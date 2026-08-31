//! Load and save datasets in COCO's `instances_*.json` annotation format.
//!
//! Only the fields this crate's [`DetectionDataset`]
//! can represent are read/written: `images[].{id,file_name,width,height}`,
//! `annotations[].{image_id,category_id,bbox,score,segmentation}`, and
//! `categories[].{id,name}`. Other COCO task types (keypoints, captions)
//! and fields (`iscrowd` beyond round-tripping it as `0`, licenses, info)
//! are not read.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::core::{Detection, Detections};
use crate::dataset::{DatasetImage, DetectionDataset};
use crate::error::Error;

#[derive(Debug, Serialize, Deserialize)]
struct CocoRoot {
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnnotation>,
    categories: Vec<CocoCategory>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CocoImage {
    id: u32,
    file_name: String,
    width: u32,
    height: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct CocoAnnotation {
    id: u32,
    image_id: u32,
    category_id: u32,
    /// `[x, y, width, height]` in absolute pixels.
    bbox: [f32; 4],
    #[serde(default)]
    score: Option<f32>,
    #[serde(default)]
    segmentation: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    area: Option<f32>,
    #[serde(default)]
    iscrowd: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CocoCategory {
    id: u32,
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    supercategory: Option<String>,
}

/// Loads a COCO-format dataset from `annotations_path` (the
/// `instances_*.json` file), resolving each image's
/// [`DatasetImage::path`](crate::dataset::DatasetImage::path) against
/// `images_dir`.
///
/// Class ids are assigned by sorting `categories` by their COCO `id` and
/// numbering them `0..N` in that order (COCO category ids are otherwise
/// arbitrary, often 1-indexed with gaps).
pub fn load(annotations_path: &Path, images_dir: &Path) -> crate::Result<DetectionDataset> {
    let text = fs::read_to_string(annotations_path)?;
    let root: CocoRoot = serde_json::from_str(&text).map_err(|e| Error::Parse(e.to_string()))?;

    let mut categories = root.categories.clone();
    categories.sort_by_key(|c| c.id);
    let classes: Vec<String> = categories.iter().map(|c| c.name.clone()).collect();
    let category_index: HashMap<u32, usize> = categories
        .iter()
        .enumerate()
        .map(|(i, c)| (c.id, i))
        .collect();

    let mut images_by_id: HashMap<u32, DatasetImage> = root
        .images
        .iter()
        .map(|image| {
            (
                image.id,
                DatasetImage {
                    path: images_dir.join(&image.file_name),
                    width: image.width,
                    height: image.height,
                    detections: Detections::empty(),
                },
            )
        })
        .collect();

    let mut detections_by_image: HashMap<u32, Vec<Detection>> = HashMap::new();
    for annotation in &root.annotations {
        let Some(&class_id) = category_index.get(&annotation.category_id) else {
            continue;
        };
        let [x, y, w, h] = annotation.bbox;
        let mut detection = Detection::new(
            [x, y, x + w, y + h],
            annotation.score.unwrap_or(1.0),
            class_id,
        );
        if let Some(first_ring) = annotation.segmentation.as_ref().and_then(|s| s.first()) {
            let polygon: Vec<[f32; 2]> = first_ring
                .chunks_exact(2)
                .map(|pair| [pair[0], pair[1]])
                .collect();
            if polygon.len() >= 3 {
                detection.mask = Some(polygon);
            }
        }
        detections_by_image
            .entry(annotation.image_id)
            .or_default()
            .push(detection);
    }

    for (image_id, detections) in detections_by_image {
        if let Some(image) = images_by_id.get_mut(&image_id) {
            image.detections = Detections::new(detections);
        }
    }

    let images = root
        .images
        .iter()
        .filter_map(|image| images_by_id.remove(&image.id))
        .collect();

    Ok(DetectionDataset::new(classes, images))
}

/// Saves `dataset` as a COCO-format `instances_*.json` file at `path`.
///
/// Class ids are written as 1-indexed COCO category ids
/// (`class_id + 1`), and image ids/annotation ids are assigned
/// sequentially in dataset order.
pub fn save(dataset: &DetectionDataset, path: &Path) -> crate::Result<()> {
    let categories: Vec<CocoCategory> = dataset
        .classes
        .iter()
        .enumerate()
        .map(|(i, name)| CocoCategory {
            id: (i + 1) as u32,
            name: name.clone(),
            supercategory: None,
        })
        .collect();

    let mut images = Vec::with_capacity(dataset.images.len());
    let mut annotations = Vec::new();
    let mut next_annotation_id = 1u32;

    for (index, image) in dataset.images.iter().enumerate() {
        let image_id = (index + 1) as u32;
        let file_name = image
            .path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        images.push(CocoImage {
            id: image_id,
            file_name,
            width: image.width,
            height: image.height,
        });

        for detection in image.detections.iter() {
            let [x1, y1, x2, y2] = detection.bbox;
            let (w, h) = (x2 - x1, y2 - y1);
            annotations.push(CocoAnnotation {
                id: next_annotation_id,
                image_id,
                category_id: (detection.class_id + 1) as u32,
                bbox: [x1, y1, w, h],
                score: Some(detection.confidence),
                segmentation: detection
                    .mask
                    .as_ref()
                    .map(|polygon| vec![polygon.iter().flat_map(|p| [p[0], p[1]]).collect()]),
                area: Some(w * h),
                iscrowd: 0,
            });
            next_annotation_id += 1;
        }
    }

    let root = CocoRoot {
        images,
        annotations,
        categories,
    };
    let text = serde_json::to_string_pretty(&root).map_err(|e| Error::Parse(e.to_string()))?;
    fs::write(path, text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_dataset() -> DetectionDataset {
        let mut detection = Detection::new([10.0, 20.0, 60.0, 120.0], 0.95, 0);
        detection.mask = Some(vec![
            [10.0, 20.0],
            [60.0, 20.0],
            [60.0, 120.0],
            [10.0, 120.0],
        ]);
        DetectionDataset::new(
            vec!["cat".to_string(), "dog".to_string()],
            vec![DatasetImage {
                path: std::path::PathBuf::from("image_0.jpg"),
                width: 640,
                height: 480,
                detections: Detections::new(vec![
                    detection,
                    Detection::new([0.0, 0.0, 5.0, 5.0], 0.5, 1),
                ]),
            }],
        )
    }

    #[test]
    fn round_trips_through_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let annotations_path = dir.path().join("instances.json");
        let images_dir = dir.path().join("images");

        let dataset = sample_dataset();
        save(&dataset, &annotations_path).unwrap();
        let loaded = load(&annotations_path, &images_dir).unwrap();

        assert_eq!(loaded.classes, dataset.classes);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.images[0].width, 640);
        assert_eq!(loaded.images[0].height, 480);
        assert_eq!(loaded.images[0].path, images_dir.join("image_0.jpg"));
        assert_eq!(loaded.images[0].detections.len(), 2);

        let boxes = &loaded.images[0].detections;
        assert!((boxes.detections[0].bbox[0] - 10.0).abs() < 1e-3);
        assert!((boxes.detections[0].bbox[3] - 120.0).abs() < 1e-3);
        assert_eq!(boxes.detections[0].class_id, 0);
        assert!(boxes.detections[0].mask.is_some());
        assert_eq!(boxes.detections[1].class_id, 1);
    }

    #[test]
    fn class_ids_follow_sorted_category_id_order_regardless_of_json_order() {
        let dir = tempfile::tempdir().unwrap();
        let annotations_path = dir.path().join("instances.json");
        // category id 5 -> "dog" listed before id 2 -> "cat" in the file,
        // but class ids should follow ascending category id (2 then 5).
        let json = r#"{
            "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 5, "bbox": [0,0,1,1]},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [1,1,1,1]}
            ],
            "categories": [
                {"id": 5, "name": "dog"},
                {"id": 2, "name": "cat"}
            ]
        }"#;
        fs::write(&annotations_path, json).unwrap();

        let loaded = load(&annotations_path, dir.path()).unwrap();
        assert_eq!(loaded.classes, vec!["cat".to_string(), "dog".to_string()]);
        let class_ids: Vec<usize> = loaded.images[0]
            .detections
            .iter()
            .map(|d| d.class_id)
            .collect();
        assert!(class_ids.contains(&0)); // cat
        assert!(class_ids.contains(&1)); // dog
    }

    #[test]
    fn malformed_json_returns_a_parse_error() {
        let dir = tempfile::tempdir().unwrap();
        let annotations_path = dir.path().join("bad.json");
        fs::write(&annotations_path, "{ not json").unwrap();
        let result = load(&annotations_path, dir.path());
        assert!(matches!(result, Err(Error::Parse(_))));
    }
}
