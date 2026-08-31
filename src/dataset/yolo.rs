//! Load and save datasets in the YOLO/Darknet annotation format: one
//! `.txt` label file per image (`class_id x_center y_center width height`
//! per line, normalized to `[0, 1]`), plus a `classes.txt` listing class
//! names one per line.
//!
//! Image dimensions are read from each image file's header (via
//! [`image::image_dimensions`], which does not decode pixel data) since
//! YOLO label files store normalized rather than absolute coordinates.

use std::fs;
use std::path::{Path, PathBuf};

use crate::core::{Detection, Detections};
use crate::dataset::{is_image_file, DatasetImage, DetectionDataset};
use crate::error::Error;

/// Loads a YOLO-format dataset: images from `images_dir`, one matching
/// `<stem>.txt` label file per image from `labels_dir` (images with no
/// label file are kept with zero detections), and class names from
/// `classes_path` (one name per line).
pub fn load(
    images_dir: &Path,
    labels_dir: &Path,
    classes_path: &Path,
) -> crate::Result<DetectionDataset> {
    let classes: Vec<String> = fs::read_to_string(classes_path)?
        .lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();

    let mut image_paths: Vec<PathBuf> = fs::read_dir(images_dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| is_image_file(path))
        .collect();
    image_paths.sort();

    let mut images = Vec::with_capacity(image_paths.len());
    for image_path in image_paths {
        let (width, height) = image::image_dimensions(&image_path)
            .map_err(|e| Error::Parse(format!("{}: {e}", image_path.display())))?;
        let stem = image_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let label_path = labels_dir.join(format!("{stem}.txt"));

        let detections = if label_path.exists() {
            parse_label_file(&label_path, width, height)?
        } else {
            Vec::new()
        };

        images.push(DatasetImage {
            path: image_path,
            width,
            height,
            detections: Detections::new(detections),
        });
    }

    Ok(DetectionDataset::new(classes, images))
}

fn parse_label_file(label_path: &Path, width: u32, height: u32) -> crate::Result<Vec<Detection>> {
    let text = fs::read_to_string(label_path)?;
    let mut detections = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            return Err(Error::Parse(format!(
                "{}: expected at least 5 fields, got {}",
                label_path.display(),
                parts.len()
            )));
        }
        let parse_field = |s: &str| -> crate::Result<f32> {
            s.parse::<f32>().map_err(|_| {
                Error::Parse(format!("{}: invalid number {s:?}", label_path.display()))
            })
        };
        let class_id: usize = parts[0].parse().map_err(|_| {
            Error::Parse(format!(
                "{}: invalid class id {:?}",
                label_path.display(),
                parts[0]
            ))
        })?;
        let cx = parse_field(parts[1])?;
        let cy = parse_field(parts[2])?;
        let w = parse_field(parts[3])?;
        let h = parse_field(parts[4])?;
        let confidence = parts
            .get(5)
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0);

        let (abs_cx, abs_cy) = (cx * width as f32, cy * height as f32);
        let (abs_w, abs_h) = (w * width as f32, h * height as f32);
        let bbox = [
            abs_cx - abs_w / 2.0,
            abs_cy - abs_h / 2.0,
            abs_cx + abs_w / 2.0,
            abs_cy + abs_h / 2.0,
        ];
        detections.push(Detection::new(bbox, confidence, class_id));
    }
    Ok(detections)
}

/// Saves `dataset` in YOLO format: one `<stem>.txt` label file per image
/// under `labels_dir`, and `dataset.classes` written to `classes_path`.
/// Does not copy or write image files.
pub fn save(
    dataset: &DetectionDataset,
    labels_dir: &Path,
    classes_path: &Path,
) -> crate::Result<()> {
    fs::create_dir_all(labels_dir)?;
    fs::write(classes_path, dataset.classes.join("\n"))?;

    for image in &dataset.images {
        let stem = image
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image");
        let label_path = labels_dir.join(format!("{stem}.txt"));

        let lines: Vec<String> = image
            .detections
            .iter()
            .map(|detection| {
                let [x1, y1, x2, y2] = detection.bbox;
                let cx = (x1 + x2) / 2.0 / image.width as f32;
                let cy = (y1 + y2) / 2.0 / image.height as f32;
                let w = (x2 - x1) / image.width as f32;
                let h = (y2 - y1) / image.height as f32;
                format!(
                    "{} {:.6} {:.6} {:.6} {:.6}",
                    detection.class_id, cx, cy, w, h
                )
            })
            .collect();
        fs::write(&label_path, lines.join("\n"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_image(path: &Path, width: u32, height: u32) {
        let image = image::RgbImage::new(width, height);
        image.save(path).unwrap();
    }

    #[test]
    fn round_trips_through_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let images_dir = dir.path().join("images");
        let labels_dir = dir.path().join("labels");
        fs::create_dir_all(&images_dir).unwrap();
        write_test_image(&images_dir.join("image_0.jpg"), 100, 50);

        let dataset = DetectionDataset::new(
            vec!["cat".to_string(), "dog".to_string()],
            vec![DatasetImage {
                path: images_dir.join("image_0.jpg"),
                width: 100,
                height: 50,
                detections: Detections::new(vec![
                    Detection::new([10.0, 10.0, 30.0, 30.0], 0.9, 0),
                    Detection::new([50.0, 20.0, 90.0, 40.0], 0.8, 1),
                ]),
            }],
        );

        let classes_path = dir.path().join("classes.txt");
        save(&dataset, &labels_dir, &classes_path).unwrap();
        let loaded = load(&images_dir, &labels_dir, &classes_path).unwrap();

        assert_eq!(loaded.classes, dataset.classes);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.images[0].width, 100);
        assert_eq!(loaded.images[0].height, 50);
        assert_eq!(loaded.images[0].detections.len(), 2);

        let bbox = loaded.images[0].detections.detections[0].bbox;
        assert!((bbox[0] - 10.0).abs() < 0.5);
        assert!((bbox[1] - 10.0).abs() < 0.5);
        assert!((bbox[2] - 30.0).abs() < 0.5);
        assert!((bbox[3] - 30.0).abs() < 0.5);
    }

    #[test]
    fn image_without_a_label_file_gets_zero_detections() {
        let dir = tempfile::tempdir().unwrap();
        let images_dir = dir.path().join("images");
        let labels_dir = dir.path().join("labels");
        fs::create_dir_all(&images_dir).unwrap();
        fs::create_dir_all(&labels_dir).unwrap();
        write_test_image(&images_dir.join("lonely.jpg"), 20, 20);
        fs::write(dir.path().join("classes.txt"), "thing\n").unwrap();

        let loaded = load(&images_dir, &labels_dir, &dir.path().join("classes.txt")).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded.images[0].detections.is_empty());
    }

    #[test]
    fn malformed_label_line_returns_a_parse_error() {
        let dir = tempfile::tempdir().unwrap();
        let images_dir = dir.path().join("images");
        let labels_dir = dir.path().join("labels");
        fs::create_dir_all(&images_dir).unwrap();
        fs::create_dir_all(&labels_dir).unwrap();
        write_test_image(&images_dir.join("bad.jpg"), 20, 20);
        fs::write(labels_dir.join("bad.txt"), "not a valid label line\n").unwrap();
        fs::write(dir.path().join("classes.txt"), "thing\n").unwrap();

        let result = load(&images_dir, &labels_dir, &dir.path().join("classes.txt"));
        assert!(matches!(result, Err(Error::Parse(_))));
    }
}
