//! Load and save datasets in the Pascal VOC annotation format: one XML
//! file per image, with `<object><name>...</name><bndbox>...</bndbox></object>`
//! entries for each detection.
//!
//! Pascal VOC has no separate class list file; [`load`] derives one by
//! collecting every distinct `<object><name>` across all annotation files,
//! sorted alphabetically for a deterministic, file-order-independent class
//! id assignment.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use quick_xml::events::Event;
use quick_xml::reader::Reader;

use crate::core::{Detection, Detections};
use crate::dataset::{DatasetImage, DetectionDataset};
use crate::error::Error;

struct VocObject {
    name: String,
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
}

struct VocAnnotation {
    filename: String,
    width: u32,
    height: u32,
    objects: Vec<VocObject>,
}

/// Resolves one of the five XML-predefined named entities. `quick_xml`
/// only resolves numeric character references (`&#65;`) itself; named
/// entities are left for the caller.
fn resolve_named_entity(name: &str) -> Option<char> {
    match name {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        _ => None,
    }
}

fn parse_xml(text: &str, source: &Path) -> crate::Result<VocAnnotation> {
    let to_parse_error = |e: quick_xml::Error| Error::Parse(format!("{}: {e}", source.display()));

    let mut reader = Reader::from_str(text);
    // Trimming is applied once, to the fully-assembled `text_buffer`, in
    // the `Event::End` handler below — per-event trimming here would eat
    // meaningful interior spaces whenever text is split across multiple
    // `Text`/`GeneralRef` events by an entity reference (e.g. "A &amp; B").
    let mut buf = Vec::new();

    let mut filename = String::new();
    let mut width = 0u32;
    let mut height = 0u32;
    let mut objects = Vec::new();

    let mut tag_stack: Vec<String> = Vec::new();
    let mut current_object: Option<VocObject> = None;
    let mut text_buffer = String::new();

    loop {
        match reader.read_event_into(&mut buf).map_err(to_parse_error)? {
            Event::Start(start) => {
                let name = start.name().as_ref().to_string();
                if name == "object" {
                    current_object = Some(VocObject {
                        name: String::new(),
                        xmin: 0.0,
                        ymin: 0.0,
                        xmax: 0.0,
                        ymax: 0.0,
                    });
                }
                tag_stack.push(name);
                text_buffer.clear();
            }
            Event::Text(text) => {
                // Character data adjacent to an entity reference (e.g. the
                // "A " and " B " either side of "&amp;" in "A &amp; B")
                // arrives as separate `Text` events, so this must append,
                // not overwrite.
                text_buffer.push_str(&text.xml10_content());
            }
            Event::GeneralRef(reference) => {
                if let Some(ch) = reference.resolve_char_ref().map_err(to_parse_error)? {
                    text_buffer.push(ch);
                } else if let Some(ch) = resolve_named_entity(reference.xml10_content().as_ref()) {
                    text_buffer.push(ch);
                }
            }
            Event::End(_) => {
                let closing = tag_stack.pop().unwrap_or_default();
                let value = text_buffer.trim();
                match closing.as_str() {
                    "filename" => filename = value.to_string(),
                    "width" => width = value.parse().unwrap_or(0),
                    "height" => height = value.parse().unwrap_or(0),
                    "name" => {
                        if let Some(obj) = current_object.as_mut() {
                            obj.name = value.to_string();
                        }
                    }
                    "xmin" => {
                        if let Some(obj) = current_object.as_mut() {
                            obj.xmin = value.parse().unwrap_or(0.0);
                        }
                    }
                    "ymin" => {
                        if let Some(obj) = current_object.as_mut() {
                            obj.ymin = value.parse().unwrap_or(0.0);
                        }
                    }
                    "xmax" => {
                        if let Some(obj) = current_object.as_mut() {
                            obj.xmax = value.parse().unwrap_or(0.0);
                        }
                    }
                    "ymax" => {
                        if let Some(obj) = current_object.as_mut() {
                            obj.ymax = value.parse().unwrap_or(0.0);
                        }
                    }
                    "object" => {
                        if let Some(obj) = current_object.take() {
                            objects.push(obj);
                        }
                    }
                    _ => {}
                }
            }
            Event::Eof => break,
            _ => {}
        }
        buf.clear();
    }

    Ok(VocAnnotation {
        filename,
        width,
        height,
        objects,
    })
}

/// Loads a Pascal VOC-format dataset: one `.xml` file per image from
/// `annotations_dir`, resolving each `<filename>` against `images_dir`.
pub fn load(images_dir: &Path, annotations_dir: &Path) -> crate::Result<DetectionDataset> {
    let mut xml_paths: Vec<PathBuf> = fs::read_dir(annotations_dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("xml"))
        .collect();
    xml_paths.sort();

    let mut parsed = Vec::with_capacity(xml_paths.len());
    let mut class_set: BTreeSet<String> = BTreeSet::new();
    for xml_path in &xml_paths {
        let text = fs::read_to_string(xml_path)?;
        let annotation = parse_xml(&text, xml_path)?;
        for object in &annotation.objects {
            class_set.insert(object.name.clone());
        }
        parsed.push(annotation);
    }

    let classes: Vec<String> = class_set.into_iter().collect();
    let class_index: std::collections::HashMap<&str, usize> = classes
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();

    let images = parsed
        .into_iter()
        .map(|annotation| {
            let detections: Vec<Detection> = annotation
                .objects
                .into_iter()
                .filter_map(|object| {
                    class_index.get(object.name.as_str()).map(|&class_id| {
                        Detection::new(
                            [object.xmin, object.ymin, object.xmax, object.ymax],
                            1.0,
                            class_id,
                        )
                    })
                })
                .collect();
            let path = if annotation.filename.is_empty() {
                images_dir.join("unknown")
            } else {
                images_dir.join(&annotation.filename)
            };
            DatasetImage {
                path,
                width: annotation.width,
                height: annotation.height,
                detections: Detections::new(detections),
            }
        })
        .collect();

    Ok(DetectionDataset::new(classes, images))
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Saves `dataset` as one Pascal VOC `.xml` file per image under
/// `annotations_dir`. Does not copy or write image files.
pub fn save(dataset: &DetectionDataset, annotations_dir: &Path) -> crate::Result<()> {
    fs::create_dir_all(annotations_dir)?;

    for image in &dataset.images {
        let filename = image
            .path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("image")
            .to_string();
        let stem = image
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image");
        let xml_path = annotations_dir.join(format!("{stem}.xml"));

        let mut objects_xml = String::new();
        for detection in image.detections.iter() {
            let name = dataset
                .classes
                .get(detection.class_id)
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            let [x1, y1, x2, y2] = detection.bbox;
            objects_xml.push_str(&format!(
                "  <object>\n    <name>{}</name>\n    <bndbox>\n      <xmin>{x1}</xmin>\n      <ymin>{y1}</ymin>\n      <xmax>{x2}</xmax>\n      <ymax>{y2}</ymax>\n    </bndbox>\n  </object>\n",
                escape_xml(name)
            ));
        }

        let xml = format!(
            "<annotation>\n  <filename>{}</filename>\n  <size>\n    <width>{}</width>\n    <height>{}</height>\n    <depth>3</depth>\n  </size>\n{}</annotation>\n",
            escape_xml(&filename),
            image.width,
            image.height,
            objects_xml
        );
        fs::write(&xml_path, xml)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let images_dir = dir.path().join("images");
        let annotations_dir = dir.path().join("annotations");

        let dataset = DetectionDataset::new(
            vec!["cat".to_string(), "dog".to_string()],
            vec![DatasetImage {
                path: images_dir.join("pic.jpg"),
                width: 500,
                height: 375,
                detections: Detections::new(vec![
                    Detection::new([10.0, 20.0, 100.0, 200.0], 1.0, 0),
                    Detection::new([1.0, 1.0, 5.0, 5.0], 1.0, 1),
                ]),
            }],
        );

        save(&dataset, &annotations_dir).unwrap();
        let loaded = load(&images_dir, &annotations_dir).unwrap();

        assert_eq!(loaded.classes, dataset.classes);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.images[0].width, 500);
        assert_eq!(loaded.images[0].height, 375);
        assert_eq!(loaded.images[0].path, images_dir.join("pic.jpg"));
        assert_eq!(loaded.images[0].detections.len(), 2);
        let bbox = loaded.images[0].detections.detections[0].bbox;
        assert_eq!(bbox, [10.0, 20.0, 100.0, 200.0]);
    }

    #[test]
    fn class_list_is_alphabetically_sorted_and_deduplicated() {
        let dir = tempfile::tempdir().unwrap();
        let annotations_dir = dir.path().join("annotations");
        fs::create_dir_all(&annotations_dir).unwrap();

        let xml_a = r#"<annotation><filename>a.jpg</filename><size><width>10</width><height>10</height></size>
            <object><name>zebra</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>1</xmax><ymax>1</ymax></bndbox></object>
            <object><name>ant</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>1</xmax><ymax>1</ymax></bndbox></object>
        </annotation>"#;
        let xml_b = r#"<annotation><filename>b.jpg</filename><size><width>10</width><height>10</height></size>
            <object><name>ant</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>1</xmax><ymax>1</ymax></bndbox></object>
        </annotation>"#;
        fs::write(annotations_dir.join("a.xml"), xml_a).unwrap();
        fs::write(annotations_dir.join("b.xml"), xml_b).unwrap();

        let loaded = load(dir.path(), &annotations_dir).unwrap();
        assert_eq!(loaded.classes, vec!["ant".to_string(), "zebra".to_string()]);
    }

    #[test]
    fn special_characters_in_names_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let images_dir = dir.path().join("images");
        let annotations_dir = dir.path().join("annotations");

        let dataset = DetectionDataset::new(
            vec!["A & B <thing>".to_string()],
            vec![DatasetImage {
                path: images_dir.join("weird.jpg"),
                width: 10,
                height: 10,
                detections: Detections::new(vec![Detection::new([0.0, 0.0, 1.0, 1.0], 1.0, 0)]),
            }],
        );
        save(&dataset, &annotations_dir).unwrap();
        let loaded = load(&images_dir, &annotations_dir).unwrap();
        assert_eq!(loaded.classes, dataset.classes);
    }
}
