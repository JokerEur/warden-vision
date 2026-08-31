//! Building a [`DetectionDataset`] in memory, saving it in YOLO format,
//! then loading it back — the same [`DetectionDataset`] shape is shared
//! by the COCO, YOLO, and Pascal VOC loaders/exporters, so swapping
//! `yolo::{save, load}` for `coco::{save, load}` here would round-trip
//! through COCO JSON instead.
//!
//! Run with: `cargo run --example dataset_io --features datasets`

use warden_vision::core::{Detection, Detections};
use warden_vision::dataset::{yolo, DatasetImage, DetectionDataset};

fn main() {
    let dir = std::env::temp_dir().join("warden_vision_dataset_io_example");
    let images_dir = dir.join("images");
    let labels_dir = dir.join("labels");
    let classes_path = dir.join("classes.txt");
    std::fs::create_dir_all(&images_dir).unwrap();

    // `yolo::load` reads each image's dimensions from its file header, so
    // the loader needs real (if tiny) image files on disk to round-trip
    // through.
    let image_path = images_dir.join("frame_0.jpg");
    image::RgbImage::new(320, 240).save(&image_path).unwrap();

    let dataset = DetectionDataset::new(
        vec!["cat".to_string(), "dog".to_string()],
        vec![DatasetImage {
            path: image_path,
            width: 320,
            height: 240,
            detections: Detections::new(vec![
                Detection::new([20.0, 30.0, 120.0, 180.0], 0.9, 0),
                Detection::new([150.0, 40.0, 300.0, 200.0], 0.8, 1),
            ]),
        }],
    );

    yolo::save(&dataset, &labels_dir, &classes_path).unwrap();
    println!("wrote YOLO labels to {}", labels_dir.display());

    let loaded = yolo::load(&images_dir, &labels_dir, &classes_path).unwrap();
    println!("classes: {:?}", loaded.classes);
    for image in &loaded.images {
        println!(
            "{}: {}x{}, {} detections",
            image.path.display(),
            image.width,
            image.height,
            image.detections.len()
        );
    }
}
