//! Turning a raw YOLOv8-style ONNX output tensor into [`Detections`],
//! independent of which Rust inference runtime produced it (`ort`,
//! `tract`, `candle`, ...). Here the tensor is built by hand to keep the
//! example self-contained; in a real pipeline it comes straight out of
//! your runtime's output binding.
//!
//! Run with: `cargo run --example onnx_postprocessing`

use ndarray::array;
use warden_vision::core::Detections;

fn main() {
    // Shape (N, 4 + C): one row per candidate box, already reshaped from
    // the model's native (1, 4 + C, N) layout — see
    // `Detections::from_ultralytics_onnx` for how to do that reshape.
    // Two boxes overlap heavily and share the same top class (0), the
    // third is a separate, low-confidence detection of class 1.
    let predictions = array![
        [100.0, 100.0, 80.0, 80.0, 0.91, 0.05],
        [104.0, 98.0, 78.0, 82.0, 0.83, 0.04],
        [400.0, 220.0, 60.0, 60.0, 0.10, 0.31],
    ];

    let detections = Detections::from_ultralytics_onnx(predictions.view(), 0.25);
    println!("after confidence filtering: {} boxes", detections.len());

    // Ultralytics runs NMS after exactly this step; do the same to
    // collapse the two overlapping class-0 boxes into one.
    let detections = detections.non_max_suppression(0.5, /* class_agnostic */ false);
    println!("after NMS: {} boxes", detections.len());

    // If you preprocessed with a resize (e.g. letterboxing to 640x640),
    // map box coordinates back to the original image size.
    let (model_size, original_width, original_height) = (640.0, 1280.0, 960.0);
    let scaled = detections.scale(original_width / model_size, original_height / model_size);

    for detection in scaled.iter() {
        println!(
            "class_id={} confidence={:.2} bbox={:?}",
            detection.class_id, detection.confidence, detection.bbox
        );
    }
}
