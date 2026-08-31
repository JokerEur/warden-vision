//! Minimal end-to-end example: detections in, a tracker assigning stable
//! ids, an annotator drawing boxes and labels onto a frame.
//!
//! Run with: `cargo run --example quickstart --features annotate-image`

use warden_vision::annotators::{Annotator, BoxAnnotator};
use warden_vision::core::{Detection, Detections};
use warden_vision::tracker::SortTracker;

fn main() {
    // Detections from your detector of choice for this frame (xyxy pixel
    // boxes). See `Detections::from_ultralytics_onnx` /
    // `from_transformers_onnx` if you're running a model directly.
    let mut detections = Detections::new(vec![
        Detection::new([120.0, 40.0, 260.0, 300.0], 0.91, 0),
        Detection::new([400.0, 80.0, 520.0, 260.0], 0.76, 1),
    ]);

    // Assign stable identities across frames.
    let mut tracker = SortTracker::default();
    tracker.update(&mut detections);

    for detection in detections.iter() {
        println!(
            "tracker_id={:?} class_id={} confidence={:.2} bbox={:?}",
            detection.tracker_id, detection.class_id, detection.confidence, detection.bbox
        );
    }

    // Draw boxes + labels onto an RGBA frame.
    let mut frame = image::RgbaImage::new(640, 480);
    BoxAnnotator::default()
        .annotate(&mut frame, &detections)
        .unwrap();
    frame
        .save("quickstart_output.png")
        .expect("failed to write output image");
    println!("wrote quickstart_output.png");
}
