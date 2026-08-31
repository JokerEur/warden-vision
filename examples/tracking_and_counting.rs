//! Multi-frame tracking + line-crossing counting: run [`ByteTracker`] over
//! a short synthetic sequence of two objects walking through a horizontal
//! tripwire in opposite directions, then draw the last frame with boxes,
//! trails, and the live in/out count.
//!
//! Run with: `cargo run --example tracking_and_counting --features annotate-image`

use std::collections::HashMap;

use warden_vision::annotators::{
    Annotator, BoxAnnotator, ColorPalette, LineZoneAnnotator, TraceAnnotator,
};
use warden_vision::core::{Detection, Detections};
use warden_vision::geometry::{LineZone, Point, Position, Zone};
use warden_vision::tracker::ByteTracker;

/// One box walking down through `y = 240`, another walking up through it,
/// each moving 25px/frame (small enough that consecutive frames' boxes
/// still overlap, which is what lets the IoU-based tracker keep matching
/// the same identity frame to frame). In a real pipeline these would come
/// from your detector running on each successive video frame.
fn detections_for_frame(frame: usize) -> Detections {
    let frame = frame as f32;
    let y_down = 40.0 + frame * 25.0;
    let y_up = 380.0 - frame * 25.0;
    Detections::new(vec![
        Detection::new([100.0, y_down, 180.0, y_down + 80.0], 0.93, 0),
        Detection::new([400.0, y_up, 480.0, y_up + 80.0], 0.88, 0),
    ])
}

fn main() {
    let mut tracker = ByteTracker::default();
    let mut line = LineZone::new(Point::new(0.0, 240.0), Point::new(640.0, 240.0));
    let trace = TraceAnnotator::new(ColorPalette::default(), 2, Position::BottomCenter, 8);

    let mut previous_centroids: HashMap<usize, Point> = HashMap::new();
    let mut frame = image::RgbaImage::new(640, 480);

    let frame_count = 10;
    for frame_index in 0..frame_count {
        let detections = tracker.update(&detections_for_frame(frame_index));

        line.trigger(&detections, &previous_centroids);
        trace.annotate(&mut frame, &detections).unwrap();

        previous_centroids = detections
            .iter()
            .filter_map(|d| {
                d.tracker_id
                    .map(|id| (id, Point::new(d.centroid().0, d.centroid().1)))
            })
            .collect();

        if frame_index == frame_count - 1 {
            BoxAnnotator::default()
                .annotate(&mut frame, &detections)
                .unwrap();
            LineZoneAnnotator::default()
                .annotate(&mut frame, &line)
                .unwrap();
        }
    }

    println!("in={} out={}", line.in_count(), line.out_count());
    frame
        .save("tracking_and_counting_output.png")
        .expect("failed to write output image");
    println!("wrote tracking_and_counting_output.png");
}
