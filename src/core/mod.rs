//! Core data structures: detections and the value types built on top of them.

mod adapters;
mod classification;
mod detections;
mod inference_slicer;
mod keypoints;

pub use classification::{ClassPrediction, Classifications};
pub(crate) use detections::bbox_iou;
pub use detections::{Detection, Detections};
pub use inference_slicer::InferenceSlicer;
pub use keypoints::{KeyPoints, Keypoint, KeypointSet, COCO_17_EDGES};
