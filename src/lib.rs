//! `warden-vision`: reusable computer vision building blocks for
//! detection, tracking, and annotation, mirroring the API shape of
//! Roboflow's Python `supervision` library.
//!
//! # Modules
//! - [`core`]: [`core::Detection`] / [`core::Detections`], the shared
//!   currency between detectors, trackers, zones, and annotators, plus
//!   [`core::KeyPoints`] (pose/landmark data, [`core::COCO_17_EDGES`]
//!   skeleton), [`core::Classifications`] (whole-image classification),
//!   [`core::InferenceSlicer`] (tiled/SAHI-style inference), and
//!   [`core::Detections::from_ultralytics_onnx`] /
//!   [`core::Detections::from_transformers_onnx`] for parsing raw
//!   detector output tensors (see their docs for why these parse tensors
//!   rather than binding to `ultralytics`/`transformers` objects).
//! - [`geometry`]: [`geometry::Point`], [`geometry::Rect`], the anchor-point
//!   [`geometry::Position`] enum, polygon utilities
//!   ([`geometry::polygon_area`], [`geometry::polygon_centroid`],
//!   [`geometry::polygon_to_rect`]), the [`geometry::Zone`] trait, and its
//!   [`geometry::LineZone`] / [`geometry::PolygonZone`] implementations.
//! - [`tracker`]: [`tracker::SortTracker`] (single-stage Kalman + IoU) and
//!   [`tracker::ByteTracker`] (two-stage high/low-confidence matching,
//!   after Zhang et al.'s ByteTrack), both assigning
//!   [`core::Detection::tracker_id`]; [`tracker::DetectionsSmoother`] to
//!   damp jitter in a tracked box across frames.
//! - [`annotators`]: the [`annotators::Annotator`] trait and a family of
//!   drawing configs — boxes (round-cornered, corner-only), masks,
//!   polygons, labels (bitmap font, or under `annotate-image` real
//!   TTF/OTF via `annotators::RichLabelAnnotator`), circles, dots,
//!   triangles, ellipses, motion traces, heat maps, halos, percentage
//!   bars, icon overlays, background dimming, blur/pixelate redaction,
//!   zone outlines, and keypoint skeletons — with pure-Rust
//!   (`annotate-image`) and OpenCV (`annotate-opencv`) rendering
//!   backends (`RichLabelAnnotator` is pure-Rust only; see its docs for
//!   why).
//! - [`metrics`]: [`metrics::Precision`], [`metrics::Recall`],
//!   [`metrics::F1Score`], [`metrics::ConfusionMatrix`],
//!   [`metrics::MeanAveragePrecision`], and
//!   [`metrics::MeanAverageRecall`] for scoring predicted
//!   [`core::Detections`] against ground truth.
//! - [`utils`]: [`utils::FPSMonitor`] for throughput reporting, plus
//!   (under `annotate-image`) `utils::resize_keeping_aspect_ratio`,
//!   `utils::letterbox`, `utils::crop_image`, `utils::overlay_image`, and
//!   `utils::ImageSink` for writing a numbered frame sequence.
//! - `dataset` (under `datasets`): `dataset::DetectionDataset` plus COCO
//!   (`dataset::coco`), YOLO (`dataset::yolo`), and Pascal VOC
//!   (`dataset::pascal_voc`) loaders/exporters, and
//!   `dataset::ClassificationDataset` with an image-folder-per-class
//!   loader/exporter (`dataset::classification`).
//! - `video` (under `annotate-opencv`): `video::VideoInfo`,
//!   `video::frames`, and `video::VideoSink` for reading and writing
//!   video files via OpenCV's `videoio`.
//!
//! (Items behind a non-default feature are written above as plain
//! `code text` rather than intra-doc links, since those links only
//! resolve when this crate's docs are built with that feature enabled.
//! The published docs build with `annotate-image` and `datasets` — see
//! `[package.metadata.docs.rs]` in `Cargo.toml` — but not
//! `annotate-opencv`, since docs.rs has no system OpenCV install; read
//! `video`'s module docs directly in `src/video/mod.rs` instead.)
//!
//! # Scope
//! This crate targets the core, dependency-light building blocks of
//! Roboflow's `supervision`: detection/keypoint/classification data
//! structures, geometry, tracking, annotation, evaluation metrics,
//! dataset I/O, image utilities, and OpenCV-backed video I/O. Remaining
//! known gaps: `RichLabelAnnotator` has no `annotate-opencv`
//! implementation (OpenCV has no built-in TTF rendering without
//! FreeType); a dedicated `OrientedBoxAnnotator` doesn't exist since
//! [`annotators::PolygonAnnotator`] already draws an arbitrary 4-point
//! rotated box stored in [`core::Detection::mask`]; and detector-output
//! parsing covers Ultralytics YOLO and DETR-family `transformers` models
//! (the two most common) rather than every framework `supervision` has
//! an adapter for.

pub mod annotators;
pub mod core;
#[cfg(feature = "datasets")]
pub mod dataset;
pub mod error;
pub mod geometry;
pub mod metrics;
pub mod tracker;
pub mod utils;
#[cfg(feature = "annotate-opencv")]
pub mod video;

pub use error::{Error, Result};
