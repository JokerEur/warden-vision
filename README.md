# warden-vision

[![CI](https://github.com/<your-github-username>/warden-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-github-username>/warden-vision/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/warden-vision.svg)](https://crates.io/crates/warden-vision)
[![docs.rs](https://img.shields.io/docsrs/warden-vision)](https://docs.rs/warden-vision)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Reusable computer vision building blocks for detection, tracking, and
annotation — detection/keypoint/classification data structures, geometry
and counting zones, two multi-object trackers, ~20 annotators, evaluation
metrics, dataset I/O, and video/image utilities — in the spirit of
Roboflow's Python [`supervision`](https://github.com/roboflow/supervision)
library, reimplemented as a dependency-light, feature-gated Rust crate.

This is an independent project, not affiliated with or endorsed by
Roboflow.

## Why

Python `supervision` is the de facto toolkit for gluing together an object
detector's output, a tracker, and on-screen annotation. `warden-vision`
covers the same ground natively in Rust — no Python runtime, no PyTorch —
so it fits into Rust inference pipelines (e.g. using
[`ort`](https://crates.io/crates/ort), [`tract`](https://crates.io/crates/tract),
or [`candle`](https://crates.io/crates/candle-core) to run the model
itself) without a foreign-language boundary between inference and
post-processing.

## Features at a glance

- **Core data model** — `Detections`/`Detection` (boxes, masks,
  confidence, class, tracker id), `KeyPoints` (pose/landmarks),
  `Classifications` (whole-image classification).
- **Detector-output parsing** — `Detections::from_ultralytics_onnx` and
  `Detections::from_transformers_onnx` parse the raw tensor layouts
  YOLOv8+ and DETR-family models produce, independent of which Rust
  inference runtime you used to run them.
- **Geometry** — `Point`, `Rect`, anchor `Position`s, polygon
  utilities, `LineZone` / `PolygonZone` crossing/occupancy counters.
- **Tracking** — `SortTracker` (Kalman + IoU) and `ByteTracker`
  (two-stage high/low-confidence matching), plus `DetectionsSmoother`
  for jitter reduction and `InferenceSlicer` for SAHI-style tiled
  inference.
- **Annotation** — box, mask, polygon, label (bitmap font or, via
  `ab_glyph`, real TTF/OTF), circle, dot, triangle, ellipse, round-box,
  box-corner, trace, heat map, halo, percentage bar, icon overlay,
  background dim, blur/pixelate redaction, and keypoint skeleton
  annotators — all working against a pure-Rust `image::RgbaImage` backend
  and (optionally) an OpenCV `Mat` backend, sharing one configuration
  type.
- **Metrics** — Precision, Recall, F1, ConfusionMatrix,
  MeanAveragePrecision, MeanAverageRecall.
- **Dataset I/O** — COCO, YOLO, and Pascal VOC detection formats, plus an
  image-folder classification format.
- **Video/image utilities** — `FPSMonitor`, letterbox/resize/crop/overlay,
  `ImageSink`, and (via OpenCV `videoio`) `VideoInfo`/`VideoSink`/frame
  iteration.

## Installation

```toml
[dependencies]
warden-vision = { version = "0.1", features = ["annotate-image"] }
```

### Feature flags

| Feature           | Adds                                                                 | Native deps required |
| ------------------ | --------------------------------------------------------------------- | --------------------- |
| `core` (default)   | Detections, geometry, trackers, metrics — pure Rust, always on       | none                  |
| `annotate-image`   | Pure-Rust annotators drawing on `image::RgbaImage`, `RichLabelAnnotator`, image utilities | none |
| `annotate-opencv`  | The same annotators drawing on `opencv::core::Mat`, plus video I/O   | system OpenCV install |
| `datasets`         | COCO/YOLO/Pascal VOC/classification dataset loaders and exporters    | none                  |

`core` has no dependencies beyond `ndarray`/`nalgebra`/`lapjv`/`thiserror`
and builds anywhere Rust does. `annotate-opencv` needs a system OpenCV
install (`imgproc` + `videoio` modules) and a working `bindgen`/`libclang`
toolchain to build its native bindings.

## Quick example

```rust
use warden_vision::annotators::{Annotator, BoxAnnotator};
use warden_vision::core::{Detection, Detections};
use warden_vision::tracker::SortTracker;

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

// Draw boxes + labels onto an RGBA frame.
let mut frame = image::RgbaImage::new(640, 480);
BoxAnnotator::default().annotate(&mut frame, &detections).unwrap();
```

Runnable version: `cargo run --example quickstart --features annotate-image`
(see `examples/quickstart.rs`).

## Module overview

| Module | Contents |
| --- | --- |
| `core` | `Detection`/`Detections`, `KeyPoints`, `Classifications`, `InferenceSlicer`, ONNX-tensor adapters |
| `geometry` | `Point`, `Rect`, `Position`, polygon utilities, `LineZone`/`PolygonZone` |
| `tracker` | `SortTracker`, `ByteTracker`, `DetectionsSmoother` |
| `annotators` | The `Annotator` trait and every drawing config, pure-Rust + OpenCV backends |
| `metrics` | `Precision`, `Recall`, `F1Score`, `ConfusionMatrix`, `MeanAveragePrecision`, `MeanAverageRecall` |
| `utils` | `FPSMonitor`, image resize/letterbox/crop/overlay, `ImageSink` |
| `dataset` (feature `datasets`) | COCO/YOLO/Pascal VOC/classification loaders and exporters |
| `video` (feature `annotate-opencv`) | `VideoInfo`, `VideoSink`, frame iteration |

Full API docs: `cargo doc --open --features annotate-image,datasets`.

## Relationship to Python `supervision`

This crate mirrors `supervision`'s API shape and covers its core
detection/tracking/annotation/metrics/dataset surface, but is not a
line-for-line port:

- **Detector adapters** parse raw ONNX output tensors
  (`from_ultralytics_onnx`, `from_transformers_onnx`) instead of binding
  to `ultralytics`/`transformers` Python objects, which have no Rust
  equivalent — see the `core::adapters` module docs for the reasoning.
- **ByteTrack** is a two-stage (high/low-confidence) implementation of
  the core idea, not a byte-for-byte port of the reference codebase.
- Not (yet) implemented: `RichLabelAnnotator`'s OpenCV backend (OpenCV has
  no built-in TTF rendering without FreeType), and a handful of
  `Detections.from_*` adapters for detector libraries beyond
  Ultralytics/`transformers`.

## Contributing

Issues and PRs welcome. Run `cargo fmt`, `cargo clippy --all-targets
--features annotate-image,datasets`, and `cargo test --features
annotate-image,datasets` before submitting — CI runs the same checks,
plus an `annotate-opencv` job that's allowed to fail (it needs a system
OpenCV install and isn't well-covered yet). If you touch that backend,
check its CI result and mention in your PR how you tested it.

## License

MIT — see [LICENSE](LICENSE).
