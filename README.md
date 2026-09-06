# warden-vision

[![Crates.io](https://img.shields.io/crates/v/warden-vision.svg)](https://crates.io/crates/warden-vision)
[![docs.rs](https://img.shields.io/docsrs/warden-vision)](https://docs.rs/warden-vision)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Reusable computer vision building blocks for detection, tracking, and
annotation — detection/keypoint/classification data structures (including
oriented bounding boxes), geometry and counting zones, three multi-object
trackers, ~20 annotators, evaluation metrics, dataset I/O, and video/image
utilities — in the spirit of Roboflow's Python
[`supervision`](https://github.com/roboflow/supervision) library,
reimplemented as a dependency-light, feature-gated Rust crate.

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

- **Core data model** — `Detections`/`Detection` (boxes, masks, oriented
  bounding boxes via `Detection::from_obb`, confidence, class, tracker
  id), `KeyPoints` (pose/landmarks), `Classifications` (whole-image
  classification).
- **Detector-output parsing** — `Detections::from_ultralytics_onnx` and
  `Detections::from_transformers_onnx` parse the raw tensor layouts
  YOLOv8+ and DETR-family models produce, independent of which Rust
  inference runtime you used to run them.
- **Geometry** — `Point`, `Rect`, anchor `Position`s, polygon
  utilities, rotated-rectangle IoU (`oriented_box_iou`), `LineZone` /
  `PolygonZone` crossing/occupancy counters.
- **Tracking** — `SortTracker` (Kalman + IoU), `ByteTracker` (two-stage
  high/low-confidence matching), and `DeepSortTracker` (motion + caller-
  supplied appearance embeddings, for when paths cross), plus
  `DetectionsSmoother` for jitter reduction and `InferenceSlicer` for
  SAHI-style tiled inference (with a `rayon`-parallel `run_parallel`
  behind the `parallel` feature).
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
| `parallel`         | `rayon`-parallel `Detections::non_max_suppression_parallel`, `InferenceSlicer::run_parallel` | none |

`core` has no dependencies beyond `ndarray`/`nalgebra`/`lapjv`/`thiserror`
and builds anywhere Rust does. `annotate-opencv` needs a system OpenCV
install (`imgproc` + `videoio` modules) and a working `bindgen`/`libclang`
toolchain to build its native bindings. `parallel` is worth it once a frame
has hundreds to thousands of candidate boxes/tiles (e.g. merging
`InferenceSlicer` output) — see the benchmark below for why the sequential
versions stay the better default at typical detector output sizes.

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

More examples in `examples/`:

| Example | What it shows | Run with |
| --- | --- | --- |
| `quickstart` | Detections → tracker → box annotator | `cargo run --example quickstart --features annotate-image` |
| `tracking_and_counting` | `ByteTracker` across multiple frames, `LineZone` in/out counting, trace + line-zone annotators | `cargo run --example tracking_and_counting --features annotate-image` |
| `onnx_postprocessing` | Parsing a raw YOLOv8-style ONNX tensor into `Detections`, NMS, rescaling to the original image size | `cargo run --example onnx_postprocessing` |
| `metrics_evaluation` | Precision/Recall/F1/mAP accumulated across a small predictions-vs-targets set | `cargo run --example metrics_evaluation` |
| `dataset_io` | Building a `DetectionDataset` and round-tripping it through YOLO format | `cargo run --example dataset_io --features datasets` |

## Module overview

| Module | Contents |
| --- | --- |
| `core` | `Detection`/`Detections` (incl. `from_obb`), `KeyPoints`, `Classifications`, `InferenceSlicer`, ONNX-tensor adapters |
| `geometry` | `Point`, `Rect`, `Position`, polygon utilities, `oriented_box_iou`, `LineZone`/`PolygonZone` |
| `tracker` | `SortTracker`, `ByteTracker`, `DeepSortTracker`, `DetectionsSmoother` |
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
- **`DeepSortTracker`** is a single-pass, motion-gated approximation of
  DeepSORT's matching-cascade idea (combined IoU + nearest-neighbor cosine
  distance over a per-track embedding gallery), not a port of the
  reference implementation either. It also doesn't run the embedding
  model — you supply one feature vector per detection each frame, same
  division of responsibility as the ONNX adapters not owning inference.
- **Oriented bounding boxes** live on `Detection::obb`/`Detection::from_obb`
  rather than a separate `OrientedBoxes` type; `from_obb` also populates
  `mask` with the same four corners so the existing `PolygonAnnotator`/
  `MaskAnnotator` render them, instead of shipping a duplicate
  `OrientedBoxAnnotator` that would be identical code.
- Not (yet) implemented: `RichLabelAnnotator`'s OpenCV backend (OpenCV has
  no built-in TTF rendering without FreeType), and a handful of
  `Detections.from_*` adapters for detector libraries beyond
  Ultralytics/`transformers`.

## Benchmark vs Python `supervision`

`examples/bench_vs_supervision.rs` and `scripts/bench_vs_supervision.py` run
the same three workloads — NMS, `ByteTrack`, box+label annotation — over
identical synthetic data (a shared xorshift PRNG keeps both sides bit-for-bit
comparable; NMS keeps the same 9,394/10,000 boxes on both sides, confirming
matching semantics). One data point, single-threaded, no SIMD tuning either
side (Intel i9-9880H @ 2.30GHz, `supervision` 0.29.1):

| Scenario | warden-vision (release) | Python `supervision` | Speedup |
| --- | --- | --- | --- |
| NMS, 10,000 boxes | 187 ms/call | 4,162 ms/call | ~22x |
| NMS, 10,000 boxes, `parallel` feature (16 threads) | 54 ms/call | 4,162 ms/call | ~77x |
| ByteTrack, 500 frames x 30 dets | 7,818 FPS | 352 FPS | ~22x |
| Box+label annotate, 500 frames x 50 boxes, 1920x1080 | 2,706 FPS | 413 FPS | ~6.6x |

Reproduce:

```bash
cargo run --release --example bench_vs_supervision --features annotate-image,parallel

python3 -m venv .venv && source .venv/bin/activate
pip install supervision numpy
python scripts/bench_vs_supervision.py
```

Caveats: this is one machine, one run each, no warmup loop, and the NMS case
in particular is a worst-case stress test (10k boxes is far more than a
single detector head produces) rather than a typical-frame workload — treat
the numbers as directional, not a formal benchmark suite.

The NMS row also doubles as a lesson in *how not to parallelize* greedy NMS:
the first `non_max_suppression_parallel` attempt dispatched a `rayon` job
per surviving candidate at every step of the sequential suppression loop
and measured *slower* than plain sequential NMS — thousands of small,
shrinking parallel dispatches lose to their own overhead. Splitting it into
one large parallel pass that precomputes the full pairwise IoU relation,
followed by a cheap sequential bookkeeping pass over the precomputed
booleans, is what actually gets the ~3.5x from 16 threads shown above. See
`Detections::non_max_suppression_parallel`'s doc comment in
`src/core/detections.rs` for the full writeup.

## Contributing

Issues and PRs welcome. Run `cargo fmt`, `cargo clippy --all-targets
--features annotate-image,datasets,parallel`, and `cargo test --features
annotate-image,datasets,parallel` before submitting — CI runs the same
checks, plus an `annotate-opencv` job that's allowed to fail (it needs a
system OpenCV install and isn't well-covered yet). If you touch that
backend, check its CI result and mention in your PR how you tested it.

## License

MIT — see [LICENSE](LICENSE).
