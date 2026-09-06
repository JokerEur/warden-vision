//! Benchmark used to compare warden-vision against Python `supervision`.
//!
//! Three scenarios, sized to match `scripts/bench_vs_supervision.py`:
//! 1. NMS over 10,000 random boxes.
//! 2. ByteTrack over 500 frames x 30 detections.
//! 3. Box + label annotation over 500 frames x 50 boxes on a 1920x1080 image.
//!
//! Run with: `cargo run --release --example bench_vs_supervision --features annotate-image`

use std::time::Instant;
use warden_vision::annotators::{Annotator, BoxAnnotator, ColorPalette, LabelAnnotator};
use warden_vision::core::{Detection, Detections};
use warden_vision::geometry::Position;
use warden_vision::tracker::ByteTracker;

// Small xorshift PRNG so results are reproducible without pulling in `rand`.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f32 / (1u64 << 53) as f32
    }
}

fn bench_nms() {
    let mut rng = Rng(42);
    let n = 10_000;
    let mut dets = Vec::with_capacity(n);
    for i in 0..n {
        let x = rng.next_f32() * 1900.0;
        let y = rng.next_f32() * 1000.0;
        let w = 20.0 + rng.next_f32() * 80.0;
        let h = 20.0 + rng.next_f32() * 80.0;
        dets.push(Detection::new([x, y, x + w, y + h], rng.next_f32(), i % 10));
    }
    let detections = Detections::new(dets);

    let iters = 20;
    let start = Instant::now();
    let mut kept = 0;
    for _ in 0..iters {
        kept = detections.non_max_suppression(0.5, false).len();
    }
    let elapsed = start.elapsed();
    println!(
        "[nms]        n={n:5} iters={iters:3} kept={kept:5} total={:8.2?} avg={:8.2?}",
        elapsed,
        elapsed / iters
    );

    #[cfg(feature = "parallel")]
    {
        let start = Instant::now();
        let mut kept = 0;
        for _ in 0..iters {
            kept = detections.non_max_suppression_parallel(0.5, false).len();
        }
        let elapsed = start.elapsed();
        println!(
            "[nms-par]    n={n:5} iters={iters:3} kept={kept:5} total={:8.2?} avg={:8.2?}",
            elapsed,
            elapsed / iters
        );
    }
}

fn bench_bytetrack() {
    let mut rng = Rng(7);
    let frames = 500;
    let per_frame = 30;

    // Precompute per-frame detections so tracker time is isolated.
    let mut all_frames = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut dets = Vec::with_capacity(per_frame);
        for i in 0..per_frame {
            let base_x = (i as f32) * 60.0 + (f as f32) * 0.5;
            let base_y = (i as f32) * 15.0;
            let jitter = rng.next_f32() * 4.0 - 2.0;
            dets.push(Detection::new(
                [
                    base_x + jitter,
                    base_y + jitter,
                    base_x + 40.0 + jitter,
                    base_y + 40.0 + jitter,
                ],
                0.5 + rng.next_f32() * 0.5,
                i % 5,
            ));
        }
        all_frames.push(Detections::new(dets));
    }

    let mut tracker = ByteTracker::new(0.25, 30, 0.8, 0.5);
    let start = Instant::now();
    let mut total_tracked = 0usize;
    for frame in &all_frames {
        total_tracked += tracker.update(frame).len();
    }
    let elapsed = start.elapsed();
    println!(
        "[bytetrack]  frames={frames:4} per_frame={per_frame:3} tracked={total_tracked:6} total={:8.2?} fps={:8.1}",
        elapsed,
        frames as f64 / elapsed.as_secs_f64()
    );
}

fn bench_annotate() {
    let mut rng = Rng(99);
    let frames = 500;
    let per_frame = 50;
    let (w, h) = (1920u32, 1080u32);

    let mut all_frames = Vec::with_capacity(frames);
    for _ in 0..frames {
        let mut dets = Vec::with_capacity(per_frame);
        for i in 0..per_frame {
            let x = rng.next_f32() * (w as f32 - 100.0);
            let y = rng.next_f32() * (h as f32 - 100.0);
            let mut d = Detection::new([x, y, x + 80.0, y + 60.0], 0.9, i % 10);
            d.tracker_id = Some(i);
            dets.push(d);
        }
        all_frames.push(Detections::new(dets));
    }

    let box_annotator = BoxAnnotator::new(ColorPalette::default(), 2);
    let label_annotator = LabelAnnotator::new(ColorPalette::default(), Position::TopLeft, 1, 4.0);

    let start = Instant::now();
    for dets in &all_frames {
        let mut frame = image::RgbaImage::new(w, h);
        box_annotator.annotate(&mut frame, dets).unwrap();
        label_annotator.annotate(&mut frame, dets).unwrap();
        std::hint::black_box(&frame);
    }
    let elapsed = start.elapsed();
    println!(
        "[annotate]   frames={frames:4} per_frame={per_frame:3} {w}x{h} total={:8.2?} fps={:8.1}",
        elapsed,
        frames as f64 / elapsed.as_secs_f64()
    );
}

fn main() {
    println!("warden-vision benchmark ({})", std::env::consts::ARCH);
    bench_nms();
    bench_bytetrack();
    bench_annotate();
}
