"""Benchmark counterpart to examples/bench_vs_supervision.rs in warden-vision.

Same three scenarios, same sizes, so numbers are directly comparable:
1. NMS over 10,000 random boxes.
2. ByteTrack over 500 frames x 30 detections.
3. Box + label annotation over 500 frames x 50 boxes on a 1920x1080 image.
"""

import time
import warnings

import numpy as np
import supervision as sv

warnings.filterwarnings("ignore", category=FutureWarning)


class Rng:
    """Same xorshift PRNG as the Rust side, so both benchmarks draw identical data."""

    def __init__(self, seed):
        self.state = seed

    def next_f32(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        self.state = x
        return (x >> 11) / float(1 << 53)


def bench_nms():
    rng = Rng(42)
    n = 10_000
    xyxy = np.zeros((n, 4), dtype=np.float32)
    confidence = np.zeros(n, dtype=np.float32)
    class_id = np.zeros(n, dtype=int)
    for i in range(n):
        x = rng.next_f32() * 1900.0
        y = rng.next_f32() * 1000.0
        w = 20.0 + rng.next_f32() * 80.0
        h = 20.0 + rng.next_f32() * 80.0
        xyxy[i] = [x, y, x + w, y + h]
        confidence[i] = rng.next_f32()
        class_id[i] = i % 10

    detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    iters = 20
    start = time.perf_counter()
    kept = 0
    for _ in range(iters):
        kept = len(detections.with_nms(threshold=0.5, class_agnostic=False))
    elapsed = time.perf_counter() - start
    print(
        f"[nms]        n={n:5} iters={iters:3} kept={kept:5} "
        f"total={elapsed * 1000:8.2f}ms avg={elapsed / iters * 1000:8.2f}ms"
    )


def bench_bytetrack():
    rng = Rng(7)
    frames = 500
    per_frame = 30

    all_frames = []
    for f in range(frames):
        xyxy = np.zeros((per_frame, 4), dtype=np.float32)
        confidence = np.zeros(per_frame, dtype=np.float32)
        class_id = np.zeros(per_frame, dtype=int)
        for i in range(per_frame):
            base_x = i * 60.0 + f * 0.5
            base_y = i * 15.0
            jitter = rng.next_f32() * 4.0 - 2.0
            xyxy[i] = [
                base_x + jitter,
                base_y + jitter,
                base_x + 40.0 + jitter,
                base_y + 40.0 + jitter,
            ]
            confidence[i] = 0.5 + rng.next_f32() * 0.5
            class_id[i] = i % 5
        all_frames.append(sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id))

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
    )

    start = time.perf_counter()
    total_tracked = 0
    for frame in all_frames:
        total_tracked += len(tracker.update_with_detections(frame))
    elapsed = time.perf_counter() - start
    print(
        f"[bytetrack]  frames={frames:4} per_frame={per_frame:3} tracked={total_tracked:6} "
        f"total={elapsed * 1000:8.2f}ms fps={frames / elapsed:8.1f}"
    )


def bench_annotate():
    rng = Rng(99)
    frames = 500
    per_frame = 50
    w, h = 1920, 1080

    all_frames = []
    for _ in range(frames):
        xyxy = np.zeros((per_frame, 4), dtype=np.float32)
        confidence = np.zeros(per_frame, dtype=np.float32)
        class_id = np.zeros(per_frame, dtype=int)
        tracker_id = np.zeros(per_frame, dtype=int)
        for i in range(per_frame):
            x = rng.next_f32() * (w - 100.0)
            y = rng.next_f32() * (h - 100.0)
            xyxy[i] = [x, y, x + 80.0, y + 60.0]
            confidence[i] = 0.9
            class_id[i] = i % 10
            tracker_id[i] = i
        all_frames.append(
            sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
                tracker_id=tracker_id,
            )
        )

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=4)

    start = time.perf_counter()
    for dets in all_frames:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame = box_annotator.annotate(frame, dets)
        frame = label_annotator.annotate(frame, dets)
    elapsed = time.perf_counter() - start
    print(
        f"[annotate]   frames={frames:4} per_frame={per_frame:3} {w}x{h} "
        f"total={elapsed * 1000:8.2f}ms fps={frames / elapsed:8.1f}"
    )


if __name__ == "__main__":
    print(f"supervision {sv.__version__} benchmark")
    bench_nms()
    bench_bytetrack()
    bench_annotate()
