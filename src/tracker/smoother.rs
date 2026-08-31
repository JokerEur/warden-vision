//! Temporal smoothing of tracked bounding boxes: averages each
//! [`tracker_id`](crate::core::Detection::tracker_id)'s box over a
//! sliding window of recent frames, damping the frame-to-frame jitter a
//! detector alone tends to produce.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::core::Detections;

/// Smooths bounding boxes across frames by averaging each tracked
/// object's box over its last `length` sightings.
///
/// Meant to sit downstream of a tracker
/// ([`SortTracker`](crate::tracker::SortTracker) /
/// [`ByteTracker`](crate::tracker::ByteTracker)): it keys its history by
/// [`tracker_id`](crate::core::Detection::tracker_id), so detections
/// without one pass through with their box unchanged (there's no
/// identity to accumulate history under).
///
/// History for a tracker id is dropped the moment it's absent from a
/// frame, rather than kept around for some grace period — if it
/// reappears later it starts smoothing fresh, the same as a brand-new
/// track.
pub struct DetectionsSmoother {
    length: usize,
    history: HashMap<usize, VecDeque<[f32; 4]>>,
}

impl DetectionsSmoother {
    /// Creates a smoother averaging over the last `length` sightings of
    /// each tracker id.
    ///
    /// # Panics
    /// Panics if `length` is `0`.
    pub fn new(length: usize) -> Self {
        assert!(length > 0, "smoothing window length must be at least 1");
        Self {
            length,
            history: HashMap::new(),
        }
    }

    /// Advances the smoother by one frame, returning a new [`Detections`]
    /// with each tracked detection's `bbox` replaced by the average of
    /// its last (up to) `length` boxes, including this frame's.
    pub fn update(&mut self, detections: &Detections) -> Detections {
        let mut seen = HashSet::new();
        let mut output = Vec::with_capacity(detections.len());

        for detection in detections.iter() {
            let Some(tracker_id) = detection.tracker_id else {
                output.push(detection.clone());
                continue;
            };
            seen.insert(tracker_id);

            let track = self.history.entry(tracker_id).or_default();
            track.push_back(detection.bbox);
            while track.len() > self.length {
                track.pop_front();
            }

            let mut smoothed = detection.clone();
            smoothed.bbox = average_bbox(track);
            output.push(smoothed);
        }

        self.history
            .retain(|tracker_id, _| seen.contains(tracker_id));
        Detections::new(output)
    }
}

impl Default for DetectionsSmoother {
    /// A 5-frame smoothing window.
    fn default() -> Self {
        Self::new(5)
    }
}

fn average_bbox(boxes: &VecDeque<[f32; 4]>) -> [f32; 4] {
    let n = boxes.len() as f32;
    let mut sum = [0.0f32; 4];
    for bbox in boxes {
        for i in 0..4 {
            sum[i] += bbox[i];
        }
    }
    [sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Detection;

    fn tracked(bbox: [f32; 4], tracker_id: usize) -> Detection {
        let mut d = Detection::new(bbox, 0.9, 0);
        d.tracker_id = Some(tracker_id);
        d
    }

    #[test]
    fn averages_a_single_tracks_box_over_the_window() {
        let mut smoother = DetectionsSmoother::new(3);
        smoother.update(&Detections::new(vec![tracked([0.0, 0.0, 10.0, 10.0], 1)]));
        smoother.update(&Detections::new(vec![tracked([10.0, 0.0, 20.0, 10.0], 1)]));
        let out = smoother.update(&Detections::new(vec![tracked([20.0, 0.0, 30.0, 10.0], 1)]));

        // Average of x1 in {0, 10, 20} = 10; average of x2 in {10, 20, 30} = 20.
        assert_eq!(out.detections[0].bbox, [10.0, 0.0, 20.0, 10.0]);
    }

    #[test]
    fn window_caps_at_configured_length() {
        let mut smoother = DetectionsSmoother::new(2);
        smoother.update(&Detections::new(vec![tracked([0.0, 0.0, 10.0, 10.0], 1)]));
        smoother.update(&Detections::new(vec![tracked(
            [100.0, 0.0, 110.0, 10.0],
            1,
        )]));
        let out = smoother.update(&Detections::new(vec![tracked(
            [200.0, 0.0, 210.0, 10.0],
            1,
        )]));

        // Only the last 2 sightings (100, 200) should count, not the first (0).
        assert_eq!(out.detections[0].bbox[0], 150.0);
    }

    #[test]
    fn detections_without_a_tracker_id_pass_through_unsmoothed() {
        let mut smoother = DetectionsSmoother::new(3);
        let out = smoother.update(&Detections::new(vec![Detection::new(
            [5.0, 5.0, 15.0, 15.0],
            0.9,
            0,
        )]));
        assert_eq!(out.detections[0].bbox, [5.0, 5.0, 15.0, 15.0]);
    }

    #[test]
    fn missing_a_frame_resets_a_tracks_history() {
        let mut smoother = DetectionsSmoother::new(5);
        smoother.update(&Detections::new(vec![tracked([0.0, 0.0, 10.0, 10.0], 1)]));
        // Tracker 1 absent this frame: its history should be dropped.
        smoother.update(&Detections::empty());
        let out = smoother.update(&Detections::new(vec![tracked(
            [100.0, 0.0, 110.0, 10.0],
            1,
        )]));

        // If the old history had survived, this would be averaged with
        // the [0, 10] box from before the gap; instead it should be
        // exactly the fresh single-frame box.
        assert_eq!(out.detections[0].bbox, [100.0, 0.0, 110.0, 10.0]);
    }

    #[test]
    fn two_tracks_are_smoothed_independently() {
        let mut smoother = DetectionsSmoother::new(2);
        smoother.update(&Detections::new(vec![
            tracked([0.0, 0.0, 10.0, 10.0], 1),
            tracked([0.0, 100.0, 10.0, 110.0], 2),
        ]));
        let out = smoother.update(&Detections::new(vec![
            tracked([20.0, 0.0, 30.0, 10.0], 1),
            tracked([20.0, 100.0, 30.0, 110.0], 2),
        ]));
        assert_eq!(out.detections[0].bbox[0], 10.0);
        assert_eq!(out.detections[1].bbox[0], 10.0);
    }

    #[test]
    #[should_panic]
    fn zero_length_panics() {
        DetectionsSmoother::new(0);
    }
}
