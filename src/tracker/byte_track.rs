//! ByteTrack: a Kalman-filter multi-object tracker that, unlike
//! [`SortTracker`](crate::tracker::SortTracker), associates in two IoU
//! passes per frame — high-confidence detections first, then low-confidence
//! ones against whatever tracks are still unmatched. This lets it recover
//! tracks through brief occlusion or detector noise using boxes that a
//! confidence-thresholding preprocessing step would otherwise discard
//! entirely, which is ByteTrack's core idea (Zhang et al., 2022).
//!
//! Not a byte-for-byte port of the reference implementation: it skips the
//! separate "unconfirmed" track category for brand-new single-frame
//! tracks, and its default thresholds are tuned for this crate rather
//! than copied from the paper.

use crate::core::{Detection, Detections};
use crate::tracker::assignment::assign_by_iou;
use crate::tracker::kalman::KalmanBoxFilter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrackState {
    /// Matched within the last frame (either continuously, or just
    /// re-matched after being [`TrackState::Lost`]).
    Tracked,
    /// Unmatched for at least one frame; still retained (and still
    /// eligible for stage-1 re-matching) until `lost_track_buffer` frames
    /// have elapsed.
    Lost,
}

struct Track {
    id: usize,
    filter: KalmanBoxFilter,
    state: TrackState,
    time_since_update: usize,
}

/// A ByteTrack multi-object tracker.
///
/// Unlike [`SortTracker`](crate::tracker::SortTracker)'s
/// [`update`](crate::tracker::SortTracker::update), which mutates every
/// input detection in place, [`ByteTracker::update`] returns a new
/// [`Detections`] containing only the detections that ended up associated
/// with a track. This is a deliberate consequence of the algorithm: a
/// low-confidence detection that fails to match any existing track is, by
/// design, never allowed to spawn a new one (that's what keeps ByteTrack
/// from turning background noise into new identities), so it has no track
/// id to report and is dropped from the output rather than passed through
/// unmodified.
pub struct ByteTracker {
    tracks: Vec<Track>,
    next_id: usize,
    /// Detections at or above this confidence are "high-confidence": they
    /// are matched first, and any left over spawn new tracks.
    track_activation_threshold: f32,
    /// IoU threshold for stage-1 (high-confidence) matching.
    high_confidence_matching_threshold: f32,
    /// IoU threshold for stage-2 (low-confidence) matching. Kept separate
    /// from, and typically lower than,
    /// `high_confidence_matching_threshold` since low-confidence detections
    /// tend to have noisier box localization.
    low_confidence_matching_threshold: f32,
    /// Number of consecutive missed frames a [`TrackState::Lost`] track
    /// tolerates before being dropped.
    lost_track_buffer: usize,
}

impl ByteTracker {
    /// Creates a tracker.
    ///
    /// - `track_activation_threshold`: confidence cutoff between the
    ///   high-confidence and low-confidence detection pools.
    /// - `lost_track_buffer`: frames a track survives, unmatched, before
    ///   being dropped.
    /// - `high_confidence_matching_threshold` / `low_confidence_matching_threshold`:
    ///   minimum IoU for a match in each of the two association passes.
    pub fn new(
        track_activation_threshold: f32,
        lost_track_buffer: usize,
        high_confidence_matching_threshold: f32,
        low_confidence_matching_threshold: f32,
    ) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            track_activation_threshold,
            high_confidence_matching_threshold,
            low_confidence_matching_threshold,
            lost_track_buffer,
        }
    }

    /// Number of tracks currently being maintained, including lost-but-not-
    /// yet-aged-out ones.
    pub fn active_tracks(&self) -> usize {
        self.tracks.len()
    }

    /// Advances the tracker by one frame.
    ///
    /// Returns a new [`Detections`] holding the subset of `detections` that
    /// matched or spawned a track, each with [`Detection::tracker_id`] set;
    /// see the type-level docs for why unmatched low-confidence detections
    /// are dropped rather than passed through.
    pub fn update(&mut self, detections: &Detections) -> Detections {
        let predicted_boxes: Vec<[f32; 4]> =
            self.tracks.iter_mut().map(|t| t.filter.predict()).collect();

        let (high_idx, low_idx): (Vec<usize>, Vec<usize>) = (0..detections.len())
            .partition(|&i| detections.detections[i].confidence >= self.track_activation_threshold);
        let high_boxes: Vec<[f32; 4]> = high_idx
            .iter()
            .map(|&i| detections.detections[i].bbox)
            .collect();
        let low_boxes: Vec<[f32; 4]> = low_idx
            .iter()
            .map(|&i| detections.detections[i].bbox)
            .collect();

        let mut track_matched = vec![false; self.tracks.len()];
        let mut assigned_track_id: Vec<Option<usize>> = vec![None; detections.len()];

        // Stage 1: every current track (tracked or lost) against high-confidence detections.
        let stage1 = assign_by_iou(
            &predicted_boxes,
            &high_boxes,
            self.high_confidence_matching_threshold,
        );
        for (k, matched_track) in stage1.iter().enumerate() {
            if let Some(track_index) = matched_track {
                let detection_index = high_idx[k];
                let track = &mut self.tracks[*track_index];
                track
                    .filter
                    .update(detections.detections[detection_index].bbox);
                track.time_since_update = 0;
                track.state = TrackState::Tracked;
                assigned_track_id[detection_index] = Some(track.id);
                track_matched[*track_index] = true;
            }
        }

        // Stage 2: tracks still Tracked (not already Lost) but unmatched in
        // stage 1, against low-confidence detections.
        let stage2_track_indices: Vec<usize> = (0..self.tracks.len())
            .filter(|&i| !track_matched[i] && self.tracks[i].state == TrackState::Tracked)
            .collect();
        let stage2_track_boxes: Vec<[f32; 4]> = stage2_track_indices
            .iter()
            .map(|&i| predicted_boxes[i])
            .collect();
        let stage2 = assign_by_iou(
            &stage2_track_boxes,
            &low_boxes,
            self.low_confidence_matching_threshold,
        );
        for (k, matched_track) in stage2.iter().enumerate() {
            if let Some(local_index) = matched_track {
                let track_index = stage2_track_indices[*local_index];
                let detection_index = low_idx[k];
                let track = &mut self.tracks[track_index];
                track
                    .filter
                    .update(detections.detections[detection_index].bbox);
                track.time_since_update = 0;
                track.state = TrackState::Tracked;
                assigned_track_id[detection_index] = Some(track.id);
                track_matched[track_index] = true;
            }
        }

        // Anything still unmatched after both stages is (or remains) lost.
        for (i, track) in self.tracks.iter_mut().enumerate() {
            if !track_matched[i] {
                track.time_since_update += 1;
                track.state = TrackState::Lost;
            }
        }
        self.tracks
            .retain(|t| t.time_since_update <= self.lost_track_buffer);

        // Unmatched high-confidence detections spawn new tracks. Unmatched
        // low-confidence detections do not (see type-level docs).
        for (k, matched_track) in stage1.iter().enumerate() {
            if matched_track.is_none() {
                let detection_index = high_idx[k];
                let id = self.next_id;
                self.next_id += 1;
                let filter = KalmanBoxFilter::new(detections.detections[detection_index].bbox);
                self.tracks.push(Track {
                    id,
                    filter,
                    state: TrackState::Tracked,
                    time_since_update: 0,
                });
                assigned_track_id[detection_index] = Some(id);
            }
        }

        let output: Vec<Detection> = detections
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                assigned_track_id[i].map(|id| {
                    let mut d = d.clone();
                    d.tracker_id = Some(id);
                    d
                })
            })
            .collect();
        Detections::new(output)
    }
}

impl Default for ByteTracker {
    /// Commonly used ByteTrack-style defaults: 0.25 confidence split, a
    /// 30-frame lost-track buffer, 0.3 IoU for high-confidence matching,
    /// 0.2 IoU (more permissive, since low-confidence boxes are noisier)
    /// for low-confidence matching.
    fn default() -> Self {
        Self::new(0.25, 30, 0.3, 0.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Detection;

    fn det(bbox: [f32; 4], confidence: f32) -> Detection {
        Detection::new(bbox, confidence, 0)
    }

    #[test]
    fn high_confidence_detection_spawns_a_track() {
        let mut tracker = ByteTracker::default();
        let detections = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9)]);
        let out = tracker.update(&detections);
        assert_eq!(out.len(), 1);
        assert!(out.detections[0].tracker_id.is_some());
        assert_eq!(tracker.active_tracks(), 1);
    }

    #[test]
    fn low_confidence_unmatched_detection_is_dropped_and_spawns_nothing() {
        let mut tracker = ByteTracker::default();
        let detections = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.1)]);
        let out = tracker.update(&detections);
        assert_eq!(out.len(), 0);
        assert_eq!(tracker.active_tracks(), 0);
    }

    #[test]
    fn matching_high_confidence_detection_keeps_same_id() {
        let mut tracker = ByteTracker::default();
        let first = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9)]);
        let out1 = tracker.update(&first);
        let id = out1.detections[0].tracker_id.unwrap();

        let second = Detections::new(vec![det([1.0, 1.0, 11.0, 11.0], 0.9)]);
        let out2 = tracker.update(&second);
        assert_eq!(out2.detections[0].tracker_id, Some(id));
    }

    #[test]
    fn low_confidence_detection_recovers_an_existing_track() {
        let mut tracker = ByteTracker::default();
        let first = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9)]);
        let out1 = tracker.update(&first);
        let id = out1.detections[0].tracker_id.unwrap();

        // Same object, but now only seen at low confidence (e.g. partial
        // occlusion): stage 2 should still recover it under the same id
        // rather than dropping it.
        let second = Detections::new(vec![det([1.0, 1.0, 11.0, 11.0], 0.15)]);
        let out2 = tracker.update(&second);
        assert_eq!(out2.len(), 1);
        assert_eq!(out2.detections[0].tracker_id, Some(id));
    }

    #[test]
    fn track_survives_briefly_missing_and_is_dropped_after_lost_track_buffer() {
        let mut tracker = ByteTracker::new(0.25, 2, 0.3, 0.2);
        let first = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9)]);
        tracker.update(&first);
        assert_eq!(tracker.active_tracks(), 1);

        for _ in 0..2 {
            tracker.update(&Detections::empty());
        }
        assert_eq!(
            tracker.active_tracks(),
            1,
            "should survive up to lost_track_buffer misses"
        );

        tracker.update(&Detections::empty());
        assert_eq!(
            tracker.active_tracks(),
            0,
            "should be dropped once lost_track_buffer is exceeded"
        );
    }

    #[test]
    fn two_independent_objects_maintain_separate_identities() {
        let mut tracker = ByteTracker::default();
        let first = Detections::new(vec![
            det([0.0, 0.0, 10.0, 10.0], 0.9),
            det([100.0, 0.0, 110.0, 10.0], 0.9),
        ]);
        let out1 = tracker.update(&first);
        let id_left = out1.detections[0].tracker_id.unwrap();
        let id_right = out1.detections[1].tracker_id.unwrap();
        assert_ne!(id_left, id_right);

        let second = Detections::new(vec![
            det([2.0, 0.0, 12.0, 10.0], 0.9),
            det([102.0, 0.0, 112.0, 10.0], 0.9),
        ]);
        let out2 = tracker.update(&second);
        assert_eq!(out2.detections[0].tracker_id, Some(id_left));
        assert_eq!(out2.detections[1].tracker_id, Some(id_right));
    }
}
