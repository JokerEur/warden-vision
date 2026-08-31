//! A SORT-style multi-object tracker: Kalman-filter motion prediction plus
//! IoU-based data association solved as a linear assignment problem.

use crate::core::Detections;
use crate::tracker::assignment::assign_by_iou;
use crate::tracker::kalman::KalmanBoxFilter;

struct Track {
    id: usize,
    filter: KalmanBoxFilter,
    time_since_update: usize,
}

/// A SORT (Simple Online and Realtime Tracking) multi-object tracker.
///
/// Each call to [`SortTracker::update`] advances every active track's
/// Kalman filter by one frame, solves the assignment between predicted
/// track boxes and the frame's detections by intersection-over-union, and
/// writes the resulting identity into each detection's
/// [`tracker_id`](crate::core::Detection::tracker_id):
/// - Matched detections update their track's filter and keep its id.
/// - Unmatched detections spawn a new track with a freshly allocated id.
/// - Tracks left unmatched for more than `max_age` consecutive frames are
///   dropped.
pub struct SortTracker {
    tracks: Vec<Track>,
    next_id: usize,
    max_age: usize,
    iou_threshold: f32,
}

impl SortTracker {
    /// Creates a tracker.
    ///
    /// - `max_age`: number of consecutive missed frames a track tolerates
    ///   before being dropped.
    /// - `iou_threshold`: minimum IoU between a predicted track box and a
    ///   detection for them to be considered a match.
    pub fn new(max_age: usize, iou_threshold: f32) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_age,
            iou_threshold,
        }
    }

    /// Number of tracks currently being maintained (including ones missed
    /// this frame but not yet aged out).
    pub fn active_tracks(&self) -> usize {
        self.tracks.len()
    }

    /// Advances the tracker by one frame, assigning `tracker_id` to every
    /// detection in place.
    pub fn update(&mut self, detections: &mut Detections) {
        let predicted_boxes: Vec<[f32; 4]> =
            self.tracks.iter_mut().map(|t| t.filter.predict()).collect();

        let num_tracks = self.tracks.len();
        let detection_boxes: Vec<[f32; 4]> = detections.detections.iter().map(|d| d.bbox).collect();
        let matches = assign_by_iou(&predicted_boxes, &detection_boxes, self.iou_threshold);

        let mut track_matched = vec![false; num_tracks];
        for (detection_index, track_index) in matches.iter().enumerate() {
            if let Some(track_index) = track_index {
                let track = &mut self.tracks[*track_index];
                track
                    .filter
                    .update(detections.detections[detection_index].bbox);
                track.time_since_update = 0;
                detections.detections[detection_index].tracker_id = Some(track.id);
                track_matched[*track_index] = true;
            }
        }

        for (track, matched) in self.tracks.iter_mut().zip(track_matched.iter()) {
            if !matched {
                track.time_since_update += 1;
            }
        }

        for (detection_index, track_index) in matches.iter().enumerate() {
            if track_index.is_none() {
                let id = self.next_id;
                self.next_id += 1;
                let filter = KalmanBoxFilter::new(detections.detections[detection_index].bbox);
                self.tracks.push(Track {
                    id,
                    filter,
                    time_since_update: 0,
                });
                detections.detections[detection_index].tracker_id = Some(id);
            }
        }

        self.tracks.retain(|t| t.time_since_update <= self.max_age);
    }
}

impl Default for SortTracker {
    /// A tracker with commonly used SORT defaults: 30-frame track lifetime,
    /// 0.3 IoU match threshold.
    fn default() -> Self {
        Self::new(30, 0.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Detection;

    #[test]
    fn new_detection_gets_a_fresh_tracker_id() {
        let mut tracker = SortTracker::default();
        let mut detections = Detections::new(vec![Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        tracker.update(&mut detections);
        assert!(detections.detections[0].tracker_id.is_some());
        assert_eq!(tracker.active_tracks(), 1);
    }

    #[test]
    fn matching_detection_keeps_same_tracker_id() {
        let mut tracker = SortTracker::new(5, 0.3);

        let mut first = Detections::new(vec![Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        tracker.update(&mut first);
        let id = first.detections[0].tracker_id.unwrap();

        let mut second = Detections::new(vec![Detection::new([1.0, 1.0, 11.0, 11.0], 0.9, 0)]);
        tracker.update(&mut second);

        assert_eq!(second.detections[0].tracker_id, Some(id));
        assert_eq!(tracker.active_tracks(), 1);
    }

    #[test]
    fn two_independent_objects_maintain_separate_identities() {
        let mut tracker = SortTracker::new(5, 0.3);

        let mut first = Detections::new(vec![
            Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            Detection::new([100.0, 0.0, 110.0, 10.0], 0.9, 0),
        ]);
        tracker.update(&mut first);
        let id_left = first.detections[0].tracker_id.unwrap();
        let id_right = first.detections[1].tracker_id.unwrap();
        assert_ne!(id_left, id_right);

        let mut second = Detections::new(vec![
            Detection::new([2.0, 0.0, 12.0, 10.0], 0.9, 0),
            Detection::new([102.0, 0.0, 112.0, 10.0], 0.9, 0),
        ]);
        tracker.update(&mut second);

        assert_eq!(second.detections[0].tracker_id, Some(id_left));
        assert_eq!(second.detections[1].tracker_id, Some(id_right));
    }

    #[test]
    fn far_apart_detection_does_not_match_existing_track() {
        let mut tracker = SortTracker::new(5, 0.3);

        let mut first = Detections::new(vec![Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        tracker.update(&mut first);
        let id1 = first.detections[0].tracker_id.unwrap();

        let mut second =
            Detections::new(vec![Detection::new([200.0, 200.0, 210.0, 210.0], 0.9, 0)]);
        tracker.update(&mut second);
        let id2 = second.detections[0].tracker_id.unwrap();

        assert_ne!(id1, id2);
        // The original track is still alive (missed one frame, within
        // max_age), plus the new one: two active tracks.
        assert_eq!(tracker.active_tracks(), 2);
    }

    #[test]
    fn track_is_removed_after_max_age_missed_frames() {
        let mut tracker = SortTracker::new(2, 0.3);

        let mut first = Detections::new(vec![Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        tracker.update(&mut first);
        assert_eq!(tracker.active_tracks(), 1);

        for _ in 0..2 {
            let mut empty = Detections::empty();
            tracker.update(&mut empty);
        }
        assert_eq!(
            tracker.active_tracks(),
            1,
            "should survive up to max_age misses"
        );

        let mut empty = Detections::empty();
        tracker.update(&mut empty);
        assert_eq!(
            tracker.active_tracks(),
            0,
            "should be dropped once max_age is exceeded"
        );
    }
}
