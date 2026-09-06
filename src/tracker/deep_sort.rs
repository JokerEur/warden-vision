//! An appearance-assisted SORT tracker: the same Kalman/IoU motion model as
//! [`crate::tracker::SortTracker`], with each track's identity additionally
//! anchored by a gallery of past appearance embeddings.
//!
//! Motion-only tracking (`SortTracker`/`ByteTracker`) loses identities when
//! two objects cross paths or one is briefly occluded, because IoU alone
//! can't tell them apart once their predicted boxes overlap. Real DeepSORT
//! fixes this with a re-identification embedding compared via a matching
//! cascade; this is a single-pass approximation of that idea, not a port of
//! the reference implementation (see [`crate::tracker::ByteTracker`]'s docs
//! for the same caveat about `ByteTrack`).
//!
//! This crate doesn't run the embedding model itself — consistent with
//! [`crate::core::adapters`] not owning inference either — so the caller
//! supplies one feature vector per detection each frame, from whatever
//! embedding network (a ReID net, or even a detector's own backbone
//! features) they're already running.

use std::collections::VecDeque;

use crate::core::Detections;
use crate::tracker::assignment::assign_by_cost;
use crate::tracker::kalman::KalmanBoxFilter;

struct Track {
    id: usize,
    filter: KalmanBoxFilter,
    time_since_update: usize,
    /// Most recent embeddings, newest last, capped at
    /// [`DeepSortTracker::feature_gallery_size`]. Matching against the
    /// whole gallery (nearest-neighbor distance) survives a track's
    /// appearance changing gradually (rotation, partial occlusion) better
    /// than comparing against only the single most recent embedding.
    features: VecDeque<Vec<f32>>,
}

/// Cosine distance (`1 - cosine similarity`) between two vectors, robust to
/// unnormalized input. Returns `1.0` (maximally dissimilar) if either
/// vector is zero.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a <= 0.0 || norm_b <= 0.0 {
        return 1.0;
    }
    (1.0 - dot / (norm_a * norm_b)).clamp(0.0, 2.0)
}

/// Cost used to hard-gate out a pairing whose IoU falls below `iou_gate`,
/// regardless of appearance similarity. Kept in the same "effectively
/// infinite but sum-safe in f32" range as `assignment::PAD_COST`, rather
/// than something like `f32::MAX`, since `lapjv`'s cost-matrix reduction
/// sums entries and a true `f32::MAX` sentinel could overflow to `inf`.
const HARD_GATE_COST: f32 = 1e6;

/// Smallest cosine distance from `query` to any embedding in `gallery`.
fn nearest_neighbor_distance(gallery: &VecDeque<Vec<f32>>, query: &[f32]) -> f32 {
    gallery
        .iter()
        .map(|f| cosine_distance(f, query))
        .fold(f32::INFINITY, f32::min)
}

/// A SORT tracker whose data association also weighs appearance similarity,
/// approximating DeepSORT.
///
/// Each call to [`DeepSortTracker::update`] advances every active track's
/// Kalman filter, then matches predicted boxes against detections using a
/// cost that combines `1 - iou` (motion) and nearest-neighbor cosine
/// distance over each track's embedding gallery (appearance), weighted by
/// [`DeepSortTracker::with_appearance_weight`]. A pairing is only ever
/// considered if its IoU clears `iou_gate` — appearance similarity alone
/// can't match two objects on opposite sides of the frame — which keeps
/// this a motion-gated refinement of SORT rather than a pure appearance
/// matcher.
pub struct DeepSortTracker {
    tracks: Vec<Track>,
    next_id: usize,
    max_age: usize,
    iou_gate: f32,
    appearance_weight: f32,
    max_cosine_distance: f32,
    feature_gallery_size: usize,
}

impl DeepSortTracker {
    /// Creates a tracker with `max_age`/`iou_gate` meaning the same as
    /// [`crate::tracker::SortTracker::new`], and defaults for the
    /// appearance side (`appearance_weight = 0.5`, `max_cosine_distance =
    /// 0.2`, `feature_gallery_size = 30`) — override with the `with_*`
    /// methods.
    pub fn new(max_age: usize, iou_gate: f32) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_age,
            iou_gate,
            appearance_weight: 0.5,
            max_cosine_distance: 0.2,
            feature_gallery_size: 30,
        }
    }

    /// Sets how much appearance similarity counts relative to motion (IoU)
    /// in the match cost: `0.0` degrades to motion-only (equivalent to
    /// [`crate::tracker::SortTracker`]), `1.0` to appearance-only (subject
    /// still to the hard `iou_gate`).
    pub fn with_appearance_weight(mut self, weight: f32) -> Self {
        self.appearance_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Sets the maximum nearest-neighbor cosine distance accepted as a
    /// match, in `[0, 2]` (`0` = identical direction, `2` = opposite).
    pub fn with_max_cosine_distance(mut self, distance: f32) -> Self {
        self.max_cosine_distance = distance;
        self
    }

    /// Sets how many past embeddings each track keeps for nearest-neighbor
    /// matching (oldest evicted first).
    pub fn with_feature_gallery_size(mut self, size: usize) -> Self {
        self.feature_gallery_size = size.max(1);
        self
    }

    /// Number of tracks currently being maintained (including ones missed
    /// this frame but not yet aged out).
    pub fn active_tracks(&self) -> usize {
        self.tracks.len()
    }

    /// Advances the tracker by one frame, assigning `tracker_id` to every
    /// detection in place.
    ///
    /// `embeddings` must have exactly one feature vector per detection, in
    /// the same order as `detections`; all vectors must share the same
    /// dimensionality (whatever your embedding model outputs).
    ///
    /// # Panics
    /// Panics if `embeddings.len() != detections.len()`.
    pub fn update(&mut self, detections: &mut Detections, embeddings: &[Vec<f32>]) {
        assert_eq!(
            detections.len(),
            embeddings.len(),
            "embeddings must have exactly one feature vector per detection"
        );

        let predicted_boxes: Vec<[f32; 4]> =
            self.tracks.iter_mut().map(|t| t.filter.predict()).collect();
        let detection_boxes: Vec<[f32; 4]> = detections.detections.iter().map(|d| d.bbox).collect();

        let no_match_cost = self.appearance_weight * self.max_cosine_distance
            + (1.0 - self.appearance_weight) * (1.0 - self.iou_gate);
        let tracks = &self.tracks;
        let iou_gate = self.iou_gate;
        let appearance_weight = self.appearance_weight;
        let matches = assign_by_cost(
            predicted_boxes.len(),
            detection_boxes.len(),
            no_match_cost,
            |i, j| {
                let iou = crate::core::bbox_iou(predicted_boxes[i], detection_boxes[j]);
                if iou < iou_gate {
                    return HARD_GATE_COST;
                }
                let appearance = nearest_neighbor_distance(&tracks[i].features, &embeddings[j]);
                appearance_weight * appearance + (1.0 - appearance_weight) * (1.0 - iou)
            },
        );

        let num_tracks = self.tracks.len();
        let mut track_matched = vec![false; num_tracks];
        for (detection_index, track_index) in matches.iter().enumerate() {
            if let Some(track_index) = track_index {
                let track = &mut self.tracks[*track_index];
                track
                    .filter
                    .update(detections.detections[detection_index].bbox);
                track.time_since_update = 0;
                track
                    .features
                    .push_back(embeddings[detection_index].clone());
                while track.features.len() > self.feature_gallery_size {
                    track.features.pop_front();
                }
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
                let mut features = VecDeque::with_capacity(self.feature_gallery_size);
                features.push_back(embeddings[detection_index].clone());
                self.tracks.push(Track {
                    id,
                    filter,
                    time_since_update: 0,
                    features,
                });
                detections.detections[detection_index].tracker_id = Some(id);
            }
        }

        self.tracks.retain(|t| t.time_since_update <= self.max_age);
    }
}

impl Default for DeepSortTracker {
    /// A tracker with the same motion defaults as [`crate::tracker::SortTracker::default`]
    /// (30-frame track lifetime, 0.3 IoU gate) plus the appearance defaults
    /// documented on [`DeepSortTracker::new`].
    fn default() -> Self {
        Self::new(30, 0.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Detection;

    fn embedding(direction: &[f32]) -> Vec<f32> {
        direction.to_vec()
    }

    #[test]
    fn new_detection_gets_a_fresh_tracker_id() {
        let mut tracker = DeepSortTracker::default();
        let mut detections = Detections::new(vec![Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        tracker.update(&mut detections, &[embedding(&[1.0, 0.0])]);
        assert!(detections.detections[0].tracker_id.is_some());
        assert_eq!(tracker.active_tracks(), 1);
    }

    #[test]
    #[should_panic(expected = "one feature vector per detection")]
    fn mismatched_embeddings_length_panics() {
        let mut tracker = DeepSortTracker::default();
        let mut detections = Detections::new(vec![Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        tracker.update(&mut detections, &[]);
    }

    #[test]
    fn matching_detection_with_similar_appearance_keeps_same_id() {
        let mut tracker = DeepSortTracker::new(5, 0.3);

        let mut first = Detections::new(vec![Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        tracker.update(&mut first, &[embedding(&[1.0, 0.0, 0.0])]);
        let id = first.detections[0].tracker_id.unwrap();

        let mut second = Detections::new(vec![Detection::new([1.0, 1.0, 11.0, 11.0], 0.9, 0)]);
        tracker.update(&mut second, &[embedding(&[0.9, 0.1, 0.0])]);

        assert_eq!(second.detections[0].tracker_id, Some(id));
    }

    #[test]
    fn appearance_disambiguates_two_tracks_with_identical_motion() {
        // Two tracks start at the exact same box, so their Kalman filters
        // predict identically (zero velocity) on every later frame: motion
        // (IoU) alone can never tell them apart. Their embeddings stay
        // distinct throughout - exactly the case `SortTracker` can't
        // handle but this tracker is built for.
        let mut tracker = DeepSortTracker::new(5, 0.1).with_appearance_weight(0.9);

        let mut frame1 = Detections::new(vec![
            Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0),
        ]);
        tracker.update(
            &mut frame1,
            &[embedding(&[1.0, 0.0, 0.0]), embedding(&[0.0, 1.0, 0.0])],
        );
        let id_a = frame1.detections[0].tracker_id.unwrap();
        let id_b = frame1.detections[1].tracker_id.unwrap();
        assert_ne!(id_a, id_b);

        // Same two overlapping boxes again, but with embeddings swapped
        // across detection order.
        let mut frame2 = Detections::new(vec![
            Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0),
        ]);
        tracker.update(
            &mut frame2,
            &[embedding(&[0.0, 1.0, 0.0]), embedding(&[1.0, 0.0, 0.0])],
        );

        // Detection 0 carries object B's embedding this frame, detection 1
        // carries object A's: appearance should route the ids accordingly,
        // since motion cost is tied between every track/detection pair.
        assert_eq!(frame2.detections[0].tracker_id, Some(id_b));
        assert_eq!(frame2.detections[1].tracker_id, Some(id_a));
    }

    #[test]
    fn feature_gallery_is_capped_at_the_configured_size() {
        let mut tracker = DeepSortTracker::new(30, 0.3).with_feature_gallery_size(3);
        let mut detections = Detections::new(vec![Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        for _ in 0..10 {
            tracker.update(&mut detections, &[embedding(&[1.0, 0.0])]);
        }
        assert_eq!(tracker.tracks[0].features.len(), 3);
    }

    #[test]
    fn track_is_removed_after_max_age_missed_frames() {
        let mut tracker = DeepSortTracker::new(2, 0.3);

        let mut first = Detections::new(vec![Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        tracker.update(&mut first, &[embedding(&[1.0, 0.0])]);
        assert_eq!(tracker.active_tracks(), 1);

        for _ in 0..3 {
            let mut empty = Detections::empty();
            tracker.update(&mut empty, &[]);
        }
        assert_eq!(tracker.active_tracks(), 0);
    }
}
