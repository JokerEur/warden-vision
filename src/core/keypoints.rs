//! Keypoint (pose/landmark) data structures, mirroring the role of
//! [`crate::core::Detections`] but for per-object sets of named joints
//! rather than a single bounding box.

/// A single detected joint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    /// X coordinate, in absolute pixel space.
    pub x: f32,
    /// Y coordinate, in absolute pixel space.
    pub y: f32,
    /// Detector confidence for this specific joint, typically in `[0, 1]`.
    pub confidence: f32,
}

impl Keypoint {
    /// Creates a new keypoint.
    pub fn new(x: f32, y: f32, confidence: f32) -> Self {
        Self { x, y, confidence }
    }
}

/// The joints detected for a single object instance.
///
/// `points[i]` is `None` when joint `i` (indexed per whatever skeleton
/// layout the caller is using, e.g. [`COCO_17_EDGES`]) was not detected for
/// this instance, keeping the slot present so joint index stays meaningful
/// even when some joints are missing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KeypointSet {
    /// Per-joint keypoints, indexed by joint id.
    pub points: Vec<Option<Keypoint>>,
    /// Index of the predicted class (e.g. which kind of skeleton this is).
    pub class_id: usize,
}

impl KeypointSet {
    /// Creates a new keypoint set.
    pub fn new(points: Vec<Option<Keypoint>>, class_id: usize) -> Self {
        Self { points, class_id }
    }

    /// Number of joint slots (detected or not).
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether this set has no joint slots at all.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The keypoint at `index`, if that joint was detected.
    pub fn get(&self, index: usize) -> Option<Keypoint> {
        self.points.get(index).copied().flatten()
    }
}

/// A collection of [`KeypointSet`]s produced for a single frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct KeyPoints {
    /// The underlying keypoint sets, one per detected object instance.
    pub keypoint_sets: Vec<KeypointSet>,
}

impl KeyPoints {
    /// Creates a `KeyPoints` from an existing vector.
    pub fn new(keypoint_sets: Vec<KeypointSet>) -> Self {
        Self { keypoint_sets }
    }

    /// An empty `KeyPoints`.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Number of object instances.
    pub fn len(&self) -> usize {
        self.keypoint_sets.len()
    }

    /// Whether there are no keypoint sets.
    pub fn is_empty(&self) -> bool {
        self.keypoint_sets.is_empty()
    }

    /// Iterates over keypoint sets by reference.
    pub fn iter(&self) -> std::slice::Iter<'_, KeypointSet> {
        self.keypoint_sets.iter()
    }

    /// Iterates over keypoint sets by mutable reference.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, KeypointSet> {
        self.keypoint_sets.iter_mut()
    }

    /// Returns a new `KeyPoints` with the same instances, but with any
    /// individual joint below `threshold` confidence (or missing) cleared
    /// to `None`. Unlike [`crate::core::Detections::filter_by_confidence`],
    /// this filters per-joint rather than dropping whole instances, since a
    /// single low-confidence joint (e.g. an occluded wrist) shouldn't
    /// discard an otherwise good pose estimate.
    pub fn filter_by_confidence(&self, threshold: f32) -> KeyPoints {
        let sets = self
            .keypoint_sets
            .iter()
            .map(|set| {
                let points = set
                    .points
                    .iter()
                    .map(|p| p.filter(|kp| kp.confidence >= threshold))
                    .collect();
                KeypointSet::new(points, set.class_id)
            })
            .collect();
        KeyPoints::new(sets)
    }
}

impl IntoIterator for KeyPoints {
    type Item = KeypointSet;
    type IntoIter = std::vec::IntoIter<KeypointSet>;

    fn into_iter(self) -> Self::IntoIter {
        self.keypoint_sets.into_iter()
    }
}

impl<'a> IntoIterator for &'a KeyPoints {
    type Item = &'a KeypointSet;
    type IntoIter = std::slice::Iter<'a, KeypointSet>;

    fn into_iter(self) -> Self::IntoIter {
        self.keypoint_sets.iter()
    }
}

/// The 19 bone connections of the standard 17-joint COCO pose skeleton, as
/// `(joint_index, joint_index)` pairs into a 0-indexed
/// `[nose, left_eye, right_eye, left_ear, right_ear, left_shoulder,
/// right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
/// left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]`
/// joint layout (COCO's official 1-indexed `skeleton` field, shifted down
/// by one).
pub const COCO_17_EDGES: &[(usize, usize)] = &[
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keypoint_set_get_returns_none_for_missing_joint() {
        let set = KeypointSet::new(vec![Some(Keypoint::new(1.0, 2.0, 0.9)), None], 0);
        assert!(set.get(0).is_some());
        assert!(set.get(1).is_none());
        assert!(set.get(5).is_none());
    }

    #[test]
    fn filter_by_confidence_clears_low_confidence_joints_only() {
        let sets = vec![KeypointSet::new(
            vec![
                Some(Keypoint::new(0.0, 0.0, 0.9)),
                Some(Keypoint::new(1.0, 1.0, 0.1)),
                None,
            ],
            0,
        )];
        let filtered = KeyPoints::new(sets).filter_by_confidence(0.5);
        assert!(filtered.keypoint_sets[0].get(0).is_some());
        assert!(filtered.keypoint_sets[0].get(1).is_none());
        assert!(filtered.keypoint_sets[0].get(2).is_none());
        // Instance itself is retained even though one joint was cleared.
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn coco_17_edges_reference_valid_joint_indices() {
        for &(a, b) in COCO_17_EDGES {
            assert!(a < 17);
            assert!(b < 17);
        }
        assert_eq!(COCO_17_EDGES.len(), 19);
    }
}
