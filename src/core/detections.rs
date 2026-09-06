//! Core detection data structures shared across the crate.
//!
//! Mirrors the role of `supervision.Detections` in the Python library: a
//! single [`Detections`] value represents everything a detector (or tracker)
//! produced for one frame, and downstream modules (geometry zones,
//! trackers, annotators) all operate on it.

use ndarray::{Array1, Array2};

use crate::geometry::{
    oriented_box_area, oriented_box_iou, polygon_centroid, Point, Position, Rect,
};

/// A single detected object.
///
/// Coordinates in `bbox` are `[x1, y1, x2, y2]` in absolute pixel space,
/// following the top-left/bottom-right convention used throughout
/// `supervision`.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Bounding box as `[x1, y1, x2, y2]`.
    pub bbox: [f32; 4],
    /// Detector confidence score, typically in `[0, 1]`.
    pub confidence: f32,
    /// Index of the predicted class.
    pub class_id: usize,
    /// Identity assigned by a multi-object tracker, if any.
    pub tracker_id: Option<usize>,
    /// Optional instance segmentation mask, stored as a polygon contour.
    pub mask: Option<Vec<[f32; 2]>>,
    /// Optional oriented (rotated) bounding box, as four corner points in
    /// either winding order. When set, [`Detection::iou`] and
    /// [`Detection::area`] use it instead of the axis-aligned `bbox`,
    /// which stays populated as its envelope so untouched code (NMS,
    /// trackers, zones) keeps working. Build one with [`Detection::from_obb`]
    /// rather than setting this field directly, so `bbox` stays consistent
    /// with it.
    pub obb: Option<[[f32; 2]; 4]>,
}

impl Detection {
    /// Creates a new detection with no tracker id, mask, or oriented box.
    pub fn new(bbox: [f32; 4], confidence: f32, class_id: usize) -> Self {
        Self {
            bbox,
            confidence,
            class_id,
            tracker_id: None,
            mask: None,
            obb: None,
        }
    }

    /// Creates a detection from an oriented bounding box's four corners
    /// (any consistent winding order), e.g. from a YOLOv8-OBB-style
    /// detector.
    ///
    /// `bbox` is set to the corners' axis-aligned envelope, so this
    /// detection still works with everything that only understands
    /// axis-aligned boxes (NMS ordering by confidence, [`crate::tracker`]
    /// motion association, [`crate::geometry`] zones). `mask` is set to
    /// the same four corners, so [`crate::annotators::PolygonAnnotator`] /
    /// [`crate::annotators::MaskAnnotator`] render the rotated rectangle
    /// with no separate oriented-box annotator needed.
    pub fn from_obb(corners: [[f32; 2]; 4], confidence: f32, class_id: usize) -> Self {
        let xs = corners.iter().map(|c| c[0]);
        let ys = corners.iter().map(|c| c[1]);
        let x1 = xs.clone().fold(f32::INFINITY, f32::min);
        let x2 = xs.fold(f32::NEG_INFINITY, f32::max);
        let y1 = ys.clone().fold(f32::INFINITY, f32::min);
        let y2 = ys.fold(f32::NEG_INFINITY, f32::max);
        Self {
            bbox: [x1, y1, x2, y2],
            confidence,
            class_id,
            tracker_id: None,
            mask: Some(corners.to_vec()),
            obb: Some(corners),
        }
    }

    /// Width of the bounding box (`x2 - x1`).
    pub fn width(&self) -> f32 {
        self.bbox[2] - self.bbox[0]
    }

    /// Height of the bounding box (`y2 - y1`).
    pub fn height(&self) -> f32 {
        self.bbox[3] - self.bbox[1]
    }

    /// Area of the detection: the true rotated-quad area when
    /// [`Detection::obb`] is set (which is generally smaller than its
    /// axis-aligned envelope), otherwise the bounding box's area.
    pub fn area(&self) -> f32 {
        if let Some(obb) = &self.obb {
            return oriented_box_area(&corners_to_points(obb));
        }
        self.width().max(0.0) * self.height().max(0.0)
    }

    /// Centroid `(x, y)` of the bounding box.
    pub fn centroid(&self) -> (f32, f32) {
        (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )
    }

    /// Intersection-over-union with another detection.
    ///
    /// Uses rotated-polygon IoU over [`Detection::obb`] when both
    /// detections have one (axis-aligned IoU over-counts overlap for
    /// thin, rotated objects), falling back to axis-aligned `bbox` IoU
    /// otherwise.
    pub fn iou(&self, other: &Detection) -> f32 {
        if let (Some(a), Some(b)) = (&self.obb, &other.obb) {
            return oriented_box_iou(&corners_to_points(a), &corners_to_points(b));
        }
        bbox_iou(self.bbox, other.bbox)
    }

    /// The point at a named [`Position`] on this detection.
    ///
    /// [`Position::CenterOfMass`] uses the centroid of [`Detection::mask`]
    /// when present, falling back to the bounding box center otherwise;
    /// every other position is read off the bounding box.
    pub fn anchor_point(&self, position: Position) -> Point {
        if position == Position::CenterOfMass {
            if let Some(mask) = &self.mask {
                if !mask.is_empty() {
                    let polygon: Vec<Point> = mask.iter().map(|&[x, y]| Point::new(x, y)).collect();
                    return polygon_centroid(&polygon);
                }
            }
            return Rect::from_xyxy(self.bbox).center();
        }
        Rect::from_xyxy(self.bbox).anchor(position)
    }
}

fn corners_to_points(corners: &[[f32; 2]; 4]) -> [Point; 4] {
    corners.map(|[x, y]| Point::new(x, y))
}

/// Intersection-over-union of two `[x1, y1, x2, y2]` bounding boxes.
///
/// Shared with [`crate::tracker`], which computes IoU between raw
/// Kalman-predicted boxes and detections without constructing full
/// [`Detection`] values.
pub(crate) fn bbox_iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let [ax1, ay1, ax2, ay2] = a;
    let [bx1, by1, bx2, by2] = b;

    let ix1 = ax1.max(bx1);
    let iy1 = ay1.max(by1);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);

    let intersection = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let area_a = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0);
    let area_b = (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0);
    let union = area_a + area_b - intersection;

    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// A collection of [`Detection`]s produced for a single frame.
///
/// Analogous to `supervision.Detections`, but represented as a plain
/// `Vec<Detection>` rather than struct-of-arrays; this keeps per-detection
/// fields (like `mask`, which is variable-length) simple to model in Rust.
/// Use [`Detections::xyxy`], [`Detections::confidence`], and
/// [`Detections::class_id`] to obtain `ndarray` views for batch/vectorized
/// processing when needed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Detections {
    /// The underlying detections.
    pub detections: Vec<Detection>,
}

impl Detections {
    /// Creates a `Detections` from an existing vector.
    pub fn new(detections: Vec<Detection>) -> Self {
        Self { detections }
    }

    /// An empty `Detections`.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Number of detections.
    pub fn len(&self) -> usize {
        self.detections.len()
    }

    /// Whether there are no detections.
    pub fn is_empty(&self) -> bool {
        self.detections.is_empty()
    }

    /// Iterates over detections by reference.
    pub fn iter(&self) -> std::slice::Iter<'_, Detection> {
        self.detections.iter()
    }

    /// Iterates over detections by mutable reference.
    ///
    /// Trackers use this to populate [`Detection::tracker_id`] in place.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Detection> {
        self.detections.iter_mut()
    }

    /// Returns a new `Detections` containing only detections with
    /// `confidence >= threshold`.
    pub fn filter_by_confidence(&self, threshold: f32) -> Detections {
        self.filter(|d| d.confidence >= threshold)
    }

    /// Returns a new `Detections` containing only detections whose
    /// `class_id` is present in `class_ids`.
    pub fn filter_by_class(&self, class_ids: &[usize]) -> Detections {
        self.filter(|d| class_ids.contains(&d.class_id))
    }

    /// Returns a new `Detections` containing only detections matching
    /// `predicate`.
    pub fn filter<F>(&self, predicate: F) -> Detections
    where
        F: Fn(&Detection) -> bool,
    {
        Detections::new(
            self.detections
                .iter()
                .filter(|d| predicate(d))
                .cloned()
                .collect(),
        )
    }

    /// Greedy non-maximum suppression: keeps the highest-confidence
    /// detection among any group whose boxes overlap by more than
    /// `iou_threshold`, discarding the rest.
    ///
    /// When `class_agnostic` is `false` (the common case for a
    /// multi-class detector), suppression only compares detections that
    /// share a `class_id`, so overlapping boxes of different classes are
    /// both kept. When `true`, overlap is compared across all detections
    /// regardless of class.
    pub fn non_max_suppression(&self, iou_threshold: f32, class_agnostic: bool) -> Detections {
        let order = Self::confidence_order(&self.detections);
        let mut suppressed = vec![false; self.len()];
        let mut kept = Vec::with_capacity(self.len());

        for pos in 0..order.len() {
            let i = order[pos];
            if suppressed[i] {
                continue;
            }
            kept.push(self.detections[i].clone());
            // Only the boxes ranked below `i` still need a keep/suppress
            // decision: anything ranked above `i` was already resolved
            // (kept or suppressed) on an earlier iteration, so comparing
            // against it again can't change the output.
            for &j in &order[pos + 1..] {
                if suppressed[j] {
                    continue;
                }
                let same_class =
                    class_agnostic || self.detections[i].class_id == self.detections[j].class_id;
                if same_class && self.detections[i].iou(&self.detections[j]) > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }

        Detections::new(kept)
    }

    /// Indices into `detections`, sorted by descending confidence.
    fn confidence_order(detections: &[Detection]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..detections.len()).collect();
        order.sort_by(|&a, &b| {
            detections[b]
                .confidence
                .total_cmp(&detections[a].confidence)
        });
        order
    }

    /// Same semantics as [`Detections::non_max_suppression`], but the
    /// expensive part — per-pair IoU — is precomputed with `rayon` in one
    /// parallel pass before the (inherently sequential) greedy keep/suppress
    /// bookkeeping runs.
    ///
    /// This two-phase split, rather than parallelizing the suppression scan
    /// directly, is deliberate: the naive approach of dispatching a `rayon`
    /// job per surviving candidate at every step of the sequential loop was
    /// tried first and measured *slower* than the plain sequential version
    /// (see `examples/bench_vs_supervision.rs`) — thousands of small,
    /// decreasing-size parallel dispatches lose to their own overhead.
    /// Precomputing the whole pairwise relation up front is one large,
    /// evenly-sized parallel job, which is what `rayon` is actually good at;
    /// the sequential pass left over is then just boolean lookups, no
    /// floating-point IoU math.
    ///
    /// Trades `O(n^2)` bits of scratch memory for that speedup — worthwhile
    /// once a frame has hundreds to thousands of candidate boxes (e.g.
    /// before merging an [`crate::core::InferenceSlicer`]'s tiles); for a
    /// detector's usual few dozen boxes, or when boxes carry an
    /// [`crate::core::Detection::obb`] (rotated-polygon IoU is
    /// meaningfully more expensive than axis-aligned), the crossover point
    /// where this wins moves lower.
    #[cfg(feature = "parallel")]
    pub fn non_max_suppression_parallel(
        &self,
        iou_threshold: f32,
        class_agnostic: bool,
    ) -> Detections {
        use rayon::prelude::*;

        let order = Self::confidence_order(&self.detections);
        let n = order.len();

        // `would_suppress[pos][k]`: does `order[pos]` suppress
        // `order[pos + 1 + k]`? This relation only depends on the raw
        // detections, not on suppression bookkeeping, so every row is
        // independent and safe to compute in parallel.
        let would_suppress: Vec<Vec<bool>> = (0..n)
            .into_par_iter()
            .map(|pos| {
                let i = order[pos];
                order[pos + 1..]
                    .iter()
                    .map(|&j| {
                        let same_class = class_agnostic
                            || self.detections[i].class_id == self.detections[j].class_id;
                        same_class && self.detections[i].iou(&self.detections[j]) > iou_threshold
                    })
                    .collect()
            })
            .collect();

        let mut suppressed = vec![false; n];
        let mut kept = Vec::with_capacity(n);
        for pos in 0..n {
            let i = order[pos];
            if suppressed[i] {
                continue;
            }
            kept.push(self.detections[i].clone());
            for (k, &j) in order[pos + 1..].iter().enumerate() {
                if would_suppress[pos][k] {
                    suppressed[j] = true;
                }
            }
        }

        Detections::new(kept)
    }

    /// Merges multiple `Detections` (e.g. from multiple models) into one.
    pub fn merge(sets: &[Detections]) -> Detections {
        let mut merged = Vec::with_capacity(sets.iter().map(Detections::len).sum());
        for set in sets {
            merged.extend(set.detections.iter().cloned());
        }
        Detections::new(merged)
    }

    /// Merges `self` with `other`, returning a new combined `Detections`.
    pub fn merge_with(&self, other: &Detections) -> Detections {
        Detections::merge(&[self.clone(), other.clone()])
    }

    /// Returns a new `Detections` with every bounding box (and mask
    /// polygon, if present) scaled by `(sx, sy)` about the origin.
    ///
    /// Useful for mapping detections from a resized/model-input image
    /// back to the original image's coordinate space, e.g. after
    /// [`crate::core::Detections::from_ultralytics_onnx`] on a
    /// letterboxed input: `sx = sy = 1.0 / letterbox_scale`.
    pub fn scale(&self, sx: f32, sy: f32) -> Detections {
        Detections::new(
            self.detections
                .iter()
                .map(|d| {
                    let mut scaled = d.clone();
                    let [x1, y1, x2, y2] = d.bbox;
                    scaled.bbox = [x1 * sx, y1 * sy, x2 * sx, y2 * sy];
                    scaled.mask = d
                        .mask
                        .as_ref()
                        .map(|polygon| polygon.iter().map(|&[x, y]| [x * sx, y * sy]).collect());
                    scaled.obb = d.obb.map(|corners| corners.map(|[x, y]| [x * sx, y * sy]));
                    scaled
                })
                .collect(),
        )
    }

    /// Bounding boxes as an `N x 4` `ndarray` (`[x1, y1, x2, y2]` per row),
    /// suitable for batch/vectorized geometry or IoU computations.
    pub fn xyxy(&self) -> Array2<f32> {
        let mut arr = Array2::<f32>::zeros((self.len(), 4));
        for (i, d) in self.detections.iter().enumerate() {
            arr.row_mut(i).assign(&Array1::from_vec(d.bbox.to_vec()));
        }
        arr
    }

    /// Confidence scores as a 1-D `ndarray`.
    pub fn confidence(&self) -> Array1<f32> {
        Array1::from_iter(self.detections.iter().map(|d| d.confidence))
    }

    /// Class ids as a 1-D `ndarray`.
    pub fn class_id(&self) -> Array1<usize> {
        Array1::from_iter(self.detections.iter().map(|d| d.class_id))
    }

    /// Tracker ids as a 1-D `ndarray` of `Option<usize>`.
    pub fn tracker_id(&self) -> Array1<Option<usize>> {
        Array1::from_iter(self.detections.iter().map(|d| d.tracker_id))
    }
}

impl IntoIterator for Detections {
    type Item = Detection;
    type IntoIter = std::vec::IntoIter<Detection>;

    fn into_iter(self) -> Self::IntoIter {
        self.detections.into_iter()
    }
}

impl<'a> IntoIterator for &'a Detections {
    type Item = &'a Detection;
    type IntoIter = std::slice::Iter<'a, Detection>;

    fn into_iter(self) -> Self::IntoIter {
        self.detections.iter()
    }
}

impl FromIterator<Detection> for Detections {
    fn from_iter<T: IntoIterator<Item = Detection>>(iter: T) -> Self {
        Detections::new(iter.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(bbox: [f32; 4], confidence: f32, class_id: usize) -> Detection {
        Detection::new(bbox, confidence, class_id)
    }

    #[test]
    fn width_height_area() {
        let d = det([0.0, 0.0, 10.0, 4.0], 0.9, 0);
        assert_eq!(d.width(), 10.0);
        assert_eq!(d.height(), 4.0);
        assert_eq!(d.area(), 40.0);
    }

    #[test]
    fn centroid_is_bbox_center() {
        let d = det([0.0, 0.0, 10.0, 10.0], 0.9, 0);
        assert_eq!(d.centroid(), (5.0, 5.0));
    }

    #[test]
    fn iou_of_identical_boxes_is_one() {
        let a = det([0.0, 0.0, 10.0, 10.0], 0.9, 0);
        let b = det([0.0, 0.0, 10.0, 10.0], 0.5, 1);
        assert!((a.iou(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_of_disjoint_boxes_is_zero() {
        let a = det([0.0, 0.0, 10.0, 10.0], 0.9, 0);
        let b = det([20.0, 20.0, 30.0, 30.0], 0.9, 0);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn iou_of_partial_overlap() {
        let a = det([0.0, 0.0, 10.0, 10.0], 0.9, 0);
        let b = det([5.0, 0.0, 15.0, 10.0], 0.9, 0);
        // intersection: 5x10 = 50, union: 100 + 100 - 50 = 150
        assert!((a.iou(&b) - (50.0 / 150.0)).abs() < 1e-6);
    }

    #[test]
    fn filter_by_confidence_keeps_only_above_threshold() {
        let dets = Detections::new(vec![
            det([0.0, 0.0, 1.0, 1.0], 0.9, 0),
            det([0.0, 0.0, 1.0, 1.0], 0.2, 0),
        ]);
        let filtered = dets.filter_by_confidence(0.5);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered.detections[0].confidence, 0.9);
    }

    #[test]
    fn filter_by_class_keeps_only_matching_classes() {
        let dets = Detections::new(vec![
            det([0.0, 0.0, 1.0, 1.0], 0.9, 0),
            det([0.0, 0.0, 1.0, 1.0], 0.9, 1),
            det([0.0, 0.0, 1.0, 1.0], 0.9, 2),
        ]);
        let filtered = dets.filter_by_class(&[0, 2]);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.detections.iter().all(|d| d.class_id != 1));
    }

    #[test]
    fn scale_multiplies_bbox_and_mask_coordinates() {
        let mut d = det([1.0, 2.0, 3.0, 4.0], 0.9, 0);
        d.mask = Some(vec![[1.0, 2.0], [3.0, 4.0]]);
        let dets = Detections::new(vec![d]);
        let scaled = dets.scale(2.0, 3.0);
        assert_eq!(scaled.detections[0].bbox, [2.0, 6.0, 6.0, 12.0]);
        assert_eq!(
            scaled.detections[0].mask,
            Some(vec![[2.0, 6.0], [6.0, 12.0]])
        );
    }

    #[test]
    fn merge_combines_all_detections() {
        let a = Detections::new(vec![det([0.0, 0.0, 1.0, 1.0], 0.9, 0)]);
        let b = Detections::new(vec![det([1.0, 1.0, 2.0, 2.0], 0.8, 1)]);
        let merged = Detections::merge(&[a.clone(), b.clone()]);
        assert_eq!(merged.len(), 2);

        let merged2 = a.merge_with(&b);
        assert_eq!(merged2.len(), 2);
    }

    #[test]
    fn nms_suppresses_lower_confidence_overlapping_same_class_box() {
        let dets = Detections::new(vec![
            det([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            det([1.0, 1.0, 11.0, 11.0], 0.6, 0),
        ]);
        let kept = dets.non_max_suppression(0.5, false);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept.detections[0].confidence, 0.9);
    }

    #[test]
    fn nms_keeps_non_overlapping_boxes() {
        let dets = Detections::new(vec![
            det([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            det([100.0, 100.0, 110.0, 110.0], 0.6, 0),
        ]);
        let kept = dets.non_max_suppression(0.5, false);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn nms_class_aware_keeps_overlapping_different_class_boxes() {
        let dets = Detections::new(vec![
            det([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            det([1.0, 1.0, 11.0, 11.0], 0.6, 1),
        ]);
        let kept = dets.non_max_suppression(0.5, false);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn nms_class_agnostic_suppresses_across_classes() {
        let dets = Detections::new(vec![
            det([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            det([1.0, 1.0, 11.0, 11.0], 0.6, 1),
        ]);
        let kept = dets.non_max_suppression(0.5, true);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept.detections[0].confidence, 0.9);
    }

    #[test]
    fn xyxy_matches_bboxes() {
        let dets = Detections::new(vec![
            det([0.0, 1.0, 2.0, 3.0], 0.9, 0),
            det([4.0, 5.0, 6.0, 7.0], 0.9, 0),
        ]);
        let arr = dets.xyxy();
        assert_eq!(arr.shape(), &[2, 4]);
        assert_eq!(arr.row(0).to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(arr.row(1).to_vec(), vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn empty_detections_has_zero_length() {
        let dets = Detections::empty();
        assert!(dets.is_empty());
        assert_eq!(dets.xyxy().shape(), &[0, 4]);
    }

    fn square_obb(cx: f32, cy: f32, half: f32) -> [[f32; 2]; 4] {
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ]
    }

    #[test]
    fn from_obb_sets_bbox_to_the_axis_aligned_envelope() {
        // A 45deg-rotated square's envelope is bigger than the square
        // itself, which is exactly why axis-aligned NMS/tracking still
        // needs this envelope even when `obb` carries the true shape.
        let corners = [[10.0, 0.0], [20.0, 10.0], [10.0, 20.0], [0.0, 10.0]];
        let d = Detection::from_obb(corners, 0.9, 0);
        assert_eq!(d.bbox, [0.0, 0.0, 20.0, 20.0]);
        assert_eq!(d.obb, Some(corners));
    }

    #[test]
    fn from_obb_also_populates_mask_for_existing_polygon_annotators() {
        let corners = square_obb(10.0, 10.0, 5.0);
        let d = Detection::from_obb(corners, 0.9, 0);
        assert_eq!(d.mask, Some(corners.to_vec()));
    }

    #[test]
    fn area_uses_true_rotated_area_not_the_envelope() {
        let corners = [[10.0, 0.0], [20.0, 10.0], [10.0, 20.0], [0.0, 10.0]];
        let d = Detection::from_obb(corners, 0.9, 0);
        // Diamond of diagonal 20 has area 200, well under its 20x20 = 400
        // axis-aligned envelope.
        assert!((d.area() - 200.0).abs() < 1e-3);
    }

    #[test]
    fn iou_prefers_obb_over_axis_aligned_bbox_when_both_have_one() {
        let a = Detection::from_obb(square_obb(0.0, 0.0, 10.0), 0.9, 0);
        let rotated = square_obb(0.0, 0.0, 10.0).map(|[x, y]| {
            let theta: f32 = 45.0_f32.to_radians();
            [
                x * theta.cos() - y * theta.sin(),
                x * theta.sin() + y * theta.cos(),
            ]
        });
        let b = Detection::from_obb(rotated, 0.9, 0);
        // Both envelopes fully overlap (axis-aligned IoU would read close
        // to 1.0 for the rotated square's bounding box), but the actual
        // rotated shapes overlap far less.
        assert!(a.iou(&b) < 0.8, "iou was {}", a.iou(&b));
    }

    #[test]
    fn iou_falls_back_to_bbox_when_only_one_side_has_obb() {
        let a = Detection::from_obb(square_obb(0.0, 0.0, 5.0), 0.9, 0);
        let b = det([-5.0, -5.0, 5.0, 5.0], 0.9, 0);
        assert!((a.iou(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn scale_also_scales_obb_corners() {
        let d = Detection::from_obb(square_obb(1.0, 1.0, 1.0), 0.9, 0);
        let dets = Detections::new(vec![d]);
        let scaled = dets.scale(2.0, 3.0);
        assert_eq!(
            scaled.detections[0].obb,
            Some([[0.0, 0.0], [4.0, 0.0], [4.0, 6.0], [0.0, 6.0]])
        );
    }

    #[test]
    fn nms_keeps_only_the_remaining_scan_reachable_by_position() {
        // Regression guard for the "scan only order[pos+1..]" optimization:
        // three heavily-overlapping same-class boxes should still collapse
        // to exactly the highest-confidence one, not an artifact of
        // scanning order.
        let dets = Detections::new(vec![
            det([0.0, 0.0, 10.0, 10.0], 0.5, 0),
            det([0.5, 0.5, 10.5, 10.5], 0.95, 0),
            det([1.0, 1.0, 11.0, 11.0], 0.7, 0),
        ]);
        let kept = dets.non_max_suppression(0.3, false);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept.detections[0].confidence, 0.95);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn non_max_suppression_parallel_matches_the_sequential_result() {
        let dets = Detections::new(vec![
            det([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            det([1.0, 1.0, 11.0, 11.0], 0.6, 0),
            det([100.0, 100.0, 110.0, 110.0], 0.8, 0),
            det([1.0, 1.0, 11.0, 11.0], 0.4, 1),
        ]);
        let sequential = dets.non_max_suppression(0.5, false);
        let parallel = dets.non_max_suppression_parallel(0.5, false);
        assert_eq!(sequential.len(), parallel.len());
        let mut seq_conf: Vec<f32> = sequential.iter().map(|d| d.confidence).collect();
        let mut par_conf: Vec<f32> = parallel.iter().map(|d| d.confidence).collect();
        seq_conf.sort_by(f32::total_cmp);
        par_conf.sort_by(f32::total_cmp);
        assert_eq!(seq_conf, par_conf);
    }
}
