//! Mean Average Precision (mAP): the standard object-detection accuracy
//! metric, averaging per-class average precision (area under the
//! precision-recall curve) across one or more IoU thresholds.
//!
//! Uses continuous (all-points) precision-recall integration rather than
//! COCO's 101-point discretization (the two converge to essentially the
//! same value), and skips COCO's per-image `max_detections` cap and
//! small/medium/large area breakdown.

use std::collections::HashMap;

use crate::core::Detections;
use crate::metrics::matching::match_records;

/// The result of [`MeanAveragePrecision::compute`].
#[derive(Debug, Clone)]
pub struct MeanAveragePrecisionResult {
    /// Mean AP across every class (with at least one ground-truth
    /// instance) and every configured IoU threshold.
    pub map: f32,
    /// Mean AP across classes, at each configured IoU threshold.
    pub map_per_threshold: Vec<(f32, f32)>,
    /// Average precision per class id, itself averaged across every
    /// configured IoU threshold.
    pub ap_per_class: HashMap<usize, f32>,
}

impl MeanAveragePrecisionResult {
    /// Mean AP at the given IoU threshold, if it was one of the thresholds
    /// this metric was configured with.
    pub fn map_at(&self, iou_threshold: f32) -> Option<f32> {
        self.map_per_threshold
            .iter()
            .find(|(t, _)| (*t - iou_threshold).abs() < 1e-6)
            .map(|(_, v)| *v)
    }
}

#[derive(Debug, Clone, Default)]
struct ClassAccumulator {
    num_targets: usize,
    /// One `Vec` of `(confidence, is_true_positive)` per configured IoU
    /// threshold, pooled across every `update()` call.
    per_threshold: Vec<Vec<(f32, bool)>>,
}

/// Computes mean average precision over one or more IoU thresholds,
/// accumulated across images via repeated [`MeanAveragePrecision::update`]
/// calls.
#[derive(Debug, Clone)]
pub struct MeanAveragePrecision {
    iou_thresholds: Vec<f32>,
    classes: HashMap<usize, ClassAccumulator>,
}

impl MeanAveragePrecision {
    /// Creates a metric over an explicit set of IoU thresholds.
    pub fn new(iou_thresholds: Vec<f32>) -> Self {
        Self {
            iou_thresholds,
            classes: HashMap::new(),
        }
    }

    /// A metric over the ten COCO thresholds `0.50, 0.55, ..., 0.95`
    /// (`mAP@[.5:.95]`).
    pub fn coco() -> Self {
        Self::new((0..10).map(|i| 0.5 + 0.05 * i as f32).collect())
    }

    /// A metric over the single threshold `0.5` (`mAP@0.5`).
    pub fn at_50() -> Self {
        Self::new(vec![0.5])
    }

    /// The IoU thresholds this metric averages over.
    pub fn iou_thresholds(&self) -> &[f32] {
        &self.iou_thresholds
    }

    /// Matches one image's `predictions` against its `targets` and folds
    /// the result into the running totals for every configured threshold.
    pub fn update(&mut self, predictions: &Detections, targets: &Detections) -> &mut Self {
        let mut class_ids: Vec<usize> = predictions
            .iter()
            .chain(targets.iter())
            .map(|d| d.class_id)
            .collect();
        class_ids.sort_unstable();
        class_ids.dedup();

        let num_thresholds = self.iou_thresholds.len();
        for class_id in class_ids {
            let class_predictions = predictions.filter_by_class(&[class_id]);
            let class_targets = targets.filter_by_class(&[class_id]);

            let accumulator = self
                .classes
                .entry(class_id)
                .or_insert_with(|| ClassAccumulator {
                    num_targets: 0,
                    per_threshold: vec![Vec::new(); num_thresholds],
                });
            accumulator.num_targets += class_targets.len();
            for (threshold_index, &threshold) in self.iou_thresholds.iter().enumerate() {
                let mut records = match_records(&class_predictions, &class_targets, threshold);
                accumulator.per_threshold[threshold_index].append(&mut records);
            }
        }
        self
    }

    /// Computes mAP over every `update()` call so far. Classes with no
    /// ground-truth instances across any update are excluded (matching
    /// standard practice: a class you never had a target for shouldn't
    /// drag down the average).
    pub fn compute(&self) -> MeanAveragePrecisionResult {
        let mut ap_per_class = HashMap::new();
        let mut threshold_sums = vec![0.0f32; self.iou_thresholds.len()];
        let mut threshold_counts = vec![0usize; self.iou_thresholds.len()];

        for (&class_id, accumulator) in &self.classes {
            if accumulator.num_targets == 0 {
                continue;
            }
            let mut class_sum = 0.0;
            for (threshold_index, records) in accumulator.per_threshold.iter().enumerate() {
                let ap = average_precision(records.clone(), accumulator.num_targets);
                threshold_sums[threshold_index] += ap;
                threshold_counts[threshold_index] += 1;
                class_sum += ap;
            }
            ap_per_class.insert(class_id, class_sum / self.iou_thresholds.len() as f32);
        }

        let map_per_threshold: Vec<(f32, f32)> = self
            .iou_thresholds
            .iter()
            .zip(threshold_sums.iter().zip(threshold_counts.iter()))
            .map(|(&t, (&sum, &count))| (t, if count == 0 { 0.0 } else { sum / count as f32 }))
            .collect();

        let map = if ap_per_class.is_empty() {
            0.0
        } else {
            ap_per_class.values().sum::<f32>() / ap_per_class.len() as f32
        };

        MeanAveragePrecisionResult {
            map,
            map_per_threshold,
            ap_per_class,
        }
    }
}

impl Default for MeanAveragePrecision {
    fn default() -> Self {
        Self::coco()
    }
}

/// Area under the precision-recall curve for one class/threshold, via
/// all-points (continuous) interpolation: precision is first made
/// monotonically non-increasing from the highest-recall end, then the
/// area is integrated directly against the (confidence-sorted) recall
/// steps.
fn average_precision(mut records: Vec<(f32, bool)>, num_targets: usize) -> f32 {
    if num_targets == 0 || records.is_empty() {
        return 0.0;
    }
    records.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut precisions = Vec::with_capacity(records.len());
    let mut recalls = Vec::with_capacity(records.len());
    let mut tp = 0usize;
    let mut fp = 0usize;
    for &(_, is_tp) in &records {
        if is_tp {
            tp += 1;
        } else {
            fp += 1;
        }
        precisions.push(tp as f32 / (tp + fp) as f32);
        recalls.push(tp as f32 / num_targets as f32);
    }

    for i in (0..precisions.len().saturating_sub(1)).rev() {
        precisions[i] = precisions[i].max(precisions[i + 1]);
    }

    let mut area = 0.0;
    let mut previous_recall = 0.0;
    for i in 0..recalls.len() {
        area += (recalls[i] - previous_recall) * precisions[i];
        previous_recall = recalls[i];
    }
    area
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Detection;

    fn det(bbox: [f32; 4], confidence: f32, class_id: usize) -> Detection {
        Detection::new(bbox, confidence, class_id)
    }

    #[test]
    fn perfect_predictions_score_ap_of_one() {
        let mut metric = MeanAveragePrecision::at_50();
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        let predictions = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        metric.update(&predictions, &targets);
        let result = metric.compute();
        assert!((result.map - 1.0).abs() < 1e-4);
    }

    #[test]
    fn missed_detection_scores_ap_of_zero() {
        let mut metric = MeanAveragePrecision::at_50();
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        let predictions = Detections::empty();
        metric.update(&predictions, &targets);
        let result = metric.compute();
        assert_eq!(result.map, 0.0);
    }

    #[test]
    fn class_with_no_targets_is_excluded_from_the_average() {
        let mut metric = MeanAveragePrecision::at_50();
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        // A confident but entirely wrong-class false positive shouldn't be
        // able to create a phantom class-1 entry that drags down the mean.
        let predictions = Detections::new(vec![
            det([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            det([50.0, 50.0, 60.0, 60.0], 0.99, 1),
        ]);
        metric.update(&predictions, &targets);
        let result = metric.compute();
        assert_eq!(result.ap_per_class.len(), 1);
        assert!((result.map - 1.0).abs() < 1e-4);
    }

    #[test]
    fn map_at_returns_none_for_unconfigured_threshold() {
        let metric = MeanAveragePrecision::at_50();
        let result = metric.compute();
        assert!(result.map_at(0.75).is_none());
        assert!(result.map_at(0.5).is_some());
    }

    #[test]
    fn false_positive_before_true_positive_lowers_ap_below_one() {
        let mut metric = MeanAveragePrecision::at_50();
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        let predictions = Detections::new(vec![
            // Higher-confidence false positive ranked ahead of the true positive.
            det([50.0, 50.0, 60.0, 60.0], 0.99, 0),
            det([0.0, 0.0, 10.0, 10.0], 0.5, 0),
        ]);
        metric.update(&predictions, &targets);
        let result = metric.compute();
        assert!(result.map < 1.0);
        assert!(result.map > 0.0);
    }
}
