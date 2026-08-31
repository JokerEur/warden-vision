//! Mean Average Recall (mAR): mean, across classes and IoU thresholds, of
//! recall achieved using every available prediction (no confidence or
//! per-image detection-count cutoff).
//!
//! Doesn't cap the number of predictions considered per image (unlike
//! COCO's `AR@max_detections` family), so it corresponds most closely to
//! COCO's `AR@100`-style metric on scenes with fewer than about 100
//! detections.

use std::collections::HashMap;

use crate::core::Detections;
use crate::metrics::matching::match_counts;

/// The result of [`MeanAverageRecall::compute`].
#[derive(Debug, Clone)]
pub struct MeanAverageRecallResult {
    /// Mean recall across every class (with at least one ground-truth
    /// instance) and every configured IoU threshold.
    pub mar: f32,
    /// Recall per class id, averaged across every configured IoU
    /// threshold.
    pub recall_per_class: HashMap<usize, f32>,
}

#[derive(Debug, Clone, Default)]
struct ClassCounts {
    /// `(true_positives, false_negatives)` per configured IoU threshold.
    per_threshold: Vec<(usize, usize)>,
}

/// Computes mean average recall over one or more IoU thresholds,
/// accumulated across images via repeated [`MeanAverageRecall::update`]
/// calls.
#[derive(Debug, Clone)]
pub struct MeanAverageRecall {
    iou_thresholds: Vec<f32>,
    classes: HashMap<usize, ClassCounts>,
}

impl MeanAverageRecall {
    /// Creates a metric over an explicit set of IoU thresholds.
    pub fn new(iou_thresholds: Vec<f32>) -> Self {
        Self {
            iou_thresholds,
            classes: HashMap::new(),
        }
    }

    /// A metric over the ten COCO thresholds `0.50, 0.55, ..., 0.95`.
    pub fn coco() -> Self {
        Self::new((0..10).map(|i| 0.5 + 0.05 * i as f32).collect())
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

            let counts = self.classes.entry(class_id).or_insert_with(|| ClassCounts {
                per_threshold: vec![(0, 0); num_thresholds],
            });
            for (threshold_index, &threshold) in self.iou_thresholds.iter().enumerate() {
                let (tp, _fp, fn_) = match_counts(&class_predictions, &class_targets, threshold);
                counts.per_threshold[threshold_index].0 += tp;
                counts.per_threshold[threshold_index].1 += fn_;
            }
        }
        self
    }

    /// Computes mAR over every `update()` call so far. Classes never seen
    /// as a ground-truth target are excluded.
    pub fn compute(&self) -> MeanAverageRecallResult {
        let mut recall_per_class = HashMap::new();
        for (&class_id, counts) in &self.classes {
            let has_targets = counts.per_threshold.iter().any(|&(tp, fn_)| tp + fn_ > 0);
            if !has_targets {
                continue;
            }
            let sum: f32 = counts
                .per_threshold
                .iter()
                .map(|&(tp, fn_)| {
                    if tp + fn_ == 0 {
                        0.0
                    } else {
                        tp as f32 / (tp + fn_) as f32
                    }
                })
                .sum();
            recall_per_class.insert(class_id, sum / counts.per_threshold.len() as f32);
        }

        let mar = if recall_per_class.is_empty() {
            0.0
        } else {
            recall_per_class.values().sum::<f32>() / recall_per_class.len() as f32
        };

        MeanAverageRecallResult {
            mar,
            recall_per_class,
        }
    }
}

impl Default for MeanAverageRecall {
    fn default() -> Self {
        Self::coco()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Detection;

    fn det(bbox: [f32; 4], confidence: f32, class_id: usize) -> Detection {
        Detection::new(bbox, confidence, class_id)
    }

    #[test]
    fn perfect_predictions_score_recall_of_one() {
        let mut metric = MeanAverageRecall::new(vec![0.5]);
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        let predictions = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        metric.update(&predictions, &targets);
        assert!((metric.compute().mar - 1.0).abs() < 1e-4);
    }

    #[test]
    fn missed_detection_scores_recall_of_zero() {
        let mut metric = MeanAverageRecall::new(vec![0.5]);
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        metric.update(&Detections::empty(), &targets);
        assert_eq!(metric.compute().mar, 0.0);
    }

    #[test]
    fn accumulates_across_multiple_update_calls() {
        let mut metric = MeanAverageRecall::new(vec![0.5]);
        let targets1 = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        let predictions1 = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        metric.update(&predictions1, &targets1);

        let targets2 = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        metric.update(&Detections::empty(), &targets2);

        // One matched target and one missed target across two images: 50%.
        assert!((metric.compute().mar - 0.5).abs() < 1e-4);
    }

    #[test]
    fn extra_iou_thresholds_average_in_zero_when_unmet() {
        // Matches at 0.5 but the boxes aren't tight enough for 0.9.
        let mut metric = MeanAverageRecall::new(vec![0.5, 0.9]);
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        let predictions = Detections::new(vec![det([1.0, 1.0, 11.0, 11.0], 0.9, 0)]);
        metric.update(&predictions, &targets);
        let mar = metric.compute().mar;
        assert!(mar > 0.0 && mar < 1.0);
    }
}
