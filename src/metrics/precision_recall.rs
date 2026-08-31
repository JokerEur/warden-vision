//! Precision, recall, and F1 score, matched per image at a fixed IoU
//! threshold and accumulated across `update()` calls.

use crate::core::Detections;
use crate::metrics::matching::match_counts;

#[derive(Debug, Clone, Copy, Default)]
struct Counts {
    tp: usize,
    fp: usize,
    fn_: usize,
}

impl Counts {
    fn add(&mut self, predictions: &Detections, targets: &Detections, iou_threshold: f32) {
        let (tp, fp, fn_) = match_counts(predictions, targets, iou_threshold);
        self.tp += tp;
        self.fp += fp;
        self.fn_ += fn_;
    }

    fn precision(&self) -> f32 {
        if self.tp + self.fp == 0 {
            0.0
        } else {
            self.tp as f32 / (self.tp + self.fp) as f32
        }
    }

    fn recall(&self) -> f32 {
        if self.tp + self.fn_ == 0 {
            0.0
        } else {
            self.tp as f32 / (self.tp + self.fn_) as f32
        }
    }

    fn f1(&self) -> f32 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

/// Precision (`TP / (TP + FP)`) at a fixed IoU threshold, accumulated
/// across images.
#[derive(Debug, Clone)]
pub struct Precision {
    iou_threshold: f32,
    counts: Counts,
}

impl Precision {
    /// Creates a metric matching predictions to targets at `iou_threshold`.
    pub fn new(iou_threshold: f32) -> Self {
        Self {
            iou_threshold,
            counts: Counts::default(),
        }
    }

    /// Matches one image's `predictions` against its `targets` and folds
    /// the result into the running total.
    pub fn update(&mut self, predictions: &Detections, targets: &Detections) -> &mut Self {
        self.counts.add(predictions, targets, self.iou_threshold);
        self
    }

    /// Precision over every `update()` call so far.
    pub fn compute(&self) -> f32 {
        self.counts.precision()
    }
}

/// Recall (`TP / (TP + FN)`) at a fixed IoU threshold, accumulated across
/// images.
#[derive(Debug, Clone)]
pub struct Recall {
    iou_threshold: f32,
    counts: Counts,
}

impl Recall {
    /// Creates a metric matching predictions to targets at `iou_threshold`.
    pub fn new(iou_threshold: f32) -> Self {
        Self {
            iou_threshold,
            counts: Counts::default(),
        }
    }

    /// Matches one image's `predictions` against its `targets` and folds
    /// the result into the running total.
    pub fn update(&mut self, predictions: &Detections, targets: &Detections) -> &mut Self {
        self.counts.add(predictions, targets, self.iou_threshold);
        self
    }

    /// Recall over every `update()` call so far.
    pub fn compute(&self) -> f32 {
        self.counts.recall()
    }
}

/// F1 score (the harmonic mean of precision and recall) at a fixed IoU
/// threshold, accumulated across images.
#[derive(Debug, Clone)]
pub struct F1Score {
    iou_threshold: f32,
    counts: Counts,
}

impl F1Score {
    /// Creates a metric matching predictions to targets at `iou_threshold`.
    pub fn new(iou_threshold: f32) -> Self {
        Self {
            iou_threshold,
            counts: Counts::default(),
        }
    }

    /// Matches one image's `predictions` against its `targets` and folds
    /// the result into the running total.
    pub fn update(&mut self, predictions: &Detections, targets: &Detections) -> &mut Self {
        self.counts.add(predictions, targets, self.iou_threshold);
        self
    }

    /// F1 score over every `update()` call so far.
    pub fn compute(&self) -> f32 {
        self.counts.f1()
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
    fn perfect_match_scores_one_on_every_metric() {
        let predictions = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);

        let mut precision = Precision::new(0.5);
        precision.update(&predictions, &targets);
        assert_eq!(precision.compute(), 1.0);

        let mut recall = Recall::new(0.5);
        recall.update(&predictions, &targets);
        assert_eq!(recall.compute(), 1.0);

        let mut f1 = F1Score::new(0.5);
        f1.update(&predictions, &targets);
        assert_eq!(f1.compute(), 1.0);
    }

    #[test]
    fn no_updates_yields_zero_rather_than_nan() {
        assert_eq!(Precision::new(0.5).compute(), 0.0);
        assert_eq!(Recall::new(0.5).compute(), 0.0);
        assert_eq!(F1Score::new(0.5).compute(), 0.0);
    }

    #[test]
    fn accumulates_across_multiple_update_calls() {
        let mut precision = Precision::new(0.5);

        let hit = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        let target = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        precision.update(&hit, &target);

        let miss = Detections::new(vec![det([50.0, 50.0, 60.0, 60.0], 0.9, 0)]);
        precision.update(&miss, &target);

        // One TP, one FP across two images: 50% precision.
        assert!((precision.compute() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn f1_is_harmonic_mean_of_precision_and_recall() {
        // One TP, one FP, one FN: precision = 0.5, recall = 0.5, f1 = 0.5.
        let predictions = Detections::new(vec![
            det([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            det([50.0, 50.0, 60.0, 60.0], 0.8, 0),
        ]);
        let targets = Detections::new(vec![
            det([0.0, 0.0, 10.0, 10.0], 1.0, 0),
            det([100.0, 100.0, 110.0, 110.0], 1.0, 0),
        ]);

        let mut f1 = F1Score::new(0.5);
        f1.update(&predictions, &targets);
        assert!((f1.compute() - 0.5).abs() < 1e-6);
    }
}
