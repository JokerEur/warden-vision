//! Shared per-image greedy matching between predicted and ground-truth
//! detections, used by every metric in this module.

use crate::core::Detections;

/// Greedily matches `predictions` (processed in descending confidence
/// order) to `targets`, both restricted to detections that share a class
/// id, requiring IoU `>= iou_threshold`. Returns each prediction's
/// `(confidence, is_true_positive)`, in the same order they were matched
/// in.
///
/// Each target matches at most one prediction: the first (i.e.
/// highest-confidence) prediction whose IoU with it is both `>=
/// iou_threshold` and the best available at match time.
pub(crate) fn match_records(
    predictions: &Detections,
    targets: &Detections,
    iou_threshold: f32,
) -> Vec<(f32, bool)> {
    let mut matched = vec![false; targets.len()];
    let mut order: Vec<usize> = (0..predictions.len()).collect();
    order.sort_by(|&a, &b| {
        predictions.detections[b]
            .confidence
            .total_cmp(&predictions.detections[a].confidence)
    });

    let mut records = Vec::with_capacity(order.len());
    for i in order {
        let pred = &predictions.detections[i];
        let mut best_iou = 0.0;
        let mut best_j = None;
        for (j, gt) in targets.detections.iter().enumerate() {
            if matched[j] || gt.class_id != pred.class_id {
                continue;
            }
            let iou = pred.iou(gt);
            if iou > best_iou {
                best_iou = iou;
                best_j = Some(j);
            }
        }

        let is_tp = if best_iou >= iou_threshold {
            if let Some(j) = best_j {
                matched[j] = true;
                true
            } else {
                false
            }
        } else {
            false
        };
        records.push((pred.confidence, is_tp));
    }
    records
}

/// Like [`match_records`], but summarized as
/// `(true_positives, false_positives, false_negatives)` for this single
/// image/call, for metrics that only need a single operating point rather
/// than a full precision-recall curve.
pub(crate) fn match_counts(
    predictions: &Detections,
    targets: &Detections,
    iou_threshold: f32,
) -> (usize, usize, usize) {
    let records = match_records(predictions, targets, iou_threshold);
    let true_positives = records.iter().filter(|&&(_, is_tp)| is_tp).count();
    let false_positives = records.len() - true_positives;
    // Every match consumes exactly one distinct target, so whatever wasn't
    // matched is a false negative.
    let false_negatives = targets.len() - true_positives;
    (true_positives, false_positives, false_negatives)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Detection;

    fn det(bbox: [f32; 4], confidence: f32, class_id: usize) -> Detection {
        Detection::new(bbox, confidence, class_id)
    }

    #[test]
    fn exact_match_is_a_true_positive() {
        let predictions = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        let (tp, fp, fn_) = match_counts(&predictions, &targets, 0.5);
        assert_eq!((tp, fp, fn_), (1, 0, 0));
    }

    #[test]
    fn wrong_class_never_matches() {
        let predictions = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9, 1)]);
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        let (tp, fp, fn_) = match_counts(&predictions, &targets, 0.5);
        assert_eq!((tp, fp, fn_), (0, 1, 1));
    }

    #[test]
    fn low_iou_below_threshold_is_unmatched() {
        let predictions = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        let targets = Detections::new(vec![det([9.0, 9.0, 19.0, 19.0], 1.0, 0)]);
        let (tp, fp, fn_) = match_counts(&predictions, &targets, 0.5);
        assert_eq!((tp, fp, fn_), (0, 1, 1));
    }

    #[test]
    fn extra_prediction_is_a_false_positive_without_consuming_a_target() {
        let predictions = Detections::new(vec![
            det([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            det([50.0, 50.0, 60.0, 60.0], 0.8, 0),
        ]);
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        let (tp, fp, fn_) = match_counts(&predictions, &targets, 0.5);
        assert_eq!((tp, fp, fn_), (1, 1, 0));
    }

    #[test]
    fn higher_confidence_prediction_wins_a_contested_target() {
        // Two predictions both overlap the single target well enough to
        // match; only the higher-confidence one should win it.
        let predictions = Detections::new(vec![
            det([0.0, 0.0, 10.0, 10.0], 0.4, 0),
            det([1.0, 1.0, 11.0, 11.0], 0.95, 0),
        ]);
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        let records = match_records(&predictions, &targets, 0.5);
        assert_eq!(records[0], (0.95, true));
        assert_eq!(records[1], (0.4, false));
    }
}
