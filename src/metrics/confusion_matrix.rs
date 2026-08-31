//! A confusion matrix over predicted vs. actual class ids, matched per
//! image at a fixed IoU threshold and accumulated across `update()` calls.

use ndarray::Array2;

use crate::core::Detections;

/// Counts predicted-vs-actual class id pairs across images.
///
/// The matrix is `(num_classes + 1) x (num_classes + 1)`: rows are the
/// predicted class (or the extra "background" row, index `num_classes`,
/// for ground-truth instances no prediction matched — false negatives),
/// columns are the actual class (or the extra "background" column, for
/// predictions that matched no ground truth — false positives).
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    num_classes: usize,
    iou_threshold: f32,
    matrix: Array2<usize>,
}

impl ConfusionMatrix {
    /// Creates an empty confusion matrix for `num_classes` classes (ids
    /// `0..num_classes`), matching predictions to targets at
    /// `iou_threshold`.
    ///
    /// # Panics
    /// Panics if `num_classes` is `0`.
    pub fn new(num_classes: usize, iou_threshold: f32) -> Self {
        assert!(
            num_classes > 0,
            "a confusion matrix needs at least one class"
        );
        Self {
            num_classes,
            iou_threshold,
            matrix: Array2::zeros((num_classes + 1, num_classes + 1)),
        }
    }

    /// Index of the extra "background" row/column.
    fn background(&self) -> usize {
        self.num_classes
    }

    /// Matches one image's `predictions` against its `targets`, folding
    /// the result into the running matrix. Matching is by best IoU
    /// (ignoring class, so a wrong-class match still lands off the
    /// diagonal instead of being scored as an independent false positive
    /// and false negative), in descending prediction confidence order.
    pub fn update(&mut self, predictions: &Detections, targets: &Detections) -> &mut Self {
        let mut matched = vec![false; targets.len()];
        let mut order: Vec<usize> = (0..predictions.len()).collect();
        order.sort_by(|&a, &b| {
            predictions.detections[b]
                .confidence
                .total_cmp(&predictions.detections[a].confidence)
        });

        for i in order {
            let pred = &predictions.detections[i];
            let mut best_iou = 0.0;
            let mut best_j = None;
            for (j, gt) in targets.detections.iter().enumerate() {
                if matched[j] {
                    continue;
                }
                let iou = pred.iou(gt);
                if iou > best_iou {
                    best_iou = iou;
                    best_j = Some(j);
                }
            }

            let predicted_row = pred.class_id.min(self.num_classes - 1);
            if best_iou >= self.iou_threshold {
                if let Some(j) = best_j {
                    matched[j] = true;
                    let actual_col = targets.detections[j].class_id.min(self.num_classes - 1);
                    self.matrix[[predicted_row, actual_col]] += 1;
                    continue;
                }
            }
            let background = self.background();
            self.matrix[[predicted_row, background]] += 1;
        }

        let background = self.background();
        for (j, gt) in targets.detections.iter().enumerate() {
            if !matched[j] {
                let actual_col = gt.class_id.min(self.num_classes - 1);
                self.matrix[[background, actual_col]] += 1;
            }
        }
        self
    }

    /// The accumulated `(num_classes + 1) x (num_classes + 1)` matrix.
    pub fn matrix(&self) -> &Array2<usize> {
        &self.matrix
    }

    /// Precision per class: `matrix[c][c] / sum(matrix[c][..])` (how often
    /// a prediction of class `c` was correct).
    pub fn precision_per_class(&self) -> Vec<f32> {
        (0..self.num_classes)
            .map(|c| {
                let row_sum: usize = self.matrix.row(c).sum();
                if row_sum == 0 {
                    0.0
                } else {
                    self.matrix[[c, c]] as f32 / row_sum as f32
                }
            })
            .collect()
    }

    /// Recall per class: `matrix[c][c] / sum(matrix[..][c])` (how often an
    /// actual instance of class `c` was correctly predicted).
    pub fn recall_per_class(&self) -> Vec<f32> {
        (0..self.num_classes)
            .map(|c| {
                let col_sum: usize = self.matrix.column(c).sum();
                if col_sum == 0 {
                    0.0
                } else {
                    self.matrix[[c, c]] as f32 / col_sum as f32
                }
            })
            .collect()
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
    #[should_panic]
    fn zero_classes_panics() {
        ConfusionMatrix::new(0, 0.5);
    }

    #[test]
    fn correct_prediction_lands_on_the_diagonal() {
        let mut cm = ConfusionMatrix::new(2, 0.5);
        let predictions = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9, 1)]);
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 1)]);
        cm.update(&predictions, &targets);
        assert_eq!(cm.matrix()[[1, 1]], 1);
    }

    #[test]
    fn wrong_class_lands_off_diagonal_not_in_background() {
        let mut cm = ConfusionMatrix::new(2, 0.5);
        let predictions = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 1)]);
        cm.update(&predictions, &targets);
        assert_eq!(cm.matrix()[[0, 1]], 1);
        assert_eq!(cm.matrix()[[0, 2]], 0);
        assert_eq!(cm.matrix()[[2, 1]], 0);
    }

    #[test]
    fn unmatched_prediction_is_a_background_column_entry() {
        let mut cm = ConfusionMatrix::new(2, 0.5);
        let predictions = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        let targets = Detections::empty();
        cm.update(&predictions, &targets);
        assert_eq!(cm.matrix()[[0, 2]], 1);
    }

    #[test]
    fn unmatched_target_is_a_background_row_entry() {
        let mut cm = ConfusionMatrix::new(2, 0.5);
        let predictions = Detections::empty();
        let targets = Detections::new(vec![det([0.0, 0.0, 10.0, 10.0], 1.0, 0)]);
        cm.update(&predictions, &targets);
        assert_eq!(cm.matrix()[[2, 0]], 1);
    }

    #[test]
    fn precision_and_recall_per_class_match_expected_ratios() {
        let mut cm = ConfusionMatrix::new(1, 0.5);
        // Two correct class-0 predictions, one false positive, one missed target.
        let predictions = Detections::new(vec![
            det([0.0, 0.0, 10.0, 10.0], 0.9, 0),
            det([20.0, 20.0, 30.0, 30.0], 0.8, 0),
            det([50.0, 50.0, 60.0, 60.0], 0.7, 0),
        ]);
        let targets = Detections::new(vec![
            det([0.0, 0.0, 10.0, 10.0], 1.0, 0),
            det([20.0, 20.0, 30.0, 30.0], 1.0, 0),
            det([100.0, 100.0, 110.0, 110.0], 1.0, 0),
        ]);
        cm.update(&predictions, &targets);
        // precision = 2 correct / 3 predicted = 0.667; recall = 2 correct / 3 actual = 0.667
        assert!((cm.precision_per_class()[0] - 2.0 / 3.0).abs() < 1e-4);
        assert!((cm.recall_per_class()[0] - 2.0 / 3.0).abs() < 1e-4);
    }
}
