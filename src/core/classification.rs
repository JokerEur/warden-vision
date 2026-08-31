//! Whole-image classification results, as opposed to the per-object
//! detections in [`crate::core::Detections`].

/// A single class prediction: class id plus confidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassPrediction {
    /// Index of the predicted class.
    pub class_id: usize,
    /// Confidence score, typically in `[0, 1]`.
    pub confidence: f32,
}

impl ClassPrediction {
    /// Creates a new class prediction.
    pub fn new(class_id: usize, confidence: f32) -> Self {
        Self {
            class_id,
            confidence,
        }
    }
}

/// The classification results for a single image (or crop): one
/// [`ClassPrediction`] per class the classifier scored.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Classifications {
    /// One entry per class, in whatever order the classifier produced
    /// them (typically class id order, not sorted by confidence). Use
    /// [`Classifications::top_k`] / [`Classifications::top1`] for a
    /// confidence-sorted view.
    pub predictions: Vec<ClassPrediction>,
}

impl Classifications {
    /// Creates a new `Classifications` from an existing prediction list.
    pub fn new(predictions: Vec<ClassPrediction>) -> Self {
        Self { predictions }
    }

    /// Builds a `Classifications` from a dense per-class score vector
    /// (`scores[class_id]`), as produced directly by a softmax/sigmoid
    /// output layer.
    pub fn from_scores(scores: &[f32]) -> Self {
        Self::new(
            scores
                .iter()
                .enumerate()
                .map(|(class_id, &confidence)| ClassPrediction::new(class_id, confidence))
                .collect(),
        )
    }

    /// The `k` highest-confidence predictions, sorted descending. Returns
    /// fewer than `k` if there aren't that many predictions.
    pub fn top_k(&self, k: usize) -> Vec<ClassPrediction> {
        let mut sorted = self.predictions.clone();
        sorted.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        sorted.truncate(k);
        sorted
    }

    /// The single highest-confidence prediction, if any.
    pub fn top1(&self) -> Option<ClassPrediction> {
        self.predictions
            .iter()
            .copied()
            .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_scores_indexes_predictions_by_position() {
        let classifications = Classifications::from_scores(&[0.1, 0.7, 0.2]);
        assert_eq!(classifications.predictions.len(), 3);
        assert_eq!(classifications.predictions[1], ClassPrediction::new(1, 0.7));
    }

    #[test]
    fn top1_returns_the_highest_confidence_prediction() {
        let classifications = Classifications::from_scores(&[0.1, 0.7, 0.2]);
        assert_eq!(classifications.top1(), Some(ClassPrediction::new(1, 0.7)));
    }

    #[test]
    fn top1_of_empty_is_none() {
        assert_eq!(Classifications::default().top1(), None);
    }

    #[test]
    fn top_k_sorts_descending_and_truncates() {
        let classifications = Classifications::from_scores(&[0.1, 0.7, 0.5, 0.2]);
        let top2 = classifications.top_k(2);
        assert_eq!(
            top2,
            vec![ClassPrediction::new(1, 0.7), ClassPrediction::new(2, 0.5)]
        );
    }

    #[test]
    fn top_k_larger_than_predictions_returns_all() {
        let classifications = Classifications::from_scores(&[0.1, 0.7]);
        assert_eq!(classifications.top_k(10).len(), 2);
    }
}
