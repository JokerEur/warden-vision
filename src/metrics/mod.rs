//! Evaluation metrics for comparing predicted [`crate::core::Detections`]
//! against ground-truth annotations: [`Precision`], [`Recall`],
//! [`F1Score`], [`ConfusionMatrix`], [`MeanAveragePrecision`], and
//! [`MeanAverageRecall`].
//!
//! Every metric follows the same shape: construct it (usually with an IoU
//! matching threshold), call `update(predictions, targets)` once per
//! image/frame to fold in that image's matches, and call `compute()` at
//! the end to get the aggregated result.

mod confusion_matrix;
mod matching;
mod mean_average_precision;
mod mean_average_recall;
mod precision_recall;

pub use confusion_matrix::ConfusionMatrix;
pub use mean_average_precision::{MeanAveragePrecision, MeanAveragePrecisionResult};
pub use mean_average_recall::{MeanAverageRecall, MeanAverageRecallResult};
pub use precision_recall::{F1Score, Precision, Recall};
