//! Multi-object tracking: Kalman motion prediction and IoU-based
//! association.

mod assignment;
mod byte_track;
mod deep_sort;
mod kalman;
mod smoother;
mod sort;

pub use byte_track::ByteTracker;
pub use deep_sort::DeepSortTracker;
pub use kalman::KalmanBoxFilter;
pub use smoother::DetectionsSmoother;
pub use sort::SortTracker;
