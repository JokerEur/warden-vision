//! Multi-object tracking: Kalman motion prediction and IoU-based
//! association.

mod assignment;
mod byte_track;
mod kalman;
mod smoother;
mod sort;

pub use byte_track::ByteTracker;
pub use kalman::KalmanBoxFilter;
pub use smoother::DetectionsSmoother;
pub use sort::SortTracker;
