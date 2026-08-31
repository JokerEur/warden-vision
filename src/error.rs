//! Shared error type for `warden-vision`.

use thiserror::Error;

/// Errors produced by `warden-vision` operations.
#[derive(Debug, Error)]
pub enum Error {
    /// A vector/array had an unexpected length or shape.
    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// The expected length or shape, rendered as text.
        expected: String,
        /// The actual length or shape, rendered as text.
        actual: String,
    },

    /// A geometric primitive (polygon, line, etc.) was degenerate or invalid.
    #[error("invalid geometry: {0}")]
    InvalidGeometry(String),

    /// The linear assignment solver failed to produce an assignment.
    #[error("assignment solver failed: {0}")]
    AssignmentFailed(String),

    /// An annotation backend (e.g. OpenCV) reported an error while drawing.
    #[error("annotation backend error: {0}")]
    Backend(String),

    /// An I/O error while reading or writing a dataset file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A dataset file (COCO JSON, YOLO label, Pascal VOC XML, ...) was
    /// malformed or failed to parse.
    #[error("parse error: {0}")]
    Parse(String),
}

/// Convenience alias for `Result<T, warden_vision::Error>`.
pub type Result<T> = std::result::Result<T, Error>;
