//! Named anchor points on a bounding box, used by annotators to decide
//! where on a detection to place a dot, label, or trace point.

/// A named anchor position on a bounding box (or, for
/// [`Position::CenterOfMass`], on a detection's mask polygon).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Position {
    /// Center of the box.
    #[default]
    Center,
    /// Vertical center, left edge.
    CenterLeft,
    /// Vertical center, right edge.
    CenterRight,
    /// Horizontal center, top edge.
    TopCenter,
    /// Top-left corner.
    TopLeft,
    /// Top-right corner.
    TopRight,
    /// Horizontal center, bottom edge.
    BottomCenter,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-right corner.
    BottomRight,
    /// Centroid of the detection's mask polygon, if it has one; falls back
    /// to [`Position::Center`] otherwise.
    CenterOfMass,
}
