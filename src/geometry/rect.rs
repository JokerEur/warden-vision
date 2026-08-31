//! An axis-aligned rectangle, with anchor-point and padding helpers used by
//! annotators that need something more geometric than a raw `[f32; 4]`.

use crate::core::bbox_iou;
use crate::geometry::{Point, Position};

/// An axis-aligned rectangle in `[x1, y1, x2, y2]` (top-left/bottom-right)
/// space.
///
/// Interoperates with the raw `[f32; 4]` bounding boxes used by
/// [`crate::core::Detection`] via [`Rect::from_xyxy`] / [`Rect::to_xyxy`],
/// so callers can move between the two representations depending on
/// whether they want geometric helpers (this type) or the compact array
/// form (detections, ndarray batches).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// Left edge.
    pub x1: f32,
    /// Top edge.
    pub y1: f32,
    /// Right edge.
    pub x2: f32,
    /// Bottom edge.
    pub y2: f32,
}

impl Rect {
    /// Creates a new rectangle from its edges.
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Builds a `Rect` from an `[x1, y1, x2, y2]` array.
    pub fn from_xyxy(bbox: [f32; 4]) -> Self {
        Self::new(bbox[0], bbox[1], bbox[2], bbox[3])
    }

    /// Converts back to an `[x1, y1, x2, y2]` array.
    pub fn to_xyxy(&self) -> [f32; 4] {
        [self.x1, self.y1, self.x2, self.y2]
    }

    /// Width (`x2 - x1`).
    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    /// Height (`y2 - y1`).
    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    /// Area, clamped to zero for degenerate (negative-size) rectangles.
    pub fn area(&self) -> f32 {
        self.width().max(0.0) * self.height().max(0.0)
    }

    /// Center point.
    pub fn center(&self) -> Point {
        Point::new((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }

    /// The point at a named [`Position`] on this rectangle.
    ///
    /// [`Position::CenterOfMass`] has no meaning for a plain rectangle (it
    /// depends on a mask polygon), so it falls back to [`Rect::center`];
    /// callers that care about the mask-aware behavior should use
    /// [`crate::core::Detection::anchor_point`] instead.
    pub fn anchor(&self, position: Position) -> Point {
        match position {
            Position::Center | Position::CenterOfMass => self.center(),
            Position::CenterLeft => Point::new(self.x1, (self.y1 + self.y2) / 2.0),
            Position::CenterRight => Point::new(self.x2, (self.y1 + self.y2) / 2.0),
            Position::TopCenter => Point::new((self.x1 + self.x2) / 2.0, self.y1),
            Position::TopLeft => Point::new(self.x1, self.y1),
            Position::TopRight => Point::new(self.x2, self.y1),
            Position::BottomCenter => Point::new((self.x1 + self.x2) / 2.0, self.y2),
            Position::BottomLeft => Point::new(self.x1, self.y2),
            Position::BottomRight => Point::new(self.x2, self.y2),
        }
    }

    /// Grows (or, with negative `padding`, shrinks) the rectangle by
    /// `padding` on every side.
    pub fn pad(&self, padding: f32) -> Rect {
        Rect::new(
            self.x1 - padding,
            self.y1 - padding,
            self.x2 + padding,
            self.y2 + padding,
        )
    }

    /// Whether `point` falls within the rectangle (inclusive of edges).
    pub fn contains_point(&self, point: Point) -> bool {
        point.x >= self.x1 && point.x <= self.x2 && point.y >= self.y1 && point.y <= self.y2
    }

    /// Intersection-over-union with another rectangle.
    pub fn iou(&self, other: &Rect) -> f32 {
        bbox_iou(self.to_xyxy(), other.to_xyxy())
    }
}

impl From<[f32; 4]> for Rect {
    fn from(bbox: [f32; 4]) -> Self {
        Rect::from_xyxy(bbox)
    }
}

impl From<Rect> for [f32; 4] {
    fn from(rect: Rect) -> Self {
        rect.to_xyxy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Rect {
        Rect::new(0.0, 0.0, 10.0, 10.0)
    }

    #[test]
    fn width_height_area() {
        let r = square();
        assert_eq!(r.width(), 10.0);
        assert_eq!(r.height(), 10.0);
        assert_eq!(r.area(), 100.0);
    }

    #[test]
    fn anchor_points_match_expected_corners() {
        let r = square();
        assert_eq!(r.anchor(Position::TopLeft), Point::new(0.0, 0.0));
        assert_eq!(r.anchor(Position::BottomRight), Point::new(10.0, 10.0));
        assert_eq!(r.anchor(Position::Center), Point::new(5.0, 5.0));
        assert_eq!(r.anchor(Position::TopCenter), Point::new(5.0, 0.0));
        assert_eq!(r.anchor(Position::BottomCenter), Point::new(5.0, 10.0));
        assert_eq!(r.anchor(Position::CenterLeft), Point::new(0.0, 5.0));
        assert_eq!(r.anchor(Position::CenterRight), Point::new(10.0, 5.0));
    }

    #[test]
    fn center_of_mass_falls_back_to_center() {
        let r = square();
        assert_eq!(r.anchor(Position::CenterOfMass), r.center());
    }

    #[test]
    fn pad_grows_every_edge() {
        let r = square().pad(2.0);
        assert_eq!(r, Rect::new(-2.0, -2.0, 12.0, 12.0));
    }

    #[test]
    fn contains_point_respects_bounds() {
        let r = square();
        assert!(r.contains_point(Point::new(5.0, 5.0)));
        assert!(r.contains_point(Point::new(0.0, 0.0)));
        assert!(!r.contains_point(Point::new(-1.0, 5.0)));
    }

    #[test]
    fn iou_matches_bbox_iou() {
        let a = square();
        let b = Rect::new(5.0, 0.0, 15.0, 10.0);
        assert!((a.iou(&b) - (50.0 / 150.0)).abs() < 1e-6);
    }

    #[test]
    fn roundtrips_through_xyxy_array() {
        let bbox = [1.0, 2.0, 3.0, 4.0];
        let rect: Rect = bbox.into();
        let back: [f32; 4] = rect.into();
        assert_eq!(bbox, back);
    }
}
