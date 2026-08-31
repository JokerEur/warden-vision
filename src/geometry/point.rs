//! 2-D point primitive used throughout the geometry module.

use std::ops::{Add, Sub};

/// A point (or 2-D vector, depending on context) in image space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
}

impl Point {
    /// Creates a new point.
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point.
    pub fn distance(&self, other: &Point) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    /// 2-D cross product of `self` and `other`, treating both as vectors
    /// from the origin: `self.x * other.y - self.y * other.x`.
    ///
    /// The sign indicates the rotational direction from `self` to `other`
    /// (positive = counter-clockwise, negative = clockwise, in a standard
    /// math orientation; image coordinates with a flipped y-axis invert
    /// this).
    pub fn cross(&self, other: &Point) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// Dot product of `self` and `other`, treating both as vectors from the
    /// origin.
    pub fn dot(&self, other: &Point) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

impl From<(f32, f32)> for Point {
    fn from((x, y): (f32, f32)) -> Self {
        Point::new(x, y)
    }
}

impl Add for Point {
    type Output = Point;

    fn add(self, rhs: Point) -> Point {
        Point::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Point {
    type Output = Point;

    fn sub(self, rhs: Point) -> Point {
        Point::new(self.x - rhs.x, self.y - rhs.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_is_euclidean() {
        let a = Point::new(0.0, 0.0);
        let b = Point::new(3.0, 4.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn cross_product_sign_indicates_rotation() {
        let a = Point::new(1.0, 0.0);
        let b = Point::new(0.0, 1.0);
        assert!(a.cross(&b) > 0.0);
        assert!(b.cross(&a) < 0.0);
    }

    #[test]
    fn sub_yields_displacement_vector() {
        let a = Point::new(5.0, 5.0);
        let b = Point::new(2.0, 1.0);
        let d = a - b;
        assert_eq!(d, Point::new(3.0, 4.0));
    }
}
