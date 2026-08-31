//! Free functions over polygons represented as an ordered `&[Point]` (open;
//! the last vertex implicitly connects back to the first), matching the
//! representation used by [`crate::geometry::PolygonZone`] and
//! [`crate::core::Detection::mask`].

use crate::geometry::{Point, Rect};

/// Signed area of `polygon` via the shoelace formula. Positive for
/// counter-clockwise vertex order, negative for clockwise; use
/// [`polygon_area`] for the unsigned area.
fn signed_area(polygon: &[Point]) -> f32 {
    let n = polygon.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        let p1 = polygon[i];
        let p2 = polygon[(i + 1) % n];
        sum += p1.x * p2.y - p2.x * p1.y;
    }
    sum / 2.0
}

/// Unsigned area enclosed by `polygon`, via the shoelace formula.
///
/// Returns `0.0` for polygons with fewer than 3 vertices.
pub fn polygon_area(polygon: &[Point]) -> f32 {
    signed_area(polygon).abs()
}

/// Centroid (center of mass) of `polygon`, area-weighted over its
/// interior.
///
/// Falls back to the unweighted average of vertices for degenerate
/// (zero-area, e.g. collinear or fewer-than-3-vertex) polygons, so it
/// never returns `NaN` for a non-empty input.
///
/// # Panics
/// Panics if `polygon` is empty.
pub fn polygon_centroid(polygon: &[Point]) -> Point {
    assert!(!polygon.is_empty(), "polygon must have at least one vertex");

    let area = signed_area(polygon);
    let n = polygon.len();

    if n < 3 || area.abs() < 1e-9 {
        let (sum_x, sum_y) = polygon
            .iter()
            .fold((0.0, 0.0), |(sx, sy), p| (sx + p.x, sy + p.y));
        return Point::new(sum_x / n as f32, sum_y / n as f32);
    }

    let mut cx = 0.0;
    let mut cy = 0.0;
    for i in 0..n {
        let p1 = polygon[i];
        let p2 = polygon[(i + 1) % n];
        let cross = p1.x * p2.y - p2.x * p1.y;
        cx += (p1.x + p2.x) * cross;
        cy += (p1.y + p2.y) * cross;
    }
    let scale = 1.0 / (6.0 * area);
    Point::new(cx * scale, cy * scale)
}

/// The axis-aligned bounding [`Rect`] enclosing `polygon`.
///
/// # Panics
/// Panics if `polygon` is empty.
pub fn polygon_to_rect(polygon: &[Point]) -> Rect {
    assert!(!polygon.is_empty(), "polygon must have at least one vertex");

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for p in polygon {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    Rect::new(min_x, min_y, max_x, max_y)
}

/// Keeps only the polygons in `polygons` whose [`polygon_area`] falls
/// within `[min_area, max_area]` (either bound optional).
pub fn filter_polygons_by_area(
    polygons: &[Vec<Point>],
    min_area: Option<f32>,
    max_area: Option<f32>,
) -> Vec<&Vec<Point>> {
    polygons
        .iter()
        .filter(|polygon| {
            let area = polygon_area(polygon);
            min_area.is_none_or(|min| area >= min) && max_area.is_none_or(|max| area <= max)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Vec<Point> {
        vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]
    }

    #[test]
    fn area_of_square_is_side_squared() {
        assert!((polygon_area(&square()) - 100.0).abs() < 1e-4);
    }

    #[test]
    fn area_is_orientation_independent() {
        let mut reversed = square();
        reversed.reverse();
        assert!((polygon_area(&reversed) - 100.0).abs() < 1e-4);
    }

    #[test]
    fn area_of_degenerate_polygon_is_zero() {
        assert_eq!(
            polygon_area(&[Point::new(0.0, 0.0), Point::new(1.0, 1.0)]),
            0.0
        );
    }

    #[test]
    fn centroid_of_square_is_its_center() {
        let c = polygon_centroid(&square());
        assert!((c.x - 5.0).abs() < 1e-4);
        assert!((c.y - 5.0).abs() < 1e-4);
    }

    #[test]
    fn centroid_of_degenerate_polygon_averages_vertices() {
        let c = polygon_centroid(&[Point::new(0.0, 0.0), Point::new(2.0, 0.0)]);
        assert_eq!(c, Point::new(1.0, 0.0));
    }

    #[test]
    fn bounding_rect_matches_extents() {
        let rect = polygon_to_rect(&square());
        assert_eq!(rect, Rect::new(0.0, 0.0, 10.0, 10.0));
    }

    #[test]
    fn filter_by_area_keeps_only_matching_polygons() {
        let small = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
        ];
        let big = square();
        let polygons = vec![small.clone(), big.clone()];
        let kept = filter_polygons_by_area(&polygons, Some(50.0), None);
        assert_eq!(kept, vec![&big]);
    }
}
