//! Area and intersection-over-union for oriented (rotated) bounding boxes,
//! represented as four corner points in either winding order.
//!
//! Used by [`crate::core::Detection`] when it carries an
//! [`obb`](crate::core::Detection::obb): axis-aligned IoU over-counts
//! overlap for thin, rotated objects (aerial imagery, text lines, packages
//! on a conveyor), so [`Detection::iou`](crate::core::Detection::iou)
//! switches to [`oriented_box_iou`] whenever both sides have one.

use crate::geometry::Point;

fn signed_area(points: &[Point; 4]) -> f32 {
    let mut sum = 0.0;
    for i in 0..4 {
        let p1 = points[i];
        let p2 = points[(i + 1) % 4];
        sum += p1.x * p2.y - p2.x * p1.y;
    }
    sum / 2.0
}

/// Unsigned area of a quadrilateral given its four corners (shoelace
/// formula), independent of winding order.
pub fn oriented_box_area(points: &[Point; 4]) -> f32 {
    signed_area(points).abs()
}

/// Returns `points` reordered counter-clockwise if it currently isn't.
fn ensure_ccw(points: &[Point; 4]) -> [Point; 4] {
    if signed_area(points) < 0.0 {
        [points[3], points[2], points[1], points[0]]
    } else {
        *points
    }
}

/// Point where segment `p1`-`p2` crosses the infinite line through
/// `p3`-`p4`, or `None` if they're parallel.
fn line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Option<Point> {
    let d1 = p2 - p1;
    let d2 = p4 - p3;
    let denom = d1.cross(&d2);
    if denom.abs() < 1e-9 {
        return None;
    }
    let t = (p3 - p1).cross(&d2) / denom;
    Some(Point::new(p1.x + t * d1.x, p1.y + t * d1.y))
}

/// Area of intersection between two convex quadrilaterals, via
/// Sutherland-Hodgman clipping of `subject` against `clip`. Both are
/// normalized to counter-clockwise winding first, so the caller's winding
/// order doesn't matter.
fn intersection_area(subject: &[Point; 4], clip: &[Point; 4]) -> f32 {
    let clip = ensure_ccw(clip);
    let mut output: Vec<Point> = ensure_ccw(subject).to_vec();

    for i in 0..4 {
        if output.is_empty() {
            break;
        }
        let edge_start = clip[i];
        let edge_end = clip[(i + 1) % 4];
        let edge = edge_end - edge_start;
        let input = std::mem::take(&mut output);

        for j in 0..input.len() {
            let current = input[j];
            let previous = input[(j + input.len() - 1) % input.len()];
            let current_inside = edge.cross(&(current - edge_start)) >= 0.0;
            let previous_inside = edge.cross(&(previous - edge_start)) >= 0.0;

            if current_inside {
                if !previous_inside {
                    if let Some(p) = line_intersection(previous, current, edge_start, edge_end) {
                        output.push(p);
                    }
                }
                output.push(current);
            } else if previous_inside {
                if let Some(p) = line_intersection(previous, current, edge_start, edge_end) {
                    output.push(p);
                }
            }
        }
    }

    if output.len() < 3 {
        0.0
    } else {
        crate::geometry::polygon_area(&output)
    }
}

/// Intersection-over-union of two oriented bounding boxes, each given as
/// four corner points (any consistent winding order).
///
/// Returns `0.0` if either box is degenerate (zero area).
pub fn oriented_box_iou(a: &[Point; 4], b: &[Point; 4]) -> f32 {
    let area_a = oriented_box_area(a);
    let area_b = oriented_box_area(b);
    if area_a <= 0.0 || area_b <= 0.0 {
        return 0.0;
    }
    let inter = intersection_area(a, b);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        (inter / union).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(cx: f32, cy: f32, half: f32) -> [Point; 4] {
        [
            Point::new(cx - half, cy - half),
            Point::new(cx + half, cy - half),
            Point::new(cx + half, cy + half),
            Point::new(cx - half, cy + half),
        ]
    }

    fn rotate(points: &[Point; 4], cx: f32, cy: f32, degrees: f32) -> [Point; 4] {
        let theta = degrees.to_radians();
        let (s, c) = theta.sin_cos();
        points.map(|p| {
            let (dx, dy) = (p.x - cx, p.y - cy);
            Point::new(cx + dx * c - dy * s, cy + dx * s + dy * c)
        })
    }

    #[test]
    fn identical_boxes_have_iou_one() {
        let a = square(10.0, 10.0, 5.0);
        assert!((oriented_box_iou(&a, &a) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn disjoint_boxes_have_iou_zero() {
        let a = square(0.0, 0.0, 5.0);
        let b = square(100.0, 100.0, 5.0);
        assert_eq!(oriented_box_iou(&a, &b), 0.0);
    }

    #[test]
    fn area_matches_axis_aligned_square() {
        let a = square(0.0, 0.0, 5.0);
        assert!((oriented_box_area(&a) - 100.0).abs() < 1e-4);
    }

    #[test]
    fn area_is_winding_independent() {
        let mut reversed = square(0.0, 0.0, 5.0);
        reversed.reverse();
        assert!((oriented_box_area(&reversed) - 100.0).abs() < 1e-4);
    }

    #[test]
    fn rotated_square_overlaps_axis_aligned_one_less_than_axis_aligned_iou_would() {
        // A 45deg-rotated square inscribed the same center as an
        // axis-aligned one overlaps it in a smaller "pinwheel" region than
        // their bounding boxes would suggest, which is exactly the case
        // axis-aligned IoU gets wrong.
        let axis_aligned = square(0.0, 0.0, 10.0);
        let rotated = rotate(&square(0.0, 0.0, 10.0), 0.0, 0.0, 45.0);
        let iou = oriented_box_iou(&axis_aligned, &rotated);
        assert!(iou > 0.0 && iou < 1.0, "iou was {iou}");
    }

    #[test]
    fn touching_boxes_have_zero_area_intersection() {
        let a = square(0.0, 0.0, 5.0);
        let b = square(10.0, 0.0, 5.0);
        assert_eq!(oriented_box_iou(&a, &b), 0.0);
    }
}
