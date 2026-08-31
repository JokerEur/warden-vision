//! A polygonal region of interest that tracks which objects are currently
//! inside it, and counts entries/exits.

use std::collections::{HashMap, HashSet};

use crate::core::Detections;
use crate::geometry::{Point, Zone};

/// A region of interest bounded by an arbitrary (possibly concave) simple
/// polygon.
///
/// Like [`LineZone`](crate::geometry::LineZone), [`PolygonZone::trigger`]
/// is driven by a previous-frame centroid map supplied by the caller: for
/// each tracked detection it compares whether the previous centroid and
/// current centroid were inside the polygon, and updates `in_count`
/// (entries), `out_count` (exits), and the live [`PolygonZone::inside`] set
/// accordingly. An object with no previous centroid (first sighting) is
/// assumed not to have just transitioned, so it is only counted as an
/// entry once it is later seen leaving, or on the frame it enters if a
/// previous position outside the polygon is known.
#[derive(Debug, Clone)]
pub struct PolygonZone {
    /// Vertices of the polygon, in order (open — the last vertex implicitly
    /// connects back to the first).
    pub polygon: Vec<Point>,
    inside: HashSet<usize>,
    in_count: usize,
    out_count: usize,
}

impl PolygonZone {
    /// Creates a new zone bounded by `polygon`.
    pub fn new(polygon: Vec<Point>) -> Self {
        Self {
            polygon,
            inside: HashSet::new(),
            in_count: 0,
            out_count: 0,
        }
    }

    /// Point-in-polygon test using the ray casting algorithm: casts a ray
    /// in the +x direction from `point` and counts polygon edge crossings;
    /// an odd count means the point is inside.
    pub fn contains(&self, point: Point) -> bool {
        let n = self.polygon.len();
        if n < 3 {
            return false;
        }

        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let pi = self.polygon[i];
            let pj = self.polygon[j];

            let straddles = (pi.y > point.y) != (pj.y > point.y);
            if straddles {
                let x_intersect = (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x;
                if point.x < x_intersect {
                    inside = !inside;
                }
            }
            j = i;
        }
        inside
    }

    /// Tracker ids currently inside the zone, as of the last [`Zone::trigger`] call.
    pub fn inside(&self) -> &HashSet<usize> {
        &self.inside
    }

    /// Number of tracker ids currently inside the zone.
    pub fn current_count(&self) -> usize {
        self.inside.len()
    }
}

impl Zone for PolygonZone {
    fn trigger(&mut self, detections: &Detections, previous_centroids: &HashMap<usize, Point>) {
        for detection in detections.iter() {
            let Some(tracker_id) = detection.tracker_id else {
                continue;
            };

            let (cx, cy) = detection.centroid();
            let current = Point::new(cx, cy);
            let currently_inside = self.contains(current);

            let previously_inside = previous_centroids
                .get(&tracker_id)
                .map(|&p| self.contains(p))
                .unwrap_or(currently_inside);

            if currently_inside && !previously_inside {
                self.in_count += 1;
                self.inside.insert(tracker_id);
            } else if !currently_inside && previously_inside {
                self.out_count += 1;
                self.inside.remove(&tracker_id);
            } else if currently_inside {
                self.inside.insert(tracker_id);
            } else {
                self.inside.remove(&tracker_id);
            }
        }
    }

    fn in_count(&self) -> usize {
        self.in_count
    }

    fn out_count(&self) -> usize {
        self.out_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Detection;

    fn square() -> PolygonZone {
        PolygonZone::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ])
    }

    fn tracked_detection(tracker_id: usize, x: f32, y: f32) -> Detection {
        let mut d = Detection::new([x - 1.0, y - 1.0, x + 1.0, y + 1.0], 0.9, 0);
        d.tracker_id = Some(tracker_id);
        d
    }

    #[test]
    fn point_inside_square_is_contained() {
        let zone = square();
        assert!(zone.contains(Point::new(5.0, 5.0)));
    }

    #[test]
    fn point_outside_square_is_not_contained() {
        let zone = square();
        assert!(!zone.contains(Point::new(15.0, 5.0)));
        assert!(!zone.contains(Point::new(-5.0, 5.0)));
        assert!(!zone.contains(Point::new(5.0, -5.0)));
        assert!(!zone.contains(Point::new(5.0, 15.0)));
    }

    #[test]
    fn point_in_concave_polygon() {
        // A "U" shape (concave notch cut from the top-middle).
        let zone = PolygonZone::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(6.0, 10.0),
            Point::new(6.0, 4.0),
            Point::new(4.0, 4.0),
            Point::new(4.0, 10.0),
            Point::new(0.0, 10.0),
        ]);

        // Inside the left leg of the U.
        assert!(zone.contains(Point::new(2.0, 8.0)));
        // Inside the notch (should be outside the polygon).
        assert!(!zone.contains(Point::new(5.0, 8.0)));
        // Inside the base of the U.
        assert!(zone.contains(Point::new(5.0, 2.0)));
    }

    #[test]
    fn degenerate_polygon_contains_nothing() {
        let zone = PolygonZone::new(vec![Point::new(0.0, 0.0), Point::new(1.0, 1.0)]);
        assert!(!zone.contains(Point::new(0.5, 0.5)));
    }

    #[test]
    fn trigger_counts_entry_and_exit() {
        let mut zone = square();
        let mut previous = HashMap::new();

        // Object starts outside, moves inside: counts as an entry.
        previous.insert(1usize, Point::new(-5.0, 5.0));
        zone.trigger(
            &Detections::new(vec![tracked_detection(1, 5.0, 5.0)]),
            &previous,
        );
        assert_eq!(zone.in_count(), 1);
        assert_eq!(zone.out_count(), 0);
        assert!(zone.inside().contains(&1));

        // Object moves from inside to outside: counts as an exit.
        previous.insert(1usize, Point::new(5.0, 5.0));
        zone.trigger(
            &Detections::new(vec![tracked_detection(1, 20.0, 5.0)]),
            &previous,
        );
        assert_eq!(zone.in_count(), 1);
        assert_eq!(zone.out_count(), 1);
        assert!(!zone.inside().contains(&1));
    }

    #[test]
    fn trigger_first_sighting_inside_does_not_double_count() {
        let mut zone = square();
        let previous = HashMap::new();
        zone.trigger(
            &Detections::new(vec![tracked_detection(1, 5.0, 5.0)]),
            &previous,
        );
        // No previous position known, so no transition is recorded, but the
        // object is still tracked as currently inside.
        assert_eq!(zone.in_count(), 0);
        assert!(zone.inside().contains(&1));
    }
}
