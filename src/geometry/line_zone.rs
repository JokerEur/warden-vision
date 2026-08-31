//! A counting line: increments `in`/`out` counters when a tracked object's
//! centroid crosses a fixed line segment.

use std::collections::{HashMap, HashSet};

use crate::core::Detections;
use crate::geometry::{Point, Zone};

/// A virtual tripwire defined by two endpoints. Call [`LineZone::trigger`]
/// once per frame with the current [`Detections`] and a map of each
/// tracker id's centroid from the *previous* frame; the zone reconstructs
/// each object's short motion segment (`previous -> current`) and checks it
/// against the line for a crossing.
///
/// `LineZone` does not maintain its own centroid history — the caller
/// (typically whatever also runs the tracker) is expected to snapshot
/// [`Detection::centroid`](crate::core::Detection::centroid) after each
/// frame and pass it in as `previous_centroids` on the next call.
///
/// Each `tracker_id` is only ever counted once, the first time it is
/// observed crossing the line, guarded by an internal `HashSet`. This
/// avoids double counting from a single crossing event spread across
/// jittery detections, at the cost of not re-counting an object that
/// legitimately crosses back and forth.
#[derive(Debug, Clone)]
pub struct LineZone {
    /// Line start point.
    pub start: Point,
    /// Line end point.
    pub end: Point,
    in_count: usize,
    out_count: usize,
    counted: HashSet<usize>,
}

impl LineZone {
    /// Creates a new counting line between `start` and `end`.
    pub fn new(start: Point, end: Point) -> Self {
        Self {
            start,
            end,
            in_count: 0,
            out_count: 0,
            counted: HashSet::new(),
        }
    }

    /// Signed area of the triangle `a`, `b`, `c` (twice the actual area).
    /// Zero means the three points are collinear; the sign indicates
    /// which side of line `a -> b` the point `c` falls on.
    fn orientation(a: Point, b: Point, c: Point) -> f32 {
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    }

    /// Assuming `c` is collinear with segment `a`-`b`, checks whether `c`
    /// lies within the segment's bounding box (and therefore on it).
    fn on_segment(a: Point, b: Point, c: Point) -> bool {
        c.x <= a.x.max(b.x) && c.x >= a.x.min(b.x) && c.y <= a.y.max(b.y) && c.y >= a.y.min(b.y)
    }

    /// Standard orientation-based segment intersection test, including
    /// collinear-overlap edge cases.
    fn segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool {
        let d1 = Self::orientation(p3, p4, p1);
        let d2 = Self::orientation(p3, p4, p2);
        let d3 = Self::orientation(p1, p2, p3);
        let d4 = Self::orientation(p1, p2, p4);

        if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
            && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
        {
            return true;
        }

        if d1 == 0.0 && Self::on_segment(p3, p4, p1) {
            return true;
        }
        if d2 == 0.0 && Self::on_segment(p3, p4, p2) {
            return true;
        }
        if d3 == 0.0 && Self::on_segment(p1, p2, p3) {
            return true;
        }
        if d4 == 0.0 && Self::on_segment(p1, p2, p4) {
            return true;
        }

        false
    }
}

impl Zone for LineZone {
    /// For every detection with a `tracker_id` and a known previous
    /// centroid, checks whether the segment from the previous centroid to
    /// the current one crosses the line. On a crossing, the direction is
    /// decided by the sign of `cross(line_vector, movement_vector)`:
    /// positive counts as `in`, negative as `out`. Which physical
    /// direction that corresponds to depends on how `start`/`end` are
    /// chosen when constructing the zone.
    fn trigger(&mut self, detections: &Detections, previous_centroids: &HashMap<usize, Point>) {
        for detection in detections.iter() {
            let Some(tracker_id) = detection.tracker_id else {
                continue;
            };
            if self.counted.contains(&tracker_id) {
                continue;
            }
            let Some(&previous) = previous_centroids.get(&tracker_id) else {
                continue;
            };

            let (cx, cy) = detection.centroid();
            let current = Point::new(cx, cy);
            if previous == current {
                continue;
            }

            if Self::segments_intersect(self.start, self.end, previous, current) {
                let line_vector = self.end - self.start;
                let movement_vector = current - previous;
                let cross = line_vector.cross(&movement_vector);

                if cross > 0.0 {
                    self.in_count += 1;
                } else {
                    self.out_count += 1;
                }
                self.counted.insert(tracker_id);
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

    fn tracked_detection(tracker_id: usize, x: f32, y: f32) -> Detection {
        let mut d = Detection::new([x - 1.0, y - 1.0, x + 1.0, y + 1.0], 0.9, 0);
        d.tracker_id = Some(tracker_id);
        d
    }

    #[test]
    fn segments_intersect_simple_crossing() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(10.0, 10.0);
        let p3 = Point::new(0.0, 10.0);
        let p4 = Point::new(10.0, 0.0);
        assert!(LineZone::segments_intersect(p1, p2, p3, p4));
    }

    #[test]
    fn segments_do_not_intersect_when_parallel_and_apart() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(10.0, 0.0);
        let p3 = Point::new(0.0, 5.0);
        let p4 = Point::new(10.0, 5.0);
        assert!(!LineZone::segments_intersect(p1, p2, p3, p4));
    }

    #[test]
    fn segments_do_not_intersect_when_disjoint_on_same_line() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(1.0, 0.0);
        let p3 = Point::new(2.0, 0.0);
        let p4 = Point::new(3.0, 0.0);
        assert!(!LineZone::segments_intersect(p1, p2, p3, p4));
    }

    #[test]
    fn segments_intersect_when_collinear_and_overlapping() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(2.0, 0.0);
        let p3 = Point::new(1.0, 0.0);
        let p4 = Point::new(3.0, 0.0);
        assert!(LineZone::segments_intersect(p1, p2, p3, p4));
    }

    #[test]
    fn trigger_counts_a_crossing_exactly_once() {
        let mut zone = LineZone::new(Point::new(0.0, 0.0), Point::new(0.0, 10.0));
        let mut previous = HashMap::new();
        previous.insert(1usize, Point::new(-5.0, 5.0));

        let detections = Detections::new(vec![tracked_detection(1, 5.0, 5.0)]);
        zone.trigger(&detections, &previous);
        assert_eq!(zone.in_count() + zone.out_count(), 1);

        // Simulate the object continuing to move; since it's already
        // counted, re-triggering (even with a fresh crossing-looking
        // segment) must not increment counters again.
        previous.insert(1usize, Point::new(5.0, 5.0));
        let detections2 = Detections::new(vec![tracked_detection(1, 15.0, 5.0)]);
        zone.trigger(&detections2, &previous);
        assert_eq!(zone.in_count() + zone.out_count(), 1);
    }

    #[test]
    fn trigger_ignores_detections_without_tracker_id() {
        let mut zone = LineZone::new(Point::new(0.0, 0.0), Point::new(0.0, 10.0));
        let previous = HashMap::new();
        let detections = Detections::new(vec![Detection::new([4.0, 4.0, 6.0, 6.0], 0.9, 0)]);
        zone.trigger(&detections, &previous);
        assert_eq!(zone.in_count(), 0);
        assert_eq!(zone.out_count(), 0);
    }

    #[test]
    fn trigger_ignores_motion_that_does_not_cross_the_line() {
        let mut zone = LineZone::new(Point::new(0.0, 0.0), Point::new(0.0, 10.0));
        let mut previous = HashMap::new();
        previous.insert(1usize, Point::new(5.0, 5.0));

        // Moves from (5,5) to (8,5): stays entirely on the right side.
        let detections = Detections::new(vec![tracked_detection(1, 8.0, 5.0)]);
        zone.trigger(&detections, &previous);
        assert_eq!(zone.in_count(), 0);
        assert_eq!(zone.out_count(), 0);
    }

    #[test]
    fn trigger_opposite_directions_produce_opposite_counts() {
        let mut zone_a = LineZone::new(Point::new(0.0, 0.0), Point::new(0.0, 10.0));
        let mut previous_a = HashMap::new();
        previous_a.insert(1usize, Point::new(-5.0, 5.0));
        zone_a.trigger(
            &Detections::new(vec![tracked_detection(1, 5.0, 5.0)]),
            &previous_a,
        );

        let mut zone_b = LineZone::new(Point::new(0.0, 0.0), Point::new(0.0, 10.0));
        let mut previous_b = HashMap::new();
        previous_b.insert(1usize, Point::new(5.0, 5.0));
        zone_b.trigger(
            &Detections::new(vec![tracked_detection(1, -5.0, 5.0)]),
            &previous_b,
        );

        assert_ne!(
            zone_a.in_count() > 0,
            zone_b.in_count() > 0,
            "crossing in opposite directions should flip which counter increments"
        );
    }
}
