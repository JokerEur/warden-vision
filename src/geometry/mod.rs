//! Spatial primitives and zones: points, counting lines, and polygonal
//! regions of interest.

mod line_zone;
mod point;
mod polygon;
mod polygon_zone;
mod position;
mod rect;

pub use line_zone::LineZone;
pub use point::Point;
pub use polygon::{filter_polygons_by_area, polygon_area, polygon_centroid, polygon_to_rect};
pub use polygon_zone::PolygonZone;
pub use position::Position;
pub use rect::Rect;

use std::collections::HashMap;

use crate::core::Detections;

/// Common behavior for spatial zones that observe a stream of [`Detections`]
/// over time and count objects moving in and out.
///
/// Implementations are driven frame-by-frame: call [`Zone::trigger`] once
/// per frame with the current detections and a map from `tracker_id` to
/// that object's centroid on the *previous* frame. Implementations do not
/// maintain this history themselves, since the natural owner of "previous
/// centroid per tracker id" is whatever component already runs the
/// tracker.
pub trait Zone {
    /// Processes one frame of detections, updating internal counters and
    /// any per-object state (e.g. "currently inside").
    fn trigger(&mut self, detections: &Detections, previous_centroids: &HashMap<usize, Point>);

    /// Cumulative count of objects that have entered the zone.
    fn in_count(&self) -> usize;

    /// Cumulative count of objects that have exited the zone.
    fn out_count(&self) -> usize;
}
