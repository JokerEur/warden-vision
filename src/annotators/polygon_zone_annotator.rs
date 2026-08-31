//! Draws a [`crate::geometry::PolygonZone`]'s boundary and its live
//! occupancy/in/out counts, mirroring [`crate::annotators::LineZoneAnnotator`]
//! for polygonal (rather than line) zones.

use crate::annotators::Color;

/// Draws a [`crate::geometry::PolygonZone`]'s outline and count label.
#[derive(Debug, Clone)]
pub struct PolygonZoneAnnotator {
    /// Color of the outline and its count label.
    pub color: Color,
    /// Stroke width, in pixels, of the drawn outline.
    pub thickness: u32,
    /// Relative size of the count label text.
    pub text_scale: f32,
}

impl PolygonZoneAnnotator {
    /// Creates a new polygon-zone annotator.
    pub fn new(color: Color, thickness: u32, text_scale: f32) -> Self {
        Self {
            color,
            thickness,
            text_scale,
        }
    }
}

impl Default for PolygonZoneAnnotator {
    fn default() -> Self {
        Self::new(Color::new(255, 255, 255), 2, 1.0)
    }
}
