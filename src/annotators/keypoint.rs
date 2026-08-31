//! Annotators for [`crate::core::KeyPoints`]: individual joints
//! ([`VertexAnnotator`]), skeleton bones ([`EdgeAnnotator`]), and labeled
//! joints ([`VertexLabelAnnotator`]).

use crate::annotators::Color;

/// Draws a filled dot at each detected joint.
#[derive(Debug, Clone)]
pub struct VertexAnnotator {
    /// Dot color.
    pub color: Color,
    /// Dot radius, in pixels.
    pub radius: u32,
}

impl VertexAnnotator {
    /// Creates a new vertex annotator.
    pub fn new(color: Color, radius: u32) -> Self {
        Self { color, radius }
    }
}

impl Default for VertexAnnotator {
    fn default() -> Self {
        Self::new(Color::new(0xFF, 0x64, 0x37), 4)
    }
}

/// Draws skeleton bones connecting pairs of joints.
///
/// A bone is only drawn when both of its joints were detected (i.e.
/// neither is `None` in the [`crate::core::KeypointSet`]).
#[derive(Debug, Clone)]
pub struct EdgeAnnotator {
    /// Bone color.
    pub color: Color,
    /// Stroke width, in pixels.
    pub thickness: u32,
    /// Joint index pairs to connect, e.g. [`crate::core::COCO_17_EDGES`].
    pub edges: Vec<(usize, usize)>,
}

impl EdgeAnnotator {
    /// Creates a new edge annotator over the given skeleton `edges`.
    pub fn new(color: Color, thickness: u32, edges: Vec<(usize, usize)>) -> Self {
        Self {
            color,
            thickness,
            edges,
        }
    }

    /// An edge annotator using the standard 17-joint COCO skeleton.
    pub fn coco_17(color: Color, thickness: u32) -> Self {
        Self::new(color, thickness, crate::core::COCO_17_EDGES.to_vec())
    }
}

/// Draws each detected joint as a dot labeled with its joint index.
#[derive(Debug, Clone)]
pub struct VertexLabelAnnotator {
    /// Label text and background color.
    pub color: Color,
    /// Relative size of the label text.
    pub text_scale: u32,
}

impl VertexLabelAnnotator {
    /// Creates a new vertex-label annotator.
    pub fn new(color: Color, text_scale: u32) -> Self {
        Self { color, text_scale }
    }
}

impl Default for VertexLabelAnnotator {
    fn default() -> Self {
        Self::new(Color::new(255, 255, 255), 1)
    }
}
