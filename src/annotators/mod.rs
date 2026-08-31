//! Backend-agnostic annotation: drawing detections and zone counters onto
//! an image or video frame.
//!
//! [`BoxAnnotator`] and [`LineZoneAnnotator`] hold only rendering
//! configuration (colors, thickness) and are independent of any particular
//! image type. Enabling the `annotate-image` feature implements
//! [`Annotator`] for them against `image::RgbaImage`; enabling
//! `annotate-opencv` implements it against `opencv::core::Mat`. Both
//! backends can be enabled at once, and the same annotator value can be
//! reused against either image type.

mod blur;
mod color;
mod heatmap;
mod keypoint;
mod polygon_zone_annotator;
mod shape_annotators;
mod trace;

#[cfg(feature = "annotate-image")]
mod font;
#[cfg(feature = "annotate-image")]
mod image_backend;
#[cfg(feature = "annotate-image")]
mod rich_label;

#[cfg(feature = "annotate-opencv")]
mod opencv_backend;

pub use blur::{BlurAnnotator, PixelateAnnotator};
pub use color::{Color, ColorPalette};
pub use heatmap::HeatMapAnnotator;
pub use keypoint::{EdgeAnnotator, VertexAnnotator, VertexLabelAnnotator};
pub use polygon_zone_annotator::PolygonZoneAnnotator;
#[cfg(feature = "annotate-image")]
pub use rich_label::RichLabelAnnotator;
pub use shape_annotators::{
    BackgroundOverlayAnnotator, BoxCornerAnnotator, CircleAnnotator, DotAnnotator,
    EllipseAnnotator, HaloAnnotator, IconAnnotator, LabelAnnotator, MaskAnnotator,
    PercentageBarAnnotator, PolygonAnnotator, RoundBoxAnnotator, TriangleAnnotator,
};
pub use trace::TraceAnnotator;

/// Draws a `Subject` (detections, a zone, etc.) onto an `Image` buffer.
///
/// Implemented once per rendering backend for each annotator struct, so a
/// single [`BoxAnnotator`] or [`LineZoneAnnotator`] configuration can
/// target whichever backend feature is compiled in.
pub trait Annotator<Image> {
    /// The value being drawn, e.g. [`crate::core::Detections`] or a
    /// [`crate::geometry::LineZone`].
    type Subject;

    /// Draws `subject` onto `image` in place.
    fn annotate(&self, image: &mut Image, subject: &Self::Subject) -> crate::Result<()>;
}

/// Draws a bounding box for each detection, colored by class id, with a
/// label showing its tracker id (if any) and confidence.
#[derive(Debug, Clone)]
pub struct BoxAnnotator {
    /// Maps class ids to colors.
    pub palette: ColorPalette,
    /// Stroke width, in pixels, of the drawn box outlines.
    pub thickness: u32,
}

impl BoxAnnotator {
    /// Creates a new box annotator.
    pub fn new(palette: ColorPalette, thickness: u32) -> Self {
        Self { palette, thickness }
    }
}

impl Default for BoxAnnotator {
    fn default() -> Self {
        Self::new(ColorPalette::default(), 2)
    }
}

/// Draws a [`crate::geometry::LineZone`]'s tripwire and its live
/// in/out counts.
#[derive(Debug, Clone)]
pub struct LineZoneAnnotator {
    /// Color of the line and its count label.
    pub color: Color,
    /// Stroke width, in pixels, of the drawn line.
    pub thickness: u32,
    /// Relative size of the count label text.
    pub text_scale: f32,
}

impl LineZoneAnnotator {
    /// Creates a new line-zone annotator.
    pub fn new(color: Color, thickness: u32, text_scale: f32) -> Self {
        Self {
            color,
            thickness,
            text_scale,
        }
    }
}

impl Default for LineZoneAnnotator {
    fn default() -> Self {
        Self::new(Color::new(255, 255, 255), 2, 1.0)
    }
}
