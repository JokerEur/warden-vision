//! Backend-agnostic configuration for annotators that draw a simple marker
//! (circle, dot, ellipse), a detection's mask polygon, or a text label.
//! Rendering is implemented per backend in `image_backend` / `opencv_backend`.

use crate::annotators::{Color, ColorPalette};
use crate::geometry::Position;

/// Fills each detection's [`mask`](crate::core::Detection::mask) polygon
/// with a translucent color, colored by class id.
#[derive(Debug, Clone)]
pub struct MaskAnnotator {
    /// Maps class ids to colors.
    pub palette: ColorPalette,
    /// Fill opacity, in `[0, 1]`.
    pub opacity: f32,
}

impl MaskAnnotator {
    /// Creates a new mask annotator.
    pub fn new(palette: ColorPalette, opacity: f32) -> Self {
        Self { palette, opacity }
    }
}

impl Default for MaskAnnotator {
    fn default() -> Self {
        Self::new(ColorPalette::default(), 0.5)
    }
}

/// Draws the outline of each detection's
/// [`mask`](crate::core::Detection::mask) polygon.
///
/// Also doubles as an oriented/rotated bounding box annotator: store the
/// box's 4 corners (in order) as the mask instead of a full segmentation
/// contour, and this draws exactly that rotated rectangle. Supervision
/// has a separate `OrientedBoxAnnotator` for this; this crate doesn't,
/// since it would be identical code to what's already here.
#[derive(Debug, Clone)]
pub struct PolygonAnnotator {
    /// Maps class ids to colors.
    pub palette: ColorPalette,
    /// Stroke width, in pixels.
    pub thickness: u32,
}

impl PolygonAnnotator {
    /// Creates a new polygon annotator.
    pub fn new(palette: ColorPalette, thickness: u32) -> Self {
        Self { palette, thickness }
    }
}

impl Default for PolygonAnnotator {
    fn default() -> Self {
        Self::new(ColorPalette::default(), 2)
    }
}

/// Draws a hollow circle at each detection's anchor point.
#[derive(Debug, Clone)]
pub struct CircleAnnotator {
    /// Maps class ids to colors.
    pub palette: ColorPalette,
    /// Stroke width, in pixels.
    pub thickness: u32,
    /// Where on the bounding box to anchor the circle.
    pub position: Position,
}

impl CircleAnnotator {
    /// Creates a new circle annotator.
    pub fn new(palette: ColorPalette, thickness: u32, position: Position) -> Self {
        Self {
            palette,
            thickness,
            position,
        }
    }
}

impl Default for CircleAnnotator {
    fn default() -> Self {
        Self::new(ColorPalette::default(), 2, Position::Center)
    }
}

/// Draws a small filled dot at each detection's anchor point.
#[derive(Debug, Clone)]
pub struct DotAnnotator {
    /// Maps class ids to colors.
    pub palette: ColorPalette,
    /// Dot radius, in pixels.
    pub radius: u32,
    /// Where on the bounding box to anchor the dot.
    pub position: Position,
}

impl DotAnnotator {
    /// Creates a new dot annotator.
    pub fn new(palette: ColorPalette, radius: u32, position: Position) -> Self {
        Self {
            palette,
            radius,
            position,
        }
    }
}

impl Default for DotAnnotator {
    fn default() -> Self {
        Self::new(ColorPalette::default(), 4, Position::Center)
    }
}

/// Draws a flattened ellipse "shadow" under each detection, centered at the
/// bottom of its bounding box.
#[derive(Debug, Clone)]
pub struct EllipseAnnotator {
    /// Maps class ids to colors.
    pub palette: ColorPalette,
    /// Stroke width, in pixels.
    pub thickness: u32,
    /// Ellipse height as a fraction of the bounding box width.
    pub height_ratio: f32,
}

impl EllipseAnnotator {
    /// Creates a new ellipse annotator.
    pub fn new(palette: ColorPalette, thickness: u32, height_ratio: f32) -> Self {
        Self {
            palette,
            thickness,
            height_ratio,
        }
    }
}

impl Default for EllipseAnnotator {
    fn default() -> Self {
        Self::new(ColorPalette::default(), 2, 0.15)
    }
}

/// Draws a filled label background and text (tracker id and/or confidence)
/// anchored to each detection.
#[derive(Debug, Clone)]
pub struct LabelAnnotator {
    /// Maps class ids to colors.
    pub palette: ColorPalette,
    /// Where on the bounding box to anchor the label.
    pub position: Position,
    /// Relative size of the label text.
    pub text_scale: u32,
    /// Padding, in pixels, between the text and its background box.
    pub padding: f32,
}

impl LabelAnnotator {
    /// Creates a new label annotator.
    pub fn new(palette: ColorPalette, position: Position, text_scale: u32, padding: f32) -> Self {
        Self {
            palette,
            position,
            text_scale,
            padding,
        }
    }
}

impl Default for LabelAnnotator {
    fn default() -> Self {
        Self::new(ColorPalette::default(), Position::TopLeft, 1, 2.0)
    }
}

/// Draws each detection's bounding box with rounded corners.
#[derive(Debug, Clone)]
pub struct RoundBoxAnnotator {
    /// Maps class ids to colors.
    pub palette: ColorPalette,
    /// Stroke width, in pixels.
    pub thickness: u32,
    /// Corner radius, in pixels (clamped to at most half the box's
    /// shorter side).
    pub corner_radius: u32,
}

impl RoundBoxAnnotator {
    /// Creates a new round-box annotator.
    pub fn new(palette: ColorPalette, thickness: u32, corner_radius: u32) -> Self {
        Self {
            palette,
            thickness,
            corner_radius,
        }
    }
}

impl Default for RoundBoxAnnotator {
    fn default() -> Self {
        Self::new(ColorPalette::default(), 2, 10)
    }
}

/// Draws only the four corners of each detection's bounding box, as short
/// L-shaped marks — a lighter-weight alternative to [`crate::annotators::BoxAnnotator`]'s
/// full outline.
#[derive(Debug, Clone)]
pub struct BoxCornerAnnotator {
    /// Maps class ids to colors.
    pub palette: ColorPalette,
    /// Stroke width, in pixels.
    pub thickness: u32,
    /// Length, in pixels, of each corner mark's two arms.
    pub corner_length: u32,
}

impl BoxCornerAnnotator {
    /// Creates a new box-corner annotator.
    pub fn new(palette: ColorPalette, thickness: u32, corner_length: u32) -> Self {
        Self {
            palette,
            thickness,
            corner_length,
        }
    }
}

impl Default for BoxCornerAnnotator {
    fn default() -> Self {
        Self::new(ColorPalette::default(), 3, 15)
    }
}

/// Draws a small filled triangle, tip pointing down at each detection's
/// anchor point — a marker that flags a point without covering it the
/// way a dot does.
#[derive(Debug, Clone)]
pub struct TriangleAnnotator {
    /// Maps class ids to colors.
    pub palette: ColorPalette,
    /// Full width of the triangle's base, in pixels.
    pub base: u32,
    /// Height from base to tip, in pixels.
    pub height: u32,
    /// Where on the bounding box the triangle's tip points to.
    pub position: Position,
}

impl TriangleAnnotator {
    /// Creates a new triangle annotator.
    pub fn new(palette: ColorPalette, base: u32, height: u32, position: Position) -> Self {
        Self {
            palette,
            base,
            height,
            position,
        }
    }
}

impl Default for TriangleAnnotator {
    fn default() -> Self {
        Self::new(ColorPalette::default(), 12, 14, Position::TopCenter)
    }
}

/// Draws a soft fading glow around each detection's bounding box.
///
/// Glows outward from the box's axis-aligned rectangle rather than from
/// an instance segmentation mask's exact contour (that needs true
/// polygon dilation, not just a per-vertex offset). If you have a mask
/// and want the glow to hug its shape, layer [`PolygonAnnotator`] at
/// decreasing opacity yourself instead.
#[derive(Debug, Clone)]
pub struct HaloAnnotator {
    /// Maps class ids to colors.
    pub palette: ColorPalette,
    /// Opacity at the box edge (fades linearly to `0` over `kernel_size`
    /// pixels outward).
    pub opacity: f32,
    /// How far, in pixels, the glow extends outward from the box.
    pub kernel_size: u32,
}

impl HaloAnnotator {
    /// Creates a new halo annotator.
    pub fn new(palette: ColorPalette, opacity: f32, kernel_size: u32) -> Self {
        Self {
            palette,
            opacity,
            kernel_size,
        }
    }
}

impl Default for HaloAnnotator {
    fn default() -> Self {
        Self::new(ColorPalette::default(), 0.4, 12)
    }
}

/// Draws a small bar above each detection, filled left-to-right in
/// proportion to its confidence.
#[derive(Debug, Clone)]
pub struct PercentageBarAnnotator {
    /// Maps class ids to the bar's fill color.
    pub palette: ColorPalette,
    /// Color of the unfilled portion of the bar.
    pub background_color: Color,
    /// Full bar width, in pixels.
    pub bar_width: u32,
    /// Bar height, in pixels.
    pub bar_height: u32,
    /// Where on the bounding box to anchor the bar (its bottom-center
    /// sits at this point).
    pub position: Position,
}

impl PercentageBarAnnotator {
    /// Creates a new percentage-bar annotator.
    pub fn new(
        palette: ColorPalette,
        background_color: Color,
        bar_width: u32,
        bar_height: u32,
        position: Position,
    ) -> Self {
        Self {
            palette,
            background_color,
            bar_width,
            bar_height,
            position,
        }
    }
}

impl Default for PercentageBarAnnotator {
    fn default() -> Self {
        Self::new(
            ColorPalette::default(),
            Color::new(80, 80, 80),
            60,
            6,
            Position::TopCenter,
        )
    }
}

/// Overlays a fixed icon image, centered at each detection's anchor
/// point.
///
/// Generic over the image type so the same shape works for both
/// rendering backends: construct an `IconAnnotator<image::RgbaImage>`
/// for the pure-Rust backend or an `IconAnnotator<opencv::core::Mat>`
/// for the OpenCV one, whichever `Annotator` impl you pull in.
#[derive(Debug, Clone)]
pub struct IconAnnotator<Image> {
    /// The icon to overlay.
    pub icon: Image,
    /// Where on the bounding box to center the icon.
    pub position: Position,
}

impl<Image> IconAnnotator<Image> {
    /// Creates a new icon annotator.
    pub fn new(icon: Image, position: Position) -> Self {
        Self { icon, position }
    }
}

/// Dims/tints everything *outside* each detection's bounding box,
/// visually pulling attention onto the detected objects.
#[derive(Debug, Clone)]
pub struct BackgroundOverlayAnnotator {
    /// Tint color blended over the background.
    pub color: Color,
    /// Tint opacity, in `[0, 1]`.
    pub opacity: f32,
}

impl BackgroundOverlayAnnotator {
    /// Creates a new background-overlay annotator.
    pub fn new(color: Color, opacity: f32) -> Self {
        Self { color, opacity }
    }
}

impl Default for BackgroundOverlayAnnotator {
    fn default() -> Self {
        Self::new(Color::new(0, 0, 0), 0.6)
    }
}
