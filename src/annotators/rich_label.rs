//! A [`LabelAnnotator`](crate::annotators::LabelAnnotator) alternative
//! that rasterizes real TTF/OTF glyphs (via `ab_glyph`) instead of this
//! crate's built-in fixed 5x5 bitmap font — needed for anything beyond
//! plain ASCII digits/uppercase letters (accented characters, non-Latin
//! scripts, ...).
//!
//! Requires the `annotate-image` feature, and — unlike every other
//! annotator in this crate — has no `annotate-opencv` implementation:
//! OpenCV's built-in text rendering (`imgproc::put_text`) only supports a
//! handful of Hershey vector fonts, not arbitrary TTF/OTF files, so
//! "rich" text isn't available on that backend without pulling in
//! FreeType bindings, which is a bigger dependency than this crate takes
//! on. Use [`LabelAnnotator`](crate::annotators::LabelAnnotator) there.
//!
//! This crate does not bundle a font, to sidestep font-licensing
//! questions entirely: bring your own TTF/OTF file's bytes and load them
//! with `ab_glyph::FontRef::try_from_slice`.

use ab_glyph::{point, Font, FontRef, ScaleFont};
use image::RgbaImage;

use crate::annotators::{Annotator, Color, ColorPalette};
use crate::core::Detections;
use crate::geometry::Position;

/// Draws a filled label background and text (tracker id and/or
/// confidence) using a real font, anchored to each detection.
pub struct RichLabelAnnotator<'font> {
    /// Maps class ids to colors (used for the background; text is always
    /// drawn in black for contrast).
    pub palette: ColorPalette,
    /// The font glyphs are rasterized from.
    pub font: FontRef<'font>,
    /// Font size, in pixels.
    pub font_size: f32,
    /// Where on the bounding box to anchor the label.
    pub position: Position,
    /// Padding, in pixels, between the text and its background box.
    pub padding: f32,
}

impl<'font> RichLabelAnnotator<'font> {
    /// Creates a new rich-label annotator.
    pub fn new(
        palette: ColorPalette,
        font: FontRef<'font>,
        font_size: f32,
        position: Position,
        padding: f32,
    ) -> Self {
        Self {
            palette,
            font,
            font_size,
            position,
            padding,
        }
    }
}

/// Total advance width and line height of `text` at `font_size`, without
/// drawing anything — used to size the label's background box first.
fn measure_text(font: &FontRef, text: &str, font_size: f32) -> (f32, f32) {
    let scale = ab_glyph::PxScale::from(font_size);
    let scaled = font.as_scaled(scale);
    let width: f32 = text
        .chars()
        .map(|c| scaled.h_advance(font.glyph_id(c)))
        .sum();
    (width, scaled.height())
}

/// Rasterizes `text` with its baseline-relative top-left at `(x, y)`,
/// alpha-blending each glyph's per-pixel coverage as `color`.
fn draw_text(
    image: &mut RgbaImage,
    font: &FontRef,
    text: &str,
    x: f32,
    y: f32,
    font_size: f32,
    color: Color,
) {
    let (width, height) = image.dimensions();
    let scale = ab_glyph::PxScale::from(font_size);
    let scaled = font.as_scaled(scale);
    let ascent = scaled.ascent();

    let mut caret = x;
    for c in text.chars() {
        let glyph_id = font.glyph_id(c);
        let advance = scaled.h_advance(glyph_id);
        let glyph = glyph_id.with_scale_and_position(scale, point(caret, y + ascent));
        if let Some(outlined) = font.outline_glyph(glyph) {
            let bounds = outlined.px_bounds();
            outlined.draw(|gx, gy, coverage| {
                if coverage <= 0.0 {
                    return;
                }
                let px = bounds.min.x as i64 + gx as i64;
                let py = bounds.min.y as i64 + gy as i64;
                super::image_backend::alpha_blend_pixel(
                    image, px, py, color, coverage, width, height,
                );
            });
        }
        caret += advance;
    }
}

impl Annotator<RgbaImage> for RichLabelAnnotator<'_> {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let color = self.palette.by_class_id(detection.class_id);
            let label = match detection.tracker_id {
                Some(id) => format!("#{} {:.0}%", id, detection.confidence * 100.0),
                None => format!("{:.0}%", detection.confidence * 100.0),
            };
            let anchor = detection.anchor_point(self.position);
            let (text_w, text_h) = measure_text(&self.font, &label, self.font_size);

            let bg = [
                (anchor.x - self.padding).max(0.0),
                (anchor.y - text_h - self.padding * 2.0).max(0.0),
                anchor.x + text_w + self.padding * 2.0,
                anchor.y,
            ];
            super::image_backend::draw_filled_rect(image, bg, color);
            draw_text(
                image,
                &self.font,
                &label,
                bg[0] + self.padding,
                bg[1] + self.padding,
                self.font_size,
                Color::new(0, 0, 0),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Detection;

    // A tiny valid TrueType font (a handful of glyphs), embedded purely
    // for this test — not shipped as part of the crate's public API or
    // used as a default anywhere.
    const TEST_FONT: &[u8] = include_bytes!("../../tests/fixtures/test_font.ttf");

    #[test]
    fn measure_text_grows_with_more_characters() {
        let font = FontRef::try_from_slice(TEST_FONT).unwrap();
        let (short_w, _) = measure_text(&font, "1", 20.0);
        let (long_w, _) = measure_text(&font, "12345", 20.0);
        assert!(long_w > short_w);
    }

    #[test]
    fn annotate_draws_a_background_box() {
        let font = FontRef::try_from_slice(TEST_FONT).unwrap();
        let annotator =
            RichLabelAnnotator::new(ColorPalette::default(), font, 16.0, Position::TopLeft, 2.0);
        let mut image = RgbaImage::new(60, 60);
        let detections = Detections::new(vec![Detection::new([10.0, 20.0, 40.0, 50.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();

        let color = annotator.palette.by_class_id(0);
        let rgba = image::Rgba([color.r, color.g, color.b, 255]);
        assert!(image.pixels().any(|p| *p == rgba));
    }
}
