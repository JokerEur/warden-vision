//! A minimal built-in 5x5 bitmap font.
//!
//! The `annotate-image` backend needs to render short labels (class,
//! confidence, tracker id) without pulling in a font-rasterization crate
//! just for that; this table covers uppercase ASCII, digits, and the
//! handful of punctuation marks annotation labels actually use. Anything
//! else renders as a blank cell. Callers wanting higher-fidelity
//! typography can swap in a dedicated text-rendering crate without
//! changing the public `Annotator` API.

use image::RgbaImage;

use crate::annotators::Color;

const GLYPH_WIDTH: u32 = 5;
const GLYPH_HEIGHT: u32 = 5;

/// Pixel width of `text` if drawn with [`draw_text`] at `scale`.
pub(super) fn text_width(text: &str, scale: u32) -> f32 {
    let scale = scale.max(1);
    (text.chars().count() as u32 * (GLYPH_WIDTH + 1) * scale) as f32
}

/// Pixel height of a line of text drawn with [`draw_text`] at `scale`.
pub(super) fn text_height(scale: u32) -> f32 {
    (GLYPH_HEIGHT * scale.max(1)) as f32
}

/// Row-major bitmap for `c`: 5 rows, each the low 5 bits of a `u8` (bit 4 =
/// leftmost pixel). Returns `None` for unsupported characters.
fn glyph(c: char) -> Option<[u8; 5]> {
    Some(match c.to_ascii_uppercase() {
        ' ' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00100],
        ':' => [0b00000, 0b00100, 0b00000, 0b00100, 0b00000],
        '%' => [0b10001, 0b00010, 0b00100, 0b01000, 0b10001],
        '-' => [0b00000, 0b00000, 0b11111, 0b00000, 0b00000],
        '#' => [0b01010, 0b11111, 0b01010, 0b11111, 0b01010],
        '0' => [0b11111, 0b10001, 0b10001, 0b10001, 0b11111],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b01110],
        '2' => [0b11111, 0b00001, 0b11111, 0b10000, 0b11111],
        '3' => [0b11111, 0b00001, 0b11111, 0b00001, 0b11111],
        '4' => [0b10001, 0b10001, 0b11111, 0b00001, 0b00001],
        '5' => [0b11111, 0b10000, 0b11111, 0b00001, 0b11111],
        '6' => [0b11111, 0b10000, 0b11111, 0b10001, 0b11111],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b00100],
        '8' => [0b11111, 0b10001, 0b11111, 0b10001, 0b11111],
        '9' => [0b11111, 0b10001, 0b11111, 0b00001, 0b11111],
        'A' => [0b01110, 0b10001, 0b11111, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b11110, 0b10001, 0b11110],
        'C' => [0b01111, 0b10000, 0b10000, 0b10000, 0b01111],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b11110, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b11110, 0b10000, 0b10000],
        'G' => [0b01111, 0b10000, 0b10111, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b11111, 0b10001, 0b10001],
        'I' => [0b11111, 0b00100, 0b00100, 0b00100, 0b11111],
        'J' => [0b00001, 0b00001, 0b00001, 0b10001, 0b01110],
        'K' => [0b10001, 0b10010, 0b11100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b11110, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b11110, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b01110, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b01010, 0b00100, 0b01010, 0b10001],
        'Y' => [0b10001, 0b01010, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00010, 0b00100, 0b01000, 0b11111],
        _ => return None,
    })
}

/// Draws `text` with its top-left corner at `(x, y)`, each glyph pixel
/// scaled up to a `scale x scale` block. Out-of-bounds pixels are silently
/// clipped.
pub(super) fn draw_text(
    image: &mut RgbaImage,
    text: &str,
    x: f32,
    y: f32,
    color: Color,
    scale: u32,
) {
    let (width, height) = image.dimensions();
    let scale = scale.max(1);
    let rgba = image::Rgba([color.r, color.g, color.b, 255]);
    let mut cursor_x = x.round() as i64;
    let y = y.round() as i64;

    for ch in text.chars() {
        if let Some(rows) = glyph(ch) {
            for (row, bits) in rows.iter().enumerate() {
                for col in 0..GLYPH_WIDTH {
                    let bit = (bits >> (GLYPH_WIDTH - 1 - col)) & 1;
                    if bit == 0 {
                        continue;
                    }
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = cursor_x + (col * scale + sx) as i64;
                            let py = y + (row as u32 * scale + sy) as i64;
                            if px < 0 || py < 0 || px as u32 >= width || py as u32 >= height {
                                continue;
                            }
                            image.put_pixel(px as u32, py as u32, rgba);
                        }
                    }
                }
            }
        }
        cursor_x += ((GLYPH_WIDTH + 1) * scale) as i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_character_is_blank() {
        assert!(glyph('!').is_none());
    }

    #[test]
    fn digits_and_letters_are_supported() {
        assert!(glyph('7').is_some());
        assert!(glyph('z').is_some());
    }

    #[test]
    fn draw_text_sets_at_least_one_pixel_for_nonblank_text() {
        let mut image = RgbaImage::new(50, 20);
        draw_text(&mut image, "A", 0.0, 0.0, Color::new(255, 0, 0), 1);
        let has_colored_pixel = image.pixels().any(|p| *p == image::Rgba([255, 0, 0, 255]));
        assert!(has_colored_pixel);
    }

    #[test]
    fn draw_text_clips_to_image_bounds_without_panicking() {
        let mut image = RgbaImage::new(4, 4);
        draw_text(
            &mut image,
            "HELLO WORLD",
            0.0,
            0.0,
            Color::new(0, 255, 0),
            2,
        );
    }
}
