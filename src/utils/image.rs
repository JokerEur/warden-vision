//! Pure-Rust image utilities (resize, letterbox, crop, overlay) built on
//! `image::RgbaImage`. Requires the `annotate-image` feature.

use image::{Rgba, RgbaImage};

use crate::annotators::Color;

/// Resizes `image` to fit within `target_width x target_height`, preserving
/// aspect ratio (the result may be smaller than the target box on one
/// axis). Use [`letterbox`] if you need the exact target dimensions.
pub fn resize_keeping_aspect_ratio(
    image: &RgbaImage,
    target_width: u32,
    target_height: u32,
) -> RgbaImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || target_width == 0 || target_height == 0 {
        return RgbaImage::new(target_width, target_height);
    }
    let scale = (target_width as f32 / width as f32).min(target_height as f32 / height as f32);
    let new_width = ((width as f32 * scale).round() as u32).max(1);
    let new_height = ((height as f32 * scale).round() as u32).max(1);
    image::imageops::resize(
        image,
        new_width,
        new_height,
        image::imageops::FilterType::Triangle,
    )
}

/// Resizes `image` to fit within `target_width x target_height` (preserving
/// aspect ratio, via [`resize_keeping_aspect_ratio`]), then pads it with
/// `fill` to exactly `target_width x target_height`, centered.
pub fn letterbox(
    image: &RgbaImage,
    target_width: u32,
    target_height: u32,
    fill: Color,
) -> RgbaImage {
    let resized = resize_keeping_aspect_ratio(image, target_width, target_height);
    let mut canvas = RgbaImage::from_pixel(
        target_width,
        target_height,
        Rgba([fill.r, fill.g, fill.b, 255]),
    );
    let (resized_width, resized_height) = resized.dimensions();
    let x = target_width.saturating_sub(resized_width) / 2;
    let y = target_height.saturating_sub(resized_height) / 2;
    overlay_image(&mut canvas, &resized, x as i64, y as i64);
    canvas
}

/// Crops `image` to `bbox` (`[x1, y1, x2, y2]`), clipped to the image
/// bounds. Returns a `0x0` image if the clipped region is empty.
pub fn crop_image(image: &RgbaImage, bbox: [f32; 4]) -> RgbaImage {
    let (width, height) = image.dimensions();
    let x1 = bbox[0].round().clamp(0.0, width as f32) as u32;
    let y1 = bbox[1].round().clamp(0.0, height as f32) as u32;
    let x2 = bbox[2].round().clamp(0.0, width as f32) as u32;
    let y2 = bbox[3].round().clamp(0.0, height as f32) as u32;
    let crop_width = x2.saturating_sub(x1);
    let crop_height = y2.saturating_sub(y1);
    if crop_width == 0 || crop_height == 0 {
        return RgbaImage::new(0, 0);
    }
    image::imageops::crop_imm(image, x1, y1, crop_width, crop_height).to_image()
}

/// Alpha-blends `overlay` onto `base` with its top-left corner at
/// `(x, y)`, using each overlay pixel's alpha channel. Parts of `overlay`
/// that fall outside `base`'s bounds are silently clipped.
pub fn overlay_image(base: &mut RgbaImage, overlay: &RgbaImage, x: i64, y: i64) {
    let (base_width, base_height) = base.dimensions();
    let (overlay_width, overlay_height) = overlay.dimensions();
    for oy in 0..overlay_height {
        for ox in 0..overlay_width {
            let px = x + ox as i64;
            let py = y + oy as i64;
            if px < 0 || py < 0 || px as u32 >= base_width || py as u32 >= base_height {
                continue;
            }
            let src = overlay.get_pixel(ox, oy);
            let alpha = src[3] as f32 / 255.0;
            if alpha <= 0.0 {
                continue;
            }
            let dst = base.get_pixel_mut(px as u32, py as u32);
            for c in 0..3 {
                dst[c] = (dst[c] as f32 * (1.0 - alpha) + src[c] as f32 * alpha).round() as u8;
            }
            dst[3] = 255;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_keeping_aspect_ratio_preserves_ratio_and_fits_target() {
        let image = RgbaImage::new(100, 50);
        let resized = resize_keeping_aspect_ratio(&image, 40, 40);
        let (w, h) = resized.dimensions();
        assert!(w <= 40 && h <= 40);
        // 2:1 aspect ratio preserved.
        assert!((w as f32 / h as f32 - 2.0).abs() < 0.1);
    }

    #[test]
    fn letterbox_produces_exact_target_dimensions() {
        let image = RgbaImage::new(100, 50);
        let result = letterbox(&image, 40, 40, Color::new(0, 0, 0));
        assert_eq!(result.dimensions(), (40, 40));
    }

    #[test]
    fn letterbox_pads_with_fill_color() {
        let image = RgbaImage::from_pixel(100, 50, Rgba([255, 255, 255, 255]));
        let fill = Color::new(10, 20, 30);
        let result = letterbox(&image, 40, 40, fill);
        // Top row should be padding for a wide source image letterboxed
        // into a square target.
        let top_pixel = result.get_pixel(0, 0);
        assert_eq!(*top_pixel, Rgba([10, 20, 30, 255]));
    }

    #[test]
    fn crop_image_extracts_the_requested_region() {
        let image = RgbaImage::from_fn(20, 20, |x, _y| {
            if x < 10 {
                Rgba([255, 0, 0, 255])
            } else {
                Rgba([0, 0, 255, 255])
            }
        });
        let cropped = crop_image(&image, [10.0, 0.0, 20.0, 20.0]);
        assert_eq!(cropped.dimensions(), (10, 20));
        assert_eq!(*cropped.get_pixel(0, 0), Rgba([0, 0, 255, 255]));
    }

    #[test]
    fn crop_image_clips_out_of_bounds_bbox() {
        let image = RgbaImage::new(10, 10);
        let cropped = crop_image(&image, [-5.0, -5.0, 5.0, 5.0]);
        assert_eq!(cropped.dimensions(), (5, 5));
    }

    #[test]
    fn crop_image_returns_empty_for_fully_out_of_bounds_bbox() {
        let image = RgbaImage::new(10, 10);
        let cropped = crop_image(&image, [20.0, 20.0, 30.0, 30.0]);
        assert_eq!(cropped.dimensions(), (0, 0));
    }

    #[test]
    fn overlay_image_blends_opaque_pixels() {
        let mut base = RgbaImage::from_pixel(10, 10, Rgba([0, 0, 0, 255]));
        let overlay = RgbaImage::from_pixel(4, 4, Rgba([255, 0, 0, 255]));
        overlay_image(&mut base, &overlay, 2, 2);
        assert_eq!(*base.get_pixel(3, 3), Rgba([255, 0, 0, 255]));
        // Outside the overlay footprint, base is untouched.
        assert_eq!(*base.get_pixel(0, 0), Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn overlay_image_ignores_fully_transparent_pixels() {
        let mut base = RgbaImage::from_pixel(10, 10, Rgba([9, 9, 9, 255]));
        let overlay = RgbaImage::from_pixel(4, 4, Rgba([255, 0, 0, 0]));
        overlay_image(&mut base, &overlay, 2, 2);
        assert_eq!(*base.get_pixel(3, 3), Rgba([9, 9, 9, 255]));
    }

    #[test]
    fn overlay_image_clips_out_of_bounds_placement_without_panicking() {
        let mut base = RgbaImage::new(10, 10);
        let overlay = RgbaImage::from_pixel(4, 4, Rgba([255, 0, 0, 255]));
        overlay_image(&mut base, &overlay, -2, 8);
    }
}
