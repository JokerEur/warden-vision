//! Pure-Rust annotation backend built on the `image` crate: draws directly
//! onto an `image::RgbaImage`, with no native/C++ dependency.

use image::{Rgba, RgbaImage};
use ndarray::Array2;

use crate::annotators::font::{draw_text, text_height, text_width};
use crate::annotators::heatmap::heat_color;
use crate::annotators::{
    Annotator, BackgroundOverlayAnnotator, BlurAnnotator, BoxAnnotator, BoxCornerAnnotator,
    CircleAnnotator, Color, DotAnnotator, EdgeAnnotator, EllipseAnnotator, HaloAnnotator,
    HeatMapAnnotator, IconAnnotator, LabelAnnotator, LineZoneAnnotator, MaskAnnotator,
    PercentageBarAnnotator, PixelateAnnotator, PolygonAnnotator, PolygonZoneAnnotator,
    RoundBoxAnnotator, TraceAnnotator, TriangleAnnotator, VertexAnnotator, VertexLabelAnnotator,
};
use crate::core::{Detections, KeyPoints};
use crate::geometry::{polygon_to_rect, LineZone, Point, PolygonZone, Position, Zone};
use crate::utils::overlay_image;

impl From<Color> for Rgba<u8> {
    fn from(color: Color) -> Self {
        Rgba([color.r, color.g, color.b, 255])
    }
}

impl Annotator<RgbaImage> for BoxAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let color = self.palette.by_class_id(detection.class_id);
            draw_rect(image, detection.bbox, color, self.thickness);

            let label = match detection.tracker_id {
                Some(id) => format!("#{} {:.0}%", id, detection.confidence * 100.0),
                None => format!("{:.0}%", detection.confidence * 100.0),
            };
            let text_y = (detection.bbox[1] - 7.0).max(0.0);
            draw_text(image, &label, detection.bbox[0], text_y, color, 1);
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for LineZoneAnnotator {
    type Subject = LineZone;

    fn annotate(&self, image: &mut RgbaImage, zone: &LineZone) -> crate::Result<()> {
        draw_line(image, zone.start, zone.end, self.color, self.thickness);

        let label = format!("IN:{} OUT:{}", zone.in_count(), zone.out_count());
        let scale = self.text_scale.max(1.0).round() as u32;
        let text_y = (zone.start.y.min(zone.end.y) - 12.0).max(0.0);
        draw_text(
            image,
            &label,
            zone.start.x.min(zone.end.x),
            text_y,
            self.color,
            scale,
        );
        Ok(())
    }
}

fn put_pixel_clamped(image: &mut RgbaImage, x: i64, y: i64, color: Color, width: u32, height: u32) {
    if x < 0 || y < 0 || x as u32 >= width || y as u32 >= height {
        return;
    }
    image.put_pixel(x as u32, y as u32, color.into());
}

fn draw_hline(
    image: &mut RgbaImage,
    x1: i64,
    x2: i64,
    y: i64,
    color: Color,
    width: u32,
    height: u32,
) {
    let (lo, hi) = (x1.min(x2), x1.max(x2));
    for x in lo..=hi {
        put_pixel_clamped(image, x, y, color, width, height);
    }
}

fn draw_vline(
    image: &mut RgbaImage,
    y1: i64,
    y2: i64,
    x: i64,
    color: Color,
    width: u32,
    height: u32,
) {
    let (lo, hi) = (y1.min(y2), y1.max(y2));
    for y in lo..=hi {
        put_pixel_clamped(image, x, y, color, width, height);
    }
}

/// Draws a hollow (unfilled) rectangle outline, `thickness` pixels wide,
/// clipped to the image bounds.
fn draw_rect(image: &mut RgbaImage, bbox: [f32; 4], color: Color, thickness: u32) {
    let (width, height) = image.dimensions();
    let x1 = bbox[0].round() as i64;
    let y1 = bbox[1].round() as i64;
    let x2 = bbox[2].round() as i64;
    let y2 = bbox[3].round() as i64;

    for t in 0..thickness.max(1) as i64 {
        draw_hline(image, x1 - t, x2 + t, y1 - t, color, width, height);
        draw_hline(image, x1 - t, x2 + t, y2 + t, color, width, height);
        draw_vline(image, y1 - t, y2 + t, x1 - t, color, width, height);
        draw_vline(image, y1 - t, y2 + t, x2 + t, color, width, height);
    }
}

/// Plots a 1px-wide line segment using Bresenham's algorithm.
fn draw_bresenham(
    image: &mut RgbaImage,
    (mut x0, mut y0): (i64, i64),
    (x1, y1): (i64, i64),
    color: Color,
    width: u32,
    height: u32,
) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        put_pixel_clamped(image, x0, y0, color, width, height);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Draws a line between `p0` and `p1`, `thickness` pixels wide, by offsetting
/// parallel 1px Bresenham lines perpendicular to the line's direction.
fn draw_line(image: &mut RgbaImage, p0: Point, p1: Point, color: Color, thickness: u32) {
    let (width, height) = image.dimensions();
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let len = (dx * dx + dy * dy).sqrt();
    let (nx, ny) = if len > 0.0 {
        (-dy / len, dx / len)
    } else {
        (0.0, 0.0)
    };

    let t = thickness.max(1) as i64;
    let half = t / 2;
    for offset in -half..(t - half) {
        let ox = nx * offset as f32;
        let oy = ny * offset as f32;
        draw_bresenham(
            image,
            ((p0.x + ox).round() as i64, (p0.y + oy).round() as i64),
            ((p1.x + ox).round() as i64, (p1.y + oy).round() as i64),
            color,
            width,
            height,
        );
    }
}

/// Blends `color` into the pixel at `(x, y)` with weight `alpha` (`0` =
/// unchanged, `1` = fully replaced). Out-of-bounds coordinates are ignored.
pub(super) fn alpha_blend_pixel(
    image: &mut RgbaImage,
    x: i64,
    y: i64,
    color: Color,
    alpha: f32,
    width: u32,
    height: u32,
) {
    if x < 0 || y < 0 || x as u32 >= width || y as u32 >= height {
        return;
    }
    let a = alpha.clamp(0.0, 1.0);
    let px = image.get_pixel_mut(x as u32, y as u32);
    px[0] = (px[0] as f32 * (1.0 - a) + color.r as f32 * a).round() as u8;
    px[1] = (px[1] as f32 * (1.0 - a) + color.g as f32 * a).round() as u8;
    px[2] = (px[2] as f32 * (1.0 - a) + color.b as f32 * a).round() as u8;
    px[3] = 255;
}

/// Draws a solid-filled rectangle, clipped to the image bounds.
pub(super) fn draw_filled_rect(image: &mut RgbaImage, bbox: [f32; 4], color: Color) {
    let (width, height) = image.dimensions();
    let x1 = bbox[0].round() as i64;
    let y1 = bbox[1].round() as i64;
    let x2 = bbox[2].round() as i64;
    let y2 = bbox[3].round() as i64;
    for y in y1..=y2 {
        draw_hline(image, x1, x2, y, color, width, height);
    }
}

#[allow(clippy::too_many_arguments)]
fn plot_circle_octants(
    image: &mut RgbaImage,
    cx: i64,
    cy: i64,
    x: i64,
    y: i64,
    color: Color,
    width: u32,
    height: u32,
) {
    put_pixel_clamped(image, cx + x, cy + y, color, width, height);
    put_pixel_clamped(image, cx - x, cy + y, color, width, height);
    put_pixel_clamped(image, cx + x, cy - y, color, width, height);
    put_pixel_clamped(image, cx - x, cy - y, color, width, height);
    put_pixel_clamped(image, cx + y, cy + x, color, width, height);
    put_pixel_clamped(image, cx - y, cy + x, color, width, height);
    put_pixel_clamped(image, cx + y, cy - x, color, width, height);
    put_pixel_clamped(image, cx - y, cy - x, color, width, height);
}

/// Plots a single-pixel-wide circle outline (midpoint circle algorithm).
fn draw_circle_ring(
    image: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: f32,
    color: Color,
    width: u32,
    height: u32,
) {
    let cxi = cx.round() as i64;
    let cyi = cy.round() as i64;
    if radius < 0.5 {
        put_pixel_clamped(image, cxi, cyi, color, width, height);
        return;
    }
    let r = radius.round() as i64;
    let mut x = 0i64;
    let mut y = r;
    let mut d = 1 - r;
    plot_circle_octants(image, cxi, cyi, x, y, color, width, height);
    while x < y {
        x += 1;
        if d < 0 {
            d += 2 * x + 1;
        } else {
            y -= 1;
            d += 2 * (x - y) + 1;
        }
        plot_circle_octants(image, cxi, cyi, x, y, color, width, height);
    }
}

/// Draws a circle outline `thickness` pixels wide by stacking concentric
/// rings.
#[allow(clippy::too_many_arguments)]
fn draw_circle_outline(
    image: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: f32,
    color: Color,
    thickness: u32,
    width: u32,
    height: u32,
) {
    let thickness = thickness.max(1) as f32;
    let mut r = (radius - thickness / 2.0).max(0.0);
    let end = radius + thickness / 2.0;
    while r <= end {
        draw_circle_ring(image, cx, cy, r, color, width, height);
        r += 1.0;
    }
}

/// Draws a filled disc via horizontal scanlines.
fn draw_filled_circle(
    image: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: f32,
    color: Color,
    width: u32,
    height: u32,
) {
    let r = radius.max(0.0);
    if r < 0.5 {
        put_pixel_clamped(
            image,
            cx.round() as i64,
            cy.round() as i64,
            color,
            width,
            height,
        );
        return;
    }
    let min_y = (cy - r).floor() as i64;
    let max_y = (cy + r).ceil() as i64;
    for y in min_y..=max_y {
        let dy = y as f32 - cy;
        let inside = r * r - dy * dy;
        if inside < 0.0 {
            continue;
        }
        let dx = inside.sqrt();
        draw_hline(
            image,
            (cx - dx).round() as i64,
            (cx + dx).round() as i64,
            y,
            color,
            width,
            height,
        );
    }
}

/// Draws an axis-aligned ellipse outline (semi-axes `a`, `b`) by sampling
/// its parametric form, stacking `thickness` concentric copies.
#[allow(clippy::too_many_arguments)]
fn draw_ellipse_outline(
    image: &mut RgbaImage,
    cx: f32,
    cy: f32,
    a: f32,
    b: f32,
    color: Color,
    thickness: u32,
    width: u32,
    height: u32,
) {
    let a = a.max(0.0);
    let b = b.max(0.0);
    let steps = ((a.max(b) * 8.0) as usize).max(36);
    let thickness = thickness.max(1);
    for layer in 0..thickness {
        let offset = layer as f32 - (thickness as f32 - 1.0) / 2.0;
        for i in 0..steps {
            let theta = (i as f32 / steps as f32) * std::f32::consts::TAU;
            let x = cx + (a + offset) * theta.cos();
            let y = cy + (b + offset) * theta.sin();
            put_pixel_clamped(
                image,
                x.round() as i64,
                y.round() as i64,
                color,
                width,
                height,
            );
        }
    }
}

/// Draws a circular arc centered at `(cx, cy)` from `start_deg` to
/// `end_deg` (standard math convention: 0° = +x, 90° = +y), `thickness`
/// pixels wide, by sampling its parametric form.
#[allow(clippy::too_many_arguments)]
fn draw_arc(
    image: &mut RgbaImage,
    cx: f32,
    cy: f32,
    radius: f32,
    start_deg: f32,
    end_deg: f32,
    color: Color,
    thickness: u32,
    width: u32,
    height: u32,
) {
    let radius = radius.max(0.0);
    let steps = ((radius * (end_deg - start_deg).abs() / 90.0) as usize).max(8);
    let thickness = thickness.max(1);
    for layer in 0..thickness {
        let r = (radius - (thickness as f32 - 1.0) / 2.0 + layer as f32).max(0.0);
        for i in 0..=steps {
            let theta =
                (start_deg + (end_deg - start_deg) * (i as f32 / steps as f32)).to_radians();
            let x = cx + r * theta.cos();
            let y = cy + r * theta.sin();
            put_pixel_clamped(
                image,
                x.round() as i64,
                y.round() as i64,
                color,
                width,
                height,
            );
        }
    }
}

/// Fills a polygon (given as `(x, y)` vertices) using a horizontal-scanline
/// even-odd fill, alpha-blending `color` into each covered pixel.
fn draw_filled_polygon(
    image: &mut RgbaImage,
    polygon: &[(f32, f32)],
    color: Color,
    alpha: f32,
    width: u32,
    height: u32,
) {
    if polygon.len() < 3 {
        return;
    }
    let min_y = polygon
        .iter()
        .map(|p| p.1)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as i64;
    let max_y = polygon
        .iter()
        .map(|p| p.1)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(height as f32 - 1.0) as i64;

    let n = polygon.len();
    for y in min_y..=max_y {
        let yf = y as f32 + 0.5;
        let mut xs: Vec<f32> = Vec::new();
        for i in 0..n {
            let (x1, y1) = polygon[i];
            let (x2, y2) = polygon[(i + 1) % n];
            if (y1 <= yf && y2 > yf) || (y2 <= yf && y1 > yf) {
                let t = (yf - y1) / (y2 - y1);
                xs.push(x1 + t * (x2 - x1));
            }
        }
        xs.sort_by(f32::total_cmp);
        for pair in xs.chunks_exact(2) {
            let x_start = pair[0].round() as i64;
            let x_end = pair[1].round() as i64;
            for x in x_start..=x_end {
                alpha_blend_pixel(image, x, y, color, alpha, width, height);
            }
        }
    }
}

/// Box-blurs the `bbox` region of `target`, reading from `source` (so
/// multiple regions in one `annotate` call all blur the same pre-annotation
/// frame regardless of processing order).
fn box_blur_region(target: &mut RgbaImage, source: &RgbaImage, bbox: [f32; 4], kernel_size: u32) {
    let (width, height) = source.dimensions();
    let x1 = bbox[0].round().clamp(0.0, width as f32) as i64;
    let y1 = bbox[1].round().clamp(0.0, height as f32) as i64;
    let x2 = bbox[2].round().clamp(0.0, width as f32) as i64;
    let y2 = bbox[3].round().clamp(0.0, height as f32) as i64;
    let rw = (x2 - x1).max(0);
    let rh = (y2 - y1).max(0);
    if rw == 0 || rh == 0 {
        return;
    }

    let half = (kernel_size.max(1) as i64) / 2;
    for ly in 0..rh {
        for lx in 0..rw {
            let mut sum = [0u32; 3];
            let mut count = 0u32;
            for dy in -half..=half {
                for dx in -half..=half {
                    let sx = lx + dx;
                    let sy = ly + dy;
                    if sx < 0 || sy < 0 || sx >= rw || sy >= rh {
                        continue;
                    }
                    let p = source.get_pixel((x1 + sx) as u32, (y1 + sy) as u32);
                    sum[0] += p[0] as u32;
                    sum[1] += p[1] as u32;
                    sum[2] += p[2] as u32;
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            let avg = Rgba([
                (sum[0] / count) as u8,
                (sum[1] / count) as u8,
                (sum[2] / count) as u8,
                255,
            ]);
            target.put_pixel((x1 + lx) as u32, (y1 + ly) as u32, avg);
        }
    }
}

/// Pixelates (mosaics) the `bbox` region of `target` in `block_size`
/// squares, reading averages from `source`.
fn pixelate_region(target: &mut RgbaImage, source: &RgbaImage, bbox: [f32; 4], block_size: u32) {
    let (width, height) = source.dimensions();
    let x1 = bbox[0].round().clamp(0.0, width as f32) as i64;
    let y1 = bbox[1].round().clamp(0.0, height as f32) as i64;
    let x2 = bbox[2].round().clamp(0.0, width as f32) as i64;
    let y2 = bbox[3].round().clamp(0.0, height as f32) as i64;
    let rw = (x2 - x1).max(0);
    let rh = (y2 - y1).max(0);
    if rw == 0 || rh == 0 {
        return;
    }

    let block = block_size.max(1) as i64;
    let mut by = 0;
    while by < rh {
        let bh = block.min(rh - by);
        let mut bx = 0;
        while bx < rw {
            let bw = block.min(rw - bx);
            let mut sum = [0u32; 3];
            let mut count = 0u32;
            for ly in 0..bh {
                for lx in 0..bw {
                    let p = source.get_pixel((x1 + bx + lx) as u32, (y1 + by + ly) as u32);
                    sum[0] += p[0] as u32;
                    sum[1] += p[1] as u32;
                    sum[2] += p[2] as u32;
                    count += 1;
                }
            }
            let count = count.max(1);
            let avg = Rgba([
                (sum[0] / count) as u8,
                (sum[1] / count) as u8,
                (sum[2] / count) as u8,
                255,
            ]);
            for ly in 0..bh {
                for lx in 0..bw {
                    target.put_pixel((x1 + bx + lx) as u32, (y1 + by + ly) as u32, avg);
                }
            }
            bx += block;
        }
        by += block;
    }
}

/// Adds `intensity` to every cell of `grid` within `radius` of `(cx, cy)`.
fn splat_heat(grid: &mut Array2<f32>, cx: f32, cy: f32, radius: f32, intensity: f32) {
    let (h, w) = grid.dim();
    if w == 0 || h == 0 {
        return;
    }
    let r = radius.max(1.0);
    let min_x = (cx - r).floor().clamp(0.0, w as f32 - 1.0) as usize;
    let max_x = (cx + r).ceil().clamp(0.0, w as f32 - 1.0) as usize;
    let min_y = (cy - r).floor().clamp(0.0, h as f32 - 1.0) as usize;
    let max_y = (cy + r).ceil().clamp(0.0, h as f32 - 1.0) as usize;
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            if dx * dx + dy * dy <= r * r {
                grid[[y, x]] += intensity;
            }
        }
    }
}

impl Annotator<RgbaImage> for MaskAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        let (width, height) = image.dimensions();
        for detection in detections.iter() {
            let Some(mask) = &detection.mask else {
                continue;
            };
            if mask.len() < 3 {
                continue;
            }
            let color = self.palette.by_class_id(detection.class_id);
            let polygon: Vec<(f32, f32)> = mask.iter().map(|&[x, y]| (x, y)).collect();
            draw_filled_polygon(image, &polygon, color, self.opacity, width, height);
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for PolygonAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let Some(mask) = &detection.mask else {
                continue;
            };
            if mask.len() < 2 {
                continue;
            }
            let color = self.palette.by_class_id(detection.class_id);
            let points: Vec<Point> = mask.iter().map(|&[x, y]| Point::new(x, y)).collect();
            let n = points.len();
            for i in 0..n {
                draw_line(image, points[i], points[(i + 1) % n], color, self.thickness);
            }
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for CircleAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        let (width, height) = image.dimensions();
        for detection in detections.iter() {
            let color = self.palette.by_class_id(detection.class_id);
            let center = detection.anchor_point(self.position);
            let radius = detection.width().min(detection.height()) / 2.0;
            draw_circle_outline(
                image,
                center.x,
                center.y,
                radius,
                color,
                self.thickness,
                width,
                height,
            );
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for DotAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        let (width, height) = image.dimensions();
        for detection in detections.iter() {
            let color = self.palette.by_class_id(detection.class_id);
            let center = detection.anchor_point(self.position);
            draw_filled_circle(
                image,
                center.x,
                center.y,
                self.radius as f32,
                color,
                width,
                height,
            );
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for EllipseAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        let (width, height) = image.dimensions();
        for detection in detections.iter() {
            let color = self.palette.by_class_id(detection.class_id);
            let center = detection.anchor_point(Position::BottomCenter);
            let a = detection.width() / 2.0;
            let b = (detection.width() * self.height_ratio).max(1.0) / 2.0;
            draw_ellipse_outline(
                image,
                center.x,
                center.y,
                a,
                b,
                color,
                self.thickness,
                width,
                height,
            );
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for RoundBoxAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        let (width, height) = image.dimensions();
        for detection in detections.iter() {
            let color = self.palette.by_class_id(detection.class_id);
            let [x1, y1, x2, y2] = detection.bbox;
            let r = (self.corner_radius as f32)
                .min(detection.width().min(detection.height()) / 2.0)
                .max(0.0);

            draw_line(
                image,
                Point::new(x1 + r, y1),
                Point::new(x2 - r, y1),
                color,
                self.thickness,
            );
            draw_line(
                image,
                Point::new(x1 + r, y2),
                Point::new(x2 - r, y2),
                color,
                self.thickness,
            );
            draw_line(
                image,
                Point::new(x1, y1 + r),
                Point::new(x1, y2 - r),
                color,
                self.thickness,
            );
            draw_line(
                image,
                Point::new(x2, y1 + r),
                Point::new(x2, y2 - r),
                color,
                self.thickness,
            );

            draw_arc(
                image,
                x1 + r,
                y1 + r,
                r,
                180.0,
                270.0,
                color,
                self.thickness,
                width,
                height,
            );
            draw_arc(
                image,
                x2 - r,
                y1 + r,
                r,
                270.0,
                360.0,
                color,
                self.thickness,
                width,
                height,
            );
            draw_arc(
                image,
                x2 - r,
                y2 - r,
                r,
                0.0,
                90.0,
                color,
                self.thickness,
                width,
                height,
            );
            draw_arc(
                image,
                x1 + r,
                y2 - r,
                r,
                90.0,
                180.0,
                color,
                self.thickness,
                width,
                height,
            );
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for BoxCornerAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let color = self.palette.by_class_id(detection.class_id);
            let [x1, y1, x2, y2] = detection.bbox;
            let len = (self.corner_length as f32)
                .min(detection.width() / 2.0)
                .min(detection.height() / 2.0)
                .max(0.0);

            for &(cx, cy, dx, dy) in &[
                (x1, y1, 1.0, 1.0),
                (x2, y1, -1.0, 1.0),
                (x2, y2, -1.0, -1.0),
                (x1, y2, 1.0, -1.0),
            ] {
                draw_line(
                    image,
                    Point::new(cx, cy),
                    Point::new(cx + dx * len, cy),
                    color,
                    self.thickness,
                );
                draw_line(
                    image,
                    Point::new(cx, cy),
                    Point::new(cx, cy + dy * len),
                    color,
                    self.thickness,
                );
            }
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for TriangleAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        let (width, height) = image.dimensions();
        for detection in detections.iter() {
            let color = self.palette.by_class_id(detection.class_id);
            let tip = detection.anchor_point(self.position);
            let base_half = self.base as f32 / 2.0;
            let polygon = [
                (tip.x, tip.y),
                (tip.x - base_half, tip.y - self.height as f32),
                (tip.x + base_half, tip.y - self.height as f32),
            ];
            draw_filled_polygon(image, &polygon, color, 1.0, width, height);
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for LabelAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let color = self.palette.by_class_id(detection.class_id);
            let label = match detection.tracker_id {
                Some(id) => format!("#{} {:.0}%", id, detection.confidence * 100.0),
                None => format!("{:.0}%", detection.confidence * 100.0),
            };
            let anchor = detection.anchor_point(self.position);
            let text_w = text_width(&label, self.text_scale);
            let text_h = text_height(self.text_scale);
            let bg = [
                (anchor.x - self.padding).max(0.0),
                (anchor.y - text_h - self.padding).max(0.0),
                anchor.x + text_w + self.padding,
                anchor.y + self.padding,
            ];
            draw_filled_rect(image, bg, color);
            draw_text(
                image,
                &label,
                bg[0] + self.padding / 2.0,
                bg[1] + self.padding / 2.0,
                Color::new(0, 0, 0),
                self.text_scale,
            );
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for TraceAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        let mut history = self.history.borrow_mut();
        for detection in detections.iter() {
            let Some(tracker_id) = detection.tracker_id else {
                continue;
            };
            let point = detection.anchor_point(self.position);
            let trail = history.entry(tracker_id).or_default();
            trail.push_back((point.x, point.y));
            while trail.len() > self.trace_length.max(1) {
                trail.pop_front();
            }
        }

        for detection in detections.iter() {
            let Some(tracker_id) = detection.tracker_id else {
                continue;
            };
            let color = self.palette.by_class_id(detection.class_id);
            if let Some(trail) = history.get(&tracker_id) {
                let points: Vec<Point> = trail.iter().map(|&(x, y)| Point::new(x, y)).collect();
                for pair in points.windows(2) {
                    draw_line(image, pair[0], pair[1], color, self.thickness);
                }
            }
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for HeatMapAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        let (width, height) = image.dimensions();
        {
            let mut buffer = self.buffer.borrow_mut();
            let needs_resize = match buffer.as_ref() {
                Some(grid) => grid.dim() != (height as usize, width as usize),
                None => true,
            };
            if needs_resize {
                *buffer = Some(Array2::<f32>::zeros((height as usize, width as usize)));
            }
            let grid = buffer.as_mut().expect("buffer initialized above");
            for detection in detections.iter() {
                let (cx, cy) = detection.centroid();
                splat_heat(grid, cx, cy, self.radius as f32, self.intensity);
            }
        }

        let buffer = self.buffer.borrow();
        if let Some(grid) = buffer.as_ref() {
            let max = grid.iter().cloned().fold(0.0f32, f32::max);
            if max > 0.0 {
                for y in 0..height {
                    for x in 0..width {
                        let v = grid[[y as usize, x as usize]];
                        if v <= 0.0 {
                            continue;
                        }
                        let t = (v / max).clamp(0.0, 1.0);
                        let (r, g, b) = heat_color(t);
                        alpha_blend_pixel(
                            image,
                            x as i64,
                            y as i64,
                            Color::new(r, g, b),
                            self.opacity * t,
                            width,
                            height,
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for BlurAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        let snapshot = image.clone();
        for detection in detections.iter() {
            box_blur_region(image, &snapshot, detection.bbox, self.kernel_size);
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for PixelateAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        let snapshot = image.clone();
        for detection in detections.iter() {
            pixelate_region(image, &snapshot, detection.bbox, self.pixel_size);
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for PolygonZoneAnnotator {
    type Subject = PolygonZone;

    fn annotate(&self, image: &mut RgbaImage, zone: &PolygonZone) -> crate::Result<()> {
        if zone.polygon.is_empty() {
            return Ok(());
        }
        let n = zone.polygon.len();
        for i in 0..n {
            draw_line(
                image,
                zone.polygon[i],
                zone.polygon[(i + 1) % n],
                self.color,
                self.thickness,
            );
        }

        let label = format!("IN:{} OUT:{}", zone.in_count(), zone.out_count());
        let rect = polygon_to_rect(&zone.polygon);
        let scale = self.text_scale.max(1.0).round() as u32;
        let text_y = (rect.y1 - 12.0).max(0.0);
        draw_text(image, &label, rect.x1, text_y, self.color, scale);
        Ok(())
    }
}

impl Annotator<RgbaImage> for VertexAnnotator {
    type Subject = KeyPoints;

    fn annotate(&self, image: &mut RgbaImage, keypoints: &KeyPoints) -> crate::Result<()> {
        let (width, height) = image.dimensions();
        for set in keypoints.iter() {
            for point in set.points.iter().flatten() {
                draw_filled_circle(
                    image,
                    point.x,
                    point.y,
                    self.radius as f32,
                    self.color,
                    width,
                    height,
                );
            }
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for EdgeAnnotator {
    type Subject = KeyPoints;

    fn annotate(&self, image: &mut RgbaImage, keypoints: &KeyPoints) -> crate::Result<()> {
        for set in keypoints.iter() {
            for &(a, b) in &self.edges {
                if let (Some(pa), Some(pb)) = (set.get(a), set.get(b)) {
                    draw_line(
                        image,
                        Point::new(pa.x, pa.y),
                        Point::new(pb.x, pb.y),
                        self.color,
                        self.thickness,
                    );
                }
            }
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for VertexLabelAnnotator {
    type Subject = KeyPoints;

    fn annotate(&self, image: &mut RgbaImage, keypoints: &KeyPoints) -> crate::Result<()> {
        for set in keypoints.iter() {
            for (i, point) in set.points.iter().enumerate() {
                if let Some(p) = point {
                    let label = i.to_string();
                    draw_text(image, &label, p.x, p.y, self.color, self.text_scale);
                }
            }
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for HaloAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        let (width, height) = image.dimensions();
        let layers = 6u32;
        for detection in detections.iter() {
            let color = self.palette.by_class_id(detection.class_id);
            let [x1, y1, x2, y2] = detection.bbox;
            // Faintest/widest layer first, so the strongest (edge) layer
            // ends up drawn last, on top.
            for layer in (0..layers).rev() {
                let t = layer as f32 / (layers - 1) as f32;
                let pad = t * self.kernel_size as f32;
                let alpha = self.opacity * (1.0 - t);
                if alpha <= 0.0 {
                    continue;
                }
                let polygon = [
                    (x1 - pad, y1 - pad),
                    (x2 + pad, y1 - pad),
                    (x2 + pad, y2 + pad),
                    (x1 - pad, y2 + pad),
                ];
                draw_filled_polygon(image, &polygon, color, alpha, width, height);
            }
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for PercentageBarAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let color = self.palette.by_class_id(detection.class_id);
            let anchor = detection.anchor_point(self.position);
            let half_w = self.bar_width as f32 / 2.0;
            let bg = [
                anchor.x - half_w,
                anchor.y - self.bar_height as f32,
                anchor.x + half_w,
                anchor.y,
            ];
            draw_filled_rect(image, bg, self.background_color);

            let fraction = detection.confidence.clamp(0.0, 1.0);
            let fg = [
                bg[0],
                bg[1],
                bg[0] + self.bar_width as f32 * fraction,
                bg[3],
            ];
            draw_filled_rect(image, fg, color);
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for IconAnnotator<RgbaImage> {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        let (icon_w, icon_h) = self.icon.dimensions();
        for detection in detections.iter() {
            let anchor = detection.anchor_point(self.position);
            let x = (anchor.x - icon_w as f32 / 2.0).round() as i64;
            let y = (anchor.y - icon_h as f32 / 2.0).round() as i64;
            overlay_image(image, &self.icon, x, y);
        }
        Ok(())
    }
}

impl Annotator<RgbaImage> for BackgroundOverlayAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut RgbaImage, detections: &Detections) -> crate::Result<()> {
        let snapshot = image.clone();
        let (width, height) = image.dimensions();
        for y in 0..height {
            for x in 0..width {
                alpha_blend_pixel(
                    image,
                    x as i64,
                    y as i64,
                    self.color,
                    self.opacity,
                    width,
                    height,
                );
            }
        }

        for detection in detections.iter() {
            let [x1, y1, x2, y2] = detection.bbox;
            let x1 = x1.max(0.0) as u32;
            let y1 = y1.max(0.0) as u32;
            let x2 = (x2.min(width as f32 - 1.0)) as u32;
            let y2 = (y2.min(height as f32 - 1.0)) as u32;
            if x2 < x1 || y2 < y1 {
                continue;
            }
            for y in y1..=y2 {
                for x in x1..=x2 {
                    image.put_pixel(x, y, *snapshot.get_pixel(x, y));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotators::ColorPalette;
    use crate::core::Detection;

    fn has_color(image: &RgbaImage, color: Color) -> bool {
        let rgba: Rgba<u8> = color.into();
        image.pixels().any(|p| *p == rgba)
    }

    #[test]
    fn box_annotator_draws_within_bounds() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = BoxAnnotator::default();
        let detection = Detection::new([5.0, 5.0, 20.0, 20.0], 0.9, 0);
        let detections = Detections::new(vec![detection]);

        annotator.annotate(&mut image, &detections).unwrap();

        let color = annotator.palette.by_class_id(0);
        assert!(has_color(&image, color));
    }

    #[test]
    fn box_annotator_does_not_panic_on_out_of_bounds_bbox() {
        let mut image = RgbaImage::new(10, 10);
        let annotator = BoxAnnotator::default();
        let detection = Detection::new([-50.0, -50.0, 500.0, 500.0], 0.9, 0);
        let detections = Detections::new(vec![detection]);

        annotator.annotate(&mut image, &detections).unwrap();
    }

    #[test]
    fn line_zone_annotator_draws_the_line() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = LineZoneAnnotator::default();
        let zone = LineZone::new(Point::new(5.0, 25.0), Point::new(45.0, 25.0));

        annotator.annotate(&mut image, &zone).unwrap();

        assert!(has_color(&image, annotator.color));
    }

    fn masked_detection() -> Detection {
        let mut d = Detection::new([5.0, 5.0, 20.0, 20.0], 0.9, 0);
        d.mask = Some(vec![[5.0, 5.0], [20.0, 5.0], [20.0, 20.0], [5.0, 20.0]]);
        d
    }

    #[test]
    fn mask_annotator_fills_the_polygon() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = MaskAnnotator::new(ColorPalette::default(), 1.0);
        let detections = Detections::new(vec![masked_detection()]);
        annotator.annotate(&mut image, &detections).unwrap();
        assert!(has_color(&image, annotator.palette.by_class_id(0)));
    }

    #[test]
    fn mask_annotator_skips_detections_without_a_mask() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = MaskAnnotator::default();
        let detections = Detections::new(vec![Detection::new([5.0, 5.0, 20.0, 20.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
        assert!(!has_color(&image, annotator.palette.by_class_id(0)));
    }

    #[test]
    fn polygon_annotator_draws_the_outline() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = PolygonAnnotator::default();
        let detections = Detections::new(vec![masked_detection()]);
        annotator.annotate(&mut image, &detections).unwrap();
        assert!(has_color(&image, annotator.palette.by_class_id(0)));
    }

    #[test]
    fn circle_annotator_draws_within_bounds() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = CircleAnnotator::default();
        let detections = Detections::new(vec![Detection::new([5.0, 5.0, 20.0, 20.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
        assert!(has_color(&image, annotator.palette.by_class_id(0)));
    }

    #[test]
    fn dot_annotator_draws_at_the_anchor_point() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = DotAnnotator::default();
        let detections = Detections::new(vec![Detection::new([5.0, 5.0, 20.0, 20.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
        assert!(has_color(&image, annotator.palette.by_class_id(0)));
    }

    #[test]
    fn ellipse_annotator_draws_within_bounds() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = EllipseAnnotator::default();
        let detections = Detections::new(vec![Detection::new([5.0, 5.0, 20.0, 20.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
        assert!(has_color(&image, annotator.palette.by_class_id(0)));
    }

    #[test]
    fn label_annotator_draws_a_background_box() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = LabelAnnotator::default();
        let detections = Detections::new(vec![Detection::new([10.0, 10.0, 30.0, 30.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
        assert!(has_color(&image, annotator.palette.by_class_id(0)));
    }

    #[test]
    fn trace_annotator_accumulates_history_across_calls() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = TraceAnnotator::default();

        let mut d1 = Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0);
        d1.tracker_id = Some(1);
        annotator
            .annotate(&mut image, &Detections::new(vec![d1]))
            .unwrap();
        assert_eq!(annotator.history.borrow().get(&1).unwrap().len(), 1);

        let mut d2 = Detection::new([20.0, 0.0, 30.0, 10.0], 0.9, 0);
        d2.tracker_id = Some(1);
        annotator
            .annotate(&mut image, &Detections::new(vec![d2]))
            .unwrap();
        assert_eq!(annotator.history.borrow().get(&1).unwrap().len(), 2);
        // A line should now be drawn connecting the two recorded points.
        assert!(has_color(&image, annotator.palette.by_class_id(0)));
    }

    #[test]
    fn trace_annotator_ignores_detections_without_tracker_id() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = TraceAnnotator::default();
        let detections = Detections::new(vec![Detection::new([0.0, 0.0, 10.0, 10.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
        assert!(annotator.history.borrow().is_empty());
    }

    #[test]
    fn heatmap_annotator_accumulates_and_colors_the_image() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = HeatMapAnnotator::new(10, 1.0, 1.0);
        let detections = Detections::new(vec![Detection::new([20.0, 20.0, 30.0, 30.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
        assert!(annotator
            .buffer
            .borrow()
            .as_ref()
            .unwrap()
            .iter()
            .any(|&v| v > 0.0));
        // A single detection splats uniform (saturated) heat, which maps to
        // the "hot" (red) end of the ramp at full opacity.
        assert_eq!(*image.get_pixel(25, 25), Rgba([255, 0, 0, 255]));
    }

    #[test]
    fn blur_annotator_does_not_panic_on_out_of_bounds_bbox() {
        let mut image = RgbaImage::new(10, 10);
        let annotator = BlurAnnotator::default();
        let detections = Detections::new(vec![Detection::new([-5.0, -5.0, 50.0, 50.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
    }

    #[test]
    fn pixelate_annotator_flattens_a_region_to_block_averages() {
        let mut image = RgbaImage::from_fn(20, 20, |x, _| {
            if x < 10 {
                Rgba([255, 0, 0, 255])
            } else {
                Rgba([0, 0, 255, 255])
            }
        });
        let annotator = PixelateAnnotator::new(20);
        let detections = Detections::new(vec![Detection::new([0.0, 0.0, 20.0, 20.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
        // A single 20px block spanning the red/blue split should average to
        // a uniform color across the whole region.
        let first = *image.get_pixel(0, 0);
        assert!(image.pixels().all(|p| *p == first));
    }

    #[test]
    fn polygon_zone_annotator_draws_outline_and_label() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = PolygonZoneAnnotator::default();
        let zone = PolygonZone::new(vec![
            Point::new(5.0, 5.0),
            Point::new(30.0, 5.0),
            Point::new(30.0, 30.0),
            Point::new(5.0, 30.0),
        ]);
        annotator.annotate(&mut image, &zone).unwrap();
        assert!(has_color(&image, annotator.color));
    }

    #[test]
    fn polygon_zone_annotator_handles_empty_polygon() {
        let mut image = RgbaImage::new(10, 10);
        let annotator = PolygonZoneAnnotator::default();
        let zone = PolygonZone::new(vec![]);
        annotator.annotate(&mut image, &zone).unwrap();
    }

    fn sample_keypoints() -> KeyPoints {
        use crate::core::{Keypoint, KeypointSet};
        KeyPoints::new(vec![KeypointSet::new(
            vec![
                Some(Keypoint::new(5.0, 5.0, 0.9)),
                Some(Keypoint::new(15.0, 15.0, 0.9)),
                None,
            ],
            0,
        )])
    }

    #[test]
    fn vertex_annotator_draws_each_detected_joint() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = VertexAnnotator::default();
        annotator.annotate(&mut image, &sample_keypoints()).unwrap();
        assert!(has_color(&image, annotator.color));
    }

    #[test]
    fn edge_annotator_only_draws_when_both_joints_present() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = EdgeAnnotator::new(Color::new(1, 2, 3), 1, vec![(0, 1), (0, 2)]);
        annotator.annotate(&mut image, &sample_keypoints()).unwrap();
        // (0, 1) is drawable, (0, 2) is not since joint 2 is missing; the
        // call should still complete without panicking either way.
        assert!(has_color(&image, annotator.color));
    }

    #[test]
    fn vertex_label_annotator_does_not_panic() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = VertexLabelAnnotator::default();
        annotator.annotate(&mut image, &sample_keypoints()).unwrap();
    }

    #[test]
    fn round_box_annotator_draws_within_bounds() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = RoundBoxAnnotator::default();
        let detections = Detections::new(vec![Detection::new([5.0, 5.0, 40.0, 40.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
        assert!(has_color(&image, annotator.palette.by_class_id(0)));
    }

    #[test]
    fn round_box_annotator_does_not_panic_on_a_tiny_box() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = RoundBoxAnnotator::default();
        let detections = Detections::new(vec![Detection::new([5.0, 5.0, 8.0, 8.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
    }

    #[test]
    fn box_corner_annotator_draws_at_the_corners_not_the_full_outline() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = BoxCornerAnnotator::default();
        let detections = Detections::new(vec![Detection::new([5.0, 5.0, 40.0, 40.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
        let color = annotator.palette.by_class_id(0);
        assert!(has_color(&image, color));
        // The midpoint of the top edge is far from every corner mark and
        // should be untouched, unlike a full BoxAnnotator outline.
        assert_ne!(
            *image.get_pixel(22, 5),
            Rgba([color.r, color.g, color.b, 255])
        );
    }

    #[test]
    fn triangle_annotator_draws_a_filled_marker() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = TriangleAnnotator::default();
        let detections = Detections::new(vec![Detection::new([10.0, 20.0, 30.0, 40.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
        assert!(has_color(&image, annotator.palette.by_class_id(0)));
    }

    #[test]
    fn halo_annotator_glows_around_the_box() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = HaloAnnotator::default();
        let detections = Detections::new(vec![Detection::new([10.0, 10.0, 30.0, 30.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
        // Just outside the box, the glow should have tinted the
        // background away from pure black.
        let p = image.get_pixel(9, 20);
        assert_ne!(*p, Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn percentage_bar_annotator_fill_width_matches_confidence() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = PercentageBarAnnotator::default();
        let detections = Detections::new(vec![Detection::new([10.0, 20.0, 30.0, 40.0], 0.5, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
        assert!(has_color(&image, annotator.background_color));
        assert!(has_color(&image, annotator.palette.by_class_id(0)));
    }

    #[test]
    fn percentage_bar_annotator_clamps_out_of_range_confidence() {
        let mut image = RgbaImage::new(50, 50);
        let annotator = PercentageBarAnnotator::default();
        let detections = Detections::new(vec![Detection::new([10.0, 20.0, 30.0, 40.0], 1.5, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
    }

    #[test]
    fn icon_annotator_overlays_the_icon_centered_on_the_anchor() {
        let mut image = RgbaImage::new(50, 50);
        let icon = RgbaImage::from_pixel(6, 6, Rgba([200, 10, 10, 255]));
        let annotator = IconAnnotator::new(icon, Position::Center);
        let detections = Detections::new(vec![Detection::new([10.0, 10.0, 30.0, 30.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();
        assert_eq!(*image.get_pixel(20, 20), Rgba([200, 10, 10, 255]));
    }

    #[test]
    fn background_overlay_annotator_dims_outside_but_not_inside_boxes() {
        let mut image = RgbaImage::from_pixel(50, 50, Rgba([255, 255, 255, 255]));
        let annotator = BackgroundOverlayAnnotator::new(Color::new(0, 0, 0), 1.0);
        let detections = Detections::new(vec![Detection::new([10.0, 10.0, 30.0, 30.0], 0.9, 0)]);
        annotator.annotate(&mut image, &detections).unwrap();

        // Fully outside opacity=1.0 black overlay: background goes black.
        assert_eq!(*image.get_pixel(0, 0), Rgba([0, 0, 0, 255]));
        // Inside the box: original pixel restored untouched.
        assert_eq!(*image.get_pixel(20, 20), Rgba([255, 255, 255, 255]));
    }
}
