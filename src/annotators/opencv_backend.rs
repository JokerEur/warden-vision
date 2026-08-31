//! High-performance annotation backend built on OpenCV via the `opencv`
//! crate: draws directly onto an `opencv::core::Mat`, assumed to be in
//! OpenCV's conventional BGR channel order.
//!
//! NOTE: annotators beyond [`BoxAnnotator`] / [`LineZoneAnnotator`] here
//! are not build-verified against a real OpenCV install (CI's
//! `annotate-opencv` job is best-effort/allowed to fail). Double-check
//! against the `opencv` crate version you're pinned to if something
//! doesn't line up.

use opencv::core::{Mat, Point, Rect, Scalar, Size, Vec3b, Vector};
use opencv::imgproc::{self, FONT_HERSHEY_SIMPLEX, LINE_8};
use opencv::prelude::*;

use crate::annotators::heatmap::heat_color;
use crate::annotators::{
    Annotator, BackgroundOverlayAnnotator, BlurAnnotator, BoxAnnotator, BoxCornerAnnotator,
    CircleAnnotator, Color, DotAnnotator, EdgeAnnotator, EllipseAnnotator, HaloAnnotator,
    HeatMapAnnotator, IconAnnotator, LabelAnnotator, LineZoneAnnotator, MaskAnnotator,
    PercentageBarAnnotator, PixelateAnnotator, PolygonAnnotator, PolygonZoneAnnotator,
    RoundBoxAnnotator, TraceAnnotator, TriangleAnnotator, VertexAnnotator, VertexLabelAnnotator,
};
use crate::core::{Detections, KeyPoints};
use crate::error::Error;
use crate::geometry::{polygon_to_rect, LineZone, PolygonZone, Position, Zone};

/// Converts an RGB [`Color`] to an OpenCV BGR [`Scalar`].
fn to_scalar(color: Color) -> Scalar {
    (color.b as f64, color.g as f64, color.r as f64).into()
}

fn to_backend_error(err: opencv::Error) -> Error {
    Error::Backend(err.to_string())
}

impl Annotator<Mat> for BoxAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let color = to_scalar(self.palette.by_class_id(detection.class_id));
            let [x1, y1, x2, y2] = detection.bbox;
            let rect = Rect::new(
                x1.round() as i32,
                y1.round() as i32,
                (x2 - x1).round().max(0.0) as i32,
                (y2 - y1).round().max(0.0) as i32,
            );
            imgproc::rectangle(image, rect, color, self.thickness as i32, LINE_8, 0)
                .map_err(to_backend_error)?;

            let label = match detection.tracker_id {
                Some(id) => format!("#{} {:.0}%", id, detection.confidence * 100.0),
                None => format!("{:.0}%", detection.confidence * 100.0),
            };
            let origin = Point::new(x1.round() as i32, (y1 - 6.0).max(0.0).round() as i32);
            imgproc::put_text(
                image,
                &label,
                origin,
                FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                LINE_8,
                false,
            )
            .map_err(to_backend_error)?;
        }
        Ok(())
    }
}

impl Annotator<Mat> for LineZoneAnnotator {
    type Subject = LineZone;

    fn annotate(&self, image: &mut Mat, zone: &LineZone) -> crate::Result<()> {
        let color = to_scalar(self.color);
        let start = Point::new(zone.start.x.round() as i32, zone.start.y.round() as i32);
        let end = Point::new(zone.end.x.round() as i32, zone.end.y.round() as i32);

        imgproc::line(image, start, end, color, self.thickness as i32, LINE_8, 0)
            .map_err(to_backend_error)?;

        let label = format!("IN:{} OUT:{}", zone.in_count(), zone.out_count());
        let origin = Point::new(start.x.min(end.x), (start.y.min(end.y) - 10).max(0));
        imgproc::put_text(
            image,
            &label,
            origin,
            FONT_HERSHEY_SIMPLEX,
            self.text_scale as f64,
            color,
            1,
            LINE_8,
            false,
        )
        .map_err(to_backend_error)?;
        Ok(())
    }
}

/// Clips `bbox` to `[0, cols) x [0, rows)`, returning `None` if the
/// resulting rectangle would be empty.
fn clamp_rect(bbox: [f32; 4], cols: i32, rows: i32) -> Option<Rect> {
    let x1 = bbox[0].round().clamp(0.0, cols as f32) as i32;
    let y1 = bbox[1].round().clamp(0.0, rows as f32) as i32;
    let x2 = bbox[2].round().clamp(0.0, cols as f32) as i32;
    let y2 = bbox[3].round().clamp(0.0, rows as f32) as i32;
    let width = x2 - x1;
    let height = y2 - y1;
    if width <= 0 || height <= 0 {
        None
    } else {
        Some(Rect::new(x1, y1, width, height))
    }
}

impl Annotator<Mat> for MaskAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let Some(mask) = &detection.mask else {
                continue;
            };
            if mask.len() < 3 {
                continue;
            }
            let color = to_scalar(self.palette.by_class_id(detection.class_id));
            let points: Vector<Point> = mask
                .iter()
                .map(|&[x, y]| Point::new(x.round() as i32, y.round() as i32))
                .collect();
            let mut contours = Vector::<Vector<Point>>::new();
            contours.push(points);

            let mut overlay = image.clone();
            imgproc::fill_poly(&mut overlay, &contours, color, LINE_8, 0, Point::new(0, 0))
                .map_err(to_backend_error)?;
            opencv::core::add_weighted(
                &overlay,
                self.opacity as f64,
                image,
                1.0 - self.opacity as f64,
                0.0,
                image,
                -1,
            )
            .map_err(to_backend_error)?;
        }
        Ok(())
    }
}

impl Annotator<Mat> for PolygonAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let Some(mask) = &detection.mask else {
                continue;
            };
            if mask.len() < 2 {
                continue;
            }
            let color = to_scalar(self.palette.by_class_id(detection.class_id));
            let points: Vector<Point> = mask
                .iter()
                .map(|&[x, y]| Point::new(x.round() as i32, y.round() as i32))
                .collect();
            let mut contours = Vector::<Vector<Point>>::new();
            contours.push(points);
            imgproc::polylines(
                image,
                &contours,
                true,
                color,
                self.thickness as i32,
                LINE_8,
                0,
            )
            .map_err(to_backend_error)?;
        }
        Ok(())
    }
}

impl Annotator<Mat> for CircleAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let color = to_scalar(self.palette.by_class_id(detection.class_id));
            let center = detection.anchor_point(self.position);
            let radius = detection.width().min(detection.height()) / 2.0;
            imgproc::circle(
                image,
                Point::new(center.x.round() as i32, center.y.round() as i32),
                radius.round() as i32,
                color,
                self.thickness as i32,
                LINE_8,
                0,
            )
            .map_err(to_backend_error)?;
        }
        Ok(())
    }
}

impl Annotator<Mat> for DotAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let color = to_scalar(self.palette.by_class_id(detection.class_id));
            let center = detection.anchor_point(self.position);
            imgproc::circle(
                image,
                Point::new(center.x.round() as i32, center.y.round() as i32),
                self.radius as i32,
                color,
                -1,
                LINE_8,
                0,
            )
            .map_err(to_backend_error)?;
        }
        Ok(())
    }
}

impl Annotator<Mat> for EllipseAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let color = to_scalar(self.palette.by_class_id(detection.class_id));
            let center = detection.anchor_point(Position::BottomCenter);
            let a = detection.width() / 2.0;
            let b = (detection.width() * self.height_ratio).max(1.0) / 2.0;
            imgproc::ellipse(
                image,
                Point::new(center.x.round() as i32, center.y.round() as i32),
                Size::new(a.round() as i32, b.round() as i32),
                0.0,
                0.0,
                360.0,
                color,
                self.thickness as i32,
                LINE_8,
                0,
            )
            .map_err(to_backend_error)?;
        }
        Ok(())
    }
}

impl Annotator<Mat> for RoundBoxAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let color = to_scalar(self.palette.by_class_id(detection.class_id));
            let [x1, y1, x2, y2] = detection.bbox;
            let r = (self.corner_radius as f32)
                .min(detection.width().min(detection.height()) / 2.0)
                .max(0.0);
            let thickness = self.thickness as i32;

            let line = |image: &mut Mat, x1: f32, y1: f32, x2: f32, y2: f32| {
                imgproc::line(
                    image,
                    Point::new(x1.round() as i32, y1.round() as i32),
                    Point::new(x2.round() as i32, y2.round() as i32),
                    color,
                    thickness,
                    LINE_8,
                    0,
                )
            };
            line(image, x1 + r, y1, x2 - r, y1).map_err(to_backend_error)?;
            line(image, x1 + r, y2, x2 - r, y2).map_err(to_backend_error)?;
            line(image, x1, y1 + r, x1, y2 - r).map_err(to_backend_error)?;
            line(image, x2, y1 + r, x2, y2 - r).map_err(to_backend_error)?;

            let arc = |image: &mut Mat, cx: f32, cy: f32, start: f64, end: f64| {
                imgproc::ellipse(
                    image,
                    Point::new(cx.round() as i32, cy.round() as i32),
                    Size::new(r.round() as i32, r.round() as i32),
                    0.0,
                    start,
                    end,
                    color,
                    thickness,
                    LINE_8,
                    0,
                )
            };
            arc(image, x1 + r, y1 + r, 180.0, 270.0).map_err(to_backend_error)?;
            arc(image, x2 - r, y1 + r, 270.0, 360.0).map_err(to_backend_error)?;
            arc(image, x2 - r, y2 - r, 0.0, 90.0).map_err(to_backend_error)?;
            arc(image, x1 + r, y2 - r, 90.0, 180.0).map_err(to_backend_error)?;
        }
        Ok(())
    }
}

impl Annotator<Mat> for BoxCornerAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let color = to_scalar(self.palette.by_class_id(detection.class_id));
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
                imgproc::line(
                    image,
                    Point::new(cx.round() as i32, cy.round() as i32),
                    Point::new((cx + dx * len).round() as i32, cy.round() as i32),
                    color,
                    self.thickness as i32,
                    LINE_8,
                    0,
                )
                .map_err(to_backend_error)?;
                imgproc::line(
                    image,
                    Point::new(cx.round() as i32, cy.round() as i32),
                    Point::new(cx.round() as i32, (cy + dy * len).round() as i32),
                    color,
                    self.thickness as i32,
                    LINE_8,
                    0,
                )
                .map_err(to_backend_error)?;
            }
        }
        Ok(())
    }
}

impl Annotator<Mat> for TriangleAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let color = to_scalar(self.palette.by_class_id(detection.class_id));
            let tip = detection.anchor_point(self.position);
            let base_half = self.base as f32 / 2.0;
            let points: Vector<Point> = [
                (tip.x, tip.y),
                (tip.x - base_half, tip.y - self.height as f32),
                (tip.x + base_half, tip.y - self.height as f32),
            ]
            .iter()
            .map(|&(x, y)| Point::new(x.round() as i32, y.round() as i32))
            .collect();
            let mut contours = Vector::<Vector<Point>>::new();
            contours.push(points);
            imgproc::fill_poly(image, &contours, color, LINE_8, 0, Point::new(0, 0))
                .map_err(to_backend_error)?;
        }
        Ok(())
    }
}

impl Annotator<Mat> for LabelAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let color = to_scalar(self.palette.by_class_id(detection.class_id));
            let label = match detection.tracker_id {
                Some(id) => format!("#{} {:.0}%", id, detection.confidence * 100.0),
                None => format!("{:.0}%", detection.confidence * 100.0),
            };
            let anchor = detection.anchor_point(self.position);
            let font_scale = 0.5 * self.text_scale as f64;
            let mut baseline = 0;
            let text_size =
                imgproc::get_text_size(&label, FONT_HERSHEY_SIMPLEX, font_scale, 1, &mut baseline)
                    .map_err(to_backend_error)?;

            let bg = Rect::new(
                (anchor.x - self.padding).max(0.0).round() as i32,
                (anchor.y - text_size.height as f32 - self.padding)
                    .max(0.0)
                    .round() as i32,
                (text_size.width as f32 + self.padding * 2.0).round() as i32,
                (text_size.height as f32 + self.padding * 2.0).round() as i32,
            );
            imgproc::rectangle(image, bg, color, -1, LINE_8, 0).map_err(to_backend_error)?;

            let origin = Point::new(
                bg.x + self.padding.round() as i32,
                bg.y + bg.height - self.padding.round() as i32,
            );
            imgproc::put_text(
                image,
                &label,
                origin,
                FONT_HERSHEY_SIMPLEX,
                font_scale,
                Scalar::new(0.0, 0.0, 0.0, 0.0),
                1,
                LINE_8,
                false,
            )
            .map_err(to_backend_error)?;
        }
        Ok(())
    }
}

impl Annotator<Mat> for TraceAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
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
            let color = to_scalar(self.palette.by_class_id(detection.class_id));
            if let Some(trail) = history.get(&tracker_id) {
                let points: Vec<(f32, f32)> = trail.iter().copied().collect();
                for pair in points.windows(2) {
                    imgproc::line(
                        image,
                        Point::new(pair[0].0.round() as i32, pair[0].1.round() as i32),
                        Point::new(pair[1].0.round() as i32, pair[1].1.round() as i32),
                        color,
                        self.thickness as i32,
                        LINE_8,
                        0,
                    )
                    .map_err(to_backend_error)?;
                }
            }
        }
        Ok(())
    }
}

impl Annotator<Mat> for PolygonZoneAnnotator {
    type Subject = PolygonZone;

    fn annotate(&self, image: &mut Mat, zone: &PolygonZone) -> crate::Result<()> {
        if zone.polygon.is_empty() {
            return Ok(());
        }
        let color = to_scalar(self.color);
        let points: Vector<Point> = zone
            .polygon
            .iter()
            .map(|p| Point::new(p.x.round() as i32, p.y.round() as i32))
            .collect();
        let mut contours = Vector::<Vector<Point>>::new();
        contours.push(points);
        imgproc::polylines(
            image,
            &contours,
            true,
            color,
            self.thickness as i32,
            LINE_8,
            0,
        )
        .map_err(to_backend_error)?;

        let label = format!("IN:{} OUT:{}", zone.in_count(), zone.out_count());
        let rect = polygon_to_rect(&zone.polygon);
        let origin = Point::new(
            rect.x1.round() as i32,
            (rect.y1 - 12.0).max(0.0).round() as i32,
        );
        imgproc::put_text(
            image,
            &label,
            origin,
            FONT_HERSHEY_SIMPLEX,
            self.text_scale as f64,
            color,
            1,
            LINE_8,
            false,
        )
        .map_err(to_backend_error)?;
        Ok(())
    }
}

impl Annotator<Mat> for VertexAnnotator {
    type Subject = KeyPoints;

    fn annotate(&self, image: &mut Mat, keypoints: &KeyPoints) -> crate::Result<()> {
        let color = to_scalar(self.color);
        for set in keypoints.iter() {
            for point in set.points.iter().flatten() {
                imgproc::circle(
                    image,
                    Point::new(point.x.round() as i32, point.y.round() as i32),
                    self.radius as i32,
                    color,
                    -1,
                    LINE_8,
                    0,
                )
                .map_err(to_backend_error)?;
            }
        }
        Ok(())
    }
}

impl Annotator<Mat> for EdgeAnnotator {
    type Subject = KeyPoints;

    fn annotate(&self, image: &mut Mat, keypoints: &KeyPoints) -> crate::Result<()> {
        let color = to_scalar(self.color);
        for set in keypoints.iter() {
            for &(a, b) in &self.edges {
                if let (Some(pa), Some(pb)) = (set.get(a), set.get(b)) {
                    imgproc::line(
                        image,
                        Point::new(pa.x.round() as i32, pa.y.round() as i32),
                        Point::new(pb.x.round() as i32, pb.y.round() as i32),
                        color,
                        self.thickness as i32,
                        LINE_8,
                        0,
                    )
                    .map_err(to_backend_error)?;
                }
            }
        }
        Ok(())
    }
}

impl Annotator<Mat> for VertexLabelAnnotator {
    type Subject = KeyPoints;

    fn annotate(&self, image: &mut Mat, keypoints: &KeyPoints) -> crate::Result<()> {
        let color = to_scalar(self.color);
        for set in keypoints.iter() {
            for (i, point) in set.points.iter().enumerate() {
                if let Some(p) = point {
                    let label = i.to_string();
                    imgproc::put_text(
                        image,
                        &label,
                        Point::new(p.x.round() as i32, p.y.round() as i32),
                        FONT_HERSHEY_SIMPLEX,
                        0.5 * self.text_scale as f64,
                        color,
                        1,
                        LINE_8,
                        false,
                    )
                    .map_err(to_backend_error)?;
                }
            }
        }
        Ok(())
    }
}

/// Adds `intensity` to every cell of `grid` within `radius` of `(cx, cy)`.
fn splat_heat(grid: &mut ndarray::Array2<f32>, cx: f32, cy: f32, radius: f32, intensity: f32) {
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

impl Annotator<Mat> for HeatMapAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        let cols = image.cols();
        let rows = image.rows();
        {
            let mut buffer = self.buffer.borrow_mut();
            let needs_resize = match buffer.as_ref() {
                Some(grid) => grid.dim() != (rows as usize, cols as usize),
                None => true,
            };
            if needs_resize {
                *buffer = Some(ndarray::Array2::<f32>::zeros((
                    rows as usize,
                    cols as usize,
                )));
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
                for y in 0..rows {
                    for x in 0..cols {
                        let v = grid[[y as usize, x as usize]];
                        if v <= 0.0 {
                            continue;
                        }
                        let t = (v / max).clamp(0.0, 1.0);
                        let (r, g, b) = heat_color(t);
                        let alpha = (self.opacity * t) as f64;
                        let pixel: &mut Vec3b = image.at_2d_mut(y, x).map_err(to_backend_error)?;
                        pixel[0] =
                            (pixel[0] as f64 * (1.0 - alpha) + b as f64 * alpha).round() as u8;
                        pixel[1] =
                            (pixel[1] as f64 * (1.0 - alpha) + g as f64 * alpha).round() as u8;
                        pixel[2] =
                            (pixel[2] as f64 * (1.0 - alpha) + r as f64 * alpha).round() as u8;
                    }
                }
            }
        }
        Ok(())
    }
}

impl Annotator<Mat> for BlurAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let Some(rect) = clamp_rect(detection.bbox, image.cols(), image.rows()) else {
                continue;
            };
            let region = image.roi(rect).map_err(to_backend_error)?;
            let mut blurred = Mat::default();
            let k = self.kernel_size.max(1) as i32;
            imgproc::blur(
                &region,
                &mut blurred,
                Size::new(k, k),
                Point::new(-1, -1),
                opencv::core::BORDER_DEFAULT,
            )
            .map_err(to_backend_error)?;
            let mut dst = image.roi_mut(rect).map_err(to_backend_error)?;
            blurred.copy_to(&mut dst).map_err(to_backend_error)?;
        }
        Ok(())
    }
}

impl Annotator<Mat> for PixelateAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let Some(rect) = clamp_rect(detection.bbox, image.cols(), image.rows()) else {
                continue;
            };
            let region = image.roi(rect).map_err(to_backend_error)?;
            let block = self.pixel_size.max(1) as i32;
            let small_size = Size::new((rect.width / block).max(1), (rect.height / block).max(1));
            let mut small = Mat::default();
            imgproc::resize(
                &region,
                &mut small,
                small_size,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )
            .map_err(to_backend_error)?;
            let mut pixelated = Mat::default();
            imgproc::resize(
                &small,
                &mut pixelated,
                Size::new(rect.width, rect.height),
                0.0,
                0.0,
                imgproc::INTER_NEAREST,
            )
            .map_err(to_backend_error)?;
            let mut dst = image.roi_mut(rect).map_err(to_backend_error)?;
            pixelated.copy_to(&mut dst).map_err(to_backend_error)?;
        }
        Ok(())
    }
}

impl Annotator<Mat> for HaloAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        let layers = 6i32;
        for detection in detections.iter() {
            let color = to_scalar(self.palette.by_class_id(detection.class_id));
            let [x1, y1, x2, y2] = detection.bbox;
            for layer in (0..layers).rev() {
                let t = layer as f32 / (layers - 1) as f32;
                let pad = t * self.kernel_size as f32;
                let alpha = self.opacity * (1.0 - t);
                if alpha <= 0.0 {
                    continue;
                }
                let rect = Rect::new(
                    (x1 - pad).round() as i32,
                    (y1 - pad).round() as i32,
                    ((x2 - x1) + 2.0 * pad).round().max(0.0) as i32,
                    ((y2 - y1) + 2.0 * pad).round().max(0.0) as i32,
                );
                let mut overlay = image.clone();
                imgproc::rectangle(&mut overlay, rect, color, -1, LINE_8, 0)
                    .map_err(to_backend_error)?;
                opencv::core::add_weighted(
                    &overlay,
                    alpha as f64,
                    image,
                    1.0 - alpha as f64,
                    0.0,
                    image,
                    -1,
                )
                .map_err(to_backend_error)?;
            }
        }
        Ok(())
    }
}

impl Annotator<Mat> for PercentageBarAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        for detection in detections.iter() {
            let color = to_scalar(self.palette.by_class_id(detection.class_id));
            let background = to_scalar(self.background_color);
            let anchor = detection.anchor_point(self.position);
            let half_w = self.bar_width as f32 / 2.0;

            let bg_rect = Rect::new(
                (anchor.x - half_w).round() as i32,
                (anchor.y - self.bar_height as f32).round() as i32,
                self.bar_width as i32,
                self.bar_height as i32,
            );
            imgproc::rectangle(image, bg_rect, background, -1, LINE_8, 0)
                .map_err(to_backend_error)?;

            let fraction = detection.confidence.clamp(0.0, 1.0);
            let fg_rect = Rect::new(
                bg_rect.x,
                bg_rect.y,
                (self.bar_width as f32 * fraction).round() as i32,
                self.bar_height as i32,
            );
            imgproc::rectangle(image, fg_rect, color, -1, LINE_8, 0).map_err(to_backend_error)?;
        }
        Ok(())
    }
}

impl Annotator<Mat> for IconAnnotator<Mat> {
    type Subject = Detections;

    /// Pastes the icon opaquely (no alpha blending — `Mat` has no
    /// standardized alpha-channel convention the way `RgbaImage` does).
    /// Icons that would be clipped by the image bounds are skipped
    /// entirely rather than partially pasted.
    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        let icon_w = self.icon.cols();
        let icon_h = self.icon.rows();
        for detection in detections.iter() {
            let anchor = detection.anchor_point(self.position);
            let x = (anchor.x - icon_w as f32 / 2.0).round() as i32;
            let y = (anchor.y - icon_h as f32 / 2.0).round() as i32;
            if x < 0 || y < 0 || x + icon_w > image.cols() || y + icon_h > image.rows() {
                continue;
            }
            let rect = Rect::new(x, y, icon_w, icon_h);
            let mut dst = image.roi_mut(rect).map_err(to_backend_error)?;
            self.icon.copy_to(&mut dst).map_err(to_backend_error)?;
        }
        Ok(())
    }
}

impl Annotator<Mat> for BackgroundOverlayAnnotator {
    type Subject = Detections;

    fn annotate(&self, image: &mut Mat, detections: &Detections) -> crate::Result<()> {
        let snapshot = image.clone();
        let color = to_scalar(self.color);
        let opacity = self.opacity as f64;

        let mut tint =
            Mat::new_rows_cols_with_default(image.rows(), image.cols(), image.typ(), color)
                .map_err(to_backend_error)?;
        opencv::core::add_weighted(&tint, opacity, image, 1.0 - opacity, 0.0, &mut tint, -1)
            .map_err(to_backend_error)?;
        tint.copy_to(image).map_err(to_backend_error)?;

        for detection in detections.iter() {
            let Some(rect) = clamp_rect(detection.bbox, image.cols(), image.rows()) else {
                continue;
            };
            let original = snapshot.roi(rect).map_err(to_backend_error)?;
            let mut dst = image.roi_mut(rect).map_err(to_backend_error)?;
            original.copy_to(&mut dst).map_err(to_backend_error)?;
        }
        Ok(())
    }
}
