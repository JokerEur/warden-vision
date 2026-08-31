//! Tiled ("SAHI-style") inference: slices a large image into overlapping
//! tiles small enough for a detector to run accurately on, runs a
//! caller-supplied callback per tile, and merges the results back into
//! full-image coordinates.
//!
//! Deliberately has no opinion on image representation — it never touches
//! pixels itself. [`InferenceSlicer::run`] hands the caller each tile's
//! bounding box (in full-image coordinates) and expects back a
//! [`Detections`] in *that tile's local coordinates* (as if the tile were
//! its own image, starting at `(0, 0)`); how the caller crops and infers
//! on that region — via `image::RgbaImage`, an `opencv::core::Mat`, or
//! anything else — is entirely up to them.

use crate::core::Detections;

/// Slices an image into overlapping tiles for per-tile inference.
#[derive(Debug, Clone, Copy)]
pub struct InferenceSlicer {
    slice_width: u32,
    slice_height: u32,
    overlap_ratio_w: f32,
    overlap_ratio_h: f32,
    iou_threshold: f32,
}

impl InferenceSlicer {
    /// Creates a slicer producing `slice_width x slice_height` tiles, with
    /// a 20% overlap on each axis and a 0.5 IoU merge threshold (override
    /// with [`InferenceSlicer::with_overlap_ratio`] /
    /// [`InferenceSlicer::with_iou_threshold`]).
    pub fn new(slice_width: u32, slice_height: u32) -> Self {
        Self {
            slice_width: slice_width.max(1),
            slice_height: slice_height.max(1),
            overlap_ratio_w: 0.2,
            overlap_ratio_h: 0.2,
            iou_threshold: 0.5,
        }
    }

    /// Sets the fraction of each tile's width/height that overlaps its
    /// neighbor (`[0, 1)`; values are clamped to `[0, 0.9]`). Overlap
    /// keeps objects that straddle a tile boundary from being clipped in
    /// every tile that contains them.
    pub fn with_overlap_ratio(mut self, ratio_w: f32, ratio_h: f32) -> Self {
        self.overlap_ratio_w = ratio_w.clamp(0.0, 0.9);
        self.overlap_ratio_h = ratio_h.clamp(0.0, 0.9);
        self
    }

    /// Sets the IoU threshold used to merge duplicate detections of the
    /// same object found in multiple overlapping tiles (via
    /// [`Detections::non_max_suppression`]).
    pub fn with_iou_threshold(mut self, iou_threshold: f32) -> Self {
        self.iou_threshold = iou_threshold;
        self
    }

    /// The tile bounding boxes (`[x1, y1, x2, y2]`, in full-image pixel
    /// coordinates) covering an `image_width x image_height` image.
    pub fn slices(&self, image_width: u32, image_height: u32) -> Vec<[f32; 4]> {
        let xs = axis_starts(image_width, self.slice_width, self.overlap_ratio_w);
        let ys = axis_starts(image_height, self.slice_height, self.overlap_ratio_h);
        let tile_w = self.slice_width.min(image_width.max(1));
        let tile_h = self.slice_height.min(image_height.max(1));

        let mut tiles = Vec::with_capacity(xs.len() * ys.len());
        for &y in &ys {
            for &x in &xs {
                tiles.push([x as f32, y as f32, (x + tile_w) as f32, (y + tile_h) as f32]);
            }
        }
        tiles
    }

    /// Runs `callback` once per tile (see the module docs for the
    /// tile-local-coordinates contract), offsets each tile's detections
    /// back into full-image coordinates, and merges everything with
    /// [`Detections::non_max_suppression`] to collapse duplicates found
    /// in more than one overlapping tile.
    pub fn run<F>(&self, image_width: u32, image_height: u32, mut callback: F) -> Detections
    where
        F: FnMut([f32; 4]) -> Detections,
    {
        let mut merged = Vec::new();
        for tile in self.slices(image_width, image_height) {
            let [offset_x, offset_y, _, _] = tile;
            let tile_detections = callback(tile);
            merged.extend(tile_detections.iter().map(|detection| {
                let mut offset = detection.clone();
                let [x1, y1, x2, y2] = detection.bbox;
                offset.bbox = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y];
                offset.mask = detection.mask.as_ref().map(|polygon| {
                    polygon
                        .iter()
                        .map(|&[x, y]| [x + offset_x, y + offset_y])
                        .collect()
                });
                offset
            }));
        }
        Detections::new(merged).non_max_suppression(self.iou_threshold, false)
    }
}

/// Tile start positions along one axis: steps of `slice_len * (1 -
/// overlap_ratio)` from `0`, with the final tile always flush against the
/// far edge (rather than overshooting it) so every tile is exactly
/// `slice_len` wide/tall whenever `image_len >= slice_len`.
fn axis_starts(image_len: u32, slice_len: u32, overlap_ratio: f32) -> Vec<u32> {
    if image_len <= slice_len {
        return vec![0];
    }
    let step = ((slice_len as f32) * (1.0 - overlap_ratio))
        .round()
        .max(1.0) as u32;
    let last_start = image_len - slice_len;

    let mut starts = Vec::new();
    let mut x = 0u32;
    while x < last_start {
        starts.push(x);
        x += step;
    }
    starts.push(last_start);
    starts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Detection;

    #[test]
    fn slices_cover_an_image_smaller_than_one_tile_with_a_single_tile() {
        let slicer = InferenceSlicer::new(100, 100);
        let tiles = slicer.slices(50, 40);
        assert_eq!(tiles, vec![[0.0, 0.0, 50.0, 40.0]]);
    }

    #[test]
    fn slices_tile_an_exact_multiple_with_no_overlap_needed() {
        let slicer = InferenceSlicer::new(100, 100).with_overlap_ratio(0.0, 0.0);
        let tiles = slicer.slices(200, 100);
        assert_eq!(
            tiles,
            vec![[0.0, 0.0, 100.0, 100.0], [100.0, 0.0, 200.0, 100.0]]
        );
    }

    #[test]
    fn slices_last_tile_is_flush_with_the_far_edge_not_overshooting_it() {
        let slicer = InferenceSlicer::new(100, 100).with_overlap_ratio(0.2, 0.2);
        let tiles = slicer.slices(250, 100);
        for [x1, _, x2, _] in &tiles {
            assert!(*x2 <= 250.0, "tile [{x1}, {x2}] overshoots the image width");
        }
        // The rightmost tile must still be exactly reachable at x=150 to
        // stay flush: [150, 0, 250, 100].
        assert!(tiles.contains(&[150.0, 0.0, 250.0, 100.0]));
    }

    #[test]
    fn slices_cover_the_full_image_with_no_gaps() {
        let slicer = InferenceSlicer::new(64, 64).with_overlap_ratio(0.1, 0.1);
        let tiles = slicer.slices(300, 217);
        let max_x = tiles.iter().map(|t| t[2] as u32).max().unwrap();
        let max_y = tiles.iter().map(|t| t[3] as u32).max().unwrap();
        assert_eq!(max_x, 300);
        assert_eq!(max_y, 217);
        assert!(tiles.iter().any(|t| t[0] == 0.0 && t[1] == 0.0));
    }

    #[test]
    fn run_offsets_tile_local_detections_into_full_image_coordinates() {
        let slicer = InferenceSlicer::new(10, 10).with_overlap_ratio(0.0, 0.0);
        let detections = slicer.run(20, 10, |_tile| {
            // One detection per tile, at the tile's local origin.
            Detections::new(vec![Detection::new([0.0, 0.0, 2.0, 2.0], 0.9, 0)])
        });
        assert_eq!(detections.len(), 2);
        let mut x1s: Vec<f32> = detections.iter().map(|d| d.bbox[0]).collect();
        x1s.sort_by(f32::total_cmp);
        assert_eq!(x1s, vec![0.0, 10.0]);
    }

    #[test]
    fn run_merges_duplicate_detections_from_overlapping_tiles() {
        // A single wide tile setup where the same real-world box, found
        // independently by two overlapping tiles, should collapse to one
        // detection after NMS.
        let slicer = InferenceSlicer::new(10, 10)
            .with_overlap_ratio(0.5, 0.0)
            .with_iou_threshold(0.3);
        let detections = slicer.run(15, 10, |tile| {
            // Every tile reports the same full-image-local box (5,0)-(9,10)
            // translated to that tile's local frame.
            let local_x1 = (5.0 - tile[0]).clamp(0.0, 10.0);
            Detections::new(vec![Detection::new(
                [local_x1, 0.0, local_x1 + 4.0, 10.0],
                0.9,
                0,
            )])
        });
        assert_eq!(detections.len(), 1);
    }
}
