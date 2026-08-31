//! A cumulative "where things have been" heat overlay, blended over the
//! image using a fixed blue-to-red color ramp.

use std::cell::RefCell;

use ndarray::Array2;

/// Accumulates a heat value at each detection's centroid, frame over frame,
/// and blends the running total over the image as a translucent color
/// ramp (cool colors for low visitation, hot colors for high).
///
/// Stateful like [`crate::annotators::TraceAnnotator`]: the accumulation
/// buffer lives in a [`RefCell`] and is lazily sized to the first image it
/// sees, so it must be reused across a whole video/frame sequence to be
/// meaningful (a single frame just shows one dot's worth of heat).
#[derive(Debug)]
pub struct HeatMapAnnotator {
    /// Radius, in pixels, of the heat splatted at each detection centroid.
    pub radius: u32,
    /// Amount added to the buffer at a centroid on each frame it is seen.
    pub intensity: f32,
    /// Overlay opacity, in `[0, 1]`, at full (saturated) heat.
    pub opacity: f32,
    pub(crate) buffer: RefCell<Option<Array2<f32>>>,
}

impl HeatMapAnnotator {
    /// Creates a new heatmap annotator with an empty accumulation buffer.
    pub fn new(radius: u32, intensity: f32, opacity: f32) -> Self {
        Self {
            radius,
            intensity,
            opacity,
            buffer: RefCell::new(None),
        }
    }

    /// Discards the accumulated heat buffer, restarting from empty.
    pub fn clear(&self) {
        *self.buffer.borrow_mut() = None;
    }
}

impl Default for HeatMapAnnotator {
    fn default() -> Self {
        Self::new(20, 0.15, 0.6)
    }
}

/// Maps a normalized heat value in `[0, 1]` to a color, ramping
/// blue (cold) -> green -> yellow -> red (hot).
///
/// Only called from the `annotate-image` / `annotate-opencv` backends, so
/// it's otherwise dead code in a `core`-only build.
#[allow(dead_code)]
pub(crate) fn heat_color(t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    let stops: [(f32, (u8, u8, u8)); 4] = [
        (0.0, (0, 0, 255)),
        (0.33, (0, 255, 0)),
        (0.66, (255, 255, 0)),
        (1.0, (255, 0, 0)),
    ];
    for pair in stops.windows(2) {
        let (t0, c0) = pair[0];
        let (t1, c1) = pair[1];
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * f).round() as u8;
            return (lerp(c0.0, c1.0), lerp(c0.1, c1.1), lerp(c0.2, c1.2));
        }
    }
    stops[3].1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_color_endpoints_are_blue_and_red() {
        assert_eq!(heat_color(0.0), (0, 0, 255));
        assert_eq!(heat_color(1.0), (255, 0, 0));
    }

    #[test]
    fn heat_color_clamps_out_of_range_input() {
        assert_eq!(heat_color(-1.0), heat_color(0.0));
        assert_eq!(heat_color(2.0), heat_color(1.0));
    }
}
