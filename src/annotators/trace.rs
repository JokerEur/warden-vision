//! A per-tracker_id position trail, drawn as a polyline connecting an
//! object's recent anchor points.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};

use crate::annotators::ColorPalette;
use crate::geometry::Position;

/// Draws a fading trail behind each tracked object, connecting its anchor
/// point over the last [`TraceAnnotator::trace_length`] frames it was seen
/// in.
///
/// Unlike the other annotators in this crate, `TraceAnnotator` is
/// stateful: each call to `annotate` both records the current frame's
/// positions and draws the trail accumulated so far, keyed by
/// [`tracker_id`](crate::core::Detection::tracker_id). Detections with no
/// tracker id are ignored, since there is no identity to accumulate a
/// trail under. History is kept in a [`RefCell`] so the annotator can still
/// be driven through the shared `&self` [`crate::annotators::Annotator::annotate`]
/// signature.
#[derive(Debug, Clone)]
pub struct TraceAnnotator {
    /// Maps class ids to colors.
    pub palette: ColorPalette,
    /// Stroke width, in pixels.
    pub thickness: u32,
    /// Which anchor point to trail.
    pub position: Position,
    /// Maximum number of past positions kept per tracker id.
    pub trace_length: usize,
    pub(crate) history: RefCell<HashMap<usize, VecDeque<(f32, f32)>>>,
}

impl TraceAnnotator {
    /// Creates a new trace annotator with empty history.
    pub fn new(
        palette: ColorPalette,
        thickness: u32,
        position: Position,
        trace_length: usize,
    ) -> Self {
        Self {
            palette,
            thickness,
            position,
            trace_length,
            history: RefCell::new(HashMap::new()),
        }
    }

    /// Discards all recorded trails.
    pub fn clear(&self) {
        self.history.borrow_mut().clear();
    }
}

impl Default for TraceAnnotator {
    fn default() -> Self {
        Self::new(ColorPalette::default(), 2, Position::Center, 30)
    }
}
