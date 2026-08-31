//! Shared linear-assignment-by-IoU helper, used by both
//! [`crate::tracker::SortTracker`] and [`crate::tracker::ByteTracker`] to
//! match predicted track boxes against detection boxes.

use lapjv::Matrix;

use crate::core::bbox_iou;

/// Cost assigned to cross pairings between two different padding blocks in
/// the assignment matrix; large enough that the solver never prefers it
/// over a real or self-paired dummy match.
const PAD_COST: f32 = 1e6;

/// Solves assignment between `rows` and `cols` (e.g. predicted track boxes
/// and detection boxes) by IoU, returning, for each index into `cols`, the
/// matched index into `rows` (if any).
///
/// `lapjv` only solves square cost matrices, so `rows` and `cols` are
/// matched via an `(m + n) x (m + n)` padded matrix: the top-left `m x n`
/// block holds real `1 - iou` costs; the remaining blocks let any row or
/// column be assigned to a same-index "dummy" at a fixed `1 -
/// iou_threshold` cost, which is how an entry ends up unmatched when no
/// real pairing is good enough.
pub(crate) fn assign_by_iou(
    rows: &[[f32; 4]],
    cols: &[[f32; 4]],
    iou_threshold: f32,
) -> Vec<Option<usize>> {
    let mut matches = vec![None; cols.len()];
    if rows.is_empty() || cols.is_empty() {
        return matches;
    }

    let dim = rows.len() + cols.len();
    let no_match_cost = 1.0 - iou_threshold;
    let mut cost = Matrix::<f32>::from_elem((dim, dim), PAD_COST);

    for i in 0..rows.len() {
        for j in 0..cols.len() {
            let iou = bbox_iou(rows[i], cols[j]);
            cost[(i, j)] = 1.0 - iou;
        }
    }
    for i in 0..rows.len() {
        for k in 0..rows.len() {
            cost[(i, cols.len() + k)] = if i == k { no_match_cost } else { PAD_COST };
        }
    }
    for j in 0..cols.len() {
        for k in 0..cols.len() {
            cost[(rows.len() + k, j)] = if j == k { no_match_cost } else { PAD_COST };
        }
    }
    for a in 0..cols.len() {
        for b in 0..rows.len() {
            cost[(rows.len() + a, cols.len() + b)] = 0.0;
        }
    }

    let Ok((row_to_col, _col_to_row)) = lapjv::lapjv(&cost) else {
        return matches;
    };

    for i in 0..rows.len() {
        let j = row_to_col[i];
        if j < cols.len() && cost[(i, j)] < no_match_cost {
            matches[j] = Some(i);
        }
    }

    matches
}
