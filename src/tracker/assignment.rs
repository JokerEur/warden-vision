//! Shared linear-assignment helper, used by [`crate::tracker::SortTracker`],
//! [`crate::tracker::ByteTracker`], and [`crate::tracker::DeepSortTracker`]
//! to match predicted track state against detections.

use lapjv::Matrix;

use crate::core::bbox_iou;

/// Cost assigned to cross pairings between two different padding blocks in
/// the assignment matrix; large enough that the solver never prefers it
/// over a real or self-paired dummy match.
const PAD_COST: f32 = 1e6;

/// Solves assignment between `num_rows` and `num_cols` items (e.g. tracks
/// and detections) given an arbitrary `cost(row, col)`, returning, for each
/// index into `0..num_cols`, the matched index into `0..num_rows` (if any).
///
/// `lapjv` only solves square cost matrices, so rows and columns are
/// matched via an `(m + n) x (m + n)` padded matrix: the top-left `m x n`
/// block holds `cost_fn`'s real costs; the remaining blocks let any row or
/// column be assigned to a same-index "dummy" at a fixed `no_match_cost`,
/// which is how an entry ends up unmatched when no real pairing is good
/// enough (`cost_fn` should return something `>= no_match_cost` for pairs
/// that must never be matched, e.g. a hard gating distance).
pub(crate) fn assign_by_cost<C>(
    num_rows: usize,
    num_cols: usize,
    no_match_cost: f32,
    cost_fn: C,
) -> Vec<Option<usize>>
where
    C: Fn(usize, usize) -> f32,
{
    let mut matches = vec![None; num_cols];
    if num_rows == 0 || num_cols == 0 {
        return matches;
    }

    let dim = num_rows + num_cols;
    let mut cost = Matrix::<f32>::from_elem((dim, dim), PAD_COST);

    for i in 0..num_rows {
        for j in 0..num_cols {
            cost[(i, j)] = cost_fn(i, j);
        }
    }
    for i in 0..num_rows {
        for k in 0..num_rows {
            cost[(i, num_cols + k)] = if i == k { no_match_cost } else { PAD_COST };
        }
    }
    for j in 0..num_cols {
        for k in 0..num_cols {
            cost[(num_rows + k, j)] = if j == k { no_match_cost } else { PAD_COST };
        }
    }
    for a in 0..num_cols {
        for b in 0..num_rows {
            cost[(num_rows + a, num_cols + b)] = 0.0;
        }
    }

    let Ok((row_to_col, _col_to_row)) = lapjv::lapjv(&cost) else {
        return matches;
    };

    for i in 0..num_rows {
        let j = row_to_col[i];
        if j < num_cols && cost[(i, j)] < no_match_cost {
            matches[j] = Some(i);
        }
    }

    matches
}

/// Solves assignment between `rows` and `cols` (e.g. predicted track boxes
/// and detection boxes) by IoU, returning, for each index into `cols`, the
/// matched index into `rows` (if any).
pub(crate) fn assign_by_iou(
    rows: &[[f32; 4]],
    cols: &[[f32; 4]],
    iou_threshold: f32,
) -> Vec<Option<usize>> {
    assign_by_cost(rows.len(), cols.len(), 1.0 - iou_threshold, |i, j| {
        1.0 - bbox_iou(rows[i], cols[j])
    })
}
