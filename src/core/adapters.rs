//! Constructors that turn a detection model's raw output tensor into
//! [`Detections`].
//!
//! Python `supervision` has `Detections.from_ultralytics(result)` and
//! `Detections.from_transformers(result)`, which unpack a specific
//! Python library's result *object*. There is no Rust equivalent of
//! either object — `ultralytics` and `transformers` are Python/PyTorch
//! libraries with no Rust port — so binding to them the same way isn't
//! possible.
//!
//! What *is* portable is the tensor layout each model family produces,
//! which is exactly what you get back after running an ONNX export of
//! one of these models through a Rust inference runtime (`ort`, `tract`,
//! `candle`, ...). This module parses those tensor layouts directly, as
//! plain `ndarray` views, so it stays runtime-agnostic: it doesn't matter
//! which crate you used to actually run the model.

use ndarray::ArrayView2;

use crate::core::{Detection, Detections};

impl Detections {
    /// Parses predictions in the Ultralytics YOLOv8/v9/v10/v11 ONNX
    /// export's raw-output layout: one row per candidate box, `[cx, cy,
    /// w, h, class_0_score, class_1_score, ..., class_{C-1}_score]`.
    ///
    /// The raw ONNX output tensor is `(1, 4 + C, N)` (channel-first);
    /// reshape/transpose it to `(N, 4 + C)` before calling this (e.g. via
    /// `array.index_axis_move(Axis(0), 0).reversed_axes()` on an
    /// `ndarray` output, or the equivalent in whatever tensor type your
    /// inference runtime hands back).
    ///
    /// `cx, cy, w, h` are read as-is — typically in the model's input
    /// resolution (e.g. 640x640 pixels), *not* normalized to `[0, 1]`.
    /// If you preprocessed with letterboxing/resizing, use
    /// [`Detections::scale`] (and offset the box coordinates yourself, if
    /// you also padded) to map back to the original image size.
    ///
    /// This performs confidence filtering only — no NMS. Ultralytics'
    /// own postprocessing runs NMS after exactly this step; call
    /// [`Detections::non_max_suppression`] yourself afterward.
    pub fn from_ultralytics_onnx(
        predictions: ArrayView2<f32>,
        confidence_threshold: f32,
    ) -> Detections {
        let mut detections = Vec::new();
        for row in predictions.rows() {
            if row.len() < 5 {
                continue;
            }
            let (cx, cy, w, h) = (row[0], row[1], row[2], row[3]);
            let class_scores = row.slice(ndarray::s![4..]);
            let Some((class_id, &confidence)) = class_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
            else {
                continue;
            };
            if confidence < confidence_threshold {
                continue;
            }
            let bbox = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
            detections.push(Detection::new(bbox, confidence, class_id));
        }
        Detections::new(detections)
    }

    /// Parses predictions in a DETR-family (HuggingFace `transformers`:
    /// DETR, Deformable DETR, RT-DETR, ...) model's raw-output layout:
    /// `logits` of shape `(num_queries, num_classes + 1)` — DETR-family
    /// models predict an explicit extra "no object" class, always the
    /// last column — and `pred_boxes` of shape `(num_queries, 4)`,
    /// holding `[cx, cy, w, h]` normalized to `[0, 1]` relative to
    /// `(image_width, image_height)`.
    ///
    /// Applies softmax over `logits` per query, drops the no-object
    /// class, and keeps the best-scoring remaining class per query if it
    /// clears `confidence_threshold`. `image_width`/`image_height` should
    /// be the dimensions of the image you actually want box coordinates
    /// in (typically the original image, since DETR-family
    /// preprocessing commonly resizes without letterbox padding).
    ///
    /// # Panics
    /// Panics if `logits` and `pred_boxes` have a different number of
    /// rows (queries).
    pub fn from_transformers_onnx(
        logits: ArrayView2<f32>,
        pred_boxes: ArrayView2<f32>,
        image_width: u32,
        image_height: u32,
        confidence_threshold: f32,
    ) -> Detections {
        assert_eq!(
            logits.nrows(),
            pred_boxes.nrows(),
            "logits and pred_boxes must have the same number of queries"
        );
        let num_classes = logits.ncols().saturating_sub(1);
        if num_classes == 0 {
            return Detections::empty();
        }

        let mut detections = Vec::new();
        for (query_logits, box_row) in logits.rows().into_iter().zip(pred_boxes.rows()) {
            let max_logit = query_logits
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = query_logits.iter().map(|&v| (v - max_logit).exp()).sum();

            let Some((class_id, confidence)) = query_logits
                .iter()
                .take(num_classes)
                .map(|&v| (v - max_logit).exp() / exp_sum)
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
            else {
                continue;
            };
            if confidence < confidence_threshold {
                continue;
            }

            let cx = box_row[0] * image_width as f32;
            let cy = box_row[1] * image_height as f32;
            let w = box_row[2] * image_width as f32;
            let h = box_row[3] * image_height as f32;
            let bbox = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0];
            detections.push(Detection::new(bbox, confidence, class_id));
        }
        Detections::new(detections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn from_ultralytics_onnx_picks_the_best_class_and_filters_by_confidence() {
        // Two candidate boxes, 3 classes each. Row 0: class 1 wins at 0.9.
        // Row 1: best score 0.2, below threshold.
        let predictions = array![
            [10.0, 10.0, 4.0, 4.0, 0.1, 0.9, 0.05],
            [50.0, 50.0, 4.0, 4.0, 0.2, 0.1, 0.05],
        ];
        let detections = Detections::from_ultralytics_onnx(predictions.view(), 0.5);
        assert_eq!(detections.len(), 1);
        assert_eq!(detections.detections[0].class_id, 1);
        assert!((detections.detections[0].confidence - 0.9).abs() < 1e-6);
        assert_eq!(detections.detections[0].bbox, [8.0, 8.0, 12.0, 12.0]);
    }

    #[test]
    fn from_ultralytics_onnx_handles_empty_input() {
        let predictions = ArrayView2::<f32>::from_shape((0, 5), &[]).unwrap();
        let detections = Detections::from_ultralytics_onnx(predictions, 0.5);
        assert!(detections.is_empty());
    }

    #[test]
    fn from_transformers_onnx_picks_the_best_real_class_and_converts_coordinates() {
        // 2 real classes + 1 "no object" class (index 2). Row 0 favors
        // class 0, row 1 favors class 1; the no-object logit is kept
        // small in both so it doesn't crush the real-class probabilities
        // through softmax normalization.
        let logits = array![[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]];
        let pred_boxes = array![[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]];
        let detections =
            Detections::from_transformers_onnx(logits.view(), pred_boxes.view(), 100, 200, 0.5);
        assert_eq!(detections.len(), 2);
        assert_eq!(detections.detections[0].class_id, 0);
        assert_eq!(detections.detections[1].class_id, 1);
        // cx=0.5*100=50, w=0.2*100=20 -> [40, ...]
        assert!((detections.detections[0].bbox[0] - 40.0).abs() < 1e-3);
    }

    #[test]
    fn from_transformers_onnx_never_returns_the_no_object_class_id() {
        // The no-object class (index 2) has by far the largest logit;
        // whichever real class wins must still be < num_classes.
        let logits = array![[5.0, 1.0, 10.0]];
        let pred_boxes = array![[0.5, 0.5, 0.2, 0.2]];
        let detections = Detections::from_transformers_onnx(
            logits.view(),
            pred_boxes.view(),
            100,
            100,
            0.0, // accept regardless of how small the crushed probability is
        );
        assert_eq!(detections.len(), 1);
        assert!(detections.detections[0].class_id < 2);
    }

    #[test]
    fn from_transformers_onnx_filters_by_confidence() {
        let logits = array![[0.0, 0.0, 0.0]]; // uniform softmax: 1/3 per class, including no-object
        let pred_boxes = array![[0.5, 0.5, 0.2, 0.2]];
        let detections =
            Detections::from_transformers_onnx(logits.view(), pred_boxes.view(), 100, 100, 0.9);
        assert!(detections.is_empty());
    }

    #[test]
    #[should_panic]
    fn from_transformers_onnx_panics_on_mismatched_row_counts() {
        let logits = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let pred_boxes = array![[0.5, 0.5, 0.2, 0.2]];
        Detections::from_transformers_onnx(logits.view(), pred_boxes.view(), 100, 100, 0.1);
    }
}
