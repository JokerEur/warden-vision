//! Scoring a detector's predictions against ground truth: Precision,
//! Recall, F1, and mean Average Precision, accumulated across a small
//! two-image "validation set".
//!
//! Run with: `cargo run --example metrics_evaluation`

use warden_vision::core::{Detection, Detections};
use warden_vision::metrics::{F1Score, MeanAveragePrecision, Precision, Recall};

fn main() {
    // Image 1: one true positive, one false positive (the second
    // prediction has no matching ground-truth box).
    let predictions_1 = Detections::new(vec![
        Detection::new([10.0, 10.0, 50.0, 50.0], 0.95, 0),
        Detection::new([200.0, 200.0, 240.0, 240.0], 0.60, 0),
    ]);
    let targets_1 = Detections::new(vec![Detection::new([12.0, 9.0, 49.0, 52.0], 1.0, 0)]);

    // Image 2: one true positive, one missed detection (false negative).
    let predictions_2 = Detections::new(vec![Detection::new([30.0, 30.0, 70.0, 70.0], 0.88, 1)]);
    let targets_2 = Detections::new(vec![
        Detection::new([31.0, 28.0, 69.0, 72.0], 1.0, 1),
        Detection::new([300.0, 100.0, 340.0, 140.0], 1.0, 1),
    ]);

    let iou_threshold = 0.5;
    let mut precision = Precision::new(iou_threshold);
    let mut recall = Recall::new(iou_threshold);
    let mut f1 = F1Score::new(iou_threshold);
    let mut map = MeanAveragePrecision::coco();

    for (predictions, targets) in [(&predictions_1, &targets_1), (&predictions_2, &targets_2)] {
        precision.update(predictions, targets);
        recall.update(predictions, targets);
        f1.update(predictions, targets);
        map.update(predictions, targets);
    }

    println!("precision@{iou_threshold}: {:.3}", precision.compute());
    println!("recall@{iou_threshold}:    {:.3}", recall.compute());
    println!("f1@{iou_threshold}:        {:.3}", f1.compute());

    let map_result = map.compute();
    println!("mAP@50:95:          {:.3}", map_result.map);
    if let Some(map_50) = map_result.map_at(0.5) {
        println!("mAP@50:              {:.3}", map_50);
    }
}
