//! Dataset loaders and exporters: three object-detection annotation
//! formats ([`coco`], [`yolo`], [`pascal_voc`]) built around the shared
//! [`DetectionDataset`] shape, plus a folder-per-class classification
//! layout ([`classification`]) built around [`ClassificationDataset`].
//!
//! Every detection loader/exporter reads and writes the same
//! [`DetectionDataset`], so a dataset loaded from one format can be
//! re-exported to another with no extra glue.
//!
//! Requires the `datasets` feature.

pub mod classification;
pub mod coco;
pub mod pascal_voc;
pub mod yolo;

use std::path::{Path, PathBuf};

use crate::core::Detections;

/// Whether `path`'s extension looks like a common raster image format.
/// Shared by every loader in this module that walks a directory of
/// images.
pub(crate) fn is_image_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("jpg") | Some("jpeg") | Some("png") | Some("bmp") | Some("gif") | Some("tiff")
    )
}

/// One image's worth of annotations within a [`DetectionDataset`].
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetImage {
    /// Path to the image file. Loaders resolve this relative to whatever
    /// images directory they were given; exporters use it only to derive
    /// the output file's base name (they don't copy image pixels).
    pub path: PathBuf,
    /// Image width in pixels, as recorded in (or, for YOLO, read from) the
    /// source annotation.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// This image's detections, in absolute pixel coordinates.
    pub detections: Detections,
}

/// An object detection dataset: a shared class list plus one
/// [`DatasetImage`] per annotated image.
///
/// `Detection::class_id` on every image's detections indexes into
/// `classes`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DetectionDataset {
    /// Class names, indexed by `Detection::class_id`.
    pub classes: Vec<String>,
    /// One entry per annotated image.
    pub images: Vec<DatasetImage>,
}

impl DetectionDataset {
    /// Creates a new dataset from an explicit class list and image set.
    pub fn new(classes: Vec<String>, images: Vec<DatasetImage>) -> Self {
        Self { classes, images }
    }

    /// An empty dataset with no classes and no images.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Number of images in the dataset.
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Whether the dataset has no images.
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Total number of annotated object instances across every image.
    pub fn num_annotations(&self) -> usize {
        self.images.iter().map(|image| image.detections.len()).sum()
    }
}

/// One image's label within a [`ClassificationDataset`].
#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationImage {
    /// Path to the image file.
    pub path: PathBuf,
    /// Index into [`ClassificationDataset::classes`].
    pub class_id: usize,
}

/// A single-label image classification dataset: a shared class list plus
/// one [`ClassificationImage`] per image.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClassificationDataset {
    /// Class names, indexed by [`ClassificationImage::class_id`].
    pub classes: Vec<String>,
    /// One entry per labeled image.
    pub images: Vec<ClassificationImage>,
}

impl ClassificationDataset {
    /// Creates a new dataset from an explicit class list and image set.
    pub fn new(classes: Vec<String>, images: Vec<ClassificationImage>) -> Self {
        Self { classes, images }
    }

    /// An empty dataset with no classes and no images.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Number of images in the dataset.
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Whether the dataset has no images.
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }
}
