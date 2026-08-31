//! Load and save classification datasets in the "image folder" layout
//! (as used by, e.g., `torchvision.datasets.ImageFolder`): one
//! subdirectory per class, containing that class's images —
//! `root/<class_name>/<image>.jpg`.
//!
//! Unlike the detection formats in [`crate::dataset`], class membership
//! *is* the directory structure here, so [`save`] physically copies image
//! files into their class's subdirectory rather than just writing a
//! separate annotation file next to images that stay where they are.

use std::fs;
use std::path::{Path, PathBuf};

use crate::dataset::{is_image_file, ClassificationDataset, ClassificationImage};
use crate::error::Error;

/// Loads a classification dataset from `root`: each direct subdirectory
/// of `root` becomes a class (named after the subdirectory, in sorted
/// order), and every image file directly inside it becomes one labeled
/// image.
pub fn load(root: &Path) -> crate::Result<ClassificationDataset> {
    let mut class_dirs: Vec<PathBuf> = fs::read_dir(root)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.is_dir())
        .collect();
    class_dirs.sort();

    let classes: Vec<String> = class_dirs
        .iter()
        .map(|dir| {
            dir.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string()
        })
        .collect();

    let mut images = Vec::new();
    for (class_id, class_dir) in class_dirs.iter().enumerate() {
        let mut image_paths: Vec<PathBuf> = fs::read_dir(class_dir)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| is_image_file(path))
            .collect();
        image_paths.sort();
        images.extend(
            image_paths
                .into_iter()
                .map(|path| ClassificationImage { path, class_id }),
        );
    }

    Ok(ClassificationDataset::new(classes, images))
}

/// Saves `dataset` in image-folder layout under `root`, copying each
/// image into `root/<class_name>/<original_file_name>`.
pub fn save(dataset: &ClassificationDataset, root: &Path) -> crate::Result<()> {
    for image in &dataset.images {
        let class_name = dataset
            .classes
            .get(image.class_id)
            .map(|s| s.as_str())
            .ok_or_else(|| {
                Error::Parse(format!("no class name for class_id {}", image.class_id))
            })?;
        let class_dir = root.join(class_name);
        fs::create_dir_all(&class_dir)?;

        let file_name = image.path.file_name().ok_or_else(|| {
            Error::Parse(format!(
                "image path has no file name: {}",
                image.path.display()
            ))
        })?;
        fs::copy(&image.path, class_dir.join(file_name))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_image(path: &Path) {
        let image = image::RgbImage::new(4, 4);
        image.save(path).unwrap();
    }

    #[test]
    fn round_trips_through_save_and_load() {
        let source_dir = tempfile::tempdir().unwrap();
        let cat_path = source_dir.path().join("cat.jpg");
        let dog_path = source_dir.path().join("dog.jpg");
        write_test_image(&cat_path);
        write_test_image(&dog_path);

        let dataset = ClassificationDataset::new(
            vec!["cat".to_string(), "dog".to_string()],
            vec![
                ClassificationImage {
                    path: cat_path,
                    class_id: 0,
                },
                ClassificationImage {
                    path: dog_path,
                    class_id: 1,
                },
            ],
        );

        let export_dir = tempfile::tempdir().unwrap();
        save(&dataset, export_dir.path()).unwrap();
        let loaded = load(export_dir.path()).unwrap();

        assert_eq!(loaded.classes, vec!["cat".to_string(), "dog".to_string()]);
        assert_eq!(loaded.len(), 2);
        assert!(loaded
            .images
            .iter()
            .any(|i| i.class_id == 0 && i.path.ends_with("cat.jpg")));
        assert!(loaded
            .images
            .iter()
            .any(|i| i.class_id == 1 && i.path.ends_with("dog.jpg")));
    }

    #[test]
    fn class_names_follow_sorted_directory_order() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("zebra")).unwrap();
        fs::create_dir_all(dir.path().join("ant")).unwrap();
        write_test_image(&dir.path().join("zebra").join("a.jpg"));
        write_test_image(&dir.path().join("ant").join("b.jpg"));

        let loaded = load(dir.path()).unwrap();
        assert_eq!(loaded.classes, vec!["ant".to_string(), "zebra".to_string()]);
    }

    #[test]
    fn empty_root_yields_an_empty_dataset() {
        let dir = tempfile::tempdir().unwrap();
        let loaded = load(dir.path()).unwrap();
        assert!(loaded.is_empty());
        assert!(loaded.classes.is_empty());
    }
}
