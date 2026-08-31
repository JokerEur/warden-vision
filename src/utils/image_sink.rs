//! Writes a numbered sequence of frames to a directory — the save-side
//! counterpart to iterating video frames (e.g. via
//! [`crate::video::frames`] when the `annotate-opencv` feature is on).

use std::path::{Path, PathBuf};

use image::{DynamicImage, RgbaImage};

use crate::error::Error;

fn to_io_error(err: image::ImageError) -> Error {
    Error::Parse(err.to_string())
}

/// Saves `image` to `path`, dropping the alpha channel first for formats
/// (JPEG) whose encoder doesn't support one — saving an `RgbaImage`
/// straight to `.jpg` otherwise fails outright.
fn save_rgba(image: &RgbaImage, path: &Path) -> crate::Result<()> {
    let is_jpeg = matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("jpg") | Some("jpeg")
    );
    if is_jpeg {
        DynamicImage::ImageRgba8(image.clone())
            .to_rgb8()
            .save(path)
            .map_err(to_io_error)
    } else {
        image.save(path).map_err(to_io_error)
    }
}

/// Saves images to a target directory under auto-incrementing (or
/// explicit) file names.
#[derive(Debug, Clone)]
pub struct ImageSink {
    target_dir: PathBuf,
    prefix: String,
    extension: String,
    padding: usize,
    overwrite: bool,
    next_index: usize,
}

impl ImageSink {
    /// Creates a sink writing into `target_dir`, creating it (and any
    /// missing parent directories) if it doesn't exist. Defaults to
    /// `image_00000.png`, `image_00001.png`, ... file names.
    pub fn new(target_dir: impl Into<PathBuf>) -> crate::Result<Self> {
        let target_dir = target_dir.into();
        std::fs::create_dir_all(&target_dir)?;
        Ok(Self {
            target_dir,
            prefix: "image".to_string(),
            extension: "png".to_string(),
            padding: 5,
            overwrite: false,
            next_index: 0,
        })
    }

    /// Sets the file name prefix (default `"image"`).
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Sets the file extension, without a leading dot (default `"png"`).
    /// Determines the encoded image format.
    pub fn with_extension(mut self, extension: impl Into<String>) -> Self {
        self.extension = extension.into();
        self
    }

    /// Sets the zero-padded width of the auto-incrementing index
    /// (default `5`, i.e. `00000`).
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// If `true`, [`ImageSink::save_image`] may overwrite a file that
    /// already exists at its generated name; if `false` (the default),
    /// it instead skips ahead to the next free name.
    pub fn overwrite(mut self, overwrite: bool) -> Self {
        self.overwrite = overwrite;
        self
    }

    /// Saves `image` under the next auto-generated name (`<prefix>_<index>.<extension>`),
    /// returning the path it was written to.
    pub fn save_image(&mut self, image: &RgbaImage) -> crate::Result<PathBuf> {
        loop {
            let name = format!(
                "{}_{:0width$}.{}",
                self.prefix,
                self.next_index,
                self.extension,
                width = self.padding
            );
            let path = self.target_dir.join(name);
            self.next_index += 1;
            if !self.overwrite && path.exists() {
                continue;
            }
            save_rgba(image, &path)?;
            return Ok(path);
        }
    }

    /// Saves `image` under an explicit `file_name` (resolved relative to
    /// this sink's target directory), bypassing the auto-incrementing
    /// counter entirely.
    pub fn save_image_as(&self, image: &RgbaImage, file_name: &str) -> crate::Result<PathBuf> {
        let path = self.target_dir.join(file_name);
        save_rgba(image, &path)?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_image_uses_incrementing_zero_padded_names() {
        let dir = tempfile::tempdir().unwrap();
        let mut sink = ImageSink::new(dir.path()).unwrap();
        let image = RgbaImage::new(2, 2);

        let first = sink.save_image(&image).unwrap();
        let second = sink.save_image(&image).unwrap();

        assert_eq!(first.file_name().unwrap(), "image_00000.png");
        assert_eq!(second.file_name().unwrap(), "image_00001.png");
        assert!(first.exists());
        assert!(second.exists());
    }

    #[test]
    fn with_prefix_extension_and_padding_customize_the_name() {
        let dir = tempfile::tempdir().unwrap();
        let mut sink = ImageSink::new(dir.path())
            .unwrap()
            .with_prefix("frame")
            .with_extension("jpg")
            .with_padding(3);
        let image = RgbaImage::new(2, 2);

        let path = sink.save_image(&image).unwrap();
        assert_eq!(path.file_name().unwrap(), "frame_000.jpg");
    }

    #[test]
    fn without_overwrite_skips_past_existing_files() {
        let dir = tempfile::tempdir().unwrap();
        let image = RgbaImage::new(2, 2);
        // Pre-create the name the sink would generate first.
        image.save(dir.path().join("image_00000.png")).unwrap();

        let mut sink = ImageSink::new(dir.path()).unwrap();
        let path = sink.save_image(&image).unwrap();
        assert_eq!(path.file_name().unwrap(), "image_00001.png");
    }

    #[test]
    fn with_overwrite_reuses_an_existing_name() {
        let dir = tempfile::tempdir().unwrap();
        let image = RgbaImage::new(2, 2);
        image.save(dir.path().join("image_00000.png")).unwrap();

        let mut sink = ImageSink::new(dir.path()).unwrap().overwrite(true);
        let path = sink.save_image(&image).unwrap();
        assert_eq!(path.file_name().unwrap(), "image_00000.png");
    }

    #[test]
    fn save_image_as_bypasses_the_counter() {
        let dir = tempfile::tempdir().unwrap();
        let sink = ImageSink::new(dir.path()).unwrap();
        let image = RgbaImage::new(2, 2);
        let path = sink.save_image_as(&image, "custom.png").unwrap();
        assert_eq!(path.file_name().unwrap(), "custom.png");
        assert!(path.exists());
    }
}
