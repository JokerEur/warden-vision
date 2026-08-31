//! Privacy/redaction annotators: blur or pixelate the region inside each
//! detection's bounding box instead of drawing on top of it.

/// Box-blurs the pixels inside each detection's bounding box.
#[derive(Debug, Clone)]
pub struct BlurAnnotator {
    /// Box-blur kernel size, in pixels. Larger values blur more.
    pub kernel_size: u32,
}

impl BlurAnnotator {
    /// Creates a new blur annotator.
    pub fn new(kernel_size: u32) -> Self {
        Self { kernel_size }
    }
}

impl Default for BlurAnnotator {
    fn default() -> Self {
        Self::new(15)
    }
}

/// Pixelates (mosaics) the region inside each detection's bounding box.
#[derive(Debug, Clone)]
pub struct PixelateAnnotator {
    /// Side length, in pixels, of each pixelated block.
    pub pixel_size: u32,
}

impl PixelateAnnotator {
    /// Creates a new pixelate annotator.
    pub fn new(pixel_size: u32) -> Self {
        Self { pixel_size }
    }
}

impl Default for PixelateAnnotator {
    fn default() -> Self {
        Self::new(10)
    }
}
