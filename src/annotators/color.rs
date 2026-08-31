//! Deterministic color assignment for classes and tracks.

/// An RGB color used when drawing annotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
}

impl Color {
    /// Creates a new color from `(r, g, b)` channels.
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

/// A repeating sequence of visually distinct colors, indexed by class id so
/// that annotators can render each class consistently across frames.
#[derive(Debug, Clone)]
pub struct ColorPalette {
    colors: Vec<Color>,
}

impl ColorPalette {
    /// Builds a palette from an explicit color list.
    ///
    /// # Panics
    /// Panics if `colors` is empty, since [`ColorPalette::by_class_id`]
    /// would otherwise have nothing to index into.
    pub fn new(colors: Vec<Color>) -> Self {
        assert!(
            !colors.is_empty(),
            "a color palette needs at least one color"
        );
        Self { colors }
    }

    /// The color assigned to `class_id`. Class ids beyond the palette's
    /// length wrap around, so every class still gets a (repeating) color.
    pub fn by_class_id(&self, class_id: usize) -> Color {
        self.colors[class_id % self.colors.len()]
    }

    /// Number of distinct colors in the palette before it repeats.
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    /// Whether the palette has no colors (always false for a palette built
    /// via [`ColorPalette::new`] or [`ColorPalette::default`]).
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }
}

impl Default for ColorPalette {
    /// A fixed rotation of high-contrast colors, so the same class id
    /// renders the same color across an entire run without any
    /// configuration.
    fn default() -> Self {
        Self::new(vec![
            Color::new(0xA3, 0x51, 0xFB),
            Color::new(0xFF, 0x64, 0x37),
            Color::new(0x00, 0xD4, 0xBB),
            Color::new(0xFF, 0xD7, 0x00),
            Color::new(0xEF, 0x38, 0x8E),
            Color::new(0x4A, 0xD1, 0x2C),
            Color::new(0x2E, 0x86, 0xFF),
            Color::new(0xFF, 0x8C, 0x00),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn by_class_id_is_stable() {
        let palette = ColorPalette::default();
        assert_eq!(palette.by_class_id(3), palette.by_class_id(3));
    }

    #[test]
    fn by_class_id_wraps_around_palette_length() {
        let palette = ColorPalette::default();
        let len = palette.len();
        assert_eq!(palette.by_class_id(0), palette.by_class_id(len));
    }

    #[test]
    fn distinct_class_ids_within_palette_get_distinct_colors() {
        let palette = ColorPalette::default();
        assert_ne!(palette.by_class_id(0), palette.by_class_id(1));
    }

    #[test]
    #[should_panic]
    fn empty_palette_panics() {
        ColorPalette::new(vec![]);
    }
}
