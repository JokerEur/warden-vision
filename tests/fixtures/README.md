# Test fixtures

- `test_font.ttf`: a tiny synthetic TrueType font used only to exercise
  `RichLabelAnnotator`'s glyph-rasterization code path in tests. Copied
  from the [`ttf-parser`](https://github.com/RazrFalcon/ttf-parser) crate's
  own test suite (`tests/fonts/demo.ttf`), which is dual-licensed
  MIT/Apache-2.0, same as this crate. It is not used at runtime and is not
  part of this crate's public API.
