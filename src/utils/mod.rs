//! Small standalone utilities that don't belong to any single pipeline
//! stage: [`FPSMonitor`] for throughput reporting, and (under the
//! `annotate-image` feature) pure-Rust image helpers.

mod fps;

#[cfg(feature = "annotate-image")]
mod image;
#[cfg(feature = "annotate-image")]
mod image_sink;

pub use fps::FPSMonitor;

#[cfg(feature = "annotate-image")]
pub use self::image::{crop_image, letterbox, overlay_image, resize_keeping_aspect_ratio};
#[cfg(feature = "annotate-image")]
pub use self::image_sink::ImageSink;
