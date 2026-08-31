//! Video I/O built on OpenCV's `videoio` module: reading frames from a
//! video file ([`VideoInfo`], [`frames`]) and writing them to a new one
//! ([`VideoSink`]).
//!
//! Requires the `annotate-opencv` feature (which already enables
//! `opencv`'s `videoio` feature). NOTE: not build-verified against a real
//! OpenCV install — same caveat as the rest of the OpenCV backend.

use std::path::Path;

use opencv::core::{Mat, Size};
use opencv::prelude::*;
use opencv::videoio::{self, VideoCapture, VideoWriter};

use crate::error::Error;

fn to_backend_error(err: opencv::Error) -> Error {
    Error::Backend(err.to_string())
}

/// Static metadata about a video file: dimensions, frame rate, and
/// (when the container reports it) total frame count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoInfo {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frames per second, as reported by the container.
    pub fps: f64,
    /// Total frame count, if the container reports one (some streams,
    /// e.g. live captures, do not).
    pub total_frames: Option<u32>,
}

impl VideoInfo {
    /// Reads video metadata by briefly opening `path`.
    pub fn from_path(path: &Path) -> crate::Result<Self> {
        let capture = VideoCapture::from_file(path, videoio::CAP_ANY).map_err(to_backend_error)?;
        if !capture.is_opened().map_err(to_backend_error)? {
            return Err(Error::Backend(format!(
                "could not open video: {}",
                path.display()
            )));
        }
        let width = capture
            .get(videoio::CAP_PROP_FRAME_WIDTH)
            .map_err(to_backend_error)? as u32;
        let height = capture
            .get(videoio::CAP_PROP_FRAME_HEIGHT)
            .map_err(to_backend_error)? as u32;
        let fps = capture
            .get(videoio::CAP_PROP_FPS)
            .map_err(to_backend_error)?;
        let count = capture
            .get(videoio::CAP_PROP_FRAME_COUNT)
            .map_err(to_backend_error)?;
        let total_frames = if count > 0.0 {
            Some(count as u32)
        } else {
            None
        };
        Ok(Self {
            width,
            height,
            fps,
            total_frames,
        })
    }
}

/// Iterates over the frames of a video file, decoding one `Mat` per call
/// to [`Iterator::next`] until the stream ends. Created by [`frames`].
pub struct FrameIterator {
    capture: VideoCapture,
}

impl Iterator for FrameIterator {
    type Item = crate::Result<Mat>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut frame = Mat::default();
        match self.capture.read(&mut frame) {
            Ok(true) => Some(Ok(frame)),
            Ok(false) => None,
            Err(e) => Some(Err(to_backend_error(e))),
        }
    }
}

/// Opens `path` and returns an iterator over its decoded frames, each as
/// a BGR [`Mat`] (matching [`crate::annotators`]'s OpenCV backend
/// convention).
pub fn frames(path: &Path) -> crate::Result<FrameIterator> {
    let capture = VideoCapture::from_file(path, videoio::CAP_ANY).map_err(to_backend_error)?;
    if !capture.is_opened().map_err(to_backend_error)? {
        return Err(Error::Backend(format!(
            "could not open video: {}",
            path.display()
        )));
    }
    Ok(FrameIterator { capture })
}

/// Writes frames to a new video file.
pub struct VideoSink {
    writer: VideoWriter,
}

impl VideoSink {
    /// Creates a new video file at `path`, matching `info`'s dimensions
    /// and frame rate, encoded with the FourCC codec `fourcc` (e.g.
    /// `('m', 'p', '4', 'v')` for MP4 — see [`VideoSink::mp4`]).
    pub fn new(
        path: &Path,
        info: VideoInfo,
        fourcc: (char, char, char, char),
    ) -> crate::Result<Self> {
        let (c1, c2, c3, c4) = fourcc;
        let code = VideoWriter::fourcc(c1, c2, c3, c4).map_err(to_backend_error)?;
        let size = Size::new(info.width as i32, info.height as i32);
        let writer =
            VideoWriter::new(path, code, info.fps, size, true).map_err(to_backend_error)?;
        if !writer.is_opened().map_err(to_backend_error)? {
            return Err(Error::Backend(format!(
                "could not open video for writing: {}",
                path.display()
            )));
        }
        Ok(Self { writer })
    }

    /// A sink using the widely-supported `mp4v` codec.
    pub fn mp4(path: &Path, info: VideoInfo) -> crate::Result<Self> {
        Self::new(path, info, ('m', 'p', '4', 'v'))
    }

    /// Appends one frame to the video.
    pub fn write_frame(&mut self, frame: &Mat) -> crate::Result<()> {
        self.writer.write(frame).map_err(to_backend_error)?;
        Ok(())
    }

    /// Finalizes and closes the video file.
    ///
    /// Dropping a `VideoSink` without calling this also releases the
    /// underlying writer, but only this method surfaces a write/flush
    /// error instead of silently discarding it.
    pub fn release(mut self) -> crate::Result<()> {
        self.writer.release().map_err(to_backend_error)
    }
}
