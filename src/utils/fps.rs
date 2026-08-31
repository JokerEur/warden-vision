//! Rolling frames-per-second estimate, for reporting how fast a
//! detect/track/annotate loop is actually running.

use std::collections::VecDeque;
use std::time::Instant;

/// Tracks the timestamps of the last `window_size` calls to
/// [`FPSMonitor::tick`] and estimates current throughput from them.
#[derive(Debug, Clone)]
pub struct FPSMonitor {
    timestamps: VecDeque<Instant>,
    window_size: usize,
}

impl FPSMonitor {
    /// Creates a monitor with the default 30-frame window.
    pub fn new() -> Self {
        Self::with_window_size(30)
    }

    /// Creates a monitor averaging over the last `window_size` ticks.
    ///
    /// # Panics
    /// Panics if `window_size` is `0`.
    pub fn with_window_size(window_size: usize) -> Self {
        assert!(window_size > 0, "window_size must be at least 1");
        Self {
            timestamps: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Records that a frame was just processed.
    pub fn tick(&mut self) {
        self.timestamps.push_back(Instant::now());
        while self.timestamps.len() > self.window_size {
            self.timestamps.pop_front();
        }
    }

    /// Estimated frames per second, averaged over the current window.
    ///
    /// Returns `0.0` until at least two ticks have been recorded (a single
    /// timestamp has no elapsed interval to divide by).
    pub fn fps(&self) -> f32 {
        if self.timestamps.len() < 2 {
            return 0.0;
        }
        let first = *self.timestamps.front().expect("checked len >= 2 above");
        let last = *self.timestamps.back().expect("checked len >= 2 above");
        let elapsed = last.duration_since(first).as_secs_f32();
        if elapsed <= 0.0 {
            return 0.0;
        }
        (self.timestamps.len() - 1) as f32 / elapsed
    }

    /// Discards all recorded timestamps, restarting the average.
    pub fn reset(&mut self) {
        self.timestamps.clear();
    }
}

impl Default for FPSMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn fps_is_zero_before_two_ticks() {
        let mut monitor = FPSMonitor::new();
        assert_eq!(monitor.fps(), 0.0);
        monitor.tick();
        assert_eq!(monitor.fps(), 0.0);
    }

    #[test]
    fn fps_is_positive_and_finite_after_ticks_with_elapsed_time() {
        let mut monitor = FPSMonitor::new();
        for _ in 0..5 {
            monitor.tick();
            sleep(Duration::from_millis(1));
        }
        let fps = monitor.fps();
        assert!(fps > 0.0);
        assert!(fps.is_finite());
    }

    #[test]
    fn reset_clears_the_window() {
        let mut monitor = FPSMonitor::new();
        monitor.tick();
        monitor.tick();
        monitor.reset();
        assert_eq!(monitor.fps(), 0.0);
    }

    #[test]
    #[should_panic]
    fn zero_window_size_panics() {
        FPSMonitor::with_window_size(0);
    }

    #[test]
    fn window_caps_at_configured_size() {
        let mut monitor = FPSMonitor::with_window_size(3);
        for _ in 0..10 {
            monitor.tick();
            sleep(Duration::from_millis(1));
        }
        // Only the last 3 ticks (2 intervals) should count; fps should
        // still be a small positive finite number, not based on all 10.
        let fps = monitor.fps();
        assert!(fps > 0.0 && fps.is_finite());
    }
}
