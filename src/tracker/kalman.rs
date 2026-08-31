//! A constant-velocity Kalman filter over `[cx, cy, w, h]` bounding-box
//! parameters, used by [`SortTracker`](crate::tracker::SortTracker) to
//! predict where an existing track should be before matching it against new
//! detections.

use nalgebra::{SMatrix, SVector};

/// State: `[cx, cy, w, h, vcx, vcy, vw, vh]`.
type State = SVector<f32, 8>;
type StateCovariance = SMatrix<f32, 8, 8>;
/// Measurement: `[cx, cy, w, h]`.
type Measurement = SVector<f32, 4>;
type MeasurementMatrix = SMatrix<f32, 4, 8>;
type MeasurementCovariance = SMatrix<f32, 4, 4>;

/// Kalman filter tracking a single bounding box under a constant-velocity
/// assumption: center `(cx, cy)` and size `(w, h)` each have an associated
/// velocity, and one time step (one frame) elapses between calls to
/// [`predict`](KalmanBoxFilter::predict).
#[derive(Debug, Clone)]
pub struct KalmanBoxFilter {
    state: State,
    covariance: StateCovariance,
}

impl KalmanBoxFilter {
    /// Initializes a filter from an observed bounding box, with zero
    /// initial velocity and high uncertainty on the (unobserved) velocity
    /// components.
    pub fn new(bbox: [f32; 4]) -> Self {
        let (cx, cy, w, h) = bbox_to_measurement(bbox);

        let mut state = State::zeros();
        state[0] = cx;
        state[1] = cy;
        state[2] = w;
        state[3] = h;

        let mut covariance = StateCovariance::identity() * 10.0;
        for k in 4..8 {
            covariance[(k, k)] = 1000.0;
        }

        Self { state, covariance }
    }

    /// State transition matrix for one time step: position components
    /// advance by their velocity, velocities are held constant.
    fn transition_matrix() -> StateCovariance {
        let mut f = StateCovariance::identity();
        for k in 0..4 {
            f[(k, k + 4)] = 1.0;
        }
        f
    }

    /// Measurement matrix: extracts `[cx, cy, w, h]` from the full state.
    fn measurement_matrix() -> MeasurementMatrix {
        let mut h = MeasurementMatrix::zeros();
        for k in 0..4 {
            h[(k, k)] = 1.0;
        }
        h
    }

    /// Process noise covariance. Velocity components are noisier than
    /// position components, reflecting that acceleration is unmodeled.
    fn process_noise() -> StateCovariance {
        let mut q = StateCovariance::identity();
        for k in 0..4 {
            q[(k, k)] = 1.0;
        }
        for k in 4..8 {
            q[(k, k)] = 0.1;
        }
        q
    }

    /// Measurement noise covariance, reflecting detector localization
    /// error.
    fn measurement_noise() -> MeasurementCovariance {
        MeasurementCovariance::identity()
    }

    /// Advances the filter by one time step and returns the predicted
    /// bounding box.
    pub fn predict(&mut self) -> [f32; 4] {
        let f = Self::transition_matrix();
        self.state = f * self.state;
        self.covariance = f * self.covariance * f.transpose() + Self::process_noise();
        self.bbox()
    }

    /// Incorporates an observed bounding box, correcting the filter's
    /// state estimate.
    pub fn update(&mut self, bbox: [f32; 4]) {
        let (cx, cy, w, h) = bbox_to_measurement(bbox);
        let z = Measurement::new(cx, cy, w, h);

        let h_mat = Self::measurement_matrix();
        let innovation = z - h_mat * self.state;
        let innovation_covariance =
            h_mat * self.covariance * h_mat.transpose() + Self::measurement_noise();

        let Some(innovation_covariance_inv) = innovation_covariance.try_inverse() else {
            // Singular innovation covariance: skip the correction rather
            // than propagate NaNs into the state.
            return;
        };

        let kalman_gain = self.covariance * h_mat.transpose() * innovation_covariance_inv;
        self.state += kalman_gain * innovation;

        let identity = StateCovariance::identity();
        self.covariance = (identity - kalman_gain * h_mat) * self.covariance;
    }

    /// The filter's current bounding box estimate, as `[x1, y1, x2, y2]`.
    pub fn bbox(&self) -> [f32; 4] {
        measurement_to_bbox(self.state[0], self.state[1], self.state[2], self.state[3])
    }
}

fn bbox_to_measurement(bbox: [f32; 4]) -> (f32, f32, f32, f32) {
    let [x1, y1, x2, y2] = bbox;
    let w = (x2 - x1).max(0.0);
    let h = (y2 - y1).max(0.0);
    (x1 + w / 2.0, y1 + h / 2.0, w, h)
}

fn measurement_to_bbox(cx: f32, cy: f32, w: f32, h: f32) -> [f32; 4] {
    let w = w.max(0.0);
    let h = h.max(0.0);
    [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_bbox_matches_observation() {
        let kf = KalmanBoxFilter::new([0.0, 0.0, 10.0, 20.0]);
        let bbox = kf.bbox();
        assert!((bbox[0] - 0.0).abs() < 1e-4);
        assert!((bbox[1] - 0.0).abs() < 1e-4);
        assert!((bbox[2] - 10.0).abs() < 1e-4);
        assert!((bbox[3] - 20.0).abs() < 1e-4);
    }

    #[test]
    fn predict_with_zero_velocity_is_stationary() {
        let mut kf = KalmanBoxFilter::new([0.0, 0.0, 10.0, 10.0]);
        let predicted = kf.predict();
        for k in 0..4 {
            assert!((predicted[k] - [0.0, 0.0, 10.0, 10.0][k]).abs() < 1e-4);
        }
    }

    #[test]
    fn update_pulls_state_toward_observation() {
        let mut kf = KalmanBoxFilter::new([0.0, 0.0, 10.0, 10.0]);
        kf.predict();
        kf.update([4.0, 0.0, 14.0, 10.0]);
        let bbox = kf.bbox();
        // Should have moved from x1=0 toward the new observation at x1=4,
        // without jumping exactly onto it (partial correction).
        assert!(bbox[0] > 0.0);
        assert!(bbox[0] <= 4.0);
    }

    #[test]
    fn tracks_constant_velocity_motion() {
        let mut kf = KalmanBoxFilter::new([0.0, 0.0, 10.0, 10.0]);
        // Object moves 2px/frame to the right; feed 20 frames so the
        // filter's velocity estimate converges.
        for i in 1..=20 {
            kf.predict();
            let shift = (i * 2) as f32;
            kf.update([shift, 0.0, shift + 10.0, 10.0]);
        }
        let predicted = kf.predict();
        // Next true position would be at x1=42; filter should be close.
        assert!(
            (predicted[0] - 42.0).abs() < 5.0,
            "expected prediction near 42.0, got {}",
            predicted[0]
        );
    }
}
