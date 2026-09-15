//! Preserve request age when DP ingress and its worker have different clock origins.

use axum::http::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};

const RECEIVED_TIMING_HEADER: &str = "x-sglang-received-timing";

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ForwardedReceivedTiming {
    pub age: f64,
    pub sent_at: f64,
    pub created_is_positive: bool,
}

impl ForwardedReceivedTiming {
    pub fn forward(headers: &mut HeaderMap, received_time: Option<f64>) -> Result<(), String> {
        // Only ingress can derive the age from the client's monotonic timestamp.
        headers.remove(RECEIVED_TIMING_HEADER);
        let Some(received) = received_time.filter(|value| *value != 0.) else {
            return Ok(());
        };
        let timing = Self {
            age: crate::metrics::monotonic_seconds()? - received,
            sent_at: crate::metrics::realtime_seconds(),
            created_is_positive: received > 0.,
        };
        let encoded = serde_json::to_string(&timing).map_err(|error| error.to_string())?;
        let value = HeaderValue::from_str(&encoded).map_err(|error| error.to_string())?;
        headers.insert(RECEIVED_TIMING_HEADER, value);
        Ok(())
    }

    pub fn from_headers(headers: &HeaderMap) -> Result<Option<Self>, String> {
        headers
            .get(RECEIVED_TIMING_HEADER)
            .map(|value| {
                serde_json::from_slice(value.as_bytes())
                    .map_err(|error| format!("invalid {RECEIVED_TIMING_HEADER} header: {error}"))
            })
            .transpose()
    }

    pub fn age_at(self, wall_time: f64) -> Result<f64, String> {
        // Wall clocks correlate the handoff across hosts; subsequent durations
        // remain on the worker's monotonic clock.
        let age = self.age + (wall_time - self.sent_at);
        if !age.is_finite() {
            return Err("forwarded request age is not finite".into());
        }
        Ok(age)
    }
}
