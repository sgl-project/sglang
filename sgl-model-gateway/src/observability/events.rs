//! Request events for observability and monitoring.
//!
//! Events use conditional log levels:
//! - DEBUG when OTEL is disabled (keeps logs quiet)
//! - INFO when OTEL is enabled (passes through EnvFilter to OTEL layer)

use tracing::{debug, event, Level};

use super::otel_trace::is_otel_enabled;

/// Module path used by CustomOtelFilter to identify events for OTEL export.
pub fn get_module_path() -> &'static str {
    module_path!()
}

/// Trait for emitting observability events.
pub trait Event {
    fn emit(&self);
}

/// Event emitted when a prefill-decode request pair is sent.
#[derive(Debug)]
pub struct RequestPDSentEvent {
    pub prefill_url: String,
    pub decode_url: String,
}

impl Event for RequestPDSentEvent {
    fn emit(&self) {
        if is_otel_enabled() {
            event!(
                Level::INFO,
                prefill_url = %self.prefill_url,
                decode_url = %self.decode_url,
                "Sending concurrent requests"
            );
        } else {
            debug!(
                prefill_url = %self.prefill_url,
                decode_url = %self.decode_url,
                "Sending concurrent requests"
            );
        }
    }
}

/// Event emitted when a request is sent to a worker.
#[derive(Debug)]
pub struct RequestSentEvent {
    pub url: String,
}

impl Event for RequestSentEvent {
    fn emit(&self) {
        if is_otel_enabled() {
            event!(Level::INFO, url = %self.url, "Sending request");
        } else {
            debug!(url = %self.url, "Sending request");
        }
    }
}

/// Event emitted when concurrent requests are received.
#[derive(Debug)]
pub struct RequestReceivedEvent;

impl Event for RequestReceivedEvent {
    fn emit(&self) {
        if is_otel_enabled() {
            event!(Level::INFO, "Received concurrent requests");
        } else {
            debug!("Received concurrent requests");
        }
    }
}
