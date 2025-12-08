//! request events for observability and monitoring

use tracing::{debug, event, Level};

use crate::observability::otel_trace::is_otel_enabled;

pub fn get_module_path() -> &'static str {
    module_path!()
}

pub trait Event {
    fn emit(&self);
}

#[derive(Debug)]
pub struct RequestPDSentEvent {
    pub prefill_url: String,
    pub decode_url: String,
}

impl Event for RequestPDSentEvent {
    fn emit(&self) {
        if !is_otel_enabled() {
            debug!(
                "Sending concurrent requests to prefill={} decode={}",
                self.prefill_url, self.decode_url
            );
        } else {
            event!(
                Level::INFO,
                prefill_url = %self.prefill_url,
                decode_url = %self.decode_url,
                "Sending concurrent requests"
            );
        }
    }
}

#[derive(Debug)]
pub struct RequestSentEvent {
    pub url: String,
}

impl Event for RequestSentEvent {
    fn emit(&self) {
        if !is_otel_enabled() {
            debug!("Sending request to {}", self.url);
        } else {
            event!(
                Level::INFO,
                url = %self.url,
                "Sending requests"
            );
        }
    }
}

#[derive(Debug)]
pub struct RequestReceivedEvent {}

impl Event for RequestReceivedEvent {
    fn emit(&self) {
        if !is_otel_enabled() {
            debug!("Received concurrent requests");
        } else {
            event!(Level::INFO, "Received concurrent requests");
        }
    }
}
