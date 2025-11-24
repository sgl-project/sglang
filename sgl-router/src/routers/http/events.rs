//! request events for observability and monitoring

use tracing::{Level, event};

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
        event!(
            Level::DEBUG,
            prefill_url = %self.prefill_url,
            decode_url = %self.decode_url,
            "Sending concurrent requests"
        );
    }
}

#[derive(Debug)]
pub struct RequestSentEvent {
    pub url: String,
}

impl Event for RequestSentEvent {
    fn emit(&self) {
        event!(
            Level::DEBUG,
            url = %self.url,
            "Sending requests"
        );
    }
}

#[derive(Debug)]
pub struct RequestReceivedEvent {}

impl Event for RequestReceivedEvent {
    fn emit(&self) {
        event!(
            Level::DEBUG,
            "Received concurrent requests"
        );
    }
}
