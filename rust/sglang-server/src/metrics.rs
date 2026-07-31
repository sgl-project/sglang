//! Low-cardinality Prometheus metrics for the embedded Rust frontend.
//!
//! The Python scheduler owns the GPU/runtime metrics. This module only tracks the
//! Rust api-server/tokenizer-manager boundary with cheap atomics, then renders a
//! fixed Prometheus text exposition for `/metrics`.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum HttpEndpointLabel {
    Generate = 0,
    Health,
    HealthGenerate,
    ServerInfo,
    GetModelInfo,
    ModelInfo,
    V1Models,
    Other,
}

impl HttpEndpointLabel {
    const COUNT: usize = 8;
    const ALL: [Self; Self::COUNT] = [
        Self::Generate,
        Self::Health,
        Self::HealthGenerate,
        Self::ServerInfo,
        Self::GetModelInfo,
        Self::ModelInfo,
        Self::V1Models,
        Self::Other,
    ];

    pub fn from_path(path: &str) -> Option<Self> {
        match path {
            "/generate" => Some(Self::Generate),
            "/health" => Some(Self::Health),
            "/health_generate" => Some(Self::HealthGenerate),
            "/server_info" => Some(Self::ServerInfo),
            "/get_model_info" => Some(Self::GetModelInfo),
            "/model_info" => Some(Self::ModelInfo),
            "/v1/models" => Some(Self::V1Models),
            // Do not let a scrape perturb the values it is reading.
            "/metrics" => None,
            _ => Some(Self::Other),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Generate => "/generate",
            Self::Health => "/health",
            Self::HealthGenerate => "/health_generate",
            Self::ServerInfo => "/server_info",
            Self::GetModelInfo => "/get_model_info",
            Self::ModelInfo => "/model_info",
            Self::V1Models => "/v1/models",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum HttpMethodLabel {
    Get = 0,
    Post,
    Other,
}

impl HttpMethodLabel {
    const COUNT: usize = 3;
    const ALL: [Self; Self::COUNT] = [Self::Get, Self::Post, Self::Other];

    pub fn from_method(method: &axum::http::Method) -> Self {
        match *method {
            axum::http::Method::GET => Self::Get,
            axum::http::Method::POST => Self::Post,
            _ => Self::Other,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Other => "OTHER",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
enum HttpStatusLabel {
    Ok = 0,
    BadRequest,
    NotFound,
    MethodNotAllowed,
    ClientClosed,
    InternalServerError,
    ServiceUnavailable,
    Other,
}

impl HttpStatusLabel {
    const COUNT: usize = 8;
    const ALL: [Self; Self::COUNT] = [
        Self::Ok,
        Self::BadRequest,
        Self::NotFound,
        Self::MethodNotAllowed,
        Self::ClientClosed,
        Self::InternalServerError,
        Self::ServiceUnavailable,
        Self::Other,
    ];

    fn from_u16(code: u16) -> Self {
        match code {
            200 => Self::Ok,
            400 => Self::BadRequest,
            404 => Self::NotFound,
            405 => Self::MethodNotAllowed,
            499 => Self::ClientClosed,
            500 => Self::InternalServerError,
            503 => Self::ServiceUnavailable,
            _ => Self::Other,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Ok => "200",
            Self::BadRequest => "400",
            Self::NotFound => "404",
            Self::MethodNotAllowed => "405",
            Self::ClientClosed => "499",
            Self::InternalServerError => "500",
            Self::ServiceUnavailable => "503",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum RequestKindLabel {
    Generate = 0,
    HealthGenerate,
    Control,
}

impl RequestKindLabel {
    const COUNT: usize = 3;
    const ALL: [Self; Self::COUNT] = [Self::Generate, Self::HealthGenerate, Self::Control];

    pub fn label(self) -> &'static str {
        match self {
            Self::Generate => "generate",
            Self::HealthGenerate => "health_generate",
            Self::Control => "control",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum InputSourceLabel {
    Text = 0,
    InputIds,
    Control,
}

impl InputSourceLabel {
    const COUNT: usize = 3;
    const ALL: [Self; Self::COUNT] = [Self::Text, Self::InputIds, Self::Control];

    fn label(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::InputIds => "input_ids",
            Self::Control => "control",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum RequestStageLabel {
    Parse = 0,
    Validate,
    Normalize,
    Tokenize,
    PreSend,
    Queue,
    Submit,
    Egress,
    Detokenize,
}

impl RequestStageLabel {
    const COUNT: usize = 9;
    const ALL: [Self; Self::COUNT] = [
        Self::Parse,
        Self::Validate,
        Self::Normalize,
        Self::Tokenize,
        Self::PreSend,
        Self::Queue,
        Self::Submit,
        Self::Egress,
        Self::Detokenize,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Parse => "parse",
            Self::Validate => "validate",
            Self::Normalize => "normalize",
            Self::Tokenize => "tokenize",
            Self::PreSend => "pre_send",
            Self::Queue => "queue",
            Self::Submit => "submit",
            Self::Egress => "egress",
            Self::Detokenize => "detokenize",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum TerminalOutcomeLabel {
    Success = 0,
    Error,
    Disconnect,
    ClientBackpressure,
    ControlResult,
}

impl TerminalOutcomeLabel {
    const COUNT: usize = 5;
    const ALL: [Self; Self::COUNT] = [
        Self::Success,
        Self::Error,
        Self::Disconnect,
        Self::ClientBackpressure,
        Self::ControlResult,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Error => "error",
            Self::Disconnect => "disconnect",
            Self::ClientBackpressure => "client_backpressure",
            Self::ControlResult => "control_result",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum EgressFrameLabel {
    Batch = 0,
    Result,
    Error,
    BadBatch,
    Unknown,
}

impl EgressFrameLabel {
    const COUNT: usize = 5;
    const ALL: [Self; Self::COUNT] = [
        Self::Batch,
        Self::Result,
        Self::Error,
        Self::BadBatch,
        Self::Unknown,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::Result => "result",
            Self::Error => "error",
            Self::BadBatch => "bad_batch",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum RingLabel {
    Ingress = 0,
    Egress,
    Channel,
}

impl RingLabel {
    const COUNT: usize = 3;
    const ALL: [Self; Self::COUNT] = [Self::Ingress, Self::Egress, Self::Channel];

    fn label(self) -> &'static str {
        match self {
            Self::Ingress => "ingress",
            Self::Egress => "egress",
            Self::Channel => "channel",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum ThreadPoolLabel {
    Api = 0,
    Tokenizer,
    Detokenizer,
    TmIngress,
    TmEgress,
}

impl ThreadPoolLabel {
    const COUNT: usize = 5;
    const ALL: [Self; Self::COUNT] = [
        Self::Api,
        Self::Tokenizer,
        Self::Detokenizer,
        Self::TmIngress,
        Self::TmEgress,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Api => "api",
            Self::Tokenizer => "tokenizer",
            Self::Detokenizer => "detokenizer",
            Self::TmIngress => "tm_ingress",
            Self::TmEgress => "tm_egress",
        }
    }
}

const HTTP_REQUEST_CARDINALITY: usize = HttpEndpointLabel::COUNT * HttpMethodLabel::COUNT;
const HTTP_RESPONSE_CARDINALITY: usize =
    HttpEndpointLabel::COUNT * HttpMethodLabel::COUNT * HttpStatusLabel::COUNT;
const RUST_REQUEST_CARDINALITY: usize = RequestKindLabel::COUNT * InputSourceLabel::COUNT * 2;
const REQUEST_ERROR_CARDINALITY: usize = RequestStageLabel::COUNT * HttpStatusLabel::COUNT;
const TERMINAL_CARDINALITY: usize = RequestKindLabel::COUNT * TerminalOutcomeLabel::COUNT;

pub struct MetricsState {
    http_requests_total: [AtomicU64; HTTP_REQUEST_CARDINALITY],
    http_responses_total: [AtomicU64; HTTP_RESPONSE_CARDINALITY],
    http_requests_active: [AtomicU64; HTTP_REQUEST_CARDINALITY],
    rust_server_requests_total: [AtomicU64; RUST_REQUEST_CARDINALITY],
    request_errors_total: [AtomicU64; REQUEST_ERROR_CARDINALITY],
    request_terminal_total: [AtomicU64; TERMINAL_CARDINALITY],
    inflight_requests: [AtomicU64; RequestKindLabel::COUNT],
    ingress_ring_push_total: [AtomicU64; RequestKindLabel::COUNT],
    ingress_ring_backpressure_total: [AtomicU64; RequestKindLabel::COUNT],
    ingress_ring_depth: AtomicU64,
    egress_ring_push_total: [AtomicU64; EgressFrameLabel::COUNT],
    egress_ring_backpressure_total: [AtomicU64; EgressFrameLabel::COUNT],
    egress_ring_depth: AtomicU64,
    egress_frames_total: [AtomicU64; EgressFrameLabel::COUNT],
    ring_capacity: [AtomicU64; RingLabel::COUNT],
    threads: [AtomicU64; ThreadPoolLabel::COUNT],
}

impl Default for MetricsState {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsState {
    pub fn new() -> Self {
        Self {
            http_requests_total: atomic_array(),
            http_responses_total: atomic_array(),
            http_requests_active: atomic_array(),
            rust_server_requests_total: atomic_array(),
            request_errors_total: atomic_array(),
            request_terminal_total: atomic_array(),
            inflight_requests: atomic_array(),
            ingress_ring_push_total: atomic_array(),
            ingress_ring_backpressure_total: atomic_array(),
            ingress_ring_depth: AtomicU64::new(0),
            egress_ring_push_total: atomic_array(),
            egress_ring_backpressure_total: atomic_array(),
            egress_ring_depth: AtomicU64::new(0),
            egress_frames_total: atomic_array(),
            ring_capacity: atomic_array(),
            threads: atomic_array(),
        }
    }

    #[inline]
    pub fn http_request_started(&self, endpoint: HttpEndpointLabel, method: HttpMethodLabel) {
        self.http_requests_total[http_idx(endpoint, method)].fetch_add(1, Ordering::Relaxed);
        self.http_requests_active[http_idx(endpoint, method)].fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn http_request_finished(
        &self,
        endpoint: HttpEndpointLabel,
        method: HttpMethodLabel,
        status_code: u16,
    ) {
        let status = HttpStatusLabel::from_u16(status_code);
        self.http_responses_total[http_response_idx(endpoint, method, status)]
            .fetch_add(1, Ordering::Relaxed);
        saturating_sub(&self.http_requests_active[http_idx(endpoint, method)], 1);
    }

    #[inline]
    pub fn request_received(
        &self,
        kind: RequestKindLabel,
        input_source: InputSourceLabel,
        stream: bool,
    ) {
        self.rust_server_requests_total[request_idx(kind, input_source, stream)]
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn request_error(&self, stage: RequestStageLabel, status_code: u16) {
        let status = HttpStatusLabel::from_u16(status_code);
        self.request_errors_total[request_error_idx(stage, status)].fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn request_terminal(&self, kind: RequestKindLabel, outcome: TerminalOutcomeLabel) {
        self.request_terminal_total[terminal_idx(kind, outcome)].fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inflight_inc(&self, kind: RequestKindLabel) {
        self.inflight_requests[kind as usize].fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn inflight_dec(&self, kind: RequestKindLabel) {
        saturating_sub(&self.inflight_requests[kind as usize], 1);
    }

    #[inline]
    pub fn ingress_ring_push(&self, kind: RequestKindLabel) {
        self.ingress_ring_push_total[kind as usize].fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn ingress_ring_backpressure(&self, kind: RequestKindLabel) {
        self.ingress_ring_backpressure_total[kind as usize].fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn ingress_depth_inc(&self) {
        self.ingress_ring_depth.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn ingress_depth_dec(&self, n: u64) {
        saturating_sub(&self.ingress_ring_depth, n);
    }

    #[inline]
    pub fn egress_ring_push(&self, frame: EgressFrameLabel) {
        self.egress_ring_push_total[frame as usize].fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn egress_ring_backpressure(&self, frame: EgressFrameLabel) {
        self.egress_ring_backpressure_total[frame as usize].fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn egress_depth_inc(&self) {
        self.egress_ring_depth.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn egress_depth_dec(&self, n: u64) {
        saturating_sub(&self.egress_ring_depth, n);
    }

    #[inline]
    pub fn egress_frame(&self, frame: EgressFrameLabel) {
        self.egress_frames_total[frame as usize].fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn set_ring_capacity(&self, ring: RingLabel, value: u64) {
        self.ring_capacity[ring as usize].store(value, Ordering::Relaxed);
    }

    #[inline]
    pub fn set_threads(&self, pool: ThreadPoolLabel, value: u64) {
        self.threads[pool as usize].store(value, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub fn ingress_depth(&self) -> u64 {
        self.ingress_ring_depth.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub fn egress_depth(&self) -> u64 {
        self.egress_ring_depth.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub fn inflight_requests(&self, kind: RequestKindLabel) -> u64 {
        self.inflight_requests[kind as usize].load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub fn terminal_total(&self, kind: RequestKindLabel, outcome: TerminalOutcomeLabel) -> u64 {
        self.request_terminal_total[terminal_idx(kind, outcome)].load(Ordering::Relaxed)
    }

    pub fn render_prometheus(&self) -> String {
        let mut out = String::with_capacity(8192);
        append_header(
            &mut out,
            "sglang:http_requests_total",
            "Total number of HTTP requests by endpoint and method.",
            "counter",
        );
        for endpoint in HttpEndpointLabel::ALL {
            for method in HttpMethodLabel::ALL {
                append_counter_sample(
                    &mut out,
                    "sglang:http_requests_total",
                    &[("endpoint", endpoint.label()), ("method", method.label())],
                    self.http_requests_total[http_idx(endpoint, method)].load(Ordering::Relaxed),
                );
            }
        }

        append_header(
            &mut out,
            "sglang:http_responses_total",
            "Total number of HTTP responses by endpoint, status code, and method.",
            "counter",
        );
        for endpoint in HttpEndpointLabel::ALL {
            for method in HttpMethodLabel::ALL {
                for status in HttpStatusLabel::ALL {
                    append_counter_sample(
                        &mut out,
                        "sglang:http_responses_total",
                        &[
                            ("endpoint", endpoint.label()),
                            ("status_code", status.label()),
                            ("method", method.label()),
                        ],
                        self.http_responses_total[http_response_idx(endpoint, method, status)]
                            .load(Ordering::Relaxed),
                    );
                }
            }
        }

        append_header(
            &mut out,
            "sglang:http_requests_active",
            "Number of currently active HTTP requests.",
            "gauge",
        );
        for endpoint in HttpEndpointLabel::ALL {
            for method in HttpMethodLabel::ALL {
                append_sample(
                    &mut out,
                    "sglang:http_requests_active",
                    &[("endpoint", endpoint.label()), ("method", method.label())],
                    self.http_requests_active[http_idx(endpoint, method)].load(Ordering::Relaxed),
                );
            }
        }

        append_header(
            &mut out,
            "sglang:rust_server_requests_total",
            "Total number of Rust frontend logical requests.",
            "counter",
        );
        for kind in RequestKindLabel::ALL {
            for source in InputSourceLabel::ALL {
                for stream in [false, true] {
                    append_counter_sample(
                        &mut out,
                        "sglang:rust_server_requests_total",
                        &[
                            ("kind", kind.label()),
                            ("input_source", source.label()),
                            ("stream", if stream { "true" } else { "false" }),
                        ],
                        self.rust_server_requests_total[request_idx(kind, source, stream)]
                            .load(Ordering::Relaxed),
                    );
                }
            }
        }

        append_header(
            &mut out,
            "sglang:rust_server_request_errors_total",
            "Total number of Rust frontend request errors by stage and status code.",
            "counter",
        );
        for stage in RequestStageLabel::ALL {
            for status in HttpStatusLabel::ALL {
                append_counter_sample(
                    &mut out,
                    "sglang:rust_server_request_errors_total",
                    &[("stage", stage.label()), ("status_code", status.label())],
                    self.request_errors_total[request_error_idx(stage, status)]
                        .load(Ordering::Relaxed),
                );
            }
        }

        append_header(
            &mut out,
            "sglang:rust_server_request_terminal_total",
            "Total number of Rust frontend terminal request outcomes.",
            "counter",
        );
        for kind in RequestKindLabel::ALL {
            for outcome in TerminalOutcomeLabel::ALL {
                append_counter_sample(
                    &mut out,
                    "sglang:rust_server_request_terminal_total",
                    &[("kind", kind.label()), ("outcome", outcome.label())],
                    self.request_terminal_total[terminal_idx(kind, outcome)]
                        .load(Ordering::Relaxed),
                );
            }
        }

        append_header(
            &mut out,
            "sglang:rust_server_inflight_requests",
            "Number of Rust frontend requests registered with detokenizer shards.",
            "gauge",
        );
        for kind in RequestKindLabel::ALL {
            append_sample(
                &mut out,
                "sglang:rust_server_inflight_requests",
                &[("kind", kind.label())],
                self.inflight_requests[kind as usize].load(Ordering::Relaxed),
            );
        }

        append_by_kind_counter(
            &mut out,
            "sglang:rust_server_ingress_ring_push_total",
            "Total number of successful pushes into the Rust ingress ring.",
            &self.ingress_ring_push_total,
        );
        append_by_kind_counter(
            &mut out,
            "sglang:rust_server_ingress_ring_backpressure_total",
            "Total number of ingress ring backpressure events.",
            &self.ingress_ring_backpressure_total,
        );

        append_header(
            &mut out,
            "sglang:rust_server_ingress_ring_depth",
            "Current number of messages queued in the Rust ingress ring.",
            "gauge",
        );
        append_sample(
            &mut out,
            "sglang:rust_server_ingress_ring_depth",
            &[],
            self.ingress_ring_depth.load(Ordering::Relaxed),
        );

        append_by_frame_counter(
            &mut out,
            "sglang:rust_server_egress_ring_push_total",
            "Total number of successful pushes into the Rust egress ring.",
            &self.egress_ring_push_total,
        );
        append_by_frame_counter(
            &mut out,
            "sglang:rust_server_egress_ring_backpressure_total",
            "Total number of egress ring backpressure events.",
            &self.egress_ring_backpressure_total,
        );

        append_header(
            &mut out,
            "sglang:rust_server_egress_ring_depth",
            "Current number of messages queued in the Rust egress ring.",
            "gauge",
        );
        append_sample(
            &mut out,
            "sglang:rust_server_egress_ring_depth",
            &[],
            self.egress_ring_depth.load(Ordering::Relaxed),
        );

        append_by_frame_counter(
            &mut out,
            "sglang:rust_server_egress_frames_total",
            "Total number of Rust egress frames drained by frame type.",
            &self.egress_frames_total,
        );

        append_header(
            &mut out,
            "sglang:rust_server_ring_capacity",
            "Configured Rust server ring or channel capacity.",
            "gauge",
        );
        for ring in RingLabel::ALL {
            append_sample(
                &mut out,
                "sglang:rust_server_ring_capacity",
                &[("ring", ring.label())],
                self.ring_capacity[ring as usize].load(Ordering::Relaxed),
            );
        }

        append_header(
            &mut out,
            "sglang:rust_server_threads",
            "Configured Rust server worker threads by pool.",
            "gauge",
        );
        for pool in ThreadPoolLabel::ALL {
            append_sample(
                &mut out,
                "sglang:rust_server_threads",
                &[("pool", pool.label())],
                self.threads[pool as usize].load(Ordering::Relaxed),
            );
        }

        out
    }
}

fn atomic_array<const N: usize>() -> [AtomicU64; N] {
    std::array::from_fn(|_| AtomicU64::new(0))
}

#[inline]
fn saturating_sub(value: &AtomicU64, delta: u64) {
    let _ = value.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_sub(delta))
    });
}

#[inline]
const fn http_idx(endpoint: HttpEndpointLabel, method: HttpMethodLabel) -> usize {
    (endpoint as usize) * HttpMethodLabel::COUNT + method as usize
}

#[inline]
const fn http_response_idx(
    endpoint: HttpEndpointLabel,
    method: HttpMethodLabel,
    status: HttpStatusLabel,
) -> usize {
    ((endpoint as usize) * HttpMethodLabel::COUNT + method as usize) * HttpStatusLabel::COUNT
        + status as usize
}

#[inline]
const fn request_idx(
    kind: RequestKindLabel,
    input_source: InputSourceLabel,
    stream: bool,
) -> usize {
    ((kind as usize) * InputSourceLabel::COUNT + input_source as usize) * 2 + stream as usize
}

#[inline]
const fn request_error_idx(stage: RequestStageLabel, status: HttpStatusLabel) -> usize {
    (stage as usize) * HttpStatusLabel::COUNT + status as usize
}

#[inline]
const fn terminal_idx(kind: RequestKindLabel, outcome: TerminalOutcomeLabel) -> usize {
    (kind as usize) * TerminalOutcomeLabel::COUNT + outcome as usize
}

fn append_by_kind_counter(
    out: &mut String,
    name: &'static str,
    help: &'static str,
    values: &[AtomicU64; RequestKindLabel::COUNT],
) {
    append_header(out, name, help, "counter");
    for kind in RequestKindLabel::ALL {
        append_counter_sample(
            out,
            name,
            &[("kind", kind.label())],
            values[kind as usize].load(Ordering::Relaxed),
        );
    }
}

fn append_by_frame_counter(
    out: &mut String,
    name: &'static str,
    help: &'static str,
    values: &[AtomicU64; EgressFrameLabel::COUNT],
) {
    append_header(out, name, help, "counter");
    for frame in EgressFrameLabel::ALL {
        append_counter_sample(
            out,
            name,
            &[("frame", frame.label())],
            values[frame as usize].load(Ordering::Relaxed),
        );
    }
}

fn append_header(out: &mut String, name: &str, help: &str, metric_type: &str) {
    out.push_str("# HELP ");
    out.push_str(name);
    out.push(' ');
    out.push_str(help);
    out.push('\n');
    out.push_str("# TYPE ");
    out.push_str(name);
    out.push(' ');
    out.push_str(metric_type);
    out.push('\n');
}

fn append_counter_sample(out: &mut String, name: &str, labels: &[(&str, &str)], value: u64) {
    if value > 0 {
        append_sample(out, name, labels, value);
    }
}

fn append_sample(out: &mut String, name: &str, labels: &[(&str, &str)], value: u64) {
    out.push_str(name);
    if !labels.is_empty() {
        out.push('{');
        for (i, (k, v)) in labels.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(k);
            out.push_str("=\"");
            append_escaped_label_value(out, v);
            out.push('"');
        }
        out.push('}');
    }
    out.push(' ');
    out.push_str(&value.to_string());
    out.push('\n');
}

fn append_escaped_label_value(out: &mut String, value: &str) {
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            _ => out.push(ch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_values_are_escaped() {
        let mut out = String::new();
        append_sample(&mut out, "m", &[("label", "quote\" slash\\ newline\n")], 1);
        assert_eq!(out, "m{label=\"quote\\\" slash\\\\ newline\\n\"} 1\n");
    }

    #[test]
    fn counters_and_gauges_render_prometheus_text() {
        let metrics = MetricsState::new();
        metrics.request_received(RequestKindLabel::Generate, InputSourceLabel::Text, false);
        metrics.request_error(RequestStageLabel::Validate, 400);
        metrics.inflight_inc(RequestKindLabel::Generate);
        metrics.set_ring_capacity(RingLabel::Ingress, 7);
        metrics.set_threads(ThreadPoolLabel::Api, 2);

        let text = metrics.render_prometheus();
        assert!(text.contains("# HELP sglang:rust_server_requests_total"));
        assert!(text.contains(
            "sglang:rust_server_requests_total{kind=\"generate\",input_source=\"text\",stream=\"false\"} 1"
        ));
        assert!(text.contains(
            "sglang:rust_server_request_errors_total{stage=\"validate\",status_code=\"400\"} 1"
        ));
        assert!(text.contains("sglang:rust_server_inflight_requests{kind=\"generate\"} 1"));
        assert!(text.contains("sglang:rust_server_ring_capacity{ring=\"ingress\"} 7"));
        assert!(text.contains("sglang:rust_server_threads{pool=\"api\"} 2"));
    }

    #[test]
    fn gauges_saturate_at_zero() {
        let metrics = MetricsState::new();
        metrics.inflight_dec(RequestKindLabel::Generate);
        metrics.ingress_depth_dec(1);
        metrics.egress_depth_dec(1);
        assert_eq!(metrics.inflight_requests(RequestKindLabel::Generate), 0);
        assert_eq!(metrics.ingress_depth(), 0);
        assert_eq!(metrics.egress_depth(), 0);
    }
}
