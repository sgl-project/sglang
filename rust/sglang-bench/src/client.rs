//! Per-request HTTP + SSE streaming, mirroring `async_request_sglang_generate`.

use std::time::Instant;

use futures::StreamExt;

use crate::output::RequestOutput;
use crate::sse::{FrameAccumulator, FrameError, LineBuffer};

/// One benchmark request. `payload` is the pre-serialized `/generate` JSON
/// body built by the Python side, so payload semantics (sampling params,
/// extra_request_body, ...) never diverge between the two clients.
pub struct RequestSpec {
    pub payload: Vec<u8>,
    pub prompt_len: u64,
    pub output_len: i64,
    pub arrival_offset_s: f64,
    pub routing_key: Option<String>,
}

pub struct RunConfig {
    pub api_url: String,
    /// Shared headers (auth + `--header` extras), applied to every request.
    pub headers: Vec<(String, String)>,
    /// Header name for the per-request `routing_key` (`X-SMG-Routing-Key`).
    pub routing_key_header: String,
    pub max_concurrency: Option<usize>,
    pub cache_report: bool,
}

enum RequestError {
    /// Non-200 status; Python formats this as `"{reason}: {body}"`.
    Http {
        reason: String,
        body: String,
    },
    Transport(String),
    Frame(FrameError),
}

impl RequestError {
    fn into_message(self) -> String {
        match self {
            RequestError::Http { reason, body } => format!("{reason}: {body}"),
            RequestError::Transport(msg) => msg,
            RequestError::Frame(e) => e.to_string(),
        }
    }
}

/// Flatten a reqwest error and its source chain into one line (the Python
/// client stores a full traceback string; the chain carries the same info,
/// e.g. "error sending request: ...: Connection refused").
fn error_chain(e: &dyn std::error::Error) -> String {
    let mut msg = e.to_string();
    let mut source = e.source();
    while let Some(cause) = source {
        msg.push_str(": ");
        msg.push_str(&cause.to_string());
        source = cause.source();
    }
    msg
}

pub async fn run_one(
    client: &reqwest::Client,
    cfg: &RunConfig,
    spec: RequestSpec,
    anchor: Instant,
) -> RequestOutput {
    let RequestSpec {
        payload,
        prompt_len,
        output_len,
        arrival_offset_s: _,
        routing_key,
    } = spec;

    let st = Instant::now();
    let start_time = st.duration_since(anchor).as_secs_f64();
    let mut acc = FrameAccumulator::new(cfg.cache_report, output_len);
    let result = stream_generate(client, cfg, payload, routing_key.as_deref(), st, &mut acc).await;

    // On failure, partially-accumulated ttft/itl/spec/cache fields are kept
    // while generated_text/latency/output_len reset — exactly like the Python
    // client, which writes the former onto the output object mid-stream and
    // the latter only on success.
    let success = result.is_ok();
    RequestOutput {
        generated_text: if success {
            acc.generated_text
        } else {
            String::new()
        },
        success,
        latency: if success { acc.latency } else { 0.0 },
        ttft: acc.ttft,
        itl: acc.itl,
        prompt_len,
        error: result
            .err()
            .map(RequestError::into_message)
            .unwrap_or_default(),
        output_len: if success { acc.output_len } else { 0 },
        start_time,
        cached_tokens: acc.cached_tokens,
        cached_tokens_details_json: acc.cached_tokens_details_json,
        spec_accept_length: acc.spec_accept_length,
    }
}

async fn stream_generate(
    client: &reqwest::Client,
    cfg: &RunConfig,
    payload: Vec<u8>,
    routing_key: Option<&str>,
    st: Instant,
    acc: &mut FrameAccumulator,
) -> Result<(), RequestError> {
    let mut request = client
        .post(&cfg.api_url)
        .header("Content-Type", "application/json");
    for (name, value) in &cfg.headers {
        request = request.header(name.as_str(), value.as_str());
    }
    if let Some(key) = routing_key {
        request = request.header(cfg.routing_key_header.as_str(), key);
    }

    let response = request
        .body(payload)
        .send()
        .await
        .map_err(|e| RequestError::Transport(error_chain(&e)))?;

    let status = response.status();
    if status.as_u16() != 200 {
        return Err(RequestError::Http {
            reason: status.canonical_reason().unwrap_or_default().to_string(),
            body: response.text().await.unwrap_or_default(),
        });
    }

    let mut stream = response.bytes_stream();
    let mut buf = LineBuffer::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| RequestError::Transport(error_chain(&e)))?;
        buf.feed(&chunk, |line| acc.on_line(line, st.elapsed().as_secs_f64()))
            .map_err(RequestError::Frame)?;
    }
    if let Some(rest) = buf.take_remainder() {
        acc.on_line(&rest, st.elapsed().as_secs_f64())
            .map_err(RequestError::Frame)?;
    }
    Ok(())
}
