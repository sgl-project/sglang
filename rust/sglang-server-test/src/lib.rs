//! A whole `sglang-server` frontend with no scheduler behind it: the test
//! reads what reaches the scheduler and answers for it. For tests of a model
//! package's extensions and for end-to-end tests of the server itself, over
//! real sockets.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

pub use sglang_server::test_utils::{
    ApiKeys, CapturedLogs, PreGenerateAborts, PreferredSamplingParams, capture_logs,
    pre_generate_app,
};
use sglang_server::test_utils::{
    BatchHeader, DeploymentInfo, Runtime, RuntimeConfig, RustServerEndpoint, frame_control_result,
    frame_decode_batch_cols, frame_error, start,
};
use sglang_server::{
    AbortHook, GenerateHook, GenerateValidator, HttpLayer, MmProcessor, RustServerServerArgs,
    ServerArgs,
};

/// `N` distinct free loopback addresses. All are probed before any is
/// released, so the OS cannot hand out the same ephemeral port twice.
pub fn free_loopback_addrs<const N: usize>() -> [SocketAddr; N] {
    let probes: [std::net::TcpListener; N] =
        std::array::from_fn(|_| std::net::TcpListener::bind("127.0.0.1:0").expect("bind"));
    probes
        .each_ref()
        .map(|probe| probe.local_addr().expect("local addr"))
}

/// A frontend serving HTTP on one address, whose scheduler is the test.
pub struct SchedulerlessServer(Runtime);

/// The package extensions a [`SchedulerlessServer`] serves with.
#[derive(Default)]
pub struct Extensions {
    pub generate_hook: Option<Arc<dyn GenerateHook>>,
    pub abort_hook: Option<Arc<dyn AbortHook>>,
}

/// One message that reached the scheduler: its type (`AbortReq`,
/// `TokenizedGenerateReqInput`, ...) and the rid it carries.
#[derive(Debug)]
pub struct SchedulerMessage {
    pub kind: String,
    pub rid: String,
}

/// What one scheduler output says about its request beside the tokens. A
/// field left at its default is absent from the wire, as the scheduler
/// leaves it.
#[derive(Debug, Default)]
pub struct OutputDetails {
    pub prompt_tokens: u32,
    /// Running total, as the scheduler sends it.
    pub cached_tokens: u32,
    pub reasoning_tokens: u64,
    /// Python's `FINISH_*.to_json()`; the request's end.
    pub finish_reason: Option<serde_json::Value>,
    /// The scheduler ended the row outside the output stream.
    pub abort: bool,
    /// `{"device", "host", "storage"}`.
    pub cached_tokens_split: Option<serde_json::Value>,
    pub storage_backend: Option<String>,
    pub decode_latency: Option<f64>,
    /// `{key: [values, one per new token]}` for this row.
    pub customized_info: Option<serde_json::Value>,
    /// A finished request's `meta_info` stats, as an object.
    pub request_stats: Option<serde_json::Value>,
    /// `[[version, start, end], ...]`.
    pub weight_versions: Option<serde_json::Value>,
    /// The new tokens' logprobs, as `(logprob, token id)`.
    pub output_token_logprobs: Vec<(f32, i32)>,
    /// Per new token, its top candidates as `(logprob, token id)`.
    pub output_top_logprobs: Vec<Vec<(f32, i32)>>,
}

/// One message on the scheduler ring as the scheduler receives it: the
/// msgpack header and the named buffers that travel beside it.
#[derive(Debug)]
#[non_exhaustive]
pub struct RingRequest {
    pub header: Vec<u8>,
    pub buffers: Vec<RingBuffer>,
}

/// A buffer of a [`RingRequest`]: its name (`input_ids`, ...) and shape.
#[derive(Debug)]
#[non_exhaustive]
pub struct RingBuffer {
    pub name: String,
    pub shape: Vec<usize>,
}

impl RingRequest {
    /// The message type, the header's first field.
    pub fn kind(&self) -> Option<String> {
        self.field(0)
    }

    /// The rid, the header's second field, of a message that carries one.
    pub fn rid(&self) -> Option<String> {
        self.field(1)
    }

    fn field(&self, index: usize) -> Option<String> {
        let header = rmpv::decode::read_value(&mut &self.header[..]).ok()?;
        header.as_array()?.get(index)?.as_str().map(str::to_owned)
    }

    /// The shape of the buffer `name`, if the message carries it.
    pub fn buffer_shape(&self, name: &str) -> Option<&[usize]> {
        let buffer = self.buffers.iter().find(|buffer| buffer.name == name)?;
        Some(&buffer.shape)
    }
}

/// The launch of a [`SchedulerlessServer`]: no tokenizer and no startup
/// warmup unless `server_args` says otherwise.
pub struct ServerBuilder {
    rust_server_args: RustServerServerArgs,
    server_args: ServerArgs,
    extensions: Extensions,
    generate_validator: Option<Arc<dyn GenerateValidator>>,
    http_layer: Option<HttpLayer>,
    server_info_state_keys: Vec<String>,
}

impl ServerBuilder {
    /// Also serve gRPC, on `addr`.
    pub fn grpc_addr(mut self, addr: SocketAddr) -> Self {
        self.rust_server_args.grpc_addr = Some(addr);
        self
    }

    /// Serve as the listener of DP rank `dp_rank` among `listeners`, one per
    /// rank of an attention-DP deployment: control requests and aborts are
    /// relayed to the others.
    pub fn dp_listeners(mut self, listeners: &[SocketAddr], dp_rank: usize) -> Self {
        self.rust_server_args.http_addr = listeners[dp_rank];
        self.server_args.deployment_info = DeploymentInfo {
            dp_size: listeners.len() as u32,
            dp_rank: Some(dp_rank as u32),
            enable_dp_attention: true,
            rust_server_endpoints: listeners
                .iter()
                .enumerate()
                .map(|(rank, addr)| RustServerEndpoint {
                    dp_rank: rank as u32,
                    endpoint: format!("http://{addr}"),
                })
                .collect(),
            ..Default::default()
        };
        self
    }

    /// Change the scheduler's launch arguments.
    pub fn server_args(mut self, change: impl FnOnce(&mut ServerArgs)) -> Self {
        change(&mut self.server_args);
        self
    }

    /// Change the frontend's own launch arguments.
    pub fn rust_server_args(mut self, change: impl FnOnce(&mut RustServerServerArgs)) -> Self {
        change(&mut self.rust_server_args);
        self
    }

    pub fn extensions(mut self, extensions: Extensions) -> Self {
        self.extensions = extensions;
        self
    }

    pub fn generate_validator(mut self, validator: Arc<dyn GenerateValidator>) -> Self {
        self.generate_validator = Some(validator);
        self
    }

    /// Wrap the served HTTP app, as an embedding package does.
    pub fn http_layer(mut self, layer: HttpLayer) -> Self {
        self.http_layer = Some(layer);
        self
    }

    pub fn server_info_state_keys(mut self, keys: Vec<String>) -> Self {
        self.server_info_state_keys = keys;
        self
    }

    pub fn start(self) -> Result<SchedulerlessServer, String> {
        let cfg = RuntimeConfig {
            rust_server_args: self.rust_server_args,
            server_args: Arc::new(self.server_args),
            generate_hook: self.extensions.generate_hook,
            generate_validator: self.generate_validator,
            abort_hook: self.extensions.abort_hook,
            server_info_state_keys: self.server_info_state_keys,
        };
        start(cfg, self.http_layer).map(SchedulerlessServer)
    }
}

impl SchedulerlessServer {
    /// A launch serving HTTP on `http_addr`.
    pub fn builder(http_addr: SocketAddr) -> ServerBuilder {
        ServerBuilder {
            rust_server_args: RustServerServerArgs {
                http_addr,
                http_api_worker_num: 1,
                ..Default::default()
            },
            server_args: ServerArgs {
                skip_tokenizer_init: true,
                skip_server_warmup: true,
                ..Default::default()
            },
            extensions: Extensions::default(),
            generate_validator: None,
            http_layer: None,
            server_info_state_keys: Vec::new(),
        }
    }

    /// Serve on `listeners[dp_rank]` as that DP rank's listener: with more
    /// than one listener, as one of an attention-DP deployment's, which
    /// relays aborts and control requests to the others.
    pub fn start(
        listeners: &[SocketAddr],
        dp_rank: usize,
        extensions: Extensions,
    ) -> Result<Self, String> {
        let builder = Self::builder(listeners[dp_rank]);
        let builder = if listeners.len() > 1 {
            builder.dp_listeners(listeners, dp_rank)
        } else {
            builder
        };
        builder.extensions(extensions).start()
    }

    /// Start the multimodal worker pool with `processor`.
    pub fn start_mm_workers(&self, processor: Arc<dyn MmProcessor>, workers: usize) {
        self.0
            .start_mm_workers_with_processor(processor, workers, false);
    }

    /// The next message on the scheduler ring, whole, waiting up to
    /// `timeout`. The first one after a launch is the `FreezeGCReq`.
    pub fn next_request(&self, timeout: Duration) -> Option<RingRequest> {
        if !self.0.to_scheduler_rx.wait(timeout) {
            return None;
        }
        let request = self.0.to_scheduler_rx.drain(1).pop()?;
        Some(RingRequest {
            header: request.header.to_vec(),
            buffers: request
                .buffers
                .into_iter()
                .map(|buffer| RingBuffer {
                    name: buffer.name,
                    shape: buffer.shape,
                })
                .collect(),
        })
    }

    /// The next message on the scheduler ring, waiting up to `timeout`.
    pub fn next_message(&self, timeout: Duration) -> Option<SchedulerMessage> {
        let request = self.next_request(timeout)?;
        Some(SchedulerMessage {
            kind: request.kind()?,
            rid: request.rid()?,
        })
    }

    /// Answer for the scheduler with one output for `rid` (the rid it
    /// received): `output_ids` more tokens, and the request's end when
    /// `finish_reason` is given (Python's `FINISH_*.to_json()`).
    pub fn push_output(
        &self,
        rid: &str,
        output_ids: &[i64],
        finish_reason: Option<serde_json::Value>,
    ) -> Result<(), String> {
        let details = OutputDetails {
            prompt_tokens: 1,
            finish_reason,
            ..Default::default()
        };
        self.push_output_with(rid, output_ids, details)
    }

    /// Answer for the scheduler with one output for `rid`: `output_ids` more
    /// tokens and everything else `details` says about the row.
    pub fn push_output_with(
        &self,
        rid: &str,
        output_ids: &[i64],
        details: OutputDetails,
    ) -> Result<(), String> {
        let json = |error: serde_json::Error| error.to_string();
        let mut header = BatchHeader {
            rids: vec![rid.to_owned()],
            finish_reasons: vec![
                details
                    .finish_reason
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(json)?,
            ],
            prompt_tokens: vec![details.prompt_tokens],
            tok_lens: vec![output_ids.len() as u32],
            cached_tokens: vec![details.cached_tokens],
            reasoning_tokens: vec![details.reasoning_tokens],
            storage_backend: details.storage_backend,
            ..Default::default()
        };
        if details.abort {
            header.abort_outputs = vec![true];
        }
        if let Some(split) = details.cached_tokens_split {
            header.cached_tokens_split = vec![serde_json::from_value(split).map_err(json)?];
        }
        if let Some(latency) = details.decode_latency {
            header.decode_latencies = vec![Some(latency)];
        }
        if let Some(stats) = details.request_stats {
            header.request_stats = vec![Some(serde_json::from_value(stats).map_err(json)?)];
        }
        if let Some(spans) = details.weight_versions {
            header.weight_versions = vec![Some(serde_json::from_value(spans).map_err(json)?)];
        }
        if let Some(info) = details.customized_info {
            // The wire carries it as msgpack bin of JSON text.
            let mut text = Vec::new();
            rmpv::encode::write_value(&mut text, &rmpv::Value::Binary(info.to_string().into()))
                .map_err(|error| error.to_string())?;
            header.customized_info =
                rmp_serde::from_slice(&text).map_err(|error| error.to_string())?;
        }
        // After the ids, the logprob columns: every value, then every id.
        let logprobs = &details.output_token_logprobs;
        if !logprobs.is_empty() {
            header.out_lp_lens = vec![logprobs.len() as u32];
        }
        // Then the top logprobs, flat: positions per row in the header, and
        // candidates per position.
        let top = &details.output_top_logprobs;
        if !top.is_empty() {
            header.out_top_reqlens = vec![top.len() as u32];
            header.out_top_poslens = top.iter().map(|position| position.len() as u32).collect();
        }
        let header = rmp_serde::to_vec(&header).map_err(|error| error.to_string())?;
        let ids: Vec<u8> = output_ids
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect();
        let values: Vec<u8> = logprobs
            .iter()
            .flat_map(|(value, _)| value.to_le_bytes())
            .collect();
        let tokens: Vec<u8> = logprobs
            .iter()
            .flat_map(|(_, token)| token.to_le_bytes())
            .collect();
        let top_values: Vec<u8> = top
            .iter()
            .flatten()
            .flat_map(|(value, _)| value.to_le_bytes())
            .collect();
        let top_tokens: Vec<u8> = top
            .iter()
            .flatten()
            .flat_map(|(_, token)| token.to_le_bytes())
            .collect();
        let columns: [&[u8]; 5] = [&ids, &values, &tokens, &top_values, &top_tokens];
        self.push(frame_decode_batch_cols(&header, &columns))
    }

    /// Answer the control request `rid` with `result`, as the scheduler
    /// answers a flush, a profile or a HiCache operation.
    pub fn push_control_result(&self, rid: &str, result: &serde_json::Value) -> Result<(), String> {
        let payload = rmp_serde::to_vec_named(result).map_err(|error| error.to_string())?;
        self.push(frame_control_result(rid, &payload))
    }

    /// Answer the control request `rid` with `payload` as it stands, for a
    /// reply the scheduler would never encode.
    pub fn push_control_payload(&self, rid: &str, payload: &[u8]) -> Result<(), String> {
        self.push(frame_control_result(rid, payload))
    }

    /// Fail the request `rid` with the scheduler error `message`.
    pub fn push_error(&self, rid: &str, message: &str) -> Result<(), String> {
        self.push(frame_error(rid, message))
    }

    fn push(&self, frame: bytes::Bytes) -> Result<(), String> {
        if self.0.from_scheduler_tx.push(frame) {
            Ok(())
        } else {
            Err("the frontend shut down".into())
        }
    }

    /// Begin the graceful drain, or with `force` close at once.
    pub fn begin_drain(&self, force: bool) {
        self.0.begin_drain(force);
    }

    /// Whether a drain that began has ended.
    pub fn drain_complete(&self) -> bool {
        self.0.drain_complete()
    }

    /// Stop the frontend and join its threads; dropping does the same.
    pub fn shutdown(&self) {
        self.0.request_shutdown();
    }
}
