//! Python `TokenizerManager`'s request log (`--log-requests`, text format to
//! stdout) and crash-dump recording (`--crash-dump-folder`), taken at the same
//! request boundaries: `Receive` before any preprocessing, `Finish` on the
//! terminal output (scheduler aborts included), and, for a dump, the finished
//! requests of the last five minutes plus every unfinished one.
//!
//! A request is described in Python's `GenerateReqInput` terms: a native
//! `/generate` body as the client sent it, with the values Python's
//! `normalize_batch_and_arguments` fills in taken from the normalized
//! [`GenerateRequest`]; other entry points describe the request they built.
//! The lines are rendered with Python's rules (`request_logger.py`) from the
//! settings Python hands over ([`RequestLogConfig`]). Strings print as
//! Python's `repr` does, except that non-ASCII characters print as they are
//! (Python escapes the rare non-printable ones).
//!
//! The crash dump itself is written by Python (`RustServer
//! .dump_requests_before_crash`), from [`RequestLog::crash_dump_snapshot`].

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use serde_json::{Map, Value};

use super::frame::CoreOutput;
use crate::message::config::{RequestLogConfig, ServerArgs};
use crate::message::multimodal::MmItem;
use crate::message::request::GenerateRequest;
use crate::message::response::ResponseItem;

/// Python's `record_request_for_crash_dump` keeps a finished request this long.
const CRASH_DUMP_WINDOW_SECS: f64 = 300.0;

/// Request logging and crash-dump recording, whichever is enabled.
pub(crate) struct RequestLog {
    logger: Option<RequestLogger>,
    recorder: Option<CrashRecorder>,
    /// `sampling_params.custom_params` keys the server adds itself, which the
    /// log lines leave out (a crash dump's recorder holds its own copy).
    server_added_custom_params: Vec<String>,
    /// Python `_set_default_priority` runs before the request is logged.
    default_priority: Option<i64>,
    /// The launch weight version the core stamps on every output.
    weight_version: Option<Arc<str>>,
    /// `--incremental-streaming-output`: a stream's `Finish` shows its last
    /// delta, as Python's does.
    incremental_streaming_output: bool,
}

/// The whitelisted request headers a request's lines carry.
pub(crate) type LoggedHeaders = Arc<Vec<(String, String)>>;

/// A native `/generate` body captured before the launch-time sampling defaults
/// are merged into it (Python merges them into a copy), for the request's log
/// lines and crash-dump record.
pub(crate) struct BodySnapshot {
    /// The top-level fields the log shows (empty when not logging).
    fields: Map<String, Value>,
    /// The whole body for a crash dump (`None` when not recording).
    dump: Option<Arc<DumpBody>>,
    headers: LoggedHeaders,
    /// When an HTTP layer started the request before `/generate` ran (work
    /// an `HttpLayer` does before the handler), so the `Finish` latency
    /// includes that wait.
    started_at: Option<Instant>,
}

impl BodySnapshot {
    pub(crate) fn started_at(mut self, at: Option<Instant>) -> Self {
        self.started_at = at;
        self
    }
}

/// A native `/generate` body's bytes as received, and the header overrides
/// written into the parsed body (`None` when overrides are off). A crash
/// dump keeps both and applies the overrides only when it is written, rather
/// than serializing every request's body again.
pub(crate) struct RawBody {
    pub(crate) bytes: Bytes,
    pub(crate) overrides: Option<Map<String, Value>>,
}

/// How a generation request reached the core.
pub(crate) enum ReceivedAs {
    /// A native `/generate` body ([`CoreHandle::snapshot_body`](crate::api_server::handlers::CoreHandle::snapshot_body)).
    Body(Option<BodySnapshot>),
    /// Built by an adapter (OpenAI, gRPC) or by the core itself, with the
    /// headers of the HTTP request it came from.
    Typed(LoggedHeaders),
}

impl ReceivedAs {
    /// A built request with no HTTP headers (gRPC, the startup warmup).
    pub(crate) fn without_headers() -> Self {
        Self::Typed(LoggedHeaders::default())
    }
}

impl RequestLog {
    /// `None` when neither `--log-requests` nor `--crash-dump-folder` is set.
    pub(crate) fn from_server_args(args: &ServerArgs) -> Option<Arc<Self>> {
        if args.request_log.is_none() && !args.crash_dump {
            return None;
        }
        Some(Arc::new(Self {
            logger: args
                .request_log
                .clone()
                .map(|config| RequestLogger::new(config, LogSink::Stdout)),
            recorder: args
                .crash_dump
                .then(|| CrashRecorder::omitting(args.server_added_custom_params.clone())),
            server_added_custom_params: args.server_added_custom_params.clone(),
            default_priority: args.default_priority_value,
            weight_version: args.weight_version.as_deref().map(Arc::from),
            incremental_streaming_output: args.incremental_streaming_output,
        }))
    }

    /// Lines kept in memory instead of printed.
    #[cfg(test)]
    pub(crate) fn for_test(config: Option<RequestLogConfig>, crash_dump: bool) -> Arc<Self> {
        Self::for_test_omitting(config, crash_dump, &[])
    }

    /// [`Self::for_test`] for a server that adds `omitted` to `custom_params`.
    #[cfg(test)]
    pub(crate) fn for_test_omitting(
        config: Option<RequestLogConfig>,
        crash_dump: bool,
        omitted: &[&str],
    ) -> Arc<Self> {
        let omitted: Vec<String> = omitted.iter().map(|key| key.to_string()).collect();
        Arc::new(Self {
            logger: config
                .map(|config| RequestLogger::new(config, LogSink::Memory(Mutex::default()))),
            recorder: crash_dump.then(|| CrashRecorder::omitting(omitted.clone())),
            server_added_custom_params: omitted,
            default_priority: None,
            weight_version: Some(Arc::from("default")),
            incremental_streaming_output: false,
        })
    }

    #[cfg(test)]
    pub(crate) fn logged_lines(&self) -> Vec<String> {
        match &self.logger.as_ref().expect("logging").sink {
            LogSink::Memory(lines) => std::mem::take(&mut *lock(lines)),
            LogSink::Stdout => unreachable!("tests log to memory"),
        }
    }

    /// `raw` is what a crash dump keeps of a body that came over HTTP instead
    /// of serializing `body`; `None` for a body built in process.
    pub(crate) fn snapshot_body(
        &self,
        body: &Value,
        raw: Option<RawBody>,
        headers: &http::HeaderMap,
    ) -> Option<BodySnapshot> {
        let object = body.as_object()?;
        let fields = match &self.logger {
            Some(logger) => object
                .iter()
                .filter_map(|(name, value)| {
                    let value = if !logger.skip_names.contains(name.as_str()) {
                        let mut value = value.clone();
                        if name == "sampling_params" {
                            omit_custom_params(&mut value, &self.server_added_custom_params);
                        }
                        value
                    } else if name == "image_data" {
                        // Its layout, for a batch's `modalities`.
                        image_layout(value)
                    } else {
                        return None;
                    };
                    Some((name.clone(), value))
                })
                .collect(),
            None => Map::new(),
        };
        let dump = self.recorder.as_ref().map(|recorder| {
            recorder.body(raw.unwrap_or_else(|| RawBody {
                bytes: Bytes::from(serde_json::to_vec(body).expect("a JSON value serializes")),
                overrides: None,
            }))
        });
        Some(BodySnapshot {
            fields,
            dump,
            headers: self.logged_headers(headers),
            started_at: None,
        })
    }

    /// The whitelisted headers present on `headers` (none when not logging).
    pub(crate) fn logged_headers(&self, headers: &http::HeaderMap) -> LoggedHeaders {
        let Some(logger) = &self.logger else {
            return LoggedHeaders::default();
        };
        Arc::new(
            logger
                .config
                .headers
                .iter()
                .filter_map(|name| {
                    let value = headers.get(name)?.to_str().ok()?;
                    (!value.is_empty()).then(|| (name.clone(), value.to_owned()))
                })
                .collect(),
        )
    }

    /// Log `Receive` for `requests` (one line for a whole batch) and start one
    /// record per request, which logs `Finish` and keeps the crash-dump entry.
    pub(crate) fn receive(
        self: &Arc<Self>,
        requests: &[GenerateRequest],
        received: ReceivedAs,
        batch: bool,
        log_metrics: bool,
    ) -> Vec<RequestRecord> {
        let created_at = match &received {
            ReceivedAs::Body(Some(snapshot)) => snapshot.started_at,
            _ => None,
        }
        .unwrap_or_else(Instant::now);
        let created_time = unix_time();
        let (obj, items, dumps, headers) = match received {
            ReceivedAs::Body(Some(snapshot)) => self.body_views(snapshot, requests, batch),
            // A body that is not a JSON object fails to decode before this.
            ReceivedAs::Body(None) => self.typed_views(requests, batch, LoggedHeaders::default()),
            ReceivedAs::Typed(headers) => self.typed_views(requests, batch, headers),
        };
        if let Some(logger) = &self.logger {
            logger.received(&obj.into(), &headers);
        }
        items
            .into_iter()
            .zip(dumps)
            .zip(requests)
            .map(|((obj, dump), request)| {
                let output = Arc::new(Mutex::new(CoreOutput {
                    weight_version: self.weight_version.clone(),
                    ..Default::default()
                }));
                let rid = request.rid.client_facing().to_owned();
                let in_flight = match (&self.recorder, dump) {
                    (Some(recorder), Some((body, batch_index))) => Some(recorder.register(
                        DumpRequest {
                            body,
                            batch_index,
                            rid: rid.clone(),
                        },
                        created_time,
                        output.clone(),
                    )),
                    _ => None,
                };
                RequestRecord {
                    log: self.clone(),
                    obj,
                    headers: headers.clone(),
                    rid,
                    created_at,
                    incremental: request.stream && self.incremental_streaming_output,
                    output,
                    in_flight,
                    log_metrics,
                    admitted: false,
                }
            })
            .collect()
    }

    /// The Python objects of a native `/generate` body: the client's fields,
    /// plus what `normalize_batch_and_arguments` sets, read off the requests
    /// `into_requests` normalized the same way.
    fn body_views(
        &self,
        snapshot: BodySnapshot,
        requests: &[GenerateRequest],
        batch: bool,
    ) -> Views {
        let BodySnapshot {
            fields,
            dump,
            headers,
            ..
        } = snapshot;
        let logging = self.logger.is_some();
        // Held once for the whole batch, as Python's items share them.
        let shared = Arc::new(match &self.logger {
            Some(logger) if batch => logger.batch_shared(&fields),
            _ => Map::new(),
        });
        let mut items = Vec::with_capacity(requests.len());
        let mut dumps = Vec::with_capacity(requests.len());
        for (i, request) in requests.iter().enumerate() {
            let mut item = Map::new();
            if let Some(logger) = &self.logger {
                if batch {
                    item = logger.batch_item(&fields, i);
                    // Python's batch normalization: a single `bootstrap_room`
                    // counts up per item, as the normalized requests carry it,
                    // and each item's images become a list with their modality.
                    item.insert("bootstrap_room".into(), request.bootstrap_room.into());
                    if let Some((images, modality)) = fields
                        .get("image_data")
                        .and_then(|images| batch_images(images, i))
                    {
                        item.insert("image_data".into(), images);
                        item.insert("modalities".into(), modality);
                    }
                } else {
                    item = fields.clone();
                }
                self.normalize(&mut item, request);
            }
            items.push(PyObj {
                own: item,
                shared: shared.clone(),
            });
            dumps.push(dump.clone().map(|body| (body, batch.then_some(i))));
        }
        let obj = if !logging {
            Map::new()
        } else if batch {
            // Python's batch object holds these as per-item lists.
            let mut obj = fields;
            for name in [
                "rid",
                "sampling_params",
                "return_logprob",
                "logprob_start_len",
                "top_logprobs_num",
            ] {
                let column = items.iter().map(|item| item.own[name].clone()).collect();
                obj.insert(name.into(), Value::Array(column));
            }
            obj
        } else {
            items[0].own.clone()
        };
        (obj, items, dumps, headers)
    }

    /// Python's single-request normalization of the fields the request
    /// carries.
    fn normalize(&self, item: &mut Map<String, Value>, request: &GenerateRequest) {
        // The deprecated `data_parallel_rank` moves to `routed_dp_rank` (the
        // Rust server itself reads only `routed_dp_rank`).
        if let Some(rank) = item
            .remove("data_parallel_rank")
            .filter(|rank| !rank.is_null())
            && item.get("routed_dp_rank").is_none_or(Value::is_null)
        {
            item.insert("routed_dp_rank".into(), rank);
        }
        if item.get("sampling_params").is_none_or(Value::is_null) {
            item.insert("sampling_params".into(), Value::Object(Map::new()));
        }
        item.insert("rid".into(), request.rid.client_facing().into());
        item.insert("return_logprob".into(), request.return_logprob.into());
        item.insert("logprob_start_len".into(), request.logprob_start_len.into());
        item.insert("top_logprobs_num".into(), request.top_logprobs_num.into());
        item.insert(
            "token_ids_logprob".into(),
            request.token_ids_logprob.clone().into(),
        );
        item.insert(
            "return_sampling_mask".into(),
            request.return_sampling_mask.into(),
        );
        item.insert("extra_key".into(), request.extra_key.clone().into());
        item.insert("cache_salt".into(), request.cache_salt.clone().into());
        item.insert(
            "priority".into(),
            request.priority.or(self.default_priority).into(),
        );
    }

    /// The Python object of a request an adapter or the core built. Its
    /// fields are all known, so the log and the crash dump read one map.
    fn typed_views(
        &self,
        requests: &[GenerateRequest],
        batch: bool,
        headers: LoggedHeaders,
    ) -> Views {
        let items: Vec<Map<String, Value>> = requests.iter().map(typed_fields).collect();
        let dumps = items
            .iter()
            .map(|item| {
                self.recorder.as_ref().map(|recorder| {
                    let json = serde_json::to_vec(item).expect("a JSON map serializes");
                    let body = recorder.body(RawBody {
                        bytes: Bytes::from(json),
                        overrides: None,
                    });
                    (body, None)
                })
            })
            .collect();
        let obj = if batch {
            let mut obj = items.first().cloned().unwrap_or_default();
            obj.insert(
                "rid".into(),
                Value::Array(items.iter().map(|item| item["rid"].clone()).collect()),
            );
            obj
        } else {
            items.first().cloned().unwrap_or_default()
        };
        (
            obj,
            items.into_iter().map(PyObj::from).collect(),
            dumps,
            headers,
        )
    }

    /// The crash dump's requests as JSON (see `CrashRecorder::snapshot`);
    /// `None` when not recording.
    pub(crate) fn crash_dump_snapshot(&self) -> Option<Vec<u8>> {
        self.recorder
            .as_ref()
            .map(|recorder| recorder.snapshot(unix_time()))
    }
}

/// `(Receive object, per-request Finish objects, per-request dump sources,
/// headers)`.
type Views = (
    Map<String, Value>,
    Vec<PyObj>,
    Vec<Option<(Arc<DumpBody>, Option<usize>)>>,
    LoggedHeaders,
);

/// A request's Python object: its own fields over the ones its batch shares,
/// which every item references as Python's `obj[i]` does.
#[derive(Default)]
struct PyObj {
    own: Map<String, Value>,
    shared: Arc<Map<String, Value>>,
}

impl PyObj {
    fn get(&self, name: &str) -> Option<&Value> {
        self.own.get(name).or_else(|| self.shared.get(name))
    }
}

impl From<Map<String, Value>> for PyObj {
    fn from(own: Map<String, Value>) -> Self {
        Self {
            own,
            shared: Arc::default(),
        }
    }
}

/// Python `_normalize_image_data` for item `i` of a batch: its images as a
/// list, and the `modalities` entry that sets; `None` when the batch has no
/// images, which leaves both fields as the client sent them.
fn batch_images(images: &Value, i: usize) -> Option<(Value, Value)> {
    Some(match images {
        Value::Null => return None,
        Value::Array(lists) if lists.is_empty() => return None,
        // One list per item already.
        Value::Array(lists) if lists[0].is_array() => {
            let images = lists.get(i)?.clone();
            let modality = match &images {
                Value::Array(list) if list.len() > 1 => "multi-images".into(),
                Value::Array(list) if list.len() == 1 && !list[0].is_null() => "image".into(),
                _ => Value::Null,
            };
            (images, modality)
        }
        Value::Array(list) => (Value::Array(vec![list.get(i)?.clone()]), "image".into()),
        // One image for every item.
        image => (Value::Array(vec![image.clone()]), "image".into()),
    })
}

/// What [`batch_images`] reads of `image_data` (nesting, lengths, `None`s)
/// without the images, for a level that does not show them.
fn image_layout(images: &Value) -> Value {
    match images {
        Value::Array(items) => Value::Array(items.iter().map(image_layout).collect()),
        Value::Null => Value::Null,
        _ => Value::Bool(true),
    }
}

fn typed_fields(request: &GenerateRequest) -> Map<String, Value> {
    let mut item = Map::new();
    let mut set = |name: &str, value: Value| {
        item.insert(name.into(), value);
    };
    set("rid", request.rid.client_facing().into());
    set("text", request.text.clone().into());
    set("input_ids", request.input_ids.clone().into());
    if let Some(mm) = request.mm.as_deref() {
        for (name, items) in [
            ("image_data", &mm.image_data),
            ("video_data", &mm.video_data),
            ("audio_data", &mm.audio_data),
        ] {
            if !items.is_empty() {
                set(name, items.iter().map(media_item).collect());
            }
        }
    }
    set(
        "sampling_params",
        serde_json::to_value(&request.sampling_params).unwrap_or(Value::Null),
    );
    set("return_logprob", request.return_logprob.into());
    set("logprob_start_len", request.logprob_start_len.into());
    set("top_logprobs_num", request.top_logprobs_num.into());
    set(
        "token_ids_logprob",
        request.token_ids_logprob.clone().into(),
    );
    set("return_sampling_mask", request.return_sampling_mask.into());
    if let Some(value) = request.return_text_in_logprobs {
        set("return_text_in_logprobs", value.into());
    }
    set("stream", request.stream.into());
    set("return_hidden_states", request.return_hidden_states.into());
    set("bootstrap_host", request.bootstrap_host.clone().into());
    set("bootstrap_port", request.bootstrap_port.into());
    set("bootstrap_room", request.bootstrap_room.into());
    set(
        "bootstrap_pair_key",
        request.bootstrap_pair_key.clone().into(),
    );
    set("decode_tp_size", request.decode_tp_size.into());
    set("routed_dp_rank", request.routed_dp_rank.into());
    set(
        "disagg_prefill_dp_rank",
        request.disagg_prefill_dp_rank.into(),
    );
    set("require_reasoning", request.require_reasoning.into());
    set("priority", request.priority.into());
    set("extra_key", request.extra_key.clone().into());
    set("cache_salt", request.cache_salt.clone().into());
    item
}

/// A media item as Python's request takes it: the bare source, or
/// `{"url", ...}` with its hints.
fn media_item(item: &MmItem) -> Value {
    match item {
        MmItem::Source(source) => source.as_str().into(),
        MmItem::Ref { url, hints } => {
            let mut item = Map::new();
            item.insert("url".into(), url.as_str().into());
            if let Some(fps) = hints.fps {
                item.insert("fps".into(), fps.into());
            }
            Value::Object(item)
        }
    }
}

/// One request's log and crash-dump state, carried by its core call.
pub(crate) struct RequestRecord {
    log: Arc<RequestLog>,
    /// The Python object of its `Finish` line (empty when not logging).
    obj: PyObj,
    headers: LoggedHeaders,
    rid: String,
    created_at: Instant,
    /// An incremental stream, whose `Finish` shows the last delta.
    incremental: bool,
    /// The output so far, shared with the crash recorder's in-flight entry.
    output: Arc<Mutex<CoreOutput>>,
    /// Its in-flight entry, until it finishes or is dropped.
    in_flight: Option<u64>,
    /// Python's `log_metrics`: false for the health probe, which a crash dump
    /// leaves out.
    log_metrics: bool,
    /// TM intake accepted the request: if its call is dropped unfinished (the
    /// client went away), Python still records it once the abort lands.
    admitted: bool,
}

impl RequestRecord {
    /// TM intake accepted the request (its core call exists).
    pub(crate) fn admit(&mut self) {
        self.admitted = true;
    }

    /// Observe one runtime item of the call.
    pub(crate) fn observe(&mut self, item: &ResponseItem) {
        match item {
            ResponseItem::Frame(chunk) => {
                let delta = CoreOutput::from(chunk.clone());
                lock(&self.output).append_delta(&delta);
            }
            ResponseItem::Done(chunk) => {
                let delta = CoreOutput::from(chunk.clone());
                let output = {
                    let mut output = lock(&self.output);
                    output.append_delta(&delta);
                    if self.incremental {
                        // The stream's terminal frame: the last delta, with
                        // the cumulative completion-token count.
                        CoreOutput {
                            completion_tokens: output.completion_tokens,
                            weight_version: output.weight_version.clone(),
                            ..delta
                        }
                    } else {
                        output.clone()
                    }
                };
                self.finish(output);
            }
            _ => self.unregister(),
        }
    }

    /// The `/generate` frame of the finished request is Python's `out`.
    fn finish(&mut self, output: CoreOutput) {
        let e2e_latency = self.created_at.elapsed().as_secs_f64();
        let out = python_out(output, &self.rid, e2e_latency);
        if let Some(logger) = &self.log.logger {
            logger.finished(&self.obj, &self.headers, &out, e2e_latency);
        }
        if let (Some(recorder), Some(id)) = (&self.log.recorder, self.in_flight.take()) {
            recorder.finish(id, self.log_metrics, &out, unix_time());
        }
    }

    fn unregister(&mut self) {
        if let (Some(recorder), Some(id)) = (&self.log.recorder, self.in_flight.take()) {
            recorder.unregister(id);
        }
    }
}

impl Drop for RequestRecord {
    fn drop(&mut self) {
        let (Some(recorder), Some(id)) = (&self.log.recorder, self.in_flight.take()) else {
            return;
        };
        if self.admitted {
            // Dropped unfinished: kept with what it generated, unlogged.
            let output = lock(&self.output).clone();
            let out = python_out(output, &self.rid, self.created_at.elapsed().as_secs_f64());
            recorder.finish(id, self.log_metrics, &out, unix_time());
        } else {
            recorder.unregister(id);
        }
    }
}

/// `output`'s `/generate` frame, with its keys in the order of Python's output
/// dict (`text`, `output_ids`, `meta_info`) and `output_ids` present even when
/// empty as Python's is.
fn python_out(output: CoreOutput, rid: &str, e2e_latency: f64) -> Map<String, Value> {
    let frame = output.frame(rid, None, Some(e2e_latency));
    let Value::Object(mut frame) = serde_json::to_value(&frame).expect("frames serialize") else {
        unreachable!("a frame is a JSON object");
    };
    let mut out = Map::new();
    out.insert("text".into(), frame.remove("text").unwrap_or_default());
    out.insert(
        "output_ids".into(),
        frame
            .remove("output_ids")
            .unwrap_or_else(|| Value::Array(Vec::new())),
    );
    if let Some(meta_info) = frame.remove("meta_info") {
        out.insert("meta_info".into(), meta_info);
    }
    out.extend(frame);
    out
}

fn unix_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

// --- the log lines ----------------------------------------------------------

/// Kept with each batch item, whose normalization moves `data_parallel_rank`
/// to `routed_dp_rank`.
const DP_RANK_FIELDS: [&str; 2] = ["data_parallel_rank", "routed_dp_rank"];

struct RequestLogger {
    config: RequestLogConfig,
    skip_names: HashSet<String>,
    out_skip_names: HashSet<String>,
    batch_item_fields: HashSet<String>,
    batch_shared_fields: HashSet<String>,
    sink: LogSink,
}

impl RequestLogger {
    fn new(config: RequestLogConfig, sink: LogSink) -> Self {
        let set = |names: &[String]| names.iter().cloned().collect();
        Self {
            skip_names: set(&config.skip_names),
            out_skip_names: set(&config.out_skip_names),
            batch_item_fields: set(&config.batch_item_fields),
            batch_shared_fields: set(&config.batch_shared_fields),
            config,
            sink,
        }
    }

    /// Item `i`'s own fields of a batch body, as Python's `obj[i]` takes
    /// them: its entry of each per-item field, and the dp rank fields the
    /// item's normalization reads.
    fn batch_item(&self, fields: &Map<String, Value>, i: usize) -> Map<String, Value> {
        fields
            .iter()
            .filter_map(|(name, value)| {
                let value = if self.batch_item_fields.contains(name) {
                    match value {
                        Value::Array(values) => values.get(i)?.clone(),
                        // One value for every item.
                        value => value.clone(),
                    }
                } else if DP_RANK_FIELDS.contains(&name.as_str()) {
                    value.clone()
                } else {
                    return None;
                };
                Some((name.clone(), value))
            })
            .collect()
    }

    /// The fields every item of a batch body shares.
    fn batch_shared(&self, fields: &Map<String, Value>) -> Map<String, Value> {
        fields
            .iter()
            .filter(|(name, _)| {
                self.batch_shared_fields.contains(name.as_str())
                    && !DP_RANK_FIELDS.contains(&name.as_str())
            })
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }

    fn received(&self, obj: &PyObj, headers: &[(String, String)]) {
        self.sink.emit(&self.received_line(obj, headers));
    }

    fn finished(
        &self,
        obj: &PyObj,
        headers: &[(String, String)],
        out: &Map<String, Value>,
        e2e_latency: f64,
    ) {
        let exceeded_ms = self.config.exceeded_ms;
        if exceeded_ms > 0 && e2e_latency * 1000.0 < exceeded_ms as f64 {
            return;
        }
        self.sink.emit(&self.finished_line(obj, headers, out));
    }

    /// Python `log_received_request`'s message.
    fn received_line(&self, obj: &PyObj, headers: &[(String, String)]) -> String {
        let mut line = String::from("Receive: obj=");
        self.write_obj(&mut line, obj);
        write_headers(&mut line, headers);
        line
    }

    /// Python `log_finished_request`'s message.
    fn finished_line(
        &self,
        obj: &PyObj,
        headers: &[(String, String)],
        out: &Map<String, Value>,
    ) -> String {
        let mut line = String::from("Finish: obj=");
        self.write_obj(&mut line, obj);
        write_headers(&mut line, headers);
        line.push_str(", out=");
        write_dict(
            &mut line,
            out.iter()
                .filter(|(name, _)| !self.out_skip_names.contains(name.as_str())),
            self.config.max_length,
            false,
        );
        line
    }

    /// `_dataclass_to_string_truncated` of the request object: every field in
    /// declaration order, the request's value or the class default.
    fn write_obj(&self, line: &mut String, obj: &PyObj) {
        line.push_str(&self.config.class_name);
        line.push('(');
        let mut first = true;
        for (name, default) in &self.config.fields {
            if self.skip_names.contains(name) {
                continue;
            }
            if !first {
                line.push_str(", ");
            }
            first = false;
            line.push_str(name);
            line.push('=');
            match obj.get(name) {
                Some(value) => write_field(line, name, value, self.config.max_length, false),
                None => line.push_str(default),
            }
        }
        line.push(')');
    }
}

/// `, headers={...}` when any whitelisted header is present.
fn write_headers(line: &mut String, headers: &[(String, String)]) {
    if headers.is_empty() {
        return;
    }
    line.push_str(", headers={");
    for (i, (name, value)) in headers.iter().enumerate() {
        if i > 0 {
            line.push_str(", ");
        }
        write_str_repr(line, name);
        line.push_str(": ");
        write_str_repr(line, value);
    }
    line.push('}');
}

/// Where the lines go: stdout through the process's request-log writer (see
/// [`StdoutWriter`]), each line prefixed with the local time as Python's
/// request-log formatter does.
enum LogSink {
    Stdout,
    #[cfg(test)]
    Memory(Mutex<Vec<String>>),
}

/// One background writer for the process, so lines from every listener keep
/// their order. It holds up to [`BUFFERED_LINES`] lines, then blocks the
/// request thread as Python's synchronous logging would, rather than
/// buffering without bound. The first line starts it; a server's shutdown
/// stops it (see [`flush_request_log`]) and the next line starts it again.
struct StdoutWriter {
    writer: tracing_appender::non_blocking::NonBlocking,
    /// Dropped to stop the writer, which writes out what is still buffered.
    _guard: tracing_appender::non_blocking::WorkerGuard,
}

/// Lines buffered before a request thread blocks on the write, as a request
/// blocks on Python's synchronous logging handler.
const BUFFERED_LINES: usize = 1024;

static STDOUT: Mutex<Option<StdoutWriter>> = Mutex::new(None);

impl StdoutWriter {
    fn new() -> Self {
        let (writer, guard) = tracing_appender::non_blocking::NonBlockingBuilder::default()
            .lossy(false)
            .buffered_lines_limit(BUFFERED_LINES)
            .thread_name("request-log")
            .finish(std::io::stdout());
        Self {
            writer,
            _guard: guard,
        }
    }
}

/// Write out the buffered request-log lines and stop the writer. Called when
/// a server shuts down or dumps before a crash; a line logged afterwards
/// starts a new writer.
pub(crate) fn flush_request_log() {
    // Held while the old writer drains, so no line can overtake its lines.
    drop(lock(&STDOUT).take());
}

impl LogSink {
    fn emit(&self, message: &str) {
        match self {
            Self::Stdout => {
                let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
                let line = format!("[{now}] {message}\n");
                let mut stdout = lock(&STDOUT);
                let writer = &mut stdout.get_or_insert_with(StdoutWriter::new).writer;
                let _ = writer.write_all(line.as_bytes());
            }
            #[cfg(test)]
            Self::Memory(lines) => lock(lines).push(message.to_owned()),
        }
    }
}

// --- Python's rendering (`_dataclass_to_string_truncated`) -----------------

/// Python's watermark redaction (`redact_watermark_secrets`), by key: the
/// secret fields anywhere, and a watermark config's keys. Python redacts only
/// with `--enable-watermark`; these lines always do.
const WATERMARK_SECRET_FIELDS: [&str; 3] = ["watermark_key", "watermark_key_b", "watermark_config"];

/// Whether the value under key `name` is a watermark secret; `in_watermark`
/// is set under a `watermark` key.
fn is_watermark_secret(name: &str, in_watermark: bool) -> bool {
    WATERMARK_SECRET_FIELDS.contains(&name) || (in_watermark && matches!(name, "key" | "key_b"))
}

/// One field or dict entry: the value, or `'<redacted>'` for a watermark
/// secret.
fn write_field(out: &mut String, name: &str, value: &Value, max_length: usize, in_watermark: bool) {
    if is_watermark_secret(name, in_watermark) {
        out.push_str("'<redacted>'");
    } else {
        write_truncated(out, value, max_length, in_watermark || name == "watermark");
    }
}

/// One value as `_dataclass_to_string_truncated(value, max_length)` prints it.
fn write_truncated(out: &mut String, value: &Value, max_length: usize, in_watermark: bool) {
    match value {
        Value::String(s) => {
            let len = s.chars().count();
            if len > max_length {
                let (head, tail) = halves(len, max_length);
                write_str_repr(out, &s.chars().take(head).collect::<String>());
                out.push_str(" ... ");
                write_str_repr(out, &s.chars().skip(tail).collect::<String>());
            } else {
                write_str_repr(out, s);
            }
        }
        Value::Array(items) if items.len() > max_length => {
            let (head, tail) = halves(items.len(), max_length);
            write_list(out, &items[..head], in_watermark);
            out.push_str(" ... ");
            write_list(out, &items[tail..], in_watermark);
        }
        Value::Object(map) => write_dict(out, map.iter(), max_length, in_watermark),
        _ => write_repr(out, value, in_watermark),
    }
}

/// `data[:half]` ends at the first index, `data[-half:]` starts at the second
/// (all of it when `half` is 0, as Python's `-0` slice is).
fn halves(len: usize, max_length: usize) -> (usize, usize) {
    let half = max_length / 2;
    (half, if half == 0 { 0 } else { len - half })
}

/// A dict level of `_dataclass_to_string_truncated`: keys quoted as they are,
/// values truncated in turn.
fn write_dict<'a>(
    out: &mut String,
    entries: impl Iterator<Item = (&'a String, &'a Value)>,
    max_length: usize,
    in_watermark: bool,
) {
    out.push('{');
    for (i, (name, value)) in entries.enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('\'');
        out.push_str(name);
        out.push_str("': ");
        write_field(out, name, value, max_length, in_watermark);
    }
    out.push('}');
}

fn write_list(out: &mut String, items: &[Value], in_watermark: bool) {
    out.push('[');
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write_repr(out, item, in_watermark);
    }
    out.push(']');
}

/// Python `repr` of the object `json.loads` makes of `value`, watermark
/// secrets redacted as [`write_field`] does.
fn write_repr(out: &mut String, value: &Value, in_watermark: bool) {
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) => {
                let _ = write!(out, "{i}");
            }
            (None, Some(u), _) => {
                let _ = write!(out, "{u}");
            }
            (None, None, Some(f)) => write_float(out, f),
            (None, None, None) => unreachable!("a JSON number is an integer or a float"),
        },
        Value::String(s) => write_str_repr(out, s),
        Value::Array(items) => write_list(out, items, in_watermark),
        Value::Object(map) => {
            out.push('{');
            for (i, (name, value)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str_repr(out, name);
                out.push_str(": ");
                if is_watermark_secret(name, in_watermark) {
                    out.push_str("'<redacted>'");
                } else {
                    write_repr(out, value, in_watermark || name == "watermark");
                }
            }
            out.push('}');
        }
    }
}

/// Python `repr(str)`: single quotes unless the text has one and no double
/// quote; the quote, backslash, `\t` `\n` `\r` and control characters escaped.
/// Other characters print as they are (Python also escapes non-printable
/// non-ASCII ones, such as U+2028).
fn write_str_repr(out: &mut String, s: &str) {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            // Control characters are all below U+00A0.
            c if c.is_control() => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
}

/// Python `repr(float)`: the shortest round-trip digits, positional unless
/// the exponent is below -4 or above 15 (`1e-05`, `1e+16`), with `.0` on a
/// positional integer.
fn write_float(out: &mut String, f: f64) {
    if f.is_nan() {
        out.push_str("nan");
        return;
    }
    if f.is_infinite() {
        out.push_str(if f > 0.0 { "inf" } else { "-inf" });
        return;
    }
    // Rust's `{:e}` prints the same shortest round-trip digits.
    let sci = format!("{f:e}");
    let (mantissa, exponent) = sci.split_once('e').expect("`{:e}` has an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` exponent is an integer");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(mantissa) => ("-", mantissa),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let point = exponent + 1; // digits before the decimal point
    out.push_str(sign);
    if !(-4 < point && point <= 16) {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    } else if point <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', point.unsigned_abs() as usize));
        out.push_str(&digits);
    } else if point as usize >= digits.len() {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', point as usize - digits.len()));
        out.push_str(".0");
    } else {
        out.push_str(&digits[..point as usize]);
        out.push('.');
        out.push_str(&digits[point as usize..]);
    }
}

/// Drop `keys` (the server-added ones) from a request's, or each batch item's,
/// `sampling_params.custom_params`.
fn omit_custom_params(sampling_params: &mut Value, keys: &[String]) {
    if keys.is_empty() {
        return;
    }
    let strip = |params: &mut Value| {
        if let Some(custom) = params
            .get_mut("custom_params")
            .and_then(Value::as_object_mut)
        {
            for key in keys {
                custom.remove(key);
            }
        }
    };
    match sampling_params {
        Value::Array(items) => items.iter_mut().for_each(strip),
        params => strip(params),
    }
}

// --- crash-dump recording ---------------------------------------------------

/// A poisoned lock still holds whole records (nothing panics halfway through
/// an update), and a crash dump must not lose them.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The finished requests of the last five minutes and every unfinished one,
/// as Python's `crash_dump_request_list` and `rid_to_state` hold them. One
/// lock covers both, so a request moves from one to the other atomically and
/// a snapshot sees it exactly once; JSON is written outside the lock.
#[derive(Default)]
struct CrashRecorder {
    state: Mutex<RecorderState>,
    next_id: AtomicU64,
    /// Server-added `custom_params` keys, left out of the bodies it writes.
    omit_custom_params: Vec<String>,
}

#[derive(Default)]
struct RecorderState {
    finished: VecDeque<FinishedRequest>,
    in_flight: HashMap<u64, InFlightRequest>,
}

/// The JSON object Python rebuilds requests from: a native body, shared by
/// the items of a batch, or the fields the core knows of a request it built.
pub(crate) struct DumpBody {
    id: u64,
    json: RawBody,
}

impl DumpBody {
    /// The body's JSON. With header overrides it is the parsed body with the
    /// overrides written in, serialized: what the request decoded. The
    /// server-added `omit` keys are left out of its `custom_params`.
    fn write(&self, out: &mut Vec<u8>, omit: &[String]) {
        let RawBody { bytes, overrides } = &self.json;
        // A body none of the keys can appear in is written as it came.
        let mentions =
            |key: &String| !key.is_empty() && bytes.windows(key.len()).any(|w| w == key.as_bytes());
        if overrides.is_none() && !omit.iter().any(mentions) {
            out.extend_from_slice(bytes);
            return;
        }
        let mut body: Value = serde_json::from_slice(bytes).expect("parsed when received");
        if let Some(fields) = body.as_object_mut() {
            if let Some(overrides) = overrides {
                fields.extend(overrides.clone());
            }
            if let Some(sampling_params) = fields.get_mut("sampling_params") {
                omit_custom_params(sampling_params, omit);
            }
        }
        serde_json::to_writer(out, &body).expect("a JSON value serializes");
    }
}

/// One request of a [`DumpBody`]: its batch item and the rid the client sees.
#[derive(Clone)]
struct DumpRequest {
    body: Arc<DumpBody>,
    batch_index: Option<usize>,
    rid: String,
}

struct FinishedRequest {
    request: DumpRequest,
    out: Bytes,
    created_time: f64,
    finished_time: f64,
}

struct InFlightRequest {
    request: DumpRequest,
    created_time: f64,
    output: Arc<Mutex<CoreOutput>>,
}

impl CrashRecorder {
    fn omitting(omit_custom_params: Vec<String>) -> Self {
        Self {
            omit_custom_params,
            ..Self::default()
        }
    }

    fn body(&self, json: RawBody) -> Arc<DumpBody> {
        Arc::new(DumpBody {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            json,
        })
    }

    fn register(
        &self,
        request: DumpRequest,
        created_time: f64,
        output: Arc<Mutex<CoreOutput>>,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        lock(&self.state).in_flight.insert(
            id,
            InFlightRequest {
                request,
                created_time,
                output,
            },
        );
        id
    }

    fn unregister(&self, id: u64) {
        lock(&self.state).in_flight.remove(&id);
    }

    /// Move a finished request to the window, evicting what finished five
    /// minutes before it (Python `record_request_for_crash_dump`).
    fn finish(&self, id: u64, log_metrics: bool, out: &Map<String, Value>, now: f64) {
        let out = log_metrics
            .then(|| Bytes::from(serde_json::to_vec(out).expect("a JSON map serializes")));
        let mut state = lock(&self.state);
        let Some(entry) = state.in_flight.remove(&id) else {
            return;
        };
        let Some(out) = out else {
            return;
        };
        state.finished.push_back(FinishedRequest {
            request: entry.request,
            out,
            created_time: entry.created_time,
            finished_time: now,
        });
        while state
            .finished
            .front()
            .is_some_and(|oldest| now - oldest.finished_time >= CRASH_DUMP_WINDOW_SECS)
        {
            state.finished.pop_front();
        }
    }

    /// `{"bodies": {id: body}, "finished": [record], "unfinished": [record]}`,
    /// each record `{"body", "batch_index", "rid", "out", "created_time",
    /// "finished_time"}` and each body written once, however many items of a
    /// batch name it.
    fn snapshot(&self, now: f64) -> Vec<u8> {
        // Capture under the lock (reference counts and the partial outputs),
        // write outside it.
        let (finished, unfinished) = {
            let state = lock(&self.state);
            let finished: Vec<(DumpRequest, Bytes, f64, f64)> = state
                .finished
                .iter()
                .map(|r| {
                    (
                        r.request.clone(),
                        r.out.clone(),
                        r.created_time,
                        r.finished_time,
                    )
                })
                .collect();
            let unfinished: Vec<(DumpRequest, Map<String, Value>, f64)> = state
                .in_flight
                .values()
                .map(|e| {
                    (
                        e.request.clone(),
                        partial_out(&lock(&e.output)),
                        e.created_time,
                    )
                })
                .collect();
            (finished, unfinished)
        };
        let mut bodies = std::collections::BTreeMap::new();
        let mut buf = Vec::new();
        buf.extend_from_slice(b"\"finished\":[");
        for (i, (request, out, created, finished)) in finished.iter().enumerate() {
            if i > 0 {
                buf.push(b',');
            }
            bodies.insert(request.body.id, request.body.clone());
            write_record(&mut buf, request, out, *created, *finished);
        }
        buf.extend_from_slice(b"],\"unfinished\":[");
        for (i, (request, out, created)) in unfinished.iter().enumerate() {
            if i > 0 {
                buf.push(b',');
            }
            bodies.insert(request.body.id, request.body.clone());
            let out = serde_json::to_vec(out).expect("a JSON map serializes");
            write_record(&mut buf, request, &out, *created, now);
        }
        buf.extend_from_slice(b"]}");

        let mut snapshot = b"{\"bodies\":{".to_vec();
        for (i, (id, body)) in bodies.iter().enumerate() {
            if i > 0 {
                snapshot.push(b',');
            }
            write!(snapshot, "\"{id}\":").expect("writes to a Vec");
            body.write(&mut snapshot, &self.omit_custom_params);
        }
        snapshot.extend_from_slice(b"},");
        snapshot.extend_from_slice(&buf);
        snapshot
    }
}

/// Python's `ReqState.get_crash_dump_output`: what was generated so far.
fn partial_out(output: &CoreOutput) -> Map<String, Value> {
    let mut out = Map::new();
    if !output.text.is_empty() {
        out.insert("text".into(), output.text.clone().into());
    }
    if !output.token_ids.is_empty() {
        out.insert("output_ids".into(), output.token_ids.clone().into());
    }
    out
}

fn write_record(buf: &mut Vec<u8>, request: &DumpRequest, out: &[u8], created: f64, finished: f64) {
    write!(buf, "{{\"body\":{},\"batch_index\":", request.body.id).expect("writes to a Vec");
    serde_json::to_writer(&mut *buf, &request.batch_index).expect("serializes");
    buf.extend_from_slice(b",\"rid\":");
    serde_json::to_writer(&mut *buf, &request.rid).expect("serializes");
    buf.extend_from_slice(b",\"out\":");
    buf.extend_from_slice(out);
    buf.extend_from_slice(b",\"created_time\":");
    serde_json::to_writer(&mut *buf, &created).expect("serializes");
    buf.extend_from_slice(b",\"finished_time\":");
    serde_json::to_writer(&mut *buf, &finished).expect("serializes");
    buf.push(b'}');
}

#[cfg(test)]
mod tests;
