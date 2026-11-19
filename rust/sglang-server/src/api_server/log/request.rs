//! Python `TokenizerManager`'s request log (`--log-requests`, in its text or
//! JSON format, to stdout and/or hourly-rotated files) and crash-dump
//! recording (`--crash-dump-folder`), taken at the same request boundaries:
//! `Receive` before any preprocessing, `Finish` on the terminal output
//! (scheduler aborts included), and, for a dump, the finished requests of the
//! last five minutes plus every unfinished one. From level 2 an OpenAI
//! request also gets its own line as received, before it is adapted
//! (Python's `serving_base`).
//!
//! A request is described in Python's `GenerateReqInput` terms: a native
//! `/generate` body as the client sent it, with the values Python's
//! `normalize_batch_and_arguments` fills in taken from the normalized
//! [`GenerateRequest`]; other entry points describe the request they built.
//! The lines are rendered with Python's rules (`request_logger.py`) from the
//! settings Python hands over
//! ([`RequestLogConfig`](crate::message::config::RequestLogConfig)). Strings print as
//! Python's `repr` does, except that non-ASCII characters print as they are
//! (Python escapes the rare non-printable ones). An OpenAI request's line is
//! its model's `model_dump(exclude_none=True)` at the top level (declared
//! fields in order, defaults filled in, the validators that set
//! `routed_dp_rank` and the chat `tool_choice`); nested objects print as the
//! client sent them.
//!
//! The crash dump itself is written by Python (`RustServer
//! .dump_requests_before_crash`), from [`RequestLog::crash_dump_snapshot`].

// TODO(rainj-me): revisit whether these lines must match Python's byte for byte;
// logging the same information would drop most of `py_object` and `logger`.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use serde_json::{Map, Value};

use crate::api_server::core::CoreOutput;
#[cfg(test)]
use crate::message::config::RequestLogConfig;
use crate::message::config::ServerArgs;
use crate::message::request::GenerateRequest;
use crate::message::response::{ChunkEvent, ResponseItem};

mod crash_recorder;
mod logger;
mod py_object;

use crash_recorder::{CrashRecorder, DumpBody, DumpRequest};
pub(crate) use logger::flush_request_log;
use logger::{LogSink, RequestLogger, Targets};
use py_object::{
    MEDIA_FIELDS, MediaSummary, PyObj, Views, batch_images, media_layout, omit_custom_params,
    typed_fields,
};

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
    /// This listener's data-parallel rank, which the core stamps on every
    /// output; `None` without data parallelism.
    dp_rank: Option<u32>,
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
    Body {
        snapshot: Option<BodySnapshot>,
        /// When an HTTP layer started the request before `/generate` ran
        /// (work an `HttpLayer` does before the handler).
        started_at: Option<Instant>,
    },
    /// Built by an adapter (OpenAI, gRPC) or by the core itself, with the
    /// headers of the HTTP request it came from.
    Typed(LoggedHeaders),
}

impl ReceivedAs {
    /// A built request with no HTTP headers (gRPC, the startup warmup).
    pub(crate) fn without_headers() -> Self {
        Self::Typed(LoggedHeaders::default())
    }

    /// When the request started: when an HTTP layer started it before
    /// `/generate` ran, else now. The `Finish` latency, `/generate`'s
    /// `e2e_latency` and the latency metrics count from it, as Python's
    /// TokenizerManager counts from before the work such a layer takes over
    /// from it.
    pub(crate) fn created_at(&self) -> Instant {
        match self {
            Self::Body {
                started_at: Some(at),
                ..
            } => *at,
            _ => Instant::now(),
        }
    }
}

/// Which OpenAI request model a body is logged as.
#[derive(Clone, Copy, Debug)]
pub(crate) enum OpenAiRequest {
    ChatCompletion,
    Completion,
}

impl RequestLog {
    /// `None` when neither `--log-requests` nor `--crash-dump-folder` is set;
    /// an error when a log file cannot be opened.
    pub(crate) fn from_server_args(args: &ServerArgs) -> Result<Option<Arc<Self>>, String> {
        if args.request_log.is_none() && !args.crash_dump {
            return Ok(None);
        }
        let logger = match args.request_log.clone() {
            Some(config) => {
                let targets = Arc::new(Targets::open(&config)?);
                Some(RequestLogger::new(config, LogSink::Targets(targets))?)
            }
            None => None,
        };
        Ok(Some(Arc::new(Self {
            logger,
            recorder: args
                .crash_dump
                .then(|| CrashRecorder::omitting(args.server_added_custom_params.clone())),
            server_added_custom_params: args.server_added_custom_params.clone(),
            default_priority: args.default_priority_value,
            weight_version: args.weight_version.as_deref().map(Arc::from),
            dp_rank: args.deployment_info.reported_dp_rank(),
            incremental_streaming_output: args.incremental_streaming_output,
        })))
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
            logger: config.map(|config| {
                RequestLogger::new(config, LogSink::Memory(Mutex::default())).expect("valid config")
            }),
            recorder: crash_dump.then(|| CrashRecorder::omitting(omitted.clone())),
            server_added_custom_params: omitted,
            default_priority: None,
            weight_version: Some(Arc::from("default")),
            dp_rank: None,
            incremental_streaming_output: false,
        })
    }

    #[cfg(test)]
    pub(crate) fn logged_lines(&self) -> Vec<String> {
        match &self.logger.as_ref().expect("logging").sink {
            LogSink::Memory(lines) => std::mem::take(&mut *lock(lines)),
            LogSink::Targets(_) => unreachable!("tests log to memory"),
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
                    } else if MEDIA_FIELDS.contains(&name.as_str()) {
                        // Its layout, for the `multimodal` summary and a
                        // batch's `modalities`.
                        media_layout(value)
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

    /// Whether OpenAI requests get their own line (from level 2).
    pub(crate) fn logs_openai_requests(&self) -> bool {
        self.logger
            .as_ref()
            .is_some_and(|logger| logger.config.openai_requests)
    }

    /// Log an OpenAI request's `body` as received (Python
    /// `log_openai_received_request`), when [`Self::logs_openai_requests`].
    /// A body that is not a JSON object is not logged.
    pub(crate) fn openai_received(
        &self,
        request: OpenAiRequest,
        body: &[u8],
        headers: &http::HeaderMap,
    ) {
        let Some(logger) = self
            .logger
            .as_ref()
            .filter(|logger| logger.config.openai_requests)
        else {
            return;
        };
        if let Ok(body) = serde_json::from_slice::<Map<String, Value>>(body) {
            logger.openai_received(request, body, &self.logged_headers(headers));
        }
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
        let created_at = received.created_at();
        let created_time = unix_time();
        let (obj, items, dumps, headers) = match received {
            ReceivedAs::Body {
                snapshot: Some(snapshot),
                ..
            } => self.body_views(snapshot, requests, batch),
            // A body that is not a JSON object fails to decode before this.
            ReceivedAs::Body { snapshot: None, .. } => {
                self.typed_views(requests, batch, LoggedHeaders::default())
            }
            ReceivedAs::Typed(headers) => self.typed_views(requests, batch, headers),
        };
        if let Some(logger) = &self.logger {
            logger.received(&obj, &headers);
        }
        items
            .into_iter()
            .zip(dumps)
            .zip(requests)
            .map(|((obj, dump), request)| {
                let output = Arc::new(Mutex::new(CoreOutput {
                    weight_version: self.weight_version.clone(),
                    dp_rank: self.dp_rank,
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
            let multimodal = MediaSummary::of(&item);
            items.push(PyObj {
                own: item,
                shared: shared.clone(),
                multimodal,
            });
            dumps.push(dump.clone().map(|body| (body, batch.then_some(i))));
        }
        let obj = if !logging {
            PyObj::default()
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
            PyObj {
                own: obj,
                shared: Arc::default(),
                // Python's normalization gives every item its media (one
                // image for all of them included), so the batch's summary is
                // the items' put together.
                multimodal: MediaSummary::sum(items.iter().map(|item| &item.multimodal)),
            }
        } else {
            items[0].clone()
        };
        (obj, items, dumps, headers)
    }

    /// Python's single-request normalization of the fields the request
    /// carries.
    fn normalize(&self, item: &mut Map<String, Value>, request: &GenerateRequest) {
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
        let items: Vec<PyObj> = items.into_iter().map(PyObj::from).collect();
        let obj = if batch {
            let mut obj = items.first().cloned().unwrap_or_default();
            obj.own.insert(
                "rid".into(),
                Value::Array(items.iter().map(|item| item.own["rid"].clone()).collect()),
            );
            obj.multimodal = MediaSummary::sum(items.iter().map(|item| &item.multimodal));
            obj
        } else {
            items.first().cloned().unwrap_or_default()
        };
        (obj, items, dumps, headers)
    }

    /// The crash dump's requests as JSON (see `CrashRecorder::snapshot`);
    /// `None` when not recording.
    pub(crate) fn crash_dump_snapshot(&self) -> Option<Vec<u8>> {
        self.recorder
            .as_ref()
            .map(|recorder| recorder.snapshot(unix_time()))
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
                let delta = self.delta(chunk);
                lock(&self.output).append_delta(&delta);
            }
            ResponseItem::Done(chunk) => {
                let delta = self.delta(chunk);
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

    /// `chunk` as an output of the call, which the core stamps with the
    /// listener's DP rank.
    fn delta(&self, chunk: &ChunkEvent) -> CoreOutput {
        CoreOutput {
            dp_rank: self.log.dp_rank,
            ..CoreOutput::from(chunk.clone())
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

/// A poisoned lock still holds whole records (nothing panics halfway through
/// an update), and a crash dump must not lose them.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::finish_reason::{AbortReason, FinishKind, FinishReason};
    use crate::message::ids::Rid;
    use crate::message::request::{GenerationPolicy, into_requests};
    use sglang_api_types::api::v1 as api;

    /// Python's lines for the same requests, from
    /// `test/registered/unit/utils/test_rust_request_log.py`, which checks this
    /// file against Python's `RequestLogger`.
    pub(super) const FIXTURE: &str = include_str!("../../../testdata/request_log.json");

    /// The settings Python hands over in `format` at `level`.
    pub(super) fn config_in(fixture: &Value, format: &str, level: &str) -> RequestLogConfig {
        let settings = &fixture["levels"][format][level];
        let strings = |key: &str| -> Vec<String> {
            serde_json::from_value(settings[key].clone()).expect("a string list")
        };
        let openai_requests = settings["openai_requests"].as_bool().unwrap();
        let openai_fields = |key: &str| match openai_requests {
            true => serde_json::from_value(fixture["openai_fields"][key].clone()).unwrap(),
            false => Vec::new(),
        };
        RequestLogConfig {
            max_length: settings["max_length"].as_u64().unwrap() as usize,
            skip_names: strings("skip_names"),
            out_skip_names: strings("out_skip_names"),
            json: settings["json"].as_bool().unwrap(),
            class_name: settings["class_name"].as_str().unwrap().into(),
            fields: serde_json::from_value(settings["fields"].clone()).unwrap(),
            batch_item_fields: strings("batch_item_fields"),
            batch_shared_fields: strings("batch_shared_fields"),
            headers: strings("headers"),
            exceeded_ms: settings["exceeded_ms"].as_i64().unwrap(),
            openai_requests,
            chat_completion_fields: openai_fields("chat_completion_fields"),
            completion_fields: openai_fields("completion_fields"),
            stdout: true,
            files: Vec::new(),
        }
    }

    pub(super) fn config(fixture: &Value, level: &str) -> RequestLogConfig {
        config_in(fixture, "text", level)
    }

    /// A body through the native `/generate` path: snapshot, then normalization.
    pub(super) fn receive_body(
        log: &Arc<RequestLog>,
        body: &Value,
        headers: &http::HeaderMap,
    ) -> Vec<RequestRecord> {
        let snapshot = log.snapshot_body(body, None, headers);
        let decoded: api::GenerateRequest = serde_json::from_value(body.clone()).unwrap();
        let (requests, batch) = into_requests(decoded, &GenerationPolicy::default()).unwrap();
        let received = ReceivedAs::Body {
            snapshot,
            started_at: None,
        };
        log.receive(&requests, received, batch, true)
    }

    pub(super) fn chunk(
        rid: &Rid,
        text: &str,
        ids: &[i64],
        finish: Option<FinishReason>,
    ) -> ChunkEvent {
        ChunkEvent {
            rid: rid.clone(),
            token_ids: ids.to_vec(),
            text: text.into(),
            completion_tokens: ids.len() as u64,
            prompt_tokens: 3,
            finish_reason: finish,
            ..Default::default()
        }
    }

    pub(super) fn level1(fixture: &Value) -> RequestLogConfig {
        config(fixture, "1")
    }

    /// The record follows the call: `Finish` (abort included) and the crash-dump
    /// entry come from the scheduler's terminal output.
    #[test]
    fn terminal_output_logs_finish_and_keeps_the_request() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let log = RequestLog::for_test(Some(level1(&fixture)), true);
        let body = serde_json::json!({"rid": "r-abort", "text": "hi"});
        let mut record = receive_body(&log, &body, &http::HeaderMap::new()).remove(0);
        let rid = Rid::from_client("r-abort");
        record.observe(&ResponseItem::Frame(chunk(&rid, "a", &[1], None)));
        let abort = FinishReason::from(FinishKind::Abort(Box::new(AbortReason {
            message: Some("Abort in waiting queue".into()),
            status_code: Some(503),
            err_type: None,
        })));
        record.observe(&ResponseItem::Done(chunk(&rid, "b", &[2], Some(abort))));

        let lines = log.logged_lines();
        assert!(lines[0].starts_with("Receive: obj=GenerateReqInput(rid='r-abort', "));
        let finish = &lines[1];
        assert!(finish.starts_with("Finish: obj=GenerateReqInput(rid='r-abort', "));
        assert!(
            finish.contains(", out={'meta_info': {'id': 'r-abort', ")
                && finish.contains(
                    "'finish_reason': {'type': 'abort', 'message': 'Abort in waiting queue', \
                     'status_code': 503, 'err_type': None}"
                ),
            "{finish}"
        );

        let snapshot: Value = serde_json::from_slice(&log.crash_dump_snapshot().unwrap()).unwrap();
        assert_eq!(snapshot["unfinished"], serde_json::json!([]));
        let finished = &snapshot["finished"][0];
        assert_eq!(snapshot["bodies"][finished["body"].to_string()], body);
        assert_eq!(finished["batch_index"], Value::Null);
        assert_eq!(finished["rid"], "r-abort");
        assert_eq!(finished["out"]["text"], "ab");
        assert_eq!(finished["out"]["output_ids"], serde_json::json!([1, 2]));
        assert!(finished["finished_time"].as_f64() >= finished["created_time"].as_f64());
    }

    /// A batch logs one `Receive` with every rid and one `Finish` per item, and
    /// each item's dump names its index in the shared body.
    #[test]
    fn batch_items_log_and_dump_separately() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let log = RequestLog::for_test(Some(level1(&fixture)), true);
        let body = serde_json::json!({
            "rid": ["a", "b"],
            "text": ["x", "y"],
            "sampling_params": [{"max_new_tokens": 1}, {"max_new_tokens": 2}],
        });
        let mut records = receive_body(&log, &body, &http::HeaderMap::new());
        let receive = log.logged_lines().remove(0);
        assert!(receive.contains("rid=['a', 'b'], "), "{receive}");
        assert!(
            receive.contains("sampling_params=[{'max_new_tokens': 1}, {'max_new_tokens': 2}], "),
            "{receive}"
        );
        let rid = Rid::from_client("b");
        records[1].observe(&ResponseItem::Done(chunk(&rid, "z", &[3], None)));
        let finish = log.logged_lines().remove(0);
        assert!(finish.contains("(rid='b', "), "{finish}");
        assert!(
            finish.contains("sampling_params={'max_new_tokens': 2}, "),
            "{finish}"
        );

        // Both items name one copy of the body.
        let snapshot: Value = serde_json::from_slice(&log.crash_dump_snapshot().unwrap()).unwrap();
        assert_eq!(snapshot["bodies"].as_object().unwrap().len(), 1);
        let (finished, unfinished) = (&snapshot["finished"][0], &snapshot["unfinished"][0]);
        assert_eq!(finished["batch_index"], 1);
        assert_eq!(unfinished["batch_index"], 0);
        assert_eq!(unfinished["rid"], "a");
        assert_eq!(finished["body"], unfinished["body"]);
        assert_eq!(snapshot["bodies"][finished["body"].to_string()], body);
    }
}
