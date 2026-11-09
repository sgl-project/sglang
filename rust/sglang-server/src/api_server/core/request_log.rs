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
//! settings Python hands over ([`RequestLogConfig`]). Strings print as
//! Python's `repr` does, except that non-ASCII characters print as they are
//! (Python escapes the rare non-printable ones). An OpenAI request's line is
//! its model's `model_dump(exclude_none=True)` at the top level (declared
//! fields in order, defaults filled in, the validators that set
//! `routed_dp_rank` and the chat `tool_choice`); nested objects print as the
//! client sent them.
//!
//! The crash dump itself is written by Python (`RustServer
//! .dump_requests_before_crash`), from [`RequestLog::crash_dump_snapshot`].

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use serde::ser::{SerializeMap, SerializeSeq}; // codespell:ignore ser
use serde_json::ser::Formatter; // codespell:ignore ser
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
            logger.received(&obj, &headers);
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
        // The deprecated `data_parallel_rank` moves to `routed_dp_rank`, as
        // `/generate` decodes it.
        crate::message::request::move_deprecated_dp_rank(item);
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

/// `(Receive object, per-request Finish objects, per-request dump sources,
/// headers)`.
type Views = (
    PyObj,
    Vec<PyObj>,
    Vec<Option<(Arc<DumpBody>, Option<usize>)>>,
    LoggedHeaders,
);

/// A request's Python object: its own fields over the ones its batch shares,
/// which every item references as Python's `obj[i]` does.
#[derive(Clone, Default)]
struct PyObj {
    own: Map<String, Value>,
    shared: Arc<Map<String, Value>>,
    /// Python `_summarize_multimodal` of its media fields.
    multimodal: MediaSummary,
}

impl PyObj {
    fn get(&self, name: &str) -> Option<&Value> {
        self.own.get(name).or_else(|| self.shared.get(name))
    }
}

impl From<Map<String, Value>> for PyObj {
    fn from(own: Map<String, Value>) -> Self {
        Self {
            multimodal: MediaSummary::of(&own),
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

/// The media fields, in the order Python's `_summarize_multimodal` reads
/// them.
const MEDIA_FIELDS: [&str; 3] = ["image_data", "video_data", "audio_data"];

/// What the log reads of a media field at a level that does not show it: its
/// nesting (`None`s included) with each item replaced by its payload length,
/// for the `multimodal` summary and a batch's `modalities`.
fn media_layout(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(media_layout).collect()),
        Value::Null => Value::Null,
        item => payload_len(item).into(),
    }
}

/// Python `_media_payload_len`: the characters of an item's data URI, URL or
/// path, bare or under `url` (else `data`); a layout holds the length.
fn payload_len(item: &Value) -> u64 {
    match item {
        Value::String(s) => s.chars().count() as u64,
        Value::Object(map) => map
            .get("url")
            .or_else(|| map.get("data"))
            .and_then(Value::as_str)
            .map_or(0, |s| s.chars().count() as u64),
        Value::Number(len) => len.as_u64().unwrap_or(0),
        _ => 0,
    }
}

/// The modalities of [`MEDIA_FIELDS`], as the summary names them.
const MODALITIES: [&str; 3] = ["image", "video", "audio"];

/// Python `_summarize_multimodal`: per media field, the item count and the
/// total payload length (zero items for a field the request leaves out),
/// and the order the summary dict lists them in.
#[derive(Clone)]
struct MediaSummary {
    slots: [(u64, u64); 3],
    order: [usize; 3],
}

impl Default for MediaSummary {
    fn default() -> Self {
        Self {
            slots: [(0, 0); 3],
            order: [0, 1, 2],
        }
    }
}

impl MediaSummary {
    fn of(fields: &Map<String, Value>) -> Self {
        let mut summary = Self::default();
        for (slot, name) in summary.slots.iter_mut().zip(MEDIA_FIELDS) {
            if let Some(value) = fields.get(name) {
                count_media(value, 0, slot);
            }
        }
        summary
    }

    /// Python `_summarize_openai_multimodal` of an OpenAI request's dump: its
    /// messages' media content parts, each modality listed where it first
    /// appears. Nested objects are as the client sent them, so an
    /// `input_audio` part counts the data URI its validation turns it into.
    fn of_openai(dump: &Map<String, Value>) -> Self {
        let mut summary = Self::default();
        let mut seen = 0;
        let Some(Value::Array(messages)) = dump.get("messages") else {
            return summary;
        };
        let parts = messages
            .iter()
            .filter_map(|message| match message.get("content") {
                Some(Value::Array(parts)) => Some(parts),
                _ => None,
            })
            .flatten();
        for part in parts {
            let Some(part_type) = part.get("type").and_then(Value::as_str) else {
                continue;
            };
            let (modality, len) = match (part_type, part.get(part_type)) {
                ("image_url", payload) => (0, payload.map_or(0, payload_len)),
                ("video_url", payload) => (1, payload.map_or(0, payload_len)),
                ("audio_url", payload) => (2, payload.map_or(0, payload_len)),
                ("input_audio", payload) => (2, payload.map_or(0, inline_audio_len)),
                _ => continue,
            };
            if summary.slots[modality].0 == 0 {
                summary.order[seen] = modality;
                seen += 1;
            }
            summary.slots[modality].0 += 1;
            summary.slots[modality].1 += len;
        }
        // The modalities never seen keep their places after the seen ones.
        for modality in 0..3 {
            if summary.slots[modality].0 == 0 {
                summary.order[seen] = modality;
                seen += 1;
            }
        }
        summary
    }

    fn sum<'a>(summaries: impl Iterator<Item = &'a Self>) -> Self {
        let mut total = Self::default();
        for summary in summaries {
            for (slot, (count, len)) in total.slots.iter_mut().zip(summary.slots) {
                slot.0 += count;
                slot.1 += len;
            }
        }
        total
    }

    fn is_empty(&self) -> bool {
        self.slots.iter().all(|(count, _)| *count == 0)
    }

    /// `(modality, count, payload length)` of each modality with any items,
    /// in the summary dict's order.
    fn entries(&self) -> impl Iterator<Item = (&'static str, u64, u64)> + '_ {
        self.order.iter().filter_map(|&modality| {
            let (count, len) = self.slots[modality];
            (count > 0).then_some((MODALITIES[modality], count, len))
        })
    }
}

/// The summary dict, for the JSON format.
impl serde::Serialize for MediaSummary {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.entries().map(|(name, count, len)| {
            (
                name,
                serde_json::json!({"count": count, "payload_len": len}),
            )
        }))
    }
}

/// The characters of the data URI an `input_audio` part validates into
/// (`_to_audio_url_part`): `data:<mime>;base64,` and its data.
fn inline_audio_len(audio: &Value) -> u64 {
    let data = audio
        .get("data")
        .and_then(Value::as_str)
        .map_or(0, |data| data.chars().count() as u64);
    let mime = match audio.get("format").and_then(Value::as_str) {
        Some("wav") => "audio/wav",
        Some("mp3") => "audio/mpeg",
        _ => return data,
    };
    ("data:;base64,".len() + mime.len()) as u64 + data
}

/// Python counts a field's items and, for a batch, each item's leaves; a
/// `None` at either level is skipped, and anything deeper is one item.
fn count_media(value: &Value, depth: usize, slot: &mut (u64, u64)) {
    match value {
        Value::Null => {}
        Value::Array(items) if depth < 2 => {
            for item in items {
                count_media(item, depth + 1, slot);
            }
        }
        item => {
            slot.0 += 1;
            slot.1 += payload_len(item);
        }
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
            if let Some(detail) = &hints.detail {
                item.insert("detail".into(), detail.as_str().into());
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
    /// The JSON format's `config.fields` defaults, parsed (empty for text).
    json_defaults: Vec<Value>,
    /// The OpenAI request models' fields with their defaults (`Null` for
    /// none).
    chat_completion_fields: Vec<(String, Value)>,
    completion_fields: Vec<(String, Value)>,
    sink: LogSink,
}

impl RequestLogger {
    fn new(config: RequestLogConfig, sink: LogSink) -> Result<Self, String> {
        let set = |names: &[String]| names.iter().cloned().collect();
        let parse = |name: &str, json: &str| {
            serde_json::from_str(json)
                .map_err(|e| format!("request log: field {name} has an invalid default: {e}"))
        };
        let model_fields = |fields: &[(String, Option<String>)]| {
            fields
                .iter()
                .map(|(name, default)| {
                    let default = match default {
                        Some(json) => parse(name, json)?,
                        None => Value::Null,
                    };
                    Ok((name.clone(), default))
                })
                .collect::<Result<Vec<_>, String>>()
        };
        let json_defaults = if config.json {
            config
                .fields
                .iter()
                .map(|(name, json)| parse(name, json))
                .collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };
        Ok(Self {
            skip_names: set(&config.skip_names),
            out_skip_names: set(&config.out_skip_names),
            batch_item_fields: set(&config.batch_item_fields),
            batch_shared_fields: set(&config.batch_shared_fields),
            json_defaults,
            chat_completion_fields: model_fields(&config.chat_completion_fields)?,
            completion_fields: model_fields(&config.completion_fields)?,
            config,
            sink,
        })
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
        let line = if self.config.json {
            self.json_line("request.received", obj, headers, None)
        } else {
            self.received_line(obj, headers)
        };
        self.sink.emit(&line);
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
        let line = if self.config.json {
            self.json_line("request.finished", obj, headers, Some(out))
        } else {
            self.finished_line(obj, headers, out)
        };
        self.sink.emit(&line);
    }

    /// Python `log_openai_received_request`'s message.
    fn openai_received(
        &self,
        request: OpenAiRequest,
        body: Map<String, Value>,
        headers: &[(String, String)],
    ) {
        let obj = self.openai_dump(request, body);
        let multimodal = MediaSummary::of_openai(&obj);
        let max_length = self.config.max_length;
        let line = if self.config.json {
            let obj = JsonObject::transformed(obj.iter(), max_length);
            let mut line = JsonLine::new("request.received.openai", obj, headers);
            line.multimodal = (!multimodal.is_empty()).then_some(&multimodal);
            line.render()
        } else {
            let mut line = String::from("Receive OpenAI: obj=");
            write_dict(&mut line, obj.iter(), max_length, false);
            write_multimodal(&mut line, &multimodal);
            write_headers(&mut line, headers);
            line
        };
        self.sink.emit(&line);
    }

    /// The request model's `model_dump(exclude_none=True)` of `body`: its
    /// fields in declaration order, the client's value or the default, the
    /// `None`s and undeclared keys left out; after the `mode="before"`
    /// validators that set top-level fields.
    fn openai_dump(
        &self,
        request: OpenAiRequest,
        mut body: Map<String, Value>,
    ) -> Map<String, Value> {
        let is_null = |value: Option<&Value>| value.is_none_or(Value::is_null);
        // `_migrate_deprecated_dp_rank`.
        if !is_null(body.get("data_parallel_rank")) && is_null(body.get("routed_dp_rank")) {
            let rank = body["data_parallel_rank"].clone();
            body.insert("routed_dp_rank".into(), rank);
        }
        let fields = match request {
            OpenAiRequest::ChatCompletion => {
                // `set_tool_choice_default`.
                if is_null(body.get("tool_choice")) {
                    let tools = !is_null(body.get("tools"))
                        || has_message_level_tools(body.get("messages"));
                    let choice = if tools { "auto" } else { "none" };
                    body.insert("tool_choice".into(), choice.into());
                }
                &self.chat_completion_fields
            }
            OpenAiRequest::Completion => &self.completion_fields,
        };
        fields
            .iter()
            .filter_map(|(name, default)| {
                let value = body.remove(name).unwrap_or_else(|| default.clone());
                (!value.is_null()).then(|| (name.clone(), value))
            })
            .collect()
    }

    /// Python `log_json`'s object for a request line: the request object
    /// (each field's value or the class default), its headers and, for
    /// `Finish`, the output.
    fn json_line(
        &self,
        event: &str,
        obj: &PyObj,
        headers: &[(String, String)],
        out: Option<&Map<String, Value>>,
    ) -> String {
        let max_length = self.config.max_length;
        let fields = self
            .config
            .fields
            .iter()
            .zip(&self.json_defaults)
            .filter(|((name, _), _)| !self.skip_names.contains(name))
            .map(|((name, _), default)| match obj.get(name) {
                Some(value) => (name.as_str(), value, false),
                None => (name.as_str(), default, true),
            })
            .collect();
        let obj_fields = JsonObject { fields, max_length };
        let mut line = JsonLine::new(event, obj_fields, headers);
        line.rid = Some(obj.get("rid").unwrap_or(&Value::Null));
        line.multimodal = (!obj.multimodal.is_empty()).then_some(&obj.multimodal);
        line.out = out.map(|out| {
            let kept = out
                .iter()
                .filter(|(name, _)| !self.out_skip_names.contains(name.as_str()));
            JsonObject::transformed(kept, max_length)
        });
        line.render()
    }

    /// Python `log_received_request`'s message.
    fn received_line(&self, obj: &PyObj, headers: &[(String, String)]) -> String {
        let mut line = String::from("Receive: obj=");
        self.write_obj(&mut line, obj);
        write_multimodal(&mut line, &obj.multimodal);
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
        write_multimodal(&mut line, &obj.multimodal);
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

/// `, multimodal={...}` when the request carries media: Python's summary
/// dict, a modality's item count and payload length for each with any.
fn write_multimodal(line: &mut String, summary: &MediaSummary) {
    if summary.is_empty() {
        return;
    }
    line.push_str(", multimodal={");
    for (i, (name, count, len)) in summary.entries().enumerate() {
        if i > 0 {
            line.push_str(", ");
        }
        let _ = write!(line, "'{name}': {{'count': {count}, 'payload_len': {len}}}");
    }
    line.push('}');
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

/// Python `datetime.isoformat()`: microseconds unless they are zero.
fn iso_timestamp(at: chrono::DateTime<chrono::Local>) -> String {
    if at.timestamp_subsec_micros() == 0 {
        at.format("%Y-%m-%dT%H:%M:%S").to_string()
    } else {
        at.format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
    }
}

/// Python `ChatCompletionRequest.set_tool_choice_default`'s `has_message_level_tools` check.
fn has_message_level_tools(messages: Option<&Value>) -> bool {
    let Some(Value::Array(messages)) = messages else {
        return false;
    };
    messages.iter().any(|message| {
        let role = message.get("role").and_then(Value::as_str);
        role.is_some_and(|role| {
            matches!(
                role.to_lowercase().as_str(),
                "root" | "system" | "developer"
            )
        }) && message.get("tools").is_some_and(is_truthy)
    })
}

/// Python truthiness of a JSON value.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// Where the lines go: the process's request-log writer (see [`Writer`]),
/// each line prefixed with the local time as Python's request-log formatter
/// does.
enum LogSink {
    Targets(Arc<Targets>),
    #[cfg(test)]
    Memory(Mutex<Vec<String>>),
}

/// `--log-requests-target`: stdout, and the files Python names for the
/// target directories, opened at startup as Python's handlers are.
struct Targets {
    stdout: bool,
    paths: Vec<String>,
    /// The files at `paths`; only the writer thread writes them.
    files: Mutex<Vec<RotatingFile>>,
}

impl Targets {
    fn open(config: &RequestLogConfig) -> Result<Self, String> {
        let files = config
            .files
            .iter()
            .map(|path| RotatingFile::open(PathBuf::from(path), SystemTime::now()))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            stdout: config.stdout,
            paths: config.files.clone(),
            files: Mutex::new(files),
        })
    }

    /// Whether `other` writes to the same places.
    fn same_as(&self, other: &Self) -> bool {
        self.stdout == other.stdout && self.paths == other.paths
    }
}

/// The writer thread's output: each line to every target. A target that
/// fails does not keep the line from the others; a file's first failure is
/// reported.
struct TargetsWriter(Arc<Targets>);

impl std::io::Write for TargetsWriter {
    fn write(&mut self, line: &[u8]) -> std::io::Result<usize> {
        if self.0.stdout {
            let _ = std::io::stdout().write_all(line);
        }
        let now = SystemTime::now();
        for file in lock(&self.0.files).iter_mut() {
            if let Err(error) = file.write_line(line, now)
                && !std::mem::replace(&mut file.failed, true)
            {
                let path = file.path.display();
                tracing::warn!(%path, %error, "request log file write failed; not reported again");
            }
        }
        Ok(line.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.0.stdout {
            std::io::stdout().flush()?;
        }
        Ok(())
    }
}

/// How often `TimedRotatingFileHandler(when="H")` rolls a file over.
const ROTATE_INTERVAL: Duration = Duration::from_secs(3600);

/// A log file in append mode that `TimedRotatingFileHandler(when="H",
/// backupCount=0)` rolls over: an hour after the existing file was last
/// modified (or after it was created), then an hour after each rollover (not
/// on the hour), renaming it `<name>.%Y-%m-%d_%H` after the hour it started
/// and keeping every old one.
struct RotatingFile {
    path: PathBuf,
    /// Closed by a rollover; the next line opens the file again, so a failed
    /// reopen is retried rather than losing the file for good.
    file: Option<File>,
    rollover_at: SystemTime,
    /// A write failed (and was reported).
    failed: bool,
}

impl RotatingFile {
    fn open(path: PathBuf, now: SystemTime) -> Result<Self, String> {
        let modified = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .unwrap_or(now);
        let modified = whole_seconds(modified);
        let file = open_append(&path)
            .map_err(|e| format!("cannot open request log file {}: {e}", path.display()))?;
        Ok(Self {
            path,
            file: Some(file),
            rollover_at: modified + ROTATE_INTERVAL,
            failed: false,
        })
    }

    fn write_line(&mut self, line: &[u8], now: SystemTime) -> std::io::Result<()> {
        let now = whole_seconds(now);
        if now >= self.rollover_at {
            self.roll_over(now, local_offset)?;
        }
        let file = match &mut self.file {
            Some(file) => file,
            None => self.file.insert(open_append(&self.path)?),
        };
        file.write_all(line)
    }

    /// Python's `doRollover`: the file is renamed after the local hour its
    /// interval started, shifted by an hour across a DST change (toward the
    /// offset now, whatever size the change) as Python shifts it. A file
    /// already gone is not
    /// renamed (Python's `rotate`). Python takes a name that already exists as
    /// already rolled over and leaves the file be; so does this, instead of
    /// replacing that file, and tries again an hour later (Python never
    /// does). `offset_at` is the local UTC offset at a time.
    fn roll_over(
        &mut self,
        now: SystemTime,
        offset_at: impl Fn(SystemTime) -> chrono::FixedOffset,
    ) -> std::io::Result<()> {
        let started = self.rollover_at - ROTATE_INTERVAL;
        let change = offset_at(now).local_minus_utc() - offset_at(started).local_minus_utc();
        let named = chrono::DateTime::<chrono::Utc>::from(started)
            + chrono::TimeDelta::hours(i64::from(change.signum()));
        let named = named.with_timezone(&offset_at(named.into()));
        let mut rotated = self.path.clone().into_os_string();
        rotated.push(named.format(".%Y-%m-%d_%H").to_string());
        if !std::path::Path::new(&rotated).exists() {
            // Closed first, as Python closes it.
            self.file = None;
            match std::fs::rename(&self.path, &rotated) {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error),
                _ => {}
            }
        }
        self.rollover_at = now + ROTATE_INTERVAL;
        Ok(())
    }
}

/// `at` in whole seconds, as Python's handler reads the clock and the file
/// time (`int(...)`).
fn whole_seconds(at: SystemTime) -> SystemTime {
    let seconds = at
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    UNIX_EPOCH + Duration::from_secs(seconds)
}

/// The local UTC offset at `at`.
fn local_offset(at: SystemTime) -> chrono::FixedOffset {
    use chrono::Offset as _;
    chrono::DateTime::<chrono::Local>::from(at).offset().fix()
}

fn open_append(path: &std::path::Path) -> std::io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

/// One background writer for the process, so lines from every listener keep
/// their order. It holds up to [`BUFFERED_LINES`] lines, then blocks the
/// request thread as Python's synchronous logging would, rather than
/// buffering without bound. The first line starts it; a server's shutdown
/// stops it (see [`flush_request_log`]) and the next line starts it again.
struct Writer {
    writer: tracing_appender::non_blocking::NonBlocking,
    /// The targets it writes to.
    targets: Arc<Targets>,
    /// Dropped to stop the writer, which writes out what is still buffered.
    _guard: tracing_appender::non_blocking::WorkerGuard,
}

/// Lines buffered before a request thread blocks on the write, as a request
/// blocks on Python's synchronous logging handler.
const BUFFERED_LINES: usize = 1024;

static WRITER: Mutex<Option<Writer>> = Mutex::new(None);

impl Writer {
    fn new(targets: Arc<Targets>) -> Self {
        let (writer, guard) = tracing_appender::non_blocking::NonBlockingBuilder::default()
            .lossy(false)
            .buffered_lines_limit(BUFFERED_LINES)
            .thread_name("request-log")
            .finish(TargetsWriter(targets.clone()));
        Self {
            writer,
            targets,
            _guard: guard,
        }
    }
}

/// Write out the buffered request-log lines and stop the writer. Called when
/// a server shuts down or dumps before a crash; a line logged afterwards
/// starts a new writer.
pub(crate) fn flush_request_log() {
    // Held while the old writer drains, so no line can overtake its lines.
    drop(lock(&WRITER).take());
}

impl LogSink {
    fn emit(&self, message: &str) {
        match self {
            Self::Targets(targets) => {
                let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
                let line = format!("[{now}] {message}\n");
                let mut writer = lock(&WRITER);
                // A server started in the same process after another one's
                // shutdown may write elsewhere.
                if !writer
                    .as_ref()
                    .is_some_and(|writer| writer.targets.same_as(targets))
                {
                    *writer = Some(Writer::new(targets.clone()));
                }
                let writer = &mut writer.as_mut().expect("set above").writer;
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

// --- Python's JSON format (`_transform_data_for_logging`, `log_json`) -------

/// One field or dict entry as `_transform_data_for_logging` returns it,
/// serialized without building it: `"<redacted>"` for a watermark secret (as
/// [`write_field`] redacts), else the [`Transformed`] value.
struct TransformedField<'a> {
    name: &'a str,
    value: &'a Value,
    max_length: usize,
    in_watermark: bool,
}

impl serde::Serialize for TransformedField<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if is_watermark_secret(self.name, self.in_watermark) {
            return serializer.serialize_str("<redacted>");
        }
        let in_watermark = self.in_watermark || self.name == "watermark";
        Transformed::new(self.value, self.max_length, in_watermark).serialize(serializer)
    }
}

/// A value as `_transform_data_for_logging` returns it, serialized without
/// building it: a string or list longer than `max_length` keeps its two
/// halves around `"..."`, at every depth.
struct Transformed<'a> {
    value: &'a Value,
    max_length: usize,
    in_watermark: bool,
}

impl<'a> Transformed<'a> {
    fn new(value: &'a Value, max_length: usize, in_watermark: bool) -> Self {
        Self {
            value,
            max_length,
            in_watermark,
        }
    }
}

impl serde::Serialize for Transformed<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let max_length = self.max_length;
        match self.value {
            Value::String(s) => {
                let len = s.chars().count();
                if len <= max_length {
                    return serializer.serialize_str(s);
                }
                let (head, tail) = halves(len, max_length);
                let mut cut: String = s.chars().take(head).collect();
                cut.push_str("...");
                cut.extend(s.chars().skip(tail));
                serializer.serialize_str(&cut)
            }
            Value::Array(items) => {
                let item = |item| Transformed::new(item, max_length, self.in_watermark);
                let mut seq = serializer.serialize_seq(None)?;
                if items.len() > max_length {
                    let (head, tail) = halves(items.len(), max_length);
                    for value in &items[..head] {
                        seq.serialize_element(&item(value))?;
                    }
                    seq.serialize_element("...")?;
                    for value in &items[tail..] {
                        seq.serialize_element(&item(value))?;
                    }
                } else {
                    for value in items {
                        seq.serialize_element(&item(value))?;
                    }
                }
                seq.end()
            }
            Value::Object(map) => {
                let mut entries = serializer.serialize_map(Some(map.len()))?;
                for (name, value) in map {
                    entries.serialize_entry(
                        name,
                        &TransformedField {
                            name,
                            value,
                            max_length,
                            in_watermark: self.in_watermark,
                        },
                    )?;
                }
                entries.end()
            }
            scalar => scalar.serialize(serializer),
        }
    }
}

/// A JSON-format line: `log_json`'s `timestamp` and `event`, then the
/// logger's fields in its order.
#[derive(serde::Serialize)]
struct JsonLine<'a> {
    timestamp: String,
    event: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    rid: Option<&'a Value>,
    obj: JsonObject<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    multimodal: Option<&'a MediaSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    headers: Option<JsonHeaders<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    out: Option<JsonObject<'a>>,
}

impl<'a> JsonLine<'a> {
    fn new(event: &'a str, obj: JsonObject<'a>, headers: &'a [(String, String)]) -> Self {
        Self {
            timestamp: iso_timestamp(chrono::Local::now()),
            event,
            rid: None,
            obj,
            multimodal: None,
            headers: (!headers.is_empty()).then_some(JsonHeaders(headers)),
            out: None,
        }
    }

    /// `json.dumps(line, ensure_ascii=False)`.
    fn render(&self) -> String {
        let mut out = Vec::new();
        let mut serializer = serde_json::Serializer::with_formatter(&mut out, PythonJsonFormatter);
        serde::Serialize::serialize(self, &mut serializer).expect("a log line serializes");
        String::from_utf8(out).expect("serde_json writes UTF-8")
    }
}

/// The whitelisted headers as a JSON object.
struct JsonHeaders<'a>(&'a [(String, String)]);

impl serde::Serialize for JsonHeaders<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().map(|(name, value)| (name, value)))
    }
}

/// A dict or dataclass as `_transform_data_for_logging` returns it: each
/// field's value transformed, or (`verbatim`) already Python's.
struct JsonObject<'a> {
    fields: Vec<(&'a str, &'a Value, bool)>,
    max_length: usize,
}

impl<'a> JsonObject<'a> {
    fn transformed(
        entries: impl Iterator<Item = (&'a String, &'a Value)>,
        max_length: usize,
    ) -> Self {
        Self {
            fields: entries
                .map(|(name, value)| (name.as_str(), value, false))
                .collect(),
            max_length,
        }
    }
}

impl serde::Serialize for JsonObject<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.fields.len()))?;
        for &(name, value, verbatim) in &self.fields {
            if verbatim {
                map.serialize_entry(name, value)?;
            } else {
                let field = TransformedField {
                    name,
                    value,
                    max_length: self.max_length,
                    in_watermark: false,
                };
                map.serialize_entry(name, &field)?;
            }
        }
        map.end()
    }
}

/// `json.dumps`' default separators (`", "`, `": "`) and float spelling
/// (`repr`). Its string escapes are serde_json's.
struct PythonJsonFormatter;

impl Formatter for PythonJsonFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }

    fn write_f64<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        value: f64,
    ) -> std::io::Result<()> {
        let mut out = String::new();
        write_float(&mut out, value);
        writer.write_all(out.as_bytes())
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
