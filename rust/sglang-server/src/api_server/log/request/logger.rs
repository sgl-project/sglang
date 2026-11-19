//! The request log's lines: what is written at `Receive` and `Finish`, in
//! Python's text or JSON format, and where they go: stdout and hourly-rotated
//! files, written by one background thread.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

use crate::message::config::RequestLogConfig;

use super::py_object::{
    JsonLine, JsonObject, MediaSummary, PyObj, write_dict, write_field, write_str_repr,
};
use super::{OpenAiRequest, lock};

pub(super) struct RequestLogger {
    pub(super) config: RequestLogConfig,
    pub(super) skip_names: HashSet<String>,
    out_skip_names: HashSet<String>,
    batch_item_fields: HashSet<String>,
    batch_shared_fields: HashSet<String>,
    /// The JSON format's `config.fields` defaults, parsed (empty for text).
    json_defaults: Vec<Value>,
    /// The OpenAI request models' fields with their defaults (`Null` for
    /// none).
    chat_completion_fields: Vec<(String, Value)>,
    completion_fields: Vec<(String, Value)>,
    pub(super) sink: LogSink,
}

impl RequestLogger {
    pub(super) fn new(config: RequestLogConfig, sink: LogSink) -> Result<Self, String> {
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
    /// them: its entry of each per-item field.
    pub(super) fn batch_item(&self, fields: &Map<String, Value>, i: usize) -> Map<String, Value> {
        fields
            .iter()
            .filter_map(|(name, value)| {
                if !self.batch_item_fields.contains(name) {
                    return None;
                }
                let value = match value {
                    Value::Array(values) => values.get(i)?.clone(),
                    // One value for every item.
                    value => value.clone(),
                };
                Some((name.clone(), value))
            })
            .collect()
    }

    /// The fields every item of a batch body shares.
    pub(super) fn batch_shared(&self, fields: &Map<String, Value>) -> Map<String, Value> {
        fields
            .iter()
            .filter(|(name, _)| self.batch_shared_fields.contains(name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }

    pub(super) fn received(&self, obj: &PyObj, headers: &[(String, String)]) {
        let line = if self.config.json {
            self.json_line("request.received", obj, headers, None)
        } else {
            self.received_line(obj, headers)
        };
        self.sink.emit(&line);
    }

    pub(super) fn finished(
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
    pub(super) fn openai_received(
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
pub(super) fn iso_timestamp(at: chrono::DateTime<chrono::Local>) -> String {
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
pub(super) enum LogSink {
    Targets(Arc<Targets>),
    #[cfg(test)]
    Memory(Mutex<Vec<String>>),
}

/// `--log-requests-target`: stdout, and the files Python names for the
/// target directories, opened at startup as Python's handlers are.
pub(super) struct Targets {
    stdout: bool,
    paths: Vec<String>,
    /// The files at `paths`; only the writer thread writes them.
    files: Mutex<Vec<RotatingFile>>,
}

impl Targets {
    pub(super) fn open(config: &RequestLogConfig) -> Result<Self, String> {
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
    pub(super) fn open(path: PathBuf, now: SystemTime) -> Result<Self, String> {
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
    pub(super) fn new(targets: Arc<Targets>) -> Self {
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

#[cfg(test)]
mod tests {
    use super::super::tests::{FIXTURE, chunk, config, config_in, level1, receive_body};
    use super::super::{ReceivedAs, RequestLog, python_out};
    use super::*;
    use crate::api_server::core::CoreOutput;
    use crate::message::ids::Rid;
    use crate::message::request::{GenerationPolicy, into_requests};
    use crate::message::response::ResponseItem;
    use sglang_api_types::api::v1 as api;

    /// `line` with its JSON `timestamp` (the time it was logged) replaced by the
    /// fixture's.
    fn at_fixture_time(line: &str, fixture: &Value) -> String {
        const KEY: &str = "{\"timestamp\": \"";
        let Some(rest) = line.strip_prefix(KEY) else {
            return line.to_owned();
        };
        let end = rest.find('"').expect("a closed timestamp");
        format!(
            "{KEY}{}{}",
            fixture["timestamp"].as_str().unwrap(),
            &rest[end..]
        )
    }

    fn header_map(headers: &Value) -> http::HeaderMap {
        let mut map = http::HeaderMap::new();
        for (name, value) in headers.as_object().unwrap() {
            map.insert(
                http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                http::HeaderValue::from_str(value.as_str().unwrap()).unwrap(),
            );
        }
        map
    }

    /// Every fixture case renders exactly Python's `Receive` and `Finish` lines at
    /// every level it lists, in both formats; and every OpenAI case Python's
    /// `Receive OpenAI` line.
    #[test]
    fn lines_match_python_request_logger() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let mut checked = 0;
        for case in fixture["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            for (format, expected) in case["expected"].as_object().unwrap() {
                for (level, expected) in expected.as_object().unwrap() {
                    let log = RequestLog::for_test(Some(config_in(&fixture, format, level)), false);
                    let records = receive_body(&log, &case["body"], &header_map(&case["headers"]));
                    let item = case["batch_item"].as_u64().unwrap_or(0) as usize;
                    let record = &records[item];
                    let out = match case.get("core_output") {
                        // The Rust server's own frame of that output.
                        Some(output) => python_out(
                            core_output(output),
                            &record.rid,
                            output["e2e_latency"].as_f64().unwrap(),
                        ),
                        None => case["out"].as_object().unwrap().clone(),
                    };
                    assert_eq!(Value::Object(out.clone()), case["out"], "{name}: out");
                    log.logger
                        .as_ref()
                        .unwrap()
                        .finished(&record.obj, &record.headers, &out, 1.0);
                    let lines: Vec<String> = log
                        .logged_lines()
                        .iter()
                        .map(|line| at_fixture_time(line, &fixture))
                        .collect();
                    // A batch's `Receive` is best effort and not compared.
                    if !expected["receive"].is_null() {
                        assert_eq!(
                            lines[0], expected["receive"],
                            "{name} at {format} level {level}: Receive"
                        );
                    }
                    assert_eq!(
                        lines[1], expected["finish"],
                        "{name} at {format} level {level}: Finish"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked >= 42, "the fixture lost cases: {checked}");

        let mut checked = 0;
        for case in fixture["openai_cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let request = match case["request"].as_str().unwrap() {
                "chat" => OpenAiRequest::ChatCompletion,
                _ => OpenAiRequest::Completion,
            };
            let body = serde_json::to_vec(&case["body"]).unwrap();
            for (format, expected) in case["expected"].as_object().unwrap() {
                for (level, expected) in expected.as_object().unwrap() {
                    let log = RequestLog::for_test(Some(config_in(&fixture, format, level)), false);
                    log.openai_received(request, &body, &header_map(&case["headers"]));
                    let lines = log.logged_lines();
                    assert_eq!(
                        at_fixture_time(&lines[0], &fixture),
                        *expected,
                        "{name} at {format} level {level}"
                    );
                    checked += 1;
                }
            }
            // Not logged below level 2.
            let log = RequestLog::for_test(Some(config(&fixture, "1")), false);
            log.openai_received(request, &body, &header_map(&case["headers"]));
            assert!(log.logged_lines().is_empty(), "{name} at level 1");
        }
        assert!(checked >= 8, "the fixture lost OpenAI cases: {checked}");
    }

    /// A chat request's media content parts give its `Receive OpenAI` line
    /// Python's `multimodal` summary, in both formats.
    #[test]
    fn openai_multimodal_summary_matches_python() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let cases = fixture["openai_multimodal_cases"].as_array().unwrap();
        assert!(cases.len() >= 2, "the fixture lost OpenAI media cases");
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let body = serde_json::to_vec(&case["body"]).unwrap();
            for (format, expected) in case["expected"].as_object().unwrap() {
                let log = RequestLog::for_test(Some(config_in(&fixture, format, "2")), false);
                log.openai_received(
                    OpenAiRequest::ChatCompletion,
                    &body,
                    &http::HeaderMap::new(),
                );
                let lines = log.logged_lines();
                if format == "json" {
                    let line: Value = serde_json::from_str(&lines[0]).unwrap();
                    let summary = line.get("multimodal").cloned().unwrap_or(Value::Null);
                    assert_eq!(summary, *expected, "{name} at json");
                } else {
                    let segment = lines[0]
                        .find(", multimodal=")
                        .map_or("", |start| &lines[0][start..]);
                    assert_eq!(segment, expected.as_str().unwrap(), "{name} at text");
                }
            }
        }
    }

    fn core_output(output: &Value) -> CoreOutput {
        CoreOutput {
            text: output["text"].as_str().unwrap().into(),
            token_ids: serde_json::from_value(output["token_ids"].clone()).unwrap(),
            prompt_tokens: output["prompt_tokens"].as_u64().unwrap() as u32,
            completion_tokens: output["completion_tokens"].as_u64().unwrap(),
            cached_tokens: output["cached_tokens"].as_u64().unwrap() as u32,
            finish_reason: serde_json::from_value(output["finish_reason"].clone()).unwrap(),
            weight_version: Some(Arc::from("default")),
            ..Default::default()
        }
    }

    /// With `--incremental-streaming-output`, a stream's `Finish` shows its
    /// terminal frame: the last delta, with the cumulative completion count.
    #[test]
    fn incremental_stream_logs_last_delta() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let log = Arc::new(RequestLog {
            logger: Some(
                RequestLogger::new(config(&fixture, "3"), LogSink::Memory(Mutex::default()))
                    .unwrap(),
            ),
            recorder: None,
            server_added_custom_params: Vec::new(),
            default_priority: None,
            weight_version: Some(Arc::from("default")),
            dp_rank: None,
            incremental_streaming_output: true,
        });
        let body = serde_json::json!({"rid": "r-inc", "text": "hi", "stream": true});
        let mut record = receive_body(&log, &body, &http::HeaderMap::new()).remove(0);
        let rid = Rid::from_client("r-inc");
        record.observe(&ResponseItem::Frame(chunk(&rid, "a", &[1], None)));
        record.observe(&ResponseItem::Done(chunk(&rid, "b", &[2], None)));

        let finish = log.logged_lines().remove(1);
        assert!(
            finish.contains(", out={'text': 'b', 'output_ids': [2], 'meta_info': {'id': 'r-inc', ")
                && finish.contains("'completion_tokens': 2, "),
            "{finish}"
        );
    }

    /// `SGLANG_LOG_REQUEST_EXCEEDED_MS` logs only the slow finishes.
    #[test]
    fn exceeded_ms_skips_fast_finishes() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let log = RequestLog::for_test(
            Some(RequestLogConfig {
                exceeded_ms: 500,
                ..level1(&fixture)
            }),
            false,
        );
        let logger = log.logger.as_ref().unwrap();
        let out = Map::new();
        logger.finished(&PyObj::default(), &[], &out, 0.499);
        logger.finished(&PyObj::default(), &[], &out, 0.5);
        let lines = log.logged_lines();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("Finish: "));
    }

    /// A key the server itself adds to `custom_params` never reaches a line; the
    /// client's own custom params still do.
    #[test]
    fn server_added_custom_params_are_not_logged() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let log = RequestLog::for_test_omitting(Some(level1(&fixture)), false, &["server_only"]);
        let body = serde_json::json!({
            "rid": ["a", "b"],
            "text": ["x", "y"],
            "sampling_params": [
                {"custom_params": {"server_only": "srv-added-1", "client": 1}},
                {"custom_params": {"server_only": "srv-added-2"}},
            ],
        });
        let mut records = receive_body(&log, &body, &http::HeaderMap::new());
        let receive = log.logged_lines().remove(0);
        assert!(!receive.contains("server_only"), "{receive}");
        assert!(!receive.contains("srv-added"), "{receive}");
        assert!(receive.contains("'client': 1"), "{receive}");
        let rid = Rid::from_client("a");
        records[0].observe(&ResponseItem::Done(chunk(&rid, "z", &[3], None)));
        let finish = log.logged_lines().remove(0);
        assert!(!finish.contains("srv-added"), "{finish}");
        assert!(finish.contains("'client': 1"), "{finish}");
    }

    /// A request an HTTP layer held before `/generate` (work an `HttpLayer` does
    /// before the handler runs) counts that wait in its `Finish` latency, so
    /// `SGLANG_LOG_REQUEST_EXCEEDED_MS` doesn't drop it as fast.
    #[test]
    fn finish_latency_counts_the_wait_before_generate() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let log = RequestLog::for_test(
            Some(RequestLogConfig {
                exceeded_ms: 1000,
                ..level1(&fixture)
            }),
            false,
        );
        let body = serde_json::json!({"rid": "a", "text": "x"});
        let received = ReceivedAs::Body {
            snapshot: log.snapshot_body(&body, None, &http::HeaderMap::new()),
            started_at: Some(std::time::Instant::now() - std::time::Duration::from_secs(2)),
        };
        let decoded: api::GenerateRequest = serde_json::from_value(body).unwrap();
        let (requests, batch) = into_requests(decoded, &GenerationPolicy::default()).unwrap();
        let mut records = log.receive(&requests, received, batch, true);
        log.logged_lines();
        records[0].observe(&ResponseItem::Done(chunk(
            &Rid::from_client("a"),
            "z",
            &[3],
            None,
        )));
        let lines = log.logged_lines();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].starts_with("Finish: "), "{lines:?}");
    }

    /// A target directory's file, written through the process writer, rolls over
    /// as `TimedRotatingFileHandler(when="H")` rolls it: an hour after it was
    /// opened (not on the hour), renamed after the hour that started, then an hour
    /// after the rollover.
    #[test]
    fn log_file_rotates_hourly() {
        let dir = std::env::temp_dir().join(format!("request-log-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("host_3.log");
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let targets = Arc::new(
            Targets::open(&RequestLogConfig {
                stdout: false,
                files: vec![path.to_string_lossy().into_owned()],
                ..level1(&fixture)
            })
            .unwrap(),
        );
        LogSink::Targets(targets.clone()).emit("through the writer");
        flush_request_log();
        let written = std::fs::read_to_string(&path).unwrap();
        assert!(written.ends_with("] through the writer\n"), "{written}");

        let mut files = lock(&targets.files);
        let file = &mut files[0];
        let opened = file.rollover_at - ROTATE_INTERVAL;
        let later = opened + ROTATE_INTERVAL * 2 + Duration::from_secs(1);
        file.write_line(b"next hour\n", later).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "next hour\n");
        let started: chrono::DateTime<chrono::Local> = opened.into();
        let rotated = format!("{}{}", path.display(), started.format(".%Y-%m-%d_%H"));
        assert_eq!(std::fs::read_to_string(rotated).unwrap(), written);
        assert_eq!(file.rollover_at, later + ROTATE_INTERVAL);
        drop(files);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A rollover names the file across a DST change as Python does, never
    /// replaces an archive (Python takes a taken name as already rolled over and
    /// keeps writing), outlives a log file deleted under it, and reads times in
    /// whole seconds as Python does.
    #[test]
    fn rollover_keeps_archives_and_survives_a_deleted_file() {
        let dir = std::env::temp_dir().join(format!("request-log-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("host_0.log");
        let read = |suffix: &str| std::fs::read_to_string(format!("{}{suffix}", path.display()));
        // America/Los_Angeles, whose DST ends at 09:00 UTC on 2026-11-01.
        let dst_end = UNIX_EPOCH + Duration::from_secs(1_793_523_600);
        let la = |at: SystemTime| {
            let hours = if at < dst_end { 7 } else { 8 };
            chrono::FixedOffset::west_opt(hours * 3600).unwrap()
        };
        let utc = |minutes: u64| dst_end + Duration::from_secs(minutes * 60);
        let mut file = RotatingFile::open(path.clone(), SystemTime::now()).unwrap();
        let roll = |file: &mut RotatingFile, minutes, line: &[u8]| {
            file.rollover_at = utc(minutes);
            file.roll_over(utc(minutes), la).unwrap();
            file.write_line(line, utc(minutes)).unwrap();
        };
        file.write_line(b"01:30 PDT\n", SystemTime::now()).unwrap();
        // 01:30 PDT to 01:30 PST: named after 00:30 PDT, as Python shifts it.
        roll(&mut file, 30, b"01:30 PST\n");
        assert_eq!(read(".2026-11-01_00").unwrap(), "01:30 PDT\n");
        // So the repeated hour gets its own name.
        roll(&mut file, 90, b"02:30 PST\n");
        assert_eq!(read(".2026-11-01_01").unwrap(), "01:30 PST\n");
        // A taken name is left alone; the file keeps its lines.
        std::fs::write(format!("{}.2026-11-01_02", path.display()), "kept\n").unwrap();
        roll(&mut file, 150, b"03:30 PST\n");
        assert_eq!(read(".2026-11-01_02").unwrap(), "kept\n");
        assert_eq!(read("").unwrap(), "02:30 PST\n03:30 PST\n");
        // A deleted file is not renamed and is created again.
        std::fs::remove_file(&path).unwrap();
        roll(&mut file, 210, b"04:30 PST\n");
        assert_eq!(read("").unwrap(), "04:30 PST\n");
        assert!(read(".2026-11-01_03").is_err());

        // Australia/Lord_Howe, whose DST ends by half an hour at 15:00 UTC on
        // 2026-04-04: Python still shifts by a whole hour.
        let path = dir.join("host_1.log");
        let mut file = RotatingFile::open(path.clone(), SystemTime::now()).unwrap();
        let dst_end = UNIX_EPOCH + Duration::from_secs(1_775_314_800);
        let lord_howe = |at: SystemTime| {
            let minutes = if at < dst_end { 11 * 60 } else { 10 * 60 + 30 };
            chrono::FixedOffset::east_opt(minutes * 60).unwrap()
        };
        // 01:45 to 02:15 (+10:30), named after 00:45 (+11).
        file.rollover_at = dst_end + Duration::from_secs(45 * 60);
        file.roll_over(file.rollover_at, lord_howe).unwrap();
        assert!(std::fs::exists(format!("{}.2026-04-05_00", path.display())).unwrap());

        // Python reads the file time and the clock in whole seconds.
        let path = dir.join("host_2.log");
        let created = std::fs::File::create(&path).unwrap();
        created
            .set_modified(dst_end + Duration::from_millis(900))
            .unwrap();
        let mut file = RotatingFile::open(path, SystemTime::now()).unwrap();
        assert_eq!(file.rollover_at, dst_end + ROTATE_INTERVAL);
        let later = dst_end + ROTATE_INTERVAL * 2;
        file.write_line(b"x\n", later + Duration::from_millis(100))
            .unwrap();
        assert_eq!(file.rollover_at, later + ROTATE_INTERVAL);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
