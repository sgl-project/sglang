//! A request as the Python object the log lines print, and Python's two
//! renderings of it: its text form (`_dataclass_to_string_truncated`) and its
//! JSON format (`_transform_data_for_logging`, `log_json`).

use std::fmt::Write as _;
use std::sync::Arc;

use serde::ser::{SerializeMap, SerializeSeq}; // codespell:ignore ser
use serde_json::ser::Formatter; // codespell:ignore ser
use serde_json::{Map, Value};

use crate::message::multimodal::MmItem;
use crate::message::request::GenerateRequest;

use super::LoggedHeaders;
use super::crash_recorder::DumpBody;
use super::logger::iso_timestamp;

/// `(Receive object, per-request Finish objects, per-request dump sources,
/// headers)`.
pub(super) type Views = (
    PyObj,
    Vec<PyObj>,
    Vec<Option<(Arc<DumpBody>, Option<usize>)>>,
    LoggedHeaders,
);

/// A request's Python object: its own fields over the ones its batch shares,
/// which every item references as Python's `obj[i]` does.
#[derive(Clone, Default)]
pub(super) struct PyObj {
    pub(super) own: Map<String, Value>,
    pub(super) shared: Arc<Map<String, Value>>,
    /// Python `_summarize_multimodal` of its media fields.
    pub(super) multimodal: MediaSummary,
}

impl PyObj {
    pub(super) fn get(&self, name: &str) -> Option<&Value> {
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
pub(super) fn batch_images(images: &Value, i: usize) -> Option<(Value, Value)> {
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
pub(super) const MEDIA_FIELDS: [&str; 3] = ["image_data", "video_data", "audio_data"];

/// What the log reads of a media field at a level that does not show it: its
/// nesting (`None`s included) with each item replaced by its payload length,
/// for the `multimodal` summary and a batch's `modalities`.
pub(super) fn media_layout(value: &Value) -> Value {
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
pub(super) struct MediaSummary {
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
    pub(super) fn of(fields: &Map<String, Value>) -> Self {
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
    pub(super) fn of_openai(dump: &Map<String, Value>) -> Self {
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

    pub(super) fn sum<'a>(summaries: impl Iterator<Item = &'a Self>) -> Self {
        let mut total = Self::default();
        for summary in summaries {
            for (slot, (count, len)) in total.slots.iter_mut().zip(summary.slots) {
                slot.0 += count;
                slot.1 += len;
            }
        }
        total
    }

    pub(super) fn is_empty(&self) -> bool {
        self.slots.iter().all(|(count, _)| *count == 0)
    }

    /// `(modality, count, payload length)` of each modality with any items,
    /// in the summary dict's order.
    pub(super) fn entries(&self) -> impl Iterator<Item = (&'static str, u64, u64)> + '_ {
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

pub(super) fn typed_fields(request: &GenerateRequest) -> Map<String, Value> {
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
            if let Some(max_frames) = hints.max_frames {
                item.insert("max_frames".into(), max_frames.into());
            }
            if let Some(max_tokens_per_frame) = hints.max_tokens_per_frame {
                item.insert("max_tokens_per_frame".into(), max_tokens_per_frame.into());
            }
            Value::Object(item)
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
pub(super) fn write_field(
    out: &mut String,
    name: &str,
    value: &Value,
    max_length: usize,
    in_watermark: bool,
) {
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
pub(super) fn write_dict<'a>(
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
pub(super) fn write_str_repr(out: &mut String, s: &str) {
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
    pub(super) fn new(value: &'a Value, max_length: usize, in_watermark: bool) -> Self {
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
pub(super) struct JsonLine<'a> {
    timestamp: String,
    event: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) rid: Option<&'a Value>,
    obj: JsonObject<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) multimodal: Option<&'a MediaSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    headers: Option<JsonHeaders<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) out: Option<JsonObject<'a>>,
}

impl<'a> JsonLine<'a> {
    pub(super) fn new(
        event: &'a str,
        obj: JsonObject<'a>,
        headers: &'a [(String, String)],
    ) -> Self {
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
    pub(super) fn render(&self) -> String {
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
pub(super) struct JsonObject<'a> {
    pub(super) fields: Vec<(&'a str, &'a Value, bool)>,
    pub(super) max_length: usize,
}

impl<'a> JsonObject<'a> {
    pub(super) fn transformed(
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
pub(super) fn omit_custom_params(sampling_params: &mut Value, keys: &[String]) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Python `repr(float)` for the values a request or `meta_info` carries.
    #[test]
    fn floats_print_as_python_repr() {
        for (value, python) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1.0 / 3.0, "0.3333333333333333"),
            (123.456, "123.456"),
            (0.0001, "0.0001"),
            (1e-05, "1e-05"),
            (1.5e-07, "1.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (2.5e20, "2.5e+20"),
            (1234567890.1234567, "1234567890.1234567"),
            (-1e300, "-1e+300"),
        ] {
            let mut out = String::new();
            write_float(&mut out, value);
            assert_eq!(out, python, "{value:e}");
        }
    }
}
