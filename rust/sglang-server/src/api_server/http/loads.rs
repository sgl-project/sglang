//! `GET /v1/loads`: the port of Python `entrypoints/v1_loads.py`. Each
//! scheduler publishes its per-DP-rank load snapshot into a shared-memory file
//! (`managers/load_snapshot.py`, `ShmLoadSnapshotWriter`); this reads that file
//! and renders Python's `LoadSnapshot.to_dict`, as JSON or Prometheus text.
//!
//! Unlike the Python server, which fronts every DP rank, a Rust listener serves
//! only its own scheduler's rank, so it reports itself as a single-rank engine:
//! its own slot, as rank 0.

use std::collections::BTreeSet;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use rustix::fs::{FlockOperation, flock};
use serde::Deserialize;
use serde_json::{Map, Value};

use super::app::AppState;
use super::native_api::native_error;

/// The file header, Python's `HEADER_STRUCT` (`<4sHHI`): magic, version,
/// dp_size, slot_size. Each rank's slot follows at `HEADER_LEN + rank *
/// slot_size`: a little-endian `u32` length, then the msgpack snapshot.
const MAGIC: &[u8; 4] = b"SLNS";
const VERSION: u16 = 2;
const HEADER_LEN: u64 = 12;
const SLOT_LEN_PREFIX: u64 = 4;

/// The zero value a field takes when the snapshot omits it.
#[derive(Clone, Copy)]
enum Zero {
    Int,
    Float,
    Null,
}

impl Zero {
    fn value(self) -> Value {
        match self {
            Zero::Int => Value::from(0),
            Zero::Float => Value::from(0.0),
            Zero::Null => Value::Null,
        }
    }
}

/// `LoadSnapshot`'s flat fields in declaration order. The snapshot is encoded
/// with `omit_defaults`, so an absent key takes its zero.
const CORE: &[(&str, Zero)] = &[
    ("timestamp", Zero::Float),
    ("dp_rank", Zero::Int),
    ("num_running_reqs", Zero::Int),
    ("num_waiting_reqs", Zero::Int),
    ("num_waiting_uncached_tokens", Zero::Int),
    ("num_used_tokens", Zero::Int),
    ("num_total_tokens", Zero::Int),
    ("num_active_tokens", Zero::Int),
    ("num_prealloc_ready_tokens", Zero::Int),
    ("max_total_num_tokens", Zero::Int),
    ("max_running_requests", Zero::Int),
    ("token_usage", Zero::Float),
    ("gen_throughput", Zero::Float),
    ("cache_hit_rate", Zero::Float),
    ("utilization", Zero::Float),
    ("total_prefill_uncached_tokens", Zero::Int),
    ("total_prefill_busy_us", Zero::Int),
    ("decode_moments", Zero::Null),
];

/// One optional sub-struct of `LoadSnapshot`, encoded as a positional array.
/// `None` marks a field without a default.
struct Section {
    field: &'static str,
    include: &'static str,
    prometheus_prefix: &'static str,
    fields: &'static [(&'static str, Option<Zero>)],
}

const SECTIONS: &[Section] = &[
    Section {
        field: "memory",
        include: "memory",
        prometheus_prefix: "memory",
        fields: &[
            ("weight_gb", None),
            ("kv_cache_gb", None),
            ("graph_gb", None),
            ("token_capacity", None),
        ],
    },
    Section {
        field: "speculative",
        include: "spec",
        prometheus_prefix: "spec",
        fields: &[("accept_length", None), ("accept_rate", None)],
    },
    Section {
        field: "lora",
        include: "lora",
        prometheus_prefix: "lora",
        fields: &[
            ("slots_used", None),
            ("slots_total", None),
            ("utilization", None),
        ],
    },
    Section {
        field: "disaggregation",
        include: "disagg",
        prometheus_prefix: "disagg",
        fields: &[
            ("mode", None),
            ("prefill_bootstrap_queue_reqs", Some(Zero::Int)),
            ("prefill_inflight_queue_reqs", Some(Zero::Int)),
            ("decode_prealloc_queue_reqs", Some(Zero::Int)),
            ("decode_transfer_queue_reqs", Some(Zero::Int)),
            ("decode_retracted_queue_reqs", Some(Zero::Int)),
            ("kv_transfer_speed_gb_s", Some(Zero::Float)),
            ("kv_transfer_latency_ms", Some(Zero::Float)),
        ],
    },
    Section {
        field: "queues",
        include: "queues",
        prometheus_prefix: "queues",
        fields: &[
            ("waiting", None),
            ("grammar", None),
            ("paused", None),
            ("retracted", None),
            ("prealloc_ready", None),
        ],
    },
];

const VALID_INCLUDES: &[&str] = &["all", "core", "disagg", "lora", "memory", "queues", "spec"];

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/v1/loads", get(loads))
}

#[derive(Deserialize)]
struct LoadsQuery {
    dp_rank: Option<i64>,
    include: Option<String>,
    format: Option<String>,
}

async fn loads(State(state): State<Arc<AppState>>, Query(query): Query<LoadsQuery>) -> Response {
    let args = &state.server_args;
    let Some(path) = args.load_snapshot_path.as_deref() else {
        return native_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "load snapshots are not published to shared memory in this deployment",
            false,
        );
    };
    let include = match parse_include(query.include.as_deref()) {
        Ok(include) => include,
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    let ranks = match query.dp_rank {
        None | Some(0) => vec![args.load_snapshot_dp_rank],
        Some(_) => Vec::new(),
    };
    let started = Instant::now();
    let loads: Vec<Map<String, Value>> =
        read_snapshots(Path::new(path), args.load_snapshot_dp_size, &ranks)
            .iter()
            .filter_map(|snapshot| to_dict(snapshot, include.as_ref()))
            .map(|mut load| {
                load.insert("dp_rank".into(), Value::from(0));
                load
            })
            .collect();
    if let Some(metrics) = state.core.metrics() {
        metrics.observe_get_loads(started.elapsed());
    }

    if query.format.as_deref() == Some("prometheus") {
        return (
            [(
                header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            prometheus(&loads),
        )
            .into_response();
    }
    let mut body = Map::new();
    body.insert("timestamp".into(), Value::from(utc_now_isoformat()));
    body.insert("version".into(), Value::from(args.version.as_str()));
    body.insert("accelerator".into(), args.accelerator.clone().into());
    body.insert(
        "num_accelerators".into(),
        Value::from(args.num_accelerators),
    );
    body.insert(
        "loads".into(),
        Value::Array(loads.into_iter().map(Value::Object).collect()),
    );
    Json(Value::Object(body)).into_response()
}

/// Python `to_dict`'s include rules: absent or `all` means every section.
/// Returns the selected section names, `None` for all of them.
fn parse_include(include: Option<&str>) -> Result<Option<BTreeSet<String>>, String> {
    let Some(include) = include.filter(|include| !include.is_empty()) else {
        return Ok(None);
    };
    let names: BTreeSet<String> = include
        .split(',')
        .map(|name| name.trim().to_owned())
        .collect();
    if names.contains("all") {
        return Ok(None);
    }
    let invalid: Vec<&String> = names
        .iter()
        .filter(|name| !VALID_INCLUDES.contains(&name.as_str()))
        .collect();
    if !invalid.is_empty() {
        return Err(format!(
            "Invalid include sections: {invalid:?}. Valid options: {VALID_INCLUDES:?}"
        ));
    }
    Ok(Some(names))
}

/// The decoded snapshots of `ranks`, read under the shared lock Python's
/// reader takes. A missing or mismatched file, or an empty or malformed slot,
/// yields nothing for that rank, as Python's reader does.
fn read_snapshots(path: &Path, dp_size: usize, ranks: &[usize]) -> Vec<rmpv::Value> {
    let Ok(file) = File::open(path) else {
        return Vec::new();
    };
    if flock(&file, FlockOperation::LockShared).is_err() {
        return Vec::new();
    }
    let mut header = [0u8; HEADER_LEN as usize];
    if file.read_exact_at(&mut header, 0).is_err() {
        return Vec::new();
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    let file_dp_size = usize::from(u16::from_le_bytes([header[6], header[7]]));
    let slot_size = u64::from(u32::from_le_bytes([
        header[8], header[9], header[10], header[11],
    ]));
    let file_len = file.metadata().map(|meta| meta.len()).unwrap_or(0);
    if &header[..4] != MAGIC
        || version != VERSION
        || file_dp_size != dp_size
        || slot_size < SLOT_LEN_PREFIX
        || file_len < HEADER_LEN + dp_size as u64 * slot_size
    {
        tracing::warn!(path = %path.display(), "load snapshot header mismatch");
        return Vec::new();
    }
    ranks
        .iter()
        .filter_map(|&rank| {
            let offset = HEADER_LEN + rank as u64 * slot_size;
            let mut len = [0u8; SLOT_LEN_PREFIX as usize];
            file.read_exact_at(&mut len, offset).ok()?;
            let len = u64::from(u32::from_le_bytes(len));
            if len == 0 || len > slot_size - SLOT_LEN_PREFIX {
                return None;
            }
            let mut payload = vec![0u8; len as usize];
            file.read_exact_at(&mut payload, offset + SLOT_LEN_PREFIX)
                .ok()?;
            rmpv::decode::read_value(&mut payload.as_slice()).ok()
        })
        .collect()
}

/// Python `LoadSnapshot.to_dict`: every flat field, then each present section
/// the include set selects (`None` selects all). `None` for a payload Python's
/// decoder would reject.
fn to_dict(
    snapshot: &rmpv::Value,
    include: Option<&BTreeSet<String>>,
) -> Option<Map<String, Value>> {
    let fields = snapshot.as_map()?;
    let get = |name: &str| {
        fields
            .iter()
            .find(|(key, _)| key.as_str() == Some(name))
            .map(|(_, value)| value)
    };
    let mut load = Map::new();
    for &(name, zero) in CORE {
        let value = match get(name) {
            Some(value) => json(value)?,
            None => zero.value(),
        };
        load.insert(name.to_owned(), value);
    }
    for section in SECTIONS {
        let selected = include.is_none_or(|include| include.contains(section.include));
        let Some(values) = get(section.field).filter(|value| !value.is_nil()) else {
            continue;
        };
        let values = values.as_array()?;
        let mut object = Map::new();
        for (i, &(name, zero)) in section.fields.iter().enumerate() {
            let value = match (values.get(i), zero) {
                (Some(value), _) => json(value)?,
                (None, Some(zero)) => zero.value(),
                (None, None) => return None,
            };
            object.insert(name.to_owned(), value);
        }
        if selected {
            load.insert(section.field.to_owned(), Value::Object(object));
        }
    }
    Some(load)
}

fn json(value: &rmpv::Value) -> Option<Value> {
    Some(match value {
        rmpv::Value::Nil => Value::Null,
        rmpv::Value::Boolean(b) => Value::from(*b),
        rmpv::Value::Integer(n) => n
            .as_i64()
            .map(Value::from)
            .or_else(|| n.as_u64().map(Value::from))?,
        rmpv::Value::F32(x) => Value::from(f64::from(*x)),
        rmpv::Value::F64(x) => Value::from(*x),
        rmpv::Value::String(s) => Value::from(s.as_str()?),
        rmpv::Value::Array(items) => Value::Array(items.iter().map(json).collect::<Option<_>>()?),
        _ => return None,
    })
}

/// Python `_format_loads_prometheus`: one gauge per numeric field, samples
/// labelled by `dp_rank`, metrics in first-seen order.
fn prometheus(loads: &[Map<String, Value>]) -> String {
    let mut metrics: Vec<(String, Vec<String>)> = Vec::new();
    let mut add = |name: String, rank: &str, value: &Value| {
        let sample = format!("{name}{{dp_rank=\"{rank}\"}} {}", prometheus_value(value));
        match metrics.iter_mut().find(|(seen, _)| *seen == name) {
            Some((_, samples)) => samples.push(sample),
            None => metrics.push((name, vec![sample])),
        }
    };
    for load in loads {
        let rank = load
            .get("dp_rank")
            .map(Value::to_string)
            .unwrap_or_default();
        for (key, value) in load.iter().filter(|(key, _)| *key != "dp_rank") {
            match value {
                Value::Object(section) => {
                    let prefix = SECTIONS
                        .iter()
                        .find(|s| s.field == key)
                        .map_or(key.as_str(), |s| s.prometheus_prefix);
                    for (sub_key, sub_value) in section.iter().filter(|(_, v)| v.is_number()) {
                        add(format!("sglang_{prefix}_{sub_key}"), &rank, sub_value);
                    }
                }
                Value::Number(_) => add(format!("sglang_{key}"), &rank, value),
                _ => {}
            }
        }
    }
    let mut lines = Vec::new();
    for (name, samples) in metrics {
        lines.push(format!("# TYPE {name} gauge"));
        lines.extend(samples);
    }
    lines.join("\n") + "\n"
}

/// Python's `str()` of a number: integers bare, floats with a decimal point.
fn prometheus_value(value: &Value) -> String {
    match value.as_f64() {
        Some(x) if !value.is_i64() && !value.is_u64() => format!("{x:?}"),
        _ => value.to_string(),
    }
}

/// `datetime.now(timezone.utc).isoformat()`.
fn utc_now_isoformat() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::config::ServerArgs;
    use serde_json::json;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../../testdata/load_snapshot.json")).unwrap()
    }

    /// The fixture's payloads in Python's file format, one slot per rank.
    struct SnapshotFile(std::path::PathBuf);

    impl SnapshotFile {
        fn new(dp_size: usize) -> Self {
            let slot_size = 16 * 1024u32;
            let mut bytes = Vec::from(&MAGIC[..]);
            bytes.extend(VERSION.to_le_bytes());
            bytes.extend((dp_size as u16).to_le_bytes());
            bytes.extend(slot_size.to_le_bytes());
            for payload in fixture()["payloads"].as_array().unwrap() {
                let hex = payload.as_str().unwrap();
                let payload: Vec<u8> = (0..hex.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                    .collect();
                let mut slot = (payload.len() as u32).to_le_bytes().to_vec();
                slot.extend(payload);
                slot.resize(slot_size as usize, 0);
                bytes.extend(slot);
            }
            let path = std::env::temp_dir().join(format!("sglang-loads-{}", uuid::Uuid::new_v4()));
            std::fs::write(&path, bytes).unwrap();
            Self(path)
        }
    }

    impl Drop for SnapshotFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// The JSON and Prometheus renderings match what Python produces from the
    /// same bytes, key order and float formatting included.
    #[test]
    fn renders_snapshots_as_python_does() {
        let file = SnapshotFile::new(2);
        let snapshots = read_snapshots(&file.0, 2, &[0, 1]);
        assert_eq!(snapshots.len(), 2);
        let expected = fixture();
        for name in ["all", "core", "core,queues"] {
            let include = parse_include(Some(name)).unwrap();
            let loads: Vec<_> = snapshots
                .iter()
                .map(|s| to_dict(s, include.as_ref()).unwrap())
                .collect();
            assert_eq!(
                serde_json::to_string(&loads).unwrap(),
                serde_json::to_string(&expected["to_dict"][name]).unwrap(),
                "to_dict({name})"
            );
            assert_eq!(
                prometheus(&loads),
                expected["prometheus"][name],
                "prometheus({name})"
            );
        }
    }

    async fn get(server_args: ServerArgs, query: &str) -> (StatusCode, String) {
        let (intake_tx, _intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let state = Arc::new(AppState {
            core: crate::api_server::core::CoreHandle::new(
                intake_tx,
                abort_tx,
                crate::api_server::core::CoreConfig {
                    response_capacity: 8,
                    response_activity: Default::default(),
                    startup_ready: true,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    metadata: crate::api_server::core::CoreMetadata::default(),
                    metrics: None,
                },
            ),
            generation_policy: crate::message::request::GenerationPolicy::from(&server_args),
            server_args: Arc::new(server_args),
            chat_formatter: None,
            generate_hook: None,
        });
        let query = Query::try_from_uri(&format!("/v1/loads?{query}").parse().unwrap()).unwrap();
        let response = loads(State(state), query).await;
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    /// The endpoint wraps the listener's own slot, as rank 0, in Python's
    /// envelope, and degrades as Python's reader does; it rejects unknown
    /// sections and answers 503 when snapshots are not in shared memory.
    #[tokio::test]
    async fn serves_own_rank_in_python_envelope() {
        let file = SnapshotFile::new(2);
        let args = |path: &Path, dp_size, dp_rank| ServerArgs {
            load_snapshot_path: Some(path.display().to_string()),
            load_snapshot_dp_size: dp_size,
            load_snapshot_dp_rank: dp_rank,
            version: "1.2.3".into(),
            accelerator: Some("GPU".into()),
            num_accelerators: 4,
            ..Default::default()
        };
        let json = |body: &str| serde_json::from_str::<Value>(body).unwrap();
        let expected = fixture();

        let (status, body) = get(args(&file.0, 2, 0), "include=core").await;
        assert_eq!(status, StatusCode::OK);
        let body = json(&body);
        let keys: Vec<_> = body.as_object().unwrap().keys().collect();
        assert_eq!(
            keys,
            [
                "timestamp",
                "version",
                "accelerator",
                "num_accelerators",
                "loads"
            ]
        );
        let timestamp = body["timestamp"].as_str().unwrap();
        assert!(
            timestamp.len() == 32 && timestamp.ends_with("+00:00"),
            "{timestamp}"
        );
        assert_eq!(
            (&body["version"], &body["accelerator"]),
            (&json!("1.2.3"), &json!("GPU"))
        );
        assert_eq!(body["num_accelerators"], 4);
        assert_eq!(body["loads"], json!([expected["to_dict"]["core"][0]]));

        // The rank-1 listener reports the rank-1 slot, renumbered to 0.
        let (_, body) = get(args(&file.0, 2, 1), "dp_rank=0").await;
        let mut rank1 = expected["to_dict"]["all"][1].clone();
        rank1["dp_rank"] = json!(0);
        assert_eq!(json(&body)["loads"], json!([rank1]));

        let ranks = |body: &str| json(body)["loads"].as_array().unwrap().len();
        assert_eq!(ranks(&get(args(&file.0, 2, 1), "dp_rank=1").await.1), 0);
        assert_eq!(
            ranks(&get(args(&file.0, 3, 0), "").await.1),
            0,
            "dp_size mismatch"
        );
        let missing = Path::new("/nonexistent/sglang_loads.shm");
        assert_eq!(ranks(&get(args(missing, 2, 0), "").await.1), 0);

        let (status, body) = get(args(&file.0, 2, 0), "format=prometheus").await;
        assert_eq!(status, StatusCode::OK);
        let rank0 = expected["to_dict"]["all"][0].as_object().unwrap().clone();
        assert_eq!(body, prometheus(&[rank0]));

        let (status, body) = get(args(&file.0, 2, 0), "include=core,bogus").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("bogus"), "{body}");

        let (status, _) = get(ServerArgs::default(), "").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }
}
