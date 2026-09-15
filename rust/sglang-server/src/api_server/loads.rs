//! Scheduler-owned load snapshots, served without polling the GPU thread.

use std::collections::HashSet;
use std::sync::Arc;

use arc_swap::ArcSwapOption;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, routing::get};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::message::config::ServerArgs;

const SECTIONS: [(&str, &str); 5] = [
    ("memory", "memory"),
    ("speculative", "spec"),
    ("lora", "lora"),
    ("disaggregation", "disagg"),
    ("queues", "queues"),
];

/// The named wire contract is Python's public `LoadSnapshot.to_dict()`, never
/// the scheduler's internal state or server arguments. Preserve new load
/// counters without requiring positional Python/Rust schema updates.
#[derive(Debug, Deserialize, Serialize)]
struct Snapshot {
    dp_rank: u32,
    timestamp: f64,
    num_running_reqs: u64,
    num_waiting_reqs: u64,
    num_used_tokens: u64,
    num_total_tokens: u64,
    #[serde(flatten)]
    counters: Map<String, Value>,
}

#[derive(Default)]
pub(crate) struct LoadSnapshotStore(ArcSwapOption<Snapshot>);

impl LoadSnapshotStore {
    pub fn publish(&self, msgpack: &[u8]) -> Result<(), String> {
        let snapshot: Snapshot =
            rmp_serde::from_slice(msgpack).map_err(|error| error.to_string())?;
        if !snapshot.timestamp.is_finite() || snapshot.timestamp < 0.0 {
            return Err("load snapshot timestamp must be finite and nonnegative".into());
        }
        if snapshot
            .num_running_reqs
            .checked_add(snapshot.num_waiting_reqs)
            .is_none()
        {
            return Err("load snapshot request count overflow".into());
        }
        self.0.store(Some(Arc::new(snapshot)));
        Ok(())
    }
}

struct LoadState {
    snapshots: Arc<LoadSnapshotStore>,
    version: String,
    accelerator: Option<String>,
    num_accelerators: usize,
}

pub(crate) fn router(snapshots: Arc<LoadSnapshotStore>, args: &ServerArgs) -> Router {
    Router::new()
        .route("/v1/loads", get(loads))
        .route("/get_load", get(legacy_load))
        .with_state(Arc::new(LoadState {
            snapshots,
            version: args.version.clone(),
            accelerator: args.accelerator.clone(),
            num_accelerators: args.num_accelerators,
        }))
}

#[derive(Default, Deserialize)]
struct LoadQuery {
    dp_rank: Option<i64>,
    include: Option<String>,
    format: Option<String>,
}

fn selected(snapshot: &Snapshot, include: Option<&str>) -> Result<Value, String> {
    let include: HashSet<&str> = include
        .filter(|value| !value.is_empty())
        .map(|value| value.split(',').map(str::trim).collect())
        .unwrap_or_else(|| HashSet::from(["all"]));
    if !include.contains("all")
        && include
            .iter()
            .any(|section| *section != "core" && !SECTIONS.iter().any(|(_, name)| name == section))
    {
        return Err("Invalid include sections. Valid options: all, core, disagg, lora, memory, queues, spec".into());
    }
    let mut value = serde_json::to_value(snapshot).map_err(|error| error.to_string())?;
    if let Value::Object(fields) = &mut value {
        for (field, section) in SECTIONS {
            if !include.contains("all") && !include.contains(section) {
                fields.remove(field);
            }
        }
    }
    Ok(value)
}

async fn loads(State(state): State<Arc<LoadState>>, Query(query): Query<LoadQuery>) -> Response {
    let snapshot = state.snapshots.0.load_full();
    let mut loads = Vec::new();
    if let Some(snapshot) = snapshot
        && query
            .dp_rank
            .is_none_or(|rank| rank == i64::from(snapshot.dp_rank))
    {
        match selected(&snapshot, query.include.as_deref()) {
            Ok(value) => loads.push(value),
            Err(detail) => {
                return (StatusCode::BAD_REQUEST, Json(json!({ "detail": detail })))
                    .into_response();
            }
        }
    }
    if query.format.as_deref() == Some("prometheus") {
        return (
            [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
            prometheus(&loads),
        )
            .into_response();
    }
    Json(json!({
        "timestamp": chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true),
        "version": state.version,
        "accelerator": state.accelerator,
        "num_accelerators": state.num_accelerators,
        "loads": loads,
    }))
    .into_response()
}

pub(super) fn prometheus(loads: &[Value]) -> String {
    use std::fmt::Write;

    let mut output = String::new();
    let mut declared = HashSet::new();
    for load in loads {
        let Some(fields) = load.as_object() else {
            continue;
        };
        let rank = &fields["dp_rank"];
        let mut sample = |name: String, value: &Value| {
            if let Value::Number(value) = value {
                if declared.insert(name.clone()) {
                    let _ = writeln!(output, "# TYPE {name} gauge");
                }
                let _ = writeln!(output, "{name}{{dp_rank=\"{rank}\"}} {value}");
            }
        };
        for (name, value) in fields {
            if name == "dp_rank" {
                continue;
            }
            if let Value::Object(section) = value {
                let prefix = SECTIONS
                    .iter()
                    .find(|(field, _)| *field == name)
                    .map_or(name.as_str(), |(_, alias)| *alias);
                for (field, value) in section {
                    sample(format!("sglang_{prefix}_{field}"), value);
                }
            } else {
                sample(format!("sglang_{name}"), value);
            }
        }
    }
    if output.is_empty() {
        output.push('\n');
    }
    output
}

async fn legacy_load(State(state): State<Arc<LoadState>>) -> Response {
    let loads: Vec<Value> = state.snapshots.0.load_full().into_iter().map(|snapshot| {
        // CPython's perf_counter uses this clock, including across processes.
        let mut now = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        // SAFETY: `now` is a valid, writable timespec for the duration of the call.
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
        json!({
            "dp_rank": snapshot.dp_rank,
            "num_reqs": snapshot.num_running_reqs + snapshot.num_waiting_reqs,
            "num_waiting_reqs": snapshot.num_waiting_reqs,
            "num_tokens": snapshot.num_total_tokens,
            "num_pending_tokens": i128::from(snapshot.num_total_tokens) - i128::from(snapshot.num_used_tokens),
            "ts_tic": now.tv_sec as f64 + now.tv_nsec as f64 / 1e9,
        })
    }).collect();
    Json(loads).into_response()
}

#[cfg(test)]
mod tests {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    use super::*;

    #[tokio::test]
    async fn snapshots_support_filtering_sections_formats_and_legacy_projection() {
        let snapshots = Arc::new(LoadSnapshotStore::default());
        let app = router(
            snapshots.clone(),
            &ServerArgs {
                version: "test-version".into(),
                accelerator: Some("NVIDIA GB300".into()),
                num_accelerators: 2,
                ..Default::default()
            },
        );
        async fn get(app: &Router, uri: &str) -> (StatusCode, Vec<u8>) {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            (
                response.status(),
                to_bytes(response.into_body(), 65536)
                    .await
                    .unwrap()
                    .to_vec(),
            )
        }
        let (_, body) = get(&app, "/v1/loads").await;
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["loads"],
            json!([])
        );
        let fixture: Value =
            serde_json::from_str(include_str!("../../testdata/load_snapshot_python.json")).unwrap();
        let hex = fixture["msgpack_hex"].as_str().unwrap();
        let payload: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        snapshots.publish(&payload).unwrap();
        let snapshot = &fixture["expected"];
        for (uri, expected) in [
            ("/v1/loads", 1),
            ("/v1/loads?dp_rank=1", 1),
            ("/v1/loads?dp_rank=0", 0),
            ("/v1/loads?dp_rank=-1", 0),
        ] {
            let (status, body) = get(&app, uri).await;
            assert_eq!(status, StatusCode::OK);
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["loads"].as_array().unwrap().len(), expected);
            assert_eq!(body["num_accelerators"], 2);
            assert_eq!(body["accelerator"], "NVIDIA GB300");
            assert!(
                chrono::DateTime::parse_from_rfc3339(body["timestamp"].as_str().unwrap()).is_ok()
            );
            if expected != 0 {
                assert_eq!(&body["loads"][0], snapshot);
            }
        }
        let (_, body) = get(&app, "/v1/loads?include=core,queues").await;
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["loads"][0]["queues"]["prealloc_ready"], 1);
        assert!(body["loads"][0].get("memory").is_none());
        assert_eq!(
            get(&app, "/v1/loads?include=unknown").await.0,
            StatusCode::BAD_REQUEST
        );
        let (_, body) = get(&app, "/v1/loads?format=prometheus").await;
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains("sglang_spec_accept_length{dp_rank=\"1\"} 3.0"));
        assert!(text.contains("sglang_disagg_decode_transfer_queue_reqs{dp_rank=\"1\"} 1"));
        assert!(!text.contains("decode_moments"));
        assert!(!text.contains("sglang_disagg_mode"));
        let (_, body) = get(&app, "/get_load").await;
        let mut body: Value = serde_json::from_slice(&body).unwrap();
        assert!(body[0]["ts_tic"].as_f64().unwrap() > 0.0);
        body[0].as_object_mut().unwrap().remove("ts_tic");
        assert_eq!(
            body,
            json!([{"dp_rank": 1, "num_reqs": 5, "num_waiting_reqs": 2, "num_tokens": 150, "num_pending_tokens": 50}])
        );
        assert!(snapshots.publish(b"bad msgpack").is_err());
        assert_eq!(snapshots.0.load_full().unwrap().num_total_tokens, 150);
    }
}
