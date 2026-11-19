//! PD KV bootstrap registry — rust port of Python `CommonKVBootstrapServer` (shared by all
//! transfer backends): prefill ranks PUT `/route`, decode ranks GET routes and the `-1`-sentinel
//! topology, the PD router tracks per-room dp ranks; the wire format is Python-owned parity.
//! Mounted on the prefill api listener (bootstrap port = api port) before `init_disaggregation`.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::utils::environ;
use crate::utils::response::json_error;
use crate::utils::serialize::{parse_int, parse_int_opt, parse_int_vec};

/// Python default: `SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL`.
const ENTRY_CLEANUP_INTERVAL_ENV: &str = "SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL";
const ENTRY_CLEANUP_INTERVAL_DEFAULT_SECS: u64 = 120;
const ROOM_SHARD_COUNT: usize = 64;

/// Python's (`PrefillRankInfo`).
#[derive(Clone, Serialize)]
struct PrefillRankInfo {
    rank_ip: String,
    rank_port: i64,
}

/// Python's (`PrefillServerInfo`).
#[derive(Serialize)]
struct PrefillServerInfo {
    attn_tp_size: i64,
    attn_cp_size: i64,
    dp_size: i64,
    pp_size: i64,
    page_size: Option<i64>,
    kv_cache_dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_layout: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dsv41_spec_layout: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    spec_kv_layout: Option<serde_json::Value>,
    /// The PD state entries every pipeline stage registers alike; omitted
    /// when there are none, as a server that predates them answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    state_wire: Option<serde_json::Value>,
    /// Each stage's state entries, by stage, when the stages differ.
    #[serde(skip_serializing_if = "Option::is_none")]
    state_wires: Option<Vec<Option<serde_json::Value>>>,
    follow_bootstrap_room: bool,
    enable_dsa_cache_layer_split: bool,
    stores_to_linker: bool,
    prefill_http_port: Option<i64>,
}

struct RoomEntry {
    dp_rank: i64,
    registered_at: Instant,
}

/// Mirror of the Python server's mutable state, split by write pattern.
#[derive(Default)]
struct Registry {
    /// Copy-on-write topology.
    topology: ArcSwap<Topology>,
    /// Per shard locking map.
    rooms: RoomShards,
}

/// Room→dp-rank entries, sharded `room % `[`ROOM_SHARD_COUNT`].
struct RoomShards([Mutex<HashMap<i64, RoomEntry>>; ROOM_SHARD_COUNT]);

// Manual: `Default` is only derivable for arrays up to 32 elements.
impl Default for RoomShards {
    fn default() -> Self {
        Self(std::array::from_fn(|_| Mutex::new(HashMap::new())))
    }
}

impl RoomShards {
    fn shard(&self, room: i64) -> &Mutex<HashMap<i64, RoomEntry>> {
        &self.0[(room as u64 % ROOM_SHARD_COUNT as u64) as usize]
    }

    fn insert(&self, room: i64, entry: RoomEntry) {
        self.shard(room).lock().unwrap().insert(room, entry);
    }

    fn dp_rank(&self, room: i64) -> Option<i64> {
        self.shard(room)
            .lock()
            .unwrap()
            .get(&room)
            .map(|entry| entry.dp_rank)
    }

    /// Drop entries older than `ttl`, one shard at a time.
    fn sweep(&self, ttl: Duration) {
        for shard in &self.0 {
            shard
                .lock()
                .unwrap()
                .retain(|_, entry| entry.registered_at.elapsed() <= ttl);
        }
    }
}

/// The registration topology.
#[derive(Clone, Default)]
struct Topology {
    attn_tp_size: Option<i64>,
    attn_cp_size: Option<i64>,
    dp_size: Option<i64>,
    pp_size: Option<i64>,
    page_size: Option<i64>,
    kv_cache_dtype: Option<String>,
    kv_layout: Option<String>,
    dsv41_spec_layout: Option<serde_json::Value>,
    spec_kv_layout: Option<serde_json::Value>,
    /// Each pipeline stage's PD state entries (`state_wire`), as its ranks
    /// registered them; every rank of a stage registers the same ones.
    state_wire_by_stage: BTreeMap<i64, Option<serde_json::Value>>,
    follow_bootstrap_room: Option<bool>,
    enable_dsa_cache_layer_split: Option<bool>,
    stores_to_linker: Option<bool>,
    prefill_http_port: Option<i64>,
    /// Keyed `(dp_group, attn_cp_rank, attn_tp_rank, pp_rank)` — the flat form
    /// of Python's nested `prefill_port_table` dicts.
    prefill_ranks: HashMap<(i64, i64, i64, i64), PrefillRankInfo>,
    registered_count: i64,
}

impl Topology {
    /// `dp * cp * tp * pp` once every size is known (saturating: absurd sizes
    /// stay "never ready" instead of overflowing).
    fn expected(&self) -> Option<i64> {
        Some(
            self.dp_size?
                .saturating_mul(self.attn_cp_size?)
                .saturating_mul(self.attn_tp_size?)
                .saturating_mul(self.pp_size?),
        )
    }

    fn is_ready(&self) -> bool {
        self.expected()
            .is_some_and(|expected| self.registered_count >= expected)
    }

    /// The state entries every pipeline stage registers alike, if so (Python
    /// `_state_wire`).
    fn state_wire(&self) -> Option<serde_json::Value> {
        let mut wires = self.state_wire_by_stage.values();
        let first = wires.next()?;
        if wires.all(|wire| wire == first) {
            first.clone()
        } else {
            None
        }
    }

    /// Each stage's state entries, by stage, when the stages differ; None when
    /// no stage registers any (Python `_state_wires`).
    fn state_wires(&self) -> Option<Vec<Option<serde_json::Value>>> {
        let any = self
            .state_wire_by_stage
            .values()
            .any(|wire| wire.as_ref().is_some_and(truthy));
        if !any || self.state_wire().is_some() {
            return None;
        }
        let last = *self.state_wire_by_stage.keys().next_back()?;
        Some(
            (0..=last)
                .map(|stage| self.state_wire_by_stage.get(&stage).cloned().flatten())
                .collect(),
        )
    }
}

/// Python truthiness of a JSON value: null, false, zero and empty strings,
/// lists and objects are false.
fn truthy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(b) => *b,
        serde_json::Value::Number(n) => n.as_f64() != Some(0.0),
        serde_json::Value::String(s) => !s.is_empty(),
        serde_json::Value::Array(a) => !a.is_empty(),
        serde_json::Value::Object(o) => !o.is_empty(),
    }
}

/// PUT /route payload (`CommonKVManager.register_to_bootstrap`).
#[derive(Deserialize)]
struct Route {
    attn_tp_size: i64,
    attn_tp_rank: i64,
    attn_cp_size: i64,
    attn_cp_rank: i64,
    attn_dp_size: i64,
    attn_dp_rank: i64,
    pp_size: i64,
    pp_rank: i64,
    system_dp_size: i64,
    system_dp_rank: i64,
    rank_ip: String,
    #[serde(deserialize_with = "parse_int")]
    rank_port: i64,
    #[serde(deserialize_with = "parse_int")]
    page_size: i64,
    #[serde(default)]
    kv_cache_dtype: Option<String>,
    #[serde(default)]
    kv_layout: Option<String>,
    #[serde(default)]
    dsv41_spec_layout: Option<serde_json::Value>,
    #[serde(default)]
    spec_kv_layout: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "parse_int_opt")]
    prefill_http_port: Option<i64>,
    #[serde(default)]
    load_balance_method: Option<String>,
    #[serde(default)]
    enable_dsa_cache_layer_split: Option<bool>,
    #[serde(default)]
    stores_to_linker: Option<bool>,
    /// The rank's PD state entries (Python `state_wire(kv_args)`); absent from
    /// a rank that registers none.
    #[serde(default)]
    state_wire: Option<serde_json::Value>,
}

async fn route_put(State(state): State<Arc<Registry>>, Json(body): Json<Route>) -> Response {
    // `system_dp_size == 1` → attention-dp topology; else system-dp topology.
    let dp_size = if body.system_dp_size == 1 {
        body.attn_dp_size
    } else {
        body.system_dp_size
    };
    let dp_group = if body.system_dp_size == 1 {
        body.attn_dp_rank
    } else {
        body.system_dp_rank
    };

    // Copy-on-write update. `rcu` may re-run the closure under write
    // contention, so it only reads `body` and clones what it stores.
    let mut layout_mismatch = false;
    let mut spec_mismatch = false;
    let mut state_wire_mismatch = false;
    state.topology.rcu(|current| {
        let mut topo = (**current).clone();
        layout_mismatch =
            topo.registered_count > 0 && topo.dsv41_spec_layout != body.dsv41_spec_layout;
        spec_mismatch = topo.spec_kv_layout.is_some()
            && body.spec_kv_layout.is_some()
            && topo.spec_kv_layout != body.spec_kv_layout;
        state_wire_mismatch = topo
            .state_wire_by_stage
            .get(&body.pp_rank)
            .is_some_and(|wire| *wire != body.state_wire);
        if layout_mismatch || spec_mismatch || state_wire_mismatch {
            return topo;
        }
        topo.dsv41_spec_layout = body.dsv41_spec_layout.clone();
        if topo.spec_kv_layout.is_none() {
            topo.spec_kv_layout = body.spec_kv_layout.clone();
        }
        topo.state_wire_by_stage
            .insert(body.pp_rank, body.state_wire.clone());
        topo.attn_tp_size.get_or_insert(body.attn_tp_size);
        topo.attn_cp_size.get_or_insert(body.attn_cp_size);
        topo.dp_size.get_or_insert(dp_size);
        topo.pp_size.get_or_insert(body.pp_size);
        topo.page_size.get_or_insert(body.page_size);
        if topo.kv_cache_dtype.is_none() {
            topo.kv_cache_dtype = body.kv_cache_dtype.clone();
        }
        if topo.kv_layout.is_none() {
            topo.kv_layout = body.kv_layout.clone();
        }
        if topo.prefill_http_port.is_none() {
            topo.prefill_http_port = body.prefill_http_port;
        }
        topo.follow_bootstrap_room.get_or_insert(
            body.load_balance_method
                .as_deref()
                .unwrap_or("follow_bootstrap_room")
                == "follow_bootstrap_room",
        );
        topo.enable_dsa_cache_layer_split
            .get_or_insert(body.enable_dsa_cache_layer_split.unwrap_or(false));
        topo.stores_to_linker
            .get_or_insert(body.stores_to_linker.unwrap_or(false));
        topo.prefill_ranks.insert(
            (dp_group, body.attn_cp_rank, body.attn_tp_rank, body.pp_rank),
            PrefillRankInfo {
                rank_ip: body.rank_ip.clone(),
                rank_port: body.rank_port,
            },
        );
        topo.registered_count += 1;
        topo
    });

    if layout_mismatch {
        return json_error(
            StatusCode::BAD_REQUEST,
            "DeepSeek-V4.1 DSpark PD layout differs across prefill ranks",
        );
    }
    if spec_mismatch {
        return json_error(
            StatusCode::BAD_REQUEST,
            "Speculative KV layout differs across prefill ranks",
        );
    }
    if state_wire_mismatch {
        return json_error(
            StatusCode::BAD_REQUEST,
            "PD state entries differ across the ranks of one pipeline stage",
        );
    }

    let topo = state.topology.load();
    tracing::debug!(
        dp_group,
        cp = body.attn_cp_rank,
        tp = body.attn_tp_rank,
        pp = body.pp_rank,
        rank_ip = %body.rank_ip,
        rank_port = body.rank_port,
        registered = topo.registered_count,
        expected = topo.expected(),
        "registered prefill bootstrap rank"
    );
    "OK".into_response()
}

async fn route_get(
    State(state): State<Arc<Registry>>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    // A missing, empty (Python truthiness), or non-integer param → 400.
    let rank = |k: &str| query.get(k).and_then(|v| v.trim().parse::<i64>().ok());
    let (Some(dp), Some(cp), Some(tp), Some(pp)) = (
        rank("prefill_dp_rank"),
        rank("prefill_cp_rank"),
        rank("target_tp_rank"),
        rank("target_pp_rank"),
    ) else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "Missing inputs for bootstrap server.",
        );
    };

    let topo = state.topology.load();
    // Python checks readiness in both branches; hoisted, same behavior.
    if !topo.is_ready() {
        let registered_count = topo.registered_count;
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            &format!(
                "Prefill server not fully registered yet ({registered_count} workers registered)."
            ),
        );
    }

    if (dp, cp, tp, pp) == (-1, -1, -1, -1) {
        // Aggregate-topology sentinel.
        return Json(PrefillServerInfo {
            attn_tp_size: topo.attn_tp_size.unwrap(),
            attn_cp_size: topo.attn_cp_size.unwrap(),
            dp_size: topo.dp_size.unwrap(),
            pp_size: topo.pp_size.unwrap(),
            page_size: topo.page_size,
            kv_cache_dtype: topo.kv_cache_dtype.clone(),
            kv_layout: topo.kv_layout.clone(),
            dsv41_spec_layout: topo.dsv41_spec_layout.clone(),
            spec_kv_layout: topo.spec_kv_layout.clone(),
            // No state entries are stated as none, as a server that predates
            // them answers, so an older decode reads a dense prefill's answer.
            state_wire: topo.state_wire().filter(truthy),
            state_wires: topo.state_wires().filter(|wires| !wires.is_empty()),
            follow_bootstrap_room: topo.follow_bootstrap_room.unwrap_or(true),
            enable_dsa_cache_layer_split: topo.enable_dsa_cache_layer_split.unwrap_or(false),
            stores_to_linker: topo.stores_to_linker.unwrap_or(false),
            prefill_http_port: topo.prefill_http_port,
        })
        .into_response();
    }

    match topo.prefill_ranks.get(&(dp, cp, tp, pp)) {
        Some(info) => Json(info.clone()).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            format!(
                "Bootstrap info not found for dp_rank={dp} cp_rank={cp} \
                 tp_rank={tp} pp_rank={pp}"
            ),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct RegisterDpRank {
    #[serde(deserialize_with = "parse_int")]
    bootstrap_room: i64,
    #[serde(deserialize_with = "parse_int")]
    dp_rank: i64,
}

async fn register_dp_rank(
    State(state): State<Arc<Registry>>,
    Json(body): Json<RegisterDpRank>,
) -> Response {
    state.rooms.insert(
        body.bootstrap_room,
        RoomEntry {
            dp_rank: body.dp_rank,
            registered_at: Instant::now(),
        },
    );
    "OK".into_response()
}

#[derive(Deserialize)]
struct QueryDpRanks {
    #[serde(deserialize_with = "parse_int_vec")]
    bootstrap_rooms: Vec<i64>,
}

/// Unknown rooms are silently omitted from the response, not an error. JSON
/// object keys are strings — Python's `str(room_int)` for free.
async fn query_dp_ranks(
    State(state): State<Arc<Registry>>,
    Json(body): Json<QueryDpRanks>,
) -> Response {
    let result: HashMap<String, i64> = body
        .bootstrap_rooms
        .iter()
        .filter_map(|room| Some((room.to_string(), state.rooms.dp_rank(*room)?)))
        .collect();
    Json(result).into_response()
}

/// No `/health` here: the merged api router already serves it (same 200 "OK"
/// the standalone Python bootstrap server answered, so probes are unchanged).
fn router(state: Arc<Registry>) -> Router {
    Router::new()
        // Unmatched methods on a routed path get axum's built-in 405, matching
        // Python's explicit method_not_allowed branch.
        .route("/route", put(route_put).get(route_get))
        .route("/register_dp_rank", post(register_dp_rank))
        .route("/query_dp_ranks", post(query_dp_ranks))
        .with_state(state)
}

/// Drop room entries
async fn cleanup_sweeper(state: Arc<Registry>) {
    let cleanup_interval = Duration::from_secs(
        environ::env_i64(
            ENTRY_CLEANUP_INTERVAL_ENV,
            ENTRY_CLEANUP_INTERVAL_DEFAULT_SECS as i64,
        )
        .max(0) as u64,
    );
    loop {
        tokio::time::sleep(cleanup_interval).await;
        state.rooms.sweep(cleanup_interval);
    }
}

pub(crate) fn router_and_sweeper() -> (Router, impl std::future::Future<Output = ()>) {
    let state = Arc::new(Registry::default());
    let sweeper = cleanup_sweeper(state.clone());
    (router(state), sweeper)
}
