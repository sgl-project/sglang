//! PD KV bootstrap registry — rust port of Python `CommonKVBootstrapServer` (shared by all
//! transfer backends): prefill ranks PUT `/route`, decode ranks GET routes and the `-1`-sentinel
//! topology, the PD router tracks per-room dp ranks; the wire format is Python-owned parity.
//! Mounted on the prefill api listener (bootstrap port = api port) before `init_disaggregation`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

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
    follow_bootstrap_room: bool,
    enable_dsa_cache_layer_split: bool,
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
    follow_bootstrap_room: Option<bool>,
    enable_dsa_cache_layer_split: Option<bool>,
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
    #[serde(default, deserialize_with = "parse_int_opt")]
    prefill_http_port: Option<i64>,
    #[serde(default)]
    load_balance_method: Option<String>,
    #[serde(default)]
    enable_dsa_cache_layer_split: Option<bool>,
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
    state.topology.rcu(|current| {
        let mut topo = (**current).clone();
        topo.attn_tp_size.get_or_insert(body.attn_tp_size);
        topo.attn_cp_size.get_or_insert(body.attn_cp_size);
        topo.dp_size.get_or_insert(dp_size);
        topo.pp_size.get_or_insert(body.pp_size);
        topo.page_size.get_or_insert(body.page_size);
        if topo.kv_cache_dtype.is_none() {
            topo.kv_cache_dtype = body.kv_cache_dtype.clone();
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
            follow_bootstrap_room: topo.follow_bootstrap_room.unwrap_or(true),
            enable_dsa_cache_layer_split: topo.enable_dsa_cache_layer_split.unwrap_or(false),
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
    let cleanup_interval = Duration::from_secs(crate::environ::env_u64(
        ENTRY_CLEANUP_INTERVAL_ENV,
        ENTRY_CLEANUP_INTERVAL_DEFAULT_SECS,
    ));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{Runtime, RuntimeConfig, RustServerServerArgs, ServerArgs};
    use std::io::{Read, Write};
    use std::net::SocketAddr;

    /// Minimal HTTP/1.1 client (same style as the `runtime` tests): returns
    /// `(status, body)`.
    fn request(
        addr: SocketAddr,
        method: &str,
        path_query: &str,
        body: Option<&serde_json::Value>,
    ) -> (u16, String) {
        let body = body.map(|b| b.to_string()).unwrap_or_default();
        let mut conn = std::net::TcpStream::connect(addr).expect("connect");
        let req = format!(
            "{method} {path_query} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        conn.write_all(req.as_bytes()).unwrap();
        let mut response = String::new();
        conn.read_to_string(&mut response).unwrap();
        let status = response
            .split_whitespace()
            .nth(1)
            .expect("status line")
            .parse()
            .expect("status code");
        let body = response
            .split_once("\r\n\r\n")
            .map(|(_, b)| b.to_string())
            .unwrap_or_default();
        (status, body)
    }

    fn put_route(overrides: serde_json::Value) -> serde_json::Value {
        let mut body = serde_json::json!({
            "attn_tp_size": 1, "attn_tp_rank": 0,
            "attn_cp_size": 1, "attn_cp_rank": 0,
            "attn_dp_size": 1, "attn_dp_rank": 0,
            "pp_size": 1, "pp_rank": 0,
            "system_dp_size": 1, "system_dp_rank": 0,
            "rank_ip": "10.0.0.1", "rank_port": 17000,
            "page_size": 64, "kv_cache_dtype": "auto",
            "load_balance_method": "follow_bootstrap_room",
            "enable_dsa_cache_layer_split": false,
            "prefill_http_port": 30000,
        });
        body.as_object_mut()
            .unwrap()
            .extend(overrides.as_object().unwrap().clone());
        body
    }

    const SENTINEL: &str =
        "/route?prefill_dp_rank=-1&prefill_cp_rank=-1&target_tp_rank=-1&target_pp_rank=-1";

    /// Minimal prefill boot blob (same shape as the `runtime` tests): no
    /// tokenizer load, the two mandatory `model_config` fields, and the
    /// prefill role that mounts the registry.
    const TEST_SERVER_ARGS: &str = r#"{
        "skip_tokenizer_init": true,
        "disaggregation_mode": "prefill",
        "model_config": {"context_len": 2048, "vocab_size": 1000}
    }"#;

    /// Pick a free port (probe-bind pattern, as in the `runtime` tests) and
    /// boot the full runtime there with the bootstrap registry mounted — the
    /// registry serves on the api listener, so these tests also pin the merge
    /// wiring (including the `enable_pd_bootstrap()` derivation from the
    /// blob), not just the handlers.
    fn start_on_free_port() -> (Runtime, SocketAddr) {
        start_runtime(TEST_SERVER_ARGS)
    }

    fn start_runtime(server_args_json: &str) -> (Runtime, SocketAddr) {
        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = probe.local_addr().unwrap();
        drop(probe);
        let cfg = RuntimeConfig {
            rust_server_args: RustServerServerArgs {
                http_addr: addr,
                api_worker_num: 1,
                ..Default::default()
            },
            server_args: Arc::new(ServerArgs::from_json(server_args_json).unwrap()),
        };
        (crate::runtime::start(cfg).expect("start runtime"), addr)
    }

    /// The full wire contract the Python decode side / PD router depends on:
    /// 503 until every rank is registered, the `-1` sentinel returning
    /// `PrefillServerInfo` with Python's exact field names, per-rank lookup
    /// returning `PrefillRankInfo`, 404 for an unknown rank, 400 for missing
    /// params, and `int(...)`-style acceptance of a string `rank_port`. Field
    /// names and status codes are external literals owned by
    /// disaggregation/common/conn.py — this pins the copy.
    #[test]
    fn route_contract_matches_python_client() {
        let (_rt, addr) = start_on_free_port();

        // Not registered yet → 503 (the decode side retries on exactly this).
        let (status, _) = request(addr, "GET", SENTINEL, None);
        assert_eq!(status, 503);

        // Missing/empty params → 400.
        let (status, _) = request(addr, "GET", "/route?prefill_dp_rank=0", None);
        assert_eq!(status, 400);
        let (status, _) = request(
            addr,
            "GET",
            "/route?prefill_dp_rank=0&prefill_cp_rank=&target_tp_rank=0&target_pp_rank=0",
            None,
        );
        assert_eq!(status, 400);

        // Register the single rank; rank_port as a STRING (Python coerces
        // with `int(data["rank_port"])`, so the wire tolerates it).
        let body = put_route(serde_json::json!({"rank_port": "17000"}));
        let (status, text) = request(addr, "PUT", "/route", Some(&body));
        assert_eq!((status, text.as_str()), (200, "OK"));

        // Sentinel now serves the topology, keys verbatim from
        // `dataclasses.asdict(PrefillServerInfo)`.
        let (status, body) = request(addr, "GET", SENTINEL, None);
        assert_eq!(status, 200);
        let info: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            info,
            serde_json::json!({
                "attn_tp_size": 1, "attn_cp_size": 1, "dp_size": 1, "pp_size": 1,
                "page_size": 64, "kv_cache_dtype": "auto",
                "follow_bootstrap_room": true,
                "enable_dsa_cache_layer_split": false,
                "prefill_http_port": 30000,
            })
        );

        // Per-rank lookup: `PrefillRankInfo` shape, rank_port back as an int.
        let (status, body) = request(
            addr,
            "GET",
            "/route?prefill_dp_rank=0&prefill_cp_rank=0&target_tp_rank=0&target_pp_rank=0",
            None,
        );
        assert_eq!(status, 200);
        let rank: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(
            rank,
            serde_json::json!({"rank_ip": "10.0.0.1", "rank_port": 17000})
        );

        // Unknown rank → 404 (ready, but no such entry).
        let (status, _) = request(
            addr,
            "GET",
            "/route?prefill_dp_rank=1&prefill_cp_rank=0&target_tp_rank=0&target_pp_rank=0",
            None,
        );
        assert_eq!(status, 404);
    }

    /// System-dp topology derivation: with `system_dp_size > 1` the dp axis
    /// (readiness expectation AND rank keying) comes from `system_dp_*`, not
    /// `attn_dp_*` — a "looks equivalent" simplification to always using
    /// `attn_dp_*` passes single-dp tests but strands multi-dp deployments at
    /// 503 / wrong-rank routes.
    #[test]
    fn system_dp_drives_readiness_and_rank_keys() {
        let (_rt, addr) = start_on_free_port();

        let rank0 = put_route(serde_json::json!({
            "system_dp_size": 2, "system_dp_rank": 0, "rank_ip": "10.0.0.1",
        }));
        let (status, _) = request(addr, "PUT", "/route", Some(&rank0));
        assert_eq!(status, 200);

        // dp_size resolved to system_dp_size=2 → one registration isn't ready.
        let (status, _) = request(addr, "GET", SENTINEL, None);
        assert_eq!(status, 503);

        let rank1 = put_route(serde_json::json!({
            "system_dp_size": 2, "system_dp_rank": 1, "rank_ip": "10.0.0.2",
        }));
        let (status, _) = request(addr, "PUT", "/route", Some(&rank1));
        assert_eq!(status, 200);

        let (status, body) = request(addr, "GET", SENTINEL, None);
        assert_eq!(status, 200);
        let info: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(info["dp_size"], 2);

        // Ranks are keyed by system_dp_rank: dp=1 must be the second rank's ip.
        let (status, body) = request(
            addr,
            "GET",
            "/route?prefill_dp_rank=1&prefill_cp_rank=0&target_tp_rank=0&target_pp_rank=0",
            None,
        );
        assert_eq!(status, 200);
        let rank: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(rank["rank_ip"], "10.0.0.2");
    }

    /// The PD router's room→dp-rank side channel: register/query round-trip
    /// with Python's `{str(room): dp_rank}` response shape, unknown rooms
    /// silently omitted (not an error). (`/health` liveness now belongs to the
    /// api router the registry is merged into.)
    #[test]
    fn dp_rank_round_trip() {
        let (_rt, addr) = start_on_free_port();

        let (status, text) = request(
            addr,
            "POST",
            "/register_dp_rank",
            Some(&serde_json::json!({"bootstrap_room": 42, "dp_rank": 3})),
        );
        assert_eq!((status, text.as_str()), (200, "OK"));

        let (status, body) = request(
            addr,
            "POST",
            "/query_dp_ranks",
            Some(&serde_json::json!({"bootstrap_rooms": [42, 99]})),
        );
        assert_eq!(status, 200);
        let result: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(result, serde_json::json!({"42": 3}));
    }

    /// The registry mounts only on prefill (`enable_pd_bootstrap()`): a
    /// non-prefill server must 404 the bootstrap routes rather than host an
    /// empty replica — that replica would answer 503 "not registered" forever,
    /// hiding a misdirected decode/router behind its retry loop.
    #[test]
    fn routes_absent_off_prefill() {
        let non_prefill = r#"{
            "skip_tokenizer_init": true,
            "model_config": {"context_len": 2048, "vocab_size": 1000}
        }"#;
        let (_rt, addr) = start_runtime(non_prefill);

        let (status, _) = request(addr, "GET", SENTINEL, None);
        assert_eq!(status, 404);
        let (status, _) = request(
            addr,
            "PUT",
            "/route",
            Some(&put_route(serde_json::json!({}))),
        );
        assert_eq!(status, 404);
    }
}
