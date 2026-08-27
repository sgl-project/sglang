//! PD KV bootstrap registry — the rust port of Python
//! `CommonKVBootstrapServer` (disaggregation/common/conn.py), which every
//! transfer backend (mooncake / mori / nixl / ascend) subclasses without
//! overrides, so this one implementation covers them all.
//!
//! Prefill ranks PUT their `{rank_ip, rank_port}` to `/route`; decode ranks
//! GET per-rank routes and the aggregate topology (the all `-1` sentinel
//! query); the PD router registers/queries per-room dp ranks. The wire
//! protocol is Python-owned — field names, status codes, and the `-1`
//! sentinel below are parity pins, not this crate's design.
//!
//! Served on the api listener itself: `api_server::serve` merges
//! [`router_and_sweeper`]'s routes on every prefill server
//! (`ServerArgs::enable_pd_bootstrap()`). In rust-server mode the resolved
//! `disaggregation_bootstrap_port` is aliased to the api port, so KV managers
//! register here without knowing about the merge. The scheduler starts the
//! rust server BEFORE `init_disaggregation` — the KV managers register
//! synchronously there, with only a few bounded retries, so the routes must
//! already be accepting.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{post, put};
use axum::{Json, Router};

/// Python default: `SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL = EnvInt(120)`.
const ENTRY_CLEANUP_INTERVAL_ENV: &str = "SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL";
const ENTRY_CLEANUP_INTERVAL_DEFAULT_SECS: u64 = 120;

/// One registered prefill rank, exactly the JSON the Python decode side
/// consumes (`PrefillRankInfo`).
#[derive(Clone, serde::Serialize)]
struct RankInfo {
    rank_ip: String,
    rank_port: i64,
}

/// The all-`-1` sentinel response, exactly Python's
/// `dataclasses.asdict(PrefillServerInfo)` shape.
#[derive(serde::Serialize)]
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

/// Mirror of the Python server's mutable state. First PUT wins for the
/// topology scalars (Python's `if self.x is None` pattern); `registered_count`
/// counts raw PUTs, and readiness is `count >= dp*cp*tp*pp` — both verbatim
/// from Python, re-registrations included.
#[derive(Default)]
struct Registry {
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
    prefill_ranks: HashMap<(i64, i64, i64, i64), RankInfo>,
    room_to_dp_rank: HashMap<i64, RoomEntry>,
    registered_count: i64,
}

impl Registry {
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

type Shared = Arc<Mutex<Registry>>;

/// i64 that also accepts a numeric string — used exactly where Python coerces
/// with `int(data[...])`, so the wire stays as tolerant as the original.
#[derive(Clone, Copy)]
struct Int(i64);

impl<'de> serde::Deserialize<'de> for Int {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Int(i64),
            Str(String),
        }
        match Raw::deserialize(d)? {
            Raw::Int(v) => Ok(Int(v)),
            // `int(...)` tolerates surrounding whitespace.
            Raw::Str(s) => s
                .trim()
                .parse()
                .map(Int)
                .map_err(|_| serde::de::Error::custom(format!("invalid int: {s:?}"))),
        }
    }
}

/// PUT /route payload (`CommonKVManager.register_to_bootstrap`).
#[derive(serde::Deserialize)]
struct RoutePut {
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
    rank_port: Int,
    page_size: Int,
    #[serde(default)]
    kv_cache_dtype: Option<String>,
    #[serde(default)]
    prefill_http_port: Option<Int>,
    #[serde(default)]
    load_balance_method: Option<String>,
    #[serde(default)]
    enable_dsa_cache_layer_split: Option<bool>,
}

fn not_ready(registered_count: i64) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        format!("Prefill server not fully registered yet ({registered_count} workers registered)."),
    )
        .into_response()
}

async fn route_put(State(state): State<Shared>, Json(body): Json<RoutePut>) -> Response {
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

    let mut reg = state.lock().unwrap();
    reg.attn_tp_size.get_or_insert(body.attn_tp_size);
    reg.attn_cp_size.get_or_insert(body.attn_cp_size);
    reg.dp_size.get_or_insert(dp_size);
    reg.pp_size.get_or_insert(body.pp_size);
    reg.page_size.get_or_insert(body.page_size.0);
    if reg.kv_cache_dtype.is_none() {
        reg.kv_cache_dtype = body.kv_cache_dtype;
    }
    if reg.prefill_http_port.is_none() {
        reg.prefill_http_port = body.prefill_http_port.map(|p| p.0);
    }
    reg.follow_bootstrap_room.get_or_insert(
        body.load_balance_method
            .as_deref()
            .unwrap_or("follow_bootstrap_room")
            == "follow_bootstrap_room",
    );
    reg.enable_dsa_cache_layer_split
        .get_or_insert(body.enable_dsa_cache_layer_split.unwrap_or(false));

    reg.prefill_ranks.insert(
        (dp_group, body.attn_cp_rank, body.attn_tp_rank, body.pp_rank),
        RankInfo {
            rank_ip: body.rank_ip.clone(),
            rank_port: body.rank_port.0,
        },
    );
    reg.registered_count += 1;

    tracing::debug!(
        dp_group,
        cp = body.attn_cp_rank,
        tp = body.attn_tp_rank,
        pp = body.pp_rank,
        rank_ip = %body.rank_ip,
        rank_port = body.rank_port.0,
        registered = reg.registered_count,
        expected = reg.expected(),
        "registered prefill bootstrap rank"
    );
    "OK".into_response()
}

async fn route_get(
    State(state): State<Shared>,
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
        return (
            StatusCode::BAD_REQUEST,
            "Missing inputs for bootstrap server.",
        )
            .into_response();
    };

    let reg = state.lock().unwrap();
    // Python checks readiness in both branches; hoisted, same behavior.
    if !reg.is_ready() {
        return not_ready(reg.registered_count);
    }

    if (dp, cp, tp, pp) == (-1, -1, -1, -1) {
        // Aggregate-topology sentinel. The sizes are Some — `is_ready` above
        // requires all four.
        return Json(PrefillServerInfo {
            attn_tp_size: reg.attn_tp_size.unwrap(),
            attn_cp_size: reg.attn_cp_size.unwrap(),
            dp_size: reg.dp_size.unwrap(),
            pp_size: reg.pp_size.unwrap(),
            page_size: reg.page_size,
            kv_cache_dtype: reg.kv_cache_dtype.clone(),
            follow_bootstrap_room: reg.follow_bootstrap_room.unwrap_or(true),
            enable_dsa_cache_layer_split: reg.enable_dsa_cache_layer_split.unwrap_or(false),
            prefill_http_port: reg.prefill_http_port,
        })
        .into_response();
    }

    match reg.prefill_ranks.get(&(dp, cp, tp, pp)) {
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

#[derive(serde::Deserialize)]
struct RegisterDpRank {
    bootstrap_room: Int,
    dp_rank: Int,
}

async fn register_dp_rank(
    State(state): State<Shared>,
    Json(body): Json<RegisterDpRank>,
) -> Response {
    state.lock().unwrap().room_to_dp_rank.insert(
        body.bootstrap_room.0,
        RoomEntry {
            dp_rank: body.dp_rank.0,
            registered_at: Instant::now(),
        },
    );
    "OK".into_response()
}

#[derive(serde::Deserialize)]
struct QueryDpRanks {
    bootstrap_rooms: Vec<Int>,
}

/// Unknown rooms are silently omitted from the response, not an error. JSON
/// object keys are strings — Python's `str(room_int)` for free.
async fn query_dp_ranks(State(state): State<Shared>, Json(body): Json<QueryDpRanks>) -> Response {
    let reg = state.lock().unwrap();
    let result: HashMap<String, i64> = body
        .bootstrap_rooms
        .iter()
        .filter_map(|room| {
            let entry = reg.room_to_dp_rank.get(&room.0)?;
            Some((room.0.to_string(), entry.dp_rank))
        })
        .collect();
    Json(result).into_response()
}

/// No `/health` here: the merged api router already serves it (same 200 "OK"
/// the standalone Python bootstrap server answered, so probes are unchanged).
fn router(state: Shared) -> Router {
    Router::new()
        // Unmatched methods on a routed path get axum's built-in 405, matching
        // Python's explicit method_not_allowed branch.
        .route("/route", put(route_put).get(route_get))
        .route("/register_dp_rank", post(register_dp_rank))
        .route("/query_dp_ranks", post(query_dp_ranks))
        .with_state(state)
}

/// Drop `room_to_dp_rank` entries older than `interval` — Python's
/// `_cleanup_expired_entries` loop (interval doubles as both period and TTL).
async fn cleanup_expired_entries(state: Shared, interval: Duration) {
    loop {
        tokio::time::sleep(interval).await;
        state
            .lock()
            .unwrap()
            .room_to_dp_rank
            .retain(|_, entry| entry.registered_at.elapsed() <= interval);
    }
}

/// The registry routes plus their expiry sweeper, ready to mount on the api
/// router (`merge` the router, `tokio::spawn` the sweeper on the api runtime —
/// the runtime drop on shutdown cancels it along with the handlers).
pub(crate) fn router_and_sweeper() -> (Router, impl std::future::Future<Output = ()>) {
    let state = Shared::default();
    let cleanup_interval = Duration::from_secs(crate::environ::env_u64(
        ENTRY_CLEANUP_INTERVAL_ENV,
        ENTRY_CLEANUP_INTERVAL_DEFAULT_SECS,
    ));
    (
        router(state.clone()),
        cleanup_expired_entries(state, cleanup_interval),
    )
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
