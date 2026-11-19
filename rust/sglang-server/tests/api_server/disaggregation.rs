//! The PD bootstrap registry a prefill server hosts on its api listener,
//! over the wire the Python decode side and the PD router speak.

use std::io::{Read, Write};
use std::net::SocketAddr;

use sglang_server::DisaggregationMode;
use sglang_server_test::{SchedulerlessServer, free_loopback_addrs};

/// Minimal HTTP/1.1 client: returns `(status, body)`.
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

/// Boot the full frontend on a free port in the PD role `disaggregation_mode`,
/// with no tokenizer load. The bootstrap registry serves on the api listener,
/// so these tests also pin the merge wiring (including its derivation from
/// the role), not just the handlers.
fn start_runtime(disaggregation_mode: DisaggregationMode) -> (SchedulerlessServer, SocketAddr) {
    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .server_args(|args| {
            args.disaggregation_mode = disaggregation_mode;
            // The moved tests ran with the startup warmup on; keep that.
            args.skip_server_warmup = false;
        })
        .start()
        .expect("start runtime");
    (rt, addr)
}

fn start_on_free_port() -> (SchedulerlessServer, SocketAddr) {
    start_runtime(DisaggregationMode::Prefill)
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
            "stores_to_linker": false,
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

#[test]
fn dspark_layout_round_trip_and_rank_mismatch() {
    let (_rt, addr) = start_on_free_port();
    let layout = serde_json::json!({"num_draft_tokens": 6, "state_item_lens": [[4096]]});
    let body = put_route(serde_json::json!({"dsv41_spec_layout": layout}));
    assert_eq!(request(addr, "PUT", "/route", Some(&body)).0, 200);
    let (status, body) = request(addr, "GET", SENTINEL, None);
    assert_eq!(status, 200);
    let info: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(info["dsv41_spec_layout"], layout);

    let incompatible = put_route(serde_json::json!({"dsv41_spec_layout": null}));
    assert_eq!(request(addr, "PUT", "/route", Some(&incompatible)).0, 400);
    let (_, body) = request(addr, "GET", SENTINEL, None);
    let info: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(info["dsv41_spec_layout"], layout);
}

#[test]
fn spec_kv_layout_rank_compatibility() {
    let (_rt, addr) = start_on_free_port();
    let layout = serde_json::json!({
        "algorithm": "DSPARK", "num_draft_entries": 6, "dspark_block_size": 4
    });
    let body = put_route(serde_json::json!({"pp_size": 2, "spec_kv_layout": layout}));
    assert_eq!(request(addr, "PUT", "/route", Some(&body)).0, 200);

    let incompatible = put_route(serde_json::json!({
        "pp_size": 2, "pp_rank": 1,
        "spec_kv_layout": {"algorithm": "DSPARK", "num_draft_entries": 7},
    }));
    assert_eq!(request(addr, "PUT", "/route", Some(&incompatible)).0, 400);
    assert_eq!(request(addr, "GET", SENTINEL, None).0, 503);

    let draftless = put_route(serde_json::json!({"pp_size": 2, "pp_rank": 1}));
    assert_eq!(request(addr, "PUT", "/route", Some(&draftless)).0, 200);
    let (status, body) = request(addr, "GET", SENTINEL, None);
    assert_eq!(status, 200);
    let info: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(info["spec_kv_layout"], layout);
}

/// A window entry as Python's `state_wire(kv_args)` states it: (state
/// type, sub-pool, window tail).
fn wire(sub_pool: &str, tail: i64) -> serde_json::Value {
    serde_json::json!([["swa", sub_pool, tail]])
}

fn sentinel_info(addr: SocketAddr) -> serde_json::Value {
    let (status, body) = request(addr, "GET", SENTINEL, None);
    assert_eq!(status, 200, "{body}");
    serde_json::from_str(&body).unwrap()
}

/// The prefill's state entries reach the decode through the topology
/// sentinel as Python's server answers them: `state_wire` when every stage
/// registers the same entries, `state_wires` by stage when they differ,
/// neither when no stage registers any.
#[test]
fn state_wire_round_trips_through_the_sentinel() {
    let (_rt, addr) = start_on_free_port();
    let body = put_route(serde_json::json!({"state_wire": wire("window_4096", 4096)}));
    assert_eq!(request(addr, "PUT", "/route", Some(&body)).0, 200);
    let info = sentinel_info(addr);
    assert_eq!(info["state_wire"], wire("window_4096", 4096));
    assert!(info.get("state_wires").is_none(), "{info}");

    let (_rt, addr) = start_on_free_port();
    for (pp_rank, state_wire) in [(0, wire("window_4096", 4096)), (1, wire("window_512", 512))] {
        let body = put_route(serde_json::json!({
            "pp_size": 2, "pp_rank": pp_rank, "state_wire": state_wire,
        }));
        assert_eq!(request(addr, "PUT", "/route", Some(&body)).0, 200);
    }
    let info = sentinel_info(addr);
    assert!(info.get("state_wire").is_none(), "{info}");
    assert_eq!(
        info["state_wires"],
        serde_json::json!([wire("window_4096", 4096), wire("window_512", 512)])
    );

    // A stage registering no entries is null in its slot.
    let (_rt, addr) = start_on_free_port();
    let stage0 = put_route(serde_json::json!({"pp_size": 2, "pp_rank": 0}));
    let stage1 = put_route(serde_json::json!({
        "pp_size": 2, "pp_rank": 1, "state_wire": wire("window_512", 512),
    }));
    for body in [stage0, stage1] {
        assert_eq!(request(addr, "PUT", "/route", Some(&body)).0, 200);
    }
    let info = sentinel_info(addr);
    assert!(info.get("state_wire").is_none(), "{info}");
    assert_eq!(
        info["state_wires"],
        serde_json::json!([null, wire("window_512", 512)])
    );

    // A dense prefill registers no state entries (or an empty list): the
    // sentinel states neither field, so a decode reads it as a peer that
    // predates them.
    for registered in [None, Some(serde_json::json!([]))] {
        let (_rt, addr) = start_on_free_port();
        let body = match &registered {
            None => put_route(serde_json::json!({})),
            Some(wire) => put_route(serde_json::json!({"state_wire": wire})),
        };
        assert_eq!(request(addr, "PUT", "/route", Some(&body)).0, 200);
        let info = sentinel_info(addr);
        assert!(info.get("state_wire").is_none(), "{info}");
        assert!(info.get("state_wires").is_none(), "{info}");
    }
}

/// Every rank of one pipeline stage must register the same state entries:
/// a rank that disagrees is refused and not counted, as in Python.
#[test]
fn conflicting_state_wire_in_one_stage_is_refused() {
    let (_rt, addr) = start_on_free_port();
    let rank0 = put_route(serde_json::json!({
        "attn_tp_size": 2, "attn_tp_rank": 0, "state_wire": wire("window_4096", 4096),
    }));
    assert_eq!(request(addr, "PUT", "/route", Some(&rank0)).0, 200);
    for other in [wire("window_512", 512), serde_json::Value::Null] {
        let rank1 = put_route(serde_json::json!({
            "attn_tp_size": 2, "attn_tp_rank": 1, "state_wire": other,
        }));
        let (status, body) = request(addr, "PUT", "/route", Some(&rank1));
        assert_eq!(status, 400, "{body}");
        assert!(body.contains("PD state entries differ"), "{body}");
    }
    // The refused rank is not registered.
    assert_eq!(request(addr, "GET", SENTINEL, None).0, 503);
    let rank1 = put_route(serde_json::json!({
        "attn_tp_size": 2, "attn_tp_rank": 1, "state_wire": wire("window_4096", 4096),
    }));
    assert_eq!(request(addr, "PUT", "/route", Some(&rank1)).0, 200);
    assert_eq!(sentinel_info(addr)["state_wire"], wire("window_4096", 4096));
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
    let (_rt, addr) = start_runtime(DisaggregationMode::Null);

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
