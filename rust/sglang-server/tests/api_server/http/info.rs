//! The control routes over HTTP: what each sends the scheduler and how the
//! scheduler's outcome is answered, in Python's statuses and bodies.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use sglang_server::{DisaggregationMode, ServerArgs};
use sglang_server_test::{ApiKeys, OutputDetails, SchedulerlessServer, free_loopback_addrs};

use crate::common::{
    HttpResponse, Sse, header_json, http_bytes, next_scheduler_request, next_scheduler_work,
    take_startup_freeze_gc,
};

const ADMIN: &str = "Authorization: Bearer sk-admin";
const JSON: &str = "Content-Type: application/json";

/// A server with an admin key, which the profile and weight routes need.
fn server() -> (SchedulerlessServer, SocketAddr) {
    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .server_args(|args| args.api_keys = ApiKeys::new(None, Some("sk-admin".into())))
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    (rt, addr)
}

/// Send the request as the admin, with a JSON content type when `typed`.
fn send(addr: SocketAddr, method: &str, path: &str, typed: bool, body: &str) -> HttpResponse {
    let headers: &[&str] = if typed { &[ADMIN, JSON] } else { &[ADMIN] };
    http_bytes(addr, method, path, headers, body.as_bytes())
}

/// Send the request, answer the one control message it puts on the ring with
/// `reply`, and return that message's header and the response.
fn round_trip(
    (rt, addr): &(SchedulerlessServer, SocketAddr),
    method: &'static str,
    path: &'static str,
    typed: bool,
    body: &'static str,
    reply: Value,
) -> (Vec<Value>, HttpResponse) {
    let addr = *addr;
    let client = std::thread::spawn(move || send(addr, method, path, typed, body));
    let request = next_scheduler_work(rt);
    let rid = request.rid().expect("control rid");
    rt.push_control_result(&rid, &reply).unwrap();
    let Value::Array(header) = header_json(&request) else {
        panic!("{path}: the header must be an array");
    };
    (header, client.join().unwrap())
}

fn assert_nothing_submitted(rt: &SchedulerlessServer, what: &str) {
    assert!(
        rt.next_request(Duration::from_millis(200)).is_none(),
        "{what} reached the scheduler"
    );
}

fn outcome(success: bool, message: &str) -> Value {
    json!({"rid": "x", "http_worker_ipc": null, "success": success, "message": message})
}

/// Python's `get_bool_env_var`: unset is the default, else `true` or `1`.
fn bool_env(name: &str, default: bool) -> bool {
    std::env::var(name).map_or(default, |raw| {
        matches!(raw.to_lowercase().as_str(), "true" | "1")
    })
}

/// `/flush_cache` relays one `FlushCacheReqInput` carrying the query
/// timeout and answers with Python's status and plain-text body; a
/// negative timeout is refused before anything reaches the scheduler.
#[test]
fn flush_cache_relays_the_scheduler_outcome() {
    let app = server();
    for (method, uri, timeout, reply, status, body) in [
        (
            "POST",
            "/flush_cache?timeout=30",
            30.0,
            (true, ""),
            200,
            "Cache flushed.\nPlease check backend logs for more details. (When there \
             are running or waiting requests, the operation will not be performed.)\n",
        ),
        (
            "GET",
            "/flush_cache",
            0.0,
            (false, "Timed out waiting for idle state."),
            400,
            "Timed out waiting for idle state.",
        ),
        (
            "POST",
            "/flush_cache",
            0.0,
            (false, ""),
            400,
            "Flush cache failed.\n",
        ),
    ] {
        let (header, response) =
            round_trip(&app, method, uri, false, "", outcome(reply.0, reply.1));
        assert_eq!(header[0], "FlushCacheReqInput");
        assert_eq!(header[3].as_f64(), Some(timeout), "{uri}");
        assert_eq!(response.status, status, "{uri}");
        assert_eq!(response.header("content-type"), None, "{uri}");
        assert_eq!(response.body, body);
    }

    let response = send(app.1, "POST", "/flush_cache?timeout=-1", false, "");
    assert_eq!(response.status, 400);
    assert_nothing_submitted(&app.0, "a negative timeout");
}

/// `/weights_checker` relays one `CheckWeightsReqInput` carrying the body's
/// fields (Python's defaults for a missing body) and answers with Python's
/// JSON: the rank records as sent, in order, with the SHA-256 of their
/// `per_gpu_checksum`s; no `ranks` when no rank sent one; 400 on failure.
/// A malformed body is refused before anything reaches the scheduler.
#[test]
fn weights_checker_relays_the_scheduler_outcome() {
    let app = server();
    let ranks = json!([
        {"checksums": {"model.w": "a"}, "per_gpu_checksum": "a",
         "parallelism_info": [{"role": "target", "tp_rank": 0, "tp_size": 2}]},
        {"checksums": {"model.w": "b"}, "per_gpu_checksum": "b",
         "parallelism_info": [{"role": "target", "tp_rank": 1, "tp_size": 2}]},
    ]);
    for (method, body, request, reply, status, response_body) in [
        (
            "POST",
            r#"{"action": "fingerprint", "allow_quant_error": true,
                "skip_tensor_list": ["lm_head"], "selector": "target", "rid": "x",
                "names": ["model.weight"]}"#,
            (
                "fingerprint",
                true,
                Some("lm_head"),
                "target",
                Some("model.weight"),
            ),
            (true, "Success.", ranks.clone()),
            200,
            json!({
                "success": true,
                "message": "Success.",
                "ranks": ranks,
                "per_engine_checksum":
                    "fb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603",
            }),
        ),
        (
            "GET",
            "",
            ("checksum", false, None, "all", None),
            (false, "rank 1 failed", Value::Null),
            400,
            json!({"success": false, "message": "rank 1 failed"}),
        ),
        (
            "POST",
            "null",
            ("checksum", false, None, "all", None),
            (true, "Success.", json!([])),
            200,
            json!({
                "success": true,
                "message": "Success.",
                "ranks": [],
                "per_engine_checksum":
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            }),
        ),
    ] {
        let reply = json!({
            "rid": "x", "http_worker_ipc": null,
            "success": reply.0, "message": reply.1, "payload": reply.2,
        });
        let (header, response) = round_trip(&app, method, "/weights_checker", false, body, reply);
        assert_eq!(header[0], "CheckWeightsReqInput");
        assert_eq!(header[3], request.0, "{body}");
        assert_eq!(header[4], request.1, "{body}");
        assert_eq!(header[5][0].as_str(), request.2, "{body}");
        assert_eq!(header[6], request.3, "{body}");
        assert_eq!(header.len(), 8);
        assert_eq!(header[7][0].as_str(), request.4, "{body}");
        assert_eq!(response.status, status, "{body}");
        // Field order too: Python's response is an ordered dict.
        assert_eq!(response.body, response_body.to_string(), "{body}");
    }

    for body in [r#"{"selector": "both"}"#, r#"{"action": null}"#, "{"] {
        let response = send(app.1, "POST", "/weights_checker", false, body);
        assert_eq!(response.status, 400, "{body}");
    }
    assert_nothing_submitted(&app.0, "a malformed body");
}

/// `/start_profile` relays one `ProfileReq` carrying the body's fields
/// (Python's defaults for a missing or `null` body, a numeric-string
/// `start_step` as Python's validation coerces it) and answers with
/// Python's plain-text 200; a refused start is Python's bare 500. A
/// malformed body is refused before anything reaches the scheduler.
#[test]
fn start_profile_relays_the_body_and_the_scheduler_outcome() {
    let app = server();
    // The stack/shape defaults come from the process env; expect what this
    // env resolves to instead of assuming the knobs are unset.
    let env_with_stack = bool_env("SGLANG_PROFILE_WITH_STACK", true);
    let env_record_shapes = bool_env("SGLANG_PROFILE_RECORD_SHAPES", true);
    let (header, response) = round_trip(
        &app,
        "POST",
        "/start_profile",
        false,
        r#"{"output_dir": "/traces", "start_step": "15", "num_steps": 5,
            "activities": ["CPU", "GPU"], "profile_by_stage": true,
            "with_stack": false, "profile_id": "run-1", "merge_profiles": true,
            "profile_prefix": "pre", "profile_stages": ["decode"],
            "detailed_annotations": true, "req_type": 2, "rid": "ignored",
            "unknown": 1}"#,
        outcome(true, "Succeeded"),
    );
    let expected = vec![
        json!("ProfileReq"),
        header[1].clone(), // the minted rid
        Value::Null,
        json!(1), // START_PROFILE even though the body said 2
        json!("/traces"),
        json!(15),
        json!(5),
        json!(["CPU", "GPU"]),
        json!(true),
        json!(false),
        json!(env_record_shapes), // absent from the body
        json!("run-1"),
        json!(true),
        json!("pre"),
        json!(["decode"]),
        json!(true),
    ];
    assert_eq!(header, expected);
    assert_ne!(header[1], "ignored");
    assert_eq!(
        (
            response.status,
            response.header("content-type"),
            response.body.as_str()
        ),
        (200, None, "Start profiling.\n")
    );

    for (method, body) in [("GET", ""), ("POST", "null"), ("POST", "{}")] {
        let (header, response) = round_trip(
            &app,
            method,
            "/start_profile",
            false,
            body,
            outcome(true, "Succeeded"),
        );
        assert_eq!(header.len(), 16, "{body:?}");
        assert_eq!(header[3], 1, "{body:?}");
        for i in [4, 5, 6, 7, 13, 14] {
            assert!(header[i].is_null(), "{body:?}: idx {i}");
        }
        for i in [8, 12, 15] {
            assert_eq!(header[i], false, "{body:?}: idx {i}");
        }
        assert_eq!(header[9], env_with_stack, "{body:?}: with_stack");
        assert_eq!(header[10], env_record_shapes, "{body:?}: record_shapes");
        assert!(
            header[11].as_str().unwrap().parse::<f64>().is_ok(),
            "{body:?}: profile_id is the start time"
        );
        assert_eq!(response.status, 200, "{body:?}");
    }

    let (_, response) = round_trip(
        &app,
        "POST",
        "/start_profile",
        false,
        "",
        outcome(
            false,
            "Profiling is already in progress. Call /stop_profile first.",
        ),
    );
    assert_eq!(
        (
            response.status,
            response.header("content-type"),
            response.body.as_str()
        ),
        (
            500,
            Some("text/plain; charset=utf-8"),
            "Internal Server Error"
        )
    );

    for body in [
        r#"{"num_steps": 5.5}"#,
        r#"{"num_steps": "five"}"#,
        r#"{"profile_by_stage": null}"#,
        r#"{"with_stack": "yes"}"#,
        r#"{"activities": "CPU"}"#,
        r#"{"output_dir": 5}"#,
        r#"{"req_type": 7}"#,
        "{",
    ] {
        let response = send(app.1, "POST", "/start_profile", false, body);
        assert_eq!(response.status, 400, "{body}");
    }
    assert_nothing_submitted(&app.0, "a malformed body");
}

/// `/stop_profile` ignores its body, relays a STOP `ProfileReq` with every
/// other field at its default, and answers with Python's plain-text 200;
/// a refused stop (nothing running) is Python's bare 500.
#[test]
fn stop_profile_relays_the_scheduler_outcome() {
    let app = server();
    for (method, body) in [("POST", "{}"), ("GET", ""), ("POST", r#"{"num_steps": 5}"#)] {
        let (header, response) = round_trip(
            &app,
            method,
            "/stop_profile",
            false,
            body,
            outcome(true, "Succeeded."),
        );
        assert_eq!(header.len(), 16, "{body:?}");
        assert_eq!(header[0], "ProfileReq");
        assert_eq!(header[3], 2, "{body:?}: STOP_PROFILE");
        for i in [4, 5, 6, 7, 9, 10, 11, 13, 14] {
            assert!(header[i].is_null(), "{body:?}: idx {i}");
        }
        assert_eq!(
            (
                response.status,
                response.header("content-type"),
                response.body.as_str()
            ),
            (200, None, "Stop profiling. This will take some time.\n")
        );
    }

    let (_, response) = round_trip(
        &app,
        "POST",
        "/stop_profile",
        false,
        "",
        outcome(
            false,
            "Profiling is not in progress. Call /start_profile first.",
        ),
    );
    assert_eq!(
        (
            response.status,
            response.header("content-type"),
            response.body.as_str()
        ),
        (
            500,
            Some("text/plain; charset=utf-8"),
            "Internal Server Error"
        )
    );
}

/// `/set_internal_state` relays the object to the scheduler and answers
/// Python's bare list of flags with 200; a non-object body is a 400 that
/// reaches no scheduler.
#[test]
fn set_internal_state_answers_the_per_rank_flags() {
    let app = server();
    let (header, response) = round_trip(
        &app,
        "PUT",
        "/set_internal_state",
        true,
        r#"{"server_args": {"pp_max_micro_batch_size": 4}}"#,
        json!({"updated": true}),
    );
    assert_eq!(header[0], "SetInternalStateReq");
    assert_eq!(response.status, 200);
    assert_eq!(response.json(), json!([true]));

    let response = send(
        app.1,
        "POST",
        "/set_internal_state",
        true,
        r#"{"server_args": [1, 2]}"#,
    );
    assert_eq!(response.status, 400);
    assert_nothing_submitted(&app.0, "a non-object body");
}

/// `/start_profile` takes an absent, empty, or null body as all defaults
/// and answers Python's text line; `/stop_profile` too; a failing rank is
/// a 500 carrying the scheduler's reason.
#[test]
fn profile_routes_answer_python_text_lines() {
    let app = server();
    for (uri, method, body, reply, status, expected) in [
        (
            "/start_profile",
            "GET",
            "",
            (true, ""),
            200,
            "Start profiling.\n",
        ),
        (
            "/start_profile",
            "POST",
            "null",
            (true, ""),
            200,
            "Start profiling.\n",
        ),
        (
            "/start_profile",
            "POST",
            r#"{"num_steps": 3, "activities": ["CPU"]}"#,
            (true, ""),
            200,
            "Start profiling.\n",
        ),
        (
            "/stop_profile",
            "POST",
            "",
            (true, ""),
            200,
            "Stop profiling. This will take some time.\n",
        ),
        (
            "/stop_profile",
            "GET",
            "",
            (false, "Profiler is not running"),
            500,
            "Internal Server Error",
        ),
    ] {
        let reply = json!({"success": reply.0, "message": reply.1});
        let (_, response) = round_trip(&app, method, uri, !body.is_empty(), body, reply);
        assert_eq!(response.status, status, "{method} {uri} {body:?}");
        assert!(
            response.body.contains(expected),
            "{method} {uri}: {}",
            response.body
        );
    }

    let response = send(
        app.1,
        "POST",
        "/start_profile",
        true,
        r#"{"num_steps": "many"}"#,
    );
    assert_eq!(response.status, 400);
    assert_nothing_submitted(&app.0, "a malformed body");
}

/// The acknowledged routes answer Python's shapes: a text line for the
/// expert-distribution recorder, a JSON `null` for the others.
#[test]
fn acknowledged_routes_answer_python_shapes() {
    let app = server();
    for (uri, body, expected) in [
        (
            "/start_expert_distribution_record",
            "",
            "Start recording the expert distribution.\n",
        ),
        (
            "/stop_expert_distribution_record",
            "",
            "Stop recording the expert distribution.\n",
        ),
        (
            "/dump_expert_distribution_record",
            "",
            "Dump expert distribution record.\n",
        ),
        ("/slow_down", r#"{"forward_sleep_time": 0.5}"#, "null"),
        (
            "/release_memory_occupation",
            r#"{"tags": ["weights"]}"#,
            "null",
        ),
        ("/resume_memory_occupation", "{}", "null"),
    ] {
        let bare = json!({"rid": "x", "http_worker_ipc": null});
        let (_, response) = round_trip(&app, "POST", uri, !body.is_empty(), body, bare);
        assert_eq!(response.status, 200, "{uri}");
        assert_eq!(response.body, expected, "{uri}");
    }
}

/// The pause family answers Python's bodies: `{"message", "status"}` for
/// pause and continue, a text line for freeze_gc, and the weight-version
/// envelope with the version echoed.
#[test]
fn pause_family_routes_answer_python_shapes() {
    let app = server();
    for (method, uri, body, expected) in [
        (
            "POST",
            "/pause_generation",
            r#"{"mode": "in_place"}"#,
            r#"{"message":"Generation paused successfully.","status":"ok"}"#,
        ),
        (
            "POST",
            "/continue_generation",
            "{}",
            r#"{"message":"Generation continued successfully.","status":"ok"}"#,
        ),
        ("GET", "/freeze_gc", "", "Garbage collection frozen.\n"),
        (
            "POST",
            "/update_weight_version",
            r#"{"new_version": "v2"}"#,
            r#"{"success":true,"message":"Weight version updated to v2","new_version":"v2"}"#,
        ),
    ] {
        let (_, response) = round_trip(
            &app,
            method,
            uri,
            !body.is_empty(),
            body,
            json!({"success": true}),
        );
        assert_eq!(response.status, 200, "{uri}");
        assert_eq!(response.body, expected, "{uri}");
    }
}

/// `/get_hicache_l3_cache_layout` answers the merged layout as the bare
/// object; `/configure_logging` is Python's empty 200 and refuses an
/// unknown level before any rank.
#[test]
fn l3_layout_and_configure_logging_routes_answer_python_shapes() {
    let app = server();
    let layout = json!({"enabled": false, "hicache_object_layout": {"pools": []}, "reason": "off"});
    let (_, response) = round_trip(
        &app,
        "GET",
        "/get_hicache_l3_cache_layout",
        false,
        "",
        json!({"layout": layout}),
    );
    assert_eq!(response.status, 200);
    assert_eq!(response.json(), layout);

    let response = send(
        app.1,
        "POST",
        "/configure_logging",
        true,
        r#"{"log_level": "loud"}"#,
    );
    assert_eq!(response.status, 400);
    assert_nothing_submitted(&app.0, "an unknown level");
    let response = send(
        app.1,
        "POST",
        "/configure_logging",
        true,
        r#"{"dump_requests_threshold": 3}"#,
    );
    assert_eq!(response.status, 200);
    assert!(response.body.is_empty());
    assert_nothing_submitted(&app.0, "a change with no level");
}

/// Without `--elastic-ep-backend` both elastic routes are Python's 404,
/// after the body checks `/scale_elastic_ep` runs first; with one, the
/// state is idle and a scale is refused.
#[test]
fn elastic_ep_routes_gate_on_the_backend() {
    let (rt, addr) = server();
    let scale = |addr, body: &str| {
        let response = send(addr, "POST", "/scale_elastic_ep", true, body);
        (response.status, response.json())
    };
    for (body, message) in [
        ("[1]", "request body must be a JSON object"),
        (
            r#"{"new_ep_size": 0}"#,
            "new_ep_size must be a positive integer",
        ),
        (
            r#"{"new_ep_size": true}"#,
            "new_ep_size must be a positive integer",
        ),
        (
            r#"{"new_ep_size": "4"}"#,
            "new_ep_size must be a positive integer",
        ),
    ] {
        let (status, json) = scale(addr, body);
        assert_eq!(status, 400, "{body}");
        assert_eq!(json["error"], message, "{body}");
    }
    let (status, json) = scale(addr, "{not json");
    assert_eq!(status, 400);
    assert!(
        json["error"]
            .as_str()
            .unwrap()
            .starts_with("Invalid JSON: ")
    );
    let (status, json) = scale(addr, r#"{"new_ep_size": 8}"#);
    assert_eq!(status, 404);
    assert_eq!(
        json["error"],
        "elastic EP is not enabled (set --elastic-ep-backend)"
    );
    let response = send(addr, "GET", "/is_scaling_elastic_ep", false, "");
    assert_eq!(response.status, 404);
    assert_nothing_submitted(&rt, "an elastic EP route");

    let [addr] = free_loopback_addrs();
    let _rt = SchedulerlessServer::builder(addr)
        .server_args(|args| {
            args.api_keys = ApiKeys::new(None, Some("sk-admin".into()));
            args.deployment_info.elastic_ep_backend = Some("mooncake".into());
            args.deployment_info.dp_size = 2;
        })
        .start()
        .expect("start runtime");
    let (status, json) = scale(addr, r#"{"new_ep_size": 8}"#);
    assert_eq!(status, 400);
    assert_eq!(
        json["error"],
        "elastic EP scaling is not served by the Rust frontend"
    );
    let response = send(addr, "GET", "/is_scaling_elastic_ep", false, "");
    assert_eq!(response.status, 200);
    assert_eq!(
        response.json(),
        json!({
            "is_scaling_elastic_ep": false, "effective_ep_size": 2,
            "pending_ep_size": null, "scale_phase": "idle", "last_error": null
        })
    );
}

/// `/dumper/{method}` is mounted only under `DUMPER_SERVER_PORT=reuse`;
/// it takes an object body (empty when absent), flattens the ranks'
/// responses, and answers the failures as Python's `{"error": [...]}`.
#[test]
fn dumper_route_is_env_gated_and_flattens_rank_responses() {
    // The env is read when a server builds its router; the tests share one
    // process, so the gate is only ever turned on here.
    if std::env::var("DUMPER_SERVER_PORT").as_deref() != Ok("reuse") {
        let (_rt, addr) = server();
        let response = send(addr, "POST", "/dumper/get_state", false, "");
        assert_eq!(response.status, 404, "not mounted by default");
        unsafe { std::env::set_var("DUMPER_SERVER_PORT", "reuse") };
    }
    let app = server();
    let response = send(app.1, "POST", "/dumper/get_state", false, "[1]");
    assert_eq!(response.status, 400);
    assert_eq!(
        response.json()["error"],
        "Request body must be a JSON object."
    );
    assert_nothing_submitted(&app.0, "a non-object body");

    let (header, response) = round_trip(
        &app,
        "POST",
        "/dumper/get_state",
        false,
        "",
        json!({"success": true, "response": [{"step": 3}], "error": ""}),
    );
    assert_eq!(header[3], "get_state");
    assert_eq!(header[4], json!({}), "an absent body is an empty object");
    assert_eq!(response.status, 200);
    assert_eq!(response.json(), json!([{"step": 3}]));

    let (_, response) = round_trip(
        &app,
        "POST",
        "/dumper/get_state",
        false,
        r#"{"k": 1}"#,
        json!({"success": false, "response": [], "error": "boom"}),
    );
    assert_eq!(response.status, 400);
    assert_eq!(response.json(), json!({"error": ["boom"]}));
}

/// `/open_session` answers the id as a bare JSON string, a refused open
/// as Python's `{"error": {"message"}}` 400, and `/close_session` an
/// empty 200; both take GET as Python does.
#[test]
fn session_routes_answer_python_shapes() {
    let app = server();
    let open = r#"{"capacity_of_str_len": 100, "session_id": "s"}"#;
    let reply = |success: bool| json!({"session_id": "s", "success": success});
    let (_, response) = round_trip(&app, "GET", "/open_session", true, open, reply(true));
    assert_eq!(response.status, 200);
    assert_eq!(response.header("content-type"), Some("application/json"));
    assert_eq!(response.json(), json!("s"));

    let (_, response) = round_trip(&app, "POST", "/open_session", true, open, reply(false));
    assert_eq!(response.status, 400);
    assert_eq!(
        response.json(),
        json!({"error": {"message": "Failed to open the session. Check if a session with the same id is still open."}})
    );

    let response = send(
        app.1,
        "POST",
        "/open_session",
        true,
        r#"{"capacity_of_str_len": 100, "streaming": true}"#,
    );
    assert_eq!(response.status, 400);
    assert!(
        response.json()["error"]["message"]
            .as_str()
            .unwrap()
            .contains("--enable-streaming-session")
    );
    assert_nothing_submitted(&app.0, "a streaming session without the flag");

    let (_, response) = round_trip(
        &app,
        "POST",
        "/close_session",
        true,
        r#"{"session_id": "s"}"#,
        json!({"rid": "x"}),
    );
    assert_eq!(response.status, 200);
    assert!(response.body.is_empty());
}

/// A server launched with `change` to its arguments and no API keys.
fn open_server(
    change: impl FnOnce(&mut ServerArgs),
    state_keys: &[&str],
) -> (SchedulerlessServer, SocketAddr) {
    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .server_args(change)
        .server_info_state_keys(state_keys.iter().map(|key| (*key).to_owned()).collect())
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    (rt, addr)
}

fn json_object(value: Value) -> Arc<serde_json::Map<String, Value>> {
    Arc::new(value.as_object().expect("an object").clone())
}

#[test]
fn model_info_routes_render_the_typed_frontend_contract() {
    let (_rt, addr) = open_server(
        |args| {
            args.model_path = "/model".into();
            args.served_model_name = "served".into();
            // The server loads a local copy; clients get the launched path.
            args.tokenizer_path = "/local-tokenizer-copy".into();
            args.public_tokenizer_path = "/tokenizer".into();
            args.weight_version = Some("weights-v1".into());
            args.load_format = Some("safetensors".into());
            args.reasoning_parser = Some("reasoner".into());
            args.tool_call_parser = Some("tools".into());
            args.disaggregation_mode = DisaggregationMode::Prefill;
            args.launch_model_info.0 = json_object(json!({
                "has_image_understanding": true,
                "architectures": ["Arch"],
            }));
        },
        &[],
    );
    for path in ["/get_model_info", "/model_info"] {
        let response = http_bytes(addr, "GET", path, &[], b"");
        assert_eq!(response.status, 200, "{path}");
        assert_eq!(response.header("content-type"), Some("application/json"));
        let body = response.json();
        assert_eq!(body["model_path"], "/model");
        assert_eq!(body["served_model_name"], "served");
        assert_eq!(body["tokenizer_path"], "/tokenizer");
        assert_eq!(body["is_generation"], true);
        assert_eq!(body["weight_version"], "weights-v1");
        assert_eq!(body["load_format"], "safetensors");
        assert_eq!(body["reasoning_parser"], "reasoner");
        assert_eq!(body["tool_call_parser"], "tools");
        assert_eq!(body["disaggregation_mode"], "prefill");
        assert!(body["preferred_sampling_params"].is_null());
        // Python's launch-time model fields follow the typed ones.
        let keys: Vec<_> = body.as_object().unwrap().keys().collect();
        assert_eq!(
            keys[keys.len() - 2..],
            ["has_image_understanding", "architectures"]
        );
        assert_eq!(body["architectures"], json!(["Arch"]));
    }
}

/// `/server_info` serializes only the typed public subset of the
/// scheduler's internal state. Native floats stay JSON numbers; NumPy
/// scalars Python encodes as strings stay strings.
#[test]
fn server_info_route_serializes_only_the_typed_public_subset() {
    for (kvcache, expected_kvcache) in [
        (json!(23.616), json!(23.616)),
        (json!("23.616"), json!("23.616")),
    ] {
        // An embedding package registers its own scheduler fields by name.
        let app = open_server(
            |args| {
                args.model_path = "/model".into();
                args.served_model_name = "served".into();
                args.tokenizer_path = "/local-tokenizer-copy".into();
                args.public_tokenizer_path = "/tokenizer".into();
                args.model_config.context_len = 4096;
                args.max_total_num_tokens = 8192;
                args.version = "1.2.3".into();
                args.launch_server_info.0 = json_object(json!({
                    "model_path": "/model",
                    "tp_size": 4,
                    "max_total_num_tokens": 1,
                    "launch_command": "launch --tp-size 4",
                    "instance_id": "stale-launch-instance-id",
                }));
            },
            &["package_state", "unset_package_state"],
        );
        // Integer keys, as the scheduler's msgpack carries them.
        let mut payload = Vec::new();
        let int_keyed = |key: i64, value: rmpv::Value| rmpv::Value::Map(vec![(key.into(), value)]);
        let named = |entries: Vec<(&str, rmpv::Value)>| {
            rmpv::Value::Map(
                entries
                    .into_iter()
                    .map(|(key, value)| (key.into(), value))
                    .collect(),
            )
        };
        let kvcache = match &kvcache {
            Value::String(text) => rmpv::Value::from(text.as_str()),
            other => rmpv::Value::from(other.as_f64().unwrap()),
        };
        let internal_state = named(vec![
            (
                "rust_mm_transport",
                named(vec![("inline_features", 3.into())]),
            ),
            ("api_key", "must-not-leak".into()),
            (
                "package_state",
                named(vec![("tokens_per_owner", int_keyed(0, 7.into()))]),
            ),
            ("last_gen_throughput", 1.5.into()),
            (
                "rust_server_startup_time",
                named(vec![
                    ("load_weight", 1.5.into()),
                    ("tokenizer_e2e", 9.0.into()),
                ]),
            ),
            (
                "memory_usage",
                named(vec![
                    ("kvcache", kvcache),
                    ("token_capacity", 8192.into()),
                    ("token_capacity_swa", rmpv::Value::Nil),
                    (
                        "token_capacity_window",
                        named(vec![("window_4096", 2048.into())]),
                    ),
                    ("future_uncontracted_metric", 99.into()),
                ]),
            ),
            (
                "step_time_dict",
                int_keyed(4, rmpv::Value::Array(vec![0.01.into()])),
            ),
        ]);
        rmpv::encode::write_value(
            &mut payload,
            &named(vec![("internal_state", internal_state)]),
        )
        .unwrap();

        let addr = app.1;
        let client = std::thread::spawn(move || http_bytes(addr, "GET", "/server_info", &[], b""));
        let request = next_scheduler_work(&app.0);
        app.0
            .push_control_payload(&request.rid().expect("rid"), &payload)
            .unwrap();
        let response = client.join().unwrap();
        assert_eq!(response.status, 200, "{}", response.body);
        assert_eq!(response.header("content-type"), Some("application/json"));
        let body = response.json();
        // Python's launch-time fields come first, in order; a runtime field
        // replaces a launch-time one in place, and the rest follow.
        let keys: Vec<&str> = body
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys[..6],
            [
                "model_path",
                "tp_size",
                "max_total_num_tokens",
                "launch_command",
                "instance_id",
                "served_model_name"
            ]
        );
        assert_eq!(body["tp_size"], 4);
        assert_eq!(body["launch_command"], "launch --tp-size 4");
        // The embedded server's startup timings move to the top level.
        assert_eq!(
            body["startup_time"],
            json!({"load_weight": 1.5, "tokenizer_e2e": 9.0})
        );
        let state = &body["internal_states"][0];
        assert!(state.get("rust_server_startup_time").is_none());
        assert_eq!(body["model_path"], "/model");
        assert_eq!(body["served_model_name"], "served");
        assert_eq!(body["tokenizer_path"], "/tokenizer");
        assert_eq!(body["max_context_length"], 4096);
        assert_eq!(body["max_total_num_tokens"], 8192);
        assert_eq!(body["version"], "1.2.3");
        let instance_id = body["instance_id"]
            .as_str()
            .expect("HTTP /server_info must carry typed instance_id");
        assert_ne!(instance_id, "stale-launch-instance-id");
        assert_eq!(
            uuid::Uuid::parse_str(instance_id).unwrap().get_version(),
            Some(uuid::Version::Random)
        );
        assert_eq!(body["frontend"], "rust");
        assert_eq!(state["rust_mm_transport"]["inline_features"], 3);
        assert_eq!(state["last_gen_throughput"], 1.5);
        // A registered field passes through after the typed ones, its integer
        // keys stringified as Python's are; one the scheduler did not set
        // stays absent.
        assert_eq!(
            state["package_state"],
            json!({"tokens_per_owner": {"0": 7}})
        );
        assert_eq!(
            state.as_object().unwrap().keys().next_back().unwrap(),
            "package_state"
        );
        assert!(state.get("unset_package_state").is_none());
        assert_eq!(state["memory_usage"]["kvcache"], expected_kvcache);
        assert_eq!(state["memory_usage"]["token_capacity"], 8192);
        assert_eq!(state["memory_usage"]["token_capacity_swa"], Value::Null);
        assert_eq!(
            state["memory_usage"]["token_capacity_window"],
            json!({"window_4096": 2048})
        );
        assert_eq!(state["step_time_dict"]["4"], json!([0.01]));
        assert!(!body.to_string().contains("must-not-leak"));
        assert!(
            state["memory_usage"]
                .get("future_uncontracted_metric")
                .is_none(),
            "uncontracted scheduler fields must not silently become public API"
        );
        // Without data parallelism there are no other listeners to name.
        assert!(body.get("dp_listeners").is_none());
        assert_nothing_submitted(&app.0, "an abort of a served read");
    }
}

/// `/abort_request` cancels the live generation submitted under the body's
/// rid, or every one with `abort_all`, and answers Python's empty 200: the
/// scheduler is told to abort the request, and the client's call ends on
/// the scheduler's answer. A rid nothing runs under is still a 200 with nothing
/// aborted, an empty rid without `abort_all` is ignored, and a malformed
/// body is a 400.
#[test]
fn abort_request_cancels_the_named_generations() {
    let (rt, addr) = open_server(|_| {}, &[]);
    for (body, status, aborted) in [
        (r#"{"rid": "client-rid"}"#, 200, true),
        (r#"{"abort_all": true}"#, 200, true),
        (r#"{"rid": "other-rid"}"#, 200, false),
        (r#"{"rid": ""}"#, 200, false),
        ("{}", 200, false),
        (r#"{"rid": 3}"#, 400, false),
    ] {
        let generate = json!({"rid": "client-rid", "input_ids": [1], "stream": true});
        let mut events = Sse::post(addr, "/generate", &generate);
        let rid = next_scheduler_request(&rt).rid().expect("rid");

        let response = http_bytes(addr, "POST", "/abort_request", &[JSON], body.as_bytes());
        assert_eq!(response.status, status, "{body}");
        if status == 200 {
            assert!(response.body.is_empty(), "{body}");
        }
        let end = |details: OutputDetails| rt.push_output_with(&rid, &[], details).unwrap();
        if aborted {
            let abort = next_scheduler_request(&rt);
            assert_eq!(abort.kind().as_deref(), Some("AbortReq"), "{body}");
            assert_eq!(abort.rid().as_deref(), Some(rid.as_str()), "{body}");
            // The call waits for the scheduler's answer, then ends on it.
            end(OutputDetails {
                abort: true,
                finish_reason: Some(json!({"type": "abort", "message": ""})),
                ..Default::default()
            });
            let last = events.next_json();
            assert_eq!(
                last["meta_info"]["finish_reason"]["type"], "abort",
                "{body}"
            );
        } else {
            assert_nothing_submitted(&rt, "an abort of nothing");
            end(OutputDetails {
                finish_reason: Some(json!({"type": "stop"})),
                ..Default::default()
            });
        }
        while events.next().is_some() {}
    }
}

/// A message no route fans out (a generate, an unknown tag) is refused
/// before it reaches the scheduler.
#[test]
fn peer_route_refuses_a_message_no_listener_relays() {
    let (rt, addr) = open_server(|_| {}, &[]);
    for tag in ["TokenizedGenerateReqInput", "NotAControlMessage"] {
        let body = rmp_serde::to_vec(&(tag, "sender-rid", ())).unwrap();
        let response = http_bytes(addr, "POST", "/internal/peer", &[], &body);
        assert_eq!(response.status, 400, "{tag}");
    }
    assert_nothing_submitted(&rt, "a message no listener relays");
}

/// A forwarded control request reaches the local scheduler under the rid
/// its sender encoded, and the scheduler's bytes come back as the body.
#[test]
fn forward_control_relays_the_sender_rid_and_the_reply_bytes() {
    let (rt, addr) = open_server(|_| {}, &[]);
    let header = rmp_serde::to_vec(&("FlushCacheReqInput", "sender-rid", (), 2.0)).unwrap();
    let sent = header.clone();
    let client = std::thread::spawn(move || http_bytes(addr, "POST", "/internal/peer", &[], &sent));
    let request = next_scheduler_work(&rt);
    assert_eq!(request.rid().as_deref(), Some("sender-rid"));
    assert_eq!(request.header, header, "relayed unchanged");
    rt.push_control_payload("sender-rid", b"scheduler-reply")
        .unwrap();
    let response = client.join().unwrap();
    assert_eq!(response.status, 200);
    assert_eq!(response.body, "scheduler-reply");

    // A body without a rid cannot be routed back: rejected before intake.
    let no_rid = rmp_serde::to_vec(&("FlushCacheReqInput", (), ())).unwrap();
    let response = http_bytes(addr, "POST", "/internal/peer", &[], &no_rid);
    assert_eq!(response.status, 400);
    assert_nothing_submitted(&rt, "a body without a rid");
}

/// Without an admin key, the peer route refuses the operations of the
/// forced admin routes, as those routes do, before they reach the
/// scheduler; with one, they go through.
#[test]
fn peer_route_refuses_forced_admin_operations_without_an_admin_key() {
    let nil = rmpv::Value::Nil;
    let message = |fields: Vec<rmpv::Value>| {
        let mut body = Vec::new();
        rmpv::encode::write_value(&mut body, &rmpv::Value::Array(fields)).unwrap();
        body
    };
    let mut stop_profile: Vec<rmpv::Value> = vec![
        "ProfileReq".into(),
        "sender-rid".into(),
        nil.clone(),
        2.into(),
    ];
    stop_profile.extend([
        nil.clone(),
        nil.clone(),
        nil.clone(),
        nil.clone(),
        false.into(),
        nil.clone(),
        nil.clone(),
        nil.clone(),
        false.into(),
        nil.clone(),
        nil.clone(),
        false.into(),
    ]);
    let detach = message(vec![
        "DetachHiCacheStorageReqInput".into(),
        "sender-rid".into(),
        nil.clone(),
    ]);
    let forced = [
        message(vec![
            "AttachHiCacheStorageReqInput".into(),
            "sender-rid".into(),
            nil.clone(),
            "file".into(),
            nil.clone(),
            nil.clone(),
            nil.clone(),
        ]),
        detach.clone(),
        message(vec![
            "CheckWeightsReqInput".into(),
            "sender-rid".into(),
            nil.clone(),
            "checksum".into(),
            false.into(),
            nil.clone(),
            "target".into(),
            nil.clone(),
        ]),
        message(stop_profile),
    ];

    let (rt, addr) = open_server(|_| {}, &[]);
    for body in &forced {
        let response = http_bytes(addr, "POST", "/internal/peer", &[], body);
        assert_eq!(response.status, 403, "{}", response.body);
        assert!(
            response.body.contains("requires admin API key"),
            "{}",
            response.body
        );
    }
    assert_nothing_submitted(&rt, "a forced admin operation");

    // With an admin key, the relayed detach reaches the scheduler.
    let (rt, addr) = server();
    let client =
        std::thread::spawn(move || http_bytes(addr, "POST", "/internal/peer", &[ADMIN], &detach));
    let request = next_scheduler_work(&rt);
    assert_eq!(
        request.kind().as_deref(),
        Some("DetachHiCacheStorageReqInput")
    );
    rt.push_control_result("sender-rid", &json!({"success": true}))
        .unwrap();
    client.join().unwrap();
}
