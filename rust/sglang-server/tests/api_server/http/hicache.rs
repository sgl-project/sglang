//! The HiCache storage routes over HTTP: what each sends the scheduler, how
//! its reply is answered, and the configuration `GET` reports afterwards.

use std::net::SocketAddr;

use serde_json::{Value, json};
use sglang_server::ServerArgs;
use sglang_server_test::{ApiKeys, SchedulerlessServer, free_loopback_addrs};

use crate::common::{HttpResponse, header_json, http, next_scheduler_work, take_startup_freeze_gc};

const ADMIN: &str = "Authorization: Bearer sk-admin";
const STORAGE: &str = "/hicache/storage-backend";
const CLEAR: &str = "/hicache/storage-backend/clear";
/// Python's refusal of an admin operation on a server launched without an
/// admin key.
const ADMIN_KEY_MISSING: &str = "This endpoint requires admin API key, but this \
    server was started without one (admin-api-key). Restart with --admin-api-key to enable.";

/// What the test answers a control request with.
enum Reply {
    Result(Value),
    /// Bytes no scheduler would send as a result.
    Raw(&'static [u8]),
    Error(&'static str),
}

/// A server with an admin key, after `change` to its launch arguments.
fn server_with(change: impl FnOnce(&mut ServerArgs)) -> (SchedulerlessServer, SocketAddr) {
    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .server_args(|args| {
            args.api_keys = ApiKeys::new(None, Some("sk-admin".into()));
            change(args);
        })
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    (rt, addr)
}

fn server() -> (SchedulerlessServer, SocketAddr) {
    server_with(|_| {})
}

/// Send the request as the admin, check the control message it puts on the
/// ring (`tag`, then `fields`), answer it with `reply`, and return the
/// response.
fn exchange(
    (rt, addr): &(SchedulerlessServer, SocketAddr),
    method: &'static str,
    path: &'static str,
    body: Value,
    tag: &str,
    fields: Value,
    reply: Reply,
) -> HttpResponse {
    let addr = *addr;
    let body = if body.is_null() {
        String::new()
    } else {
        body.to_string()
    };
    let client = std::thread::spawn(move || http(addr, method, path, &[ADMIN], &body));
    // A failed reply is followed by the frontend's abort of that request.
    let request = next_scheduler_work(rt);
    let rid = request.rid().expect("control rid");
    let mut expected = vec![json!(tag), json!(rid), Value::Null];
    expected.extend(fields.as_array().unwrap().iter().cloned());
    assert_eq!(header_json(&request), Value::Array(expected));
    match reply {
        Reply::Result(result) => rt.push_control_result(&rid, &result),
        Reply::Raw(payload) => rt.push_control_payload(&rid, payload),
        Reply::Error(message) => rt.push_error(&rid, message),
    }
    .unwrap();
    let response = client.join().unwrap();
    if matches!(response.status, 200 | 400) {
        assert!(
            response
                .head
                .to_ascii_lowercase()
                .contains("content-type: application/json"),
            "{}",
            response.head
        );
    }
    response
}

fn read_status(addr: SocketAddr) -> Value {
    let response = http(addr, "GET", STORAGE, &[ADMIN], "");
    assert_eq!(response.status, 200, "{}", response.body);
    assert!(
        response
            .head
            .to_ascii_lowercase()
            .contains("content-type: application/json"),
        "{}",
        response.head
    );
    response.json()
}

/// Python refuses the storage-backend status, attach and detach without
/// an admin key: 403 from the key check, before any handler.
#[test]
fn storage_backend_needs_a_configured_admin_key() {
    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    for method in ["GET", "PUT", "DELETE"] {
        let response = http(
            addr,
            method,
            STORAGE,
            &[],
            r#"{"hicache_storage_backend": "file"}"#,
        );
        assert_eq!(response.status, 403, "{method}");
        assert_eq!(response.json()["error"], ADMIN_KEY_MISSING, "{method}");
    }
    assert!(
        rt.next_request(std::time::Duration::from_millis(200))
            .is_none(),
        "refused before the scheduler"
    );
}

/// Clear maps scheduler success to HTTP 200 and failure to HTTP 400.
#[test]
fn clear_relays_success_and_failure() {
    let app = server();
    for (success, status) in [(true, 200), (false, 400)] {
        let response = exchange(
            &app,
            "POST",
            CLEAR,
            Value::Null,
            "ClearHiCacheReqInput",
            json!([]),
            Reply::Result(json!({"success": success})),
        );
        assert_eq!(response.status, status);
        assert_eq!(response.json(), json!({"success": success, "message": ""}));
    }
}

/// Detach returns the generated response, defaulting absent messages to empty.
#[test]
fn detach_preserves_message_and_handles_empty_or_missing_message() {
    let app = server();
    for payload in [
        json!({"success": true, "message": "Already detached."}),
        json!({"success": false, "message": "scheduler is not idle"}),
        json!({"success": true, "message": ""}),
        json!({"success": false}),
    ] {
        let response = exchange(
            &app,
            "DELETE",
            STORAGE,
            Value::Null,
            "DetachHiCacheStorageReqInput",
            json!([]),
            Reply::Result(payload.clone()),
        );
        let success = payload["success"].as_bool().unwrap();
        assert_eq!(response.status, if success { 200 } else { 400 });
        assert_eq!(
            response.json(),
            json!({
                "success": success,
                "message": payload.get("message").cloned().unwrap_or(json!("")),
            })
        );
    }
}

/// Malformed replies and scheduler errors produce HTTP 500 for all three operations.
#[test]
fn control_operations_report_scheduler_errors() {
    let app = server();
    for (method, path, tag, operation, input, fields) in [
        (
            "POST",
            CLEAR,
            "ClearHiCacheReqInput",
            "clear",
            Value::Null,
            json!([]),
        ),
        (
            "DELETE",
            STORAGE,
            "DetachHiCacheStorageReqInput",
            "detach",
            Value::Null,
            json!([]),
        ),
        (
            "PUT",
            STORAGE,
            "AttachHiCacheStorageReqInput",
            "attach",
            json!({"hicache_storage_backend": "file"}),
            json!(["file", null, null, null]),
        ),
    ] {
        let invalid = format!("invalid HiCache {operation} response");
        for (reply, expected_error) in [
            (
                Reply::Result(json!({"message": "missing success"})),
                invalid.as_str(),
            ),
            (Reply::Result(json!({"success": "false"})), invalid.as_str()),
            (Reply::Raw(&[0xc1]), invalid.as_str()),
            (Reply::Error("backend unavailable"), "backend unavailable"),
        ] {
            let response = exchange(
                &app,
                method,
                path,
                input.clone(),
                tag,
                fields.clone(),
                reply,
            );
            assert_eq!(response.status, 500, "{}", response.body);
            assert!(response.body.contains(expected_error), "{}", response.body);
        }
    }
}

/// Attach forwards all options and returns the scheduler result and message.
#[test]
fn attach_http_forwards_options_and_relays_result() {
    let app = server();
    let body = json!({
        "hicache_storage_backend": "file",
        "hicache_storage_backend_extra_config_json": "{}",
        "hicache_storage_prefetch_policy": "timeout",
        "hicache_write_policy": "write_through",
    });
    for (success, status) in [(true, 200), (false, 400)] {
        let response = exchange(
            &app,
            "PUT",
            STORAGE,
            body.clone(),
            "AttachHiCacheStorageReqInput",
            json!(["file", "{}", "timeout", "write_through"]),
            Reply::Result(json!({"success": success, "message": "backend detail"})),
        );
        assert_eq!(response.status, status);
        assert_eq!(
            response.json(),
            json!({"success": success, "message": "backend detail"})
        );
    }
}

/// Python accepts empty names and null options; backend validation belongs to the scheduler.
#[test]
fn attach_preserves_empty_strings_and_accepts_null_options() {
    let app = server();
    for (body, fields) in [
        (
            json!({"hicache_storage_backend": ""}),
            json!(["", null, null, null]),
        ),
        (
            json!({
                "hicache_storage_backend": "file",
                "hicache_storage_backend_extra_config_json": null,
                "hicache_storage_prefetch_policy": null,
                "hicache_write_policy": null,
            }),
            json!(["file", null, null, null]),
        ),
        (
            json!({
                "hicache_storage_backend": "file",
                "hicache_storage_backend_extra_config_json": "",
                "hicache_storage_prefetch_policy": "",
                "hicache_write_policy": "",
            }),
            json!(["file", "", "", ""]),
        ),
    ] {
        let response = exchange(
            &app,
            "PUT",
            STORAGE,
            body,
            "AttachHiCacheStorageReqInput",
            fields,
            Reply::Result(json!({"success": true})),
        );
        assert_eq!(response.status, 200);
    }
}

/// Empty bodies, invalid JSON, and missing backend fields fail before scheduler submission.
#[test]
fn attach_rejects_invalid_body_before_scheduler_submission() {
    let (rt, addr) = server();
    for (body, expected_error) in [
        ("", "EOF while parsing"),
        ("{", "EOF while parsing"),
        ("{}", "missing field `hicache_storage_backend`"),
        (
            r#"{"hicache_storage_backend":null}"#,
            "missing field `hicache_storage_backend`",
        ),
        (r#"{"hicache_storage_backend":1}"#, "invalid type"),
        (
            r#"{"hicache_storage_backend":"file","hicache_write_policy":{}}"#,
            "invalid type",
        ),
    ] {
        let response = http(addr, "PUT", STORAGE, &[ADMIN], body);
        assert_eq!(response.status, 422, "{body:?}");
        assert!(response.body.contains(expected_error), "{}", response.body);
    }
    assert!(
        rt.next_request(std::time::Duration::from_millis(200))
            .is_none(),
        "an invalid request reached the scheduler"
    );
}

/// GET returns a JSON object with the same four configuration fields as Python.
#[test]
fn get_status_returns_python_configuration_fields() {
    let (_rt, addr) = server();
    assert_eq!(
        read_status(addr),
        json!({
            "hicache_storage_backend": null,
            "hicache_storage_backend_extra_config": null,
            "hicache_storage_prefetch_policy": null,
            "hicache_write_policy": null,
        })
    );
}

/// Only successful attach/detach operations update GET. Omitted options and
/// clear preserve the current configuration.
#[test]
fn get_status_tracks_only_successful_control_operations() {
    let app = server_with(|args| {
        args.hicache_storage_backend = Some("file".into());
        args.hicache_storage_backend_extra_config = Some("{}".into());
        args.hicache_storage_prefetch_policy = Some("timeout".into());
        args.hicache_write_policy = Some("write_through".into());
    });
    let addr = app.1;
    let initial = json!({
        "hicache_storage_backend": "file",
        "hicache_storage_backend_extra_config": "{}",
        "hicache_storage_prefetch_policy": "timeout",
        "hicache_write_policy": "write_through",
    });
    assert_eq!(read_status(addr), initial);
    assert!(
        app.0
            .next_request(std::time::Duration::from_millis(200))
            .is_none(),
        "GET must not submit scheduler work"
    );

    let body = json!({
        "hicache_storage_backend": "file",
        "hicache_storage_backend_extra_config_json": "{\"path\":\"/cache\"}",
        "hicache_storage_prefetch_policy": "wait_complete",
        "hicache_write_policy": "write_back",
    });
    let fields = json!([
        "file",
        "{\"path\":\"/cache\"}",
        "wait_complete",
        "write_back"
    ]);
    for success in [false, true] {
        exchange(
            &app,
            "PUT",
            STORAGE,
            body.clone(),
            "AttachHiCacheStorageReqInput",
            fields.clone(),
            Reply::Result(json!({"success": success})),
        );
        if !success {
            assert_eq!(read_status(addr), initial);
        }
    }
    let attached = json!({
        "hicache_storage_backend": "file",
        "hicache_storage_backend_extra_config": "{\"path\":\"/cache\"}",
        "hicache_storage_prefetch_policy": "wait_complete",
        "hicache_write_policy": "write_back",
    });
    assert_eq!(read_status(addr), attached);
    // Omitted options retain their current values, like Python's config updates.
    exchange(
        &app,
        "PUT",
        STORAGE,
        json!({"hicache_storage_backend": "file"}),
        "AttachHiCacheStorageReqInput",
        json!(["file", null, null, null]),
        Reply::Result(json!({"success": true})),
    );
    assert_eq!(read_status(addr), attached);
    exchange(
        &app,
        "POST",
        CLEAR,
        Value::Null,
        "ClearHiCacheReqInput",
        json!([]),
        Reply::Result(json!({"success": true})),
    );
    assert_eq!(read_status(addr), attached);
    for success in [false, true] {
        exchange(
            &app,
            "DELETE",
            STORAGE,
            Value::Null,
            "DetachHiCacheStorageReqInput",
            json!([]),
            Reply::Result(json!({"success": success})),
        );
        let mut expected = attached.clone();
        if success {
            expected["hicache_storage_backend"] = Value::Null;
            expected["hicache_storage_backend_extra_config"] = Value::Null;
        }
        assert_eq!(read_status(addr), expected);
    }
}
