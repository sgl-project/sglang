use axum::{
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

use crate::core::{is_retryable_status, Worker};

const BOOTSTRAP_HOST_KEY: &str = "bootstrap_host";
const BOOTSTRAP_PORT_KEY: &str = "bootstrap_port";
const BOOTSTRAP_ROOM_KEY: &str = "bootstrap_room";
const BOOTSTRAP_ATTEMPT_ID_KEY: &str = "bootstrap_attempt_id";

#[derive(Clone, Copy)]
pub(super) enum PdRawError {
    RequestInvalid,
    Unsupported,
}

fn optional_pd_bool(body: &Value, field: &str) -> Result<Option<bool>, PdRawError> {
    match body.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(PdRawError::RequestInvalid),
    }
}

fn pd_i64(value: Option<&Value>, default: i64) -> Result<i64, PdRawError> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_i64()
            .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
            .ok_or(PdRawError::RequestInvalid),
    }
}

fn pd_f64(value: Option<&Value>, default: f64) -> Result<f64, PdRawError> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or(PdRawError::RequestInvalid),
    }
}

fn validate_pd_sampling_map(map: &serde_json::Map<String, Value>) -> Result<(), PdRawError> {
    for forbidden in [
        "regex",
        "json_schema",
        "ebnf",
        "grammar",
        "structural_tag",
        "custom_logit_processor",
    ] {
        if map.get(forbidden).is_some_and(|value| !value.is_null()) {
            return Err(PdRawError::Unsupported);
        }
    }
    if pd_f64(map.get("temperature"), 1.0)? != 0.0
        || pd_i64(map.get("top_k"), 1)? != 1
        || pd_f64(map.get("top_p"), 1.0)? != 1.0
        || pd_f64(map.get("min_p"), 0.0)? != 0.0
    {
        return Err(PdRawError::Unsupported);
    }
    if !(0..=256).contains(&pd_i64(map.get("max_new_tokens"), 128)?) {
        return Err(PdRawError::RequestInvalid);
    }
    Ok(())
}

pub(super) fn validate_pd_support(body: &Value) -> Result<(), PdRawError> {
    let object = body.as_object().ok_or(PdRawError::RequestInvalid)?;
    const ALLOWED_FIELDS: &[&str] = &[
        "rid",
        "text",
        "input_ids",
        "stream",
        "sampling_params",
        "return_logprob",
        "logprob_start_len",
        "top_logprobs_num",
        "token_ids_logprob",
        "return_hidden_states",
        "return_sampling_mask",
        "return_text_in_logprobs",
        "n",
        "lora_path",
        "return_routed_experts",
        "image_data",
        "bootstrap_host",
        "bootstrap_port",
        "bootstrap_room",
        "bootstrap_attempt_id",
    ];
    if object
        .keys()
        .any(|field| !ALLOWED_FIELDS.contains(&field.as_str()))
    {
        return Err(PdRawError::RequestInvalid);
    }

    for field in [
        "return_logprob",
        "return_hidden_states",
        "return_sampling_mask",
        "return_routed_experts",
    ] {
        if optional_pd_bool(body, field)?.unwrap_or(false) {
            return Err(PdRawError::Unsupported);
        }
    }
    if pd_i64(object.get("top_logprobs_num"), 0)? != 0
        || pd_i64(object.get("n"), 1)? != 1
        || object
            .get("token_ids_logprob")
            .is_some_and(|value| !value.is_null())
        || object
            .get("lora_path")
            .is_some_and(|value| !value.is_null())
        || object
            .get("image_data")
            .is_some_and(|value| !value.is_null())
    {
        return Err(PdRawError::Unsupported);
    }

    match object.get("sampling_params") {
        Some(Value::Object(map)) => validate_pd_sampling_map(map)?,
        Some(Value::Array(items)) => {
            for item in items {
                validate_pd_sampling_map(item.as_object().ok_or(PdRawError::RequestInvalid)?)?;
            }
        }
        Some(_) => return Err(PdRawError::RequestInvalid),
        None => return Err(PdRawError::Unsupported),
    }
    Ok(())
}

fn valid_token_id(value: &Value) -> bool {
    value.as_u64().is_some_and(|token| token <= u32::MAX as u64)
}

fn optional_bool(value: Option<&Value>, field: &str) -> Result<Option<bool>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(format!("{field} must be a boolean")),
    }
}

pub(super) fn generate_shape(body: &Value) -> Result<(Option<usize>, bool, bool), String> {
    let object = body
        .as_object()
        .ok_or_else(|| "Request must be a JSON object".to_string())?;
    let text = object.get("text").filter(|value| !value.is_null());
    let input_ids = object.get("input_ids").filter(|value| !value.is_null());
    if text.is_some() == input_ids.is_some() {
        return Err("Exactly one of text or input_ids is required".to_string());
    }

    let batch_size = if let Some(text) = text {
        match text {
            Value::String(_) => None,
            Value::Array(items)
                if (1..=8).contains(&items.len()) && items.iter().all(Value::is_string) =>
            {
                Some(items.len())
            }
            _ => return Err("text must be a string or a batch of 1..8 strings".to_string()),
        }
    } else {
        let items = input_ids
            .and_then(Value::as_array)
            .ok_or_else(|| "input_ids must be an array".to_string())?;
        if items.is_empty() {
            return Err("input_ids must not be empty".to_string());
        }
        if items.len() <= 4096 && items.iter().all(valid_token_id) {
            None
        } else if (1..=8).contains(&items.len())
            && items.iter().all(|item| {
                item.as_array().is_some_and(|tokens| {
                    (1..=4096).contains(&tokens.len()) && tokens.iter().all(valid_token_id)
                })
            })
        {
            Some(items.len())
        } else {
            return Err(
                "input_ids must be token ids or a batch of 1..8 token-id arrays".to_string(),
            );
        }
    };
    let is_stream = optional_bool(object.get("stream"), "stream")?.unwrap_or(false);
    let return_logprob =
        optional_bool(object.get("return_logprob"), "return_logprob")?.unwrap_or(false);
    Ok((batch_size, is_stream, return_logprob))
}

pub(super) fn request_text(body: &Value) -> Option<String> {
    match body.get("text") {
        Some(Value::String(text)) => Some(text.clone()),
        Some(Value::Array(items)) => items.first().and_then(Value::as_str).map(str::to_owned),
        _ => None,
    }
}

pub(super) fn response_retryable(response: &Response) -> bool {
    let Some(reason) = response.headers().get("x-sglang-pd-reason") else {
        return is_retryable_status(response.status());
    };
    if !matches!(
        reason.to_str().ok(),
        Some(
            "PD_REQUEST_INVALID"
                | "PD_UNSUPPORTED"
                | "PD_CAPACITY_EXHAUSTED"
                | "PD_PROTOCOL_MISMATCH"
                | "PD_PEER_UNAVAILABLE"
                | "PD_RENDEZVOUS_TIMEOUT"
                | "PD_TRANSFER_TIMEOUT"
                | "PD_TRANSFER_FAILED"
                | "PD_ACK_TIMEOUT"
                | "PD_ABORTED"
                | "PD_STALE_EPOCH"
                | "PD_LOCAL_FATAL"
        )
    ) {
        return false;
    }
    matches!(
        response
            .headers()
            .get("x-sglang-retryable")
            .and_then(|value| value.to_str().ok()),
        Some("true")
    )
}

pub(super) fn has_typed_error(headers: &HeaderMap) -> bool {
    headers.contains_key("x-sglang-pd-reason") && headers.contains_key("x-sglang-retryable")
}

pub(super) fn request_invalid_response() -> Response {
    pd_error_response(PdRawError::RequestInvalid)
}

pub(super) fn pd_error_response(error: PdRawError) -> Response {
    let (status, message, reason) = match error {
        PdRawError::RequestInvalid => (
            StatusCode::BAD_REQUEST,
            "PD request is invalid",
            "PD_REQUEST_INVALID",
        ),
        PdRawError::Unsupported => (
            StatusCode::UNPROCESSABLE_ENTITY,
            "PD request uses an unsupported capability",
            "PD_UNSUPPORTED",
        ),
    };
    let mut response = (
        status,
        Json(json!({
            "error": {
                "code": status.as_u16(),
                "message": message,
                "pd_reason": reason,
                "retryable": false,
                "type": "pd_error"
            }
        })),
    )
        .into_response();
    response
        .headers_mut()
        .insert("x-sglang-pd-reason", HeaderValue::from_static(reason));
    response
        .headers_mut()
        .insert("x-sglang-retryable", HeaderValue::from_static("false"));
    response
}

pub(super) fn inject_bootstrap(
    original: Value,
    prefill_worker: &dyn Worker,
    batch_size: Option<usize>,
    include_attempt_id: bool,
) -> Result<Value, String> {
    inject_bootstrap_with(
        original,
        prefill_worker,
        batch_size,
        include_attempt_id,
        || {
            (
                super::super::pd_types::generate_room_id(),
                uuid::Uuid::new_v4().to_string(),
            )
        },
    )
}

fn inject_bootstrap_with<F>(
    mut original: Value,
    prefill_worker: &dyn Worker,
    batch_size: Option<usize>,
    include_attempt_id: bool,
    mut next_identity: F,
) -> Result<Value, String>
where
    F: FnMut() -> (u64, String),
{
    let object = original
        .as_object_mut()
        .ok_or_else(|| "Request must be a JSON object".to_string())?;

    let count = batch_size.unwrap_or(1);
    let mut hosts = Vec::with_capacity(count);
    let mut ports = Vec::with_capacity(count);
    let mut rooms = Vec::with_capacity(count);
    let mut attempts = Vec::with_capacity(count);
    for _ in 0..count {
        let (room, attempt_id) = next_identity();
        if room > i64::MAX as u64 {
            return Err("bootstrap_room must be in 0..2^63-1".to_string());
        }
        hosts.push(Value::from(prefill_worker.bootstrap_host()));
        ports.push(
            prefill_worker
                .bootstrap_port()
                .map_or(Value::Null, Value::from),
        );
        rooms.push(Value::from(room));
        attempts.push(Value::from(attempt_id));
    }

    if batch_size.is_some() {
        object.insert(BOOTSTRAP_HOST_KEY.to_string(), Value::Array(hosts));
        object.insert(BOOTSTRAP_PORT_KEY.to_string(), Value::Array(ports));
        object.insert(BOOTSTRAP_ROOM_KEY.to_string(), Value::Array(rooms));
        if include_attempt_id {
            object.insert(BOOTSTRAP_ATTEMPT_ID_KEY.to_string(), Value::Array(attempts));
        }
    } else {
        object.insert(BOOTSTRAP_HOST_KEY.to_string(), hosts.remove(0));
        object.insert(BOOTSTRAP_PORT_KEY.to_string(), ports.remove(0));
        object.insert(BOOTSTRAP_ROOM_KEY.to_string(), rooms.remove(0));
        if include_attempt_id {
            object.insert(BOOTSTRAP_ATTEMPT_ID_KEY.to_string(), attempts.remove(0));
        }
    }
    Ok(original)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet, convert::Infallible, future::pending, sync::Arc, time::Duration,
    };

    use axum::{
        body::Body,
        http::{
            header::{AUTHORIZATION, CONTENT_LENGTH},
            HeaderValue, StatusCode,
        },
        response::Response,
        routing::post,
        Router as AxumRouter,
    };
    use serde_json::json;

    use super::*;
    use crate::{
        config::{types::RetryConfig, PolicyConfig},
        core::{BasicWorkerBuilder, WorkerRegistry, WorkerType},
        policies::PolicyRegistry,
        routers::RouterTrait,
    };

    fn prefill_worker() -> impl Worker {
        BasicWorkerBuilder::new("http://configured-prefill.internal:30000")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build()
    }

    fn router() -> super::super::PDRouter {
        super::super::PDRouter {
            worker_registry: Arc::new(WorkerRegistry::new()),
            policy_registry: Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            client: reqwest::Client::new(),
            retry_config: RetryConfig::default(),
            api_key: Some("client-secret".to_string()),
            enable_igw: false,
            rendezvous_gate: Arc::new(tokio::sync::Mutex::new(())),
            active_item_permits: Arc::new(tokio::sync::Semaphore::new(
                super::super::PD_ACTIVE_ITEM_CAPACITY,
            )),
        }
    }

    struct PendingRequestGuard {
        dropped: Arc<tokio::sync::Notify>,
    }

    impl Drop for PendingRequestGuard {
        fn drop(&mut self) {
            self.dropped.notify_one();
        }
    }

    #[derive(Clone)]
    struct PendingRequestState {
        started: Arc<tokio::sync::Notify>,
        dropped: Arc<tokio::sync::Notify>,
    }

    async fn pending_prefill(
        axum::extract::State(state): axum::extract::State<PendingRequestState>,
    ) -> Result<Response, Infallible> {
        let _guard = PendingRequestGuard {
            dropped: Arc::clone(&state.dropped),
        };
        state.started.notify_one();
        pending::<Result<Response, Infallible>>().await
    }

    async fn start_test_server(app: AxumRouter) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), task)
    }

    async fn route_typed_decode_failure(stream: bool) -> Response {
        const ERROR_BODY: &str = concat!(
            "{\"error\":{\"message\":\"PD peer is unavailable\",",
            "\"type\":\"pd_error\",\"code\":503,",
            "\"pd_reason\":\"PD_PEER_UNAVAILABLE\",\"retryable\":true}}"
        );
        let state = PendingRequestState {
            started: Arc::new(tokio::sync::Notify::new()),
            dropped: Arc::new(tokio::sync::Notify::new()),
        };
        let (prefill_url, prefill_task) = start_test_server(
            AxumRouter::new()
                .route("/generate", post(pending_prefill))
                .with_state(state.clone()),
        )
        .await;
        let started = Arc::clone(&state.started);
        let (decode_url, decode_task) = start_test_server(AxumRouter::new().route(
            "/generate",
            post(move || {
                let started = Arc::clone(&started);
                async move {
                    started.notified().await;
                    Response::builder()
                        .status(StatusCode::SERVICE_UNAVAILABLE)
                        .header("x-sglang-pd-reason", "PD_PEER_UNAVAILABLE")
                        .header("x-sglang-retryable", "true")
                        .body(Body::from(ERROR_BODY))
                        .unwrap()
                }
            }),
        ))
        .await;

        let mut router = router();
        router.retry_config.max_retries = 0;
        let prefill = BasicWorkerBuilder::new(prefill_url)
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();
        prefill.set_healthy(true);
        let decode = BasicWorkerBuilder::new(decode_url)
            .worker_type(WorkerType::Decode)
            .build();
        decode.set_healthy(true);
        router.worker_registry.register(Arc::new(prefill));
        router.worker_registry.register(Arc::new(decode));

        let response = tokio::time::timeout(
            Duration::from_secs(2),
            router.route_generate_raw(
                None,
                &json!({
                    "text": "hello",
                    "stream": stream,
                    "sampling_params": {"temperature": 0}
                }),
                None,
            ),
        )
        .await
        .expect("decode failure must not wait for pending prefill");
        tokio::time::timeout(Duration::from_secs(2), state.dropped.notified())
            .await
            .expect("paired prefill request was not cancelled");
        prefill_task.abort();
        decode_task.abort();
        response
    }

    async fn route_typed_prefill_failure(stream: bool) -> Response {
        const ERROR_BODY: &str = concat!(
            "{\"error\":{\"message\":\"PD protocol mismatch\",",
            "\"type\":\"pd_error\",\"code\":503,",
            "\"pd_reason\":\"PD_PROTOCOL_MISMATCH\",\"retryable\":false}}"
        );
        let state = PendingRequestState {
            started: Arc::new(tokio::sync::Notify::new()),
            dropped: Arc::new(tokio::sync::Notify::new()),
        };
        let started = Arc::clone(&state.started);
        let (prefill_url, prefill_task) = start_test_server(AxumRouter::new().route(
            "/generate",
            post(move || {
                let started = Arc::clone(&started);
                async move {
                    started.notified().await;
                    Response::builder()
                        .status(StatusCode::SERVICE_UNAVAILABLE)
                        .header("x-sglang-pd-reason", "PD_PROTOCOL_MISMATCH")
                        .header("x-sglang-retryable", "false")
                        .body(Body::from(ERROR_BODY))
                        .unwrap()
                }
            }),
        ))
        .await;
        let (decode_url, decode_task) = start_test_server(
            AxumRouter::new()
                .route("/generate", post(pending_prefill))
                .with_state(state.clone()),
        )
        .await;

        let mut router = router();
        router.retry_config.max_retries = 0;
        let prefill = BasicWorkerBuilder::new(prefill_url)
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();
        prefill.set_healthy(true);
        let decode = BasicWorkerBuilder::new(decode_url)
            .worker_type(WorkerType::Decode)
            .build();
        decode.set_healthy(true);
        router.worker_registry.register(Arc::new(prefill));
        router.worker_registry.register(Arc::new(decode));

        let response = tokio::time::timeout(
            Duration::from_secs(2),
            router.route_generate_raw(
                None,
                &json!({
                    "text": "hello",
                    "stream": stream,
                    "sampling_params": {"temperature": 0}
                }),
                None,
            ),
        )
        .await
        .expect("prefill failure must not wait for pending decode");
        tokio::time::timeout(Duration::from_secs(2), state.dropped.notified())
            .await
            .expect("paired decode request was not cancelled");
        prefill_task.abort();
        decode_task.abort();
        response
    }

    #[test]
    fn raw_shape_accepts_scalar_and_batch_text_or_token_ids() {
        let cases = [
            (json!({"text": "hello"}), (None, false, false)),
            (
                json!({"text": ["a", "b"], "stream": true}),
                (Some(2), true, false),
            ),
            (
                json!({"input_ids": [0, u32::MAX], "return_logprob": true}),
                (None, false, true),
            ),
            (json!({"input_ids": [[1], [2, 3]]}), (Some(2), false, false)),
        ];
        for (body, expected) in cases {
            assert_eq!(generate_shape(&body), Ok(expected), "{body}");
        }
    }

    #[test]
    fn raw_shape_rejects_ambiguous_invalid_or_oversized_inputs() {
        let invalid = [
            json!({}),
            json!({"text": "x", "input_ids": [1]}),
            json!({"text": []}),
            json!({"text": ["x", 1]}),
            json!({"text": ["0", "1", "2", "3", "4", "5", "6", "7", "8"]}),
            json!({"input_ids": []}),
            json!({"input_ids": [[]]}),
            json!({"input_ids": [u32::MAX as u64 + 1]}),
            json!({"input_ids": [-1]}),
            json!({"text": "x", "stream": "true"}),
            json!({"text": "x", "return_logprob": 1}),
        ];
        for body in invalid {
            assert!(generate_shape(&body).is_err(), "{body}");
        }
    }

    #[test]
    fn injection_preserves_payload_and_overwrites_all_client_identity_shapes() {
        let worker = prefill_worker();
        let hostile_values = [
            json!("http://attacker.invalid"),
            json!(65535),
            json!(["attacker.invalid"]),
            Value::Null,
            json!({"host": "attacker.invalid"}),
        ];
        for key in [
            BOOTSTRAP_HOST_KEY,
            BOOTSTRAP_PORT_KEY,
            BOOTSTRAP_ROOM_KEY,
            BOOTSTRAP_ATTEMPT_ID_KEY,
        ] {
            for hostile in &hostile_values {
                let mut body = json!({
                    "text": "hello",
                    "sampling_params": {"temperature": 0, "stop": ["END"]},
                    "custom": [null, {"nested": true}]
                });
                body[key] = hostile.clone();
                let injected = inject_bootstrap_with(body, &worker, None, true, || {
                    (0, "00000000-0000-4000-8000-000000000001".to_string())
                })
                .unwrap();
                assert_eq!(injected["bootstrap_host"], "configured-prefill.internal");
                assert_eq!(injected["bootstrap_port"], 8998);
                assert_eq!(injected["bootstrap_room"], 0);
                assert_eq!(
                    injected["bootstrap_attempt_id"],
                    "00000000-0000-4000-8000-000000000001"
                );
                assert_eq!(
                    injected["sampling_params"],
                    json!({"temperature": 0, "stop": ["END"]})
                );
                assert_eq!(injected["custom"], json!([null, {"nested": true}]));
            }
        }
    }

    #[tokio::test]
    async fn batch_identity_supports_room_boundaries_and_is_shared_by_both_sides() {
        let worker = prefill_worker();
        let decode = BasicWorkerBuilder::new("http://configured-decode.internal:30001")
            .worker_type(WorkerType::Decode)
            .build();
        let identities = [
            (0, "00000000-0000-4000-8000-000000000001".to_string()),
            (
                i64::MAX as u64,
                "00000000-0000-4000-8000-000000000002".to_string(),
            ),
        ];
        let mut identities = identities.into_iter();
        let injected = inject_bootstrap_with(
            json!({
                "text": ["a", "b"],
                "bootstrap_host": null,
                "bootstrap_port": "attacker",
                "bootstrap_room": {"bad": true},
                "bootstrap_attempt_id": 7
            }),
            &worker,
            Some(2),
            true,
            || identities.next().unwrap(),
        )
        .unwrap();
        assert_eq!(
            injected["bootstrap_host"],
            json!(["configured-prefill.internal", "configured-prefill.internal"])
        );
        assert_eq!(injected["bootstrap_port"], json!([8998, 8998]));
        assert_eq!(injected["bootstrap_room"], json!([0, i64::MAX as u64]));
        assert_eq!(
            injected["bootstrap_attempt_id"],
            json!([
                "00000000-0000-4000-8000-000000000001",
                "00000000-0000-4000-8000-000000000002"
            ])
        );
        let (prefill_request, decode_request) = super::super::PDRouter::prepare_pd_worker_requests(
            "/generate",
            &injected,
            &worker,
            &decode,
        )
        .await
        .unwrap();
        assert_eq!(prefill_request.body.as_ref(), &injected);
        assert_eq!(decode_request.body.as_ref(), &injected);
    }

    #[test]
    fn generated_attempt_ids_are_distinct_uuid_v4_values_and_rooms_are_u63() {
        let worker = prefill_worker();
        let injected =
            inject_bootstrap(json!({"text": ["a", "b"]}), &worker, Some(2), true).unwrap();
        let attempts = injected["bootstrap_attempt_id"].as_array().unwrap();
        let unique = attempts
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect::<HashSet<_>>();
        assert_eq!(unique.len(), 2);
        for attempt in unique {
            assert_eq!(uuid::Uuid::parse_str(attempt).unwrap().get_version_num(), 4);
        }
        for room in injected["bootstrap_room"].as_array().unwrap() {
            assert!(room.as_u64().unwrap() <= i64::MAX as u64);
        }
    }

    #[test]
    fn typed_retry_headers_take_priority_over_status_fallback() {
        let mut retryable = Response::new(Body::empty());
        *retryable.status_mut() = StatusCode::BAD_REQUEST;
        retryable.headers_mut().insert(
            "x-sglang-pd-reason",
            HeaderValue::from_static("PD_PEER_UNAVAILABLE"),
        );
        retryable
            .headers_mut()
            .insert("x-sglang-retryable", HeaderValue::from_static("true"));
        assert!(response_retryable(&retryable));

        let mut terminal = Response::new(Body::empty());
        *terminal.status_mut() = StatusCode::SERVICE_UNAVAILABLE;
        terminal.headers_mut().insert(
            "x-sglang-pd-reason",
            HeaderValue::from_static("PD_REQUEST_INVALID"),
        );
        terminal
            .headers_mut()
            .insert("x-sglang-retryable", HeaderValue::from_static("false"));
        assert!(!response_retryable(&terminal));

        let mut fallback = Response::new(Body::empty());
        *fallback.status_mut() = StatusCode::SERVICE_UNAVAILABLE;
        assert!(response_retryable(&fallback));

        let mut unknown = Response::new(Body::empty());
        *unknown.status_mut() = StatusCode::SERVICE_UNAVAILABLE;
        unknown.headers_mut().insert(
            "x-sglang-pd-reason",
            HeaderValue::from_static("attacker-controlled-message"),
        );
        unknown
            .headers_mut()
            .insert("x-sglang-retryable", HeaderValue::from_static("true"));
        assert!(!response_retryable(&unknown));
    }

    #[tokio::test]
    async fn invalid_raw_generate_shape_returns_frozen_typed_pd_error() {
        let response = router()
            .route_generate_raw(
                None,
                &json!({"sampling_params": {"temperature": 0, "max_new_tokens": 1}}),
                None,
            )
            .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers().get("x-sglang-pd-reason").unwrap(),
            "PD_REQUEST_INVALID"
        );
        assert_eq!(
            response.headers().get("x-sglang-retryable").unwrap(),
            "false"
        );
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap(),
            json!({
                "error": {
                    "message": "PD request is invalid",
                    "type": "pd_error",
                    "code": 400,
                    "pd_reason": "PD_REQUEST_INVALID",
                    "retryable": false
                }
            })
        );
    }

    #[tokio::test]
    async fn prevalidation_errors_match_native_api_wire_bytes() {
        let cases = [
            (
                PdRawError::RequestInvalid,
                concat!(
                    "{\"error\":{\"code\":400,\"message\":\"PD request is invalid\",",
                    "\"pd_reason\":\"PD_REQUEST_INVALID\",\"retryable\":false,",
                    "\"type\":\"pd_error\"}}"
                ),
            ),
            (
                PdRawError::Unsupported,
                concat!(
                    "{\"error\":{\"code\":422,",
                    "\"message\":\"PD request uses an unsupported capability\",",
                    "\"pd_reason\":\"PD_UNSUPPORTED\",\"retryable\":false,",
                    "\"type\":\"pd_error\"}}"
                ),
            ),
        ];
        for (error, expected) in cases {
            let response = pd_error_response(error);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(body.as_ref(), expected.as_bytes());
        }
    }

    #[tokio::test]
    async fn raw_pd_support_and_range_errors_are_typed_before_worker_selection() {
        let unsupported = [
            json!({"input_ids": [1], "sampling_params": {"temperature": 0.5, "max_new_tokens": 1}}),
            json!({"input_ids": [1], "sampling_params": {"temperature": 0, "max_new_tokens": 1, "json_schema": {"type": "object"}}}),
            json!({"input_ids": [1], "sampling_params": {"temperature": 0, "top_k": 2, "max_new_tokens": 1}}),
            json!({"input_ids": [1], "sampling_params": {"temperature": 0, "max_new_tokens": 1}, "return_logprob": true}),
            json!({"input_ids": [1], "sampling_params": {"temperature": 0, "max_new_tokens": 1}, "n": 2}),
            json!({"input_ids": [1], "sampling_params": {"temperature": 0, "max_new_tokens": 1}, "image_data": "image"}),
        ];
        for payload in unsupported {
            let response = router().route_generate_raw(None, &payload, None).await;
            assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
            assert_eq!(
                response.headers().get("x-sglang-pd-reason").unwrap(),
                "PD_UNSUPPORTED"
            );
        }

        let invalid = [
            json!({"input_ids": vec![1; 4097], "sampling_params": {"temperature": 0, "max_new_tokens": 1}}),
            json!({"input_ids": [1], "sampling_params": {"temperature": 0, "max_new_tokens": -1}}),
            json!({"input_ids": [1], "sampling_params": {"temperature": 0, "max_new_tokens": 257}}),
            json!({"input_ids": [1], "sampling_params": {"temperature": 0, "max_new_tokens": 1}, "pd07_unknown": true}),
        ];
        for payload in invalid {
            let response = router().route_generate_raw(None, &payload, None).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert_eq!(
                response.headers().get("x-sglang-pd-reason").unwrap(),
                "PD_REQUEST_INVALID"
            );
        }
    }

    #[test]
    fn client_authorization_is_forwarded_unchanged() {
        let router = router();
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_static("Bearer client-secret"),
        );
        let request = router
            .build_post_with_headers(
                &router.client,
                "http://configured-prefill.internal:30000/generate",
                &json!({"text": "hello"}),
                Some(&headers),
                false,
            )
            .build()
            .unwrap();
        assert_eq!(
            request.headers().get(AUTHORIZATION),
            Some(&HeaderValue::from_static("Bearer client-secret"))
        );
    }

    #[tokio::test]
    async fn decode_failure_cancels_prefill_and_preserves_typed_unary_error() {
        let response = route_typed_decode_failure(false).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.headers().get("x-sglang-pd-reason").unwrap(),
            "PD_PEER_UNAVAILABLE"
        );
        assert_eq!(
            response.headers().get("x-sglang-retryable").unwrap(),
            "true"
        );
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            body,
            concat!(
                "{\"error\":{\"message\":\"PD peer is unavailable\",",
                "\"type\":\"pd_error\",\"code\":503,",
                "\"pd_reason\":\"PD_PEER_UNAVAILABLE\",\"retryable\":true}}"
            )
        );
    }

    #[tokio::test]
    async fn typed_sse_error_has_one_terminal_and_cancels_prefill() {
        let response = route_typed_decode_failure(true).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            body,
            concat!(
                "data: {\"error\":{\"message\":\"PD peer is unavailable\",",
                "\"type\":\"pd_error\",\"code\":503,",
                "\"pd_reason\":\"PD_PEER_UNAVAILABLE\",\"retryable\":true}}\n\n",
                "data: [DONE]\n\n"
            )
        );
        assert_eq!(String::from_utf8_lossy(&body).matches("[DONE]").count(), 1);
    }

    #[tokio::test]
    async fn prefill_first_typed_failure_is_symmetric_and_cancels_decode() {
        let unary = route_typed_prefill_failure(false).await;
        assert_eq!(unary.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            unary.headers().get("x-sglang-pd-reason").unwrap(),
            "PD_PROTOCOL_MISMATCH"
        );

        let stream = route_typed_prefill_failure(true).await;
        assert_eq!(stream.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            stream.headers().get(CONTENT_LENGTH).is_none(),
            "synthetic SSE must not retain the unary upstream content-length",
        );
        let body = axum::body::to_bytes(stream.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            body,
            concat!(
                "data: {\"error\":{\"message\":\"PD protocol mismatch\",",
                "\"type\":\"pd_error\",\"code\":503,",
                "\"pd_reason\":\"PD_PROTOCOL_MISMATCH\",\"retryable\":false}}\n\n",
                "data: [DONE]\n\n"
            )
        );
        assert_eq!(String::from_utf8_lossy(&body).matches("[DONE]").count(), 1);
    }
}
