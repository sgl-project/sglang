use std::{borrow::Cow, sync::Arc, time::Instant};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{
        header::{CONTENT_LENGTH, CONTENT_TYPE},
        HeaderMap, HeaderValue, StatusCode,
    },
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use memchr::memmem;
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, warn};

mod dispatch;
mod raw_generate;
mod response;
mod selection;

use super::pd_types::api_path;
use crate::{
    config::types::RetryConfig,
    core::{
        AttachedBody, HashRing, RetryExecutor, Worker, WorkerLoadGuard, WorkerRegistry, WorkerType,
        UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo},
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        common::{GenerationRequest, InputIds, StringOrArray},
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
    },
    routers::{
        error,
        grpc::utils::{error_type_from_status, route_to_endpoint},
        header_utils,
        streaming_utils::BreakerTrackedStream,
        RouterTrait,
    },
};

const PD_ACTIVE_ITEM_CAPACITY: usize = 32;

#[derive(Debug)]
pub struct PDRouter {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub retry_config: RetryConfig,
    pub api_key: Option<String>,
    pub enable_igw: bool,
    rendezvous_gate: Arc<Mutex<()>>,
    active_item_permits: Arc<tokio::sync::Semaphore>,
}

struct PreparedWorkerRequest<'a> {
    endpoint_url: String,
    body: Cow<'a, Value>,
}

#[derive(Clone)]
struct PDRequestContext<'a> {
    route: &'static str,
    batch_size: Option<usize>,
    is_stream: bool,
    return_logprob: bool,
    request_text: Option<String>,
    model_id: Option<&'a str>,
    headers: Option<HeaderMap>,
}

/// Marker placed on a `Response` by paths inside
/// `execute_dual_dispatch_internal` that have already recorded prefill and
/// decode breaker outcomes against the workers' actual per-side results
/// (rather than the final response status). The outer dispatcher reads this
/// and skips its own status-based `record_outcome` calls so a decode-only
/// transport failure can't be misattributed to a healthy prefill.
#[derive(Clone, Copy)]
struct BreakerOutcomesRecorded;

#[async_trait]
impl RouterTrait for PDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair

        // Select a random worker pair using the policy
        let (prefill, decode) = match self.select_pd_pair(None, None, None).await {
            Ok(pair) => pair,
            Err(e) => {
                return error::service_unavailable(
                    "no_healthy_worker_pair",
                    format!("No healthy worker pair available: {}", e),
                );
            }
        };

        let prefill_url = Self::worker_endpoint_url(prefill.as_ref(), "health_generate");
        let decode_url = Self::worker_endpoint_url(decode.as_ref(), "health_generate");
        let (prefill_result, decode_result) = tokio::join!(
            self.client.get(&prefill_url).send(),
            self.client.get(&decode_url).send()
        );

        // Check results
        let mut errors = Vec::new();

        match prefill_result {
            Ok(res) if res.status().is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(res) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(res) if res.status().is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(res) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            }
        }

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!(
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    prefill.url(),
                    decode.url()
                ),
            )
                .into_response()
        } else {
            error::service_unavailable(
                "health_generate_failed",
                format!("Health generate failed: {:?}", errors),
            )
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // Get info from the first decode server to match sglang's server info format
        // Note: We use decode workers for server info to match expected format
        self.proxy_to_first_prefill_worker("server_info", None)
            .await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
            .await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("model_info", Some(headers))
            .await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.return_logprob.unwrap_or(false);

        let request_text = if self.policies_need_request_text() {
            body.text.as_deref().map(|s| s.to_string())
        } else {
            None
        };

        let batch_size = Self::get_generate_batch_size(body);

        let context = PDRequestContext {
            route: "/generate",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_generate_raw(
        &self,
        headers: Option<&HeaderMap>,
        body: &Value,
        model_id: Option<&str>,
    ) -> Response {
        if let Err(error) = raw_generate::validate_pd_support(body) {
            return raw_generate::pd_error_response(error);
        }
        let (batch_size, is_stream, return_logprob) = match raw_generate::generate_shape(body) {
            Ok(shape) => shape,
            Err(_) => return raw_generate::request_invalid_response(),
        };
        let request_text = self
            .policies_need_request_text()
            .then(|| raw_generate::request_text(body))
            .flatten();
        let context = PDRequestContext {
            route: "/generate",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };
        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs;

        let request_text = if self.policies_need_request_text() {
            Self::build_chat_request_text(body)
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_chat_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        let request_text = if self.policies_need_request_text() {
            match &body.prompt {
                StringOrArray::String(s) => Some(s.clone()),
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            }
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_completion_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Extract text for cache-aware routing
        let req_text = if self.policies_need_request_text() {
            Some(body.query.clone())
        } else {
            None
        };

        let context = PDRequestContext {
            route: "/v1/rerank",
            batch_size: None,
            is_stream: false,
            return_logprob: false,
            request_text: req_text,
            model_id,
            headers: headers.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/embeddings; returning bad request");
        error::bad_request(
            "pd_unsupported_embeddings",
            "PD mode does not support /v1/embeddings",
        )
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/classify; returning bad request");
        error::bad_request(
            "pd_unsupported_classify",
            "PD mode does not support /v1/classify",
        )
    }

    fn router_type(&self) -> &'static str {
        "pd"
    }
}

#[cfg(test)]
mod tests {
    use std::{
        convert::Infallible,
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    use super::*;
    use crate::core::{BasicWorkerBuilder, DPAwareWorkerBuilder, WorkerType};

    fn create_test_pd_router() -> PDRouter {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));

        PDRouter {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            api_key: Some("test_api_key".to_string()),
            enable_igw: false,
            rendezvous_gate: Arc::new(Mutex::new(())),
            active_item_permits: Arc::new(tokio::sync::Semaphore::new(PD_ACTIVE_ITEM_CAPACITY)),
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .build();
        worker.set_healthy(healthy);
        Box::new(worker)
    }

    #[tokio::test]
    async fn pd_item_admission_is_weighted_and_exactly_32() {
        let router = create_test_pd_router();
        let all_items = Arc::clone(&router.active_item_permits)
            .acquire_many_owned(32)
            .await
            .unwrap();
        assert_eq!(router.active_item_permits.available_permits(), 0);
        assert!(tokio::time::timeout(
            Duration::from_millis(10),
            Arc::clone(&router.active_item_permits).acquire_owned(),
        )
        .await
        .is_err());

        drop(all_items);
        let one_item = Arc::clone(&router.active_item_permits)
            .acquire_owned()
            .await
            .unwrap();
        assert_eq!(router.active_item_permits.available_permits(), 31);
        drop(one_item);
        assert_eq!(router.active_item_permits.available_permits(), 32);
    }

    #[tokio::test]
    async fn pd_item_permit_is_held_until_response_body_drop() {
        let router = create_test_pd_router();
        let permit = Arc::clone(&router.active_item_permits)
            .acquire_many_owned(8)
            .await
            .unwrap();
        let response = AttachedBody::wrap_response(Response::new(Body::empty()), permit);
        assert_eq!(router.active_item_permits.available_permits(), 24);

        drop(response);
        assert_eq!(router.active_item_permits.available_permits(), 32);
    }

    #[tokio::test]
    async fn pd_rendezvous_gate_is_held_until_prefill_body_is_consumed() {
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let prefill_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let prefill_address = prefill_listener.local_addr().unwrap();
        let prefill_app = axum::Router::new().route(
            "/generate",
            axum::routing::post({
                let active = Arc::clone(&active);
                let maximum = Arc::clone(&maximum);
                move || {
                    let active = Arc::clone(&active);
                    let maximum = Arc::clone(&maximum);
                    async move {
                        let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                        maximum.fetch_max(current, Ordering::SeqCst);
                        let body = futures_util::stream::once(async move {
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            active.fetch_sub(1, Ordering::SeqCst);
                            Ok::<_, Infallible>(bytes::Bytes::from_static(b"{}"))
                        });
                        Response::new(Body::from_stream(body))
                    }
                }
            }),
        );
        let prefill_server =
            tokio::spawn(async move { axum::serve(prefill_listener, prefill_app).await });

        let decode_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let decode_address = decode_listener.local_addr().unwrap();
        let decode_app = axum::Router::new().route(
            "/generate",
            axum::routing::post(|| async { (StatusCode::OK, "{}") }),
        );
        let decode_server =
            tokio::spawn(async move { axum::serve(decode_listener, decode_app).await });

        let router = create_test_pd_router();
        let prefill: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("http://{prefill_address}"))
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(8998),
                })
                .build(),
        );
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("http://{decode_address}"))
                .worker_type(WorkerType::Decode)
                .build(),
        );
        let request = raw_generate::inject_bootstrap(
            json!({
                "text": "rendezvous ordering",
                "stream": false,
                "sampling_params": {"temperature": 0}
            }),
            prefill.as_ref(),
            None,
            true,
        )
        .unwrap();
        let context = || PDRequestContext {
            route: "/generate",
            batch_size: None,
            is_stream: false,
            return_logprob: false,
            request_text: None,
            model_id: None,
            headers: None,
        };

        let first = router.execute_dual_dispatch_internal(
            None,
            request.clone(),
            context(),
            Arc::clone(&prefill),
            Arc::clone(&decode),
            Instant::now(),
        );
        let second = router.execute_dual_dispatch_internal(
            None,
            request,
            context(),
            prefill,
            decode,
            Instant::now(),
        );
        let (first, second) = tokio::join!(first, second);

        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(second.status(), StatusCode::OK);
        assert_eq!(
            maximum.load(Ordering::SeqCst),
            1,
            "a second Prefill request entered before the first response body reached terminal"
        );

        prefill_server.abort();
        decode_server.abort();
    }

    #[test]
    fn test_chat_request_text_uses_full_conversation() {
        // Regression test for https://github.com/sgl-project/sglang/issues/26263
        // Cache-aware routing must build its text from the full conversation, not
        // just the first message, so that KV-cache prefix matching reflects what
        // the worker will actually process in a multi-turn chat.
        let body: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "First question about apples."},
                {"role": "assistant", "content": "Apples are red."},
                {"role": "user", "content": "Follow up question about oranges."}
            ]
        }))
        .expect("valid chat request");

        let text = PDRouter::build_chat_request_text(&body)
            .expect("multi-message chat should produce routing text");

        assert!(
            text.contains("apples"),
            "routing text must include earlier turns, got: {text:?}"
        );
        assert!(
            text.contains("oranges"),
            "routing text must include later turns (not only the first message), got: {text:?}"
        );
    }

    #[test]
    fn test_chat_request_text_none_when_no_text() {
        // When the conversation carries no text content, no routing text should
        // be produced (None) rather than an empty string, preserving the prior
        // PD behavior. See https://github.com/sgl-project/sglang/issues/26263.
        let body: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": ""}
            ]
        }))
        .expect("valid chat request");

        assert!(
            PDRouter::build_chat_request_text(&body).is_none(),
            "empty conversation text should produce None, not Some(\"\")"
        );
    }

    #[tokio::test]
    async fn test_select_healthy_prefill_worker() {
        let router = create_test_pd_router();

        let healthy_worker = create_test_worker(
            "http://healthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let unhealthy_worker = create_test_worker(
            "http://unhealthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            false,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(unhealthy_worker));
        router.worker_registry.register(Arc::from(healthy_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_ok());
        let (prefill, _decode) = result.unwrap();

        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
    }

    #[tokio::test]
    async fn test_empty_worker_lists() {
        let router = create_test_pd_router();

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No prefill workers available"));
    }

    #[test]
    fn test_worker_endpoint_url_uses_base_url_for_dp_aware_worker() {
        let worker = DPAwareWorkerBuilder::new("http://prefill:30000", 2, 4)
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();

        assert_eq!(
            PDRouter::worker_endpoint_url(&worker, "health_generate"),
            "http://prefill:30000/health_generate"
        );
        assert_eq!(
            PDRouter::worker_endpoint_url(&worker, "/v1/models"),
            "http://prefill:30000/v1/models"
        );
    }

    #[tokio::test]
    async fn test_prepare_pd_worker_requests_uses_dp_aware_rank() {
        let prefill = DPAwareWorkerBuilder::new("http://prefill:30000", 2, 4)
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();
        let decode = DPAwareWorkerBuilder::new("http://decode:30001", 1, 4)
            .worker_type(WorkerType::Decode)
            .build();
        let request = json!({
            "prompt": "shared prefix",
            "max_tokens": 8,
            "bootstrap_host": "prefill",
            "bootstrap_port": 8998,
            "bootstrap_room": 1234,
        });

        let (prefill_request, decode_request) =
            PDRouter::prepare_pd_worker_requests("/v1/completions", &request, &prefill, &decode)
                .await
                .unwrap();

        assert_eq!(
            prefill_request.endpoint_url,
            "http://prefill:30000/v1/completions"
        );
        assert_eq!(prefill_request.body["data_parallel_rank"], 2);
        assert!(prefill_request.body.get("disagg_prefill_dp_rank").is_none());

        assert_eq!(
            decode_request.endpoint_url,
            "http://decode:30001/v1/completions"
        );
        assert_eq!(decode_request.body["data_parallel_rank"], 1);
        assert_eq!(decode_request.body["disagg_prefill_dp_rank"], 2);
        assert_eq!(decode_request.body["bootstrap_room"], 1234);
        assert!(matches!(prefill_request.body, Cow::Owned(_)));
        assert!(matches!(decode_request.body, Cow::Owned(_)));
    }

    #[tokio::test]
    async fn test_prepare_pd_worker_requests_preserves_non_dp_workers() {
        let prefill = BasicWorkerBuilder::new("http://prefill:30000")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(8998),
            })
            .build();
        let decode = BasicWorkerBuilder::new("http://decode:30001")
            .worker_type(WorkerType::Decode)
            .build();
        let request = json!({
            "prompt": "shared prefix",
            "max_tokens": 8,
            "bootstrap_room": 1234,
        });

        let (prefill_request, decode_request) =
            PDRouter::prepare_pd_worker_requests("/v1/completions", &request, &prefill, &decode)
                .await
                .unwrap();

        assert_eq!(
            prefill_request.endpoint_url,
            "http://prefill:30000/v1/completions"
        );
        assert_eq!(
            decode_request.endpoint_url,
            "http://decode:30001/v1/completions"
        );
        assert!(prefill_request.body.get("data_parallel_rank").is_none());
        assert!(decode_request.body.get("data_parallel_rank").is_none());
        assert!(decode_request.body.get("disagg_prefill_dp_rank").is_none());
        assert!(matches!(prefill_request.body, Cow::Borrowed(_)));
        assert!(matches!(decode_request.body, Cow::Borrowed(_)));
    }

    #[test]
    fn test_worker_load_metrics() {
        let prefill_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        ));
        let decode_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".to_string(),
            WorkerType::Decode,
            true,
        ));

        let _prefill_guard = WorkerLoadGuard::new(prefill_worker.clone(), None);
        let _decode_guard = WorkerLoadGuard::new(decode_worker.clone(), None);

        assert_eq!(prefill_worker.load(), 1);
        assert_eq!(decode_worker.load(), 1);

        drop(_prefill_guard);
        drop(_decode_guard);

        assert_eq!(prefill_worker.load(), 0);
        assert_eq!(decode_worker.load(), 0);
    }

    #[tokio::test]
    async fn test_streaming_load_tracking() {
        use futures_util::StreamExt;
        use tokio::time::{sleep, Duration};

        let router = create_test_pd_router();

        let prefill_worker = create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(prefill_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let prefill_workers = router.worker_registry.get_prefill_workers();
        let decode_workers = router.worker_registry.get_decode_workers();

        let prefill_ref = prefill_workers[0].clone();
        let decode_ref = decode_workers[0].clone();

        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);

        {
            let response = router.create_streaming_response(
                stream.map(Ok),
                StatusCode::OK,
                None,
                false,
                None,
                prefill_ref.clone(),
                decode_ref.clone(),
            );

            // Guards are now attached to response body, so load should be 1
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            tx.send(bytes::Bytes::from("test data")).unwrap();

            sleep(Duration::from_millis(10)).await;

            // Load still 1 while response body exists
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            drop(tx);

            // Response (and its body with guards) dropped here
            drop(response);
        }

        // Guards dropped when response dropped
        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);
    }
}
