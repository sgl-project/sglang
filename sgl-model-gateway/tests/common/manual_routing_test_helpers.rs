//! Common helpers for ManualPolicy tests

use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::{
    core::{BasicWorkerBuilder, Worker, WorkerType},
    policies::{ManualConfig, ManualPolicy},
};
use tower::ServiceExt;

use super::redis_test_server::get_shared_server;

pub const ROUTING_KEY_HEADER: &str = "X-SMG-Routing-Key";

pub struct RedisConfig {
    pub url: String,
    pub key_prefix: String,
}

pub fn create_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
    urls.iter()
        .map(|url| {
            Arc::new(
                BasicWorkerBuilder::new(*url)
                    .worker_type(WorkerType::Regular)
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect()
}

pub fn headers_with_routing_key(key: &str) -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-smg-routing-key", key.parse().unwrap());
    headers
}

pub fn random_prefix(test_name: &str) -> String {
    let random_id: u64 = rand::random();
    format!("{}:{}:", test_name, random_id)
}

pub fn get_redis_config(test_name: &str) -> RedisConfig {
    let server = get_shared_server();
    RedisConfig {
        url: server.url().to_string(),
        key_prefix: random_prefix(test_name),
    }
}

pub fn create_policy(redis_url: Option<String>, redis_key_prefix: Option<String>) -> ManualPolicy {
    ManualPolicy::with_config(ManualConfig {
        redis_url,
        redis_key_prefix,
        ..Default::default()
    })
}

pub fn create_redis_policy(test_name: &str) -> ManualPolicy {
    let cfg = get_redis_config(test_name);
    create_policy(Some(cfg.url), Some(cfg.key_prefix))
}

pub fn create_redis_policy_with_explicit_prefix(prefix: &str) -> ManualPolicy {
    let server = get_shared_server();
    create_policy(Some(server.url().to_string()), Some(prefix.to_string()))
}

pub fn create_redis_policy_with_ttl(test_name: &str, max_idle_secs: u64) -> ManualPolicy {
    let server = get_shared_server();
    ManualPolicy::with_config(ManualConfig {
        redis_url: Some(server.url().to_string()),
        redis_key_prefix: Some(random_prefix(test_name)),
        max_idle_secs,
        ..Default::default()
    })
}

pub async fn send_request(app: axum::Router, routing_key: &str) -> (String, String) {
    let payload = json!({
        "text": format!("Request for {}", routing_key),
        "stream": false
    });

    let req = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .header(ROUTING_KEY_HEADER, routing_key)
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let worker_id = resp
        .headers()
        .get("x-worker-id")
        .expect("Response should have x-worker-id header")
        .to_str()
        .unwrap()
        .to_string();

    (routing_key.to_string(), worker_id)
}

#[macro_export]
macro_rules! all_backend_test {
    ($name:ident) => {
        paste::paste! {
            #[tokio::test]
            async fn [<$name _local_backend>]() {
                [<$name _impl>](None, None).await;
            }

            #[tokio::test]
            async fn [<$name _redis_backend>]() {
                let cfg = $crate::common::manual_routing_test_helpers::get_redis_config(stringify!($name));
                [<$name _impl>](Some(cfg.url), Some(cfg.key_prefix)).await;
            }
        }
    };
}

#[macro_export]
macro_rules! all_backend_e2e_test {
    ($name:ident, $base_port:expr) => {
        paste::paste! {
            #[tokio::test]
            async fn [<$name _local_backend>]() {
                [<$name _impl>]($base_port, None, None).await;
            }

            #[tokio::test]
            async fn [<$name _redis_backend>]() {
                let cfg = $crate::common::manual_routing_test_helpers::get_redis_config(stringify!($name));
                [<$name _impl>]($base_port + 1000, Some(cfg.url), Some(cfg.key_prefix)).await;
            }
        }
    };
}

pub use all_backend_e2e_test;
pub use all_backend_test;
