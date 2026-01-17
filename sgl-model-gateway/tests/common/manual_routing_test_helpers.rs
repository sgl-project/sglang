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

#[derive(Clone)]
pub struct TestManualConfig {
    pub redis_url: Option<String>,
    pub redis_key_prefix: Option<String>,
    pub max_idle_secs: u64,
}

impl Default for TestManualConfig {
    fn default() -> Self {
        Self {
            redis_url: None,
            redis_key_prefix: None,
            max_idle_secs: 4 * 3600,
        }
    }
}

impl TestManualConfig {
    pub fn local() -> Self {
        Self::default()
    }

    pub fn redis(test_name: &str) -> Self {
        let server = get_shared_server();
        Self {
            redis_url: Some(server.url().to_string()),
            redis_key_prefix: Some(random_prefix(test_name)),
            max_idle_secs: 4 * 3600,
        }
    }

    pub fn redis_with_prefix(prefix: &str) -> Self {
        let server = get_shared_server();
        Self {
            redis_url: Some(server.url().to_string()),
            redis_key_prefix: Some(prefix.to_string()),
            max_idle_secs: 4 * 3600,
        }
    }

    pub fn with_ttl(mut self, secs: u64) -> Self {
        self.max_idle_secs = secs;
        self
    }

    pub fn build_policy(&self) -> ManualPolicy {
        ManualPolicy::with_config(ManualConfig {
            redis_url: self.redis_url.clone(),
            redis_key_prefix: self.redis_key_prefix.clone(),
            max_idle_secs: self.max_idle_secs,
            ..Default::default()
        })
    }
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
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);

    format!("{}:{:x}:{:x}:", test_name, timestamp, count)
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
macro_rules! manual_routing_all_backend_test {
    ($name:ident, $base_port:expr) => {
        paste::paste! {
            #[tokio::test]
            async fn [<$name _local_backend>]() {
                [<$name _impl>]($crate::common::manual_routing_test_helpers::TestManualConfig::local(), $base_port).await;
            }

            #[tokio::test]
            async fn [<$name _redis_backend>]() {
                [<$name _impl>]($crate::common::manual_routing_test_helpers::TestManualConfig::redis(stringify!($name)), $base_port + 500).await;
            }
        }
    };
    ($name:ident) => {
        paste::paste! {
            #[tokio::test]
            async fn [<$name _local_backend>]() {
                [<$name _impl>]($crate::common::manual_routing_test_helpers::TestManualConfig::local()).await;
            }

            #[tokio::test]
            async fn [<$name _redis_backend>]() {
                [<$name _impl>]($crate::common::manual_routing_test_helpers::TestManualConfig::redis(stringify!($name))).await;
            }
        }
    };
}

#[allow(unused_imports)]
pub use manual_routing_all_backend_test;
