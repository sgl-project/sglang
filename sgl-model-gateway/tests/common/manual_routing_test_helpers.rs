//! Common helpers for ManualPolicy tests

use std::sync::Arc;

use smg::core::{BasicWorkerBuilder, Worker, WorkerType};

use super::redis_test_server::get_shared_server;

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
