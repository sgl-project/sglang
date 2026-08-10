use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{
    body::to_bytes,
    extract::State,
    http::{HeaderMap, HeaderValue},
    routing::post,
    Json, Router as AxumRouter,
};
use serde_json::{json, Value};
use smg::{
    app_context::AppContext,
    config::{PolicyConfig, RouterConfig, RoutingMode},
    core::{BasicWorkerBuilder, ConnectionMode, ModelCard, Worker, WorkerType, UNKNOWN_MODEL_ID},
    protocols::generate::GenerateRequest,
    routers::{http::router::Router, RouterTrait},
};

const SERVED_MODEL: &str = "served-model";
const ARTIFICIAL_PREFERRED_LOAD: usize = 4;
const REPETITIONS: usize = 5;

#[derive(Clone)]
struct BackendState {
    name: &'static str,
    hits: Arc<AtomicUsize>,
}

async fn generate(State(state): State<BackendState>) -> Json<Value> {
    state.hits.fetch_add(1, Ordering::SeqCst);
    Json(json!({"worker": state.name}))
}

async fn start_backend(
    name: &'static str,
) -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
    let hits = Arc::new(AtomicUsize::new(0));
    let app = AxumRouter::new()
        .route("/generate", post(generate))
        .with_state(BackendState {
            name,
            hits: Arc::clone(&hits),
        });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (url, hits, task)
}

fn make_worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new(SERVED_MODEL))
            .build(),
    )
}

fn fallback_index(key: &str, worker_count: usize) -> usize {
    let hash = blake3::hash(key.as_bytes());
    let hash_value = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
    (hash_value as usize) % worker_count
}

fn hit_url(urls: &[String], hits: &[Arc<AtomicUsize>]) -> String {
    let selected: Vec<usize> = hits
        .iter()
        .enumerate()
        .filter_map(|(index, count)| (count.load(Ordering::SeqCst) == 1).then_some(index))
        .collect();
    assert_eq!(
        selected.len(),
        1,
        "exactly one backend must receive the request"
    );
    urls[selected[0]].clone()
}

#[tokio::test]
async fn non_igw_bounded_spill_uses_key_relative_clockwise_ring() {
    let (url_a, hits_a, task_a) = start_backend("backend-a").await;
    let (url_b, hits_b, task_b) = start_backend("backend-b").await;
    let (url_c, hits_c, task_c) = start_backend("backend-c").await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    let urls = vec![url_a, url_b, url_c];
    let hits = vec![hits_a, hits_b, hits_c];

    let mut config = RouterConfig::new(
        RoutingMode::Regular {
            worker_urls: vec![],
        },
        PolicyConfig::BoundedConsistentHashing {
            max_load_skew: 1.5,
            min_load_gap: 1,
        },
    );
    config.disable_retries = true;
    config.disable_circuit_breaker = true;
    config.health_check.disable_health_check = true;
    config.enable_igw = false;

    let context = Arc::new(AppContext::from_config(config, 5).await.unwrap());
    for url in &urls {
        context.worker_registry.register(make_worker(url));
    }

    let available = context.worker_registry.get_workers_filtered(
        None,
        Some(WorkerType::Regular),
        Some(ConnectionMode::Http),
        None,
        false,
    );
    assert_eq!(available.len(), 3);
    let slice_order: Vec<String> = available
        .iter()
        .map(|worker| worker.url().to_owned())
        .collect();

    assert!(context
        .worker_registry
        .get_hash_ring(SERVED_MODEL)
        .is_some());
    assert!(context
        .worker_registry
        .get_hash_ring(UNKNOWN_MODEL_ID)
        .is_none());
    let aggregate_ring = context
        .worker_registry
        .get_non_igw_regular_http_hash_ring()
        .expect("regular HTTP registration must build the aggregate ring");
    assert_eq!(aggregate_ring.worker_count(), 3);

    let (key, preferred_url, expected_clockwise_url, slice_first_url) = (0..100_000)
        .find_map(|index| {
            let key = format!("non-igw-bounded-http-key-{index}");
            let preferred = aggregate_ring.find_healthy_url(&key, |_| true)?.to_owned();
            let fallback_preferred = &slice_order[fallback_index(&key, slice_order.len())];
            if fallback_preferred != &preferred {
                return None;
            }
            let expected = aggregate_ring
                .find_healthy_url(&key, |url| url != preferred)
                .map(str::to_owned)?;
            let slice_first = slice_order.iter().find(|url| **url != preferred)?.clone();
            (expected != slice_first).then_some((key, preferred, expected, slice_first))
        })
        .expect("could not distinguish clockwise ring walk from slice-first");

    let preferred_worker = available
        .iter()
        .find(|worker| worker.url() == preferred_url)
        .unwrap();
    for _ in 0..ARTIFICIAL_PREFERRED_LOAD {
        preferred_worker.increment_load();
    }

    let mean_load = ARTIFICIAL_PREFERRED_LOAD as f64 / available.len() as f64;
    assert!(ARTIFICIAL_PREFERRED_LOAD > 1);
    assert!((ARTIFICIAL_PREFERRED_LOAD as f64) > mean_load * 1.5);

    let request: GenerateRequest = serde_json::from_value(json!({
        "text": "non-igw-key-relative-ring",
        "model": SERVED_MODEL,
        "stream": false
    }))
    .unwrap();
    let mut headers = HeaderMap::new();
    headers.insert("x-smg-routing-key", HeaderValue::from_str(&key).unwrap());
    let router = Router::new(&context).await.unwrap();

    let mut observed = Vec::new();
    for repetition in 0..REPETITIONS {
        for count in &hits {
            count.store(0, Ordering::SeqCst);
        }
        let response = router
            .route_generate(Some(&headers), &request, Some(SERVED_MODEL))
            .await;
        let status = response.status();
        let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        assert!(
            status.is_success(),
            "repetition={repetition} body={}",
            String::from_utf8_lossy(&body)
        );
        let actual_url = hit_url(&urls, &hits);
        assert_eq!(actual_url, expected_clockwise_url);
        assert_ne!(actual_url, slice_first_url);
        observed.push(actual_url);

        for worker in &available {
            let expected_load =
                usize::from(worker.url() == preferred_url) * ARTIFICIAL_PREFERRED_LOAD;
            assert_eq!(worker.load(), expected_load);
            assert_eq!(worker.worker_routing_key_load().value(), 0);
        }
    }

    assert_eq!(
        observed
            .iter()
            .filter(|url| **url == expected_clockwise_url)
            .count(),
        REPETITIONS
    );
    assert_eq!(
        observed
            .iter()
            .filter(|url| **url == slice_first_url)
            .count(),
        0
    );
    assert!(Arc::ptr_eq(
        &aggregate_ring,
        &context
            .worker_registry
            .get_non_igw_regular_http_hash_ring()
            .unwrap()
    ));

    for _ in 0..ARTIFICIAL_PREFERRED_LOAD {
        preferred_worker.decrement_load();
    }
    for worker in &available {
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.worker_routing_key_load().value(), 0);
    }

    task_a.abort();
    task_b.abort();
    task_c.abort();
}
