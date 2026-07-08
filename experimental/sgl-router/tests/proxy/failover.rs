// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use axum::http::Request;
use sgl_router::config::*;
use sgl_router::discovery::{spawn_discovery, ModelId};
use sgl_router::policies::factory::build_registry_with_defaults as build_policy_registry;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::manager;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

#[tokio::test]
async fn failover_when_one_worker_dies() {
    // Three mock workers. Each advertises served_model_name = "tiny" on
    // /server_info, so the worker manager's introspect step resolves the
    // registry's model_ids without us having to hand-declare them here.
    let w1 = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let w2 = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let w3 = crate::common::mock_worker::MockWorker::start(vec![]).await;

    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: Default::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            tokenizer_shards: 1,
            tokenizer_backend: Default::default(),
            tokenizer_l1_cache_mb: 0,
            policy: PolicyKind::RoundRobin,
            circuit_breaker: Some(CircuitBreakerConfig {
                threshold: std::num::NonZeroU32::new(1).unwrap(), // open after first failure
                cool_down_secs: 30,
            }),
            cache_aware: None,
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec![w1.url.clone(), w2.url.clone(), w3.url.clone()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
        admission: AdmissionConfig::default(),
        retry: RetryConfig::default(),
    };

    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());

    let (event_rx, _disc) = spawn_discovery(&cfg).await.unwrap();
    let _mgr = tokio::spawn(manager::run_with_config(
        event_rx,
        registry.clone(),
        Some(Arc::new(cfg.clone())),
        None,
        None,
    ));

    // Poll for the registry to converge — `register_one` introspect is
    // a per-task spawn (manager.rs:127), so order of registration is
    // non-deterministic under load. Cap the wait so a real hang surfaces
    // instead of becoming a flake.
    let converged = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if registry.workers_for(&ModelId("tiny".into())).len() == 3 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(
        converged.is_ok(),
        "registry should contain all 3 workers after discovery; have {}",
        registry.workers_for(&ModelId("tiny".into())).len()
    );

    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let ctx = Arc::new(AppContext::new(
        cfg,
        tokenizers,
        proxy,
        registry.clone(),
        policies,
    ));
    ctx.mark_ready();
    let app = build_router(ctx);

    // Kill w2 by dropping its handle, then poll until its socket
    // actually refuses connections. Without this, the first request
    // routed to w2 can race against the listener's graceful shutdown
    // and succeed, masking the failover assertion below.
    let w2_url = w2.url.clone();
    drop(w2);
    let host_port = w2_url.trim_start_matches("http://");
    let down = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if tokio::net::TcpStream::connect(host_port).await.is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(down.is_ok(), "w2 socket never went down");

    // Send 6 requests; round-robin would route 2 to w2 → connection refused →
    // breaker opens (threshold=1); subsequent round-robin picks rotate among
    // the 2 healthy workers (#1 and #3) because healthy_workers_for filters out w2.
    let mut errs = 0usize;
    let mut oks = 0usize;
    for i in 0..6 {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": format!("hi {i}")}],
        }))
        .unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        if res.status().is_success() {
            oks += 1;
        } else {
            errs += 1;
        }
    }
    // We expect exactly 1 error — the first call routed to w2 fails and opens
    // its breaker; subsequent round-robin picks rotate among the 2 healthy
    // workers since registry.healthy_workers_for filters out the open breaker.
    assert_eq!(errs, 1, "exactly the first w2 pick should error");
    assert_eq!(oks, 5, "remaining 5 picks should succeed via filtered RR");
}

/// With retry enabled and the circuit breaker effectively neutralized, a
/// plain-mode request that round-robins onto a dead worker is re-dispatched to
/// a healthy one instead of failing — the router-side failover the breaker
/// alone can't provide. The breaker threshold is set unreachably high on
/// purpose (there is no true "off": `circuit_breaker: None` means the DEFAULT
/// breaker, threshold 3) so the dead worker stays in the candidate set every
/// round: each request that picks it MUST be recovered by retry, not by the
/// breaker quietly filtering it out. So every request succeeds.
#[tokio::test]
async fn retry_recovers_request_that_lands_on_a_dead_worker() {
    let w1 = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let w2 = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let w3 = crate::common::mock_worker::MockWorker::start(vec![]).await;

    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: Default::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            tokenizer_shards: 1,
            tokenizer_backend: Default::default(),
            tokenizer_l1_cache_mb: 0,
            policy: PolicyKind::RoundRobin,
            // Unreachably-high threshold: the dead worker's failures never trip
            // the breaker, so it is never filtered out of `healthy_workers_for`
            // and recovery is attributable to retry alone. (`None` would mean
            // the DEFAULT breaker — threshold 3 — not "no breaker".)
            circuit_breaker: Some(CircuitBreakerConfig {
                threshold: std::num::NonZeroU32::new(u32::MAX).unwrap(),
                cool_down_secs: 30,
            }),
            cache_aware: None,
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec![w1.url.clone(), w2.url.clone(), w3.url.clone()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
        // Admission disabled → no per-worker cap, so the load gate is a no-op
        // and the single retry always proceeds to a different worker.
        admission: AdmissionConfig::default(),
        retry: RetryConfig { enabled: true },
    };

    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());

    let (event_rx, _disc) = spawn_discovery(&cfg).await.unwrap();
    let _mgr = tokio::spawn(manager::run_with_config(
        event_rx,
        registry.clone(),
        Some(Arc::new(cfg.clone())),
        None,
        None,
    ));

    let converged = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if registry.workers_for(&ModelId("tiny".into())).len() == 3 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(
        converged.is_ok(),
        "registry should contain all 3 workers after discovery; have {}",
        registry.workers_for(&ModelId("tiny".into())).len()
    );

    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let ctx = Arc::new(AppContext::new(
        cfg,
        tokenizers,
        proxy,
        registry.clone(),
        policies,
    ));
    ctx.mark_ready();
    let app = build_router(ctx.clone());

    // Kill w2 and wait until its socket refuses connections.
    let w2_url = w2.url.clone();
    drop(w2);
    let host_port = w2_url.trim_start_matches("http://");
    let down = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if tokio::net::TcpStream::connect(host_port).await.is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(down.is_ok(), "w2 socket never went down");

    // Round-robin routes ~2 of these 6 onto the dead w2; retry must recover
    // each by re-dispatching to w1/w3. All 6 succeed.
    let mut errs = 0usize;
    let mut oks = 0usize;
    for i in 0..6 {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": format!("hi {i}")}],
        }))
        .unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        if res.status().is_success() {
            oks += 1;
        } else {
            errs += 1;
        }
    }
    assert_eq!(errs, 0, "retry should recover every dead-worker pick");
    assert_eq!(oks, 6, "all 6 requests should succeed via retry");

    // Confirm the retry path actually ran (rather than every request happening
    // to miss the dead worker): at least one re-dispatch was recorded.
    let metrics = ctx.metrics.render();
    let retried = metrics
        .lines()
        .find_map(|l| l.strip_prefix("sgl_router_retries_total{model_id=\"tiny\"} "))
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(0);
    assert!(
        retried >= 1,
        "expected at least one recorded retry; metrics:\n{metrics}"
    );
}

/// The load gate: retry never fails over onto a worker that is already at its
/// in-flight cap. Two workers, admission cap = 1. The live worker's single slot
/// is held for the whole test, so it is "full"; the other worker is dead. Every
/// request is admitted onto the (free) dead worker, fails, and finds the only
/// alternative full — so the retry is SKIPPED and the error surfaces. No retry
/// is performed; the "exhausted" counter records the un-recovered requests.
#[tokio::test]
async fn retry_skipped_when_the_only_other_worker_is_full() {
    let w_dead = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let w_live = crate::common::mock_worker::MockWorker::start(vec![]).await;

    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: Default::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            tokenizer_shards: 1,
            tokenizer_backend: Default::default(),
            tokenizer_l1_cache_mb: 0,
            policy: PolicyKind::RoundRobin,
            // Unreachably-high threshold: without it, w_dead's third refused
            // dispatch trips the DEFAULT breaker (threshold 3 — `None` does not
            // mean "no breaker"), `healthy_workers_for` then drops w_dead, and
            // the 4th request's `acquire` sees only the permanently-full w_live
            // and parks forever — deadlocking the test.
            circuit_breaker: Some(CircuitBreakerConfig {
                threshold: std::num::NonZeroU32::new(u32::MAX).unwrap(),
                cool_down_secs: 30,
            }),
            cache_aware: None,
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec![w_dead.url.clone(), w_live.url.clone()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
        // Cap = 1 gives "full" a concrete meaning for the load gate.
        admission: AdmissionConfig::Enabled {
            max_concurrent_per_worker: std::num::NonZeroUsize::new(1).unwrap(),
            max_queued_requests: None,
        },
        retry: RetryConfig { enabled: true },
    };

    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());

    let (event_rx, _disc) = spawn_discovery(&cfg).await.unwrap();
    let _mgr = tokio::spawn(manager::run_with_config(
        event_rx,
        registry.clone(),
        Some(Arc::new(cfg.clone())),
        None,
        None,
    ));

    let converged = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if registry.workers_for(&ModelId("tiny".into())).len() == 2 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(converged.is_ok(), "registry should contain both workers");

    // Occupy w_live's single slot for the whole test → it is "full" (its
    // in-flight load == cap), so the load gate must never retry onto it.
    let live_arc = registry
        .workers_for(&ModelId("tiny".into()))
        .into_iter()
        .find(|w| w.url == w_live.url)
        .expect("w_live must be registered");
    let _held_slot = live_arc.load_guard();
    assert_eq!(live_arc.active_load(), 1, "w_live must be at cap");

    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let ctx = Arc::new(AppContext::new(
        cfg,
        tokenizers,
        proxy,
        registry.clone(),
        policies,
    ));
    ctx.mark_ready();
    let app = build_router(ctx.clone());

    // Kill w_dead and wait until its socket refuses connections. It stays the
    // only worker with a free slot, so every request is admitted onto it.
    let dead_url = w_dead.url.clone();
    drop(w_dead);
    let host_port = dead_url.trim_start_matches("http://");
    let down = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if tokio::net::TcpStream::connect(host_port).await.is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(down.is_ok(), "w_dead socket never went down");

    let mut errs = 0usize;
    for i in 0..4 {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": format!("hi {i}")}],
        }))
        .unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        if !res.status().is_success() {
            errs += 1;
        }
    }
    assert_eq!(
        errs, 4,
        "no request can be recovered — the only alternative is full"
    );

    // The load gate held: NO retry was performed (the alternative was full),
    // and every un-recovered request is counted as exhausted.
    let metrics = ctx.metrics.render();
    let count = |name: &str| -> u64 {
        metrics
            .lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0)
    };
    assert_eq!(
        count("sgl_router_retries_total{model_id=\"tiny\"} "),
        0,
        "retry must be skipped when the only other worker is full; metrics:\n{metrics}"
    );
    assert!(
        count("sgl_router_retries_exhausted_total{model_id=\"tiny\"} ") >= 1,
        "skipped retries must be recorded as exhausted; metrics:\n{metrics}"
    );
}

/// Streaming variant of the recovery test: the streaming arm has the most
/// per-attempt state churn (admission + active-load guards move into the
/// `stream_guards` box each attempt; the pre-headers abort guard and the
/// stream-end/TTFT/ITL hooks are rebuilt per attempt), so pin that a
/// `stream: true` request that lands on a dead worker is re-dispatched and
/// the SSE body arrives intact from the failover worker.
#[tokio::test]
async fn retry_recovers_streaming_request_that_lands_on_a_dead_worker() {
    use http_body_util::BodyExt;

    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let w1 = crate::common::mock_worker::MockWorker::start(chunks.clone()).await;
    let w2 = crate::common::mock_worker::MockWorker::start(chunks.clone()).await;
    let w3 = crate::common::mock_worker::MockWorker::start(chunks).await;

    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: Default::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            tokenizer_shards: 1,
            tokenizer_backend: Default::default(),
            tokenizer_l1_cache_mb: 0,
            policy: PolicyKind::RoundRobin,
            // Unreachably-high threshold — see the non-streaming recovery
            // test for why (`None` means the DEFAULT breaker, threshold 3).
            circuit_breaker: Some(CircuitBreakerConfig {
                threshold: std::num::NonZeroU32::new(u32::MAX).unwrap(),
                cool_down_secs: 30,
            }),
            cache_aware: None,
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec![w1.url.clone(), w2.url.clone(), w3.url.clone()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
        admission: AdmissionConfig::default(),
        retry: RetryConfig { enabled: true },
    };

    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());

    let (event_rx, _disc) = spawn_discovery(&cfg).await.unwrap();
    let _mgr = tokio::spawn(manager::run_with_config(
        event_rx,
        registry.clone(),
        Some(Arc::new(cfg.clone())),
        None,
        None,
    ));

    let converged = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if registry.workers_for(&ModelId("tiny".into())).len() == 3 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(converged.is_ok(), "registry should contain all 3 workers");

    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let ctx = Arc::new(AppContext::new(
        cfg,
        tokenizers,
        proxy,
        registry.clone(),
        policies,
    ));
    ctx.mark_ready();
    let app = build_router(ctx.clone());

    // Kill w2 and wait until its socket refuses connections.
    let w2_url = w2.url.clone();
    drop(w2);
    let host_port = w2_url.trim_start_matches("http://");
    let down = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if tokio::net::TcpStream::connect(host_port).await.is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(down.is_ok(), "w2 socket never went down");

    // Round-robin routes ~2 of these 6 onto the dead w2; each must be
    // recovered by a streaming re-dispatch, and every SSE body must be
    // complete (all 3 chunks from the failover worker).
    for i in 0..6 {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": format!("hi {i}")}],
            "stream": true,
        }))
        .unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        assert!(
            res.status().is_success(),
            "streaming request {i} must be recovered by retry; got {}",
            res.status()
        );
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let data = crate::common::streaming::parse_sse_data(&bytes);
        assert_eq!(
            data.len(),
            3,
            "streaming request {i} must receive the full SSE body from the failover worker"
        );
    }

    // At least one re-dispatch actually happened (RR must have picked the
    // dead worker at least once in 6 requests over 3 workers).
    let metrics = ctx.metrics.render();
    let retried = metrics
        .lines()
        .find_map(|l| l.strip_prefix("sgl_router_retries_total{model_id=\"tiny\"} "))
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(0);
    assert!(
        retried >= 1,
        "expected at least one recorded streaming retry; metrics:\n{metrics}"
    );
}

/// At-most-one-retry termination: with EVERY worker dead, each request makes
/// exactly two upstream attempts (one dispatch + one retry) and then surfaces
/// the error — the `retried` flag, not worker health, is what stops the loop.
/// The test completing at all is the termination assertion; the exact counter
/// values pin the retry budget (retries_total == N, exhausted == N — this is
/// the retried-and-failed-again exhausted branch, which the skip-path test
/// cannot reach).
#[tokio::test]
async fn retry_is_bounded_to_one_when_every_worker_is_dead() {
    let w1 = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let w2 = crate::common::mock_worker::MockWorker::start(vec![]).await;

    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: Default::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            tokenizer_shards: 1,
            tokenizer_backend: Default::default(),
            tokenizer_l1_cache_mb: 0,
            policy: PolicyKind::RoundRobin,
            // Unreachably-high threshold so both dead workers stay in the
            // candidate set for every request (`None` = DEFAULT breaker).
            circuit_breaker: Some(CircuitBreakerConfig {
                threshold: std::num::NonZeroU32::new(u32::MAX).unwrap(),
                cool_down_secs: 30,
            }),
            cache_aware: None,
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec![w1.url.clone(), w2.url.clone()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
        // Admission disabled → the retry load gate always admits, so the only
        // thing bounding attempts is the at-most-one-retry flag.
        admission: AdmissionConfig::default(),
        retry: RetryConfig { enabled: true },
    };

    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());

    let (event_rx, _disc) = spawn_discovery(&cfg).await.unwrap();
    let _mgr = tokio::spawn(manager::run_with_config(
        event_rx,
        registry.clone(),
        Some(Arc::new(cfg.clone())),
        None,
        None,
    ));

    let converged = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if registry.workers_for(&ModelId("tiny".into())).len() == 2 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(converged.is_ok(), "registry should contain both workers");

    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let ctx = Arc::new(AppContext::new(
        cfg,
        tokenizers,
        proxy,
        registry.clone(),
        policies,
    ));
    ctx.mark_ready();
    let app = build_router(ctx.clone());

    // Kill both workers and wait until both sockets refuse connections.
    for w in [w1, w2] {
        let url = w.url.clone();
        drop(w);
        let host_port_owned = url.trim_start_matches("http://").to_string();
        let down = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if tokio::net::TcpStream::connect(host_port_owned.as_str())
                    .await
                    .is_err()
                {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(down.is_ok(), "worker socket never went down");
    }

    const N: u64 = 4;
    for i in 0..N {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": format!("hi {i}")}],
        }))
        .unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        assert!(
            !res.status().is_success(),
            "request {i} cannot succeed — every worker is dead"
        );
    }

    let metrics = ctx.metrics.render();
    let count = |name: &str| -> u64 {
        metrics
            .lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0)
    };
    assert_eq!(
        count("sgl_router_retries_total{model_id=\"tiny\"} "),
        N,
        "each request must perform EXACTLY one retry — no more; metrics:\n{metrics}"
    );
    assert_eq!(
        count("sgl_router_retries_exhausted_total{model_id=\"tiny\"} "),
        N,
        "each retried-and-failed request must count as exhausted; metrics:\n{metrics}"
    );
}

/// A well-formed engine non-2xx is a RESPONSE, not a dispatch failure: it is
/// forwarded to the client verbatim and never retried, even with retry
/// enabled. Both workers answer 503, so if the retry logic (wrongly) treated
/// an engine error status as retryable, retries_total would be non-zero.
#[tokio::test]
async fn engine_error_status_is_forwarded_verbatim_and_never_retried() {
    use axum::http::StatusCode;

    let w1 = crate::common::mock_worker::MockWorker::start_returning_error(
        StatusCode::SERVICE_UNAVAILABLE,
        serde_json::json!({"error": {"message": "engine overloaded"}}),
    )
    .await;
    let w2 = crate::common::mock_worker::MockWorker::start_returning_error(
        StatusCode::SERVICE_UNAVAILABLE,
        serde_json::json!({"error": {"message": "engine overloaded"}}),
    )
    .await;

    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: Default::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            tokenizer_shards: 1,
            tokenizer_backend: Default::default(),
            tokenizer_l1_cache_mb: 0,
            policy: PolicyKind::RoundRobin,
            circuit_breaker: Some(CircuitBreakerConfig {
                threshold: std::num::NonZeroU32::new(u32::MAX).unwrap(),
                cool_down_secs: 30,
            }),
            cache_aware: None,
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec![w1.url.clone(), w2.url.clone()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
        admission: AdmissionConfig::default(),
        retry: RetryConfig { enabled: true },
    };

    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());

    let (event_rx, _disc) = spawn_discovery(&cfg).await.unwrap();
    let _mgr = tokio::spawn(manager::run_with_config(
        event_rx,
        registry.clone(),
        Some(Arc::new(cfg.clone())),
        None,
        None,
    ));

    let converged = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if registry.workers_for(&ModelId("tiny".into())).len() == 2 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(converged.is_ok(), "registry should contain both workers");

    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let ctx = Arc::new(AppContext::new(
        cfg,
        tokenizers,
        proxy,
        registry.clone(),
        policies,
    ));
    ctx.mark_ready();
    let app = build_router(ctx.clone());

    for i in 0..4 {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": format!("hi {i}")}],
        }))
        .unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            res.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "the engine's own status must be forwarded verbatim"
        );
    }

    let metrics = ctx.metrics.render();
    let retried = metrics
        .lines()
        .find_map(|l| l.strip_prefix("sgl_router_retries_total{model_id=\"tiny\"} "))
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(0);
    assert_eq!(
        retried, 0,
        "a well-formed engine non-2xx must never be retried; metrics:\n{metrics}"
    );
}
