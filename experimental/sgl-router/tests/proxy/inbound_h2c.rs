// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Inbound cleartext-HTTP/2 (h2c) listener tests for the router's own server.
//!
//! Where `h2c_forward.rs` proves the router's *outbound* client speaks h2c to
//! an HTTP/2-only worker, these prove the *inbound* side: the router's
//! `axum::serve` listener accepts an h2c prior-knowledge client on the same
//! cleartext port it serves HTTP/1.1 on, auto-negotiating per connection.
//!
//! Protocol negotiation happens at the connection layer, below the tower
//! service — so unlike the `oneshot` tests in `chat_routing.rs`, these must
//! drive a real socket via `axum::serve` (mirroring `main.rs`). The h2 path of
//! `axum::serve`'s hyper-util auto builder is gated behind axum's `http2` Cargo
//! feature: dropping it makes the listener HTTP/1.1-only and fails
//! `inbound_accepts_h2c_prior_knowledge`, while `inbound_still_accepts_http1`
//! stays green — so this pair pins both the capability and its backward
//! compatibility on the one cleartext port.

use std::sync::Arc;
use std::time::Duration;

use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::policies::factory::build_registry_with_defaults as build_policy_registry;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;

use axum::http::Version;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

fn build_ctx() -> Arc<AppContext> {
    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            tokenizer_shards: 1,
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
        admission: sgl_router::config::AdmissionConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(TEST_TIMEOUT).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

/// Spawn the real router (`build_router`) behind `axum::serve` on an ephemeral
/// port — the exact serve path used in `main.rs`. `/healthz` returns 200
/// unconditionally, so no worker is needed. Returns the base URL; the accept
/// loop is dropped when the test runtime shuts down.
async fn spawn_router() -> String {
    let ctx = build_ctx();
    ctx.mark_ready();
    let app = build_router(ctx);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn inbound_accepts_h2c_prior_knowledge() {
    let base = spawn_router().await;

    // `http2_prior_knowledge()` sends the HTTP/2 connection preface directly
    // over cleartext (no ALPN, no h1 upgrade) — the same way a service-mesh
    // sidecar dials h2c. The listener must recognize the preface and serve h2.
    let client = reqwest::Client::builder()
        .http2_prior_knowledge()
        .build()
        .unwrap();

    let resp = client
        .get(format!("{base}/healthz"))
        .send()
        .await
        .expect("h2c prior-knowledge client must reach the router listener");

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.version(),
        Version::HTTP_2,
        "listener must negotiate HTTP/2 for an h2c prior-knowledge client",
    );
}

#[tokio::test]
async fn inbound_still_accepts_http1() {
    let base = spawn_router().await;

    // The default reqwest client speaks HTTP/1.1 over cleartext. Enabling h2c
    // must not break existing HTTP/1.1 callers (load balancers, probes, curl)
    // — the auto builder serves both on the same port.
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base}/healthz"))
        .send()
        .await
        .expect("HTTP/1.1 client must still reach the router listener");

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.version(),
        Version::HTTP_11,
        "an HTTP/1.1 client must keep negotiating HTTP/1.1 on the same port",
    );
}
