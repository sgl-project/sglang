// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end checks that the right token reaches the HuggingFace request.
//!
//! Own test binary, and deliberately ONE `#[test]`: these mutate process-wide
//! environment (`HF_HOME`, `HF_ENDPOINT`, `HF_TOKEN`). A second `#[test]` here
//! would run on another thread of the SAME process and race them. Keep the
//! assertions sequential inside this one test.
//!
//! The hub is mocked on loopback via `HF_ENDPOINT`, so no network and no
//! credentials are needed.

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, Uri};
use std::net::SocketAddr;
use std::sync::{mpsc, Arc, Mutex};

/// Every request the mock hub saw, as `(uri, authorization header)`.
type SeenRequests = Arc<Mutex<Vec<(String, Option<String>)>>>;

async fn capture(State(seen): State<SeenRequests>, uri: Uri, headers: HeaderMap) -> StatusCode {
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    seen.lock().unwrap().push((uri.to_string(), auth));
    // 404 every path: we assert on what went out, not on a successful
    // download, so the mock never has to serve a real tokenizer blob.
    StatusCode::NOT_FOUND
}

/// Start a loopback stand-in for huggingface.co and return its base URL.
/// The server thread is detached; it dies with the test process.
fn start_mock_hub(seen: SeenRequests) -> String {
    let (tx, rx) = mpsc::channel::<SocketAddr>();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(listener.local_addr().unwrap()).unwrap();
            let app = axum::Router::new().fallback(capture).with_state(seen);
            axum::serve(listener, app).await.unwrap();
        });
    });
    format!("http://{}", rx.recv().unwrap())
}

fn auths(seen: &SeenRequests) -> Vec<Option<String>> {
    seen.lock()
        .unwrap()
        .iter()
        .map(|(_, a)| a.clone())
        .collect()
}

#[test]
fn hf_token_reaches_the_hub_without_clobbering_the_token_file() {
    // hf-hub's agent honors ALL_PROXY / HTTPS_PROXY / HTTP_PROXY
    // (`try_proxy_from_env(true)`). An ambient proxy would divert the loopback
    // request and fail this test with a misleading "endpoint not honored".
    for proxy in [
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
    ] {
        std::env::remove_var(proxy);
    }

    // Empty HF_HOME: no `$HF_HOME/token` file yet, so a token on the wire in
    // part 1 can ONLY have come from HF_TOKEN. Set before the server thread
    // starts — `setenv` is not thread-safe, so keep the window narrow.
    let home = tempfile::tempdir().unwrap();
    std::env::set_var("HF_HOME", home.path());
    // Surrounding whitespace is what a secret mount can leave behind; it must
    // be trimmed off before it reaches the header.
    std::env::set_var("HF_TOKEN", "  hf_test_token\n");

    let seen: SeenRequests = Arc::new(Mutex::new(Vec::new()));
    let endpoint = start_mock_hub(Arc::clone(&seen));
    std::env::set_var("HF_ENDPOINT", &endpoint);

    // --- Part 1: HF_TOKEN is honored at all. -----------------------------
    // Regression: hf-hub 0.4's `ApiBuilder::from_env()` reads HF_HOME and
    // HF_ENDPOINT but NOT HF_TOKEN, so without an explicit read a gated repo
    // 401s with the token correctly exported.
    //
    // The mock 404s, so the load fails; the assertion is on the request.
    let _ = sgl_router::tokenizer::adapter::load("fake-org/fake-repo");

    let got = auths(&seen);
    assert!(
        !got.is_empty(),
        "no request reached the mock hub — either HF_ENDPOINT was not honored, \
         or the Authorization value was rejected locally by ureq's header validation",
    );
    // Every request, not just the first: hf-hub may issue metadata and blob
    // requests, and all of them must be authenticated.
    assert!(
        got.iter()
            .all(|a| a.as_deref() == Some("Bearer hf_test_token")),
        "HF_TOKEN must be sent as a trimmed bearer token on every hub request; got: {got:?}",
    );

    // --- Part 2: an unset HF_TOKEN must NOT clobber the token file. ------
    // This is the load-bearing half. `with_token` OVERWRITES, so collapsing
    // the caller's `if let` into `with_token(hf_token_override(..))` type-checks,
    // reads cleaner, and silently drops the file-based token on every
    // `huggingface-cli login` deployment. Every unit test still passes under
    // that refactor; only this assertion catches it.
    std::fs::write(home.path().join("token"), "hf_file_token\n").unwrap();
    std::env::remove_var("HF_TOKEN");
    seen.lock().unwrap().clear();

    let _ = sgl_router::tokenizer::adapter::load("fake-org/fake-repo-two");

    let got = auths(&seen);
    assert!(!got.is_empty(), "no request reached the mock hub in part 2");
    assert!(
        got.iter()
            .all(|a| a.as_deref() == Some("Bearer hf_file_token")),
        "with HF_TOKEN unset, hf-hub's $HF_HOME/token value must survive; got: {got:?}",
    );
}
