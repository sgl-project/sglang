// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Exercise the real Hub client against loopback. Each environment configuration
//! runs in a subprocess so changing credentials cannot race Tokio/server threads.

use axum::extract::State;
use axum::http::{HeaderMap, Uri};
use axum::response::{IntoResponse, Response};
use sgl_router::tokenizer::adapter::{load, ModelFiles};
use std::process::Command;
use std::sync::{mpsc, Arc, Mutex};

type Requests = Arc<Mutex<Vec<(String, Option<String>)>>>;

async fn hub(State(seen): State<Requests>, uri: Uri, headers: HeaderMap) -> Response {
    let auth = headers
        .get("authorization")
        .map(|v| v.to_str().unwrap().to_owned());
    seen.lock().unwrap().push((uri.path().to_owned(), auth));
    if uri.path().starts_with("/api/models/") {
        return axum::Json(serde_json::json!({
            "id": "test/model",
            "siblings": [{"rfilename": "tokenizer.json"}, {"rfilename": "tokenizer_config.json"}]
        }))
        .into_response();
    }
    let body: &'static [u8] = if uri.path().ends_with("/tokenizer.json") {
        include_bytes!("fixtures/tiny_tokenizer.json")
    } else {
        b"{}"
    };
    (
        [
            (
                "etag",
                format!("\"{}\"", uri.path().rsplit('/').next().unwrap()),
            ),
            (
                "x-repo-commit",
                "0123456789012345678901234567890123456789".to_owned(),
            ),
            ("content-length", body.len().to_string()),
        ],
        body,
    )
        .into_response()
}

fn exercise_downloads() {
    // The production router also loads tokenizers inside a Tokio runtime.
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        if std::env::var("HF_HUB_TEST_CASE").unwrap() == "invalid" {
            let error = load("test/model")
                .err()
                .expect("invalid token was accepted");
            let message = format!("{error:#}");
            assert!(message.contains("HF_TOKEN"));
            assert!(!message.contains("secret"));
            return;
        }
        load("test/model").unwrap();
        load("test/model").unwrap(); // A cache hit must not contact the Hub again.
        let files = ModelFiles::open("test/model");
        assert_eq!(
            files.text("tokenizer_config.json").unwrap().as_deref(),
            Some("{}")
        );
        assert_eq!(
            files.text("tokenizer_config.json").unwrap().as_deref(),
            Some("{}")
        );
        assert!(files.text("absent.json").unwrap().is_none());
    });
}

#[test]
fn hub_auth_downloads_and_cache() {
    if std::env::var_os("HF_HUB_TEST_CASE").is_some() {
        exercise_downloads();
        return;
    }
    let seen = Requests::default();
    let server_seen = Arc::clone(&seen);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(listener.local_addr().unwrap()).unwrap();
            axum::serve(
                listener,
                axum::Router::new().fallback(hub).with_state(server_seen),
            )
            .await
            .unwrap();
        });
    });
    let endpoint = format!("http://{}", rx.recv().unwrap());
    for (case, token, use_path, expected) in [
        ("env", Some("  hf_env\n"), true, Some("Bearer hf_env")),
        ("path", None, true, Some("Bearer hf_path")),
        ("home", None, false, Some("Bearer hf_home")),
        ("empty", Some(""), false, Some("Bearer hf_home")),
        ("anonymous", None, false, None),
        ("disabled", Some("hf_secret\ninvalid"), true, None),
        ("invalid", Some("hf_secret\ninvalid"), true, None),
    ] {
        let home = tempfile::tempdir().unwrap();
        if case != "anonymous" {
            std::fs::write(home.path().join("token"), "hf_home\n").unwrap();
        }
        let token_path = home.path().join("custom-token");
        std::fs::write(&token_path, "hf_path\n").unwrap();
        let mut cmd = Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", "hub_auth_downloads_and_cache", "--nocapture"]);
        for key in [
            "HF_TOKEN",
            "HF_TOKEN_PATH",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN",
            "HF_HUB_OFFLINE",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ] {
            cmd.env_remove(key);
        }
        cmd.env("HF_HUB_TEST_CASE", case)
            .env("HF_HOME", home.path())
            .env("HF_ENDPOINT", &endpoint);
        if let Some(token) = token {
            cmd.env("HF_TOKEN", token);
        }
        if use_path {
            cmd.env("HF_TOKEN_PATH", token_path);
        }
        if case == "disabled" {
            cmd.env("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1");
        }
        seen.lock().unwrap().clear();
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{case}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let requests = seen.lock().unwrap();
        // HEAD + GET per file and one listing. Repeated reads and absent files
        // must cause no extra requests, and malformed credentials must fail locally.
        assert_eq!(
            requests.len(),
            if case == "invalid" { 0 } else { 5 },
            "{case}: {requests:?}"
        );
        assert!(
            requests.iter().all(|(_, auth)| auth.as_deref() == expected),
            "{case}: {requests:?}"
        );
    }
}
