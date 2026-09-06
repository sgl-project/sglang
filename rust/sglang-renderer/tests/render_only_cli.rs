// Keep the inner attribute off the first line so shebang lint does not misclassify it.
#![cfg(feature = "http")]

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use reqwest::StatusCode;
use serde_json::{Value, json};

struct ChildGuard(Child);

struct TestDirectory(std::path::PathBuf);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn available_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

async fn wait_until_ready(child: &mut Child, client: &reqwest::Client, health_url: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            panic!("render-only process exited during startup with {status}");
        }
        if client
            .get(health_url)
            .send()
            .await
            .is_ok_and(|response| response.status() == StatusCode::OK)
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "render-only process did not start"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

#[tokio::test]
async fn binary_starts_without_an_engine_and_serves_only_preprocessing() {
    let source_tokenizer = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../experimental/sgl-router/tests/fixtures/tiny_tokenizer.json");
    let model = TestDirectory(
        std::env::temp_dir().join(format!("sglang-render-only-{}", uuid::Uuid::new_v4())),
    );
    std::fs::create_dir(&model.0).unwrap();
    std::fs::copy(source_tokenizer, model.0.join("tokenizer.json")).unwrap();
    let port = available_port();
    let mut child = ChildGuard(
        Command::new(env!("CARGO_BIN_EXE_sglang-renderer"))
            .arg(&model.0)
            .arg("--tokenizer-path")
            .arg(&model.0)
            .arg("--served-model-name")
            .arg("model")
            .arg("--resolved-sampling-params")
            .arg("{}")
            .arg("--context-length")
            .arg("64")
            .arg("--vocab-size")
            .arg("512")
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let origin = format!("http://127.0.0.1:{port}");
    wait_until_ready(&mut child.0, &client, &format!("{origin}/health")).await;

    let tokenized = client
        .post(format!("{origin}/v1/tokenize"))
        .json(&json!({"prompt": "hello"}))
        .send()
        .await
        .unwrap();
    assert_eq!(tokenized.status(), StatusCode::OK);
    let tokenized: Value = tokenized.json().await.unwrap();
    assert!(tokenized["count"].as_u64().is_some_and(|count| count > 0));

    let completion = json!({"model": "model", "prompt": "hello"});
    let rendered = client
        .post(format!("{origin}/v1/completions/render"))
        .json(&completion)
        .send()
        .await
        .unwrap();
    assert_eq!(rendered.status(), StatusCode::OK);
    let rendered: Value = rendered.json().await.unwrap();
    assert!(
        rendered
            .as_array()
            .is_some_and(|requests| requests.len() == 1)
    );

    let inference = client
        .post(format!("{origin}/v1/completions"))
        .json(&completion)
        .send()
        .await
        .unwrap();
    assert_eq!(inference.status(), StatusCode::NOT_FOUND);
}
