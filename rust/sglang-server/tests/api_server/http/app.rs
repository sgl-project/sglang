//! The HTTP listener's behaviour that only a served app shows: which
//! listener serves what, body limits and decompression, an embedding
//! package's layer, and how a media processor's refusal is answered.

use std::sync::Arc;

use sglang_server::{
    DisaggregationMode, HttpLayer, MediaError, MetricsConfig, MmData, MmProcessOutput, MmProcessor,
    TokenIds,
};
use sglang_server_test::{ApiKeys, SchedulerlessServer, free_loopback_addrs};

use crate::common::{push_terminal, take_startup_freeze_gc};

/// Only the base listener serves the node's `/metrics`; a DP rank's
/// offset listener answers 404, as it would without metrics.
#[test]
fn only_the_metrics_listener_serves_metrics() {
    use std::io::{Read, Write};

    let dir = std::env::temp_dir().join(format!("runtime-metrics-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (metrics_listener, expected) in [(true, "200"), (false, "404")] {
        let [addr] = free_loopback_addrs();
        let metrics = MetricsConfig {
            multiproc_dir: dir.to_string_lossy().into_owned(),
            labels: Default::default(),
            priority_label: false,
            bucket_time_to_first_token: None,
            bucket_inter_token_latency: None,
            bucket_e2e_request_latency: None,
            bucket_prompt_tokens: None,
            bucket_generation_tokens: None,
        };
        let rt = SchedulerlessServer::builder(addr)
            .rust_server_args(|args| args.metrics_listener = metrics_listener)
            .server_args(|args| args.metrics = Some(metrics))
            .start()
            .expect("start runtime");
        let mut conn = std::net::TcpStream::connect(addr).expect("connect");
        conn.write_all(b"GET /metrics HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut response = String::new();
        conn.read_to_string(&mut response).unwrap();
        rt.shutdown();
        let status = response.split(' ').nth(1).unwrap_or_default();
        assert_eq!(status, expected, "metrics_listener={metrics_listener}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Regression: a >2MB body must reach the JSON layer and fail on its
/// *content* (unknown field → 4xx), never on size (413).
#[test]
fn accepts_multi_megabyte_generate_body() {
    use std::io::{Read, Write};

    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .start()
        .expect("start runtime");

    // ~3MB of input_ids plus a `text`, which is mutually exclusive with them:
    // the body parses in full and is then rejected by `into_requests` with a
    // 400, proving it got past any size limit (a 413 would fire before
    // parsing). The rejection must come from OUR validation, not from serde —
    // an unknown field used to serve here, but unknown fields are now ignored
    // to match Python, so such a body would be accepted, dispatched to a ring
    // nobody drains in this test, and hang the connection.
    let ids = "1,".repeat(1_500_000);
    let body = format!(
        r#"{{"input_ids":[{}1],"text":"x","sampling_params":{{"max_new_tokens":1}}}}"#,
        ids
    );
    assert!(body.len() > 2 * 1024 * 1024, "test body must exceed 2MB");

    let mut conn = std::net::TcpStream::connect(addr).expect("connect");
    let req = format!(
        "POST /generate HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    conn.write_all(req.as_bytes()).unwrap();
    conn.flush().unwrap();

    let mut response = String::new();
    conn.read_to_string(&mut response).unwrap();
    let status_line = response.lines().next().unwrap_or("");
    let code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    // A 400 from the mutually-exclusive-inputs check proves the body was read
    // and parsed in full; 413 would mean it was rejected on size beforehand.
    assert!(
        (400..500).contains(&code) && code != 413,
        "expected a JSON-layer 4xx (not 413), got: {status_line}"
    );

    rt.shutdown();
}

/// Media the processor refuses fails `/generate` as the client's error with
/// the processor's message, as Python answers its processor's `ValueError`:
/// a 400, or for a stream an error event with code 400. The processor's own
/// failure (Python's `RuntimeError`, e.g. an allocation failure) stays a 500.
#[test]
fn refused_media_is_a_bad_request() {
    use std::io::{Read, Write};

    struct FailingProcessor(MediaError);
    impl MmProcessor for FailingProcessor {
        fn process(
            &self,
            _input_ids: TokenIds,
            _mm: MmData,
        ) -> Result<MmProcessOutput, MediaError> {
            Err(self.0.clone())
        }
    }

    for (error, status) in [
        (
            MediaError::Client("image decode failed: CRC error".into()),
            400,
        ),
        (
            MediaError::Server("audio decode: Cannot allocate memory".into()),
            500,
        ),
    ] {
        let message = error.to_string();
        let [addr] = free_loopback_addrs();
        let rt = SchedulerlessServer::builder(addr)
            .server_args(|args| args.model_config.is_multimodal = true)
            .start()
            .expect("start runtime");
        rt.start_mm_workers(Arc::new(FailingProcessor(error)), 1);
        for stream in [false, true] {
            let body = format!(
                r#"{{"input_ids":[1,2,3],"image_data":"data:image/png;base64,AAAA","stream":{stream}}}"#
            );
            let mut conn = std::net::TcpStream::connect(addr).expect("connect");
            conn.write_all(
                format!(
                    "POST /generate HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .unwrap();
            let mut response = String::new();
            conn.read_to_string(&mut response).unwrap();
            let http_status = response.split(' ').nth(1).unwrap_or_default();
            let expected = if stream { "200" } else { &status.to_string() };
            assert_eq!(http_status, expected, "{response}");
            assert!(
                response.contains(&format!(r#""message":"{message}""#))
                    && response.contains(&format!(r#""code":{status}"#)),
                "{response}"
            );
        }
        rt.shutdown();
    }
}

/// An embedding package's `http_layer` wraps the served app: its own route
/// serves, and its middleware sees the built-in routes' responses. With an
/// API key, its route needs the key as the built-in ones do, and so does
/// an unknown path; the PD bootstrap registry of a prefill server stays
/// open, as Python serves it on its own server.
#[test]
fn http_layer_wraps_the_served_app() {
    use std::io::{Read, Write};

    let [addr] = free_loopback_addrs();
    let layer: HttpLayer = Arc::new(|app: axum::Router| {
        app.route("/embedder", axum::routing::get(|| async { "embedded" }))
            .layer(axum::middleware::map_response(
                |mut response: axum::response::Response| async move {
                    response
                        .headers_mut()
                        .insert("x-embedder", axum::http::HeaderValue::from_static("1"));
                    response
                },
            ))
    });
    let rt = SchedulerlessServer::builder(addr)
        .server_args(|args| {
            args.api_keys = ApiKeys::new(Some("sk-test".into()), None);
            args.disaggregation_mode = DisaggregationMode::Prefill;
        })
        .http_layer(layer)
        .start()
        .expect("start runtime");
    let get = |path: &str, key: Option<&str>| {
        let mut conn = std::net::TcpStream::connect(addr).expect("connect");
        let authorization = key.map_or(String::new(), |key| {
            format!("Authorization: Bearer {key}\r\n")
        });
        write!(
            conn,
            "GET {path} HTTP/1.1\r\nHost: t\r\n{authorization}Connection: close\r\n\r\n"
        )
        .unwrap();
        let mut response = String::new();
        conn.read_to_string(&mut response).unwrap();
        response
    };
    let key = Some("sk-test");
    let model_info = get("/model_info", key);
    assert!(model_info.starts_with("HTTP/1.1 200"), "{model_info}");
    assert!(model_info.contains("x-embedder: 1"), "{model_info}");
    let embedded = get("/embedder", key);
    assert!(embedded.starts_with("HTTP/1.1 200"), "{embedded}");
    assert!(embedded.ends_with("\r\n\r\nembedded"), "{embedded}");

    for path in ["/model_info", "/embedder", "/no_such_route"] {
        let response = get(path, None);
        assert!(response.starts_with("HTTP/1.1 401"), "{path}: {response}");
    }
    let unknown = get("/no_such_route", key);
    assert!(unknown.starts_with("HTTP/1.1 404"), "{unknown}");
    let bootstrap = get("/route", None);
    assert!(!bootstrap.starts_with("HTTP/1.1 401"), "{bootstrap}");

    rt.shutdown();
}

/// `SGLANG_ENABLE_REQUEST_DECOMPRESSION` answers as Python's middleware
/// does. A body tagged `x-body-compressed: zstd` reaches an embedding
/// package's layer and every route decompressed, with the header dropped
/// and the content length fixed, and without a size cap; multiple frames
/// decode as one body, and an empty body stays empty. An unknown method, a corrupt payload or a frame
/// needing a window over libzstd's 128 MiB default is a plain-text 400
/// that reaches neither the layer nor the scheduler. Turned off, the
/// header is ignored.
#[test]
fn compressed_request_bodies_match_python_decompression() {
    use std::io::Write;
    use std::sync::Mutex;
    use std::time::Duration;

    fn http_request(
        addr: std::net::SocketAddr,
        path: &str,
        compression: Option<&str>,
        body: &[u8],
    ) -> String {
        let mut conn = std::net::TcpStream::connect(addr).expect("connect");
        conn.set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let compression = compression
            .map(|method| format!("x-body-compressed: {method}\r\n"))
            .unwrap_or_default();
        let head = format!(
            "POST {path} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\n\
             {compression}Content-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let response = crate::common::exchange(&mut conn, &[head.as_bytes(), body].concat());
        String::from_utf8(response).unwrap()
    }
    // The status line and body: the `date` header can differ between two
    // otherwise identical responses.
    fn status_and_body(response: &str) -> (&str, &str) {
        let (head, body) = response.split_once("\r\n\r\n").expect("response head");
        (head.lines().next().unwrap_or_default(), body)
    }
    fn zstd_frames(body: &[u8], frames: usize, window_log: u32) -> Vec<u8> {
        body.chunks(body.len().div_ceil(frames))
            .flat_map(|chunk| {
                let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 3).unwrap();
                encoder
                    .set_parameter(zstd::zstd_safe::CParameter::WindowLog(window_log))
                    .unwrap();
                encoder.write_all(chunk).unwrap();
                encoder.finish().unwrap()
            })
            .collect()
    }
    let zstd = |body: &[u8]| zstd_frames(body, 1, 20);

    // What the embedding package's layer read: path, whether the
    // compression header was still set, content length, body.
    type Seen = (String, bool, Option<String>, Vec<u8>);
    let seen = Arc::new(Mutex::new(Vec::<Seen>::new()));
    let recorder = seen.clone();
    let layer: HttpLayer = Arc::new(move |app: axum::Router| {
        let recorder = recorder.clone();
        app.layer(axum::middleware::from_fn(
            move |request: axum::extract::Request, next: axum::middleware::Next| {
                let recorder = recorder.clone();
                async move {
                    let (parts, body) = request.into_parts();
                    let body = axum::body::to_bytes(body, usize::MAX).await.unwrap();
                    recorder.lock().unwrap().push((
                        parts.uri.path().to_owned(),
                        parts.headers.contains_key("x-body-compressed"),
                        parts
                            .headers
                            .get("content-length")
                            .map(|value| value.to_str().unwrap().to_owned()),
                        body.to_vec(),
                    ));
                    let request =
                        axum::extract::Request::from_parts(parts, axum::body::Body::from(body));
                    next.run(request).await
                }
            },
        ))
    });
    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .server_args(|args| args.enable_request_decompression = true)
        .http_layer(layer)
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);

    let generate =
        br#"{"input_ids":[1,2,3],"stream":false,"sampling_params":{"max_new_tokens":8}}"#;
    let compressed = zstd(generate);
    let request =
        std::thread::spawn(move || http_request(addr, "/generate", Some("zstd"), &compressed));
    let request_on_ring = rt
        .next_request(Duration::from_secs(5))
        .expect("the compressed request did not reach the scheduler ring");
    assert_eq!(request_on_ring.buffer_shape("input_ids"), Some(&[3][..]));
    let rid = request_on_ring.rid().expect("request rid");
    assert!(push_terminal(&rt, &rid, &[9, 10]));
    let response = request.join().unwrap();
    let (status, body) = status_and_body(&response);
    assert_eq!(status, "HTTP/1.1 200 OK", "{response}");
    assert!(body.contains(r#""output_ids":[9,10]"#), "{response}");
    assert_eq!(
        seen.lock().unwrap().pop(),
        Some((
            "/generate".to_owned(),
            false,
            Some(generate.len().to_string()),
            generate.to_vec()
        ))
    );

    // Each route answers a compressed body as it answers the plain one;
    // the padded body inflates ~16 MiB from a few KiB.
    let padded = format!(r#"{{"input_ids":[1],"text":"x"{}}}"#, " ".repeat(16 << 20));
    for (path, plain) in [
        ("/generate", br#"{"input_ids":[1],"text":"x"}"#.to_vec()),
        ("/generate", padded.into_bytes()),
        ("/v1/completions", b"{}".to_vec()),
        ("/v1/chat/completions", b"{}".to_vec()),
    ] {
        let expected = http_request(addr, path, None, &plain);
        assert!(
            status_and_body(&expected).0.starts_with("HTTP/1.1 400"),
            "{expected}"
        );
        for compressed in [zstd(&plain), zstd_frames(&plain, 2, 20)] {
            let response = http_request(addr, path, Some("zstd"), &compressed);
            assert_eq!(status_and_body(&response), status_and_body(&expected));
            let (seen_path, still_compressed, _, seen_body) = seen.lock().unwrap().pop().unwrap();
            assert_eq!(seen_path, path);
            assert!(!still_compressed);
            assert!(seen_body == plain, "{path}: the layer saw another body");
        }
    }
    let empty = http_request(addr, "/generate", None, b"");
    assert!(
        status_and_body(&empty).0.starts_with("HTTP/1.1 400"),
        "{empty}"
    );
    let tagged_empty = http_request(addr, "/generate", Some("zstd"), b"");
    assert_eq!(status_and_body(&tagged_empty), status_and_body(&empty));

    seen.lock().unwrap().clear();
    let unsupported =
        |method: &str| format!("unsupported x-body-compressed '{method}'; supported: ['zstd']");
    for (method, body, message) in [
        ("gzip", zstd(generate), unsupported("gzip")),
        ("ZSTD", zstd(generate), unsupported("ZSTD")),
        (
            "zstd",
            b"not-zstd-data".to_vec(),
            "decompress failed".to_owned(),
        ),
        (
            "zstd",
            [zstd(generate), b"garbage".to_vec()].concat(),
            "decompress failed".to_owned(),
        ),
        (
            "zstd",
            zstd_frames(generate, 1, 28),
            "decompress failed".to_owned(),
        ),
    ] {
        let response = http_request(addr, "/generate", Some(method), &body);
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        assert!(head.starts_with("HTTP/1.1 400"), "{response}");
        assert!(
            !head.to_ascii_lowercase().contains("content-type"),
            "{response}"
        );
        assert_eq!(body, message);
    }
    assert!(seen.lock().unwrap().is_empty());
    assert!(rt.next_request(Duration::from_millis(200)).is_none());
    rt.shutdown();

    let [addr] = free_loopback_addrs();
    let rt = SchedulerlessServer::builder(addr)
        .start()
        .expect("start runtime");
    take_startup_freeze_gc(&rt);
    let compressed = zstd(generate);
    let tagged = http_request(addr, "/generate", Some("zstd"), &compressed);
    let untagged = http_request(addr, "/generate", None, &compressed);
    assert!(
        status_and_body(&untagged).0.starts_with("HTTP/1.1 400"),
        "{untagged}"
    );
    assert_eq!(status_and_body(&tagged), status_and_body(&untagged));
    assert!(rt.next_request(Duration::from_millis(200)).is_none());
    rt.shutdown();
}
