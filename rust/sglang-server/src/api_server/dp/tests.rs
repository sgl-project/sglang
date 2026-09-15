use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::extract::Query;
use axum::http::HeaderMap;
use axum::routing::get;
use bytes::Bytes;
use futures::future;
use tokio::sync::mpsc;

use super::*;

fn listener() -> TcpListener {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    listener
}

struct Active(Arc<AtomicUsize>);

impl Drop for Active {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn public_dp_ingress_preserves_batches_routing_streams_loads_and_cancellation() {
    let (received, mut requests) = mpsc::unbounded_channel();
    let active = Arc::new(AtomicUsize::new(0));
    let mut workers = Vec::new();
    let mut tasks = Vec::new();
    for rank in 0..2 {
        let generate = {
            let received = received.clone();
            let active = active.clone();
            move |headers: HeaderMap, Json(body): Json<Value>| {
                let received = received.clone();
                let active = active.clone();
                async move {
                    let intake: crate::message::request::GenerateBody =
                        serde_json::from_value(body.clone()).unwrap();
                    assert_eq!(intake.into_requests().unwrap().0.len(), 1);
                    received.send((rank, body.clone(), headers)).unwrap();
                    let output = json!({
                        "text": "α", "output_ids": body["input_ids"],
                        "meta_info": { "id": body["rid"], "dp_rank": rank }
                    });
                    if body["stream"] != true {
                        return Json(output).into_response();
                    }
                    let held = body["input_ids"][0] == 99;
                    let stream = async_stream::stream! {
                        active.fetch_add(1, Ordering::SeqCst);
                        let _guard = Active(active);
                        let frame = format!("data: {output}\n\n");
                        yield Ok::<_, io::Error>(Bytes::copy_from_slice(&frame.as_bytes()[..11]));
                        tokio::time::sleep(Duration::from_millis(5)).await;
                        yield Ok(Bytes::copy_from_slice(&frame.as_bytes()[11..]));
                        if held { future::pending::<()>().await; }
                        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
                    };
                    (
                        [("content-type", "text/event-stream")],
                        Body::from_stream(stream),
                    )
                        .into_response()
                }
            }
        };
        let loads = move |Query(query): Query<HashMap<String, String>>| async move {
            assert_ne!(query.get("format").map(String::as_str), Some("prometheus"));
            let rows = if query
                .get("dp_rank")
                .is_none_or(|wanted| wanted == &rank.to_string())
            {
                vec![
                    json!({"dp_rank":rank,"timestamp":1.0,"num_running_reqs":rank,"num_waiting_reqs":0,"num_used_tokens":rank,"num_total_tokens":rank}),
                ]
            } else {
                vec![]
            };
            Json(json!({"version":"test","accelerator":"GPU","num_accelerators":1,"loads":rows}))
        };
        let app = Router::new()
            .route(
                "/generate",
                get(|| async { StatusCode::METHOD_NOT_ALLOWED })
                    .post(generate.clone())
                    .put(generate),
            )
            .route("/v1/loads", get(loads))
            .route(
                "/get_load",
                get(move || async move { Json(json!([{"dp_rank": rank}])) }),
            )
            .route(
                "/health_generate",
                get(|| async { ([("x-router-dp-size", "2")], "OK") }),
            )
            .route(
                "/metrics",
                get(move || async move { format!("rank_{rank} 1\n") }),
            )
            .route(
                "/abort_request",
                axum::routing::post(|Json(body): Json<Value>| async move {
                    (
                        [(
                            super::super::common::ABORT_DISPATCHED_HEADER,
                            if body["rid"] == "present" || body["abort_all"] == true {
                                "1"
                            } else {
                                "0"
                            },
                        )],
                        StatusCode::OK,
                    )
                }),
            );
        let listener = listener();
        workers.push((rank, format!("http://{}", listener.local_addr().unwrap())));
        tasks.push(tokio::spawn(serve_listener(listener, app, None)));
    }
    let metrics = crate::metrics::FrontendMetrics::new(&crate::ServerArgs::default()).unwrap();
    let ingress = DpIngress::start(
        listener(),
        workers,
        LoadBalanceMethod::RoundRobin,
        true,
        true,
        Some(json!({"temperature":0.25})),
        Some(Http2Settings {
            max_concurrent_streams: 17,
            initial_connection_window_size: 1 << 20,
        }),
        Some(metrics.clone()),
    )
    .unwrap();
    let url = format!("http://{}", ingress.address);
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let post = |body: Value| {
        client
            .post(format!("{url}/generate"))
            .header("content-type", "application/json")
            .body(body.to_string())
    };

    let received_time = crate::metrics::monotonic_seconds().unwrap() - 2.;
    let response = post(json!({
        "input_ids":[[11],[12],[13]],"rid":"body","bootstrap_room":1,
        "extra_key":["tenant-a","tenant-b",""], "cache_salt":"namespace",
        "routing_key":"session",
        "received_time": received_time,
        "image_data":[["/a"],["/b"],["/c"]],
        "multimodal_placeholders":vec![json!([{"type":"image","token_index":0,"item_index":0}]);3],
        "sampling_params":{"max_new_tokens":3}
    }))
    .header("x-override-rid", "router")
    .header("x-override-bootstrap-room", "1000")
    .header("connection", "x-hop, x-sglang-received-timing")
    .header("x-hop", "removed")
    .header("x-sglang-received-timing", "untrusted")
    .send()
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let output: Value = serde_json::from_str(&response.text().await.unwrap()).unwrap();
    let families = metrics.registry.gather();
    let http = families
        .iter()
        .find(|family| family.get_name() == "sglang:http_requests_total")
        .unwrap();
    assert_eq!(
        http.get_metric()[0].get_counter().get_value(),
        1.,
        "one public batch, three worker calls"
    );
    for (index, rank) in [0, 1, 0].into_iter().enumerate() {
        assert_eq!(output[index]["meta_info"]["id"], format!("router_{index}"));
        assert_eq!(output[index]["meta_info"]["dp_rank"], rank);
    }
    for _ in 0..3 {
        let (_, body, headers) = requests.recv().await.unwrap();
        let index = body["input_ids"][0].as_i64().unwrap() - 11;
        assert_eq!(body["bootstrap_room"], 1000 + index);
        assert_eq!(body["sampling_params"]["temperature"], 0.25);
        assert_eq!(body["sampling_params"]["max_new_tokens"], 3);
        assert_eq!(
            body["extra_key"],
            json!(["tenant-a", "tenant-b", null])[index as usize]
        );
        assert_eq!(body["cache_salt"], "namespace");
        assert_eq!(body["routing_key"], "session");
        assert_eq!(body["received_time"], received_time);
        let timing: Value = serde_json::from_slice(
            headers
                .get("x-sglang-received-timing")
                .expect("DP forwarding must preserve request age across host clock origins")
                .as_bytes(),
        )
        .unwrap();
        let age = timing["age"].as_f64().unwrap();
        assert!(age >= 2.);
        assert!(age <= crate::metrics::monotonic_seconds().unwrap() - received_time);
        assert_eq!(timing["created_is_positive"], true);
        assert_eq!(body["multimodal_placeholders"][0]["token_index"], 0);
        assert!(!headers.contains_key("x-override-rid"));
        assert!(!headers.contains_key("x-hop"));
    }
    let response = post(json!({"input_ids":[7],"data_parallel_rank":0,"routed_dp_rank":0}))
        .header("x-override-routed-dp-rank", "1")
        .header("x-sglang-received-timing", "untrusted")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let (rank, _, headers) = requests.recv().await.unwrap();
    assert_eq!(rank, 1);
    assert!(!headers.contains_key("x-sglang-received-timing"));
    let response = post(json!({"input_ids":[7],"data_parallel_rank":2}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(requests.try_recv().is_err());

    let response = post(json!({"input_ids":[[21],[22]],"stream":true,"rid":"stream"}))
        .send()
        .await
        .unwrap();
    let output = response.text().await.unwrap();
    assert_eq!(output.matches("data: [DONE]").count(), 1);
    let frames: Vec<Value> = output
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|line| *line != "[DONE]")
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(frames.len(), 2);
    for frame in frames {
        let index = frame["index"].as_u64().unwrap();
        assert_eq!(frame["output_ids"][0], 21 + index);
        assert_eq!(frame["meta_info"]["id"], format!("stream_{index}"));
        assert_eq!(frame["text"], "α");
    }
    let response = client
        .get(format!("{url}/v1/loads?include=core&dp_rank=1"))
        .send()
        .await
        .unwrap();
    let output: Value = serde_json::from_str(&response.text().await.unwrap()).unwrap();
    assert_eq!(output["num_accelerators"], 2);
    assert_eq!(output["loads"].as_array().unwrap().len(), 1);
    assert_eq!(output["loads"][0]["dp_rank"], 1);
    let output = client
        .get(format!("{url}/v1/loads?format=prometheus"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(output.matches("# TYPE sglang_num_running_reqs").count(), 1);
    assert!(output.contains("dp_rank=\"0\""));
    assert!(output.contains("dp_rank=\"1\""));
    let output = client
        .get(format!("{url}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(output, "rank_0 1\n");

    let http2 = reqwest::Client::builder()
        .no_proxy()
        .http2_prior_knowledge()
        .build()
        .unwrap();
    let response = http2
        .get(format!("{url}/health_generate"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.version(), reqwest::Version::HTTP_2);
    assert_eq!(response.headers()["x-router-dp-size"], "2");

    let mut response = post(json!({"input_ids":[[99],[99]],"stream":true}))
        .send()
        .await
        .unwrap();
    assert!(!response.chunk().await.unwrap().unwrap().is_empty());
    drop(response);
    tokio::time::timeout(Duration::from_secs(2), async {
        while active.load(Ordering::SeqCst) > 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("disconnect must cancel upstream streams");
    while requests.try_recv().is_ok() {}
    for method in [reqwest::Method::POST, reqwest::Method::PUT] {
        let response = client
            .request(method, format!("{url}/generate"))
            .header("content-type", "application/json")
            .header("x-body-compressed", "zstd")
            .body(super::super::decompression::tests::python_zstd_request())
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body = response.text().await.unwrap();
        assert_eq!(status, StatusCode::OK, "{body}");
        let (_, request, headers) = requests.recv().await.unwrap();
        assert_eq!(request["input_ids"], json!([11]));
        assert!(!headers.contains_key("x-body-compressed"));
    }
    for body in [
        json!({"rid":"present"}),
        json!({"rid":"absent"}),
        json!({"abort_all":true}),
    ] {
        let response = client
            .post(format!("{url}/abort_request"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            !response
                .headers()
                .contains_key(super::super::common::ABORT_DISPATCHED_HEADER)
        );
        assert!(response.text().await.unwrap().is_empty());
    }
    let text = client
        .get(format!("{url}/metrics/native"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        text.contains(
            "sglang:num_aborted_requests_total{engine_type=\"unified\",model_name=\"\"} 2"
        ),
        "{text}"
    );
    assert!(!text.contains("endpoint=\"/metrics/native\""));
    ingress.stop();
    for family in metrics.registry.gather() {
        if family.get_name() == "sglang:http_requests_active"
            || family.get_name() == "sglang:routing_keys_active"
        {
            assert!(
                family
                    .get_metric()
                    .iter()
                    .all(|metric| metric.get_gauge().get_value() == 0.)
            );
        }
    }
    assert!(
        client
            .get(format!("{url}/health_generate"))
            .send()
            .await
            .is_err()
    );
    for task in tasks {
        task.abort();
    }
}

use std::io;
