use super::*;
use axum::{extract::State, http::HeaderMap, routing::post, Json, Router};
use serde_json::{json, Value};
use tokio::sync::mpsc;

type Events = mpsc::UnboundedSender<(&'static str, Value)>;

async fn chat(State(events): State<Events>, Json(body): Json<Value>) -> Body {
    events.send(("chat", body.clone())).unwrap();
    if body["hold"] == true {
        if body["stream"] == true {
            return Body::from_stream(futures::stream::pending::<
                Result<bytes::Bytes, std::io::Error>,
            >());
        }
        std::future::pending::<()>().await;
    }
    Body::from(if body["stream"] == true {
        "data: [DONE]\n\n"
    } else {
        "{}"
    })
}

async fn abort(State(events): State<Events>, headers: HeaderMap, Json(body): Json<Value>) {
    assert_eq!(headers["authorization"], "Bearer test");
    events.send(("abort", body)).unwrap();
}

#[tokio::test]
async fn cancellation_aborts_only_unfinished_requests() {
    let (events, mut rx) = mpsc::unbounded_channel();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new()
                .route("/v1/chat/completions", post(chat))
                .route("/abort_request", post(abort))
                .with_state(events),
        )
        .await
        .unwrap();
    });
    for streaming in [false, true] {
        for hold in [false, true] {
            let app = build_router(build_ctx_with_worker(&url));
            let mut body = json!({"model": "tiny", "stream": streaming, "hold": hold});
            if streaming {
                body["rid"] = json!("caller-rid");
            }
            let request = Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("authorization", "Bearer test")
                .header("x-request-id", "gateway-rid")
                .body(Body::from(body.to_string()))
                .unwrap();
            let task = tokio::spawn(app.oneshot(request));
            let (event, forwarded) = tokio::time::timeout(TEST_TIMEOUT, rx.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(event, "chat");
            assert_eq!(
                forwarded["rid"],
                if streaming {
                    "caller-rid"
                } else {
                    "gateway-rid"
                }
            );
            if hold && !streaming {
                task.abort();
                assert!(task.await.unwrap_err().is_cancelled());
            } else {
                let response = tokio::time::timeout(TEST_TIMEOUT, task)
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                if hold {
                    drop(response);
                } else {
                    response.into_body().collect().await.unwrap();
                }
            }
            if hold {
                let (event, aborted) = tokio::time::timeout(TEST_TIMEOUT, rx.recv())
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(event, "abort");
                assert_eq!(
                    aborted,
                    json!({"rid": forwarded["rid"], "abort_all": false})
                );
            }
            assert!(tokio::time::timeout(Duration::from_millis(20), rx.recv())
                .await
                .is_err());
        }
    }
    server.abort();
}
