// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION},
    Client, Url,
};
use std::time::Duration;

/// Cancels unfinished plain-mode work, including cancellation before response headers.
pub(super) struct AbortOnDrop {
    client: Client,
    url: Url,
    rid: Option<String>,
    authorization: Option<HeaderValue>,
}

impl AbortOnDrop {
    pub fn new(
        client: &Client,
        worker: &Url,
        rid: Option<&str>,
        headers: &HeaderMap,
    ) -> Option<Self> {
        // The engine treats rid as a prefix; an empty ID would cancel every request.
        let rid = rid.filter(|id| !id.is_empty())?;
        Some(Self {
            client: client.clone(),
            url: worker.join("/abort_request").ok()?,
            rid: Some(rid.to_owned()),
            authorization: headers.get(AUTHORIZATION).cloned(),
        })
    }

    pub fn disarm(&mut self) {
        self.rid = None;
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        let Some(rid) = self.rid.take() else { return };
        let client = self.client.clone();
        let url = self.url.clone();
        let authorization = self.authorization.take();
        tokio::spawn(async move {
            let mut request = client.post(url);
            if let Some(value) = authorization {
                request = request.header(AUTHORIZATION, value);
            }
            let result = request
                .json(&serde_json::json!({"rid": rid, "abort_all": false}))
                .timeout(Duration::from_secs(5))
                .send()
                .await
                .and_then(reqwest::Response::error_for_status);
            if let Err(error) = result {
                tracing::warn!(%rid, %error, "could not abort unfinished engine request");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{health::circuit_breaker::CircuitBreaker, proxy::Proxy, workers::WireProtocol};
    use axum::{body::Body, routing::post, Json, Router};
    use bytes::Bytes;
    use http_body_util::BodyExt;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    async fn worker() -> (
        String,
        mpsc::UnboundedReceiver<serde_json::Value>,
        tokio::task::JoinHandle<()>,
    ) {
        let (tx, rx) = mpsc::unbounded_channel();
        let app = Router::new()
            .route(
                "/abort_request",
                post(move |headers: HeaderMap, Json(value): Json<serde_json::Value>| {
                    let tx = tx.clone();
                    async move {
                        tx.send(serde_json::json!({"body":value,"authorization":headers.get(AUTHORIZATION).and_then(|h| h.to_str().ok())})).unwrap();
                    }
                }),
            )
            .route(
                "/pending",
                post(|| async { futures::future::pending::<String>().await }),
            )
            .route(
                "/stream",
                post(|| async {
                    Body::from_stream(futures::stream::pending::<Result<Bytes, std::io::Error>>())
                }),
            )
            .route("/done", post(|| async { "data: [DONE]\n\n" }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (url, rx, task)
    }

    #[tokio::test]
    async fn timeout_and_stream_disconnect_abort_the_matching_request() {
        let (url, mut aborted, task) = worker().await;
        let proxy = Proxy::new(Duration::from_millis(50)).unwrap();
        let breaker = Arc::new(CircuitBreaker::new());
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer test"));
        for streaming in [false, true] {
            let result = if streaming {
                proxy
                    .forward_streaming_to(
                        &url,
                        WireProtocol::Http1,
                        &breaker,
                        "/pending",
                        &headers,
                        Bytes::new(),
                        None,
                        None,
                        None,
                        Some("waiting"),
                    )
                    .await
            } else {
                proxy
                    .forward_json_to(
                        &url,
                        WireProtocol::Http1,
                        &breaker,
                        "/pending",
                        &headers,
                        Bytes::new(),
                        Some("waiting"),
                    )
                    .await
            };
            assert!(result.is_err());
            let abort = tokio::time::timeout(Duration::from_secs(2), aborted.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(abort["authorization"], "Bearer test");
            assert_eq!(
                abort["body"],
                serde_json::json!({"rid": "waiting", "abort_all": false})
            );
        }
        let response = proxy
            .forward_streaming_to(
                &url,
                WireProtocol::Http1,
                &breaker,
                "/stream",
                &headers,
                Bytes::new(),
                None,
                None,
                None,
                Some("streaming"),
            )
            .await
            .unwrap();
        drop(response);
        let abort = tokio::time::timeout(Duration::from_secs(2), aborted.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(abort["body"]["rid"], "streaming");
        task.abort();
    }

    #[tokio::test]
    async fn clean_completion_and_pd_disconnect_do_not_abort() {
        let (url, mut aborted, task) = worker().await;
        let proxy = Proxy::new(Duration::from_secs(2)).unwrap();
        let breaker = Arc::new(CircuitBreaker::new());
        for request_id in [Some("complete"), None, Some("")] {
            let path = if request_id == Some("complete") {
                "/done"
            } else {
                "/stream"
            };
            let response = proxy
                .forward_streaming_to(
                    &url,
                    WireProtocol::Http1,
                    &breaker,
                    path,
                    &HeaderMap::new(),
                    Bytes::new(),
                    None,
                    None,
                    None,
                    request_id,
                )
                .await
                .unwrap();
            if request_id == Some("complete") {
                response.into_body().collect().await.unwrap();
            }
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(50), aborted.recv())
                .await
                .is_err()
        );
        task.abort();
    }
}
