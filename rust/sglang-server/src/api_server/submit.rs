//! Request submission into the ingress pipeline, shared by every endpoint
//! module: bind the request's final rid, build the `Request`, and hand it to the
//! TM with an egress receiver for the response.

use std::convert::Infallible;

use axum::{
    Json,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use tokio::sync::mpsc;

use super::AppState;
use super::frame::error_value;
use super::guard::AbortGuard;
use crate::fsm::RequestState;
use crate::ids::Rid;
use crate::message::{EgressItem, EgressSink, GenerateRequest, Request, RequestKind};
use crate::tokenizer_manager::TmEvent;

pub(super) type GenerationReceiver = (Rid, mpsc::Receiver<EgressItem>);

/// Successfully submitted generation requests and their abort-on-drop guard.
/// Keeping these together prevents a caller from losing cancellation coverage
/// between submission and handing the receivers to the unary or streaming path.
pub(super) struct SubmittedGenerations {
    receivers: Vec<GenerationReceiver>,
    guard: AbortGuard,
}

impl SubmittedGenerations {
    pub(super) fn into_parts(self) -> (Vec<GenerationReceiver>, AbortGuard) {
        (self.receivers, self.guard)
    }
}

/// Submit one request; returns its rid and egress receiver. Every request arrives
/// with its final rid: a generate request from `into_requests` (or the
/// `HEALTH_CHECK_<uuid>` the health probe sets), and a control request from its
/// constructor. This only echoes that rid back.
pub(super) async fn submit(
    state: &AppState,
    kind: RequestKind,
    // `stream`: the client is reading an SSE stream, so it expects 200 plus an
    // error frame rather than a 4xx — same rule `pre_submit_error` applies
    // everywhere else.
    stream: bool,
) -> Result<(Rid, mpsc::Receiver<EgressItem>), Response> {
    let rid = match &kind {
        // Generate rids are already final: `GenerateBody::into_requests` normalized the
        // client's, or minted one. Control requests have no client-facing rid.
        RequestKind::Generate(g) => g.rid.clone(),
        RequestKind::Control(c) => c.rid().into(),
    };
    // Two in-flight requests can name the same client rid, but they cannot share a
    // `Rid`: `into_requests` built each through `Rid::from_client`, which appends a
    // uniquifier. So nothing here needs to check for a collision — the detok table
    // key is unique by construction, and `client_facing` restores what the client
    // sent for `meta_info.id`.
    // Async-aware send so a full TM inbox yields (backpressure) instead of parking
    // a thread; Err only when the inbox is closed (shutdown).
    let (tx, rx) = mpsc::channel::<EgressItem>(state.egress_buf);
    let request = Request {
        rid: rid.clone(),
        state: RequestState::Received,
        sink: EgressSink::Local(tx),
        kind,
    };
    match state.senders.tm.send_async(TmEvent::Ingress(request)).await {
        Ok(()) => Ok((rid, rx)),
        // `SendError` has a single meaning — the channel is disconnected.
        Err(_) => {
            tracing::error!(%rid, "tm inbox closed; request rejected");
            // Return 503 so the client can retry.
            Err(pre_submit_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "service unavailable",
                stream,
            ))
        }
    }
}

/// Submit a generation batch before consuming any result. Requests are armed as
/// they enter the scheduler, so a later submission failure aborts the requests
/// that were already accepted.
pub(super) async fn submit_all(
    state: &AppState,
    requests: Vec<GenerateRequest>,
    stream: bool,
) -> Result<SubmittedGenerations, Response> {
    let mut receivers = Vec::with_capacity(requests.len());
    let mut guard = AbortGuard::new_empty(state.senders.clone());

    for request in requests {
        let (rid, receiver) =
            submit(state, RequestKind::Generate(Box::new(request)), stream).await?;
        guard.arm(rid.clone());
        receivers.push((rid, receiver));
    }

    Ok(SubmittedGenerations { receivers, guard })
}

/// Shape an error that occurs *before* (or instead of) a successful submit into a
/// client response. Two parity points with Python's `generate_request`. The body
/// is the same `{"error": {...}}` object every other path emits — not bare text,
/// which a client parsing JSON chokes on. And a streaming request gets 200 plus
/// one SSE error frame and `[DONE]`, not a 4xx: the client has already committed
/// to reading a stream, and Python answers it inside `stream_results()`.
pub(super) fn pre_submit_error(code: StatusCode, message: &str, stream: bool) -> Response {
    let body = error_value(code.as_u16(), message);
    if !stream {
        return (code, Json(body)).into_response();
    }
    let frames = [body.to_string(), "[DONE]".to_string()];
    Sse::new(futures::stream::iter(
        frames.map(|data| Ok::<_, Infallible>(Event::default().data(data))),
    ))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer_manager::{AbortSource, Senders};
    use std::sync::{Arc, atomic::AtomicU64};

    #[tokio::test]
    async fn submit_all_aborts_prior_requests_when_a_later_submit_fails() {
        // A rendezvous channel requires one receive per successful submission,
        // so dropping it after the first ingress deterministically rejects the
        // second request before it can be armed.
        let (tm, tm_rx) = flume::bounded(0);
        let (abort, abort_rx) = flume::unbounded();
        let state = AppState {
            senders: Senders {
                tm,
                abort,
                tok: flume::unbounded().0,
                detok: vec![],
            },
            egress_buf: 8,
            server_args: Arc::new(
                crate::runtime::ServerArgs::from_json(r#"{"model_path": "/m"}"#).unwrap(),
            ),
            egress_activity: Arc::new(AtomicU64::new(0)),
        };
        let requests = ["first", "second"]
            .map(|rid| GenerateRequest {
                rid: rid.into(),
                text: Some("hello".into()),
                ..Default::default()
            })
            .into();

        let submitting = tokio::spawn(async move { submit_all(&state, requests, false).await });

        let TmEvent::Ingress(first) = tm_rx.recv_async().await.unwrap() else {
            panic!("generation must enter through the TM inbox");
        };
        assert_eq!(first.rid.client_facing(), "first");
        drop(first);
        drop(tm_rx);

        let response = match submitting.await.unwrap() {
            Ok(_) => panic!("the second submission must fail"),
            Err(response) => response,
        };
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(matches!(
            abort_rx.try_recv().unwrap(),
            AbortSource::Guard(rid) if rid.client_facing() == "first"
        ));
        assert!(
            abort_rx.try_recv().is_err(),
            "only the successfully submitted request is armed"
        );
    }

    /// Unary pre-submit errors are a 4xx/5xx with a JSON `{"error":...}` body;
    /// streaming ones are 200 + an SSE error frame + `[DONE]`, because Python
    /// answers from inside `stream_results()` once the stream is committed.
    #[tokio::test]
    async fn pre_submit_errors_match_python_shape() {
        let unary = pre_submit_error(StatusCode::BAD_REQUEST, "bad input", false);
        assert_eq!(unary.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(unary.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).expect("JSON body");
        assert_eq!(v["error"]["message"], "bad input");
        assert_eq!(v["error"]["code"], 400);

        let streamed = pre_submit_error(StatusCode::BAD_REQUEST, "bad input", true);
        assert_eq!(
            streamed.status(),
            StatusCode::OK,
            "the stream itself is 200"
        );
        let body = axum::body::to_bytes(streamed.into_body(), 64 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            text.contains(r#""code":400"#),
            "carries the status in-band: {text}"
        );
        assert!(
            text.trim_end().ends_with("data: [DONE]"),
            "terminated: {text}"
        );
    }
}
