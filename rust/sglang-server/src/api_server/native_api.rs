//! The native SGLang data-plane endpoints: `/generate` (submit a request, then
//! either fold egress frames to one unary JSON response or relay them as SSE
//! `data: {json}` … `[DONE]`, byte-compatible with Python
//! `http_server.generate_request`) and `/health` + `/health_generate` (which
//! round-trip a 1-token generate probe). Frame shaping (`meta_info`, logprob
//! tuples, cumulative vs incremental streams) lives here, as does
//! generate-request submission (`submit`); the shared `AppState` lives in the
//! parent `api_server` module.

use std::convert::Infallible;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use tokio::sync::mpsc;

use super::AppState;
use super::frame::{
    OutputAccumulator, abort_status, cumulative_frame_string, error_value, sglang_frame_value,
    stream_frame_string, tag_value,
};
use super::guard::AbortGuard;
use crate::environ::env_bool;
use crate::fsm::RequestState;
use crate::ids::RidHash;
use crate::message::{EgressItem, EgressSink, GenerateBody, GenerateRequest, Request, RequestKind};
use crate::runtime::channels::TmEvent;

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/generate", post(generate))
        // `/health` runs the generation round-trip by default (env-gated, else
        // plain 200); `/health_generate` always does. Mirrors Python.
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
}

/// `GET /health` — liveness. By default (env true, mirroring Python) runs the same
/// 1-token round-trip as `/health_generate`; env false → plain 200 (routing the
/// request already proves the frontend is up).
async fn health(state: State<AppState>) -> Response {
    if env_bool("SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION", true) {
        health_generate(state).await
    } else {
        StatusCode::OK.into_response()
    }
}

/// `GET /health_generate` — deep health: confirm the scheduler → detok path is
/// producing output. 200 iff the egress heartbeat advances within the timeout,
/// else 503. (`/health` delegates here when its env gate is on.)
///
/// Fires a pre-tokenized 1-token probe (`input_ids = [0]`, skips the tokenizer) so
/// an idle pipeline produces a frame, then watches the *global*
/// [`AppState::egress_activity`] counter (not the probe's own rid) — so a busy
/// server passes immediately and a backlog never false-503s (the analogue of
/// Python's `last_receive_tstamp`). The `HEALTH_CHECK` skip + `http_worker_ipc`
/// ack are irrelevant here: this single-process server owns the egress ring.
async fn health_generate(State(state): State<AppState>) -> Response {
    let baseline = state
        .egress_activity
        .load(std::sync::atomic::Ordering::Relaxed);

    // Fire the probe (the heartbeat is the signal, not its own response). A busy
    // scheduler skips it with no terminal frame, so its detok registration is
    // cleaned up only by the `AbortGuard` below.
    let sampling_params = rmpv::Value::Map(vec![
        (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(1)),
        (rmpv::Value::from("temperature"), rmpv::Value::F64(0.0)),
    ]);
    let probe = GenerateRequest {
        input_ids: Some(vec![0]),
        sampling_params: Some(sampling_params),
        stream: false,
        // The scheduler skips this when busy so it never occupies a queue slot.
        is_health_check: true,
        ..Default::default()
    };
    let (id, rid, _keepalive) = match submit(&state, probe).await {
        // Hold the receiver so the probe's sink stays open until it completes.
        Ok((id, rid, rx)) => (id, rid, rx),
        Err(()) => return StatusCode::SERVICE_UNAVAILABLE.into_response(),
    };
    // Deregister on drop (never disarmed): a busy-skipped probe has no terminal
    // frame, so without this abort it leaks one detok entry per call.
    let _abort_guard = AbortGuard::new(state.senders.clone(), id, rid);

    // Watch the heartbeat advance. `SGLANG_HEALTH_CHECK_TIMEOUT` defaults to 20s.
    let timeout = crate::environ::env_u64("SGLANG_HEALTH_CHECK_TIMEOUT", 20);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout);
    loop {
        if state
            .egress_activity
            .load(std::sync::atomic::Ordering::Relaxed)
            != baseline
        {
            return StatusCode::OK.into_response();
        }
        if tokio::time::Instant::now() >= deadline {
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

/// Submit one generate request into the ingress pipeline; returns the
/// client-visible rid (uuid hex, Python-parity), its hashed routing key, and the
/// egress receiver. Rid policy: health probes get the Python server's
/// `HEALTH_CHECK_<uuid>` form so scheduler logs and prefix-gated handling
/// recognize them; a client-supplied rid (already fanned out per item by
/// `split`) wins over minting.
async fn submit(
    state: &AppState,
    req: GenerateRequest,
) -> Result<(RidHash, String, mpsc::Receiver<EgressItem>), ()> {
    let rid = if req.is_health_check {
        crate::ids::new_health_check_rid()
    } else {
        req.rid.clone().unwrap_or_else(crate::ids::new_rid)
    };
    let id = RidHash::from_rid(&rid);
    // Async-aware send so a full TM inbox yields (backpressure) instead of parking
    // a thread; Err only when the inbox is closed (shutdown).
    let (tx, rx) = mpsc::channel::<EgressItem>(state.egress_buf);
    let request = Request {
        rid_hash: id,
        rid: rid.clone(),
        state: RequestState::Received,
        sink: EgressSink::Local(tx),
        kind: RequestKind::Generate(req),
    };
    match state.senders.tm.send_async(TmEvent::Ingress(request)).await {
        Ok(()) => Ok((id, rid, rx)),
        Err(_) => {
            tracing::error!("tm inbox closed; request dropped");
            Err(())
        }
    }
}

async fn generate(State(state): State<AppState>, Json(body): Json<GenerateBody>) -> Response {
    let stream = body.stream;
    // Fan `text`/`input_ids`/`sampling_params` (scalar or list) into per-request
    // payloads. `is_batch` = list form → the response is a JSON array.
    let (payloads, is_batch) = match body.split() {
        Ok(v) => v,
        Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
    };
    if !is_batch {
        // `split` guarantees exactly one payload for a non-batch body.
        let payload = payloads
            .into_iter()
            .next()
            .expect("split yields >=1 payload");
        generate_single(&state, payload, stream).await
    } else {
        generate_batch(&state, payloads, stream).await
    }
}

/// A single (non-batched) `/generate`: submit one request, then either stream its
/// SSE frames or fold to one unary response.
async fn generate_single(state: &AppState, req: GenerateRequest, stream: bool) -> Response {
    // `return_text_in_logprobs` is decoded on the detok shard into `*_txt`, so
    // `sglang_frame_value` just reads them — no tokenizer needed here.
    let (id, rid_str, mut rx) = match submit(state, req).await {
        Ok(v) => v,
        Err(()) => {
            return (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response();
        }
    };
    // Abort on client disconnect: the guard fires when dropped before the request
    // finishes (axum drops the handler/SSE stream). Disarmed on a natural terminal.
    // `rid_str` is the response `meta_info.id`, reused for every frame.
    let mut guard = AbortGuard::new(state.senders.clone(), id, rid_str.clone());
    // Cumulative frames (SGLang default) vs per-step deltas.
    let incremental = state.server_args.incremental_streaming_output;

    if stream {
        // A single request is a 1-element batch without the `index` field — reuse
        // the same stream so the frame/abort/truncation logic lives in one place.
        use futures::StreamExt;
        let s = generation_event_stream(vec![(id, rid_str, rx)], guard, incremental, false)
            .map(|data| Ok::<_, Infallible>(Event::default().data(data)));
        Sse::new(s).into_response()
    } else {
        // Unary: fold to the terminal, respond once. Disarm only on a real terminal
        // (a truncation leaves the guard armed so the scheduler work is aborted).
        let (status, value, terminal) = drain_unary(&mut rx, &rid_str).await;
        if terminal {
            guard.disarm(id);
        }
        (status, Json(value)).into_response()
    }
}

/// Fold a unary request to its terminal → (HTTP status, result/`error` JSON, saw-terminal); `false` = truncation, caller keeps the abort guard armed. Shared by single + batch.
async fn drain_unary(
    rx: &mut mpsc::Receiver<EgressItem>,
    rid_str: &str,
) -> (StatusCode, serde_json::Value, bool) {
    let mut acc = OutputAccumulator::default();
    while let Some(item) = rx.recv().await {
        match item {
            EgressItem::Frame(out) => acc.fold(&out),
            EgressItem::Done(out) => {
                acc.fold(&out);
                let final_out = acc.into_output();
                // A validation abort carries its own HTTP status + diagnostic.
                if let Some((code, message)) = abort_status(&final_out.finish_reason) {
                    let status =
                        StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    return (status, error_value(code, &message), true);
                }
                return (
                    StatusCode::OK,
                    sglang_frame_value(&final_out, rid_str),
                    true,
                );
            }
            EgressItem::Error(e) => {
                let code = e.http_status();
                let status =
                    StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                return (status, error_value(code, &e.to_string()), true);
            }
            EgressItem::Control(_) => continue, // never on `/generate`
        }
    }
    // Sender dropped without a terminal item: the shard dropped this request (a
    // truncation — a client disconnect would have dropped the handler future).
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        error_value(500, "response truncated before completion"),
        false,
    )
}

/// Batch `/generate`: submit all sub-requests first (scheduler runs them together),
/// then either (unary) drain each in order into a JSON array, or (streaming)
/// multiplex their streams into one SSE response, each frame carrying its `index`.
/// One [`AbortGuard`] covers the batch. A failed unary item is its own
/// `{ "error": … }` entry; the batch response is 200.
async fn generate_batch(
    state: &AppState,
    requests: Vec<GenerateRequest>,
    stream: bool,
) -> Response {
    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let mut receivers = Vec::with_capacity(requests.len());
    for req in requests {
        match submit(state, req).await {
            Ok((id, rid, rx)) => {
                guard.arm(id, rid.clone());
                receivers.push((id, rid, rx));
            }
            Err(()) => {
                return (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response();
            }
        }
    }

    if stream {
        // Multiplex the N streams (mirrors the Python `_handle_batch_request` path);
        // `guard` moves into the stream so a disconnect aborts what's unfinished.
        use futures::StreamExt;
        let incremental = state.server_args.incremental_streaming_output;
        let s = generation_event_stream(receivers, guard, incremental, true)
            .map(|data| Ok::<_, Infallible>(Event::default().data(data)));
        Sse::new(s).into_response()
    } else {
        // Unary: drain each in order (already all submitted, so they run together).
        let mut results = Vec::with_capacity(receivers.len());
        for (id, rid_str, mut rx) in receivers {
            let (_status, value, terminal) = drain_unary(&mut rx, &rid_str).await;
            if terminal {
                guard.disarm(id);
            }
            results.push(value);
        }
        (StatusCode::OK, Json(serde_json::Value::Array(results))).into_response()
    }
}

/// Await the next item from `rx`, then drain whatever queued behind it (so the caller
/// can coalesce a backlog, as Python's `state.out_list` does), handing the receiver
/// back for `FuturesUnordered` to re-poll. Empty result = channel closed.
async fn recv_indexed(
    index: usize,
    mut rx: mpsc::Receiver<EgressItem>,
) -> (usize, mpsc::Receiver<EgressItem>, Vec<EgressItem>) {
    let mut items = Vec::new();
    match rx.recv().await {
        Some(item) => items.push(item),
        None => return (index, rx, items), // closed
    }
    while let Ok(item) = rx.try_recv() {
        items.push(item);
    }
    (index, rx, items)
}

/// Multiplex `receivers` (one per request) into SSE `data` strings + a final `[DONE]`;
/// `with_index` tags each frame (batch only), `incremental` = delta vs cumulative,
/// `guard` aborts unfinished on drop.
fn generation_event_stream(
    receivers: Vec<(RidHash, String, mpsc::Receiver<EgressItem>)>,
    mut guard: AbortGuard,
    incremental: bool,
    with_index: bool,
) -> impl futures::Stream<Item = String> {
    async_stream::stream! {
        use futures::StreamExt;

        let n = receivers.len();
        let rids: Vec<RidHash> = receivers.iter().map(|(id, _, _)| *id).collect();
        let rid_strs: Vec<String> = receivers.iter().map(|(_, rid, _)| rid.clone()).collect();
        let mut accs: Vec<OutputAccumulator> =
            (0..n).map(|_| OutputAccumulator::default()).collect();

        // Batch position, tagged onto every frame (a single request omits it).
        let idx = |i: usize| with_index.then_some(i);

        // Poll all receivers concurrently; re-arm a receiver's future after each
        // non-terminal frame so its stream keeps flowing.
        let mut futs = futures::stream::FuturesUnordered::new();
        for (i, (_, _, rx)) in receivers.into_iter().enumerate() {
            futs.push(recv_indexed(i, rx));
        }

        while let Some((i, rx, items)) = futs.next().await {
            if items.is_empty() {
                // Channel closed with no terminal → truncation for this item;
                // leave its rid armed so the scheduler work is aborted.
                yield tag_value(error_value(500, "response truncated before completion"), idx(i));
                continue;
            }

            // Cumulative frames supersede one another, so a drained backlog collapses
            // to its last (Python's `out_list[-1]`); deltas can't be dropped.
            let mut coalesced = false; // a cumulative frame is pending
            let mut terminal = None;   // (finish_reason) of a `Done` in this batch
            let mut failed = None;     // an `Error` in this batch

            for item in items {
                match item {
                    EgressItem::Frame(out) => {
                        accs[i].fold(&out);
                        if incremental {
                            yield stream_frame_string(out, &accs[i], true, &rid_strs[i], idx(i));
                        } else {
                            coalesced = true;
                        }
                    }
                    EgressItem::Done(out) => {
                        accs[i].fold(&out);
                        terminal = Some(out);
                    }
                    EgressItem::Error(e) => failed = Some(e),
                    EgressItem::Control(_) => {} // never on /generate
                }
            }

            if let Some(e) = failed {
                yield tag_value(error_value(e.http_status(), &e.to_string()), idx(i));
                guard.disarm(rids[i]);
            } else if let Some(out) = terminal {
                // A validation abort → an error object, not a frame. The final frame
                // carries the full cumulative state, so any coalesced ones are moot.
                yield match abort_status(&out.finish_reason) {
                    Some((code, message)) => tag_value(error_value(code, &message), idx(i)),
                    None => stream_frame_string(out, &accs[i], incremental, &rid_strs[i], idx(i)),
                };
                guard.disarm(rids[i]); // terminal → not re-pushed
            } else {
                if coalesced {
                    yield cumulative_frame_string(&accs[i], &rid_strs[i], idx(i));
                }
                futs.push(recv_indexed(i, rx)); // keep this item flowing
            }
        }
        yield "[DONE]".to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ChunkEvent;
    use crate::runtime::channels::Senders;
    use futures::StreamExt;
    fn senders() -> Senders {
        Senders {
            tm: flume::unbounded().0,
            tok: flume::unbounded().0,
            detok: vec![],
        }
    }

    fn frame(rid: u64, text: &str) -> EgressItem {
        EgressItem::Frame(ChunkEvent {
            rid_hash: rid,
            text: text.into(),
            completion_tokens: 1,
            ..Default::default()
        })
    }
    fn done(rid: u64, text: &str) -> EgressItem {
        EgressItem::Done(ChunkEvent {
            rid_hash: rid,
            text: text.into(),
            completion_tokens: 1,
            finish_reason: Some(serde_json::json!({ "type": "length" })),
            ..Default::default()
        })
    }
    fn parse(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("frame is JSON")
    }

    /// Two sub-requests' frames interleave into one stream, each tagged with its
    /// batch `index`; text accumulates per item; `[DONE]` comes only after both
    /// terminate, then the stream ends.
    #[tokio::test]
    async fn interleaves_indexes_and_accumulates() {
        let (tx0, rx0) = mpsc::channel(8);
        let (tx1, rx1) = mpsc::channel(8);
        let receivers = vec![
            (RidHash(10), "10".to_string(), rx0),
            (RidHash(11), "11".to_string(), rx1),
        ];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), false, true);
        futures::pin_mut!(stream);

        // Drive deterministically: exactly one channel has data before each poll.
        tx0.send(frame(10, "a")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 0);
        assert_eq!(v["text"], "a");

        tx1.send(frame(11, "b")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 1);
        assert_eq!(v["text"], "b");

        tx0.send(done(10, "!")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 0);
        assert_eq!(v["text"], "a!", "cumulative per item");
        assert_eq!(v["meta_info"]["finish_reason"]["type"], "length");

        tx1.send(done(11, "?")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 1);
        assert_eq!(v["text"], "b?");

        assert_eq!(stream.next().await.unwrap(), "[DONE]");
        assert!(stream.next().await.is_none());
    }

    /// A per-item error is surfaced with its `index` and doesn't end the batch;
    /// `[DONE]` still waits for the other item.
    #[tokio::test]
    async fn per_item_error_carries_index() {
        let (tx0, rx0) = mpsc::channel(8);
        let (tx1, rx1) = mpsc::channel(8);
        let receivers = vec![
            (RidHash(10), "10".to_string(), rx0),
            (RidHash(11), "11".to_string(), rx1),
        ];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), false, true);
        futures::pin_mut!(stream);

        tx0.send(EgressItem::Error(crate::error::Error::Validation(
            "bad".into(),
        )))
        .await
        .unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 0);
        assert_eq!(v["error"]["code"], 400);

        tx1.send(done(11, "ok")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["index"], 1);

        assert_eq!(stream.next().await.unwrap(), "[DONE]");
    }

    /// `incremental=true`: each frame carries this step's **delta** text/output_ids,
    /// but `meta_info.completion_tokens` stays cumulative (matching Python).
    #[tokio::test]
    async fn incremental_emits_deltas_with_cumulative_count() {
        let (tx, rx) = mpsc::channel(8);
        let receivers = vec![(RidHash(10), "10".to_string(), rx)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), true, true);
        futures::pin_mut!(stream);

        tx.send(frame(10, "Hello")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "Hello");
        assert_eq!(v["meta_info"]["completion_tokens"], 1);

        tx.send(frame(10, " world")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], " world", "delta, not cumulative 'Hello world'");
        assert_eq!(
            v["meta_info"]["completion_tokens"], 2,
            "count stays cumulative"
        );

        tx.send(done(10, "!")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "!");
        assert_eq!(v["meta_info"]["completion_tokens"], 3);
        assert_eq!(v["meta_info"]["finish_reason"]["type"], "length");

        assert_eq!(stream.next().await.unwrap(), "[DONE]");
    }

    /// The single-request shape (`with_index=false`, one receiver) omits the
    /// `index` field entirely, and still terminates with `[DONE]`.
    #[tokio::test]
    async fn single_shape_omits_index() {
        let (tx, rx) = mpsc::channel(8);
        let receivers = vec![(RidHash(10), "10".to_string(), rx)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), false, false);
        futures::pin_mut!(stream);

        tx.send(done(10, "hi")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "hi");
        assert!(v.get("index").is_none(), "single response has no index");

        assert_eq!(stream.next().await.unwrap(), "[DONE]");
    }

    /// A backlog of cumulative chunks collapses to a single frame carrying the latest
    /// state — each cumulative frame supersedes the last, so emitting the intermediate
    /// ones ships the full O(T) payload again for nothing. Mirrors the Python waiter's
    /// `out = out_list[-1]`. This is the whole point of draining in `recv_indexed`.
    #[tokio::test]
    async fn cumulative_backlog_coalesces_to_latest() {
        let (tx, rx) = mpsc::channel(8);
        let receivers = vec![(RidHash(10), "10".to_string(), rx)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), false, false);
        futures::pin_mut!(stream);

        // Three chunks queued before the stream is ever polled (a client falling behind).
        tx.send(frame(10, "a")).await.unwrap();
        tx.send(frame(10, "b")).await.unwrap();
        tx.send(frame(10, "c")).await.unwrap();

        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "abc", "one frame, full cumulative text");
        assert_eq!(v["meta_info"]["completion_tokens"], 3, "no tokens lost");

        // The terminal frame still carries everything, and only then does [DONE] land.
        tx.send(done(10, "!")).await.unwrap();
        let v = parse(&stream.next().await.unwrap());
        assert_eq!(v["text"], "abc!");
        assert_eq!(v["meta_info"]["finish_reason"]["type"], "length");
        assert_eq!(stream.next().await.unwrap(), "[DONE]");
    }

    /// Incremental frames are *deltas*, so a backlog must emit every one — dropping
    /// any would silently lose tokens. Only the cumulative protocol may coalesce.
    #[tokio::test]
    async fn incremental_backlog_emits_every_delta() {
        let (tx, rx) = mpsc::channel(8);
        let receivers = vec![(RidHash(10), "10".to_string(), rx)];
        let stream =
            generation_event_stream(receivers, AbortGuard::new_empty(senders()), true, false);
        futures::pin_mut!(stream);

        tx.send(frame(10, "a")).await.unwrap();
        tx.send(frame(10, "b")).await.unwrap();
        tx.send(frame(10, "c")).await.unwrap();

        for (n, expect) in [(1, "a"), (2, "b"), (3, "c")] {
            let v = parse(&stream.next().await.unwrap());
            assert_eq!(v["text"], expect, "delta {n} must not be dropped");
            assert_eq!(
                v["meta_info"]["completion_tokens"], n,
                "count stays cumulative"
            );
        }
    }
}
