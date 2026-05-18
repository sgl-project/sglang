// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{ModelId, WorkerMode};
use crate::policies::registry::{PdPoolResolver, PdResolveError};
use crate::policies::SelectionContext;
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::{RequestOutcome, StaleRequestOutcome, WorkerModeLabel};
use crate::workers::LoadGuard;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use bytes::Bytes;
use serde::de::IgnoredAny;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

/// HTTP header carrying the decode-pool URL selected via host-affinity
/// for a PD-disaggregated request. The prefill worker reads this on its
/// own bootstrap path (M5 will wire `bootstrap_host` / `bootstrap_port`
/// injection); M4 sets the header but the bootstrap injection itself
/// is deferred. The header name uses the `x-sgl-` prefix shared with
/// `x-sgl-router-error-code` to keep router-emitted metadata together.
const X_SGL_DECODE_URL: HeaderName = HeaderName::from_static("x-sgl-decode-url");

/// Coarse char-count → token-count divisor used to estimate prefill load
/// from the request body when no real tokenizer count is available. Four
/// bytes per token is the standard SGLang upstream estimate; it
/// overcounts ASCII and undercounts CJK but stays within an order of
/// magnitude of the real token count, which is plenty for load
/// scoring. The active-load counters' role is relative ordering across
/// workers — not absolute accuracy — so the estimate is fit for
/// purpose.
const CHARS_PER_TOKEN_ESTIMATE: usize = 4;

/// Per-route body-size cap on `/v1/chat/completions`. 1 MiB is comfortable
/// for normal chat traffic (a 200 k-token context tokenized as JSON is well
/// under this) while preventing a hostile client from forcing the router to
/// heap-allocate hundreds of MiB before forwarding. The cap is wired in
/// `crate::server::app::build_router` as a route-level `DefaultBodyLimit`
/// layer; axum's `Bytes` extractor enforces it and returns 413
/// PAYLOAD_TOO_LARGE before this handler runs.
pub const MAX_CHAT_BODY_BYTES: usize = 1 << 20;

/// Minimal probe over the request body — we only need the `stream` field
/// and the `model` field to decide between buffered vs SSE forwarding and
/// to select a worker. Deserializing into this struct (vs `serde_json::Value`)
/// does two things:
///
/// 1. Avoids the per-field heap allocation of `Value` for a 1 MiB body.
/// 2. Pins the contract: the body MUST be a JSON object. Degenerate
///    shapes (`null`, `[]`, `"hi"`) fail at this step rather than being
///    silently forwarded with `stream=false`.
///
/// All other fields are ignored — the worker is authoritative for the
/// full request schema.
#[derive(Debug, Deserialize)]
struct RequestProbe {
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    model: Option<String>,
}

/// POST /v1/chat/completions — parse model from body, select a healthy
/// worker via the per-model policy, then proxy the request. If the
/// request opts into streaming (`stream: true`), we pipe SSE bytes back;
/// otherwise buffer.
pub async fn chat_completions(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let probe = parse_probe(&body)?;
    let streaming = probe.stream.unwrap_or(false);
    let model_str = probe
        .model
        .ok_or_else(|| ApiError::BadRequest("missing `model` field".into()))?;
    let model_id = ModelId(model_str.clone());

    // PD pool isolation: for PD-mode deployments, prefill traffic
    // selects from the prefill pool only. Plain-mode deployments fall
    // through to the full candidate set. Partial-failure errors
    // (`no_prefill_workers_available`) are surfaced as 503 with a
    // distinct error code so operators can alert independently.
    let resolver = PdPoolResolver::new(Arc::clone(&ctx.registry));
    let workers = resolver
        .prefill_candidates(&model_id)
        .map_err(|e| match e {
            PdResolveError::NoHealthyWorkers => ApiError::NoHealthyWorkers {
                model: model_str.clone(),
            },
            PdResolveError::NoPrefillWorkersAvailable => ApiError::NoPrefillWorkersAvailable {
                model: model_str.clone(),
            },
            PdResolveError::NoDecodeWorkersAvailable => ApiError::NoDecodeWorkersAvailable {
                model: model_str.clone(),
            },
        })?;

    let policy = ctx
        .policies
        .get(&model_id)
        .ok_or_else(|| ApiError::ModelNotFound(model_str.clone()))?;
    let selection_ctx = SelectionContext::new(&model_id, Some(&body));
    let worker =
        policy
            .select(&workers, &selection_ctx)
            .ok_or_else(|| ApiError::PolicySelectionFailed {
                model: model_str.clone(),
            })?;

    // PD-mode decoder affinity (M4 gap-closer #3). When the selected
    // prefill worker is part of a PD-disagg deployment, also resolve
    // the matching decode peer (same host where possible, falling back
    // to min-load via `select_decode_with_affinity`). The decode URL is
    // forwarded to the prefill worker via the `x-sgl-decode-url`
    // header — M5 will wire the bootstrap injection itself, but
    // selecting the peer at the router edge is M4's contract
    // ("bonus tokens decoded correctly"): the prefill→decode handoff
    // becomes deterministic per host, removing one source of cross-
    // worker tail latency.
    //
    // Plain-mode workers skip the decode resolution entirely (no
    // decode peer to find). PD-mode requests that fail to resolve a
    // decode peer (`NoDecodeWorkersAvailable`) bubble up as 503 so
    // operators can alert on prefill-vs-decode pool imbalance.
    let decode_hint: Option<String> = if worker.mode() == WorkerMode::Prefill {
        Some(
            resolver
                .decode_with_affinity(&model_id, &worker.url)
                .map(|d| d.url.clone())
                .map_err(|e| match e {
                    PdResolveError::NoHealthyWorkers => ApiError::NoHealthyWorkers {
                        model: model_str.clone(),
                    },
                    PdResolveError::NoDecodeWorkersAvailable => {
                        ApiError::NoDecodeWorkersAvailable {
                            model: model_str.clone(),
                        }
                    }
                    PdResolveError::NoPrefillWorkersAvailable => {
                        ApiError::NoPrefillWorkersAvailable {
                            model: model_str.clone(),
                        }
                    }
                })?,
        )
    } else {
        None
    };
    let mut request_headers = headers;
    if let Some(url) = &decode_hint {
        match HeaderValue::from_str(url) {
            Ok(v) => {
                request_headers.insert(X_SGL_DECODE_URL, v);
            }
            Err(e) => {
                // Discovery emits URLs the proxy has already used; a
                // header-value parse failure here means the URL
                // contains a control character (e.g. CR / LF) — drop
                // it loudly so an operator notices but don't kill the
                // request: the M5 bootstrap can fall back to its own
                // probing if the header is absent.
                tracing::warn!(
                    decode_url = %url,
                    error = %e,
                    "decode worker URL rejected by header parser; sending request without decode hint",
                );
            }
        }
    }
    let headers = request_headers;

    // M2 per-worker active_requests guard. The M4 ActiveLoadGuard below
    // sits beside this one: both track in-flight load, but the M4 entry
    // is per-request (with timeout-based janitor) while the M2 entry is
    // a worker-scoped counter the cache-aware policy reads. Both must
    // drop at the same time — when the response stream ends, the client
    // disconnects, or the handler returns an error. We bundle them
    // together for the streaming pump task so the two lifetimes are
    // tied. Decode-load contribution is 0 here: the active-load
    // registry's decode axis is exclusively driven by M5's decode-pool
    // dispatch (not implemented yet — prefill-only routing in M4 means
    // every chat request adds to prefill_load and nothing else).
    let guard = worker.load_guard();
    let prefill_load = estimate_prefill_tokens(&body);
    let active_guard = ctx.active_load.register(worker.id.clone(), prefill_load, 0);
    // Snapshot the stale-request cancel token BEFORE moving the guard
    // into the streaming pump or the response future. The token is
    // cheap to clone (it's an `Arc<...>` internally) and the chat
    // handler races the upstream fetch against `token.cancelled()` to
    // surface a 504 `stale_request_expired` if the janitor expires
    // the request mid-flight.
    let stale_token = active_guard.cancel_token().clone();

    // Snapshot the labels we need for metrics BEFORE moving the worker
    // / model_str values into the per-branch fetch futures.
    let metrics_worker_url = worker.url.clone();
    let metrics_mode = match worker.mode() {
        WorkerMode::Prefill => WorkerModeLabel::Prefill,
        WorkerMode::Decode => WorkerModeLabel::Decode,
        WorkerMode::Plain => WorkerModeLabel::Plain,
    };
    let metrics_model = model_str.clone();

    let result = if streaming {
        // Hand BOTH guards to the streaming body task. They are held
        // until the SSE pump finishes (last byte / client disconnect /
        // upstream error), not just until this handler returns (which
        // happens when headers arrive). Without the M4 guard here, a
        // long-running stream would under-report active-load to the
        // cache-aware policy and over-attract new requests to the same
        // worker.
        let stream_guards: Box<dyn Send + 'static> = Box::new((guard, active_guard));
        // For streaming, the cancellation race is only over the
        // initial headers fetch (`forward_streaming_to.await`) — once
        // headers arrive, the SSE pump owns the guard. Pre-headers
        // cancellation surfaces as 504; post-headers cancellation
        // (janitor fires mid-stream) is observable in the pump task
        // dropping the M4 guard, but the response status is already
        // 200 by then and the client sees a truncated stream.
        let fetch = ctx.proxy.forward_streaming_to(
            &worker.url,
            &worker.breaker,
            "/v1/chat/completions",
            &headers,
            body,
            Some(stream_guards),
        );
        // Bias `fetch` over the cancellation branch: a successful
        // response that completes in the same poll as the token firing
        // MUST win (returning 504 for a request that already has
        // headers is a correctness regression). The cancellation
        // branch only matters when fetch is still pending — at that
        // point biasing the order is a wash.
        tokio::select! {
            biased;
            r = fetch => r,
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        }
    } else {
        // Non-streaming: the handler awaits the full buffered response,
        // so both guards live correctly in this scope. The tuple binding
        // exists only to extend the guards' lifetime to the end of the
        // function — the `forward_json_to` future does not need them
        // (it does not return until the body is buffered).
        let _holds: (LoadGuard, _) = (guard, active_guard);
        let fetch = ctx.proxy.forward_json_to(
            &worker.url,
            &worker.breaker,
            "/v1/chat/completions",
            &headers,
            body,
        );
        // Race the upstream fetch against the stale-request token. If
        // the janitor fires while we're awaiting the response, return
        // 504 instead of leaving the client hung. Same `biased` order
        // as the streaming arm: a successful fetch that completes in
        // the same poll as token cancellation MUST win.
        tokio::select! {
            biased;
            r = fetch => r,
            _ = stale_token.cancelled() => Err(ApiError::StaleRequestExpired { model: model_str }),
        }
    };

    // Record the dispatch outcome AFTER we know whether the upstream
    // accepted the request. A 504 from the stale-request branch counts as
    // `cancelled` — semantically distinct from upstream errors that bubble
    // through as `error`. The metric is per-worker so the M4 convergence
    // tests can scrape `/metrics` and assert that ≥N requests landed on
    // a single prefill worker.
    let outcome = match &result {
        Ok(_) => RequestOutcome::Success,
        Err(ApiError::StaleRequestExpired { .. }) => {
            // The janitor fired the stale-cancel and we observed it
            // user-side; record both the per-request `cancelled` outcome
            // AND the global `expired` count. The two views are useful for
            // different alerts: per-worker request_total{cancelled} flags a
            // worker that's hanging, while stale_requests_total{expired}
            // tracks the global health of the janitor.
            ctx.metrics
                .record_stale_request(StaleRequestOutcome::Expired);
            RequestOutcome::Cancelled
        }
        Err(_) => RequestOutcome::Error,
    };
    ctx.metrics
        .record_request(&metrics_worker_url, &metrics_model, metrics_mode, outcome);

    // Mirror the upstream `x-sgl-decode-url` hint onto the response so
    // external tests / sidecars can observe PD decode affinity without
    // sniffing the proxy hop. The request-side header was set above for
    // the prefill worker; copying it here makes the affinity observable
    // end-to-end. Plain-mode requests skip this (no decode peer was
    // resolved). A malformed URL was already rejected at the
    // request-side parse — we only reach this branch when the URL was
    // header-valid, so the second parse is safe.
    match (result, decode_hint) {
        (Ok(mut response), Some(url)) => {
            match HeaderValue::from_str(&url) {
                Ok(v) => {
                    response.headers_mut().insert(X_SGL_DECODE_URL, v);
                }
                Err(e) => {
                    // Already-validated upstream; defensive log only.
                    tracing::warn!(
                        decode_url = %url,
                        error = %e,
                        "decode worker URL rejected by header parser on response; omitting response-side hint",
                    );
                }
            }
            Ok(response)
        }
        (other, _) => other,
    }
}

/// Estimate prefill-token count from the raw request body for use as
/// the M4 active-load `prefill_load` counter. Returns 1 at minimum so
/// a registered request always shows up as "load > 0" — under-counting
/// to zero would hide the request from the cache-aware policy's
/// load-imbalance fast-path.
///
/// This is a coarse approximation: we count the body length in bytes
/// and divide by [`CHARS_PER_TOKEN_ESTIMATE`]. A future M5+ improvement
/// is to thread the tokenizer's actual token count through (the
/// cache-aware-zmq policy already tokenizes the prompt for tree
/// matching — that count could be reused here).
fn estimate_prefill_tokens(body: &Bytes) -> usize {
    (body.len() / CHARS_PER_TOKEN_ESTIMATE).max(1)
}

fn parse_probe(body: &Bytes) -> Result<RequestProbe, ApiError> {
    // We deliberately do NOT echo the serde error into the client-visible
    // message — that risks leaking field-level detail and is also of little
    // help to a real client (which already has its own JSON validator).
    // Server-side, the full error is logged with `tracing::debug!` for
    // operator triage.
    //
    // Two-step deserialize:
    //   1. `Map<String, IgnoredAny>` *anchors* the shape to a JSON object.
    //      This rejects `null` / `[]` / `"hi"` (all valid JSON but not
    //      request shape) without walking the full value into a
    //      `serde_json::Value` per field.
    //   2. `RequestProbe` (struct of `Option<bool>` + `Option<String>`)
    //      lifts out only the fields we care about — `stream` and `model`.
    //      Other fields are ignored; the worker is authoritative for the
    //      rest of the schema.
    let _: HashMap<String, IgnoredAny> = serde_json::from_slice(body).map_err(|e| {
        tracing::debug!(error = %e, "chat-completions body rejected as non-object JSON");
        ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
    })?;
    let probe: RequestProbe = serde_json::from_slice(body).map_err(|e| {
        tracing::debug!(error = %e, "chat-completions request-probe deserialize failed");
        ApiError::BadRequest("invalid request: body must be a JSON object".to_string())
    })?;
    Ok(probe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_probe_reads_stream_bool_from_object() {
        let b = Bytes::from_static(br#"{"stream": true, "model": "tiny"}"#);
        assert_eq!(parse_probe(&b).unwrap().stream, Some(true));
        let b = Bytes::from_static(br#"{"stream": false, "model": "tiny"}"#);
        assert_eq!(parse_probe(&b).unwrap().stream, Some(false));
    }

    #[test]
    fn parse_probe_defaults_when_stream_absent() {
        // Existing happy-path contract: well-formed object missing `stream`
        // must default to None (caller picks false). The minimal `RequestProbe`
        // (Option<bool> + #[serde(default)]) must NOT break this.
        let b = Bytes::from_static(br#"{"model": "tiny", "messages": []}"#);
        let p = parse_probe(&b).unwrap();
        assert_eq!(p.stream, None);
        assert_eq!(p.model.as_deref(), Some("tiny"));
    }

    #[test]
    fn parse_probe_rejects_non_object_shapes() {
        // Pin the contract: degenerate JSON (valid JSON but wrong shape)
        // must be rejected, not silently forwarded with `stream=false`.
        for bad in [&b"null"[..], &b"[]"[..], &b"\"hi\""[..], &b"42"[..]] {
            let b = Bytes::copy_from_slice(bad);
            let err = parse_probe(&b).unwrap_err();
            match err {
                ApiError::BadRequest(_) => {}
                other => panic!("expected BadRequest for {bad:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn parse_probe_rejects_malformed_json() {
        let b = Bytes::from_static(b"{not json}");
        let err = parse_probe(&b).unwrap_err();
        assert!(matches!(err, ApiError::BadRequest(_)));
    }

    #[test]
    fn parse_probe_handles_nested_messages_with_stream_true() {
        // Well-formed object with nested arrays/objects (real chat-completions
        // payloads carry `messages: [{role, content: [{type, text}]}]`). The
        // two-step deserialize must not balk on this — only the top-level
        // object shape and the `stream`/`model` fields matter.
        let b = Bytes::from_static(
            br#"{
              "model": "x",
              "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
              "stream": true
            }"#,
        );
        assert_eq!(parse_probe(&b).unwrap().stream, Some(true));
    }

    #[test]
    fn parse_probe_handles_nested_messages_with_stream_false() {
        let b = Bytes::from_static(
            br#"{
              "model": "x",
              "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
              "stream": false
            }"#,
        );
        assert_eq!(parse_probe(&b).unwrap().stream, Some(false));
    }

    #[test]
    fn parse_probe_handles_duplicate_stream_keys() {
        // RFC 8259 says "names within an object SHOULD be unique" but a
        // parser MAY accept duplicates. Step 1 (HashMap) silently
        // last-wins, but step 2 deserializes into the typed `RequestProbe`
        // struct, and `serde_json`'s `#[derive(Deserialize)]` REJECTS
        // duplicate fields with a `duplicate field` error.
        //
        // We map that to `BadRequest` (same path as other malformed input).
        // Pinning "reject" rather than "last-wins" is intentional —
        // ambiguous bodies should fail loudly at the edge, not silently
        // route based on which copy serde happened to see last.
        let b = Bytes::from_static(br#"{"stream": true, "stream": false}"#);
        let err = parse_probe(&b).unwrap_err();
        match err {
            ApiError::BadRequest(_) => {}
            other => panic!("expected BadRequest on duplicate `stream` key, got {other:?}"),
        }
    }

    #[test]
    fn parse_probe_bad_request_message_does_not_leak_serde_detail() {
        // Info-leak guard: the client-visible message must be a fixed
        // string, not the serde error (which can contain line/column
        // detail or hint at field shape).
        let b = Bytes::from_static(br#"{"stream": "not-a-bool"}"#);
        let err = parse_probe(&b).unwrap_err();
        match err {
            ApiError::BadRequest(msg) => assert_eq!(
                msg, "invalid request: body must be a JSON object",
                "client-visible message must be fixed; got: {msg}"
            ),
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }
}
