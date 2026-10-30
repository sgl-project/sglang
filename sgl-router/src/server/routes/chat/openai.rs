// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! The OpenAI response to a request served through the engine's `/generate`.

use axum::body::Body;
use axum::http::{header, HeaderValue, Response, StatusCode};
use bytes::{Bytes, BytesMut};
use futures::{stream, Stream, StreamExt};
use serde_json::Value;
use sglang_api_types::api::v1::{GenerateResponse, GenerateStreamError};
use sglang_processor::openai::{Reply, Responder};

pub(super) async fn respond(
    responder: &mut Option<Responder>,
    response: Response<Body>,
    streaming: bool,
) -> Response<Body> {
    let (mut parts, body) = response.into_parts();
    let status = parts.status.as_u16();
    if !streaming || status != 200 {
        let Ok(bytes) = axum::body::to_bytes(body, usize::MAX).await else {
            let mut response = Response::new(Body::empty());
            *response.status_mut() = StatusCode::BAD_GATEWAY;
            return response;
        };
        // Keep the responder for a retry when the engine rejects this attempt.
        let reply = if status != 200 {
            serde_json::from_slice(&bytes)
                .and_then(generate_result)
                .ok()
                .and_then(Result::err)
                .map(Responder::rejected)
        } else {
            unary_reply(responder.take().expect("OpenAI responder"), &bytes)
        };
        // A reply the OpenAI layer would not have made passes through as the engine sent it.
        return match reply {
            Some(reply) => json_response(reply),
            None => Response::from_parts(parts, Body::from(bytes)),
        };
    }

    let mut responder = responder.take().expect("OpenAI responder");
    // Python answers only once the first OpenAI chunk exists, so an early
    // error is still an error status.
    let mut frames = Box::pin(sse_data(body).fuse());
    let mut first = Vec::new();
    let mut failed = None;
    let mut last = Bytes::new();
    while first.is_empty() && failed.is_none() {
        match frames.next().await {
            None => break,
            Some(Err(error)) => failed = Some(error),
            Some(Ok(data)) => match stream_data(&mut responder, &data) {
                Ok(events) => (first, last) = (events, data),
                Err(reply) => return json_response(reply),
            },
        }
    }
    // The pump's terminal error still ends the converted stream as an error.
    let pending = after(frames, responder, &last);
    let rest = stream::unfold(pending, |pending| async move {
        let (mut frames, responder) = pending?;
        let Some(mut responder) = responder else {
            while frames.next().await.is_some() {}
            return None;
        };
        let (events, data) = match frames.next().await? {
            Ok(data) => (
                stream_data(&mut responder, &data)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|e| Ok(Bytes::from(e)))
                    .collect(),
                data,
            ),
            Err(error) => (vec![Err(error)], Bytes::new()),
        };
        Some((stream::iter(events), after(frames, responder, &data)))
    })
    .flatten();
    let events = stream::iter(first.into_iter().map(|e| Ok(Bytes::from(e))))
        .chain(stream::iter(failed.map(Err)))
        .chain(rest);
    parts.headers.remove(header::CONTENT_LENGTH);
    Response::from_parts(parts, Body::from_stream(events))
}

fn generate_result(
    value: Value,
) -> serde_json::Result<Result<GenerateResponse, GenerateStreamError>> {
    if value.get("error").is_some() {
        serde_json::from_value(value).map(Err)
    } else {
        serde_json::from_value(value).map(Ok)
    }
}

fn unary_reply(responder: Responder, bytes: &[u8]) -> Option<Reply> {
    let items = match serde_json::from_slice(bytes).ok()? {
        Value::Array(items) => items,
        item => vec![item],
    };
    if items.iter().any(|item| {
        item.get("error").is_none() && item.get("text").and_then(Value::as_str).is_none()
    }) {
        return Some(Responder::malformed_output("text"));
    }
    let output = items
        .into_iter()
        .map(generate_result)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    responder.unary(output)
}

fn stream_data(responder: &mut Responder, data: &[u8]) -> Result<Vec<String>, Reply> {
    if data == b"[DONE]" {
        return Ok(responder.stream_end());
    }
    match serde_json::from_slice(data).and_then(generate_result) {
        Ok(output) => responder.stream_data(output),
        Err(_) => Ok(Vec::new()),
    }
}

/// What is left to read after `data`: more frames, the rest of a finished
/// stream (no responder), or nothing. Dropping the upstream aborts the engine
/// request, which Python does at an abort or error but not after `[DONE]`.
fn after<F>(frames: F, responder: Responder, data: &[u8]) -> Option<(F, Option<Responder>)> {
    if !responder.done() {
        Some((frames, Some(responder)))
    } else if data == b"[DONE]" {
        Some((frames, None))
    } else {
        None
    }
}

fn json_response(reply: Reply) -> Response<Body> {
    let mut response = Response::new(Body::from(reply.body));
    *response.status_mut() = reply.status.try_into().unwrap_or_default();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// The `data:` payloads of an SSE body, then its error if it broke.
fn sse_data(body: Body) -> impl Stream<Item = Result<Bytes, axum::Error>> {
    let chunks = body.into_data_stream();
    stream::unfold(
        (chunks, BytesMut::new(), 0),
        |(mut chunks, mut buffer, mut scanned)| async move {
            loop {
                if let Some(end) = buffer[scanned..].windows(2).position(|w| w == b"\n\n") {
                    let event = buffer.split_to(scanned + end + 2).freeze();
                    scanned = 0;
                    if let Some(data) = event.strip_prefix(b"data: ") {
                        let data = event.slice_ref(data.trim_ascii_end());
                        return Some((Ok(data), (chunks, buffer, scanned)));
                    }
                    continue;
                }
                // A delimiter may straddle the chunk boundary.
                scanned = buffer.len().saturating_sub(1);
                match chunks.next().await? {
                    Ok(chunk) => buffer.extend_from_slice(&chunk),
                    Err(error) => return Some((Err(error), (chunks, BytesMut::new(), 0))),
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use sglang_processor::openai::{lower_completion, OpenAiHeaders, OpenAiSettings};

    fn responder() -> Responder {
        let (_, responder) = lower_completion(
            br#"{"model":"m","prompt":"hi"}"#,
            &OpenAiHeaders::default(),
            &OpenAiSettings::default(),
            None,
        )
        .unwrap();
        Responder::Completion(Box::new(responder))
    }

    #[test]
    fn unary_decoding_preserves_engine_errors_and_malformed_text() {
        for (body, status, message) in [
            (r#"{"error":{"message":"busy"}}"#, 400, "busy"),
            (
                r#"[{"text":"ok"},{"error":{"message":"busy"},"index":1}]"#,
                400,
                "busy",
            ),
            (r#"{}"#, 500, "Internal server error: 'text'"),
            (r#"{"text":7}"#, 500, "Internal server error: 'text'"),
        ] {
            let reply = unary_reply(responder(), body.as_bytes()).unwrap();
            assert_eq!(reply.status, status);
            let body: Value = serde_json::from_str(&reply.body).unwrap();
            assert_eq!(body["message"], message);
        }
    }

    #[test]
    fn streaming_errors_keep_their_timing_and_end_once() {
        let error = br#"{"error":{"message":"busy"}}"#;
        assert_eq!(
            stream_data(&mut responder(), error).unwrap_err().status,
            400
        );

        let mut responder = responder();
        let events =
            stream_data(&mut responder, br#"{"text":"hi","meta_info":{"id":"r"}}"#).unwrap();
        assert!(events[0].contains("hi"));
        let events = stream_data(&mut responder, error).unwrap();
        assert!(events[0].contains("busy"));
        assert_eq!(events[1], "data: [DONE]\n\n");
        assert!(stream_data(&mut responder, b"[DONE]").unwrap().is_empty());
    }
}
