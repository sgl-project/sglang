use std::io;

use axum::Json;
use axum::body::Body;
use axum::http::{HeaderMap, StatusCode, header, request::Parts};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures::{StreamExt, future::join_all, stream::select_all};
use serde_json::{Value, json};

use super::{DpState, error, routing};
use crate::api_server::headers;
use crate::api_server::native_api::native_error;
use crate::message::request::GenerateBody;

fn origin_form(parts: &Parts) -> &str {
    parts.uri.path_and_query().map_or("/", |path| path.as_str())
}

fn end_to_end_headers(headers: &HeaderMap) -> HeaderMap {
    let mut forwarded = headers.clone();
    for value in headers.get_all(header::CONNECTION) {
        if let Ok(value) = value.to_str() {
            for name in value.split(',').map(str::trim) {
                forwarded.remove(name);
            }
        }
    }
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        super::super::common::ABORT_DISPATCHED_HEADER,
    ] {
        forwarded.remove(name);
    }
    forwarded
}

async fn send(
    state: &DpState,
    rank: usize,
    parts: &Parts,
    body: Bytes,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut headers = end_to_end_headers(&parts.headers);
    headers.remove(header::HOST);
    headers.remove(header::CONTENT_LENGTH);
    state
        .client
        .request(
            parts.method.clone(),
            format!("{}{}", state.workers[rank], origin_form(parts)),
        )
        .headers(headers)
        .body(body)
        .send()
        .await
}

fn relay(mut upstream: reqwest::Response) -> Response {
    let status = upstream.status();
    let headers = end_to_end_headers(upstream.headers());
    let stream = async_stream::stream! {
        loop {
            match upstream.chunk().await {
                Ok(Some(chunk)) => yield Ok::<_, reqwest::Error>(chunk),
                Ok(None) => break,
                Err(error) => { yield Err(error); break; }
            }
        }
    };
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

pub(super) async fn one_worker(
    state: &DpState,
    rank: usize,
    parts: Parts,
    body: Bytes,
) -> Response {
    match send(state, rank, &parts, body).await {
        Ok(response) => relay(response),
        Err(e) => error(StatusCode::BAD_GATEWAY, e),
    }
}

pub(super) async fn all_workers(state: &DpState, parts: Parts, body: Bytes) -> Response {
    let results =
        join_all((0..state.workers.len()).map(|rank| send(state, rank, &parts, body.clone())))
            .await;
    if parts.uri.path() == "/abort_request"
        && let Some(metrics) = &state.metrics
        && results.iter().any(|result| {
            result.as_ref().is_ok_and(|response| {
                response
                    .headers()
                    .get(super::super::common::ABORT_DISPATCHED_HEADER)
                    .is_some_and(|value| value == "1")
            })
        })
    {
        metrics.aborted();
    }
    let mut first = None;
    for result in results {
        match result {
            Ok(response) if response.status().is_success() => {
                first.get_or_insert(response);
            }
            Ok(response) => return relay(response),
            Err(e) => return error(StatusCode::BAD_GATEWAY, e),
        }
    }
    match first {
        Some(response) => relay(response),
        None => error(StatusCode::SERVICE_UNAVAILABLE, "no DP workers"),
    }
}

pub(super) async fn generate(state: &DpState, mut parts: Parts, bytes: Bytes) -> Response {
    let mut body: GenerateBody = match serde_json::from_slice(&bytes) {
        Ok(body) => body,
        Err(e) => return native_error(StatusCode::BAD_REQUEST, &e.to_string(), false),
    };
    if state.header_overrides {
        if let Err(e) = headers::apply_overrides(&mut body, &parts.headers) {
            return error(StatusCode::BAD_REQUEST, e);
        }
        // Overrides have already participated in batch normalization. Reapplying
        // a scalar RID/room on each worker would erase its per-item identity.
        let names: Vec<_> = parts
            .headers
            .keys()
            .filter(|name| name.as_str().starts_with("x-override-"))
            .cloned()
            .collect();
        for name in names {
            parts.headers.remove(name);
        }
    }
    let stream = body.stream;
    if let Some(preferred) = &state.preferred_sampling
        && let Err(e) = body.apply_preferred_sampling(preferred)
    {
        return native_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string(), stream);
    }
    let (requests, batch) = match body.into_requests() {
        Ok(requests) => requests,
        Err(e) => return native_error(StatusCode::BAD_REQUEST, &e.to_string(), stream),
    };
    let mut submissions = Vec::with_capacity(requests.len());
    for request in requests {
        let rank = match routing::choose(
            state,
            request.routed_dp_rank,
            request.bootstrap_room,
            request.input_ids.as_ref().map_or(0, |ids| ids.len() as u64),
        ) {
            Ok(rank) => rank,
            Err(e) => return native_error(e.status, &e.message, stream),
        };
        let body = match serde_json::to_vec(&request) {
            Ok(body) => Bytes::from(body),
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        };
        submissions.push((rank, body));
    }
    parts.headers.insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    let responses = join_all(
        submissions
            .into_iter()
            .map(|(rank, body)| send(state, rank, &parts, body)),
    )
    .await;
    let responses = match responses.into_iter().collect::<Result<Vec<_>, _>>() {
        Ok(responses) => responses,
        Err(e) => return native_error(StatusCode::BAD_GATEWAY, &e.to_string(), stream),
    };
    if !batch {
        return match responses.into_iter().next() {
            Some(response) => relay(response),
            None => error(StatusCode::INTERNAL_SERVER_ERROR, "empty generation batch"),
        };
    }
    if stream {
        let streams: Vec<_> = responses
            .into_iter()
            .enumerate()
            .map(|(index, response)| Box::pin(indexed_stream(index, response)))
            .collect();
        let stream = async_stream::stream! {
            let mut streams = select_all(streams);
            while let Some(frame) = streams.next().await {
                yield frame;
            }
            yield Ok::<_, io::Error>(Bytes::from_static(b"data: [DONE]\n\n"));
        };
        return (
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            Body::from_stream(stream),
        )
            .into_response();
    }
    let outputs = join_all(responses.into_iter().map(|response| async {
        let bytes = response.bytes().await.map_err(|e| e.to_string())?;
        serde_json::from_slice::<Value>(&bytes).map_err(|e| e.to_string())
    }))
    .await;
    match outputs.into_iter().collect::<Result<Vec<_>, _>>() {
        Ok(outputs) => Json(outputs).into_response(),
        Err(e) => error(StatusCode::BAD_GATEWAY, e),
    }
}

fn indexed_stream(
    index: usize,
    mut response: reqwest::Response,
) -> impl futures::Stream<Item = Result<Bytes, io::Error>> {
    async_stream::try_stream! {
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let bytes = response.bytes().await.map_err(io::Error::other)?;
            let mut value = serde_json::from_slice::<Value>(&bytes).unwrap_or_else(|_| {
                json!({ "error": { "message": String::from_utf8_lossy(&bytes), "code": status } })
            });
            value["index"] = index.into();
            yield Bytes::from(format!("data: {value}\n\n"));
            return;
        }
        let mut pending = Vec::new();
        let mut done = false;
        while let Some(chunk) = response.chunk().await.map_err(io::Error::other)? {
            pending.extend_from_slice(&chunk);
            while let Some((end, delimiter)) = frame_end(&pending) {
                let frame: Vec<u8> = pending.drain(..end + delimiter).collect();
                let Some(data) = frame[..end].strip_prefix(b"data:") else { continue; };
                let data = std::str::from_utf8(data).map_err(io::Error::other)?.trim();
                if data == "[DONE]" { done = true; continue; }
                let mut value: Value = serde_json::from_str(data).map_err(io::Error::other)?;
                let object = value.as_object_mut().ok_or_else(||
                    io::Error::other("generation stream frame must be an object"))?;
                object.insert("index".into(), index.into());
                yield Bytes::from(format!("data: {value}\n\n"));
            }
        }
        if !done || !pending.is_empty() {
            Err(io::Error::new(io::ErrorKind::UnexpectedEof, "DP worker stream ended before [DONE]"))?;
        }
    }
}

fn frame_end(bytes: &[u8]) -> Option<(usize, usize)> {
    let lf = bytes
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|end| (end, 2));
    let crlf = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|end| (end, 4));
    [lf, crlf].into_iter().flatten().min_by_key(|(end, _)| *end)
}

pub(super) async fn loads(state: &DpState, mut parts: Parts) -> Response {
    let legacy = parts.uri.path() == "/get_load";
    let mut uri = match reqwest::Url::parse(&format!("http://localhost{}", origin_form(&parts))) {
        Ok(uri) => uri,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let prometheus = uri
        .query_pairs()
        .any(|(key, value)| key == "format" && value == "prometheus");
    let query: Vec<_> = uri
        .query_pairs()
        .filter(|(key, _)| key != "format")
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect();
    uri.set_query(None);
    uri.query_pairs_mut().extend_pairs(query);
    let path = match uri.query() {
        Some(query) => format!("{}?{query}", uri.path()),
        None => uri.path().to_owned(),
    };
    parts.uri = match path.parse() {
        Ok(uri) => uri,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let results =
        join_all((0..state.workers.len()).map(|rank| send(state, rank, &parts, Bytes::new())))
            .await;
    let mut first = None;
    let mut loads = Vec::new();
    let mut accelerators = 0;
    for result in results {
        let response = match result {
            Ok(response) if response.status().is_success() => response,
            Ok(response) => return relay(response),
            Err(e) => return error(StatusCode::BAD_GATEWAY, e),
        };
        let value = match response.bytes().await {
            Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
                Ok(value) => value,
                Err(e) => return error(StatusCode::BAD_GATEWAY, e),
            },
            Err(e) => return error(StatusCode::BAD_GATEWAY, e),
        };
        let items = if legacy {
            value.as_array()
        } else {
            value["loads"].as_array()
        };
        let Some(items) = items else {
            return error(StatusCode::BAD_GATEWAY, "invalid DP load response");
        };
        loads.extend_from_slice(items);
        accelerators += value["num_accelerators"].as_u64().unwrap_or(0);
        first.get_or_insert(value);
    }
    loads.sort_by_key(|load| load["dp_rank"].as_u64());
    if legacy {
        return Json(loads).into_response();
    }
    if prometheus {
        return (
            [(
                header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            crate::api_server::loads::prometheus(&loads),
        )
            .into_response();
    }
    let Some(mut output) = first else {
        return error(StatusCode::SERVICE_UNAVAILABLE, "no DP workers");
    };
    output["loads"] = loads.into();
    output["num_accelerators"] = accelerators.into();
    Json(output).into_response()
}

pub(super) async fn input_metadata(state: &DpState, parts: Parts) -> Response {
    let uri = match reqwest::Url::parse(&format!("http://localhost{}", origin_form(&parts))) {
        Ok(uri) => uri,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    if let Some((_, rank)) = uri.query_pairs().find(|(key, _)| key == "dp_rank") {
        let rank = match rank.parse::<usize>() {
            Ok(rank) if rank < state.workers.len() => rank,
            _ => return error(StatusCode::BAD_REQUEST, "metadata dp_rank is out of range"),
        };
        return one_worker(state, rank, parts, Bytes::new()).await;
    }
    let results =
        join_all((0..state.workers.len()).map(|rank| send(state, rank, &parts, Bytes::new())))
            .await;
    let mut pending = None;
    for result in results {
        match result {
            Ok(response) if response.status() == StatusCode::OK => return relay(response),
            Ok(response) => {
                pending.get_or_insert(response);
            }
            Err(e) => return error(StatusCode::BAD_GATEWAY, e),
        }
    }
    match pending {
        Some(response) => relay(response),
        None => error(StatusCode::SERVICE_UNAVAILABLE, "no DP workers"),
    }
}
