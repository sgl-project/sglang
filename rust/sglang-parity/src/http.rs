//! HTTP capture without interpreting API-specific fields or stream sentinels.

use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use futures::StreamExt;
use reqwest::{Client, Method};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::AsyncWriteExt;

use crate::compare::{ComparisonScope, Violation};
use crate::sse::{SseDecoder, SseEvent};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CaptureMode {
    Json,
    Sse,
}

/// One resolved request; transport mode and comparison scope come from its suite.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HttpCase {
    pub name: String,
    pub method: String,
    pub path: String,
    pub body: Value,
    pub expect_status: u16,
    pub capture: CaptureMode,
    pub comparison_scope: ComparisonScope,
    #[serde(default)]
    pub equivalence_group: Option<String>,
}

/// Captured bytes remain available even when a response is malformed or truncated.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct HttpObservation {
    pub status: Option<u16>,
    pub headers: BTreeMap<String, String>,
    pub raw_body: PathBuf,
    pub json: Option<Value>,
    pub events: Vec<SseEvent>,
    pub transport_error: Option<String>,
    pub violations: Vec<Violation>,
}

pub(crate) fn client() -> Result<Client, reqwest::Error> {
    Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
}

/// Collect a complete response within one deadline, retaining partial data on failure.
///
/// `request` is the same serialized body saved in the request artifact and sent to
/// both implementations. Only filesystem failures escape as errors; connection
/// failures and malformed responses are recorded in the observation.
pub(crate) async fn capture(
    client: &Client,
    base_url: &str,
    case: &HttpCase,
    request: &[u8],
    raw_body: &Path,
    timeout: Duration,
) -> std::io::Result<HttpObservation> {
    let mut observation = HttpObservation {
        raw_body: raw_body.to_path_buf(),
        ..HttpObservation::default()
    };
    let mut file = tokio::fs::File::create(raw_body).await?;
    let result = tokio::time::timeout(timeout, async {
        let method = Method::from_bytes(case.method.as_bytes())
            .expect("request method validated before execution");
        let response = match client
            .request(method, format!("{base_url}{}", case.path))
            .header("content-type", "application/json")
            .body(request.to_owned())
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                observation.transport_error = Some(error.to_string());
                return Ok::<(), std::io::Error>(());
            }
        };
        observation.status = Some(response.status().as_u16());
        for name in ["content-type", "content-length"] {
            if let Some(value) = response.headers().get(name) {
                observation.headers.insert(
                    name.into(),
                    String::from_utf8_lossy(value.as_bytes()).into_owned(),
                );
            }
        }
        if observation.status != Some(case.expect_status) {
            observation.violations.push(Violation::new(
                "/status",
                format!(
                    "expected {}, received {}",
                    case.expect_status,
                    response.status()
                ),
            ));
        }
        let media_type = observation
            .headers
            .get("content-type")
            .and_then(|header| header.split(';').next())
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        let valid_type = match case.capture {
            CaptureMode::Json => {
                media_type == "application/json"
                    || (media_type.starts_with("application/") && media_type.ends_with("+json"))
            }
            CaptureMode::Sse => media_type == "text/event-stream",
        };
        if !valid_type {
            observation.violations.push(Violation::new(
                "/headers/content-type",
                format!(
                    "unexpected media type {media_type:?} for {:?}",
                    case.capture
                ),
            ));
        }
        let mut decoder = SseDecoder::default();
        let mut parse_failed = false;
        let mut json_bytes = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    observation.transport_error = Some(error.to_string());
                    return Ok(());
                }
            };
            // Persist bytes before interpreting them, including malformed events.
            file.write_all(&chunk).await?;
            match case.capture {
                CaptureMode::Json => json_bytes.extend_from_slice(&chunk),
                CaptureMode::Sse if !parse_failed => match decoder.push(&chunk) {
                    Ok(events) => observation.events.extend(events),
                    Err(error) => {
                        observation.events.extend(decoder.take_completed());
                        observation
                            .violations
                            .push(Violation::new("/events", error));
                        parse_failed = true;
                    }
                },
                CaptureMode::Sse => {}
            }
        }
        match case.capture {
            CaptureMode::Json => match serde_json::from_slice(&json_bytes) {
                Ok(json) => observation.json = Some(json),
                Err(error) => observation
                    .violations
                    .push(Violation::new("/body", error.to_string())),
            },
            CaptureMode::Sse if !parse_failed => match decoder.finish() {
                Ok(events) => observation.events.extend(events),
                Err(error) => observation
                    .violations
                    .push(Violation::new("/events", error)),
            },
            CaptureMode::Sse => {}
        }
        Ok(())
    })
    .await;
    match result {
        Ok(result) => result?,
        Err(_) => {
            observation.transport_error =
                Some(format!("response exceeded {} seconds", timeout.as_secs()))
        }
    }
    file.flush().await?;
    Ok(observation)
}
