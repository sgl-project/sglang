//! HTTP client from renderer-owned generation requests to SGLang `/generate`.

use std::time::Duration;

use async_stream::stream;
use futures::{StreamExt, future::BoxFuture};

use super::{GenerateTransport, TokenStream, internal};
use crate::{GenerateRequest, ResponseError};
use protocol::{engine_error_message, normalize_engine_output, parse_engine_frame};

mod protocol;

// SGLang's deep health probe defaults to 20 seconds. Leave it time to return
// its own status while still bounding a peer that never sends response headers.
const ENGINE_HEALTH_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

fn unavailable(message: impl Into<String>) -> ResponseError {
    ResponseError {
        kind: crate::ResponseErrorKind::Unavailable,
        message: message.into(),
    }
}

#[derive(Clone)]
pub struct HttpGenerateClient {
    client: reqwest::Client,
    generate_url: reqwest::Url,
    health_url: reqwest::Url,
    health_timeout: Duration,
}

impl HttpGenerateClient {
    pub fn new(engine_url: impl AsRef<str>) -> Result<Self, String> {
        let engine_url = engine_url.as_ref();
        let base_url = reqwest::Url::parse(engine_url)
            .map_err(|error| format!("invalid engine URL {engine_url:?}: {error}"))?;
        let is_http_origin = matches!(base_url.scheme(), "http" | "https")
            && base_url.host_str().is_some()
            && base_url.username().is_empty()
            && base_url.password().is_none()
            && base_url.path() == "/"
            && base_url.query().is_none()
            && base_url.fragment().is_none();
        if !is_http_origin {
            return Err(format!(
                "invalid engine URL {engine_url:?}: expected an HTTP(S) origin without credentials, a path, query, or fragment"
            ));
        }
        let generate_url = base_url
            .join("/generate")
            .map_err(|error| format!("joining /generate to engine URL failed: {error}"))?;
        let health_url = base_url
            .join("/health")
            .map_err(|error| format!("joining /health to engine URL failed: {error}"))?;
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|error| format!("building engine HTTP client failed: {error}"))?;
        Ok(Self {
            client,
            generate_url,
            health_url,
            health_timeout: ENGINE_HEALTH_REQUEST_TIMEOUT,
        })
    }

    #[cfg(test)]
    pub(crate) fn with_health_timeout(mut self, timeout: Duration) -> Self {
        self.health_timeout = timeout;
        self
    }

    pub(crate) async fn health_status(&self) -> Result<reqwest::StatusCode, ResponseError> {
        let request = self
            .client
            .get(self.health_url.clone())
            .timeout(self.health_timeout);
        let response = request
            .send()
            .await
            .map_err(|error| unavailable(format!("engine health check failed: {error}")))?;
        Ok(response.status())
    }
}

impl GenerateTransport for HttpGenerateClient {
    fn generate(
        &self,
        mut request: GenerateRequest,
    ) -> BoxFuture<'_, Result<TokenStream, ResponseError>> {
        Box::pin(async move {
            // Always consume token deltas, including for unary frontend requests.
            request.stream = true;

            let response = self
                .client
                .post(self.generate_url.clone())
                .json(&request)
                .send()
                .await
                .map_err(|error| unavailable(format!("engine request failed: {error}")))?;
            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                return Err(ResponseError {
                    kind: crate::ResponseErrorKind::Upstream(crate::UpstreamErrorCode::Http(
                        status.as_u16(),
                    )),
                    message: engine_error_message(&body)
                        .unwrap_or_else(|| format!("engine returned HTTP {status}")),
                });
            }

            let mut chunks = response.bytes_stream();
            let events = stream! {
                let mut parser = SseParser::default();
                let mut terminal = false;
                let mut emitted_tokens = 0;
                while let Some(chunk) = chunks.next().await {
                    let chunk = match chunk {
                        Ok(chunk) => chunk,
                        Err(error) => {
                            yield Err(unavailable(format!("engine stream failed: {error}")));
                            return;
                        }
                    };
                    for payload in parser.push(&chunk) {
                        if payload == "[DONE]" {
                            if !terminal {
                                yield Err(internal("engine stream ended before a terminal frame"));
                            }
                            return;
                        }
                        let mut output = match parse_engine_frame(&payload) {
                            Ok(output) => output,
                            Err(error) => {
                                yield Err(error);
                                return;
                            }
                        };
                        if let Err(error) = normalize_engine_output(&mut output, &mut emitted_tokens) {
                            yield Err(error);
                            return;
                        }
                        terminal = output.finish_reason.is_some();
                        yield Ok(output);
                    }
                }
                if !terminal {
                    yield Err(internal("engine response closed before [DONE]"));
                }
            }
            .boxed();

            Ok(events)
        })
    }
}

#[derive(Default)]
struct SseParser {
    bytes: Vec<u8>,
}

impl SseParser {
    fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.bytes.extend_from_slice(chunk);
        let mut payloads = Vec::new();
        while let Some((end, separator_len)) = event_end(&self.bytes) {
            let event = self.bytes.drain(..end).collect::<Vec<_>>();
            self.bytes.drain(..separator_len);
            let event = String::from_utf8_lossy(&event);
            let data = event
                .lines()
                .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
                .collect::<Vec<_>>()
                .join("\n");
            if !data.is_empty() {
                payloads.push(data);
            }
        }
        payloads
    }
}

fn event_end(bytes: &[u8]) -> Option<(usize, usize)> {
    let crlf = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| (position, 4));
    let lf = bytes
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|position| (position, 2));
    match (crlf, lf) {
        (Some(crlf), Some(lf)) => Some(crlf.min(lf)),
        (Some(crlf), None) => Some(crlf),
        (None, Some(lf)) => Some(lf),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::test_utils::tiny_tokenizer;
    use crate::{GenerationOptions, TokenIds, TokenIdsRequest};
    use axum::{
        Json, Router,
        extract::State,
        response::sse::{Event, Sse},
        routing::post,
    };
    use std::convert::Infallible;
    use std::sync::{Arc, Mutex};

    #[test]
    fn sse_parser_handles_split_crlf_and_lf_frames() {
        let mut parser = SseParser::default();
        assert!(parser.push(b"data: {\"a\":1}\r\n").is_empty());
        assert_eq!(
            parser.push(b"\r\ndata: [DONE]\n\n"),
            ["{\"a\":1}", "[DONE]"]
        );
    }

    #[test]
    fn sse_parser_uses_the_earliest_mixed_delimiter() {
        let mut parser = SseParser::default();

        let payloads = parser.push(b"data: {\"a\":1}\n\ndata: {\"b\":2}\r\n\r\n");

        assert_eq!(payloads, ["{\"a\":1}", "{\"b\":2}"]);
    }

    #[derive(Clone)]
    struct EngineState {
        requests: Arc<Mutex<Vec<serde_json::Value>>>,
        output_ids: TokenIds,
    }

    async fn generate(
        State(state): State<EngineState>,
        Json(body): Json<serde_json::Value>,
    ) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
        state.requests.lock().unwrap().push(body);
        let frame = serde_json::json!({
            "output_ids": state.output_ids,
            "meta_info": {
                "prompt_tokens": 1,
                "completion_tokens": state.output_ids.len(),
                "finish_reason": {"type": "stop", "matched": null}
            }
        })
        .to_string();
        Sse::new(futures::stream::iter([
            Ok(Event::default().data(frame)),
            Ok(Event::default().data("[DONE]")),
        ]))
    }

    async fn streaming_generate(
        State(cumulative): State<bool>,
    ) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
        let frame = |completion_tokens, finish_reason: serde_json::Value| {
            Event::default().data(
                serde_json::json!({
                    "output_ids": if cumulative { vec![104; completion_tokens] } else { vec![104] },
                    "meta_info": {
                        "prompt_tokens": 1,
                        "completion_tokens": completion_tokens,
                        "finish_reason": finish_reason,
                    }
                })
                .to_string(),
            )
        };
        Sse::new(futures::stream::iter([
            Ok(frame(1, serde_json::Value::Null)),
            Ok(frame(2, serde_json::json!({"type": "length", "length": 2}))),
            Ok(Event::default().data("[DONE]")),
        ]))
    }

    #[test]
    fn engine_origins_are_validated_and_joined_during_client_construction() {
        for invalid_url in [
            "127.0.0.1:30001",
            "ftp://engine.example",
            "http://user@engine.example",
            "http://engine.example/base",
            "http://engine.example?query",
            "http://engine.example#fragment",
        ] {
            let error = match HttpGenerateClient::new(invalid_url) {
                Ok(_) => panic!("{invalid_url:?} must be rejected"),
                Err(error) => error,
            };
            assert!(error.contains("invalid engine URL"));
        }

        let client = HttpGenerateClient::new("http://engine.example:30001/").unwrap();
        assert_eq!(
            client.generate_url.as_str(),
            "http://engine.example:30001/generate"
        );
        assert_eq!(
            client.health_url.as_str(),
            "http://engine.example:30001/health"
        );
    }

    #[tokio::test]
    async fn backend_posts_token_ids_and_decodes_the_engine_stream() {
        let tokenizer = tiny_tokenizer();
        let output_ids = tokenizer
            .encode("hello")
            .unwrap()
            .token_ids()
            .iter()
            .map(|&id| id as i32)
            .collect::<Vec<_>>();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(
            axum::serve(
                listener,
                Router::new()
                    .route("/generate", post(generate))
                    .with_state(EngineState {
                        requests: requests.clone(),
                        output_ids: output_ids.clone(),
                    }),
            )
            .into_future(),
        );

        let client = HttpGenerateClient::new(format!("http://{address}")).unwrap();
        let request = TokenIdsRequest {
            rid: "client-request".into(),
            input_ids: vec![65],
            options: GenerationOptions {
                return_text_in_logprobs: Some(true),
                ..Default::default()
            },
            metadata: Default::default(),
        };
        let service = crate::engine::GenerationService::new(
            Arc::new(client),
            crate::engine::TokenDecoder::new(tokenizer.clone()),
        );
        let mut events = service.generate(request.into()).await.unwrap();
        let output = events.next().await.unwrap().unwrap();
        assert!(output.finish_reason.is_some());

        let mut expected_decoder = tokenizer.decode_stream(&[65], true);
        let mut expected = String::new();
        for id in output_ids {
            if let Some(delta) = expected_decoder.step(id as u32).unwrap() {
                expected.push_str(&delta);
            }
        }
        assert_eq!(output.text, expected);
        assert_eq!(output.prompt_tokens, 1);
        let request = requests.lock().unwrap().pop().unwrap();
        assert_eq!(request["rid"], "client-request");
        assert_eq!(request["input_ids"], serde_json::json!([65]));
        assert_eq!(request["stream"], true);
        assert!(request.get("incremental_streaming_output").is_none());
        assert_eq!(request["return_text_in_logprobs"], false);
        server.abort();
    }

    #[tokio::test]
    async fn transport_requires_a_terminal_frame_and_rejects_malformed_output() {
        async fn scripted(
            Json(request): Json<serde_json::Value>,
        ) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
            let case = request["rid"].as_str().unwrap();
            let terminal = case == "terminal-eof";
            let frame = serde_json::json!({
                "output_ids": [],
                "meta_info": {
                    "prompt_tokens": 1,
                    "completion_tokens": 0,
                    "finish_reason": if terminal { serde_json::json!({"type": "length"}) } else { serde_json::Value::Null },
                }
            }).to_string();
            let frames = match case {
                "malformed" => vec!["{".to_owned()],
                "early-done" => vec![frame, "[DONE]".to_owned()],
                _ => vec![frame],
            };
            Sse::new(futures::stream::iter(
                frames
                    .into_iter()
                    .map(|frame| Ok(Event::default().data(frame))),
            ))
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(
            axum::serve(listener, Router::new().route("/generate", post(scripted))).into_future(),
        );
        let client = HttpGenerateClient::new(format!("http://{address}")).unwrap();
        for (case, error_message) in [
            ("malformed", Some("invalid engine frame")),
            (
                "early-done",
                Some("engine stream ended before a terminal frame"),
            ),
            (
                "unfinished-eof",
                Some("engine response closed before [DONE]"),
            ),
            ("terminal-eof", None),
        ] {
            let request = TokenIdsRequest {
                rid: case.into(),
                input_ids: vec![65],
                options: GenerationOptions::default(),
                metadata: Default::default(),
            };
            let events = client
                .generate(request.into())
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await;
            if let Some(message) = error_message {
                let error = events.last().unwrap().as_ref().unwrap_err();
                assert_eq!(error.kind, crate::ResponseErrorKind::Internal);
                assert!(
                    error.message.starts_with(message),
                    "{case}: {}",
                    error.message
                );
                assert_eq!(events.iter().filter(|event| event.is_err()).count(), 1);
            } else {
                assert_eq!(events.len(), 1);
                assert!(events[0].as_ref().unwrap().finish_reason.is_some());
            }
        }
        server.abort();
    }

    #[tokio::test]
    async fn engine_frames_are_forwarded_once() {
        for cumulative in [false, true] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(
                axum::serve(
                    listener,
                    Router::new()
                        .route("/generate", post(streaming_generate))
                        .with_state(cumulative),
                )
                .into_future(),
            );

            let client = HttpGenerateClient::new(format!("http://{address}")).unwrap();
            let mut events = client
                .generate(
                    TokenIdsRequest {
                        rid: "incremental".into(),
                        input_ids: vec![65],
                        options: GenerationOptions::default(),
                        metadata: Default::default(),
                    }
                    .into(),
                )
                .await
                .unwrap();

            let first = events.next().await.unwrap().unwrap();
            assert!(first.finish_reason.is_none());
            assert_eq!(first.token_ids, [104]);
            assert_eq!(first.completion_tokens, 1);

            let second = events.next().await.unwrap().unwrap();
            assert!(second.finish_reason.is_some());
            assert_eq!(second.token_ids, [104]);
            assert_eq!(second.completion_tokens, 1);
            assert!(events.next().await.is_none());
            server.abort();
        }
    }

    struct DropNotice(Option<tokio::sync::oneshot::Sender<()>>);

    impl Drop for DropNotice {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }

    async fn slow_generate(
        State(notice): State<Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>>,
    ) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
        let guard = DropNotice(notice.lock().unwrap().take());
        Sse::new(stream! {
            let _guard = guard;
            yield Ok(Event::default().data(serde_json::json!({
                "output_ids": [104],
                "meta_info": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "finish_reason": null
                }
            }).to_string()));
            futures::future::pending::<()>().await;
        })
    }

    #[tokio::test]
    async fn dropping_renderer_events_closes_the_engine_stream() {
        let (notice_tx, notice_rx) = tokio::sync::oneshot::channel();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(
            axum::serve(
                listener,
                Router::new()
                    .route("/generate", post(slow_generate))
                    .with_state(Arc::new(Mutex::new(Some(notice_tx)))),
            )
            .into_future(),
        );
        let client = HttpGenerateClient::new(format!("http://{address}")).unwrap();
        let request = TokenIdsRequest {
            rid: "cancel-me".into(),
            input_ids: vec![65],
            options: GenerationOptions::default(),
            metadata: Default::default(),
        };
        let mut events = client.generate(request.into()).await.unwrap();
        assert!(events.next().await.is_some());
        drop(events);
        tokio::time::timeout(Duration::from_secs(2), notice_rx)
            .await
            .expect("engine response stream was not dropped")
            .unwrap();
        server.abort();
    }
}
