//! HTTP adapter from renderer-owned generation requests to SGLang `/generate`.

use std::sync::Arc;
use std::time::Duration;

use async_stream::stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use serde::Deserialize;

use crate::{
    DynamoTokenizer, FrontendError, GenerationEvent, GenerationFinishReason, GenerationOutput,
    GenerationOutputExtras, GenerationSubmission, InferenceBackend, InferenceSession, MatchedStop,
    PreparedGenerateRequest, TextTokenizer, TokenIds, TokenIdsRequest,
};

#[derive(Clone)]
pub struct HttpInferenceBackend {
    client: reqwest::Client,
    generate_url: Arc<str>,
    tokenizer: dynamo_tokenizers::Tokenizer,
    stop_tokenizer: Arc<DynamoTokenizer>,
    auto_specials: Arc<[i32]>,
}

impl HttpInferenceBackend {
    pub fn new(
        engine_url: impl AsRef<str>,
        tokenizer: dynamo_tokenizers::Tokenizer,
    ) -> Result<Self, String> {
        let base = engine_url.as_ref().trim_end_matches('/');
        if base.is_empty() {
            return Err("engine URL cannot be empty".into());
        }
        let generate_url: Arc<str> = format!("{base}/generate").into();
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|error| format!("building engine HTTP client failed: {error}"))?;
        let stop_tokenizer = Arc::new(DynamoTokenizer::new(tokenizer.clone()));
        let auto_specials = stop_tokenizer.auto_specials().into();
        Ok(Self {
            client,
            generate_url,
            tokenizer,
            stop_tokenizer,
            auto_specials,
        })
    }
}

pub struct HttpInferenceSession {
    backend: HttpInferenceBackend,
}

impl InferenceBackend for HttpInferenceBackend {
    type Session = HttpInferenceSession;

    fn begin_session(&self) -> Self::Session {
        HttpInferenceSession {
            backend: self.clone(),
        }
    }
}

impl InferenceSession for HttpInferenceSession {
    fn submit(
        &mut self,
        mut request: TokenIdsRequest,
        _stream: bool,
    ) -> BoxFuture<'_, Result<GenerationSubmission, FrontendError>> {
        Box::pin(async move {
            translate_text_stops(
                &mut request,
                self.backend.stop_tokenizer.as_ref(),
                &self.backend.auto_specials,
            )?;

            let prompt_ids = request
                .input_ids
                .iter()
                .map(|&id| u32::try_from(id))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| invalid("input_ids must be non-negative"))?;
            let skip_special_tokens = request.options.sampling_params.skip_special_tokens;
            let decode_logprob_text = request.options.return_text_in_logprobs.unwrap_or(false);
            let id = request.rid.clone();
            let mut wire = PreparedGenerateRequest::from(request);
            // The renderer needs token deltas even for a unary OpenAI request. It
            // collects them locally and dropping this response stream propagates
            // client cancellation to `/generate`.
            wire.stream = true;
            // The token-only engine cannot fill text columns. Decode them below.
            wire.return_text_in_logprobs = Some(false);

            let response = self
                .backend
                .client
                .post(self.backend.generate_url.as_ref())
                .json(&wire)
                .send()
                .await
                .map_err(|error| unavailable(format!("engine request failed: {error}")))?;
            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                return Err(FrontendError {
                    status_code: status.as_u16(),
                    message: engine_error_message(&body)
                        .unwrap_or_else(|| format!("engine returned HTTP {status}")),
                });
            }

            let tokenizer = self.backend.tokenizer.clone();
            let logprob_tokenizer = tokenizer.clone();
            let mut decoder = tokenizer.decode_stream(&prompt_ids, skip_special_tokens);
            let mut chunks = response.bytes_stream();
            let events = stream! {
                let mut parser = SseParser::default();
                let mut terminal = false;
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
                        output.text = match decode_ids(&mut decoder, &output.token_ids) {
                            Ok(text) => text,
                            Err(error) => {
                                yield Err(error);
                                return;
                            }
                        };
                        if decode_logprob_text {
                            fill_logprob_text(&logprob_tokenizer, output.extras.as_deref_mut());
                        }
                        terminal = output.finish_reason.is_some();
                        if terminal {
                            yield Ok(GenerationEvent::Done(output));
                        } else {
                            yield Ok(GenerationEvent::Frame(output));
                        }
                    }
                }
                if !terminal {
                    yield Err(internal("engine response closed before [DONE]"));
                }
            }
            .boxed();

            Ok(GenerationSubmission { id, events })
        })
    }

    fn detokenize(&mut self, token_ids: TokenIds) -> BoxFuture<'_, Result<String, FrontendError>> {
        Box::pin(async move {
            let ids = token_ids
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| invalid("token IDs must be non-negative"))?;
            self.backend
                .tokenizer
                .decode(&ids, true)
                .map(String::from)
                .map_err(|error| invalid(format!("detokenizing prompt failed: {error}")))
        })
    }

    fn complete(&mut self, _submission_id: &str) {}
}

fn translate_text_stops(
    request: &mut TokenIdsRequest,
    tokenizer: &dyn TextTokenizer,
    auto_specials: &[i32],
) -> Result<(), FrontendError> {
    let params = &mut request.options.sampling_params;
    if params.min_new_tokens > 0 {
        return Err(invalid(
            "min_tokens is not supported by a token-only SGLang engine",
        ));
    }
    if !params.stop_regex_strs.is_empty() {
        return Err(invalid(
            "stop_regex is not supported by a token-only SGLang engine",
        ));
    }

    let mut stop_token_ids = params.stop_token_ids.take().unwrap_or_default();
    for stop in std::mem::take(&mut params.stop_strs) {
        let mut ids = tokenizer
            .encode(&stop)
            .map_err(|error| invalid(format!("tokenizing stop {stop:?} failed: {error}")))?;
        if ids.starts_with(auto_specials) {
            ids.drain(..auto_specials.len());
        }
        if ids.len() != 1 {
            return Err(invalid(format!(
                "stop {stop:?} tokenizes to {} tokens; the token-only engine supports only single-token text stops",
                ids.len()
            )));
        }
        stop_token_ids.push(i64::from(ids[0]));
    }
    stop_token_ids.sort_unstable();
    stop_token_ids.dedup();
    params.stop_token_ids = (!stop_token_ids.is_empty()).then_some(stop_token_ids);
    params.stop = None;
    params.stop_regex = None;
    params.stop_regex_strs.clear();
    params.stop_str_max_len = 0;
    params.stop_regex_max_len = 0;
    Ok(())
}

fn decode_ids(
    decoder: &mut dynamo_tokenizers::DecodeStream,
    token_ids: &[i32],
) -> Result<String, FrontendError> {
    let mut text = String::new();
    for &id in token_ids {
        let id = u32::try_from(id).map_err(|_| internal("engine returned a negative token ID"))?;
        if let Some(delta) = decoder
            .step(id)
            .map_err(|error| internal(format!("detokenizing engine output failed: {error}")))?
        {
            text.push_str(&delta);
        }
    }
    Ok(text)
}

fn fill_logprob_text(
    tokenizer: &dynamo_tokenizers::Tokenizer,
    extras: Option<&mut GenerationOutputExtras>,
) {
    let Some(extras) = extras else { return };
    fill_texts(
        tokenizer,
        &extras.output_logprob_token_ids,
        &mut extras.output_logprob_text,
    );
    fill_texts(
        tokenizer,
        &extras.input_logprob_token_ids,
        &mut extras.input_logprob_text,
    );
    fill_texts(
        tokenizer,
        &extras.output_top_logprob_token_ids,
        &mut extras.output_top_logprob_text,
    );
    fill_texts(
        tokenizer,
        &extras.input_top_logprob_token_ids,
        &mut extras.input_top_logprob_text,
    );
}

fn fill_texts(tokenizer: &dynamo_tokenizers::Tokenizer, ids: &[i32], texts: &mut Vec<String>) {
    if texts.len() == ids.len() {
        return;
    }
    *texts = ids
        .iter()
        .map(|&id| {
            u32::try_from(id)
                .ok()
                .and_then(|id| tokenizer.decode(&[id], false).ok())
                .map(String::from)
                .unwrap_or_default()
        })
        .collect();
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
    bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| (position, 4))
        .or_else(|| {
            bytes
                .windows(2)
                .position(|window| window == b"\n\n")
                .map(|position| (position, 2))
        })
}

type WireLogprob = (Option<f32>, i32, Option<String>);
type WireTopLogprobs = Vec<Option<Vec<WireLogprob>>>;

#[derive(Deserialize)]
struct EngineFrame {
    #[serde(default)]
    output_ids: TokenIds,
    meta_info: EngineMeta,
}

#[derive(Deserialize)]
struct EngineMeta {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    finish_reason: Option<EngineFinishReason>,
    #[serde(default)]
    output_token_logprobs: Vec<WireLogprob>,
    #[serde(default)]
    input_token_logprobs: Vec<WireLogprob>,
    #[serde(default)]
    output_top_logprobs: WireTopLogprobs,
    #[serde(default)]
    input_top_logprobs: WireTopLogprobs,
}

#[derive(Deserialize)]
struct EngineFinishReason {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    matched: Option<EngineMatchedStop>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EngineMatchedStop {
    Token(i64),
    Text(String),
    Tokens(Vec<i64>),
}

#[derive(Deserialize)]
struct EngineErrorEnvelope {
    error: EngineError,
}

#[derive(Deserialize)]
struct EngineError {
    #[serde(default = "default_error_code")]
    code: u16,
    message: String,
}

fn default_error_code() -> u16 {
    500
}

fn parse_engine_frame(payload: &str) -> Result<GenerationOutput, FrontendError> {
    if let Ok(error) = serde_json::from_str::<EngineErrorEnvelope>(payload) {
        return Err(FrontendError {
            status_code: error.error.code,
            message: error.error.message,
        });
    }
    let frame: EngineFrame = serde_json::from_str(payload)
        .map_err(|error| internal(format!("invalid engine frame: {error}")))?;
    let finish_reason = frame
        .meta_info
        .finish_reason
        .map(|reason| match reason.kind.as_str() {
            "stop" => GenerationFinishReason::Stop(reason.matched.map(|matched| match matched {
                EngineMatchedStop::Token(id) => MatchedStop::Token(id),
                EngineMatchedStop::Text(text) => MatchedStop::Text(text),
                EngineMatchedStop::Tokens(ids) => MatchedStop::Tokens(ids),
            })),
            "length" => GenerationFinishReason::Length,
            "abort" => GenerationFinishReason::Abort,
            "content_filter" => GenerationFinishReason::ContentFilter,
            other => GenerationFinishReason::Other(other.to_owned()),
        });
    let has_extras = !frame.meta_info.output_token_logprobs.is_empty()
        || !frame.meta_info.input_token_logprobs.is_empty()
        || !frame.meta_info.output_top_logprobs.is_empty()
        || !frame.meta_info.input_top_logprobs.is_empty();
    let mut extras = GenerationOutputExtras::default();
    flatten_logprobs(
        frame.meta_info.output_token_logprobs,
        &mut extras.output_logprobs,
        &mut extras.output_logprob_token_ids,
        &mut extras.output_logprob_text,
    );
    flatten_logprobs(
        frame.meta_info.input_token_logprobs,
        &mut extras.input_logprobs,
        &mut extras.input_logprob_token_ids,
        &mut extras.input_logprob_text,
    );
    flatten_top_logprobs(
        frame.meta_info.output_top_logprobs,
        &mut extras.output_top_logprobs,
        &mut extras.output_top_logprob_token_ids,
        &mut extras.output_top_logprob_lengths,
        &mut extras.output_top_logprob_text,
    );
    flatten_top_logprobs(
        frame.meta_info.input_top_logprobs,
        &mut extras.input_top_logprobs,
        &mut extras.input_top_logprob_token_ids,
        &mut extras.input_top_logprob_lengths,
        &mut extras.input_top_logprob_text,
    );
    let extras = has_extras.then_some(Box::new(extras));
    Ok(GenerationOutput {
        text: String::new(),
        token_ids: frame.output_ids,
        finish_reason,
        prompt_tokens: frame.meta_info.prompt_tokens,
        completion_tokens: frame.meta_info.completion_tokens,
        extras,
    })
}

fn flatten_logprobs(
    values: Vec<WireLogprob>,
    logprobs: &mut Vec<f32>,
    token_ids: &mut TokenIds,
    texts: &mut Vec<String>,
) {
    for (value, token_id, text) in values {
        logprobs.push(value.unwrap_or(f32::NAN));
        token_ids.push(token_id);
        if let Some(text) = text {
            texts.push(text);
        }
    }
}

fn flatten_top_logprobs(
    values: WireTopLogprobs,
    logprobs: &mut Vec<f32>,
    token_ids: &mut TokenIds,
    lengths: &mut Vec<u32>,
    texts: &mut Vec<String>,
) {
    for position in values {
        let position = position.unwrap_or_default();
        lengths.push(u32::try_from(position.len()).unwrap_or(u32::MAX));
        flatten_logprobs(position, logprobs, token_ids, texts);
    }
}

fn engine_error_message(body: &str) -> Option<String> {
    serde_json::from_str::<EngineErrorEnvelope>(body)
        .ok()
        .map(|error| error.error.message)
}

fn invalid(message: impl Into<String>) -> FrontendError {
    FrontendError {
        status_code: 400,
        message: message.into(),
    }
}

fn unavailable(message: impl Into<String>) -> FrontendError {
    FrontendError {
        status_code: 503,
        message: message.into(),
    }
}

fn internal(message: impl Into<String>) -> FrontendError {
    FrontendError {
        status_code: 500,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenerationOptions, SamplingParams};
    use axum::{
        Json, Router,
        extract::State,
        response::sse::{Event, Sse},
        routing::post,
    };
    use std::convert::Infallible;
    use std::sync::Mutex;

    struct StopTokenizer;

    impl TextTokenizer for StopTokenizer {
        fn encode(&self, text: &str) -> Result<TokenIds, crate::RendererError> {
            Ok(match text {
                "<eos>" => vec![0, 9],
                "many" => vec![0, 4, 5],
                _ => vec![0, 7],
            })
        }

        fn auto_specials(&self) -> Vec<i32> {
            vec![0]
        }
    }

    fn request(stop: Vec<&str>) -> TokenIdsRequest {
        TokenIdsRequest {
            rid: "r".into(),
            input_ids: vec![1],
            options: GenerationOptions {
                sampling_params: SamplingParams {
                    stop_strs: stop.into_iter().map(str::to_owned).collect(),
                    ..Default::default()
                },
                ..Default::default()
            },
        }
    }

    #[test]
    fn single_token_stops_become_engine_token_stops() {
        let mut request = request(vec!["<eos>"]);
        translate_text_stops(&mut request, &StopTokenizer, &[0]).unwrap();
        assert_eq!(
            request.options.sampling_params.stop_token_ids,
            Some(vec![9])
        );
        assert!(request.options.sampling_params.stop_strs.is_empty());
    }

    #[test]
    fn multi_token_stops_are_rejected_before_submission() {
        let mut request = request(vec!["many"]);
        let error = translate_text_stops(&mut request, &StopTokenizer, &[0]).unwrap_err();
        assert_eq!(error.status_code, 400);
        assert!(error.message.contains("tokenizes to 2 tokens"));
    }

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
    fn engine_frame_maps_tokens_usage_finish_and_logprobs() {
        let output = parse_engine_frame(
            r#"{
                "output_ids":[7],
                "meta_info":{
                    "prompt_tokens":3,
                    "completion_tokens":1,
                    "finish_reason":{"type":"stop","matched":9},
                    "output_token_logprobs":[[-0.25,7,null]],
                    "output_top_logprobs":[[[-0.25,7,null],[-1.0,8,null]]]
                }
            }"#,
        )
        .unwrap();
        assert_eq!(output.token_ids, [7]);
        assert_eq!(output.prompt_tokens, 3);
        assert_eq!(output.completion_tokens, 1);
        assert_eq!(
            output.finish_reason,
            Some(GenerationFinishReason::Stop(Some(MatchedStop::Token(9))))
        );
        let extras = output.extras.unwrap();
        assert_eq!(extras.output_logprob_token_ids, [7]);
        assert_eq!(extras.output_top_logprob_lengths, [2]);
    }

    #[test]
    fn engine_error_frame_preserves_status_and_message() {
        let error = parse_engine_frame(
            r#"{"error":{"message":"too long","type":"BadRequestError","code":400}}"#,
        )
        .unwrap_err();
        assert_eq!(error.status_code, 400);
        assert_eq!(error.message, "too long");
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

    fn tiny_tokenizer() -> dynamo_tokenizers::Tokenizer {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../experimental/sgl-router/tests/fixtures/tiny_tokenizer.json");
        dynamo_tokenizers::Tokenizer::from_file_with_options(
            path.to_str().unwrap(),
            dynamo_tokenizers::TokenizerOptions {
                add_special_tokens: true,
            },
        )
        .unwrap()
    }

    #[tokio::test]
    async fn backend_posts_token_ids_and_decodes_the_engine_stream() {
        let tokenizer = tiny_tokenizer();
        let output_ids = DynamoTokenizer::new(tokenizer.clone())
            .encode("hello")
            .unwrap();
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

        let backend =
            HttpInferenceBackend::new(format!("http://{address}"), tokenizer.clone()).unwrap();
        let mut session = backend.begin_session();
        let request = TokenIdsRequest {
            rid: "client-request".into(),
            input_ids: vec![65],
            options: GenerationOptions {
                return_text_in_logprobs: Some(true),
                ..Default::default()
            },
        };
        let mut submission = session.submit(request, false).await.unwrap();
        let output = match submission.events.next().await.unwrap().unwrap() {
            GenerationEvent::Done(output) => output,
            GenerationEvent::Frame(_) => panic!("mock engine returned a terminal frame"),
        };

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
        assert_eq!(request["return_text_in_logprobs"], false);
        server.abort();
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
        let backend =
            HttpInferenceBackend::new(format!("http://{address}"), tiny_tokenizer()).unwrap();
        let mut session = backend.begin_session();
        let request = TokenIdsRequest {
            rid: "cancel-me".into(),
            input_ids: vec![65],
            options: GenerationOptions::default(),
        };
        let mut submission = session.submit(request, true).await.unwrap();
        assert!(submission.events.next().await.is_some());
        drop(submission.events);
        tokio::time::timeout(Duration::from_secs(2), notice_rx)
            .await
            .expect("engine response stream was not dropped")
            .unwrap();
        server.abort();
    }
}
