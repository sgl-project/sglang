//! OpenAI-compatible generation endpoints.
//!
//! The HTTP adapter stays deliberately thin: Dynamo owns the standard OpenAI
//! request and response primitives. Native [`ChunkEvent`] values remain the one
//! backend output type for both unary and streaming responses.

use axum::{Router, http::StatusCode, response::Response};
use futures::StreamExt;
use tokio::sync::mpsc;

mod chat;
mod completions;
mod models;
mod reasoning;
mod template;
mod tools;

pub(super) use template::ChatFormatter;

use dynamo_protocols::types::{ChatCompletionStreamOptions, Stop};

use super::AppState;
use super::frame::OutputAccumulator;
use super::guard::AbortGuard;
use crate::ids::Rid;
use crate::message::{ChunkEvent, EgressItem, OneOrMany, SamplingParams};
use crate::runtime::ServerArgs;
use crate::utils::response::error_response;

const MAX_OPENAI_CHOICES: usize = 4096;

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .merge(models::routes())
        .merge(completions::routes())
        .merge(chat::routes())
}

/// Resolve the chat formatter, or `None` to disable the OpenAI chat-completions
/// endpoint. Tokenization is the tokenizer pool's job (the api server never
/// encodes); the formatter needs at most `tokenizer_config.json` — a built-in
/// `--chat-template` name or a model-path-inferred legacy template resolve
/// without it, so its absence must not disable chat.
pub(super) fn load_chat_support(server_args: &ServerArgs) -> Option<ChatFormatter> {
    // Chat needs the tokenizer pool behind it: under `skip_tokenizer_init`
    // there is none (text cannot be submitted), so chat is disabled.
    if server_args.skip_tokenizer_init || server_args.tokenizer_path.is_empty() {
        return None;
    }
    let config_file = crate::tokenizer::resolve_model_file(
        &server_args.tokenizer_path,
        server_args.revision.as_deref(),
        "tokenizer_config.json",
    );

    match template::load_chat_formatter(
        config_file.as_deref(),
        (!server_args.model_path.is_empty()).then_some(server_args.model_path.as_str()),
        server_args.chat_template.as_deref(),
    ) {
        Ok(formatter) => {
            tracing::info!(
                config = ?config_file.as_deref().unwrap_or("<built-in / inferred>"),
                "loaded OpenAI chat template"
            );
            Some(formatter)
        }
        Err(error) => {
            tracing::warn!(%error, "OpenAI chat completions disabled");
            None
        }
    }
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn unix_seconds_u32() -> u32 {
    u32::try_from(unix_seconds()).unwrap_or(u32::MAX)
}

/// One submitted OpenAI choice, shared by the chat and completions
/// endpoints: OpenAI `n` (and the completions prompt list) fan out into one
/// native request per choice, and this carries what the drain/stream stages
/// need to route its output back to the right response slot.
pub(super) struct OpenAiChoice {
    pub(super) index: usize,
    /// Which prompt this choice belongs to (usage counts each prompt once).
    /// Chat has a single prompt: always 0.
    pub(super) prompt_index: usize,
    pub(super) rid: Rid,
    /// Echoed prefix for the first output chunk (completions `echo`; empty
    /// for chat).
    pub(super) echo: String,
    pub(super) rx: mpsc::Receiver<EgressItem>,
}

/// OpenAI `stop` → the native pair: strings become stop strings, token ids
/// become `stop_token_ids`. Identical for chat and completions.
pub(super) fn native_stops(stop: Option<&Stop>) -> (Option<OneOrMany<String>>, Option<Vec<i64>>) {
    match stop {
        Some(Stop::String(value)) => (Some(OneOrMany::One(value.clone())), None),
        Some(Stop::StringArray(values)) => (Some(OneOrMany::Many(values.clone())), None),
        Some(Stop::TokenIdArray(values)) => {
            (None, Some(values.iter().map(|&id| id as i64).collect()))
        }
        None => (None, None),
    }
}

/// OpenAI `logit_bias` → the native map; a non-numeric bias is a 400.
pub(super) fn parse_logit_bias(
    values: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> Result<Option<std::collections::BTreeMap<String, f64>>, String> {
    let mut logit_bias = std::collections::BTreeMap::new();
    if let Some(values) = values {
        for (token, bias) in values {
            let bias = bias
                .as_f64()
                .ok_or_else(|| format!("logit_bias[{token:?}] must be a number"))?;
            logit_bias.insert(token.clone(), bias);
        }
    }
    Ok((!logit_bias.is_empty()).then_some(logit_bias))
}

/// Early sampling validation. The ingress FSM re-normalizes authoritatively
/// (its `Normalizing` state runs the same call, and `normalize` is idempotent
/// — pinned by `normalize_is_idempotent`), so this pass exists only to
/// (a) answer invalid params on a `stream: true` request as an unary 400
/// BEFORE the SSE commits, matching Python, and (b) fail before the
/// n×prompts fan-out submits doomed requests.
pub(super) fn normalize_sampling(
    sampling: &mut SamplingParams,
    server_args: &ServerArgs,
) -> Result<(), String> {
    sampling
        .normalize(
            server_args.skip_tokenizer_init,
            server_args.model_config.vocab_size.unwrap_or(u64::MAX),
        )
        .map_err(|error| error.to_string())
}

/// `Some(400)` unless the request names the served model (the backtick
/// message format is wire-pinned). Option rather than Result: clippy's
/// `result_large_err` — a `Response` error variant is 128+ bytes.
pub(super) fn require_served_model(model: &str, server_args: &ServerArgs) -> Option<Response> {
    (model != server_args.served_model_name)
        .then(|| invalid_request(format!("The model `{model}` does not exist")))
}

/// Whether an SSE stream ends with a usage chunk: the request's
/// `stream_options` or the server-wide default.
pub(super) fn stream_include_usage(
    options: Option<&ChatCompletionStreamOptions>,
    server_args: &ServerArgs,
) -> bool {
    options.is_some_and(|options| options.include_usage)
        || server_args.stream_response_default_include_usage
}

/// The dominant OpenAI error case: an unary 400. Deliberately narrow — it
/// names one hot pattern rather than re-wrapping `error_response`'s
/// unary/stream selector; non-400 and committed-stream sites keep the
/// explicit `error_response` + `error_payload` composition.
pub(super) fn invalid_request(message: impl Into<String>) -> Response {
    error_response(
        StatusCode::BAD_REQUEST,
        error_payload(StatusCode::BAD_REQUEST, message),
        false,
    )
}

/// The OpenAI error payload.
pub(super) fn error_payload(code: StatusCode, message: impl Into<String>) -> serde_json::Value {
    let message = message.into();
    let error_type = if code == StatusCode::UNAUTHORIZED {
        "AuthenticationError"
    } else if code.is_server_error() {
        "InternalServerError"
    } else {
        "BadRequestError"
    };
    serde_json::json!({
        "error": {
            "object": "error",
            "message": message,
            "type": error_type,
            "param": null,
            "code": code.as_u16(),
        }
    })
}

/// One event from [`choice_step_stream`].
pub(super) enum ChoiceStep {
    /// A generation step for choice `index`; `first` marks the choice's
    /// first output (the completions echo prefix hangs off it).
    Output {
        index: usize,
        output: ChunkEvent,
        first: bool,
    },
    /// A ready-to-yield OpenAI error frame payload — channel truncation, an
    /// `EgressItem::Error`, or a validation abort; the exact
    /// `error_payload(..).to_string()` both endpoints previously built by
    /// hand.
    Error(String),
    /// Emitted once, after every receiver ended: totals for the trailing
    /// usage chunk. Prompt tokens first-win per `prompt_index` (chat's
    /// single prompt is the one-key case), completion tokens sum.
    Usage {
        prompt_tokens: u32,
        completion_tokens: u64,
    },
}

/// The SSE skeleton shared by chat and completions: multiplex the choice
/// receivers, classify every egress item, disarm the guard on terminals, and
/// account usage. Endpoint streams only render `Output` steps into their own
/// wire frames (and decide whether the `Usage` step becomes a chunk).
pub(super) fn choice_step_stream(
    choices: Vec<OpenAiChoice>,
    mut guard: AbortGuard,
) -> impl futures::Stream<Item = ChoiceStep> {
    async_stream::stream! {
        let count = choices.len();
        // Slot tables keyed by `choice.index`, which both endpoints mint
        // densely over 0..count (chat: 0..n; completions: prompt*n + sample).
        let mut rids: Vec<Option<Rid>> = (0..count).map(|_| None).collect();
        let mut prompt_index_of = vec![0usize; count];
        let mut streams = Vec::with_capacity(count);
        for choice in choices {
            prompt_index_of[choice.index] = choice.prompt_index;
            rids[choice.index] = Some(choice.rid);
            streams.push(indexed_egress_stream(choice.index, choice.rx));
        }
        let rids: Vec<Rid> = rids
            .into_iter()
            .map(|rid| rid.expect("choice indexes are dense"))
            .collect();
        let mut first_output = vec![true; count];
        let mut prompt_tokens_by_prompt = std::collections::BTreeMap::<usize, u32>::new();
        let mut completion_tokens: u64 = 0;

        let mut events = futures::stream::select_all(streams);
        while let Some((index, item)) = events.next().await {
            let Some(item) = item else {
                // Channel closed without a terminal → truncation; the rid
                // stays armed so the guard aborts the scheduler work.
                yield ChoiceStep::Error(
                    error_payload(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "response truncated before completion",
                    )
                    .to_string(),
                );
                continue;
            };
            let output = match item {
                EgressItem::Frame(output) => output,
                EgressItem::Done(output) => {
                    guard.disarm(&rids[index]);
                    output
                }
                EgressItem::Error(error) => {
                    guard.disarm(&rids[index]);
                    yield ChoiceStep::Error(
                        error_payload(
                            StatusCode::from_u16(error.http_status())
                                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                            error.to_string(),
                        )
                        .to_string(),
                    );
                    continue;
                }
                EgressItem::Control(_) | EgressItem::Data(_) => continue,
            };
            if let Some((code, message)) = output
                .finish_reason
                .as_ref()
                .and_then(|reason| reason.abort_status())
            {
                yield ChoiceStep::Error(
                    error_payload(
                        StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                        message,
                    )
                    .to_string(),
                );
                continue;
            }
            prompt_tokens_by_prompt
                .entry(prompt_index_of[index])
                .or_insert(output.prompt_tokens);
            completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
            let first = std::mem::replace(&mut first_output[index], false);
            yield ChoiceStep::Output { index, output, first };
        }
        yield ChoiceStep::Usage {
            prompt_tokens: prompt_tokens_by_prompt
                .values()
                .copied()
                .fold(0u32, u32::saturating_add),
            completion_tokens,
        };
    }
}

/// Drain every submitted choice in order (they already run concurrently) to
/// its terminal output, mapping the first failure to the OpenAI error
/// response. Usage totals ride along: prompt tokens count once per prompt
/// (first terminal wins per `prompt_index`), completion tokens sum over
/// choices. Returns `(index, echo, output)` triples in submit order.
pub(super) async fn collect_choices(
    choices: Vec<OpenAiChoice>,
    guard: &mut AbortGuard,
) -> Result<(Vec<(usize, String, ChunkEvent)>, u32, u64), Response> {
    let mut outputs = Vec::with_capacity(choices.len());
    let mut prompt_tokens = std::collections::BTreeMap::<usize, u32>::new();
    let mut completion_tokens: u64 = 0;
    for choice in choices {
        let output = match collect_output(choice.rx, guard, &choice.rid).await {
            Ok(output) => output,
            Err((status, message)) => {
                return Err(error_response(
                    status,
                    error_payload(status, message),
                    false,
                ));
            }
        };
        prompt_tokens
            .entry(choice.prompt_index)
            .or_insert(output.prompt_tokens);
        completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
        outputs.push((choice.index, choice.echo, output));
    }
    let prompt_tokens = prompt_tokens
        .values()
        .copied()
        .fold(0u32, u32::saturating_add);
    Ok((outputs, prompt_tokens, completion_tokens))
}

/// Drain one submitted request to its terminal output: fold frames, disarm
/// `guard` on a natural terminal, and map errors / validation aborts /
/// truncation to `(status, message)` for the OpenAI error shape.
async fn collect_output(
    mut rx: mpsc::Receiver<EgressItem>,
    guard: &mut AbortGuard,
    rid: &Rid,
) -> Result<ChunkEvent, (StatusCode, String)> {
    let mut accumulator = OutputAccumulator::default();
    let output = loop {
        match rx.recv().await {
            Some(EgressItem::Frame(output)) => accumulator.fold(&output),
            Some(EgressItem::Done(output)) => {
                accumulator.fold(&output);
                break accumulator.into_output();
            }
            Some(EgressItem::Error(error)) => {
                guard.disarm(rid);
                let status = StatusCode::from_u16(error.http_status())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                return Err((status, error.to_string()));
            }
            Some(EgressItem::Control(_)) | Some(EgressItem::Data(_)) => {}
            None => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "response truncated before completion".into(),
                ));
            }
        }
    };
    guard.disarm(rid);
    if let Some((code, message)) = output
        .finish_reason
        .as_ref()
        .and_then(|reason| reason.abort_status())
    {
        return Err((
            StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            message.to_owned(),
        ));
    }
    Ok(output)
}

fn indexed_egress_stream(
    index: usize,
    rx: mpsc::Receiver<EgressItem>,
) -> futures::stream::BoxStream<'static, (usize, Option<EgressItem>)> {
    futures::stream::unfold((rx, false), move |(mut rx, finished)| async move {
        if finished {
            return None;
        }
        match rx.recv().await {
            Some(item) => {
                let finished = matches!(item, EgressItem::Done(_) | EgressItem::Error(_));
                Some(((index, Some(item)), (rx, finished)))
            }
            None => Some(((index, None), (rx, true))),
        }
    })
    .boxed()
}

fn contains_media(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Array(values) => values.iter().any(contains_media),
        serde_json::Value::Object(object) => {
            object.keys().any(|key| {
                matches!(
                    key.as_str(),
                    "image_url" | "video_url" | "input_audio" | "audio_url" | "file"
                )
            }) || object.values().any(contains_media)
        }
        _ => false,
    }
}

#[cfg(test)]
mod test_utils {
    use std::sync::Arc;

    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::response::Response;
    use serde_json::json;

    use crate::utils::response::error_response;
    use tower::util::ServiceExt;

    use super::routes;
    use crate::message::{ChunkEvent, EgressItem};
    use crate::runtime::ServerArgs;
    use crate::tokenizer_manager::Senders;

    pub(super) fn senders() -> Senders {
        Senders {
            tm: flume::unbounded().0,
            abort: flume::unbounded().0,
            tok: flume::unbounded().0,
            detok: vec![],
        }
    }

    pub(super) fn chunk(rid: &str, text: &str, done: bool) -> EgressItem {
        let output = ChunkEvent {
            rid: rid.into(),
            text: text.into(),
            token_ids: vec![1],
            prompt_tokens: 5,
            completion_tokens: 1,
            finish_reason: done.then(|| {
                serde_json::from_value(serde_json::json!({
                    "type": "stop",
                    "matched": "</s>"
                }))
                .unwrap()
            }),
            ..Default::default()
        };
        if done {
            EgressItem::Done(output)
        } else {
            EgressItem::Frame(output)
        }
    }

    /// A submitted legacy completion choice with its egress channel.
    pub(super) fn submitted(
        index: usize,
        prompt_index: usize,
        rid: &str,
    ) -> (super::OpenAiChoice, tokio::sync::mpsc::Sender<EgressItem>) {
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        (
            super::OpenAiChoice {
                index,
                prompt_index,
                rid: rid.into(),
                echo: String::new(),
                rx,
            },
            tx,
        )
    }

    /// A submitted chat choice (the tuple `chat_event_stream` consumes) with its
    /// egress channel.
    pub(super) fn chat_submitted(
        index: usize,
        rid: &str,
    ) -> (super::OpenAiChoice, tokio::sync::mpsc::Sender<EgressItem>) {
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        (
            super::OpenAiChoice {
                index,
                prompt_index: 0,
                rid: rid.into(),
                echo: String::new(),
                rx,
            },
            tx,
        )
    }

    // ---------------------------------------------------------------------
    // Handler-level tests: full router, real extractors, no scheduler. A
    // request that reaches `submit` with an OPEN tm lane would wait on the
    // egress receiver forever, so submission-reaching cases use `senders_closed`
    // (503) and everything else fails validation before submit.
    // ---------------------------------------------------------------------

    pub(super) fn server_args() -> Arc<ServerArgs> {
        Arc::new(
            serde_json::from_value(serde_json::json!({ "served_model_name": "model" }))
                .expect("ServerArgs must deserialize"),
        )
    }

    pub(super) fn app_state(senders: Senders) -> super::AppState {
        super::AppState {
            senders,
            egress_buf: 8,
            server_args: server_args(),
            chat_formatter: None,
            egress_activity: Default::default(),
        }
    }

    pub(super) fn senders_closed() -> Senders {
        // Dropping the receivers disconnects the channels; the senders stay
        // valid (moveable) but every send reports `Err`, the shutdown state
        // `submit` surfaces as a 503.
        let (tm_tx, tm_rx) = flume::unbounded();
        drop(tm_rx);
        let (abort_tx, abort_rx) = flume::unbounded();
        drop(abort_rx);
        let (tok_tx, tok_rx) = flume::unbounded();
        drop(tok_rx);
        Senders {
            tm: tm_tx,
            abort: abort_tx,
            tok: tok_tx,
            detok: vec![],
        }
    }

    /// Serve one request through the full router (extractors, auth, routing).
    /// `with_state` consumes the state into a `Router<()>`, which is what
    /// implements `tower::Service`.
    pub(super) async fn oneshot(app: Router<()>, req: Request<Body>) -> Response {
        app.oneshot(req).await.unwrap()
    }

    pub(super) async fn post_json(
        app: Router<()>,
        path: &str,
        body: serde_json::Value,
    ) -> Response {
        let req = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        oneshot(app, req).await
    }

    pub(super) async fn body_json(response: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// The common StatusCode→error helper follows `error_response`'s shape:
    /// unary requests get the JSON error with its status; a committed stream gets
    /// 200 + one SSE error frame + `[DONE]`, and the frame carries the OpenAI
    /// error fields (`type`, `param`, `code`) that the SDKs dispatch on.
    #[tokio::test]
    async fn openai_error_response_covers_unary_and_sse() {
        let unary = error_response(
            StatusCode::BAD_REQUEST,
            super::error_payload(StatusCode::BAD_REQUEST, "bad input"),
            false,
        );
        assert_eq!(unary.status(), StatusCode::BAD_REQUEST);
        let value = body_json(unary).await;
        assert_eq!(value["error"]["message"], "bad input");
        assert_eq!(value["error"]["type"], "BadRequestError");
        assert_eq!(value["error"]["code"], 400);
        assert!(value["error"]["param"].is_null());

        let streamed = error_response(
            StatusCode::BAD_REQUEST,
            super::error_payload(StatusCode::BAD_REQUEST, "bad input"),
            true,
        );
        assert_eq!(streamed.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(streamed.into_body(), 64 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let frame = text
            .split("\n\n")
            .next()
            .unwrap()
            .strip_prefix("data: ")
            .unwrap();
        let frame: serde_json::Value = serde_json::from_str(frame).unwrap();
        assert_eq!(frame["error"]["message"], "bad input");
        assert_eq!(frame["error"]["type"], "BadRequestError");
        assert!(text.contains("[DONE]"));
    }

    #[tokio::test]
    async fn completions_handler_validates_before_submit() {
        let app = routes().with_state(app_state(senders()));
        let cases = [
            (json!({"model": "other", "prompt": "hi"}), "unknown model"),
            (json!({"model": "model", "prompt": "hi", "n": 0}), "n=0"),
            (
                json!({"model": "model", "prompt": "hi", "max_tokens": 0}),
                "max_tokens=0",
            ),
            (json!({"model": "model", "prompt": ""}), "empty prompt"),
            (
                json!({"model": "model", "prompt": "hi", "best_of": 2}),
                "best_of>1",
            ),
            (
                json!({"model": "model", "prompt": "hi", "suffix": "x"}),
                "suffix",
            ),
            (
                json!({"model": "model", "prompt": "hi", "prompt_embeds": [[1.0]]}),
                "prompt_embeds",
            ),
        ];
        for (body, label) in cases {
            let response = post_json(app.clone(), "/v1/completions", body).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
        }
        // Malformed JSON → 400 (JsonRejection path).
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from("not json"))
            .unwrap();
        let response = oneshot(app.clone(), req).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        // A closed tm inbox (shutdown) surfaces as 503.
        let app = routes().with_state(app_state(senders_closed()));
        let response = post_json(
            app.clone(),
            "/v1/completions",
            json!({"model": "model", "prompt": "hi"}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn chat_handler_validates_before_submit() {
        let app = routes().with_state(app_state(senders()));
        let cases = [
            (
                json!({"model": "other", "messages": [{"role": "user", "content": "hi"}]}),
                "unknown model",
            ),
            (json!({"model": "model", "messages": []}), "empty messages"),
            (
                json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "n": 0}),
                "n=0",
            ),
            (
                json!({"model": "model", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://example.com/x.png"}}]}]}),
                "media content",
            ),
            (
                json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "function_call": "auto"}),
                "deprecated function_call",
            ),
            (
                json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "audio": {"input_audio": {"data": "x", "format": "wav"}}}),
                "audio",
            ),
            (
                json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 0}),
                "max_completion_tokens=0",
            ),
        ];
        for (body, label) in cases {
            let response = post_json(app.clone(), "/v1/chat/completions", body).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{label}");
        }
        // A valid request with no loaded chat template → 400 (template gate).
        let response = post_json(
            app.clone(),
            "/v1/chat/completions",
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn basic_openai_router_excludes_responses_api() {
        let app = routes().with_state(app_state(senders()));
        let response = post_json(app, "/v1/responses", json!({"input": "hi"})).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// A closed tm inbox with a *streaming* request must answer inside the
    /// committed stream: 200 + one OpenAI-shaped SSE error frame + `[DONE]` (the
    /// same `error_response` rule the native API applies), not a unary 503.
    #[tokio::test]
    async fn streaming_submit_failure_answers_inside_the_stream() {
        let app = routes().with_state(app_state(senders_closed()));
        let response = post_json(
            app,
            "/v1/completions",
            json!({"model": "model", "prompt": "hi", "stream": true}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let frame = text
            .split("\n\n")
            .next()
            .unwrap()
            .strip_prefix("data: ")
            .unwrap();
        let frame: serde_json::Value = serde_json::from_str(frame).unwrap();
        assert_eq!(frame["error"]["message"], "service unavailable");
        assert_eq!(frame["error"]["type"], "InternalServerError");
        assert_eq!(frame["error"]["code"], 503);
        assert!(text.contains("[DONE]"));
    }
}
