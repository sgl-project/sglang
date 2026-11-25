// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod forward;
mod openai;
pub(crate) mod preparation;
pub(crate) use crate::routing::selection::pool_error;
use crate::routing::selection::ModelRouting;

pub(super) use preparation::enforce_minimax_m31_thinking;
pub use preparation::Preprocessor;

use crate::discovery::{ModelId, WorkerMode};
use crate::off_worker;
use crate::proxy::grpc::{GrpcResponse, RoutingView};
use crate::server::app_context::AppContext;
use crate::server::error::ApiError;
use crate::server::metrics::RequestLogContext;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Response};
use bytes::Bytes;
use forward::{forward_request, forward_request_grpc, RequestDurationGuard};
use preparation::{
    parse_embedding_request, parse_routing_fields, PreparedRequest, CHAT_PATH, CLASSIFY_PATH,
    COMPLETIONS_PATH, EMBEDDINGS_PATH, GENERATE_PATH, RERANK_PATH,
};
use sglang_api_types::api::v1::{token_ids_or_list, GenerateRequest, TokenIds, TokenIdsOrList};
use sglang_processor::openai::OpenAiSettings;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// Default cap on a buffered inference request body, including base64
/// multimodal inputs (32 MiB). [`SGLANG_ROUTER_MAX_BODY_BYTES`] overrides it;
/// [`max_chat_body_bytes`] is the value actually enforced.
pub const MAX_CHAT_BODY_BYTES: usize = 32 << 20;

/// Env var that overrides [`MAX_CHAT_BODY_BYTES`]: a positive decimal byte
/// count. A deployment that accepts larger multimodal requests (several base64
/// images or audio clips) raises it without rebuilding the router.
pub const SGLANG_ROUTER_MAX_BODY_BYTES: &str = "SGLANG_ROUTER_MAX_BODY_BYTES";

/// The request-body cap every inference route enforces, through the
/// `DefaultBodyLimit` layer in app.rs (413 past it). Read from
/// [`SGLANG_ROUTER_MAX_BODY_BYTES`] once, on first use (`build_router` at
/// startup), and logged then; an unset or empty variable gives
/// [`MAX_CHAT_BODY_BYTES`], and an unparsable or zero value gives the same
/// default with a WARN rather than failing startup. The value is bytes: `256`
/// is a 256-byte cap, not 256 MiB.
pub fn max_chat_body_bytes() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        let raw = std::env::var_os(SGLANG_ROUTER_MAX_BODY_BYTES);
        let (bytes, source) = resolve_max_body_bytes(raw.as_deref());
        match source {
            BodyCapSource::Default | BodyCapSource::Env => tracing::info!(
                max_body_bytes = bytes,
                source = if source == BodyCapSource::Env {
                    SGLANG_ROUTER_MAX_BODY_BYTES
                } else {
                    "default"
                },
                "request body cap"
            ),
            BodyCapSource::Invalid(reason) => tracing::warn!(
                value = ?raw,
                reason,
                max_body_bytes = bytes,
                "ignoring {SGLANG_ROUTER_MAX_BODY_BYTES}; using the default request body cap"
            ),
        }
        bytes
    })
}

/// Where the resolved request-body cap came from.
#[derive(Debug, PartialEq)]
enum BodyCapSource {
    Default,
    Env,
    /// The variable was set but unusable; the default applies.
    Invalid(String),
}

/// Resolve a raw [`SGLANG_ROUTER_MAX_BODY_BYTES`] value to the cap. Unset,
/// empty or whitespace-only gives the default. A positive decimal integer
/// (surrounding whitespace allowed, no unit suffix) is the cap. Anything else,
/// zero, overflow and non-UTF-8 included, gives the default and the reason.
fn resolve_max_body_bytes(raw: Option<&std::ffi::OsStr>) -> (usize, BodyCapSource) {
    let Some(raw) = raw else {
        return (MAX_CHAT_BODY_BYTES, BodyCapSource::Default);
    };
    let invalid = |reason: String| (MAX_CHAT_BODY_BYTES, BodyCapSource::Invalid(reason));
    let Some(raw) = raw.to_str() else {
        return invalid("not valid UTF-8".to_string());
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return (MAX_CHAT_BODY_BYTES, BodyCapSource::Default);
    }
    match raw.parse::<usize>() {
        Ok(0) => invalid("must be greater than zero".to_string()),
        Ok(bytes) => (bytes, BodyCapSource::Env),
        Err(e) => invalid(format!("not a byte count: {e}")),
    }
}

/// An inference route. HTTP and gRPC calls prepare and select for it alike.
#[derive(Clone, Copy)]
pub enum Endpoint {
    Chat,
    Completions,
    Generate,
    Embeddings,
    Classify,
    Rerank,
}

/// Validate, select workers, and forward a chat-completions request.
pub async fn chat_completions(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    serve_http(ctx, Endpoint::Chat, headers, body).await
}

/// OpenAI `/v1/completions`, forwarded to the engine's with the same request and response.
pub async fn completions(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    serve_http(ctx, Endpoint::Completions, headers, body).await
}

/// SGLang's native `/generate`: same request and response schema as the engine.
/// The body names no model, so it goes to the one this router serves.
pub async fn generate(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    serve_http(ctx, Endpoint::Generate, headers, body).await
}

/// OpenAI `/v1/embeddings`, forwarded to the engine's with the same request and response.
pub async fn embeddings(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    serve_http(ctx, Endpoint::Embeddings, headers, body).await
}

/// SGLang's `/v1/classify`, which takes the same `input` as embeddings.
pub async fn classify(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    serve_http(ctx, Endpoint::Classify, headers, body).await
}

/// SGLang's `/v1/rerank`, forwarded as sent to the model this router serves.
pub async fn rerank(
    State(ctx): State<Arc<AppContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    serve_http(ctx, Endpoint::Rerank, headers, body).await
}

async fn serve_http(
    ctx: Arc<AppContext>,
    endpoint: Endpoint,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response<Body>, ApiError> {
    let start = Instant::now();
    let (request, routing) = prepare(&ctx, endpoint, &headers, body).await?;
    routing.dispatch(&ctx, request, headers, start).await
}

/// Route one native Generate request without converting it through JSON.
pub async fn route_generate(
    ctx: &AppContext,
    mut request: GenerateRequest,
    headers: HeaderMap,
) -> (Result<GrpcResponse, ApiError>, Option<RequestLogContext>) {
    let start = Instant::now();
    let prepared = (|| {
        let model = ModelId(ctx.config.model.id.clone());
        let routing = ModelRouting::lookup(ctx, &model)?;
        let view = RoutingView::try_from(&request)?;
        let prepared = ctx.preprocessor.typed(model, &view)?;
        if view.text.is_some()
            && !ctx.config.model.disable_input_ids_forwarding
            && request.return_text_in_logprobs != Some(true)
            && request.image_data.is_none()
            && request.video_data.is_none()
            && request.audio_data.is_none()
        {
            if let Some(tokens) = &prepared.tokens {
                request.input_ids = Some(TokenIdsOrList {
                    value: Some(token_ids_or_list::Value::One(TokenIds {
                        ids: tokens.ids.iter().copied().map(i64::from).collect(),
                    })),
                });
                request.text = None;
            }
        }
        Ok::<_, ApiError>((prepared, routing))
    })();
    match prepared {
        Ok((mut prepared, routing)) => {
            forward_request_grpc(request, ctx, &mut prepared, &routing, headers, start).await
        }
        Err(error) => (Err(error), None),
    }
}

impl Endpoint {
    fn path(self) -> &'static str {
        match self {
            Endpoint::Chat => CHAT_PATH,
            Endpoint::Completions => COMPLETIONS_PATH,
            Endpoint::Generate => GENERATE_PATH,
            Endpoint::Embeddings => EMBEDDINGS_PATH,
            Endpoint::Classify => CLASSIFY_PATH,
            Endpoint::Rerank => RERANK_PATH,
        }
    }
}

/// Validate the body and resolve its routing policy, as `endpoint` does over HTTP.
/// Render and tokenize leave the async worker on their own. Parsing and
/// rebuilding a large body is CPU work too, so for a large body the whole step
/// runs through [`off_worker::run_for_body`].
async fn prepare<'a>(
    ctx: &'a AppContext,
    endpoint: Endpoint,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<(PreparedRequest, ModelRouting<'a>), ApiError> {
    off_worker::run_for_body(body.len(), || prepare_now(ctx, endpoint, headers, body))
}

fn prepare_now<'a>(
    ctx: &'a AppContext,
    endpoint: Endpoint,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<(PreparedRequest, ModelRouting<'a>), ApiError> {
    // One model is served, so any `model` name routes to it; the engine still
    // reads the name the body carries, with any `:adapter` suffix.
    let model = ModelId(ctx.config.model.id.clone());
    let routing = ModelRouting::lookup(ctx, &model)?;
    let request = match endpoint {
        Endpoint::Chat | Endpoint::Completions => {
            let mut fields = parse_routing_fields(&body)?;
            let sent = fields.model.take();
            let sent =
                ModelId(sent.ok_or_else(|| ApiError::BadRequest("missing `model` field".into()))?);
            let (_, body) = canonicalize_model_alias(ctx, sent, body)?;
            let needs_tokens = routing.needs_request_tokens(ctx);
            let path = if matches!(endpoint, Endpoint::Chat) {
                CHAT_PATH
            } else {
                COMPLETIONS_PATH
            };
            ctx.preprocessor.openai(
                model_openai_settings(ctx, &model).as_deref(),
                path,
                model,
                fields,
                headers,
                body,
                needs_tokens,
            )?
        }
        Endpoint::Generate => ctx.preprocessor.generate(model, body)?,
        Endpoint::Embeddings | Endpoint::Classify => {
            let path = endpoint.path();
            let (sent, mut value) = parse_embedding_request(&body)?;
            let (sent, body) = canonicalize_model_alias(ctx, sent, body)?;
            value["model"] = sent.0.into();
            require_plain_workers(ctx, &model, path)?;
            ctx.preprocessor.embeddings(path, model, body, value)?
        }
        Endpoint::Rerank => {
            require_plain_workers(ctx, &model, RERANK_PATH)?;
            PreparedRequest::rerank(model, body)?
        }
    };
    Ok((request, routing))
}

/// Resolve aliases before either lowering or forwarding. The engine only knows
/// its served name, so an unchanged URI alias would be parsed as a LoRA request.
fn canonicalize_model_alias(
    ctx: &AppContext,
    model: ModelId,
    body: Bytes,
) -> Result<(ModelId, Bytes), ApiError> {
    let Some(canonical) = ctx.config.model.canonical_alias(&model.0) else {
        return Ok((model, body));
    };
    // Keep every other JSON value verbatim, including numbers outside f64's
    // range, for the engine to validate rather than rounding or rejecting them.
    let mut fields: BTreeMap<String, &serde_json::value::RawValue> = serde_json::from_slice(&body)
        .map_err(|_| ApiError::BadRequest("invalid request: body must be a JSON object".into()))?;
    let encoded = serde_json::value::to_raw_value(&canonical)
        .map_err(|error| ApiError::Internal(error.into()))?;
    fields.insert("model".into(), &encoded);
    let body = serde_json::to_vec(&fields)
        .map_err(|error| ApiError::Internal(error.into()))?
        .into();
    Ok((ModelId(canonical), body))
}

/// Prefill and decode engines serve generation only.
fn require_plain_workers(ctx: &AppContext, model: &ModelId, path: &str) -> Result<(), ApiError> {
    let registered = ctx.registry.workers_for(model);
    if registered.iter().any(|w| w.mode() != WorkerMode::Plain) {
        return Err(ApiError::BadRequest(format!(
            "{path} is not served by prefill-decode workers"
        )));
    }
    Ok(())
}

/// The OpenAI settings every worker of `model` reports; `None` if any is unread or they differ.
fn model_openai_settings(ctx: &AppContext, model: &ModelId) -> Option<Arc<OpenAiSettings>> {
    let workers = ctx.registry.workers_for(model);
    let first = Arc::clone(workers.first()?.openai_settings()?);
    let agree = workers.iter().all(|w| w.openai_settings() == Some(&first));
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !agree && !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(model = %model.0, "workers report different OpenAI server args; \
            serving OpenAI requests through the engines' own routes");
    }
    agree.then_some(first)
}

impl ModelRouting<'_> {
    /// Select workers for `request` and forward it to them. An attempt that
    /// fails before any response reaches the client is retried, after a
    /// backoff, on workers it has not tried, up to `--retry-max-attempts`.
    async fn dispatch(
        &self,
        ctx: &AppContext,
        mut request: PreparedRequest,
        headers: HeaderMap,
        start: Instant,
    ) -> Result<Response<Body>, ApiError> {
        let mut excluded = Vec::new();
        let mut failed = None;
        let mut duration = None;
        for attempt in 0..ctx.config.proxy.max_attempts.get() {
            if attempt > 0 {
                // A retry never starts past the request's stale deadline, so the
                // backoff is capped by what remains of it. Backing off before
                // selecting lets the pick see fresh breaker and load state.
                let deadline = ctx.router_inflight_load.stale_request_timeout();
                let Some(remaining) = deadline.checked_sub(start.elapsed()) else {
                    break;
                };
                tokio::time::sleep(ctx.config.proxy.backoff(attempt).min(remaining)).await;
                if start.elapsed() >= deadline {
                    break;
                }
            }
            let workers = match self
                .select_workers(ctx, &request, &headers, &excluded)
                .await
            {
                Ok(workers) => workers,
                // Every eligible worker already failed this request: report the last failure.
                Err(error) => return failed.ok_or(error),
            };
            if failed.is_some() {
                ctx.metrics.record_retry(&request.model.0);
            }
            let duration = duration
                .get_or_insert_with(|| RequestDurationGuard::new(ctx, &request.model, start));
            let attempt =
                forward_request(ctx, &mut request, workers, headers.clone(), start, duration)
                    .await?;
            if attempt.retry_excluding.is_empty() {
                return Ok(attempt.response);
            }
            excluded.extend(attempt.retry_excluding);
            failed = Some(attempt.response);
        }
        Ok(failed.expect("at least one attempt"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::off_worker::test_support::{slow_chat_body, slow_template_dir, time_beside_a_timer};
    use crate::tokenizer::TokenizerRegistry;
    use std::time::Duration;
    #[test]
    fn max_body_bytes_resolves_from_the_env_value() {
        use std::ffi::OsStr;
        let resolve = |v: Option<&str>| resolve_max_body_bytes(v.map(OsStr::new));
        // Unset or empty: the default, no warning.
        for unset in [None, Some(""), Some("  ")] {
            assert_eq!(
                resolve(unset),
                (MAX_CHAT_BODY_BYTES, BodyCapSource::Default),
                "{unset:?}"
            );
        }
        // A positive byte count, larger or smaller than the default.
        assert_eq!(resolve(Some("268435456")), (256 << 20, BodyCapSource::Env));
        assert_eq!(resolve(Some(" 1024\n")), (1024, BodyCapSource::Env));
        // Zero and unparsable values keep the default and say why.
        for bad in [
            "0",
            "-1",
            "256MiB",
            "1.5",
            "0x100",
            "99999999999999999999999",
        ] {
            let (bytes, source) = resolve(Some(bad));
            assert_eq!(bytes, MAX_CHAT_BODY_BYTES, "{bad:?}");
            assert!(
                matches!(source, BodyCapSource::Invalid(_)),
                "{bad:?}: {source:?}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let (bytes, source) = resolve_max_body_bytes(Some(OsStr::from_bytes(b"\xff")));
            assert_eq!(bytes, MAX_CHAT_BODY_BYTES);
            assert!(matches!(source, BodyCapSource::Invalid(_)));
        }
    }

    #[test]
    fn aliases_match_whole_names_and_the_longest_lora_prefix() {
        let mut ctx = AppContext::stub();
        ctx.config.model.id = "s3://bucket/served".into();
        ctx.config.model.aliases = vec![
            "s3".into(),
            "short".into(),
            "s3://bucket/model".into(),
            "s3://bucket/model:version".into(),
        ];
        for (input, expected) in [
            ("s3://bucket/served", None),
            ("s3://bucket/served:adapter", None),
            ("short", Some("s3://bucket/served")),
            ("short:adapter", Some("s3://bucket/served:adapter")),
            ("s3://bucket/model", Some("s3://bucket/served")),
            ("s3://bucket/model:version", Some("s3://bucket/served")),
            (
                "s3://bucket/model:version:adapter",
                Some("s3://bucket/served:adapter"),
            ),
            ("unknown:adapter", None),
        ] {
            assert_eq!(
                ctx.config.model.canonical_alias(input).as_deref(),
                expected,
                "{input}"
            );
        }
    }

    #[test]
    fn alias_rewrite_preserves_other_json_values() {
        let mut ctx = AppContext::stub();
        ctx.config.model.aliases = vec!["s3://bucket/model".into()];
        let body = Bytes::from_static(
            br#"{"model":"s3://bucket/model","temperature":1e400,"extra":{"n":18446744073709551617,"text":"s3://bucket/model"}}"#,
        );
        let (model, body) =
            canonicalize_model_alias(&ctx, ModelId("s3://bucket/model".into()), body).unwrap();
        assert_eq!(model.0, "stub-model");
        let fields: BTreeMap<String, &serde_json::value::RawValue> =
            serde_json::from_slice(&body).unwrap();
        assert_eq!(fields["model"].get(), r#""stub-model""#);
        assert_eq!(fields["temperature"].get(), "1e400");
        assert_eq!(
            fields["extra"].get(),
            r#"{"n":18446744073709551617,"text":"s3://bucket/model"}"#
        );
        let original = Bytes::from_static(br#"{ "model": "stub-model", "temperature": 1e400 }"#);
        let (_, unchanged) =
            canonicalize_model_alias(&ctx, ModelId("stub-model".into()), original.clone()).unwrap();
        assert_eq!(unchanged, original);
    }

    /// A routable context over [`slow_template_dir`] that tokenizes chat
    /// requests in `prepare`. The directory must outlive the registry load.
    fn slow_render_ctx(dir: &std::path::Path) -> AppContext {
        let mut ctx = AppContext::stub();
        ctx.config.model.tokenizer_path = Some(dir.join("tokenizer.json").to_str().unwrap().into());
        // A token-aware route, so `prepare` renders and tokenizes the chat.
        ctx.config.model.dp_aware = true;
        ctx.preprocessor = Preprocessor::new(
            &ctx.config.model,
            Arc::new(TokenizerRegistry::load_from_config(&ctx.config).unwrap()),
            Arc::clone(&ctx.metrics),
        );
        ctx.policies.insert(
            ModelId("stub-model".into()),
            Arc::new(crate::policies::round_robin::RoundRobinPolicy::new()),
        );
        ctx
    }

    /// With ONE worker, a slow chat render in `prepare` must not keep another
    /// task's timer from firing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn a_slow_chat_render_leaves_the_runtime_serving() {
        let dir = slow_template_dir();
        let ctx = Arc::new(slow_render_ctx(dir.path()));
        let body = slow_chat_body(
            |b| {
                assert!(ctx
                    .preprocessor
                    .tokenizers()
                    .encode_chat("stub-model", b)
                    .is_some())
            },
            Duration::from_millis(400),
        );
        let body = Bytes::from(body.to_string());
        let (tx, rx) = tokio::sync::oneshot::channel();
        let (took, ticked_after) = time_beside_a_timer(move || async move {
            let tokens = prepare(&ctx, Endpoint::Chat, &HeaderMap::new(), body)
                .await
                .map(|(request, _)| request.tokens.is_some())
                .map_err(|e| format!("{e:?}"));
            tx.send(tokens).unwrap();
        })
        .await;
        assert_eq!(
            rx.await.unwrap(),
            Ok(true),
            "prepare must have rendered and tokenized the chat"
        );
        assert!(
            ticked_after < took / 2,
            "timer task waited {ticked_after:?} while prepare took {took:?}"
        );
    }
}
