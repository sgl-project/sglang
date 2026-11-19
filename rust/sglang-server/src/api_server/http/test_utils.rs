//! Fixtures the HTTP adapters' unit tests share: a stub core handle, an app
//! state whose core answers every generation, a test tokenizer, and helpers
//! for sending a request through a router.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::Request;
use axum::response::Response;
use serde_json::json;
use tower::util::ServiceExt;

use crate::api_server::handlers::CoreHandle;
use crate::message::config::ServerArgs;
use crate::message::response::{ChunkEvent, ResponseItem};
use crate::tokenizer_manager::wiring::TmEvent;

/// The formatter a checkpoint with `config` and Jinja `template` selects.
pub(super) fn checkpoint_formatter(
    config: &serde_json::Value,
    template: &str,
) -> crate::api_server::openai::ChatFormatter {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("jinja-{}-{id}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(dir.join("chat_template.jinja"), template).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
    let (formatter, error) =
        sglang_processor::select_chat_formatter(&sglang_processor::ChatFormatterOptions {
            tokenizer_path: dir.to_string_lossy().into_owned(),
            ..Default::default()
        });
    std::fs::remove_dir_all(&dir).unwrap();
    formatter.unwrap_or_else(|| panic!("{error:?}"))
}

pub(crate) fn core_handle() -> CoreHandle {
    CoreHandle::new(
        flume::unbounded().0,
        flume::unbounded().0,
        crate::api_server::handlers::CoreConfig {
            response_capacity: 8,
            response_activity: Default::default(),
            startup_ready: false,
            is_disaggregation: false,
            mm_limits: Default::default(),
            multimodal: false,
            peer_endpoints: Vec::new(),
            metadata: crate::api_server::handlers::CoreMetadata::from(server_args().as_ref()),
            metrics: None,
            request_log: None,
            generate_validator: None,
            abort_hook: None,
        },
    )
}

pub(super) fn chunk(rid: &str, text: &str, done: bool) -> ResponseItem {
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
        ResponseItem::Done(output)
    } else {
        ResponseItem::Frame(output)
    }
}

pub(super) fn server_args() -> Arc<ServerArgs> {
    Arc::new(ServerArgs {
        served_model_name: "model".into(),
        ..Default::default()
    })
}

pub(super) fn app_state(core: CoreHandle) -> Arc<super::app::AppState> {
    let server_args = server_args();
    Arc::new(super::app::AppState {
        core,
        generation_policy: crate::message::request::GenerationPolicy::from(server_args.as_ref()),
        server_args,
        chat_formatter: None,
        generate_hook: None,
        openai: Default::default(),
    })
}

pub(super) fn frontend_closed() -> CoreHandle {
    // Dropping the receivers makes core admission report the shutdown
    // state as a 503.
    let (tm_tx, tm_rx) = flume::unbounded();
    drop(tm_rx);
    let (abort_tx, abort_rx) = flume::unbounded();
    drop(abort_rx);
    CoreHandle::new(
        tm_tx,
        abort_tx,
        crate::api_server::handlers::CoreConfig {
            response_capacity: 8,
            response_activity: Default::default(),
            startup_ready: false,
            is_disaggregation: false,
            mm_limits: Default::default(),
            multimodal: false,
            peer_endpoints: Vec::new(),
            metadata: crate::api_server::handlers::CoreMetadata::from(server_args().as_ref()),
            metrics: None,
            request_log: None,
            generate_validator: None,
            abort_hook: None,
        },
    )
}

/// A byte-level word tokenizer's `tokenizer.json`; id 3 is `é`'s bytes.
pub(super) fn tokenizer_path() -> &'static str {
    static PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("lowered-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let vocab = json!({"[UNK]": 0, "!": 1, "#": 2, "Ã©": 3});
        let model = json!({"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"});
        let config = json!({"model": model, "pre_tokenizer": {"type": "Whitespace"}});
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, config.to_string()).unwrap();
        path.to_string_lossy().into_owned()
    })
}

/// The mocked core ignores the prompt ids.
pub(in crate::api_server::http) fn tokenizer() -> sglang_processor::dynamo_tokenizers::Tokenizer {
    sglang_processor::dynamo_tokenizers::Tokenizer::from_file(tokenizer_path()).unwrap()
}

/// App state whose core intake answers every generation with one finished
/// `ok` chunk, with the ChatML template so chat can render.
pub(in crate::api_server::http) fn answering_app_state() -> Arc<super::app::AppState> {
    let formatter = sglang_processor::load_chat_formatter(None, None, None, Some("chatml"));
    answering_app_state_with(server_args(), formatter.unwrap(), Some(tokenizer()), "ok")
}

/// [`answering_app_state`], logging its requests to `request_log`.
pub(super) fn answering_app_state_logged(
    request_log: Arc<crate::api_server::log::RequestLog>,
) -> Arc<super::app::AppState> {
    let formatter = sglang_processor::load_chat_formatter(None, None, None, Some("chatml"));
    answering_app_state_built(
        server_args(),
        formatter.unwrap(),
        Some(tokenizer()),
        |rid| chunk(rid, "ok", true),
        Some(request_log),
    )
}

/// App state whose core intake answers every generation with one finished `text` chunk.
pub(super) fn answering_app_state_with(
    server_args: Arc<ServerArgs>,
    formatter: crate::api_server::openai::ChatFormatter,
    tokenizer: Option<sglang_processor::dynamo_tokenizers::Tokenizer>,
    text: &'static str,
) -> Arc<super::app::AppState> {
    answering_app_state_with_item(server_args, formatter, tokenizer, move |rid| {
        chunk(rid, text, true)
    })
}

/// [`answering_app_state_with`], answering each request with `answer(rid)`.
pub(super) fn answering_app_state_with_item(
    server_args: Arc<ServerArgs>,
    formatter: crate::api_server::openai::ChatFormatter,
    tokenizer: Option<sglang_processor::dynamo_tokenizers::Tokenizer>,
    answer: impl Fn(&str) -> ResponseItem + Send + 'static,
) -> Arc<super::app::AppState> {
    answering_app_state_built(server_args, formatter, tokenizer, answer, None)
}

fn answering_app_state_built(
    server_args: Arc<ServerArgs>,
    formatter: crate::api_server::openai::ChatFormatter,
    tokenizer: Option<sglang_processor::dynamo_tokenizers::Tokenizer>,
    answer: impl Fn(&str) -> ResponseItem + Send + 'static,
    request_log: Option<Arc<crate::api_server::log::RequestLog>>,
) -> Arc<super::app::AppState> {
    let (intake_tx, intake_rx) = flume::unbounded();
    tokio::spawn(async move {
        // Held open, so an unfinished answer stays unfinished.
        let mut open = Vec::new();
        while let Ok(TmEvent::Intake { request, admission }) = intake_rx.recv_async().await {
            assert!(admission.try_accept());
            request.sink.try_send(answer(request.rid.as_str())).unwrap();
            open.push(request);
        }
    });
    let core = CoreHandle::new(
        intake_tx,
        flume::unbounded().0,
        crate::api_server::handlers::CoreConfig {
            response_capacity: 8,
            response_activity: Default::default(),
            startup_ready: false,
            is_disaggregation: false,
            mm_limits: Default::default(),
            multimodal: false,
            peer_endpoints: Vec::new(),
            metadata: crate::api_server::handlers::CoreMetadata::from(server_args.as_ref()),
            metrics: None,
            request_log,
            generate_validator: None,
            abort_hook: None,
        },
    );
    // The test tokenizers add no specials, so one handle serves both modes.
    let tokenizer = tokenizer.map(|tokenizer| {
        Arc::new(sglang_processor::DynamoTokenizer::new(
            tokenizer.clone(),
            tokenizer,
        ))
    });
    Arc::new(super::app::AppState {
        core,
        generation_policy: crate::message::request::GenerationPolicy::from(server_args.as_ref()),
        openai: Arc::new(crate::api_server::openai::OpenAi::new(
            &server_args,
            Some(&formatter),
            tokenizer,
        )),
        server_args,
        chat_formatter: Some(formatter),
        generate_hook: None,
    })
}

/// Serve one request through the full router (extractors, auth, routing).
/// `with_state` consumes the state into a `Router<()>`, which is what
/// implements `tower::Service`.
pub(super) async fn oneshot(app: Router<()>, req: Request<Body>) -> Response {
    app.oneshot(req).await.unwrap()
}

pub(super) async fn post_json(app: Router<()>, path: &str, body: serde_json::Value) -> Response {
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
