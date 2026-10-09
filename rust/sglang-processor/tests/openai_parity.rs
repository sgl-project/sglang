//! Every `tests/fixtures/openai_parity/*.json` case lowers to the `/generate`
//! body SGLang's OpenAI layer builds, and its recorded `/generate` output comes
//! back as the OpenAI response SGLang returns. Fixtures come from
//! `tests/scripts/generate_openai_parity.py`.

#![cfg(all(feature = "openai", feature = "render", feature = "tokenizer"))]

use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};
use sglang_processor::dynamo_tokenizers::{DecodeResult, Tokenizer};
use sglang_processor::openai::{
    ChatModel, OpenAiHeaders, OpenAiSettings, OpenAiTokenizer, Responder, TokenPieces, lower_chat,
    lower_completion,
};
use sglang_processor::{
    ChatFormatter, ChatFormatterOptions, RenderEnv, load_tokenizer, resolve_model_file,
    select_chat_formatter,
};

struct FixtureTokenizer {
    tokenizer: Tokenizer,
    pieces: TokenPieces,
}

impl OpenAiTokenizer for FixtureTokenizer {
    fn decode(&self, ids: &[u32]) -> Option<String> {
        match self.tokenizer.decode(ids, true).ok()? {
            DecodeResult::Complete(text) | DecodeResult::Partial(text) => Some(text),
        }
    }

    fn byte_level_bytes(&self, id: u32) -> Option<Vec<u8>> {
        self.pieces.byte_level_bytes(id)
    }
}

/// The cached checkpoint's renderer and tokenizer, as a host has them.
struct Checkpoint {
    formatter: ChatFormatter,
    tokenizer: Tokenizer,
    openai: Arc<dyn OpenAiTokenizer>,
}

impl Checkpoint {
    fn load(tokenizer_file: &str) -> Self {
        let dir = Path::new(tokenizer_file).parent().unwrap();
        let (formatter, error) = select_chat_formatter(&ChatFormatterOptions {
            tokenizer_path: dir.to_string_lossy().into_owned(),
            ..Default::default()
        });
        let json = std::fs::read_to_string(tokenizer_file).unwrap();
        Self {
            formatter: formatter.unwrap_or_else(|| panic!("{error:?}")),
            tokenizer: load_tokenizer(Some(tokenizer_file), None, true).unwrap(),
            openai: Arc::new(FixtureTokenizer {
                tokenizer: load_tokenizer(Some(tokenizer_file), None, false).unwrap(),
                pieces: TokenPieces::from_tokenizer_json(&serde_json::from_str(&json).unwrap()),
            }),
        }
    }

    /// The prompt ids SGLang encodes: the rendered prompt, then the
    /// `continue_final_message` prefix without its BOS.
    fn render(&self, request: &Value, settings: &OpenAiSettings) -> Option<Vec<u32>> {
        let defaults = settings
            .default_chat_template_kwargs
            .clone()
            .unwrap_or_default();
        let (prompt, prefix) = self
            .formatter
            .render_request(
                request.clone(),
                &defaults.into_iter().collect(),
                &RenderEnv::default(),
            )
            .ok()?;
        let ids = |text: &str| self.tokenizer.encode(text).unwrap().token_ids().to_vec();
        let bos = ids("").first().copied();
        let mut out = ids(&prompt);
        let mut suffix = ids(&prefix);
        if bos.is_some() && suffix.first().copied() == bos {
            suffix.remove(0);
        }
        out.extend(suffix);
        Some(out)
    }
}

/// Without the cached checkpoint, chat renders to Python's recorded prompt ids
/// and cases that need the tokenizer are skipped, so the rest still run in CI.
#[test]
fn fixtures_match_sglang() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/openai_parity");
    for entry in std::fs::read_dir(dir).unwrap() {
        let fixture: Value =
            serde_json::from_str(&std::fs::read_to_string(entry.unwrap().path()).unwrap()).unwrap();
        let model = fixture["model"].as_str().unwrap();
        let checkpoint = resolve_model_file(model, fixture["revision"].as_str(), "tokenizer.json")
            .map(|path| Checkpoint::load(&path));
        let settings: OpenAiSettings = serde_json::from_value(fixture["settings"].clone()).unwrap();
        let chat_model: ChatModel = serde_json::from_value(fixture["chat_model"].clone()).unwrap();
        let mut skipped = 0;
        for case in fixture["cases"].as_array().unwrap() {
            match check_case(case, &fixture, &settings, &chat_model, checkpoint.as_ref()) {
                Ok(true) => {}
                Ok(false) => skipped += 1,
                Err(error) => panic!("{model} {}: {error}", case["name"]),
            }
        }
        if skipped > 0 {
            eprintln!("{model}: tokenizer not cached; {skipped} cases skipped");
        }
    }
}

fn check_case(
    case: &Value,
    fixture: &Value,
    settings: &OpenAiSettings,
    chat_model: &ChatModel,
    checkpoint: Option<&Checkpoint>,
) -> Result<bool, String> {
    let header = |name: &str| case["headers"][name].as_str();
    let headers = OpenAiHeaders {
        routing_key: header("x-smg-routing-key"),
        custom_labels: None,
        sglext_ids: false,
    };
    let body = serde_json::to_vec(&case["request"]).unwrap();
    let tokenizer = checkpoint.map(|c| c.openai.clone());
    let lowered = match case["endpoint"].as_str() {
        Some("chat") => {
            let recorded = || serde_json::from_value(case["generate"]["input_ids"].clone()).ok();
            let render = |request: &Value| match checkpoint {
                Some(checkpoint) => checkpoint.render(request, settings),
                None => recorded(),
            };
            lower_chat(&body, &headers, settings, chat_model, render, tokenizer)
                .map(|(body, responder)| (body, Responder::Chat(Box::new(responder))))
        }
        _ => lower_completion(&body, &headers, settings, tokenizer)
            .map(|(body, responder)| (body, Responder::Completion(Box::new(responder)))),
    };
    let (generate, mut responder) = match (lowered, case["unsupported"].as_bool()) {
        (Err(reason), None) if checkpoint.is_none() && reason.0 == "no_tokenizer" => {
            return Ok(false);
        }
        (Err(_), Some(true)) => return Ok(true),
        (Ok(_), Some(true)) => return Err("SGLang's handler would not serve this".into()),
        (Err(reason), _) => return Err(format!("unsupported: {reason:?}")),
        (Ok(lowered), _) => lowered,
    };
    let generate = without_defaults(generate, fixture, &case["endpoint"]);
    if generate != case["generate"] {
        return Err(format!("/generate body differs:\n{generate:#}"));
    }
    if case["lower_only"].as_bool() == Some(true) {
        return Ok(true);
    }

    let engine = &case["engine"];
    let status = engine["status"].as_u64().unwrap() as u16;
    let (got, want) = match engine["frames"].as_array() {
        Some(frames) => {
            let mut events = Vec::new();
            let mut reply = None;
            for frame in frames
                .iter()
                .map(|f| f.to_string())
                .chain(["[DONE]".into()])
            {
                match responder.stream_data(frame.as_bytes()) {
                    Ok(more) => events.extend(more),
                    Err(error) => {
                        reply = Some(error);
                        break;
                    }
                }
            }
            match reply {
                Some(reply) => (
                    reply_json(reply.status, &reply.body),
                    case["openai"].clone(),
                ),
                None => (
                    json!({"events": events.iter().map(|e| event_json(e)).collect::<Vec<_>>()}),
                    json!({"events": case["openai"]["events"].as_array().unwrap().iter().map(|e| event_json(e.as_str().unwrap())).collect::<Vec<_>>()}),
                ),
            }
        }
        None => {
            let body = engine["body"].to_string();
            let reply = match case["request"]["stream"] == true || status != 200 {
                true => Responder::rejected(body.as_bytes()),
                false => responder.unary(body.as_bytes()),
            };
            let reply = reply.ok_or("no reply")?;
            (
                reply_json(reply.status, &reply.body),
                case["openai"].clone(),
            )
        }
    };
    let want = without_created(want);
    let got = without_created(got);
    if got != want {
        return Err(format!(
            "OpenAI response differs:\n got: {got}\nwant: {want}"
        ));
    }
    Ok(true)
}

/// The fields Python set away from `GenerateReqInput`'s and a bare request's sampling defaults.
fn without_defaults(body: Value, fixture: &Value, endpoint: &Value) -> Value {
    let Value::Object(mut fields) = body else {
        unreachable!()
    };
    fields.retain(|name, value| fixture["generate_defaults"].get(name) != Some(value));
    let sampling_defaults = &fixture["sampling_defaults"][endpoint.as_str().unwrap()];
    if let Some(Value::Object(params)) = fields.get_mut("sampling_params") {
        params.retain(|name, value| sampling_defaults.get(name) != Some(value));
    }
    Value::Object(fields)
}

fn reply_json(status: u16, body: &str) -> Value {
    json!({"status": status, "body": serde_json::from_str::<Value>(body).unwrap()})
}

fn event_json(event: &str) -> Value {
    let data = event.strip_prefix("data: ").unwrap().trim_end();
    serde_json::from_str(data).unwrap_or_else(|_| json!(data))
}

/// `created` is the wall clock at response time, and tool call ids are random.
fn without_created(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| match k.as_str() {
                    "created" => (k, json!(0)),
                    "id" if v.as_str().is_some_and(|id| id.starts_with("call_")) => {
                        (k, json!("call_"))
                    }
                    _ => (k, without_created(v)),
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(without_created).collect()),
        other => other,
    }
}
