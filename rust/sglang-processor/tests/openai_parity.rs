//! Every `tests/fixtures/openai_parity/*.json` case lowers to the `/generate`
//! body SGLang's OpenAI layer builds, and its recorded `/generate` output comes
//! back as the OpenAI response SGLang returns. Fixtures come from
//! `tests/scripts/generate_openai_parity.py`.

#![cfg(all(feature = "openai", feature = "tokenizer"))]

use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};
use sglang_processor::dynamo_tokenizers::{DecodeResult, Tokenizer};
use sglang_processor::openai::{
    OpenAiHeaders, OpenAiSettings, OpenAiTokenizer, Responder, TokenPieces, lower_completion,
};
use sglang_processor::{load_tokenizer, resolve_model_file};

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

    fn byte_level_piece(&self, id: u32) -> Option<String> {
        self.pieces.byte_level_piece(id).map(str::to_owned)
    }
}

impl FixtureTokenizer {
    fn load(path: &str) -> Self {
        let json = std::fs::read_to_string(path).unwrap();
        Self {
            tokenizer: load_tokenizer(Some(path), None, false).unwrap(),
            pieces: TokenPieces::from_tokenizer_json(&serde_json::from_str(&json).unwrap()),
        }
    }
}

/// Without the cached tokenizer, cases that need it are skipped, so the rest
/// still run in CI.
#[test]
fn fixtures_match_sglang() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/openai_parity");
    for entry in std::fs::read_dir(dir).unwrap() {
        let fixture: Value =
            serde_json::from_str(&std::fs::read_to_string(entry.unwrap().path()).unwrap()).unwrap();
        let model = fixture["model"].as_str().unwrap();
        let tokenizer = resolve_model_file(model, fixture["revision"].as_str(), "tokenizer.json")
            .map(|path| Arc::new(FixtureTokenizer::load(&path)) as Arc<dyn OpenAiTokenizer>);
        let settings: OpenAiSettings = serde_json::from_value(fixture["settings"].clone()).unwrap();
        let mut skipped = 0;
        for case in fixture["cases"].as_array().unwrap() {
            let defaults = &fixture["generate_defaults"];
            match check_case(case, defaults, &settings, tokenizer.clone()) {
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
    defaults: &Value,
    settings: &OpenAiSettings,
    tokenizer: Option<Arc<dyn OpenAiTokenizer>>,
) -> Result<bool, String> {
    let header = |name: &str| case["headers"][name].as_str();
    let headers = OpenAiHeaders {
        routing_key: header("x-smg-routing-key"),
        custom_labels: None,
    };
    let body = serde_json::to_vec(&case["request"]).unwrap();
    let missing_tokenizer = tokenizer.is_none();
    let lowered = lower_completion(&body, &headers, settings, tokenizer)
        .map(|(body, responder)| (body, Responder::Completion(responder)));
    let (generate, mut responder) = match (lowered, case["unsupported"].as_bool()) {
        (Err(reason), None) if missing_tokenizer && reason.0 == "no_tokenizer" => {
            return Ok(false);
        }
        (Err(_), Some(true)) => return Ok(true),
        (Ok(_), Some(true)) => return Err("SGLang's handler would not serve this".into()),
        (Err(reason), _) => return Err(format!("unsupported: {reason:?}")),
        (Ok(lowered), _) => lowered,
    };
    let generate = without_defaults(generate, defaults);
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

/// The fields Python set away from `GenerateReqInput`'s defaults.
fn without_defaults(body: Value, defaults: &Value) -> Value {
    let Value::Object(fields) = body else {
        unreachable!()
    };
    Value::Object(
        fields
            .into_iter()
            .filter(|(name, value)| defaults.get(name) != Some(value))
            .collect(),
    )
}

fn reply_json(status: u16, body: &str) -> Value {
    json!({"status": status, "body": serde_json::from_str::<Value>(body).unwrap()})
}

fn event_json(event: &str) -> Value {
    let data = event.strip_prefix("data: ").unwrap().trim_end();
    serde_json::from_str(data).unwrap_or_else(|_| json!(data))
}

/// `created` is the wall clock at response time.
fn without_created(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| match k.as_str() {
                    "created" => (k, json!(0)),
                    _ => (k, without_created(v)),
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(without_created).collect()),
        other => other,
    }
}
