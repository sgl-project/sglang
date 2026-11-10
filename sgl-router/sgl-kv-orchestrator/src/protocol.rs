//! The checked-in schemas are generated from Python contracts and drift-tested.
use crate::models::PrefixTarget;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    sync::LazyLock,
};

pub const PROTOCOL_VERSION: &str = "0.1";
pub const ACTION_VERSION: &str = "1.0";
pub const HASH_SCHEME: &str = "sglang.sha256.u32le.i64.v1";
pub const MAX_INTEGER: u64 = (1 << 53) - 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Envelope {
    pub protocol_version: String,
    pub message_id: String,
    #[serde(default)]
    pub actions: Vec<Action>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Action {
    pub action_id: String,
    pub action_type: String,
    pub action_version: String,
    #[serde(default = "empty_payload")]
    pub payload: Value,
}
fn empty_payload() -> Value {
    serde_json::json!({})
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Location {
    pub tier: String,
    pub engine_id: Option<String>,
    pub backend: Option<String>,
    pub backend_namespace: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transfer {
    // Validated union retains all legacy Content/Tokens/Session selectors.
    pub target: Value,
    pub source: Location,
    pub destination: Location,
    #[serde(default)]
    pub allow_partial: bool,
    pub timeout_ms: Option<u64>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pin {
    pub mode: String,
    pub target: Value,
    pub location: Location,
    pub claim_id: String,
    pub revision: u64,
    pub ttl_seconds: Option<f64>,
    #[serde(default = "protection")]
    pub required_protection: String,
    pub owner_task: Option<crate::models::TaskIdentity>,
}
fn protection() -> String {
    "eviction_exclusion".into()
}
#[derive(Clone, Debug, PartialEq)]
pub enum ParsedAction {
    Prefetch(Transfer),
    Demote(Transfer),
    Pin(Pin),
    /// Independent consumer payload is not inspected or rewritten.
    Unhandled(Action),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidPayload,
    Unsupported,
}

static SCHEMAS: LazyLock<HashMap<&'static str, jsonschema::Validator>> = LazyLock::new(|| {
    [
        (
            "sglang.prefetch",
            include_str!("../fixtures/prefetch.schema.json"),
        ),
        (
            "sglang.demote",
            include_str!("../fixtures/demote.schema.json"),
        ),
        ("sglang.pin", include_str!("../fixtures/pin.schema.json")),
        ("result", include_str!("../fixtures/result.schema.json")),
        ("claim", include_str!("../fixtures/claim.schema.json")),
    ]
    .into_iter()
    .map(|(name, source)| {
        let schema: Value = serde_json::from_str(source).expect("checked-in JSON schema");
        (
            name,
            jsonschema::validator_for(&schema).expect("valid protocol schema"),
        )
    })
    .collect()
});

/// Structural validation is shared verbatim; dependent range rules are below.
pub fn validate_schema(name: &str, value: &Value) -> Result<(), Error> {
    if SCHEMAS.get(name).ok_or(Error::Unsupported)?.is_valid(value) {
        Ok(())
    } else {
        Err(Error::InvalidPayload)
    }
}

pub fn parse_action(action: &Action) -> Result<ParsedAction, Error> {
    if !action.action_type.starts_with("sglang.") {
        return Ok(ParsedAction::Unhandled(action.clone()));
    }
    if action.action_version != ACTION_VERSION || !SCHEMAS.contains_key(action.action_type.as_str())
    {
        return Err(Error::Unsupported);
    }
    validate_schema(&action.action_type, &action.payload)?;
    strict_numbers(&action.payload, "")?;
    validate_target(&action.payload["target"])?;
    let decode_error = |_| Error::InvalidPayload;
    if action.action_type == "sglang.pin" {
        let pin: Pin = serde_json::from_value(action.payload.clone()).map_err(decode_error)?;
        validate_location(&pin.location)?;
        if (pin.mode == "release") != pin.ttl_seconds.is_none() {
            return Err(Error::InvalidPayload);
        }
        return Ok(ParsedAction::Pin(pin));
    }
    let transfer: Transfer =
        serde_json::from_value(action.payload.clone()).map_err(decode_error)?;
    validate_location(&transfer.source)?;
    validate_location(&transfer.destination)?;
    let rank = |tier: &str| match tier {
        "gpu" => 0,
        "host" => 1,
        _ => 2,
    };
    let src = rank(&transfer.source.tier);
    let dst = rank(&transfer.destination.tier);
    if action.action_type == "sglang.demote" {
        if src >= dst {
            return Err(Error::InvalidPayload);
        }
        Ok(ParsedAction::Demote(transfer))
    } else {
        if src <= dst {
            return Err(Error::InvalidPayload);
        }
        Ok(ParsedAction::Prefetch(transfer))
    }
}

fn strict_numbers(value: &Value, key: &str) -> Result<(), Error> {
    match value {
        Value::Number(n) => {
            if key != "ttl_seconds" && n.as_u64().is_none() && n.as_i64().is_none() {
                return Err(Error::InvalidPayload);
            }
            if n.as_f64().is_none_or(|v| v.abs() > MAX_INTEGER as f64) {
                return Err(Error::InvalidPayload);
            }
        }
        Value::Array(a) => {
            for v in a {
                strict_numbers(v, key)?;
            }
        }
        Value::Object(o) => {
            for (k, v) in o {
                strict_numbers(v, k)?;
            }
        }
        _ => (),
    }
    Ok(())
}
fn validate_location(l: &Location) -> Result<(), Error> {
    if (l.tier == "storage" && l.engine_id.is_some())
        || (l.tier != "storage" && (l.backend.is_some() || l.backend_namespace.is_some()))
    {
        return Err(Error::InvalidPayload);
    }
    Ok(())
}
fn position(v: &Value, key: &str, default: u64) -> u64 {
    v[key].as_u64().unwrap_or(default)
}
fn validate_target(target: &Value) -> Result<(), Error> {
    if target == "request_cached" {
        return Ok(());
    }
    let start = position(target, "start_token", 0);
    if target["end_token"].as_u64().is_some_and(|end| end <= start) {
        return Err(Error::InvalidPayload);
    }
    if target["kind"] == "prefix" {
        let p: PrefixTarget =
            serde_json::from_value(target.clone()).map_err(|_| Error::InvalidPayload)?;
        let c = &p.compatibility;
        if c.components.iter().collect::<HashSet<_>>().len() != c.components.len()
            || p.start_token % c.page_size != 0
            || p.end_token % c.page_size != 0
            || p.token_ids
                .as_ref()
                .is_some_and(|tokens| tokens.len() as u64 != p.end_token)
        {
            return Err(Error::InvalidPayload);
        }
        let mut previous = HashMap::new();
        for b in p.blocks {
            if !c.components.contains(&b.component)
                || b.dependency_start > b.start_token
                || b.start_token >= b.end_token
                || b.end_token > p.end_token
                || b.end_token - b.start_token != c.page_size
                || b.dependency_start % c.page_size != 0
                || b.start_token % c.page_size != 0
                || b.start_token < *previous.get(&b.component).unwrap_or(&0)
            {
                return Err(Error::InvalidPayload);
            }
            previous.insert(b.component, b.end_token);
        }
    } else if target["kind"] == "content" {
        let components = target["components"]
            .as_array()
            .ok_or(Error::InvalidPayload)?;
        if position(target, "token_count", 0) == 0
            || components
                .iter()
                .map(Value::to_string)
                .collect::<HashSet<_>>()
                .len()
                != components.len()
        {
            return Err(Error::InvalidPayload);
        }
        let spans = target["spans"].as_array().ok_or(Error::InvalidPayload)?;
        let mut by_component: HashMap<String, Vec<&Value>> = HashMap::new();
        for span in spans {
            let start = position(span, "start_token", 0);
            let end = position(span, "end_token", 0);
            let kind = span["coverage_kind"].as_str().unwrap_or("tokens");
            if position(span, "dependency_start", 0) > start
                || start > end
                || end == 0
                || end > position(target, "token_count", 0)
                || (kind == "checkpoint") != (start == end)
                || (kind == "window" && position(span, "window_size", 0) == 0)
                || (kind != "window" && !span["window_size"].is_null())
                || !components.contains(&span["component"])
            {
                return Err(Error::InvalidPayload);
            }
            by_component
                .entry(span["component"].to_string())
                .or_default()
                .push(span);
        }
        for spans in by_component.values_mut() {
            spans.sort_by_key(|s| position(s, "start_token", 0));
            for pair in spans.windows(2) {
                if position(pair[0], "end_token", 0) > position(pair[1], "start_token", 0)
                    || pair[0]["start_token"] == pair[1]["start_token"]
                    || pair[0]["coverage_kind"].as_str().unwrap_or("tokens")
                        != pair[1]["coverage_kind"].as_str().unwrap_or("tokens")
                    || pair[0]["window_size"] != pair[1]["window_size"]
                {
                    return Err(Error::InvalidPayload);
                }
            }
        }
    }
    Ok(())
}

/// Raw salt in; encoded salt out. Empty and null deliberately share Python semantics.
/// This helper has no auto-detection: callers must apply it exactly once.
pub fn encode_cache_salt(namespace: &str, salt: Option<&str>) -> String {
    let raw = salt.filter(|s| !s.is_empty());
    let bytes =
        serde_json::to_vec(&("sglang.kv.weight.v1", namespace, raw)).expect("UTF-8 strings");
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn result_and_claim_vectors_remain_separate() {
        let fixture: Value =
            serde_json::from_str(include_str!("../fixtures/results.json")).unwrap();
        for value in fixture["results"].as_array().unwrap() {
            validate_schema("result", value).unwrap();
            let result: crate::models::PublicResult =
                serde_json::from_value(value.clone()).unwrap();
            assert_eq!(serde_json::to_value(result).unwrap(), *value);
        }
        for value in fixture["claims"].as_array().unwrap() {
            validate_schema("claim", value).unwrap();
            let claim: crate::models::ClaimState = serde_json::from_value(value.clone()).unwrap();
            assert!(claim.tombstone);
            assert_eq!(claim.state, crate::models::ClaimStatus::Released);
            assert_eq!(serde_json::to_value(claim).unwrap(), *value);
        }
    }
    #[test]
    fn shared_action_vectors() {
        let cases: Value = serde_json::from_str(include_str!("../fixtures/actions.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let action: Action = serde_json::from_value(case["action"].clone()).unwrap();
            assert_eq!(
                parse_action(&action).is_ok(),
                case["valid"].as_bool().unwrap(),
                "{}",
                case["name"]
            );
        }
    }
    #[test]
    fn salt_unicode_null_and_empty() {
        let cases: Value = serde_json::from_str(include_str!("../fixtures/salt.json")).unwrap();
        for case in cases.as_array().unwrap() {
            assert_eq!(
                encode_cache_salt(case["namespace"].as_str().unwrap(), case["salt"].as_str()),
                case["encoded"]
            );
        }
    }
    #[test]
    fn unknown_consumers_and_versions() {
        let a = Action {
            action_id: "a".into(),
            action_type: "other.future".into(),
            action_version: "99".into(),
            payload: serde_json::json!({"opaque": [null, true, "9223372036854775808"]}),
        };
        assert_eq!(parse_action(&a), Ok(ParsedAction::Unhandled(a.clone())));
        let a = Action {
            action_type: "sglang.prefetch".into(),
            ..a
        };
        assert_eq!(parse_action(&a), Err(Error::Unsupported));
    }
}
