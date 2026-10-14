//! The `/generate` wire contract is the generated `sglang_api_types` request
//! (the proto schema is the single source of truth for both servers). This
//! module is the one place its carriers become the internal request types:
//! scalar-or-list columns into [`OneOrMany`], media containers into
//! [`MmDataInput`], and the wire `SamplingParams` into the internal one that
//! carries normalization state. Nothing here validates batch shape; that is
//! [`into_requests`](super::request::into_requests).

use std::collections::BTreeMap;

use sglang_api_types::api::v1 as api;

use super::multimodal::{MediaHints, MmDataInput, MmItem};
use super::sampling::{CustomParamValue, SamplingParams, WatermarkRequestConfig};
use super::types::{OneOrMany, TokenIds};
use crate::utils::error::Error;

pub fn string_or_list(v: api::StringOrList) -> Option<OneOrMany<String>> {
    use api::string_or_list::Value;
    v.value.map(|v| match v {
        Value::One(s) => OneOrMany::One(s),
        Value::Many(l) => OneOrMany::Many(l.items),
    })
}

pub fn int64_or_list(v: api::Int64OrList) -> Option<OneOrMany<i64>> {
    use api::int64_or_list::Value;
    v.value.map(|v| match v {
        Value::One(x) => OneOrMany::One(x),
        Value::Many(l) => OneOrMany::Many(l.items),
    })
}

pub fn bool_or_list(v: api::BoolOrList) -> Option<OneOrMany<bool>> {
    use api::bool_or_list::Value;
    v.value.map(|v| match v {
        Value::One(x) => OneOrMany::One(x),
        Value::Many(l) => OneOrMany::Many(l.items),
    })
}

fn token_ids(v: api::TokenIds) -> TokenIds {
    v.ids
}

pub fn token_ids_or_list(v: api::TokenIdsOrList) -> Option<OneOrMany<TokenIds>> {
    use api::token_ids_or_list::Value;
    v.value.map(|v| match v {
        Value::One(ids) => OneOrMany::One(token_ids(ids)),
        Value::Many(l) => OneOrMany::Many(l.items.into_iter().map(token_ids).collect()),
    })
}

pub fn optional_string_or_list(v: api::OptionalStringOrList) -> Option<OneOrMany<Option<String>>> {
    use api::optional_string_or_list::Value;
    v.value.map(|v| match v {
        Value::One(s) => OneOrMany::One(s.value),
        Value::Many(l) => OneOrMany::Many(l.items.into_iter().map(|s| s.value).collect()),
    })
}

pub fn optional_int64_or_list(v: api::OptionalInt64OrList) -> Option<OneOrMany<Option<i64>>> {
    use api::optional_int64_or_list::Value;
    v.value.map(|v| match v {
        Value::One(x) => OneOrMany::One(x.value),
        Value::Many(l) => OneOrMany::Many(l.items.into_iter().map(|x| x.value).collect()),
    })
}

pub fn string_list_or_list(v: api::StringListOrList) -> Option<OneOrMany<Vec<String>>> {
    use api::string_list_or_list::Value;
    v.value.map(|v| match v {
        Value::One(l) => OneOrMany::One(l.items),
        Value::Many(ll) => OneOrMany::Many(ll.items.into_iter().map(|l| l.items).collect()),
    })
}

/// A media item: a bare source, or a ref whose `url` must be present. proto3
/// strings carry no presence, so an object without `url` decodes as an empty
/// one and is rejected here (Python's `load_image` fails on it the same way).
pub fn media_item(item: api::MediaItem) -> Result<MmItem, Error> {
    use api::media_item::Value;
    match item.value {
        // Only `url` and the hints a Rust processor reads reach the MM
        // pipeline; the rest serve model families it does not run (Python's
        // `load_image` reduces an item to `.url` the same way).
        Some(Value::Ref(r)) if !r.url.is_empty() => Ok(MmItem::Ref {
            url: r.url,
            hints: MediaHints { fps: r.fps },
        }),
        Some(Value::Ref(_)) => Err(Error::Validation(
            "a multimodal item object needs a non-empty `url`".into(),
        )),
        Some(Value::Source(s)) => Ok(MmItem::Source(s)),
        // Unreachable from JSON (the untagged union always sets an arm).
        None => Err(Error::Validation("empty multimodal item".into())),
    }
}

pub fn media_input(input: api::MediaInput) -> Result<MmDataInput, Error> {
    use api::media_input::Value;
    let item = |i: api::OptionalMediaItem| i.item.map(media_item).transpose();
    Ok(match input.value {
        Some(Value::Item(i)) => MmDataInput::One(media_item(i)?),
        Some(Value::Many(l)) => {
            MmDataInput::Many(l.items.into_iter().map(item).collect::<Result<_, _>>()?)
        }
        Some(Value::Nested(ll)) => MmDataInput::Nested(
            ll.items
                .into_iter()
                .map(|entry| {
                    entry
                        .items
                        .map(|l| l.items.into_iter().map(item).collect::<Result<_, _>>())
                        .transpose()
                })
                .collect::<Result<_, _>>()?,
        ),
        None => MmDataInput::Many(Vec::new()),
    })
}

/// The wire params, with the schema defaults already applied by the generated
/// decoder (an absent key reads its default), become the internal params. Only
/// `custom_params` needs parsing: it crosses the wire as JSON text.
impl TryFrom<api::SamplingParams> for SamplingParams {
    type Error = String;

    fn try_from(p: api::SamplingParams) -> Result<Self, String> {
        let defaults = SamplingParams::default();
        let custom_params: Option<BTreeMap<String, CustomParamValue>> = match p.custom_params {
            None => None,
            Some(text) => {
                Some(serde_json::from_str(&text).map_err(|e| format!("custom_params: {e}"))?)
            }
        };
        Ok(SamplingParams {
            max_new_tokens: p.max_new_tokens,
            stop: p.stop.and_then(string_or_list),
            stop_token_ids: p.stop_token_ids.map(|l| l.items),
            stop_regex: p.stop_regex.and_then(string_or_list),
            temperature: p.temperature.unwrap_or(defaults.temperature),
            top_p: p.top_p.unwrap_or(defaults.top_p),
            top_k: p.top_k.unwrap_or(defaults.top_k),
            min_p: p.min_p.unwrap_or(defaults.min_p),
            frequency_penalty: p.frequency_penalty.unwrap_or(defaults.frequency_penalty),
            presence_penalty: p.presence_penalty.unwrap_or(defaults.presence_penalty),
            repetition_penalty: p.repetition_penalty.unwrap_or(defaults.repetition_penalty),
            min_new_tokens: p.min_new_tokens.unwrap_or(defaults.min_new_tokens),
            n: p.n.unwrap_or(defaults.n),
            beam_width: p.beam_width,
            json_schema: p.json_schema,
            regex: p.regex,
            ebnf: p.ebnf,
            structural_tag: p.structural_tag,
            ignore_eos: p.ignore_eos.unwrap_or(defaults.ignore_eos),
            skip_special_tokens: p
                .skip_special_tokens
                .unwrap_or(defaults.skip_special_tokens),
            spaces_between_special_tokens: p
                .spaces_between_special_tokens
                .unwrap_or(defaults.spaces_between_special_tokens),
            no_stop_trim: p.no_stop_trim.unwrap_or(defaults.no_stop_trim),
            stream_interval: p.stream_interval,
            logit_bias: (!p.logit_bias.is_empty()).then(|| p.logit_bias.into_iter().collect()),
            sampling_seed: p.sampling_seed,
            custom_params,
            watermark: p.watermark.map(|watermark| WatermarkRequestConfig {
                enabled: watermark.enabled,
                key: watermark.key,
                context_window: watermark.context_window,
            }),
            ..defaults
        })
    }
}

/// `sampling_params` as sent: one object (a broadcast in a batch) or one per
/// prompt. Not a [`OneOrMany`]: that carrier is sealed to wire-shaped items,
/// and the internal params are not one.
pub enum SamplingInput {
    /// Boxed: the params are ~450 bytes, so an inline variant would make
    /// every input that big regardless of which form arrived.
    One(Box<SamplingParams>),
    Many(Vec<SamplingParams>),
}

pub fn sampling_params_or_list(
    v: api::SamplingParamsOrList,
) -> Result<Option<SamplingInput>, String> {
    use api::sampling_params_or_list::Value;
    Ok(match v.value {
        None => None,
        Some(Value::One(p)) => Some(SamplingInput::One(Box::new(p.try_into()?))),
        Some(Value::Many(l)) => Some(SamplingInput::Many(
            l.items
                .into_iter()
                .map(SamplingParams::try_from)
                .collect::<Result<_, _>>()?,
        )),
    })
}

/// Launch-time preferred sampling params fill the keys a request did not
/// send, at the JSON level, before the body is decoded: `{**preferred,
/// **request}` per object, which is Python TokenizerManager's precedence. A
/// request key wins even when it carries the type's default or null, which
/// a decoded struct can no longer tell apart from an absent key.
pub fn merge_preferred_sampling(
    body: &mut serde_json::Value,
    preferred: &serde_json::Value,
) -> Result<(), String> {
    let preferred = preferred
        .as_object()
        .ok_or_else(|| "preferred_sampling_params must be a JSON object".to_string())?;
    let Some(body) = body.as_object_mut() else {
        return Ok(()); // not an object: the decoder reports it
    };
    let merge_into = |request: &mut serde_json::Value| {
        if let Some(obj) = request.as_object_mut() {
            for (k, v) in preferred {
                obj.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    };
    match body.get_mut("sampling_params") {
        None | Some(serde_json::Value::Null) => {
            body.insert(
                "sampling_params".into(),
                serde_json::Value::Object(preferred.clone()),
            );
        }
        Some(serde_json::Value::Array(items)) => items.iter_mut().for_each(merge_into),
        Some(one) => merge_into(one),
    }
    Ok(())
}

/// The same fill for a request that arrived already decoded (gRPC): the
/// carrier round-trips through its JSON form so protobuf and HTTP share one
/// precedence rule. Protobuf has no null, so an unset field is an absent key
/// (the schema's null-emitting fields are stripped before the merge), never
/// the explicit null a JSON client can send.
pub fn fill_preferred_sampling(
    params: Option<api::SamplingParamsOrList>,
    preferred: &serde_json::Value,
) -> Result<Option<api::SamplingParamsOrList>, String> {
    let mut body = serde_json::json!({ "sampling_params": params });
    let strip_nulls = |object: &mut serde_json::Value| {
        if let Some(map) = object.as_object_mut() {
            map.retain(|_, value| !value.is_null());
        }
    };
    match &mut body["sampling_params"] {
        serde_json::Value::Array(items) => items.iter_mut().for_each(strip_nulls),
        one => strip_nulls(one),
    }
    merge_preferred_sampling(&mut body, preferred)?;
    serde_json::from_value(body["sampling_params"].take()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(body: &str, preferred: serde_json::Value) -> Vec<SamplingParams> {
        let mut value: serde_json::Value = serde_json::from_str(body).unwrap();
        merge_preferred_sampling(&mut value, &preferred).unwrap();
        let req: api::GenerateRequest = serde_json::from_value(value).unwrap();
        match sampling_params_or_list(req.sampling_params.unwrap())
            .unwrap()
            .unwrap()
        {
            SamplingInput::One(p) => vec![*p],
            SamplingInput::Many(v) => v,
        }
    }

    /// A request key wins over the preferred value, even as an explicit
    /// default or null; only omitted keys take the preferred value.
    #[test]
    fn preferred_params_fill_only_omitted_request_fields() {
        let preferred =
            serde_json::json!({"temperature": 0.25, "top_p": 0.75, "max_new_tokens": 4096});
        let p = &decode(
            r#"{"sampling_params": {"temperature": 1.0, "top_p": null}}"#,
            preferred.clone(),
        )[0];
        assert_eq!(p.temperature, 1.0, "explicit default wins");
        assert_eq!(p.top_p, 1.0, "explicit null keeps the type default");
        assert_eq!(p.max_new_tokens, Some(4096), "omitted uses preferred");
        let p = &decode(r#"{"text": "hi"}"#, preferred)[0];
        assert_eq!(
            (p.temperature, p.max_new_tokens),
            (0.25, Some(4096)),
            "absent object takes all"
        );
    }

    #[test]
    fn preferred_params_apply_to_every_batched_object() {
        let preferred = serde_json::json!({"temperature": 0.25, "top_p": 0.75});
        let ps = decode(
            r#"{"sampling_params": [{"temperature": 0.5}, {"top_p": 0.9}]}"#,
            preferred,
        );
        assert_eq!((ps[0].temperature, ps[0].top_p), (0.5, 0.75));
        assert_eq!((ps[1].temperature, ps[1].top_p), (0.25, 0.9));
    }

    /// The generated `TokenIdsOrList` and the crate's own `OneOrMany<TokenIds>`
    /// accept and reject the same shapes (the first-element rule), so the
    /// swap to the wire type changed no input_ids contract.
    #[test]
    fn token_ids_decoding_matches_one_or_many() {
        for value in [
            "null",
            "[]",
            "[0]",
            "[-2147483648,2147483647]",
            "[[]]",
            "[[],[1,-2]]",
            "[[1,2],[3]]",
            "1",
            "true",
            "{}",
            r#""1""#,
            "[null]",
            "[true]",
            r#"["1"]"#,
            "[1.0]",
            "[-0]",
            "[2147483648]",
            "[0,2147483648]",
            "[1,[2]]",
            "[[1],2]",
            "[[[1]]]",
            "[[1.0]]",
        ] {
            let expected = serde_json::from_str::<Option<OneOrMany<TokenIds>>>(value);
            let actual = serde_json::from_str::<Option<api::TokenIdsOrList>>(value)
                .map(|v| v.and_then(token_ids_or_list));
            match (expected, actual) {
                (Ok(e), Ok(a)) => assert_eq!(a, e, "{value}"),
                (Err(_), Err(_)) => {}
                (e, a) => panic!("{value}: expected {e:?}, got {a:?}"),
            }
        }
    }

    #[test]
    fn media_input_converts_every_container_form() {
        let conv = |json: &str| -> MmDataInput {
            media_input(serde_json::from_str::<api::MediaInput>(json).unwrap()).unwrap()
        };
        let src = |s: &str| MmItem::Source(s.to_owned());
        assert_eq!(conv(r#""u""#), MmDataInput::One(src("u")));
        assert_eq!(
            conv(r#"{"url": "u", "detail": "high", "fps": 2.0}"#),
            MmDataInput::One(MmItem::Ref {
                url: "u".into(),
                hints: MediaHints { fps: Some(2.0) },
            })
        );
        assert_eq!(
            conv(r#"["a", null, "b"]"#),
            MmDataInput::Many(vec![Some(src("a")), None, Some(src("b"))])
        );
        assert_eq!(conv("[]"), MmDataInput::Many(vec![]));
        assert_eq!(
            conv(r#"[["a", null], null, []]"#),
            MmDataInput::Nested(vec![Some(vec![Some(src("a")), None]), None, Some(vec![])])
        );
    }

    #[test]
    fn custom_params_cross_as_json_text() {
        let p: api::SamplingParams =
            serde_json::from_str(r#"{"custom_params": {"k": [1, "x"]}, "logit_bias": {"5": 1.5}}"#)
                .unwrap();
        let p = SamplingParams::try_from(p).unwrap();
        assert!(
            p.custom_params
                .as_ref()
                .is_some_and(|m| m.contains_key("k"))
        );
        assert_eq!(
            p.logit_bias.as_ref().and_then(|m| m.get("5")).copied(),
            Some(1.5)
        );
        assert_eq!(p.temperature, 1.0, "schema default applied by the decoder");
    }

    #[test]
    fn watermark_crosses_generated_schema() {
        let key = "0123456789abcdef";
        let p: api::SamplingParams = serde_json::from_value(serde_json::json!({
            "watermark": {"enabled": true, "key": key, "context_window": 4}
        }))
        .unwrap();
        let p = SamplingParams::try_from(p).unwrap();
        let watermark = p.watermark.as_ref().unwrap();
        assert_eq!(watermark.enabled, Some(true));
        assert_eq!(watermark.key.as_deref(), Some(key));
        assert_eq!(watermark.context_window, Some(4));
        assert!(!format!("{p:?}").contains(key));
    }

    /// The protobuf entry fills the same keys the JSON entry does: a set field
    /// wins, an unset one takes the preferred value, and an absent carrier
    /// takes them all.
    #[test]
    fn preferred_params_fill_unset_protobuf_fields() {
        use api::sampling_params_or_list::Value;
        let preferred = serde_json::json!({"temperature": 0.25, "max_new_tokens": 4096});
        let sent = api::SamplingParamsOrList {
            value: Some(Value::One(api::SamplingParams {
                temperature: Some(1.0),
                ..Default::default()
            })),
        };
        let filled = fill_preferred_sampling(Some(sent), &preferred)
            .unwrap()
            .unwrap();
        let Some(Value::One(p)) = filled.value else {
            panic!("one object in, one object out");
        };
        assert_eq!((p.temperature, p.max_new_tokens), (Some(1.0), Some(4096)));

        let filled = fill_preferred_sampling(None, &preferred).unwrap().unwrap();
        let Some(Value::One(p)) = filled.value else {
            panic!("an absent carrier becomes the preferred object");
        };
        assert_eq!((p.temperature, p.max_new_tokens), (Some(0.25), Some(4096)));
    }
}
