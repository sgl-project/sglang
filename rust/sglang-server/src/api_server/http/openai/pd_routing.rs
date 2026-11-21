//! SGLang P/D extensions to Dynamo's standard OpenAI request types, and the
//! DP rank those requests name.

use axum::http::HeaderMap;
use serde::Deserialize;
use sglang_api_types::api::v1::{Int64OrList, OptionalInt64OrList, StringOrList};

use super::super::request_headers::int_header;
use crate::message::request::{
    BootstrapColumns, check_broadcast_budget, normalize_bootstrap_columns,
};
use crate::message::types::OneOrMany;
use crate::message::wire;
use crate::utils::error::Error;

/// Python's OpenAI PDRoutingFields, using generated API carriers. Unlike the
/// native API, OpenAI allows null list elements only for bootstrap_port.
#[derive(Deserialize)]
pub(crate) struct PDRoutingFields {
    pub(crate) bootstrap_host: Option<StringOrList>,
    pub(crate) bootstrap_port: Option<OptionalInt64OrList>,
    pub(crate) bootstrap_room: Option<Int64OrList>,
    #[serde(default, deserialize_with = "lax_rank")]
    pub(crate) routed_dp_rank: Option<i64>,
    pub(crate) disagg_prefill_dp_rank: Option<i64>,
}

/// Validated per-prompt columns, ready for admission.
pub(crate) struct NormalizedPDRouting {
    pub(crate) bootstrap: BootstrapColumns,
    pub(crate) routed_dp_rank: Option<i64>,
    pub(crate) disagg_prefill_dp_rank: Option<i64>,
}

impl PDRoutingFields {
    /// Normalize bootstrap fields per prompt and bound the handlers' host clones
    /// across choices. DP hints remain scalar.
    pub(crate) fn normalize(
        self,
        prompt_count: usize,
        choices_per_prompt: usize,
    ) -> Result<NormalizedPDRouting, Error> {
        let hosts = self
            .bootstrap_host
            .and_then(wire::string_or_list)
            .map(|hosts| match hosts {
                OneOrMany::One(host) => OneOrMany::One(Some(host)),
                OneOrMany::Many(hosts) => OneOrMany::Many(hosts.into_iter().map(Some).collect()),
            });
        let rooms = self
            .bootstrap_room
            .and_then(wire::int64_or_list)
            .map(|rooms| match rooms {
                OneOrMany::One(room) => OneOrMany::One(Some(room)),
                OneOrMany::Many(rooms) => OneOrMany::Many(rooms.into_iter().map(Some).collect()),
            });
        let bootstrap = normalize_bootstrap_columns(
            hosts,
            self.bootstrap_port.and_then(wire::optional_int64_or_list),
            rooms,
            prompt_count,
        )?;
        if choices_per_prompt > 1 {
            check_broadcast_budget(
                bootstrap
                    .bootstrap_hosts
                    .iter()
                    .filter_map(Option::as_ref)
                    .map(String::len)
                    .sum(),
                choices_per_prompt,
                "bootstrap_host",
            )?;
        }
        Ok(NormalizedPDRouting {
            bootstrap,
            routed_dp_rank: self.routed_dp_rank,
            disagg_prefill_dp_rank: self.disagg_prefill_dp_rank,
        })
    }
}

impl NormalizedPDRouting {
    /// A DP rank `route`'s headers name wins over the body's, as in Python
    /// ([`HintRoute::header_rank`]); an unparsable header is a 400.
    pub(super) fn apply_header_rank(
        &mut self,
        route: HintRoute,
        headers: &HeaderMap,
        overrides: bool,
    ) -> Result<(), String> {
        if let Some(rank) = route.header_rank(headers, overrides)? {
            self.routed_dp_rank = Some(rank);
        }
        Ok(())
    }
}

/// Python's router header override (`request_headers.py`), when enabled.
const OVERRIDE_HEADER: &str = "x-override-routed-dp-rank";
/// Python's OpenAI header (`OpenAIServingBase.extract_routed_dp_rank_from_header`).
const DATA_PARALLEL_RANK_HEADER: &str = "x-data-parallel-rank";

/// An OpenAI route Python reads `routed_dp_rank` on, and which headers it
/// reads it from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HintRoute {
    /// `/v1/chat/completions`: the router's `x-override-routed-dp-rank`
    /// header when header overrides are on (`OpenAIServingChat`'s
    /// `apply_header_overrides`), then `X-Data-Parallel-Rank`, then the body's
    /// fields.
    ChatCompletions,
    /// `/v1/completions`: `X-Data-Parallel-Rank`, then the body's fields.
    Completions,
}

impl HintRoute {
    /// The DP rank this route's headers name, which wins over the body's;
    /// `None` when they name none. An unparsable header is Python's 400:
    /// `X-Data-Parallel-Rank` is read first, as Python reads it before it
    /// applies the override, so a malformed one fails even under a valid
    /// override.
    pub(super) fn header_rank(
        self,
        headers: &HeaderMap,
        overrides: bool,
    ) -> Result<Option<i64>, String> {
        let data_parallel_rank = match headers.get(DATA_PARALLEL_RANK_HEADER) {
            Some(value) => {
                let text = String::from_utf8_lossy(value.as_bytes());
                Some(python_int(&text).ok_or_else(|| {
                    format!("Invalid X-Data-Parallel-Rank header: must be an integer, got '{text}'")
                })?)
            }
            None => None,
        };
        if overrides
            && self == Self::ChatCompletions
            && let Some(rank) = int_header(headers, OVERRIDE_HEADER)?
        {
            return Ok(Some(rank));
        }
        Ok(data_parallel_rank)
    }
}

/// Python's `int(text)` for an ASCII decimal: surrounding whitespace, a sign,
/// and single underscores between digits. A value past `i64` saturates, so
/// the range check refuses it rather than the parse.
fn python_int(text: &str) -> Option<i64> {
    let text = text.trim();
    let (negative, digits) = match text.as_bytes().first()? {
        b'-' => (true, &text[1..]),
        b'+' => (false, &text[1..]),
        _ => (false, text),
    };
    if digits
        .split('_')
        .any(|group| group.is_empty() || !group.bytes().all(|b| b.is_ascii_digit()))
    {
        return None;
    }
    let magnitude = digits
        .bytes()
        .filter(u8::is_ascii_digit)
        .fold(0i128, |value, digit| {
            value
                .saturating_mul(10)
                .saturating_add(i128::from(digit - b'0'))
        });
    let value = if negative { -magnitude } else { magnitude };
    Some(value.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64)
}

/// Pydantic's lax `int` (the OpenAI request models): an integer, a float or
/// numeric string with no fraction, or a bool.
fn pydantic_int(value: &serde_json::Value) -> Option<i64> {
    use serde_json::Value;
    let whole = |float: f64| {
        (float.fract() == 0.0 && float >= i64::MIN as f64 && float <= i64::MAX as f64)
            .then_some(float as i64)
    };
    match value {
        Value::Number(number) => number.as_i64().or_else(|| whole(number.as_f64()?)),
        Value::Bool(flag) => Some(i64::from(*flag)),
        // An integer, or one with a fraction of zeros (`"1.0"`).
        Value::String(text) => python_int(text).or_else(|| {
            let (int, fraction) = text.trim().split_once('.')?;
            (!fraction.is_empty() && fraction.bytes().all(|b| b == b'0'))
                .then(|| python_int(int))
                .flatten()
        }),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

/// A DP rank field of an OpenAI request, as pydantic's `Optional[int]` reads
/// it there ([`pydantic_int`]).
fn lax_rank<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<i64>, D::Error> {
    match serde_json::Value::deserialize(deserializer)? {
        serde_json::Value::Null => Ok(None),
        value => pydantic_int(&value).map(Some).ok_or_else(|| {
            serde::de::Error::custom(format!("invalid DP rank {value}: not an integer"))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        pairs
            .iter()
            .map(|&(name, value)| (name.parse().unwrap(), value.parse().unwrap()))
            .collect()
    }

    /// Python's precedence per route: the override header (when enabled, not
    /// on completions), then `X-Data-Parallel-Rank`; and Python's `int()` and
    /// pydantic's lax `int` for the values.
    #[test]
    fn header_rank_follows_python_per_route() {
        use HintRoute::{ChatCompletions, Completions};
        let both = headers(&[(OVERRIDE_HEADER, "1"), (DATA_PARALLEL_RANK_HEADER, "2")]);
        for (route, overrides, sent, expected) in [
            (ChatCompletions, true, &both, Some(1)),
            (ChatCompletions, false, &both, Some(2)),
            (Completions, true, &both, Some(2)),
            (Completions, true, &HeaderMap::new(), None),
        ] {
            assert_eq!(
                route.header_rank(sent, overrides),
                Ok(expected),
                "{route:?} overrides={overrides}"
            );
        }
        assert_eq!(
            Completions.header_rank(&headers(&[(DATA_PARALLEL_RANK_HEADER, "x")]), false),
            Err("Invalid X-Data-Parallel-Rank header: must be an integer, got 'x'".into())
        );
        // Python validates X-Data-Parallel-Rank before it applies the override,
        // and reads it with `int()`.
        let malformed = headers(&[(OVERRIDE_HEADER, "1"), (DATA_PARALLEL_RANK_HEADER, "x")]);
        assert!(ChatCompletions.header_rank(&malformed, true).is_err());
        assert!(
            ChatCompletions
                .header_rank(&headers(&[(OVERRIDE_HEADER, "1.5")]), true)
                .is_err()
        );
        for (text, rank) in [
            (" +0_1 ", Some(1)),
            ("-2", Some(-2)),
            ("1__0", None),
            ("_1", None),
        ] {
            assert_eq!(python_int(text), rank, "{text:?}");
        }
        assert_eq!(python_int(&"9".repeat(30)), Some(i64::MAX));
        assert_eq!(python_int("-9223372036854775808"), Some(i64::MIN));
        for (value, rank) in [
            (json!(1.0), Some(1)),
            (json!(true), Some(1)),
            (json!("1.0"), Some(1)),
            (json!(1.5), None),
            (json!("one"), None),
            (json!("1e0"), None),
            (json!("1.0000000000000001"), None),
        ] {
            assert_eq!(pydantic_int(&value), rank, "{value}");
        }
    }

    #[test]
    fn list_routing_preserves_per_prompt_metadata() {
        // mini_lb injects scalars; the GPU fixture does not cover lists or
        // distinguish nonzero DP hints in its single-rank topology.
        let fields: PDRoutingFields = serde_json::from_value(json!({
            "bootstrap_host": ["prefill-a", "prefill-b"],
            "bootstrap_port": [8998, null],
            "bootstrap_room": [9007199254740993_i64, 9007199254740995_i64],
            "routed_dp_rank": 1,
            "disagg_prefill_dp_rank": 3,
        }))
        .unwrap();
        let routing = fields.normalize(2, 2).unwrap();
        assert_eq!(
            routing.bootstrap.bootstrap_hosts,
            vec![Some("prefill-a".to_owned()), Some("prefill-b".to_owned())]
        );
        assert_eq!(routing.bootstrap.bootstrap_ports, vec![Some(8998), None]);
        assert_eq!(
            routing.bootstrap.bootstrap_rooms,
            vec![Some(9007199254740993), Some(9007199254740995)]
        );
        assert_eq!(routing.routed_dp_rank, Some(1));
        assert_eq!(routing.disagg_prefill_dp_rank, Some(3));
    }

    /// `routed_dp_rank` names the rank, and a header the route reads wins
    /// over it.
    #[test]
    fn dp_rank_reads_the_body_then_the_route_headers() {
        let routing = |body| {
            serde_json::from_value::<PDRoutingFields>(body)
                .unwrap()
                .normalize(1, 1)
                .unwrap()
        };
        // Pydantic reads a string of an integer.
        assert_eq!(
            routing(json!({"routed_dp_rank": "1"})).routed_dp_rank,
            Some(1)
        );
        assert!(serde_json::from_value::<PDRoutingFields>(json!({"routed_dp_rank": "x"})).is_err());
        let mut normalized = routing(json!({"routed_dp_rank": 0}));
        normalized
            .apply_header_rank(
                HintRoute::Completions,
                &headers(&[(DATA_PARALLEL_RANK_HEADER, "1")]),
                false,
            )
            .unwrap();
        assert_eq!(normalized.routed_dp_rank, Some(1));
    }

    #[test]
    fn null_list_elements_are_rejected_except_for_ports() {
        for body in [
            r#"{"bootstrap_host":[null]}"#,
            r#"{"bootstrap_room":[null]}"#,
        ] {
            assert!(serde_json::from_str::<PDRoutingFields>(body).is_err());
        }
    }

    #[test]
    fn list_hosts_cannot_bypass_choice_clone_budget() {
        // 263173 bytes fit for one choice, but 255 choices exceed 64 MiB.
        let fields: PDRoutingFields =
            serde_json::from_value(json!({"bootstrap_host": ["x".repeat(263173)]})).unwrap();
        let error = fields.normalize(1, 255).err().unwrap().to_string();
        assert!(error.contains("bootstrap_host"), "{error}");
        assert!(error.contains("would allocate more than"), "{error}");
    }
}
