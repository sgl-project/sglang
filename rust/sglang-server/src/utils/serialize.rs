//! Python-`int(...)`-tolerant integer deserialization: accepts a JSON number
//! or a numeric string (surrounding whitespace ok), so wire fields the Python
//! side coerces with `int(data[...])` stay as tolerant here as the original.
//! Field types remain plain integers — apply per field with
//! `#[serde(deserialize_with = "parse_int")]`; generic over any
//! `FromStr + Deserialize` integer width. The `_opt` / `_vec` variants exist
//! because serde's `deserialize_with` does not compose through containers.

use serde::{Deserialize, Deserializer};

/// Wire form: a number or a numeric string.
#[derive(Deserialize)]
#[serde(untagged)]
enum RawInt<T> {
    Num(T),
    Str(String),
}

impl<T: std::str::FromStr> RawInt<T> {
    fn resolve<E: serde::de::Error>(self) -> Result<T, E> {
        match self {
            RawInt::Num(v) => Ok(v),
            // `int(...)` tolerates surrounding whitespace.
            RawInt::Str(s) => s
                .trim()
                .parse()
                .map_err(|_| E::custom(format!("invalid int: {s:?}"))),
        }
    }
}

pub fn parse_int<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + std::str::FromStr,
{
    RawInt::deserialize(deserializer)?.resolve()
}

pub fn parse_int_opt<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + std::str::FromStr,
{
    Option::<RawInt<T>>::deserialize(deserializer)?
        .map(RawInt::resolve)
        .transpose()
}

pub fn parse_int_vec<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + std::str::FromStr,
{
    Vec::<RawInt<T>>::deserialize(deserializer)?
        .into_iter()
        .map(RawInt::resolve)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One struct exercising all three container shapes and two widths.
    #[derive(Deserialize)]
    struct Probe {
        #[serde(deserialize_with = "parse_int")]
        scalar: i64,
        #[serde(deserialize_with = "parse_int")]
        narrow: u16,
        #[serde(default, deserialize_with = "parse_int_opt")]
        opt: Option<i64>,
        #[serde(deserialize_with = "parse_int_vec")]
        vec: Vec<i64>,
    }

    /// Python `int(...)` parity across scalar/Option/Vec and integer widths:
    /// numbers and (whitespace-padded) numeric strings both parse; a missing
    /// optional defaults. Guards the wire tolerance for every consumer, not
    /// just the one field the pd_bootstrap HTTP contract test pins.
    #[test]
    fn accepts_numbers_and_numeric_strings() {
        let p: Probe = serde_json::from_value(serde_json::json!({
            "scalar": " 17000 ",
            "narrow": "8998",
            "opt": 3,
            "vec": [1, "2", " 3 "],
        }))
        .unwrap();
        assert_eq!(
            (p.scalar, p.narrow, p.opt, p.vec),
            (17000, 8998, Some(3), vec![1, 2, 3])
        );

        let p: Probe =
            serde_json::from_value(serde_json::json!({"scalar": 1, "narrow": 2, "vec": []}))
                .unwrap();
        assert_eq!(p.opt, None, "missing optional defaults to None");
    }

    /// Non-numeric strings and out-of-range values are errors, not silent
    /// defaults — the tolerance is exactly `int(...)`-wide, no wider.
    #[test]
    fn rejects_non_numeric_and_out_of_range() {
        for body in [
            serde_json::json!({"scalar": "abc", "narrow": 1, "vec": []}),
            serde_json::json!({"scalar": 1, "narrow": "70000", "vec": []}), // > u16::MAX
            serde_json::json!({"scalar": 1, "narrow": 1, "vec": ["4x"]}),
            serde_json::json!({"scalar": 1, "narrow": 1, "opt": "no", "vec": []}),
        ] {
            assert!(
                serde_json::from_value::<Probe>(body.clone()).is_err(),
                "{body}"
            );
        }
    }
}
