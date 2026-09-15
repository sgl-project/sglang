//! Strict JSON comparison with explicit, validated exceptions for scalar values.
//!
//! Rules never project responses into a smaller schema: the original tree stays
//! intact, and only declared values in a copy are replaced after validation.

use std::collections::BTreeSet;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

/// Where relative exception pointers apply; the complete root is still compared.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonScope {
    Root,
    TopLevelArrayItems,
}

/// The only comparison mode: no numeric tolerances or structural coercions.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonBase {
    ExactJson,
}

/// Validation that must succeed before an exception may replace a value.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValueRequirement {
    NonEmptyString,
    NonNegativeNumber,
}

/// One literal JSON Pointer and the documented reason its value may vary.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ValueException {
    pub path: String,
    pub require: ValueRequirement,
    pub reason: String,
}

/// The suite's complete comparison contract, with no implicit exceptions.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ComparisonRules {
    pub base: ComparisonBase,
    pub per_result_value_exceptions: Vec<ValueException>,
}

impl ComparisonRules {
    /// Reject malformed or duplicate pointers and undocumented exceptions.
    pub fn validate(&self) -> Result<(), String> {
        let mut paths = BTreeSet::new();
        for rule in &self.per_result_value_exceptions {
            if !valid_pointer(&rule.path) {
                return Err(format!(
                    "invalid exception path {:?}: expected a non-root JSON Pointer without wildcards",
                    rule.path
                ));
            }
            if !paths.insert(&rule.path) {
                return Err(format!("duplicate exception path {:?}", rule.path));
            }
            if rule.reason.trim().is_empty() {
                return Err(format!("exception {:?} requires a reason", rule.path));
            }
        }
        Ok(())
    }
}

fn valid_pointer(path: &str) -> bool {
    if !path.starts_with('/') {
        return false;
    }
    path[1..].split('/').all(|token| {
        if token == "*" {
            return false;
        }
        let mut chars = token.chars();
        while let Some(ch) = chars.next() {
            if ch == '~' && !matches!(chars.next(), Some('0' | '1')) {
                return false;
            }
        }
        true
    })
}

/// A malformed response or a failed precondition, not a parity difference.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Violation {
    pub path: String,
    pub message: String,
    pub event: Option<usize>,
}

impl Violation {
    pub fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
            event: None,
        }
    }
}

/// Validate every declared path, then replace only its value in a full copy.
///
/// Callers validate configuration before starting services. This function also
/// rejects invalid rules so direct library calls cannot silently weaken checks.
/// Array cardinality and API-specific response shapes belong to the suite.
pub fn prepare_comparison(
    original: &Value,
    scope: ComparisonScope,
    rules: &ComparisonRules,
) -> Result<Value, Vec<Violation>> {
    rules
        .validate()
        .map_err(|message| vec![Violation::new("", message)])?;
    let prefixes = match scope {
        ComparisonScope::Root => vec![String::new()],
        ComparisonScope::TopLevelArrayItems => {
            let Some(items) = original.as_array() else {
                return Err(vec![Violation::new("", "expected a top-level array")]);
            };
            (0..items.len()).map(|index| format!("/{index}")).collect()
        }
    };
    let mut violations = Vec::new();
    let mut replacements = Vec::new();
    for prefix in prefixes {
        for rule in &rules.per_result_value_exceptions {
            let path = format!("{prefix}{}", rule.path);
            let Some(value) = original.pointer(&path) else {
                violations.push(Violation::new(path, "required exception path is missing"));
                continue;
            };
            let (valid, replacement, message) = match rule.require {
                ValueRequirement::NonEmptyString => (
                    value.as_str().is_some_and(|text| !text.is_empty()),
                    Value::String("<dynamic>".into()),
                    "expected a non-empty string",
                ),
                ValueRequirement::NonNegativeNumber => (
                    value
                        .as_f64()
                        .is_some_and(|number| number.is_finite() && number >= 0.0),
                    Value::from(0),
                    "expected a finite non-negative number",
                ),
            };
            if valid {
                replacements.push((path, replacement));
            } else {
                violations.push(Violation::new(path, message));
            }
        }
    }
    if !violations.is_empty() {
        return Err(violations);
    }

    let mut prepared = original.clone();
    for (path, replacement) in replacements {
        // All targets were validated as scalars, so replacing one cannot remove
        // another target or change the surrounding tree.
        *prepared.pointer_mut(&path).expect("validated scalar path") = replacement;
    }
    Ok(prepared)
}

/// How a pair of JSON trees differs at a particular pointer.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DifferenceKind {
    ValueMismatch,
    TypeMismatch,
    MissingLeft,
    MissingRight,
}

/// A JSON Pointer and both observed values. An absent side is omitted in JSON;
/// a present JSON null remains an explicit null, including on deserialization.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Difference {
    pub path: String,
    pub kind: DifferenceKind,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present_value"
    )]
    pub left: Option<Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present_value"
    )]
    pub right: Option<Value>,
}

fn present_value<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Value>, D::Error> {
    Value::deserialize(deserializer).map(Some)
}

/// Compare every field and array element without normalizing either input.
///
/// Object key order is irrelevant; array order, numeric values, missing keys and
/// JSON null are significant. Paths use JSON Pointer escaping and stable order.
pub fn compare_json(left: &Value, right: &Value) -> Vec<Difference> {
    let mut differences = Vec::new();
    compare_at("", Some(left), Some(right), &mut differences);
    differences
}

fn compare_at(
    path: &str,
    left: Option<&Value>,
    right: Option<&Value>,
    differences: &mut Vec<Difference>,
) {
    if left == right {
        return;
    }
    match (left, right) {
        (Some(Value::Object(left)), Some(Value::Object(right))) => {
            let keys: BTreeSet<_> = left.keys().chain(right.keys()).collect();
            for key in keys {
                compare_at(
                    &format!("{path}/{}", key.replace('~', "~0").replace('/', "~1")),
                    left.get(key),
                    right.get(key),
                    differences,
                );
            }
        }
        (Some(Value::Array(left)), Some(Value::Array(right))) => {
            for index in 0..left.len().max(right.len()) {
                compare_at(
                    &format!("{path}/{index}"),
                    left.get(index),
                    right.get(index),
                    differences,
                );
            }
        }
        _ => differences.push(Difference {
            path: path.into(),
            kind: match (left, right) {
                (None, _) => DifferenceKind::MissingLeft,
                (_, None) => DifferenceKind::MissingRight,
                (Some(left), Some(right))
                    if std::mem::discriminant(left) != std::mem::discriminant(right) =>
                {
                    DifferenceKind::TypeMismatch
                }
                _ => DifferenceKind::ValueMismatch,
            },
            left: left.cloned(),
            right: right.cloned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn rules() -> ComparisonRules {
        serde_json::from_value(json!({
            "base": "exact_json",
            "per_result_value_exceptions": [
                {"path": "/meta/id", "require": "non_empty_string", "reason": "Request ID."},
                {"path": "/meta/time", "require": "non_negative_number", "reason": "Elapsed time."}
            ]
        }))
        .unwrap()
    }

    #[test]
    fn exceptions_preserve_inputs_structure_and_unlisted_fields() {
        let original = json!([
            {"text": " a\n", "meta": {"id": "a", "time": 0.5, "extra": null}},
            {"text": "b", "meta": {"id": "b", "time": 2, "extra": [1, 2]}}
        ]);
        let before = original.clone();
        let prepared =
            prepare_comparison(&original, ComparisonScope::TopLevelArrayItems, &rules()).unwrap();
        assert_eq!(original, before);
        assert_eq!(prepared[0]["text"], " a\n");
        assert_eq!(prepared[0]["meta"]["extra"], Value::Null);
        assert_eq!(prepared[1]["meta"]["extra"], json!([1, 2]));
        assert_eq!(prepared[0]["meta"]["id"], prepared[1]["meta"]["id"]);
        assert_eq!(prepared[0]["meta"]["time"], prepared[1]["meta"]["time"]);
        let mut changed = prepared.clone();
        changed[1]["meta"]["extra"][1] = json!(3);
        assert_eq!(compare_json(&prepared, &changed)[0].path, "/1/meta/extra/1");
        changed.as_array_mut().unwrap().pop();
        assert_eq!(
            compare_json(&prepared, &changed)[0].kind,
            DifferenceKind::MissingRight
        );
    }

    #[test]
    fn missing_null_empty_and_wrong_scalar_values_are_violations() {
        for original in [
            json!({}),
            json!({"meta": {"id": null, "time": null}}),
            json!({"meta": {"id": "", "time": -0.1}}),
            json!({"meta": {"id": [], "time": "0"}}),
        ] {
            let errors =
                prepare_comparison(&original, ComparisonScope::Root, &rules()).unwrap_err();
            assert_eq!(errors.len(), 2);
            assert_eq!(errors[0].path, "/meta/id");
            assert_eq!(errors[1].path, "/meta/time");
        }
        let errors = prepare_comparison(
            &json!([{"meta": {"id": "a", "time": 0}}, {"meta": {"id": "b"}}]),
            ComparisonScope::TopLevelArrayItems,
            &rules(),
        )
        .unwrap_err();
        assert_eq!(errors[0].path, "/1/meta/time");
        assert!(
            prepare_comparison(&json!({}), ComparisonScope::TopLevelArrayItems, &rules()).is_err()
        );
    }

    #[test]
    fn empty_exception_list_preserves_error_responses_and_has_no_defaults() {
        let mut rules = rules();
        rules.per_result_value_exceptions.clear();
        let error = json!({"error": {"message": "invalid request"}});
        assert_eq!(
            prepare_comparison(&error, ComparisonScope::Root, &rules).unwrap(),
            error
        );
    }

    #[test]
    fn rules_reject_unknown_features_and_invalid_or_duplicate_pointers() {
        let mut value = serde_json::to_value(rules()).unwrap();
        value["tolerance"] = json!(0.01);
        assert!(serde_json::from_value::<ComparisonRules>(value).is_err());
        let mut value = serde_json::to_value(rules()).unwrap();
        value["per_result_value_exceptions"][0]["unknown"] = json!(true);
        assert!(serde_json::from_value::<ComparisonRules>(value).is_err());
        for path in ["", "meta/id", "/a~", "/a~2", "/items/*/id"] {
            let mut invalid = rules();
            invalid.per_result_value_exceptions[0].path = path.into();
            assert!(invalid.validate().is_err(), "{path:?}");
        }
        let mut invalid = rules();
        invalid
            .per_result_value_exceptions
            .push(invalid.per_result_value_exceptions[0].clone());
        assert!(invalid.validate().is_err());
        let mut invalid = rules();
        invalid.per_result_value_exceptions[0].reason = " \n".into();
        assert!(invalid.validate().is_err());
        let mut value = serde_json::to_value(rules()).unwrap();
        value["per_result_value_exceptions"][0]["require"] = json!("anything");
        assert!(serde_json::from_value::<ComparisonRules>(value).is_err());
    }

    #[test]
    fn literal_pointer_escapes_resolve_without_normalizing_keys() {
        let mut rules = rules();
        rules.per_result_value_exceptions.truncate(1);
        rules.per_result_value_exceptions[0].path = "/a~1b/~0id".into();
        let original = json!({"a/b": {"~id": "abc"}, "a": {"b": "untouched"}});
        let prepared = prepare_comparison(&original, ComparisonScope::Root, &rules).unwrap();
        assert_eq!(prepared["a"]["b"], "untouched");
        let differences = compare_json(&original, &prepared);
        assert_eq!(differences.len(), 1);
        assert_eq!(differences[0].path, "/a~1b/~0id");
    }

    #[test]
    fn strict_comparison_reports_all_fields_elements_and_numeric_changes() {
        let left: Value = serde_json::from_str(r#"{"z":null,"a":["x",1.0,3]}"#).unwrap();
        let reordered: Value = serde_json::from_str(r#"{"a":["x",1.0,3],"z":null}"#).unwrap();
        assert!(compare_json(&left, &reordered).is_empty());
        let right = json!({"a": ["x", 1.0000000001], "b": null});
        let differences = compare_json(&left, &right);
        assert_eq!(
            differences
                .iter()
                .map(|d| d.path.as_str())
                .collect::<Vec<_>>(),
            ["/a/1", "/a/2", "/b", "/z"]
        );
        assert_eq!(differences[1].kind, DifferenceKind::MissingRight);
        assert_eq!(differences[2].kind, DifferenceKind::MissingLeft);
        assert_eq!(
            compare_json(&json!(""), &Value::Null)[0].kind,
            DifferenceKind::TypeMismatch
        );
    }

    #[test]
    fn serialized_differences_keep_missing_distinct_from_null() {
        let differences = compare_json(&json!({}), &json!({"x": null}));
        let value = serde_json::to_value(&differences).unwrap();
        assert!(value[0].get("left").is_none());
        assert_eq!(value[0].get("right"), Some(&Value::Null));
        assert_eq!(
            serde_json::from_value::<Vec<Difference>>(value).unwrap(),
            differences
        );
    }
}
