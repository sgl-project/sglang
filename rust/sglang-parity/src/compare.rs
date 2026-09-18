//! Strict JSON comparison with explicit, validated exceptions for scalar values.
//!
//! Rules never project responses into a smaller schema: the original tree stays
//! intact, and only declared values in a copy are replaced or rounded after validation.

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

/// Whether an exception requires its field; optional fields retain their absence.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FieldPresence {
    #[default]
    Required,
    Optional,
}

/// One literal JSON Pointer and the documented reason its value may vary.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ValueException {
    pub path: String,
    #[serde(default)]
    pub presence: FieldPresence,
    pub require: ValueRequirement,
    pub reason: String,
}

/// Precision used for explicitly selected numeric values, without a tolerance.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NumericPrecision {
    Float32,
}

/// A numeric JSON Pointer pattern; `*` selects each array item or object member.
/// Missing fields and nulls remain unchanged; other values must be numbers that
/// stay finite after rounding. API suites own presence/type contracts.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NumericRule {
    pub path: String,
    pub precision: NumericPrecision,
    pub reason: String,
}

impl NumericRule {
    /// Whether this rule applies to a concrete, escaped JSON Pointer.
    pub fn matches(&self, pointer: &str) -> bool {
        let mut tokens = pointer.split('/');
        self.path.split('/').all(|pattern| {
            tokens
                .next()
                .is_some_and(|token| pattern == "*" || pattern == token)
        }) && tokens.next().is_none()
    }
}

/// The suite's complete comparison contract, with no implicit exceptions.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ComparisonRules {
    pub base: ComparisonBase,
    pub per_result_value_exceptions: Vec<ValueException>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_result_numeric_rules: Vec<NumericRule>,
}

impl ComparisonRules {
    /// Reject malformed or duplicate pointers and undocumented exceptions.
    pub fn validate(&self) -> Result<(), String> {
        let mut paths = BTreeSet::new();
        for rule in &self.per_result_value_exceptions {
            if !valid_pointer(&rule.path, false) {
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
        for rule in &self.per_result_numeric_rules {
            if !valid_pointer(&rule.path, true) || rule.reason.trim().is_empty() {
                return Err(format!(
                    "numeric rule {:?} requires a valid non-root pointer pattern and a reason",
                    rule.path
                ));
            }
            if !paths.insert(&rule.path)
                || self
                    .per_result_value_exceptions
                    .iter()
                    .any(|exception| rule.matches(&exception.path))
            {
                return Err(format!("overlapping comparison rules at {:?}", rule.path));
            }
        }
        Ok(())
    }
}

fn valid_pointer(path: &str, wildcards: bool) -> bool {
    if !path.starts_with('/') {
        return false;
    }
    path[1..].split('/').all(|token| {
        if token == "*" && !wildcards {
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

/// Validate declared values, then replace them in a full copy, preserving absence.
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
                if rule.presence == FieldPresence::Required {
                    violations.push(Violation::new(path, "required exception path is missing"));
                }
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
    normalize_numbers(&mut prepared, scope, &rules.per_result_numeric_rules)?;
    Ok(prepared)
}

/// Round declared numeric fields in a copy of an API's semantic projection.
/// Dynamic-value exceptions do not apply: a projection may omit those fields.
pub fn prepare_numeric_comparison(
    original: &Value,
    scope: ComparisonScope,
    rules: &ComparisonRules,
) -> Result<Value, Vec<Violation>> {
    rules
        .validate()
        .map_err(|message| vec![Violation::new("", message)])?;
    let mut prepared = original.clone();
    normalize_numbers(&mut prepared, scope, &rules.per_result_numeric_rules)?;
    Ok(prepared)
}

fn normalize_numbers(
    value: &mut Value,
    scope: ComparisonScope,
    rules: &[NumericRule],
) -> Result<(), Vec<Violation>> {
    if scope == ComparisonScope::TopLevelArrayItems && !value.is_array() {
        return Err(vec![Violation::new("", "expected a top-level array")]);
    }
    let mut violations = Vec::new();
    for rule in rules {
        let tokens: Vec<_> = rule.path[1..].split('/').collect();
        match scope {
            ComparisonScope::Root => round_at(value, "", &tokens, rule.precision, &mut violations),
            ComparisonScope::TopLevelArrayItems => {
                let items = value.as_array_mut().expect("validated array scope");
                for (index, item) in items.iter_mut().enumerate() {
                    round_at(
                        item,
                        &format!("/{index}"),
                        &tokens,
                        rule.precision,
                        &mut violations,
                    );
                }
            }
        }
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

fn round_at(
    value: &mut Value,
    path: &str,
    tokens: &[&str],
    precision: NumericPrecision,
    violations: &mut Vec<Violation>,
) {
    let Some((token, rest)) = tokens.split_first() else {
        if value.is_null() {
            return;
        }
        let rounded = match precision {
            NumericPrecision::Float32 => value.as_f64().map(|n| n as f32),
        };
        if let Some(number) = rounded.filter(|n| n.is_finite()) {
            *value = Value::from(f64::from(number));
        } else {
            violations.push(Violation::new(
                path,
                "expected a number finite at float32 precision or null",
            ));
        }
        return;
    };
    if *token == "*" {
        match value {
            Value::Array(items) => {
                for (index, item) in items.iter_mut().enumerate() {
                    round_at(
                        item,
                        &format!("{path}/{index}"),
                        rest,
                        precision,
                        violations,
                    );
                }
            }
            Value::Object(items) => {
                for (key, item) in items {
                    let key = key.replace('~', "~0").replace('/', "~1");
                    round_at(item, &format!("{path}/{key}"), rest, precision, violations);
                }
            }
            _ => {}
        }
    } else if let Some(item) = value.pointer_mut(&format!("/{token}")) {
        round_at(
            item,
            &format!("{path}/{token}"),
            rest,
            precision,
            violations,
        );
    }
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

    fn numeric_rules() -> ComparisonRules {
        serde_json::from_value(json!({
            "base":"exact_json", "per_result_value_exceptions":[],
            "per_result_numeric_rules":[{"path":"/items/*/score","precision":"float32","reason":"Use the source precision."}]
        })).unwrap()
    }

    #[test]
    fn float32_comparison_preserves_evidence_structure_and_unlisted_numbers() {
        let rules = numeric_rules();
        for scope in [ComparisonScope::Root, ComparisonScope::TopLevelArrayItems] {
            let wrap = |v| {
                if scope == ComparisonScope::Root {
                    v
                } else {
                    json!([v])
                }
            };
            let prepare = |v| prepare_comparison(&wrap(v), scope, &rules).unwrap();
            for (left, right, equal) in [
                (-0.24555964767932892, -0.24555965, true),
                (
                    -0.24555964767932892,
                    f64::from(f32::from_bits((-0.24555965_f32).to_bits() + 1)),
                    false,
                ),
                (0.0, -0.0, true),
                (1e-50, 0.0, true),
                (f64::from(f32::from_bits(1)), 0.0, false),
                (f64::from(f32::MAX), f64::from(f32::MAX), true),
            ] {
                let original = json!({"items":[{"score":left}, {"score":null}, {}],"count":16777217,"time":0.123456789});
                let saved = original.clone();
                let mut other = original.clone();
                other["items"][0]["score"] = json!(right);
                let comparison = prepare(original.clone());
                assert_eq!(compare_json(&comparison, &prepare(other)).is_empty(), equal);
                assert_eq!(original, saved);
                let item = if scope == ComparisonScope::Root {
                    &comparison
                } else {
                    &comparison[0]
                };
                assert_eq!(item["count"], 16777217);
                assert_eq!(item["time"], 0.123456789);
                assert_eq!(item["items"][1], json!({"score":null}));
                assert_eq!(item["items"][2], json!({}));
            }
            assert!(
                !compare_json(
                    &prepare(json!({"items":[{}]})),
                    &prepare(json!({"items":[{"score":null}]}))
                )
                .is_empty()
            );
            assert!(
                !compare_json(
                    &prepare(json!({"items":[]})),
                    &prepare(json!({"items":[{}]}))
                )
                .is_empty()
            );
        }
    }

    #[test]
    fn numeric_patterns_handle_escaped_members_and_reject_invalid_values() {
        let mut rules = numeric_rules();
        rules.per_result_numeric_rules[0].path = "/a~1b/*/~0score".into();
        let rule = &rules.per_result_numeric_rules[0];
        assert!(rule.matches("/a~1b/x~1y/~0score"));
        assert!(!rule.matches("/a~1b/x/y/~0score"));
        assert!(!rule.matches("/a~1b/x"));
        assert!(!rule.matches("/a~1b/x/~0score/extra"));
        for number in [
            json!(-0.24555965),
            json!(1e40),
            json!("-0.1"),
            json!(true),
            json!([]),
            json!({}),
        ] {
            let value = json!({"a/b":{"x/y":{"~score":number}}});
            let result = prepare_comparison(&value, ComparisonScope::Root, &rules);
            if number == json!(-0.24555965) {
                assert_eq!(
                    result.unwrap()["a/b"]["x/y"]["~score"],
                    json!(f64::from(-0.24555965_f32))
                );
            } else {
                assert_eq!(result.unwrap_err()[0].path, "/a~1b/x~1y/~0score");
            }
        }
        rules.per_result_value_exceptions = self::rules().per_result_value_exceptions;
        let projection = json!({"a/b":{"token":{"~score":-0.24555965}}});
        assert!(prepare_comparison(&projection, ComparisonScope::Root, &rules).is_err());
        assert!(prepare_numeric_comparison(&projection, ComparisonScope::Root, &rules).is_ok());
        assert!(
            prepare_numeric_comparison(&projection, ComparisonScope::TopLevelArrayItems, &rules)
                .is_err()
        );
    }

    #[test]
    fn numeric_rules_require_valid_documented_unambiguous_paths() {
        for path in ["", "items/*/score", "/a~", "/a~2"] {
            let mut rules = numeric_rules();
            rules.per_result_numeric_rules[0].path = path.into();
            assert!(prepare_comparison(&json!({}), ComparisonScope::Root, &rules).is_err());
        }
        let mut rules = numeric_rules();
        rules
            .per_result_numeric_rules
            .push(rules.per_result_numeric_rules[0].clone());
        assert!(rules.validate().is_err());
        let mut rules = numeric_rules();
        rules.per_result_numeric_rules[0].reason.clear();
        assert!(rules.validate().is_err());
        let mut rules = numeric_rules();
        rules.per_result_numeric_rules[0].path = "/meta/*".into();
        rules.per_result_value_exceptions = self::rules().per_result_value_exceptions;
        assert!(rules.validate().is_err());
        let mut value = serde_json::to_value(numeric_rules()).unwrap();
        value["per_result_numeric_rules"][0]["precision"] = json!("approximate");
        assert!(serde_json::from_value::<ComparisonRules>(value).is_err());
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
        assert_eq!(
            prepared,
            json!([
                {"text": " a\n", "meta": {"id": "<dynamic>", "time": 0, "extra": null}},
                {"text": "b", "meta": {"id": "<dynamic>", "time": 0, "extra": [1, 2]}}
            ])
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
    fn optional_fields_validate_present_values_and_preserve_presence_differences() {
        let rules: ComparisonRules = serde_json::from_value(json!({
            "base": "exact_json",
            "per_result_value_exceptions": [{
                "path": "/time", "presence": "optional",
                "require": "non_negative_number", "reason": "Elapsed time."
            }]
        }))
        .unwrap();
        for scope in [ComparisonScope::Root, ComparisonScope::TopLevelArrayItems] {
            let wrap = |value: Value| match scope {
                ComparisonScope::Root => value,
                ComparisonScope::TopLevelArrayItems => json!([{"time": 5}, value]),
            };
            let path = if scope == ComparisonScope::Root {
                "/time"
            } else {
                "/1/time"
            };
            for (left, right, expected) in [
                (json!({"time": 1}), json!({"time": 2.5}), None),
                (json!({}), json!({}), None),
                (
                    json!({"time": 0}),
                    json!({}),
                    Some(DifferenceKind::MissingRight),
                ),
                (
                    json!({}),
                    json!({"time": 1}),
                    Some(DifferenceKind::MissingLeft),
                ),
            ] {
                let (left, right) = (wrap(left), wrap(right));
                let originals = (left.clone(), right.clone());
                let prepared_left = prepare_comparison(&left, scope, &rules).unwrap();
                let prepared_right = prepare_comparison(&right, scope, &rules).unwrap();
                let differences = compare_json(&prepared_left, &prepared_right);
                assert_eq!(
                    differences
                        .iter()
                        .map(|d| (d.path.as_str(), d.kind))
                        .collect::<Vec<_>>(),
                    expected
                        .map(|kind| (path, kind))
                        .into_iter()
                        .collect::<Vec<_>>()
                );
                assert_eq!((left, right), originals);
            }
            for invalid in [
                json!(null),
                json!("1"),
                json!(-1),
                json!(true),
                json!([]),
                json!({}),
            ] {
                let errors =
                    prepare_comparison(&wrap(json!({"time": invalid})), scope, &rules).unwrap_err();
                assert_eq!(errors.len(), 1);
                assert_eq!(errors[0].path, path);
            }
        }
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
            assert!(
                prepare_comparison(&json!([]), ComparisonScope::TopLevelArrayItems, &invalid)
                    .is_err(),
                "direct comparison accepted invalid rules: {path:?}"
            );
        }
        let mut invalid = rules();
        invalid
            .per_result_value_exceptions
            .push(invalid.per_result_value_exceptions[0].clone());
        assert!(invalid.validate().is_err());
        let mut invalid = rules();
        invalid.per_result_value_exceptions[0].reason = " \n".into();
        assert!(invalid.validate().is_err());
        for key in ["require", "presence"] {
            let mut value = serde_json::to_value(rules()).unwrap();
            value["per_result_value_exceptions"][0][key] = json!("anything");
            assert!(serde_json::from_value::<ComparisonRules>(value).is_err());
        }
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
