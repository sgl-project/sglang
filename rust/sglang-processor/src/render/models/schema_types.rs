//! `function_call/utils.py::normalize_json_schema_types`, which
//! `serving_chat._validate_request` runs on every tool's `parameters` in place,
//! so the prompt shows the rewritten schema.

use serde_json::Value;

const STANDARD_TYPES: &[&str] = &[
    "null", "boolean", "object", "array", "number", "string", "integer",
];

const TYPE_ALIASES: &[(&str, &str)] = &[
    ("str", "string"),
    ("text", "string"),
    ("varchar", "string"),
    ("char", "string"),
    ("enum", "string"),
    ("uuid", "string"),
    ("date", "string"),
    ("datetime", "string"),
    ("time", "string"),
    ("timestamp", "string"),
    ("binary", "string"),
    ("blob", "string"),
    ("bytea", "string"),
    ("bytes", "string"),
    ("varbinary", "string"),
    ("bool", "boolean"),
    ("bigint", "integer"),
    ("smallint", "integer"),
    ("tinyint", "integer"),
    ("double", "number"),
    ("decimal", "number"),
    ("real", "number"),
    ("numeric", "number"),
    ("arr", "array"),
    ("tuple", "array"),
    ("set", "array"),
    ("map", "object"),
];

const PREFIX_RULES: &[(&[&str], &str)] = &[
    (&["int", "uint", "long", "short", "unsigned"], "integer"),
    (&["num", "float"], "number"),
    (&["list"], "array"),
    (&["dict"], "object"),
];

const PREFIX_BOUNDARY_CHARS: &str = "0123456789[<( \t";

const SCHEMA_MAPS: &[&str] = &[
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
];
const SCHEMA_LISTS: &[&str] = &["anyOf", "oneOf", "allOf", "prefixItems"];
const SCHEMA_VALUES: &[&str] = &[
    "items",
    "additionalProperties",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "propertyNames",
    "unevaluatedItems",
    "unevaluatedProperties",
];

pub(super) fn normalize_json_schema_types(schema: &mut Value) {
    match schema {
        Value::Array(items) => items.iter_mut().for_each(normalize_json_schema_types),
        Value::Object(schema) => {
            match schema.get_mut("type") {
                Some(Value::String(t)) => {
                    if let Some(normalized) = normalize_type(t) {
                        *t = normalized;
                    }
                }
                Some(Value::Array(types)) => {
                    let mut normalized: Vec<Value> = Vec::with_capacity(types.len());
                    for t in types.drain(..) {
                        let t = match t.as_str().and_then(normalize_type) {
                            Some(normalized) => normalized.into(),
                            None => t,
                        };
                        if !normalized.contains(&t) {
                            normalized.push(t);
                        }
                    }
                    *types = normalized;
                }
                _ => {}
            }
            for key in SCHEMA_MAPS {
                if let Some(Value::Object(nested)) = schema.get_mut(*key) {
                    nested.values_mut().for_each(normalize_json_schema_types);
                }
            }
            for key in SCHEMA_LISTS {
                if let Some(Value::Array(nested)) = schema.get_mut(*key) {
                    nested.iter_mut().for_each(normalize_json_schema_types);
                }
            }
            for key in SCHEMA_VALUES {
                if let Some(nested) = schema.get_mut(*key) {
                    normalize_json_schema_types(nested);
                }
            }
        }
        _ => {}
    }
}

/// `_normalize_single_type`; `None` leaves the value as it is.
fn normalize_type(raw: &str) -> Option<String> {
    if STANDARD_TYPES.contains(&raw) {
        return None;
    }
    // Python's `str.strip` also strips U+001C..U+001F.
    let base = raw
        .split('(')
        .next()
        .unwrap_or_default()
        .trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
        .to_lowercase();
    if STANDARD_TYPES.contains(&base.as_str()) {
        return Some(base);
    }
    if let Some((_, mapped)) = TYPE_ALIASES.iter().find(|(alias, _)| *alias == base) {
        return Some((*mapped).to_owned());
    }
    PREFIX_RULES
        .iter()
        .find(|(prefixes, _)| prefixes.iter().any(|p| matches_prefix(&base, p)))
        .map(|(_, target)| (*target).to_owned())
}

/// A prefix spans the whole name or is followed by a non-identifier character.
fn matches_prefix(base: &str, prefix: &str) -> bool {
    base == prefix
        || base
            .strip_prefix(prefix)
            .and_then(|rest| rest.chars().next())
            .is_some_and(|c| PREFIX_BOUNDARY_CHARS.contains(c))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::normalize_json_schema_types;

    /// Expected values from `normalize_json_schema_types` in Python.
    #[test]
    fn rewrites_db_type_names_as_python_does() {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "varchar(255)"},
                "b": {"type": "Int32"},
                "c": {"type": ["float", "double", "null"]},
                "d": {"type": "internal"},
                "e": {"type": "list[str]"},
                "f": {"type": "String"},
                "g": {"type": "array", "items": {"type": "uuid"}},
                "h": {"anyOf": [{"type": "bool"}, {"type": "map"}]},
            },
            "$defs": {"x": {"type": "unsigned int"}},
        });
        normalize_json_schema_types(&mut schema);
        assert_eq!(
            schema,
            json!({
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "integer"},
                    "c": {"type": ["number", "null"]},
                    "d": {"type": "internal"},
                    "e": {"type": "array"},
                    "f": {"type": "string"},
                    "g": {"type": "array", "items": {"type": "string"}},
                    "h": {"anyOf": [{"type": "boolean"}, {"type": "object"}]},
                },
                "$defs": {"x": {"type": "integer"}},
            })
        );
    }
}
