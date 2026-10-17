//! Kimi K2.5 checkpoint-compatible tool declaration preprocessing.

use std::collections::HashMap;
use std::fmt::Write as _;

use serde_json::{Map, Value};

const INDENT: &str = "  ";
const FIELD_DELIMITER: &str = ",\n";
const MAX_RECURSION_DEPTH: usize = 32;

pub(crate) fn deep_sort(value: &mut Value) {
    match value {
        Value::Object(object) => {
            let mut entries: Vec<_> = std::mem::take(object).into_iter().collect();
            for (_, value) in &mut entries {
                deep_sort(value);
            }
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            *object = entries.into_iter().collect::<Map<_, _>>();
        }
        Value::Array(array) => {
            for value in array {
                deep_sort(value);
            }
        }
        _ => {}
    }
}

pub(crate) fn encode_tools_to_typescript(tools: &[Value]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    let mut functions = Vec::new();
    for tool in tools {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            continue;
        }
        let function = match tool.get("function") {
            Some(function)
                if function
                    .as_object()
                    .is_some_and(|object| !object.is_empty()) =>
            {
                function
            }
            _ => continue,
        };
        match encode_function(function) {
            Some(function) => functions.push(function),
            None => {
                tracing::warn!(
                    "Kimi K2.5 tool schema is unsupported by the TypeScript encoder; using the checkpoint JSON fallback"
                );
                return None;
            }
        }
    }
    if functions.is_empty() {
        return None;
    }
    Some(format!(
        "# Tools\n\n## functions\nnamespace functions {{\n{}\n}}\n",
        functions.join("\n")
    ))
}

fn encode_function(function: &Value) -> Option<String> {
    let parameters = function
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let mut registry = SchemaRegistry::default();
    let parsed = ObjectType::parse(&parameters, &mut registry);
    let mut interfaces = Vec::new();

    let root_name = if registry.has_self_ref {
        let body = parsed
            .properties
            .iter()
            .map(|parameter| parameter.to_typescript(INDENT, &registry))
            .collect::<Vec<_>>()
            .join(FIELD_DELIMITER);
        let body = if body.is_empty() {
            String::new()
        } else {
            format!("\n{body}\n")
        };
        interfaces.push(format!("interface parameters {{{body}}}"));
        Some("parameters")
    } else {
        None
    };

    let definitions = registry
        .order
        .iter()
        .filter_map(|name| {
            registry
                .definitions
                .get(name)
                .map(|schema| (name.clone(), schema.clone()))
        })
        .collect::<Vec<_>>();
    for (name, schema) in definitions {
        let object = parse_type(&schema, &mut registry);
        let mut definition = String::new();
        if let Some(description) = schema.get("description").and_then(Value::as_str)
            && !description.is_empty()
        {
            definition.push_str(&format_description(description, ""));
            definition.push('\n');
        }
        definition.push_str(&format!(
            "interface {name} {}",
            object.to_typescript("", &registry)
        ));
        interfaces.push(definition);
    }

    if registry.unsupported {
        return None;
    }
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("function");
    let type_definition = match root_name {
        Some(root_name) => format!("type {name} = (_: {root_name}) => any;"),
        None => format!(
            "type {name} = (_: {}) => any;",
            parsed.to_typescript("", &registry)
        ),
    };
    let description = function
        .get("description")
        .and_then(Value::as_str)
        .filter(|description| !description.is_empty())
        .map(|description| format_description(description, ""))
        .unwrap_or_default();
    Some(
        [interfaces.join("\n"), description, type_definition]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

#[derive(Default)]
struct SchemaRegistry {
    definitions: HashMap<String, Value>,
    order: Vec<String>,
    has_self_ref: bool,
    depth: usize,
    unsupported: bool,
}

impl SchemaRegistry {
    fn register_definitions(&mut self, definitions: &Value) {
        if let Some(definitions) = definitions.as_object() {
            for (name, schema) in definitions {
                if !self.definitions.contains_key(name) {
                    self.order.push(name.clone());
                }
                self.definitions.insert(name.clone(), schema.clone());
            }
        }
    }

    fn resolve_reference(&mut self, reference: &str) -> Option<Value> {
        if reference == "#" {
            self.has_self_ref = true;
            return Some(serde_json::json!({"$self_ref": true}));
        }
        if let Some(name) = reference.strip_prefix("#/$defs/")
            && let Some(definition) = self.definitions.get(name)
        {
            return Some(definition.clone());
        }
        self.unsupported = true;
        None
    }
}

enum ParameterType {
    Scalar(ScalarType),
    Object(ObjectType),
    Array(ArrayType),
    Enum(EnumType),
    AnyOf(AnyOfType),
    Union(UnionType),
    Reference(ReferenceType),
}

impl ParameterType {
    fn format_docstring(&self, indent: &str) -> String {
        match self {
            Self::Scalar(value) => value.base.format_docstring(indent),
            Self::Object(value) => value.base.format_docstring(indent),
            Self::Array(value) => value.base.format_docstring(indent),
            Self::Enum(value) => value.base.format_docstring(indent),
            Self::AnyOf(value) => value.base.format_docstring(indent),
            Self::Union(value) => value.base.format_docstring(indent),
            Self::Reference(value) => value.base.format_docstring(indent),
        }
    }

    fn to_typescript(&self, indent: &str, registry: &SchemaRegistry) -> String {
        match self {
            Self::Scalar(value) => value.to_typescript(),
            Self::Object(value) => value.to_typescript(indent, registry),
            Self::Array(value) => value.to_typescript(indent, registry),
            Self::Enum(value) => value.to_typescript(),
            Self::AnyOf(value) => value.to_typescript(indent, registry),
            Self::Union(value) => value.to_typescript(),
            Self::Reference(value) => value.to_typescript(),
        }
    }
}

#[derive(Default)]
struct BaseType {
    description: String,
    constraints: Vec<(String, Value)>,
}

impl BaseType {
    fn new(schema: &Value, allowed_constraints: &[&str]) -> Self {
        let description = schema
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let mut constraints = schema
            .as_object()
            .map(|object| {
                object
                    .iter()
                    .filter(|(key, _)| allowed_constraints.contains(&key.as_str()))
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        constraints.sort_by(|left, right| left.0.cmp(&right.0));
        Self {
            description,
            constraints,
        }
    }

    fn format_docstring(&self, indent: &str) -> String {
        let mut output = String::new();
        if !self.description.is_empty() {
            output.push_str(&format_description(&self.description, indent));
            output.push('\n');
        }
        if !self.constraints.is_empty() {
            let constraints = self
                .constraints
                .iter()
                .map(|(key, value)| format!("{key}: {}", json_inline(value)))
                .collect::<Vec<_>>()
                .join(", ");
            output.push_str(&format!("{indent}// {constraints}\n"));
        }
        output
    }
}

struct ScalarType {
    base: BaseType,
    kind: String,
}

impl ScalarType {
    fn parse(kind: &str, schema: &Value) -> Self {
        let constraints = match kind {
            "string" => &["maxLength", "minLength", "pattern"][..],
            "number" | "integer" => &["maximum", "minimum"][..],
            _ => &[],
        };
        Self {
            base: BaseType::new(schema, constraints),
            kind: kind.to_owned(),
        }
    }

    fn any() -> Self {
        Self {
            base: BaseType::default(),
            kind: "any".into(),
        }
    }

    fn to_typescript(&self) -> String {
        if self.kind == "integer" {
            "number".into()
        } else {
            self.kind.clone()
        }
    }
}

struct Parameter {
    name: String,
    kind: ParameterType,
    optional: bool,
    default: Option<Value>,
}

impl Parameter {
    fn to_typescript(&self, indent: &str, registry: &SchemaRegistry) -> String {
        let mut output = self.kind.format_docstring(indent);
        if let Some(default) = &self.default {
            let default = match default {
                Value::Bool(true) => "True".into(),
                Value::Bool(false) => "False".into(),
                Value::Number(_) => default.to_string(),
                _ => serde_json::to_string(default).unwrap_or_else(|_| "null".into()),
            };
            output.push_str(&format!("{indent}// Default: {default}\n"));
        }
        let optional = if self.optional { "?" } else { "" };
        let _ = write!(
            output,
            "{indent}{}{optional}: {}",
            self.name,
            self.kind.to_typescript(indent, registry)
        );
        output
    }
}

struct ObjectType {
    base: BaseType,
    properties: Vec<Parameter>,
    additional_properties: AdditionalProperties,
}

enum AdditionalProperties {
    None,
    True,
    False,
    Schema(Box<ParameterType>),
}

impl ObjectType {
    fn parse(schema: &Value, registry: &mut SchemaRegistry) -> Self {
        if let Some(definitions) = schema.get("$defs") {
            registry.register_definitions(definitions);
        }
        let additional_properties = match schema.get("additionalProperties") {
            None => AdditionalProperties::None,
            Some(Value::Bool(true)) => AdditionalProperties::True,
            Some(Value::Bool(false)) => AdditionalProperties::False,
            Some(schema) => AdditionalProperties::Schema(Box::new(parse_type(schema, registry))),
        };
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_str).collect::<Vec<_>>())
            .unwrap_or_default();
        let properties = schema
            .get("properties")
            .and_then(Value::as_object)
            .map(|properties| {
                properties
                    .iter()
                    .map(|(name, schema)| Parameter {
                        name: name.clone(),
                        kind: parse_type(schema, registry),
                        optional: !required.contains(&name.as_str()),
                        default: schema
                            .get("default")
                            .filter(|value| !value.is_null())
                            .cloned(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            base: BaseType::new(schema, &[]),
            properties,
            additional_properties,
        }
    }

    fn to_typescript(&self, indent: &str, registry: &SchemaRegistry) -> String {
        let mut required = self
            .properties
            .iter()
            .filter(|parameter| !parameter.optional)
            .collect::<Vec<_>>();
        let mut optional = self
            .properties
            .iter()
            .filter(|parameter| parameter.optional)
            .collect::<Vec<_>>();
        required.sort_by(|left, right| left.name.cmp(&right.name));
        optional.sort_by(|left, right| left.name.cmp(&right.name));
        let inner_indent = format!("{indent}{INDENT}");
        let mut fields = required
            .into_iter()
            .chain(optional)
            .map(|parameter| parameter.to_typescript(&inner_indent, registry))
            .collect::<Vec<_>>();
        match &self.additional_properties {
            AdditionalProperties::None => {}
            AdditionalProperties::True => fields.push(format!("{inner_indent}[k: string]: any")),
            AdditionalProperties::False => {
                fields.push(format!("{inner_indent}[k: string]: never"));
            }
            AdditionalProperties::Schema(schema) => fields.push(format!(
                "{inner_indent}[k: string]: {}",
                schema.to_typescript(&inner_indent, registry)
            )),
        }
        if fields.is_empty() {
            "{}".into()
        } else {
            format!("{{\n{}\n{indent}}}", fields.join(FIELD_DELIMITER))
        }
    }
}

struct ArrayType {
    base: BaseType,
    item: Box<ParameterType>,
}

impl ArrayType {
    fn parse(schema: &Value, registry: &mut SchemaRegistry) -> Self {
        let item = schema
            .get("items")
            .filter(|item| !item.is_null())
            .map(|item| parse_type(item, registry))
            .unwrap_or_else(|| ParameterType::Scalar(ScalarType::any()));
        Self {
            base: BaseType::new(schema, &["minItems", "maxItems"]),
            item: Box::new(item),
        }
    }

    fn to_typescript(&self, indent: &str, registry: &SchemaRegistry) -> String {
        let inner_indent = format!("{indent}{INDENT}");
        let docstring = self.item.format_docstring(&inner_indent);
        let item = self.item.to_typescript(&inner_indent, registry);
        if docstring.is_empty() {
            format!("Array<{item}>")
        } else {
            format!("Array<\n{docstring}{inner_indent}{item}\n{indent}>")
        }
    }
}

struct EnumType {
    base: BaseType,
    values: Vec<Value>,
}

impl EnumType {
    fn parse(schema: &Value) -> Self {
        Self {
            base: BaseType::new(schema, &[]),
            values: schema
                .get("enum")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
        }
    }

    fn to_typescript(&self) -> String {
        self.values
            .iter()
            .map(|value| match value {
                Value::String(value) => format!("\"{value}\""),
                Value::Null => "None".into(),
                Value::Bool(true) => "True".into(),
                Value::Bool(false) => "False".into(),
                value => value.to_string(),
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

struct AnyOfType {
    base: BaseType,
    branches: Vec<ParameterType>,
}

impl AnyOfType {
    fn parse(schema: &Value, registry: &mut SchemaRegistry) -> Self {
        Self {
            base: BaseType::new(schema, &[]),
            branches: schema
                .get("anyOf")
                .and_then(Value::as_array)
                .map(|branches| {
                    branches
                        .iter()
                        .map(|branch| parse_type(branch, registry))
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    fn to_typescript(&self, indent: &str, registry: &SchemaRegistry) -> String {
        self.branches
            .iter()
            .map(|branch| branch.to_typescript(indent, registry))
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

struct UnionType {
    base: BaseType,
    kinds: Vec<String>,
}

impl UnionType {
    fn parse(schema: &Value) -> Self {
        let kinds = schema
            .get("type")
            .and_then(Value::as_array)
            .map(|kinds| {
                kinds
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|kind| match kind {
                        "integer" => "number".into(),
                        "object" => "{}".into(),
                        "array" => "Array<any>".into(),
                        kind => kind.to_owned(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            base: BaseType::new(schema, &[]),
            kinds,
        }
    }

    fn to_typescript(&self) -> String {
        self.kinds.join(" | ")
    }
}

struct ReferenceType {
    base: BaseType,
    name: String,
}

impl ReferenceType {
    fn parse(schema: &Value, registry: &mut SchemaRegistry) -> Self {
        let reference = schema.get("$ref").and_then(Value::as_str).unwrap_or("");
        let resolved = registry.resolve_reference(reference);
        let name = match resolved {
            Some(value) if value.get("$self_ref").and_then(Value::as_bool) == Some(true) => {
                "parameters".into()
            }
            Some(_) => reference.rsplit('/').next().unwrap_or_default().into(),
            None => "any".into(),
        };
        Self {
            base: BaseType::new(schema, &[]),
            name,
        }
    }

    fn to_typescript(&self) -> String {
        self.name.clone()
    }
}

fn parse_type(schema: &Value, registry: &mut SchemaRegistry) -> ParameterType {
    if registry.depth >= MAX_RECURSION_DEPTH {
        return ParameterType::Scalar(ScalarType::any());
    }
    registry.depth += 1;
    let result = parse_type_inner(schema, registry);
    registry.depth -= 1;
    result
}

fn parse_type_inner(schema: &Value, registry: &mut SchemaRegistry) -> ParameterType {
    if let Some(schema) = schema.as_bool() {
        return ParameterType::Scalar(ScalarType {
            base: BaseType::default(),
            kind: if schema { "any" } else { "null" }.into(),
        });
    }
    let Some(object) = schema.as_object() else {
        registry.unsupported = true;
        return ParameterType::Scalar(ScalarType::any());
    };
    if object.contains_key("$ref") {
        return ParameterType::Reference(ReferenceType::parse(schema, registry));
    }
    if object.contains_key("anyOf") {
        return ParameterType::AnyOf(AnyOfType::parse(schema, registry));
    }
    if object.contains_key("enum") {
        return ParameterType::Enum(EnumType::parse(schema));
    }
    if let Some(kind) = object.get("type") {
        if kind.is_array() {
            return ParameterType::Union(UnionType::parse(schema));
        }
        if let Some(kind) = kind.as_str() {
            return match kind {
                "object" => ParameterType::Object(ObjectType::parse(schema, registry)),
                "array" => ParameterType::Array(ArrayType::parse(schema, registry)),
                kind => ParameterType::Scalar(ScalarType::parse(kind, schema)),
            };
        }
    }
    if object.is_empty() {
        return ParameterType::Scalar(ScalarType::any());
    }
    registry.unsupported = true;
    ParameterType::Scalar(ScalarType::any())
}

fn format_description(description: &str, indent: &str) -> String {
    description
        .split('\n')
        .map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!("{indent}// {line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn json_inline(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::Null => "null".into(),
        value => serde_json::to_string(value).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recursively_sorts_tool_schema() {
        let mut value = serde_json::json!({"z": [{"b": 1, "a": 2}], "a": 0});
        deep_sort(&mut value);
        assert_eq!(value.to_string(), r#"{"a":0,"z":[{"a":2,"b":1}]}"#);
    }

    #[test]
    fn encodes_complex_schema_byte_exactly() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Read weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "units": {"type": "string", "enum": ["c", "f"]},
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        }]);
        assert_eq!(
            encode_tools_to_typescript(tools.as_array().unwrap()).unwrap(),
            "# Tools\n\n## functions\nnamespace functions {\n// Read weather\ntype weather = (_: {\n  // City name\n  city: string,\n  units?: \"c\" | \"f\"\n}) => any;\n}\n"
        );
    }

    #[test]
    fn unsupported_schema_uses_json_fallback() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "broken",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"oneOf": [{"type": "string"}]}}
                }
            }
        }]);
        assert!(encode_tools_to_typescript(tools.as_array().unwrap()).is_none());
    }

    #[test]
    fn null_default_is_omitted_like_checkpoint_python() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "optional_value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": ["string", "null"], "default": null}
                    }
                }
            }
        }]);
        let encoded = encode_tools_to_typescript(tools.as_array().unwrap()).unwrap();
        assert_eq!(
            encoded,
            "# Tools\n\n## functions\nnamespace functions {\ntype optional_value = (_: {\n  value?: string | null\n}) => any;\n}\n"
        );
        assert!(!encoded.contains("Default"));
    }
}
