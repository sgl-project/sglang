//! Pass 2's input model: each message classified by the `sglang.json.v1`
//! options into the JSON shape the emitter renders. The options are read from
//! the FileDescriptorSet through prost-reflect, so the schema file is the
//! single source of truth.

use prost_reflect::{
    DescriptorPool, ExtensionDescriptor, FieldDescriptor, Kind, MessageDescriptor,
};

pub struct Extensions {
    message_json: ExtensionDescriptor,
    field_json: ExtensionDescriptor,
    oneof_json: ExtensionDescriptor,
}

impl Extensions {
    pub fn resolve(pool: &DescriptorPool) -> Extensions {
        let get = |name: &str| {
            pool.get_extension_by_name(name)
                .unwrap_or_else(|| panic!("options.proto extension missing: {name}"))
        };
        Extensions {
            message_json: get("sglang.json.v1.message_json"),
            field_json: get("sglang.json.v1.field_json"),
            oneof_json: get("sglang.json.v1.oneof_json"),
        }
    }
}

fn opt_bool(options: &prost_reflect::DynamicMessage, name: &str) -> bool {
    options
        .get_field_by_name(name)
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

fn opt_str(options: &prost_reflect::DynamicMessage, name: &str) -> Option<String> {
    options
        .get_field_by_name(name)
        .and_then(|v| v.as_str().map(str::to_owned))
        .filter(|s| !s.is_empty())
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarTy {
    F64,
    I32,
    I64,
    U32,
    U64,
    Bool,
    Str,
}

impl ScalarTy {
    pub fn rust(&self) -> &'static str {
        match self {
            ScalarTy::F64 => "f64",
            ScalarTy::I32 => "i32",
            ScalarTy::I64 => "i64",
            ScalarTy::U32 => "u32",
            ScalarTy::U64 => "u64",
            ScalarTy::Bool => "bool",
            ScalarTy::Str => "::prost::alloc::string::String",
        }
    }
}

#[derive(Debug, Clone)]
pub enum DefaultLit {
    F64(f64),
    I64(i64),
    Bool(bool),
    Str(String),
}

impl DefaultLit {
    pub fn rust(&self) -> String {
        match self {
            DefaultLit::F64(v) => format!("{v:?}f64"),
            DefaultLit::I64(v) => format!("{v}i64"),
            DefaultLit::Bool(v) => format!("{v}"),
            DefaultLit::Str(v) => format!("{v:?}.to_string()"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FieldKind {
    /// A plain proto3 scalar (no presence). `null_default` set =
    /// null_resets_default: absent AND explicit null yield the default.
    Scalar {
        ty: ScalarTy,
        null_default: Option<DefaultLit>,
    },
    /// proto3 `optional` scalar. `null_is_none_default` set: absent yields the
    /// default, explicit null clears to None. `null_default` set
    /// (null_resets_default): absent AND null yield Some(default), and an
    /// `*_or_default` accessor covers protobuf arrival (unset = None).
    OptScalar {
        ty: ScalarTy,
        emit_null: bool,
        null_is_none_default: Option<DefaultLit>,
        null_default: Option<DefaultLit>,
    },
    /// A message-typed field: `Option<T>` presence; the value's JSON shape is
    /// the message's own impl.
    Message {
        rust_path: String,
        emit_null: bool,
    },
    RepeatedScalar {
        ty: ScalarTy,
    },
    RepeatedMessage {
        rust_path: String,
    },
    /// repeated string with raw_json: Vec<String> of JSON texts passing
    /// through as an array of raw values.
    RepeatedRawJson,
    /// map<string, scalar>; int_key means numeric-string keys (no transform
    /// needed on the JSON side, documented parity).
    Map {
        value: ScalarTy,
    },
    /// A string field holding JSON text (raw_json): the value passes through
    /// verbatim on the HTTP side.
    RawJson {
        emit_null: bool,
    },
}

#[derive(Debug, Clone)]
pub struct FieldModel {
    /// The JSON key (= the proto field name).
    pub json_name: String,
    /// The prost struct field ident (keyword-escaped).
    pub rust_ident: String,
    pub kind: FieldKind,
}

#[derive(Debug, Clone)]
pub struct VariantModel {
    /// The JSON discriminator value / shape source (= the proto field name).
    pub json_name: String,
    /// The prost oneof enum variant (PascalCase).
    pub rust_variant: String,
    pub kind: FieldKind,
    /// For tagged unions: the payload message's fields, inlined into the
    /// tagged object.
    pub payload_fields: Vec<FieldModel>,
}

#[derive(Debug)]
pub enum MessageKind {
    /// Single-repeated-field carrier: JSON is the bare array.
    BareList { item: FieldModel },
    /// Positional tuple: fields in declaration order form a bare array.
    Tuple { fields: Vec<FieldModel> },
    /// Single-field wrapper serializing as its sole field's value.
    Transparent { field: FieldModel },
    /// Single oneof { one, many }: scalar-or-list flattening.
    OneOrMany {
        oneof_mod: String,
        one: VariantModel,
        many: VariantModel,
    },
    /// Single oneof, internally tagged by `tag`; `passthrough` catches
    /// unknown tags as a raw map (and any known-tag payload that fails to
    /// parse, matching the hand-written FinishReason).
    TaggedUnion {
        oneof_mod: String,
        oneof_ident: String,
        tag: String,
        variants: Vec<VariantModel>,
        passthrough: Option<VariantModel>,
    },
    /// Single oneof, untagged: the value's shape picks the variant.
    Untagged {
        oneof_mod: String,
        oneof_ident: String,
        variants: Vec<VariantModel>,
    },
    /// A plain object.
    Struct {
        deny_unknown: bool,
        fields: Vec<FieldModel>,
    },
    /// gRPC-only (a plain oneof container like GenerateStreamItem): no HTTP
    /// JSON contract, no serde emitted.
    Skip,
}

#[derive(Debug)]
pub struct MessageModel {
    /// Rust type name (= proto message name).
    pub rust_name: String,
    pub kind: MessageKind,
}

pub fn build_models(pool: &DescriptorPool, package: &str) -> Vec<MessageModel> {
    let ext = Extensions::resolve(pool);
    pool.all_messages()
        .filter(|m| m.full_name().starts_with(package))
        // Map-entry synthetic messages have no JSON identity of their own.
        .filter(|m| !m.is_map_entry())
        .map(|m| message_model(&m, &ext))
        .collect()
}

fn message_model(msg: &MessageDescriptor, ext: &Extensions) -> MessageModel {
    let rust_name = msg.name().to_string();
    let options = msg.options();
    let mopts = options.get_extension(&ext.message_json);
    let mopts = mopts.as_message();

    let bare_list = mopts.map(|m| opt_bool(m, "bare_list")).unwrap_or(false);
    let tuple = mopts.map(|m| opt_bool(m, "tuple")).unwrap_or(false);
    let transparent = mopts.map(|m| opt_bool(m, "transparent")).unwrap_or(false);
    let deny_unknown = mopts
        .and_then(|m| m.get_field_by_name("unknown_fields"))
        .and_then(|v| v.as_enum_number())
        .unwrap_or(0)
        == 1;

    // proto3 `optional` is implemented as a synthetic oneof: those fields are
    // plain fields with presence, not union members.
    let oneofs: Vec<_> = msg.oneofs().filter(|o| !o.is_synthetic()).collect();
    let plain_fields: Vec<FieldDescriptor> = msg
        .fields()
        .filter(|f| f.containing_oneof().is_none_or(|o| o.is_synthetic()))
        .collect();

    if bare_list {
        assert_eq!(
            plain_fields.len(),
            1,
            "{}: bare_list needs exactly one repeated field",
            msg.full_name()
        );
        return MessageModel {
            rust_name,
            kind: MessageKind::BareList {
                item: field_model(&plain_fields[0], ext),
            },
        };
    }
    if transparent {
        assert_eq!(
            plain_fields.len(),
            1,
            "{}: transparent needs exactly one field",
            msg.full_name()
        );
        return MessageModel {
            rust_name,
            kind: MessageKind::Transparent {
                field: field_model(&plain_fields[0], ext),
            },
        };
    }
    if tuple {
        return MessageModel {
            rust_name,
            kind: MessageKind::Tuple {
                fields: plain_fields.iter().map(|f| field_model(f, ext)).collect(),
            },
        };
    }
    if oneofs.len() == 1 && plain_fields.is_empty() {
        let oneof = &oneofs[0];
        let oopts = oneof.options();
        let oopts = oopts.get_extension(&ext.oneof_json);
        let oopts = oopts.as_message();
        let one_or_many = oopts.map(|m| opt_bool(m, "one_or_many")).unwrap_or(false);
        let untagged = oopts.map(|m| opt_bool(m, "untagged")).unwrap_or(false);
        let tag = oopts.and_then(|m| opt_str(m, "tag"));
        let passthrough_on = oopts
            .map(|m| opt_bool(m, "unknown_variant_passthrough"))
            .unwrap_or(false);
        let oneof_mod = snake(&rust_name);
        let oneof_ident = pascal(oneof.name());

        let mut variants: Vec<VariantModel> = oneof
            .fields()
            .map(|f| VariantModel {
                json_name: f.name().to_string(),
                rust_variant: pascal(f.name()),
                kind: field_model(&f, ext).kind,
                payload_fields: match f.kind() {
                    // The raw_json passthrough arm carries a well-known type;
                    // its innards are never inlined.
                    Kind::Message(payload)
                        if tag.is_some() && !payload.full_name().starts_with("google.protobuf") =>
                    {
                        payload
                            .fields()
                            .map(|pf| field_model(&pf, ext))
                            .collect::<Vec<_>>()
                    }
                    _ => vec![],
                },
            })
            .collect();

        if one_or_many {
            assert_eq!(variants.len(), 2, "{}: one_or_many", msg.full_name());
            let many = variants.pop().expect("many variant");
            let one = variants.pop().expect("one variant");
            return MessageModel {
                rust_name,
                kind: MessageKind::OneOrMany {
                    oneof_mod,
                    one,
                    many,
                },
            };
        }
        if let Some(tag) = tag {
            let passthrough = passthrough_on.then(|| {
                let idx = variants
                    .iter()
                    .position(|v| matches!(v.kind, FieldKind::RawJson { .. }))
                    .expect("passthrough needs a raw_json string variant");
                variants.remove(idx)
            });
            return MessageModel {
                rust_name,
                kind: MessageKind::TaggedUnion {
                    oneof_mod,
                    oneof_ident,
                    tag,
                    variants,
                    passthrough,
                },
            };
        }
        if untagged {
            return MessageModel {
                rust_name,
                kind: MessageKind::Untagged {
                    oneof_mod,
                    oneof_ident,
                    variants,
                },
            };
        }
        // A plain oneof container (e.g. GenerateStreamItem) has no HTTP JSON
        // contract of its own — it is gRPC-only. Emit nothing.
        return MessageModel {
            rust_name,
            kind: MessageKind::Skip,
        };
    }

    MessageModel {
        rust_name,
        kind: MessageKind::Struct {
            deny_unknown,
            fields: plain_fields.iter().map(|f| field_model(f, ext)).collect(),
        },
    }
}

fn field_model(f: &FieldDescriptor, ext: &Extensions) -> FieldModel {
    let options = f.options();
    let fopts = options.get_extension(&ext.field_json);
    let fopts = fopts.as_message();
    let get_bool = |name: &str| fopts.map(|m| opt_bool(m, name)).unwrap_or(false);
    let default_lit = |m: &prost_reflect::DynamicMessage| -> Option<DefaultLit> {
        // has_field_by_name: the default_* members are proto3 optional, so
        // presence is meaningful.
        if m.has_field_by_name("default_f64") {
            return m
                .get_field_by_name("default_f64")
                .and_then(|v| v.as_f64())
                .map(DefaultLit::F64);
        }
        if m.has_field_by_name("default_i64") {
            return m
                .get_field_by_name("default_i64")
                .and_then(|v| v.as_i64())
                .map(DefaultLit::I64);
        }
        if m.has_field_by_name("default_bool") {
            return m
                .get_field_by_name("default_bool")
                .and_then(|v| v.as_bool())
                .map(DefaultLit::Bool);
        }
        if m.has_field_by_name("default_string") {
            return m
                .get_field_by_name("default_string")
                .and_then(|v| v.as_str().map(str::to_owned))
                .map(DefaultLit::Str);
        }
        None
    };
    let raw_json = get_bool("raw_json");
    let emit_null = get_bool("emit_null_when_absent");
    let null_resets = get_bool("null_resets_default");
    let null_is_none = get_bool("null_is_none");
    let default = fopts.and_then(default_lit);

    let kind = if raw_json {
        if f.is_list() {
            FieldKind::RepeatedRawJson
        } else {
            assert!(
                matches!(f.kind(), Kind::String),
                "{}: raw_json requires a string field",
                f.full_name()
            );
            FieldKind::RawJson { emit_null }
        }
    } else if f.is_map() {
        let value_ty = match f.kind() {
            Kind::Message(entry) => scalar_ty(
                &entry
                    .get_field_by_name("value")
                    .expect("map value field")
                    .kind(),
            ),
            _ => unreachable!("map fields are message-kinded"),
        };
        FieldKind::Map { value: value_ty }
    } else if f.is_list() {
        match f.kind() {
            Kind::Message(m) => FieldKind::RepeatedMessage {
                rust_path: m.name().to_string(),
            },
            other => FieldKind::RepeatedScalar {
                ty: scalar_ty(&other),
            },
        }
    } else {
        match f.kind() {
            Kind::Message(m) => FieldKind::Message {
                rust_path: m.name().to_string(),
                emit_null,
            },
            other => {
                let ty = scalar_ty(&other);
                if f.supports_presence() {
                    FieldKind::OptScalar {
                        ty,
                        emit_null,
                        null_is_none_default: null_is_none
                            .then(|| default.clone().expect("null_is_none requires a default_*")),
                        null_default: null_resets.then(|| {
                            default
                                .clone()
                                .expect("null_resets_default requires a default_*")
                        }),
                    }
                } else {
                    FieldKind::Scalar {
                        ty,
                        null_default: null_resets.then(|| {
                            default
                                .clone()
                                .expect("null_resets_default requires a default_*")
                        }),
                    }
                }
            }
        }
    };

    FieldModel {
        json_name: f.name().to_string(),
        rust_ident: escape_ident(f.name()),
        kind,
    }
}

fn scalar_ty(kind: &Kind) -> ScalarTy {
    match kind {
        Kind::Double | Kind::Float => ScalarTy::F64,
        Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 => ScalarTy::I64,
        Kind::Int32 | Kind::Sint32 | Kind::Sfixed32 => ScalarTy::I32,
        Kind::Uint32 | Kind::Fixed32 => ScalarTy::U32,
        Kind::Uint64 | Kind::Fixed64 => ScalarTy::U64,
        Kind::Bool => ScalarTy::Bool,
        Kind::String => ScalarTy::Str,
        other => panic!("unsupported scalar kind {other:?}"),
    }
}

/// prost's keyword escaping for field idents.
fn escape_ident(name: &str) -> String {
    const KEYWORDS: &[&str] = &[
        "as", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern", "false",
        "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub",
        "ref", "return", "self", "static", "struct", "super", "trait", "true", "type", "unsafe",
        "use", "where", "while",
    ];
    if KEYWORDS.contains(&name) {
        format!("r#{name}")
    } else {
        name.to_string()
    }
}

pub fn pascal(name: &str) -> String {
    name.split('_')
        .filter(|s| !s.is_empty())
        .map(|s| {
            let mut chars = s.chars();
            match chars.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

pub fn snake(name: &str) -> String {
    let mut out = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}
