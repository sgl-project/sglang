//! Typed multimodal inputs of the `/generate` body — the Rust form of Python
//! `MultimodalDataInputFormat` (`io_struct.py`) — and their per-request fan-out.

use std::fmt;

use serde::de::value::{MapAccessDeserializer, SeqAccessDeserializer};
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use super::request::{HeapBytes, check_broadcast_budget};
use super::types::{OneOrMany, OneOrManyItem};
use crate::utils::error::Error;

/// Request-scoped options exposed by Python's multimodal input schema.
/// Each processor applies the hints supported by its preprocessing configuration.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
pub struct MmProcessorOptions {
    #[serde(default)]
    pub use_audio_in_video: bool,
    pub video_config: Option<serde_json::Map<String, serde_json::Value>>,
    pub modalities: Option<Vec<String>>,
    pub max_dynamic_patch: Option<i64>,
    pub min_dynamic_patch: Option<i64>,
    pub image_max_dynamic_patch: Option<i64>,
    pub video_max_dynamic_patch: Option<i64>,
    pub images_config: Option<serde_json::Map<String, serde_json::Value>>,
}

impl MmProcessorOptions {
    pub(super) fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// One media item: Python `MultimodalDataInputItem` as it can arrive over JSON.
/// `bytes` and PIL images exist only on the in-process Engine path, so they have
/// no variant here.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MmItem {
    /// URL, `file://` / absolute path, `data:` URI, or bare base64 (Python `str`).
    Source(String),
    /// Python `ImageData` / `VideoData` (`{"url": …, …}`). Content identity is
    /// distinct from the caller's processor-feature hash. The hint keys
    /// (`detail`, `max_dynamic_patch`, `preprocess_kwargs`, ...)
    /// are read by model families this pipeline does not run, and Python's
    /// `load_image` itself reduces the item to `.url`.
    Ref {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content_hash: Option<String>,
    },
    /// A preprocessed item (`{"format": "processor_output" | "precomputed_embedding", …}`).
    /// Parsed only far enough to be rejected by name at the MM stage; Python
    /// ignores it the same way on a text-only model.
    Preprocessed { format: String },
}

impl MmItem {
    /// The raw source string for the modality pipeline, `None` for a
    /// preprocessed item.
    pub fn source(&self) -> Option<&str> {
        match self {
            MmItem::Source(source) | MmItem::Ref { url: source, .. } => Some(source),
            MmItem::Preprocessed { .. } => None,
        }
    }
}

impl HeapBytes for MmItem {
    fn heap_bytes(&self) -> usize {
        match self {
            MmItem::Source(s) | MmItem::Preprocessed { format: s } => s.len(),
            MmItem::Ref { url, content_hash } => {
                url.len() + content_hash.as_ref().map_or(0, String::len)
            }
        }
    }
}

/// The object form of an item, as Python's `Dict[str, Any]`: `format` marks a
/// preprocessed item (checked first, as `glm4v` does), `url` an `ImageData`.
#[derive(Deserialize)]
struct ItemObject {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    content_hash: Option<String>,
}

impl TryFrom<ItemObject> for MmItem {
    type Error = &'static str;

    fn try_from(object: ItemObject) -> Result<Self, Self::Error> {
        match (object.format, object.url) {
            (Some(format), _) => Ok(MmItem::Preprocessed { format }),
            (None, Some(url)) => Ok(MmItem::Ref {
                url,
                content_hash: object.content_hash,
            }),
            (None, None) => Err("a multimodal item object needs a `url` or a `format` key"),
        }
    }
}

/// Align each hash column with Python's normalized image lists. A batch can
/// mix a scalar for a one-image request with a list for a multi-image request.
/// Single-request feature hashes retain the worker's warning/fallback policy
/// for a count mismatch; content-hash alignment is checked before processing.
pub(super) fn fan_out_hashes<T: OneOrManyItem + Clone>(
    hashes: Option<Vec<OneOrMany<T>>>,
    images: &[Vec<MmItem>],
    is_batch: bool,
    field: &str,
) -> Result<Vec<Option<Vec<T>>>, Error> {
    let Some(hashes) = hashes else {
        return Ok(vec![None; images.len()]);
    };
    if !is_batch {
        let values = hashes
            .into_iter()
            .map(|hash| match hash {
                OneOrMany::One(value) => Ok(value),
                OneOrMany::Many(_) => Err(Error::Validation(format!(
                    "{field} must be a flat list for a single request"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(vec![Some(values)]);
    }
    if hashes.len() != images.len() {
        return Err(Error::Validation(format!(
            "The length of {field} should equal the batch size"
        )));
    }
    hashes
        .into_iter()
        .zip(images)
        .enumerate()
        .map(|(index, (hashes, images))| {
            let values = match hashes {
                OneOrMany::One(value) if images.len() == 1 => vec![value],
                OneOrMany::One(_) => {
                    return Err(Error::Validation(format!(
                        "{field}[{index}] must be a list with one entry per image"
                    )));
                }
                OneOrMany::Many(values) => values,
            };
            if values.len() != images.len() {
                return Err(Error::Validation(format!(
                    "{field}[{index}] has {} entries for {} images",
                    values.len(),
                    images.len()
                )));
            }
            Ok(Some(values))
        })
        .collect()
}

/// Python TokenizerManager merges native and inline OpenAI identities before
/// calling a processor. These identities never replace the feature hash.
pub(crate) fn normalize_content_hashes(
    images: &[MmItem],
    explicit: Option<Vec<Option<String>>>,
) -> Result<Option<Vec<Option<String>>>, String> {
    let has_inline = images.iter().any(|image| {
        matches!(
            image,
            MmItem::Ref {
                content_hash: Some(value),
                ..
            } if !value.is_empty()
        )
    });
    if explicit.is_none() && !has_inline {
        return Ok(None);
    }
    if let Some(hashes) = &explicit
        && hashes.len() != images.len()
    {
        return Err(format!(
            "mm_content_hashes has {} entries for {} images",
            hashes.len(),
            images.len()
        ));
    }
    images
        .iter()
        .enumerate()
        .map(|(index, image)| {
            let embedded = match image {
                MmItem::Ref { content_hash, .. } => content_hash.as_deref(),
                _ => None,
            };
            let embedded = parse_content_hash(embedded)?;
            let provided = parse_content_hash(
                explicit
                    .as_ref()
                    .and_then(|hashes| hashes[index].as_deref()),
            )?;
            if provided.is_some() && embedded.is_some() && provided != embedded {
                return Err(format!(
                    "Conflicting content hashes for image_data[{index}]"
                ));
            }
            Ok(provided.or(embedded))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn parse_content_hash(value: Option<&str>) -> Result<Option<String>, String> {
    let Some(value) = value else { return Ok(None) };
    let digest = value
        .strip_prefix("sha256:")
        .ok_or("content_hash must use the form 'sha256:<64 hex digits>'")?;
    if digest.len() != 64 {
        return Err("content_hash must contain exactly 64 SHA-256 hex digits".into());
    }
    if !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("content_hash contains non-hexadecimal characters".into());
    }
    Ok(Some(format!("sha256:{}", digest.to_ascii_lowercase())))
}

/// Hand-written rather than `#[serde(untagged)]` so a bad item is reported as
/// what it is ("expected a source string or an item object"), not as "did not
/// match any variant".
impl<'de> Deserialize<'de> for MmItem {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ItemVisitor;

        impl<'de> Visitor<'de> for ItemVisitor {
            type Value = MmItem;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a media source string or a multimodal item object")
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(MmItem::Source(value.to_owned()))
            }

            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(MmItem::Source(value))
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                ItemObject::deserialize(MapAccessDeserializer::new(map))?
                    .try_into()
                    .map_err(serde::de::Error::custom)
            }
        }

        deserializer.deserialize_any(ItemVisitor)
    }
}

/// One `image_data` / `video_data` / `audio_data` field as sent: Python
/// `MultimodalDataInputFormat`, whose three shapes read differently for a single
/// request and a batch (see [`fan_out`]).
#[derive(Debug, Clone, PartialEq)]
pub enum MmDataInput {
    /// One item: a single request's whole input, or a broadcast to every batch entry.
    One(MmItem),
    /// A flat list: a single request's items, or one item per batch entry.
    Many(Vec<Option<MmItem>>),
    /// One item list per batch entry.
    Nested(Vec<Option<Vec<Option<MmItem>>>>),
}

/// One element of a list-form field, before the list is known to be flat or nested.
enum ListElement {
    Null,
    Item(MmItem),
    List(Vec<Option<MmItem>>),
}

impl<'de> Deserialize<'de> for ListElement {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ElementVisitor;

        impl<'de> Visitor<'de> for ElementVisitor {
            type Value = ListElement;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("null, a media source string, an item object, or a list of items")
            }

            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(ListElement::Null)
            }

            fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(ListElement::Null)
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(ListElement::Item(MmItem::Source(value.to_owned())))
            }

            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(ListElement::Item(MmItem::Source(value)))
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                ItemObject::deserialize(MapAccessDeserializer::new(map))?
                    .try_into()
                    .map(ListElement::Item)
                    .map_err(serde::de::Error::custom)
            }

            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
                Vec::<Option<MmItem>>::deserialize(SeqAccessDeserializer::new(seq))
                    .map(ListElement::List)
            }
        }

        deserializer.deserialize_any(ElementVisitor)
    }
}

impl<'de> Deserialize<'de> for MmDataInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct InputVisitor;

        impl<'de> Visitor<'de> for InputVisitor {
            type Value = MmDataInput;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a media item, a list of items, or a list of item lists")
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(MmDataInput::One(MmItem::Source(value.to_owned())))
            }

            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(MmDataInput::One(MmItem::Source(value)))
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                ItemObject::deserialize(MapAccessDeserializer::new(map))?
                    .try_into()
                    .map(MmDataInput::One)
                    .map_err(serde::de::Error::custom)
            }

            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
                let elements = Vec::<ListElement>::deserialize(SeqAccessDeserializer::new(seq))?;
                let nested = elements
                    .iter()
                    .any(|element| matches!(element, ListElement::List(_)));
                if !nested {
                    return Ok(MmDataInput::Many(
                        elements
                            .into_iter()
                            .map(|element| match element {
                                ListElement::Item(item) => Some(item),
                                ListElement::Null => None,
                                ListElement::List(_) => unreachable!("checked above"),
                            })
                            .collect(),
                    ));
                }
                elements
                    .into_iter()
                    .map(|element| match element {
                        ListElement::List(items) => Ok(Some(items)),
                        ListElement::Null => Ok(None),
                        ListElement::Item(_) => Err(serde::de::Error::custom(
                            "a nested list cannot mix bare items with item lists",
                        )),
                    })
                    .collect::<Result<_, _>>()
                    .map(MmDataInput::Nested)
            }
        }

        deserializer.deserialize_any(InputVisitor)
    }
}

/// The items of one modality for one request, `null` entries dropped.
fn present(items: Vec<Option<MmItem>>) -> Vec<MmItem> {
    items.into_iter().flatten().collect()
}

/// Fan one field into per-request item lists (empty = no input for that
/// request), mirroring Python `_normalize_{image,video,audio}_data`:
///   * absent, `[]`, or all-`null` → no input (Python `has_valid_data`);
///   * single request → one item or a flat list, taken as is;
///   * batch + one item → broadcast to every entry;
///   * batch + list → per entry, length must equal the batch size.
///
/// The Python image path wraps a broadcast as `[[img]] * num` while video and
/// audio broadcast bare; the difference vanishes here because every request's
/// input is already an item list.
pub fn fan_out(
    value: Option<MmDataInput>,
    n: usize,
    is_batch: bool,
    name: &str,
) -> Result<Vec<Vec<MmItem>>, Error> {
    let Some(value) = value else {
        return Ok(vec![Vec::new(); n]);
    };
    if !is_batch {
        return match value {
            MmDataInput::One(item) => Ok(vec![vec![item]]),
            MmDataInput::Many(items) => Ok(vec![present(items)]),
            MmDataInput::Nested(_) => Err(Error::Validation(format!(
                "{name}: a nested list is the batch form; a single request takes one item or a flat list"
            ))),
        };
    }
    match value {
        MmDataInput::One(item) => {
            // A broadcast deep-clones once per prompt — same blow-up as
            // sampling_params, so bound the product before any clone.
            check_broadcast_budget(item.heap_bytes(), n, name)?;
            Ok(vec![vec![item]; n])
        }
        MmDataInput::Many(items) if items.is_empty() => Ok(vec![Vec::new(); n]),
        MmDataInput::Many(items) => {
            check_len(items.len(), n, name)?;
            Ok(items
                .into_iter()
                .map(|item| item.into_iter().collect())
                .collect())
        }
        MmDataInput::Nested(lists) => {
            check_len(lists.len(), n, name)?;
            Ok(lists
                .into_iter()
                .map(|items| items.map(present).unwrap_or_default())
                .collect())
        }
    }
}

fn check_len(len: usize, n: usize, name: &str) -> Result<(), Error> {
    if len != n {
        return Err(Error::Validation(format!(
            "{name}: list length {len} does not match batch size {n}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processor_options_validate_python_types_and_reach_forwarded_media_work() {
        use crate::message::request::GenerateBody;
        use crate::multi_modality::payload::resolve_media_work;

        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../testdata/mm_processor_options_python.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let body: GenerateBody = serde_json::from_value(case["body"].clone()).unwrap();
            let options = body.processor_options.clone();
            let (requests, _) = body.into_requests().unwrap();
            assert_eq!(requests.len(), case["requests"].as_u64().unwrap() as usize);
            for request in requests {
                let forwarded: GenerateBody =
                    serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
                let (forwarded, _) = forwarded.into_requests().unwrap();
                assert_eq!(forwarded.len(), 1);
                for mut request in std::iter::once(request).chain(forwarded) {
                    assert!(request.has_multimodal());
                    let resolved = resolve_media_work(request.take_mm_work()).unwrap();
                    assert_eq!(resolved.processor_options, options);
                    assert_eq!(resolved.images.len() + resolved.videos.len(), 1);
                }
            }
        }
        for body in fixture["invalid"].as_array().unwrap() {
            assert!(
                serde_json::from_value::<GenerateBody>(body.clone()).is_err(),
                "{body}"
            );
        }
        let body: GenerateBody = serde_json::from_value(serde_json::json!({
            "input_ids": [1], "max_dynamic_patch": 2, "use_audio_in_video": true
        }))
        .unwrap();
        let (requests, _) = body.into_requests().unwrap();
        assert!(
            !requests[0].has_multimodal(),
            "hints alone must not trigger media processing"
        );
    }

    fn parse(json: &str) -> Result<MmDataInput, serde_json::Error> {
        serde_json::from_str(json)
    }

    fn src(s: &str) -> MmItem {
        MmItem::Source(s.to_owned())
    }

    /// The three Python shapes parse to their own variants, with `null`
    /// entries kept in place so batch fan-out can index them.
    #[test]
    fn parses_python_shapes() {
        assert_eq!(parse(r#""u""#).unwrap(), MmDataInput::One(src("u")));
        assert_eq!(
            parse(r#"["a", null, "b"]"#).unwrap(),
            MmDataInput::Many(vec![Some(src("a")), None, Some(src("b"))])
        );
        assert_eq!(parse("[]").unwrap(), MmDataInput::Many(vec![]));
        assert_eq!(
            parse(r#"[["a", null], null, []]"#).unwrap(),
            MmDataInput::Nested(vec![Some(vec![Some(src("a")), None]), None, Some(vec![])])
        );
    }

    /// Object items: `format` wins over `url` (a preprocessed item may carry
    /// both), and an object with neither is named in the error.
    #[test]
    fn parses_item_objects() {
        assert_eq!(
            parse(r#"{"url": "u", "detail": "high"}"#).unwrap(),
            MmDataInput::One(MmItem::Ref {
                url: "u".into(),
                content_hash: None
            })
        );
        assert_eq!(
            parse(r#"[{"format": "processor_output", "url": "u", "pixel_values": [1]}]"#).unwrap(),
            MmDataInput::Many(vec![Some(MmItem::Preprocessed {
                format: "processor_output".into()
            })])
        );
        let err = parse(r#"{"detail": "high"}"#).unwrap_err().to_string();
        assert!(err.contains("`url` or a `format`"), "{err}");
    }

    /// Anything Python's item union does not cover is rejected up front, with
    /// the expected shape in the message.
    #[test]
    fn rejects_non_items() {
        for (json, expect) in [
            ("5", "expected a media item, a list of items"),
            (r#"["a", 5]"#, "expected null, a media source string"),
            (r#"["a", ["b"]]"#, "cannot mix"),
            (
                r#"[[["a"]]]"#,
                "expected a media source string or a multimodal item object",
            ),
        ] {
            let err = parse(json).unwrap_err().to_string();
            assert!(err.contains(expect), "{json}: {err}");
        }
    }

    #[test]
    fn single_request_takes_item_or_flat_list() {
        assert_eq!(fan_out(None, 1, false, "image_data").unwrap(), vec![vec![]]);
        assert_eq!(
            fan_out(Some(MmDataInput::One(src("u"))), 1, false, "image_data").unwrap(),
            vec![vec![src("u")]]
        );
        assert_eq!(
            fan_out(
                Some(parse(r#"["a", null, "b"]"#).unwrap()),
                1,
                false,
                "image_data"
            )
            .unwrap(),
            vec![vec![src("a"), src("b")]]
        );
        assert_eq!(
            fan_out(Some(parse("[null]").unwrap()), 1, false, "image_data").unwrap(),
            vec![vec![]]
        );
        let err = fan_out(Some(parse(r#"[["a"]]"#).unwrap()), 1, false, "image_data").unwrap_err();
        assert!(err.to_string().contains("batch form"), "{err}");
    }

    #[test]
    fn batch_broadcasts_scalar_and_splits_lists() {
        let one = fan_out(Some(MmDataInput::One(src("u"))), 2, true, "video_data").unwrap();
        assert_eq!(one, vec![vec![src("u")], vec![src("u")]]);

        let flat = fan_out(
            Some(parse(r#"["a", null]"#).unwrap()),
            2,
            true,
            "image_data",
        )
        .unwrap();
        assert_eq!(flat, vec![vec![src("a")], vec![]]);

        let nested = fan_out(
            Some(parse(r#"[["a", "b"], null, [null]]"#).unwrap()),
            3,
            true,
            "image_data",
        )
        .unwrap();
        assert_eq!(nested, vec![vec![src("a"), src("b")], vec![], vec![]]);

        // `[]` is "no input", not a length-0 per-entry list.
        assert_eq!(
            fan_out(Some(parse("[]").unwrap()), 2, true, "image_data").unwrap(),
            vec![vec![], vec![]]
        );
        for json in [r#"["a"]"#, r#"[["a"]]"#] {
            let err = fan_out(Some(parse(json).unwrap()), 2, true, "image_data").unwrap_err();
            assert!(
                err.to_string().contains("does not match batch size"),
                "{json}: {err}"
            );
        }
    }
}
