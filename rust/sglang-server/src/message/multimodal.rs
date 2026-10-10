//! Typed multimodal inputs of the `/generate` body — the Rust form of Python
//! `MultimodalDataInputFormat` (`io_struct.py`) — and their per-request fan-out.

use std::fmt;

use serde::de::value::{MapAccessDeserializer, SeqAccessDeserializer};
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

use super::request::{HeapBytes, check_broadcast_budget};
use crate::utils::error::Error;

/// One media item: Python `MultimodalDataInputItem` as it can arrive over JSON.
/// `bytes` and PIL images exist only on the in-process Engine path, so they have
/// no variant here.
#[derive(Debug, Clone, PartialEq)]
pub enum MmItem {
    /// URL, `file://` / absolute path, `data:` URI, or bare base64 (Python `str`).
    Source(String),
    /// Python `ImageData` / `VideoData` (`{"url": …, …}`). Only `url` is kept:
    /// the hint keys (`detail`, `max_dynamic_patch`, `preprocess_kwargs`, ...)
    /// are read by model families this pipeline does not run, and Python's
    /// `load_image` itself reduces the item to `.url`.
    Ref { url: String },
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
            MmItem::Source(source) | MmItem::Ref { url: source } => Some(source),
            MmItem::Preprocessed { .. } => None,
        }
    }
}

impl HeapBytes for MmItem {
    fn heap_bytes(&self) -> usize {
        match self {
            MmItem::Source(s) | MmItem::Ref { url: s } | MmItem::Preprocessed { format: s } => {
                s.len()
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
}

impl TryFrom<ItemObject> for MmItem {
    type Error = &'static str;

    fn try_from(object: ItemObject) -> Result<Self, Self::Error> {
        match (object.format, object.url) {
            (Some(format), _) => Ok(MmItem::Preprocessed { format }),
            (None, Some(url)) => Ok(MmItem::Ref { url }),
            (None, None) => Err("a multimodal item object needs a `url` or a `format` key"),
        }
    }
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
            MmDataInput::One(MmItem::Ref { url: "u".into() })
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
