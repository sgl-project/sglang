//! Typed multimodal inputs of the `/generate` body — the Rust form of Python
//! `MultimodalDataInputFormat` (`io_struct.py`) — and their per-request fan-out.

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
}

impl MmItem {
    /// The raw source string for the modality pipeline.
    pub fn source(&self) -> Option<&str> {
        match self {
            MmItem::Source(source) | MmItem::Ref { url: source } => Some(source),
        }
    }
}

impl HeapBytes for MmItem {
    fn heap_bytes(&self) -> usize {
        match self {
            MmItem::Source(s) | MmItem::Ref { url: s } => s.len(),
        }
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

    /// Parsed as the wire type (the schema's `MediaInput`), then converted;
    /// either step's rejection is an `Err`.
    fn parse(json: &str) -> Result<MmDataInput, String> {
        let wire: sglang_api_types::api::v1::MediaInput =
            serde_json::from_str(json).map_err(|e| e.to_string())?;
        crate::message::api::media_input(wire).map_err(|e| e.to_string())
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

    /// Object items are typed `MediaRef`s: only `url` reaches this pipeline, a
    /// hint the schema does not name is an error, and a preprocessed-input
    /// dict is not a wire shape at all (its values are tensors; Engine-only).
    #[test]
    fn parses_item_objects() {
        assert_eq!(
            parse(r#"{"url": "u", "detail": "high"}"#).unwrap(),
            MmDataInput::One(MmItem::Ref { url: "u".into() })
        );
        let err = parse(r#"{"url": "u", "detai": "high"}"#)
            .unwrap_err()
            .to_string(); // codespell:ignore detai
        assert!(err.contains("unknown field"), "{err}");
        assert!(parse(r#"[{"format": "processor_output", "url": "u"}]"#).is_err());
        assert!(parse(r#"{"detail": "high"}"#).is_err(), "a ref needs a url");
    }

    /// Anything Python's item union does not cover is rejected up front.
    #[test]
    fn rejects_non_items() {
        for json in ["5", r#"["a", 5]"#, r#"["a", ["b"]]"#, r#"[[["a"]]]"#] {
            assert!(parse(json).is_err(), "{json}");
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
