//! The MM wire payload: parse the msgpack blob built by
//! [`super::request::GenerateRequest::to_mm_payload_msgpack`] into the typed
//! [`MmInput`] the `sglang-mm` driver consumes. Encoder and decoder live in
//! this crate so the wire contract has one owner.
//!
//! Every `Err` rejects the request back to the client (there is no Python
//! fallback path); the message says whether the input is malformed or merely
//! outside the pipeline's scope (video/audio, precomputed features, ...).

use bytes::Bytes;
use rmpv::Value;
use sglang_mm::driver::{ImageSource, MmInput};

/// True for source forms the API layer must resolve before MM dispatch: I/O —
/// network *or* disk (a network mount can hang past any HTTP timeout) — never
/// runs on the fixed MM worker pool (see `api_server::prefetch`). `data:` and
/// bare base64 are pure CPU and stay on the worker. Owned here, next to
/// `collect_images`, so the prefetch walk and the parse walk can never drift.
pub fn is_io_source(src: &str) -> bool {
    src.starts_with("http://")
        || src.starts_with("https://")
        || src.starts_with("file://")
        || src.starts_with('/')
}

/// The I/O-backed sources of an `image_data` value, in `collect_images` order.
pub fn io_sources(value: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let mut walk = |value: &Value| {
        if let Some(src) = value.as_str().filter(|s| is_io_source(s)) {
            out.push(src.to_owned());
        }
    };
    if let Value::Array(values) = value {
        values.iter().for_each(&mut walk);
    } else {
        walk(value);
    }
    out
}

/// Decode `[text, input_ids, image_data, video_data, audio_data]`.
/// `prefetched` holds the already-resolved bytes of the payload's I/O-backed
/// sources (in `io_sources` order); those sources are swapped for their
/// bytes, and one left without an entry is an internal error, not a fetch.
pub fn parse(payload: &[u8], prefetched: &[Bytes]) -> Result<MmInput, String> {
    let value = rmpv::decode::read_value(&mut &payload[..])
        .map_err(|e| format!("mm payload decode: {e}"))?;
    let Value::Array(fields) = value else {
        return Err("mm payload is not an array".into());
    };
    if fields.len() != 5 {
        return Err("mm payload arity mismatch".into());
    }
    if value_present(&fields[3]) || value_present(&fields[4]) {
        return Err("unsupported modality: video/audio input".into());
    }

    let text = match &fields[0] {
        Value::Nil => None,
        Value::String(value) => Some(
            value
                .as_str()
                .ok_or_else(|| "mm payload: non-utf8 text".to_string())?
                .to_owned(),
        ),
        _ => return Err("mm payload: non-string text".into()),
    };
    let input_ids = match &fields[1] {
        Value::Nil => None,
        Value::Array(values) => Some(
            values
                .iter()
                .map(|value| {
                    value
                        .as_i64()
                        .and_then(|id| i32::try_from(id).ok())
                        .ok_or_else(|| "mm payload: non-int input id".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        _ => return Err("mm payload: bad input_ids".into()),
    };

    let mut images = Vec::new();
    collect_images(&fields[2], &mut prefetched.iter(), &mut images)?;
    if images.is_empty() {
        return Err("no raw image sources in mm payload".into());
    }
    Ok(MmInput {
        text,
        input_ids,
        images,
    })
}

fn collect_images(
    value: &Value,
    prefetched: &mut std::slice::Iter<Bytes>,
    out: &mut Vec<ImageSource>,
) -> Result<(), String> {
    match value {
        Value::Nil => Ok(()),
        Value::String(value) => {
            let value = value
                .as_str()
                .ok_or_else(|| "non-utf8 image source".to_string())?;
            if is_io_source(value) {
                let bytes = prefetched
                    .next()
                    .ok_or_else(|| "I/O-backed image source was not prefetched".to_string())?;
                out.push(ImageSource::Bytes(bytes.to_vec()));
            } else {
                out.push(ImageSource::String(value.to_owned()));
            }
            Ok(())
        }
        Value::Binary(value) => {
            out.push(ImageSource::Bytes(value.clone()));
            Ok(())
        }
        Value::Array(values) => {
            for value in values {
                match value {
                    Value::String(_) | Value::Binary(_) | Value::Nil => {
                        collect_images(value, prefetched, out)?
                    }
                    _ => {
                        return Err("unsupported image_data shape: nested/typed item".into());
                    }
                }
            }
            Ok(())
        }
        _ => Err("unsupported image_data shape".into()),
    }
}

/// Rust mirror of Python `has_valid_data`: `nil` and (recursively) empty /
/// all-nil lists don't count as multimodal input. Shared with the ingress
/// `has_multimodal` routing check so routing and parsing can never drift.
pub fn value_present(value: &Value) -> bool {
    match value {
        Value::Nil => false,
        Value::Array(values) => values.iter().any(value_present),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(fields: Vec<Value>) -> Vec<u8> {
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &Value::Array(fields)).unwrap();
        bytes
    }

    fn image_payload(image: Value) -> Vec<u8> {
        encode(vec![
            Value::from("prompt"),
            Value::Nil,
            image,
            Value::Nil,
            Value::Nil,
        ])
    }

    #[test]
    fn parses_string_and_list_images() {
        let one = parse(&image_payload(Value::from("data:image/png;base64,x")), &[]).unwrap();
        assert_eq!(one.images.len(), 1);
        let many = parse(
            &image_payload(Value::Array(vec![Value::from("a"), Value::from("b")])),
            &[],
        )
        .unwrap();
        assert_eq!(many.images.len(), 2);
    }

    #[test]
    fn unsupported_modalities_and_shapes_rejected() {
        let video = encode(vec![
            Value::from("prompt"),
            Value::Nil,
            Value::Nil,
            Value::from("video.mp4"),
            Value::Nil,
        ]);
        assert!(parse(&video, &[]).err().unwrap().contains("video/audio"));

        let dict = Value::Map(vec![(Value::from("format"), Value::from("x"))]);
        assert!(
            parse(&image_payload(Value::Array(vec![dict])), &[])
                .err()
                .unwrap()
                .contains("image_data shape")
        );
    }

    #[test]
    fn malformed_payloads_fail() {
        // Truncated msgpack (array header, no elements) and wrong arity.
        assert!(parse(b"\x91", &[]).is_err());
        let three = encode(vec![Value::Nil, Value::Nil, Value::from("a")]);
        assert!(parse(&three, &[]).is_err());
    }

    #[test]
    fn empty_video_audio_lists_are_not_modalities() {
        // Mirrors Python `has_valid_data`: nil / empty lists don't count.
        let payload = encode(vec![
            Value::Nil,
            Value::Array(vec![Value::from(1)]),
            Value::from("a"),
            Value::Array(vec![]),
            Value::Array(vec![Value::Array(vec![])]),
        ]);
        assert_eq!(parse(&payload, &[]).unwrap().images.len(), 1);
    }

    /// I/O-backed sources (URLs and file paths) are swapped for their
    /// prefetched bytes in walk order; one left unfetched is an error, so
    /// neither network nor disk I/O can ever reach an MM worker.
    #[test]
    fn io_sources_use_prefetched_bytes() {
        let image = Value::Array(vec![
            Value::from("http://a/x.png"),
            Value::from("data:image/png;base64,x"),
            Value::from("/mnt/nfs/y.png"),
        ]);
        assert_eq!(io_sources(&image), vec!["http://a/x.png", "/mnt/nfs/y.png"]);

        let fetched = [Bytes::from_static(b"aa"), Bytes::from_static(b"bb")];
        let input = parse(&image_payload(image.clone()), &fetched).unwrap();
        let as_bytes = |i: usize| match &input.images[i] {
            ImageSource::Bytes(b) => b.as_slice(),
            other => panic!("expected bytes, got {other:?}"),
        };
        assert_eq!(as_bytes(0), b"aa");
        assert_eq!(as_bytes(2), b"bb");
        assert!(matches!(&input.images[1], ImageSource::String(_)));

        let err = parse(&image_payload(image), &[]).err().unwrap();
        assert!(err.contains("not prefetched"), "{err}");
    }

    #[test]
    fn image_free_payload_rejected() {
        assert!(
            parse(&image_payload(Value::Nil), &[])
                .err()
                .unwrap()
                .contains("no raw image sources")
        );
    }

    #[test]
    fn rejects_non_integer_input_ids() {
        for bad_id in [
            Value::from("not-an-id"),
            Value::from(i64::from(i32::MAX) + 1),
        ] {
            let payload = encode(vec![
                Value::Nil,
                Value::Array(vec![bad_id]),
                Value::from("a"),
                Value::Nil,
                Value::Nil,
            ]);
            assert!(parse(&payload, &[]).is_err());
        }
    }
}
