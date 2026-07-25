//! Server multimodal request payload parsing.
//!
//! Every `Err` rejects the request back to the client (there is no Python
//! fallback path); the message says whether the input is malformed or merely
//! outside the pipeline's scope (video/audio, precomputed features, ...).

use rmpv::Value;

#[derive(Debug)]
pub enum ImageSource {
    String(String),
    Bytes(Vec<u8>),
}

#[derive(Debug)]
pub struct Payload {
    pub text: Option<String>,
    pub input_ids: Option<Vec<i32>>,
    pub images: Vec<ImageSource>,
}

/// Decode `[text, input_ids, image_data, video_data, audio_data]`.
pub fn parse(payload: &[u8]) -> Result<Payload, String> {
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
        Value::String(value) => value.as_str().map(str::to_owned),
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
                        .map(|id| id as i32)
                        .ok_or_else(|| "mm payload: non-int input id".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        _ => return Err("mm payload: bad input_ids".into()),
    };

    let mut images = Vec::new();
    collect_images(&fields[2], &mut images)?;
    if images.is_empty() {
        return Err("no raw image sources in mm payload".into());
    }
    Ok(Payload {
        text,
        input_ids,
        images,
    })
}

fn collect_images(value: &Value, out: &mut Vec<ImageSource>) -> Result<(), String> {
    match value {
        Value::Nil => Ok(()),
        Value::String(value) => {
            let value = value
                .as_str()
                .ok_or_else(|| "non-utf8 image source".to_string())?;
            out.push(ImageSource::String(value.to_owned()));
            Ok(())
        }
        Value::Binary(value) => {
            out.push(ImageSource::Bytes(value.clone()));
            Ok(())
        }
        Value::Array(values) => {
            for value in values {
                match value {
                    Value::String(_) | Value::Binary(_) | Value::Nil => collect_images(value, out)?,
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
/// all-nil lists don't count as multimodal input. Shared with the server's
/// `has_multimodal` routing check so the two can never drift.
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
        let one = parse(&image_payload(Value::from("data:image/png;base64,x"))).unwrap();
        assert_eq!(one.images.len(), 1);
        let many = parse(&image_payload(Value::Array(vec![
            Value::from("a"),
            Value::from("b"),
        ])))
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
        assert!(parse(&video).unwrap_err().contains("video/audio"));

        let dict = Value::Map(vec![(Value::from("format"), Value::from("x"))]);
        assert!(
            parse(&image_payload(Value::Array(vec![dict])))
                .unwrap_err()
                .contains("image_data shape")
        );
    }

    #[test]
    fn malformed_payloads_fail() {
        // Truncated msgpack (array header, no elements) and wrong arity.
        assert!(parse(b"\x91").is_err());
        let three = encode(vec![Value::Nil, Value::Nil, Value::from("a")]);
        assert!(parse(&three).is_err());
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
        assert_eq!(parse(&payload).unwrap().images.len(), 1);
    }

    #[test]
    fn image_free_payload_rejected() {
        assert!(
            parse(&image_payload(Value::Nil))
                .unwrap_err()
                .contains("no raw image sources")
        );
    }

    #[test]
    fn rejects_non_integer_input_ids() {
        let payload = encode(vec![
            Value::Nil,
            Value::Array(vec![Value::from("not-an-id")]),
            Value::from("a"),
            Value::Nil,
            Value::Nil,
        ]);
        assert!(parse(&payload).is_err());
    }
}
