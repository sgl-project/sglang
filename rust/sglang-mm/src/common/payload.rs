//! Native multimodal request payload parsing.

use rmpv::Value;

#[derive(Debug)]
pub enum NativeError {
    Fallback(String),
    Failed(String),
}

impl std::fmt::Display for NativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fallback(message) => write!(f, "fallback: {message}"),
            Self::Failed(message) => write!(f, "failed: {message}"),
        }
    }
}

#[derive(Debug)]
pub enum ImageSource {
    String(String),
    Bytes(Vec<u8>),
}

pub struct NativePayload {
    pub text: Option<String>,
    pub input_ids: Option<Vec<i32>>,
    pub images: Vec<ImageSource>,
}

/// Decode `[text, input_ids, image_data, video_data, audio_data]`.
pub fn parse(payload: &[u8]) -> Result<NativePayload, NativeError> {
    let value = rmpv::decode::read_value(&mut &payload[..])
        .map_err(|e| NativeError::Failed(format!("mm payload decode: {e}")))?;
    let Value::Array(fields) = value else {
        return Err(NativeError::Failed("mm payload is not an array".into()));
    };
    if fields.len() != 5 {
        return Err(NativeError::Failed("mm payload arity mismatch".into()));
    }
    if value_present(&fields[3]) || value_present(&fields[4]) {
        return Err(NativeError::Fallback("video/audio input".into()));
    }

    let text = match &fields[0] {
        Value::Nil => None,
        Value::String(value) => value.as_str().map(str::to_owned),
        _ => return Err(NativeError::Failed("mm payload: non-string text".into())),
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
                        .ok_or_else(|| NativeError::Failed("mm payload: non-int input id".into()))
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        _ => return Err(NativeError::Failed("mm payload: bad input_ids".into())),
    };

    let mut images = Vec::new();
    collect_images(&fields[2], &mut images)?;
    if images.is_empty() {
        return Err(NativeError::Fallback("no raw image sources".into()));
    }
    Ok(NativePayload {
        text,
        input_ids,
        images,
    })
}

fn collect_images(value: &Value, out: &mut Vec<ImageSource>) -> Result<(), NativeError> {
    match value {
        Value::Nil => Ok(()),
        Value::String(value) => {
            let value = value
                .as_str()
                .ok_or_else(|| NativeError::Failed("non-utf8 image source".into()))?;
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
                        return Err(NativeError::Fallback(
                            "nested/typed image_data shape".into(),
                        ));
                    }
                }
            }
            Ok(())
        }
        _ => Err(NativeError::Fallback("unsupported image_data shape".into())),
    }
}

fn value_present(value: &Value) -> bool {
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
    fn unsupported_modalities_and_shapes_fall_back() {
        let video = encode(vec![
            Value::from("prompt"),
            Value::Nil,
            Value::Nil,
            Value::from("video.mp4"),
            Value::Nil,
        ]);
        assert!(matches!(parse(&video), Err(NativeError::Fallback(_))));

        let dict = Value::Map(vec![(Value::from("format"), Value::from("x"))]);
        assert!(matches!(
            parse(&image_payload(Value::Array(vec![dict]))),
            Err(NativeError::Fallback(_))
        ));
    }

    #[test]
    fn malformed_payloads_fail() {
        // Truncated msgpack (array header, no elements) and wrong arity.
        assert!(matches!(parse(b"\x91"), Err(NativeError::Failed(_))));
        let three = encode(vec![Value::Nil, Value::Nil, Value::from("a")]);
        assert!(matches!(parse(&three), Err(NativeError::Failed(_))));
    }

    #[test]
    fn empty_video_audio_lists_do_not_fall_back() {
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
    fn image_free_payload_falls_back() {
        assert!(matches!(
            parse(&image_payload(Value::Nil)),
            Err(NativeError::Fallback(_))
        ));
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
        assert!(matches!(parse(&payload), Err(NativeError::Failed(_))));
    }
}
