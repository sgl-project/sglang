//! msgpack (de)serialization for the proto messages, for the ZMQ transport path
//! (RFC #22558 Phase 4+). The proto is the IDL; `build.rs` derives serde on every
//! generated message. This module pins the wire convention so the Python `msgspec`
//! codec generated from the same proto interoperates byte-for-byte where it counts.
//!
//! Wire convention (must match `python/sglang/srt/grpc/messages.py`):
//!   * each message is a msgpack **map keyed by proto field name** (`with_struct_map`),
//!   * `bytes` fields are msgpack **bin**, not int arrays (`BytesMode::ForceAll`),
//!   * default-valued fields are omitted (skip attrs from `build.rs`),
//!   * decoding is tolerant — unknown keys are ignored, absent keys take defaults.
//!
//! Note: proto `float` is 32-bit but Python emits 64-bit doubles, so float fields
//! are not byte-identical across languages; both decoders coerce numerically and the
//! effective precision is float32 (the proto's declared width).

use serde::{de::DeserializeOwned, Serialize};

/// Serialize a generated proto message to msgpack bytes for the ZMQ wire.
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    let mut buf = Vec::new();
    let mut serializer = rmp_serde::Serializer::new(&mut buf)
        .with_struct_map()
        .with_bytes(rmp_serde::config::BytesMode::ForceAll);
    value.serialize(&mut serializer)?;
    Ok(buf)
}

/// Deserialize a generated proto message from msgpack bytes received on the ZMQ wire.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, rmp_serde::decode::Error> {
    rmp_serde::from_slice(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::*;

    #[test]
    fn round_trip_text_generate_request() {
        let mut trace_headers = std::collections::HashMap::new();
        trace_headers.insert("traceparent".to_string(), "00-abc-def-01".to_string());
        let req = TextGenerateRequest {
            text: "Hello, msgpack!".to_string(),
            sampling_params: Some(SamplingParams {
                temperature: Some(0.7),
                top_p: Some(0.95),
                max_new_tokens: Some(128),
                stop: vec!["</s>".to_string()],
                ..Default::default()
            }),
            stream: Some(true),
            rid: Some("req-123".to_string()),
            trace_headers,
            ..Default::default()
        };
        let bytes = encode(&req).unwrap();
        let back: TextGenerateRequest = decode(&bytes).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn unset_optionals_are_omitted() {
        // Empty SamplingParams must encode as an empty msgpack map (0x80), matching
        // the Python codec exactly — proof the skip_serializing_if attrs are applied.
        let bytes = encode(&SamplingParams::default()).unwrap();
        assert_eq!(bytes, vec![0x80]);
    }

    #[test]
    fn bytes_field_is_msgpack_bin() {
        // proto `bytes` must be msgpack bin (0xc4/0xc5/0xc6) so Python's msgspec
        // decodes it as `bytes`, not as a list of ints.
        let req = OpenAiRequest {
            json_body: br#"{"model":"x"}"#.to_vec(),
            ..Default::default()
        };
        let bytes = encode(&req).unwrap();
        assert!(bytes.iter().any(|&b| matches!(b, 0xc4 | 0xc5 | 0xc6)));
        let back: OpenAiRequest = decode(&bytes).unwrap();
        assert_eq!(req.json_body, back.json_body);
    }

    #[test]
    fn repeated_int_round_trip() {
        let req = GenerateRequest {
            input_ids: vec![1, 2, 300, 40000],
            ..Default::default()
        };
        let back: GenerateRequest = decode(&encode(&req).unwrap()).unwrap();
        assert_eq!(req.input_ids, back.input_ids);
    }

    fn from_hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    // Canonical wire vectors minted from the Python `msgspec` codec generated from
    // the same proto (see test/registered/unit/grpc/test_msgpack_codec.py). These
    // must stay byte-identical so the two codecs can never silently drift apart.
    const GV_GENERATE: &str = "83a9696e7075745f69647393010203af73616d706c696e675f706172616d7383ae6d61785f6e65775f746f6b656e7310ae73746f705f746f6b656e5f6964739102a16e01a3726964a3616263";
    const GV_OPENAI: &str = "82a96a736f6e5f626f6479c4077b2261223a317dad74726163655f6865616465727381a178a179";

    #[test]
    fn golden_vector_generate_request() {
        let expect = from_hex(GV_GENERATE);
        let req = GenerateRequest {
            input_ids: vec![1, 2, 3],
            rid: Some("abc".into()),
            sampling_params: Some(SamplingParams {
                n: Some(1),
                max_new_tokens: Some(16),
                stop_token_ids: vec![2],
                ..Default::default()
            }),
            ..Default::default()
        };
        // Byte-identical with Python (no float fields here).
        assert_eq!(encode(&req).unwrap(), expect);
        assert_eq!(decode::<GenerateRequest>(&expect).unwrap(), req);
    }

    #[test]
    fn golden_vector_openai_request() {
        let expect = from_hex(GV_OPENAI);
        let mut trace_headers = std::collections::HashMap::new();
        trace_headers.insert("x".to_string(), "y".to_string());
        let req = OpenAiRequest {
            json_body: br#"{"a":1}"#.to_vec(),
            trace_headers,
        };
        assert_eq!(encode(&req).unwrap(), expect);
        assert_eq!(decode::<OpenAiRequest>(&expect).unwrap(), req);
    }
}
