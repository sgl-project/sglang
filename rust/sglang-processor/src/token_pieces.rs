//! Token pieces from `tokenizer.json`, for Python's byte-level token text.

use std::collections::HashMap;

use serde_json::Value;

/// `tokenizer.convert_ids_to_tokens` over a `tokenizer.json`, and whether
/// `_is_byte_level_tokenizer` holds for it.
pub struct TokenPieces {
    pieces: HashMap<u32, String>,
    byte_level: bool,
}

impl TokenPieces {
    pub fn from_tokenizer_json(tokenizer: &Value) -> Self {
        let mut pieces = HashMap::new();
        if let Some(vocab) = tokenizer.pointer("/model/vocab").and_then(Value::as_object) {
            for (piece, id) in vocab {
                if let Some(id) = id.as_u64() {
                    pieces.insert(id as u32, piece.clone());
                }
            }
        }
        for added in tokenizer["added_tokens"].as_array().into_iter().flatten() {
            if let (Some(id), Some(piece)) = (added["id"].as_u64(), added["content"].as_str()) {
                pieces.insert(id as u32, piece.to_owned());
            }
        }
        // `_is_byte_level_tokenizer` answers a fast tokenizer with its declared decoder.
        let byte_level = is_byte_level_decoder(&tokenizer["decoder"]);
        Self { pieces, byte_level }
    }

    /// The token's bytes when the tokenizer is byte-level BPE.
    pub fn byte_level_bytes(&self, id: u32) -> Option<Vec<u8>> {
        let piece = self.pieces.get(&id).filter(|_| self.byte_level)?;
        let bytes = piece
            .chars()
            .map(byte_level_byte)
            .collect::<Option<Vec<u8>>>()?;
        (!bytes.is_empty()).then_some(bytes)
    }
}

fn is_byte_level_decoder(mut decoder: &Value) -> bool {
    while decoder["type"] == "Sequence" {
        match decoder["decoders"].as_array().map(Vec::as_slice) {
            Some([step]) => decoder = step,
            _ => return false,
        }
    }
    decoder["type"] == "ByteLevel"
}

/// GPT-2's `bytes_to_unicode`, inverted for one character.
fn byte_level_byte(c: char) -> Option<u8> {
    let printable = |b: u32| matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
    let c = c as u32;
    if printable(c) {
        return Some(c as u8);
    }
    let offset = c.checked_sub(256)? as usize;
    (0u32..256)
        .filter(|&b| !printable(b))
        .nth(offset)
        .map(|b| b as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn byte_level_follows_the_declared_decoder() {
        let gpt2 = json!({"model": {"vocab": {"!": 0, "\"": 1, "#": 2, "Ġa": 3, "é": 4}}, "added_tokens": [],
            "decoder": {"type": "ByteLevel"}});
        let pieces = TokenPieces::from_tokenizer_json(&gpt2);
        assert_eq!(pieces.byte_level_bytes(3), Some(b" a".to_vec()));
        // DeepSeek's first ids are full-width special tokens; its decoder is still ByteLevel.
        let deepseek = json!({"model": {"vocab": {"!": 3, "a": 4}},
            "added_tokens": [{"id": 0, "content": "<｜begin▁of▁sentence｜>"}], "decoder": {"type": "ByteLevel"}});
        assert_eq!(
            TokenPieces::from_tokenizer_json(&deepseek).byte_level_bytes(3),
            Some(b"!".to_vec())
        );
        // LFM2 wraps its ByteLevel decoder in a Sequence.
        let lfm2 = json!({"model": {"vocab": {"!": 0}}, "added_tokens": [],
            "decoder": {"type": "Sequence", "decoders": [{"type": "ByteLevel"}]}});
        assert_eq!(
            TokenPieces::from_tokenizer_json(&lfm2).byte_level_bytes(0),
            Some(b"!".to_vec())
        );
        let sentencepiece = json!({"model": {"vocab": {"!": 0, "a": 1}}, "added_tokens": [],
            "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}]}});
        assert_eq!(
            TokenPieces::from_tokenizer_json(&sentencepiece).byte_level_bytes(0),
            None
        );
        assert_eq!(byte_level_byte('Ġ'), Some(b' '));

        let nested = json!({
            "model": {"vocab": {"ð": 1, "Ł": 2, "ĺ": 3, "Ģ": 4}},
            "added_tokens": [{"id": 0, "content": "<｜begin▁of▁sentence｜>"}],
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "Sequence", "decoders": [{"type": "ByteLevel"}]}
            ]}
        });
        let pieces = TokenPieces::from_tokenizer_json(&nested);
        let bytes: Vec<_> = (1..=4)
            .flat_map(|id| pieces.byte_level_bytes(id).unwrap())
            .collect();
        assert_eq!(bytes, "😀".as_bytes());
        assert_eq!(pieces.byte_level_bytes(0), None);

        let transforming = json!({
            "model": {"vocab": {"é": 0}},
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "é"}, "content": "Ã©"},
                {"type": "ByteLevel"}
            ]}
        });
        assert_eq!(
            TokenPieces::from_tokenizer_json(&transforming).byte_level_bytes(0),
            None
        );
    }
}
