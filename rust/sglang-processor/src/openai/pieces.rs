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
        // `_is_byte_level_tokenizer` samples these ids of `get_vocab()`.
        let size = pieces
            .values()
            .collect::<std::collections::HashSet<_>>()
            .len() as u32;
        let byte_level = [0, 1, 2, 3, size / 2, size.wrapping_sub(2)]
            .into_iter()
            .filter(|&id| id < size)
            .filter_map(|id| pieces.get(&id).filter(|piece| !piece.is_empty()))
            .all(|piece| piece.chars().all(|c| byte_level_byte(c).is_some()));
        Self { pieces, byte_level }
    }

    /// The token's piece when the tokenizer is byte-level BPE.
    pub fn byte_level_piece(&self, id: u32) -> Option<&str> {
        self.byte_level
            .then(|| self.pieces.get(&id).map(String::as_str))
            .flatten()
    }
}

/// GPT-2's `bytes_to_unicode`, inverted for one character.
pub(super) fn byte_level_byte(c: char) -> Option<u8> {
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
    fn byte_level_probe_matches_python() {
        let gpt2 = json!({"model": {"vocab": {"!": 0, "\"": 1, "#": 2, "Ġa": 3, "é": 4}}, "added_tokens": []});
        let pieces = TokenPieces::from_tokenizer_json(&gpt2);
        assert_eq!(pieces.byte_level_piece(3), Some("Ġa"));
        // DeepSeek's first ids are full-width special tokens, so Python says no.
        let deepseek = json!({"model": {"vocab": {"!": 3, "a": 4}},
            "added_tokens": [{"id": 0, "content": "<｜begin▁of▁sentence｜>"}]});
        assert_eq!(
            TokenPieces::from_tokenizer_json(&deepseek).byte_level_piece(3),
            None
        );
        assert_eq!(byte_level_byte('Ġ'), Some(b' '));
    }
}
