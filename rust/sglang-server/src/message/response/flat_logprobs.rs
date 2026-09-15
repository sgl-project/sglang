use base64::{Engine, engine::general_purpose::STANDARD};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FlatTopLogprobShape {
    pub rows: u32,
    pub top_k: u64,
    pub null_prefix: u32,
}

impl FlatTopLogprobShape {
    pub fn elements(self) -> Option<usize> {
        (self.rows as usize).checked_mul(self.top_k.try_into().ok()?)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LogprobOptions {
    pub top_k: u64,
    pub token_ids: bool,
    pub flat: bool,
    pub base64: bool,
}

/// Scheduler f32/i32 arrays remain raw until the detokenizer encodes the requested
/// JSON representation. Base64 never expands them into numeric JSON values.
#[derive(Debug, Clone)]
pub struct FlatTopLogprobs {
    pub shape: FlatTopLogprobShape,
    pub values: Bytes,
    pub indices: Bytes,
}

impl FlatTopLogprobs {
    pub fn from_nested(
        values: &[f32],
        indices: &[i32],
        lengths: &[u32],
        top_k: u64,
    ) -> Option<Self> {
        let null_prefix = lengths.iter().take_while(|&&length| length == 0).count();
        let top_k = lengths
            .get(null_prefix)
            .map_or(top_k, |&width| u64::from(width));
        if lengths[null_prefix..]
            .iter()
            .any(|&length| u64::from(length) != top_k)
        {
            return None;
        }
        let shape = FlatTopLogprobShape {
            rows: (lengths.len() - null_prefix).try_into().ok()?,
            top_k,
            null_prefix: null_prefix.try_into().ok()?,
        };
        if shape.elements()? != values.len() || values.len() != indices.len() {
            return None;
        }
        Some(Self {
            shape,
            values: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
                .into(),
            indices: indices
                .iter()
                .flat_map(|index| index.to_le_bytes())
                .collect::<Vec<_>>()
                .into(),
        })
    }

    pub fn empty(top_k: u64) -> Self {
        Self {
            shape: FlatTopLogprobShape {
                rows: 0,
                top_k,
                null_prefix: 0,
            },
            values: Bytes::new(),
            indices: Bytes::new(),
        }
    }

    pub fn into_fields(self, base64: bool) -> Map<String, Value> {
        let mut fields = Map::new();
        if base64 {
            fields.insert(
                "input_top_logprobs_val_flat_b64".into(),
                json!(STANDARD.encode(self.values)),
            );
            fields.insert(
                "input_top_logprobs_idx_flat_b64".into(),
                json!(STANDARD.encode(self.indices)),
            );
            fields.insert(
                "input_top_logprobs_val_flat_b64_dtype".into(),
                json!("float32"),
            );
            fields.insert(
                "input_top_logprobs_idx_flat_b64_dtype".into(),
                json!("int32"),
            );
        } else {
            let values: Vec<_> = self
                .values
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            let indices: Vec<_> = self
                .indices
                .chunks_exact(4)
                .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            fields.insert("input_top_logprobs_val_flat".into(), json!(values));
            fields.insert("input_top_logprobs_idx_flat".into(), json!(indices));
        }
        fields.insert(
            "input_top_logprobs_shape".into(),
            json!([self.shape.rows, self.shape.top_k]),
        );
        fields.insert(
            "input_top_logprobs_null_prefix".into(),
            json!(self.shape.null_prefix),
        );
        fields
    }
}
