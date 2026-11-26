// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use base64::prelude::{Engine as _, BASE64_STANDARD};
use serde_json::Value;

/// Metadata produced by both legs of a buffered native /generate request.
#[derive(Clone, Copy, Default)]
pub(super) struct PrefillMetadata {
    pub(super) input_logprobs: bool,
    pub(super) routed_experts_start: Option<u64>,
}

impl PrefillMetadata {
    pub(super) fn requested(self) -> bool {
        self.input_logprobs || self.routed_experts_start.is_some()
    }

    pub(super) fn merge(self, prefill_body: &[u8], decode_body: &[u8]) -> Option<Vec<u8>> {
        let prefill: Value = serde_json::from_slice(prefill_body).ok()?;
        let mut decode: Value = serde_json::from_slice(decode_body).ok()?;
        let mut changed = false;
        if self.input_logprobs {
            const PATH: &str = "/meta_info/input_token_logprobs";
            if let (Some(p), Some(d)) = (
                prefill.pointer(PATH).and_then(Value::as_array),
                decode.pointer_mut(PATH).and_then(Value::as_array_mut),
            ) {
                d.splice(0..0, p.iter().cloned());
                changed = true;
            }
        }
        if let Some(start) = self.routed_experts_start {
            match (&prefill, &mut decode) {
                (Value::Array(p), Value::Array(d)) if p.len() == d.len() => {
                    for (prefill, decode) in p.iter().zip(d) {
                        changed |= merge_experts(prefill, decode, start).is_some();
                    }
                }
                (Value::Object(_), Value::Object(_)) => {
                    changed |= merge_experts(&prefill, &mut decode, start).is_some();
                }
                _ => {}
            }
        }
        if changed {
            serde_json::to_vec(&decode).ok()
        } else {
            None
        }
    }
}

fn merge_experts(prefill: &Value, decode: &mut Value, start: u64) -> Option<()> {
    let meta = decode.get_mut("meta_info")?.as_object_mut()?;
    let prompt_rows =
        usize::try_from(meta.get("prompt_tokens")?.as_u64()?.checked_sub(start)?).ok()?;
    let completion_tokens = meta.get("completion_tokens")?.as_u64()?;
    let encoded = prefill.pointer("/meta_info/routed_experts")?.as_str()?;
    if prompt_rows == 0 {
        // An empty interval is still an encoded tensor on a plain worker.
        if completion_tokens <= 1 && encoded.is_empty() {
            meta.insert("routed_experts".into(), encoded.into());
            return Some(());
        }
        return None;
    }
    let mut prompt = BASE64_STANDARD.decode(encoded).ok()?;
    let row_bytes = prompt.len() / prompt_rows;
    if row_bytes == 0
        || !row_bytes.is_multiple_of(size_of::<i32>())
        || !prompt.len().is_multiple_of(prompt_rows)
    {
        return None;
    }
    match completion_tokens {
        // PD prefill always computes its first output token. A plain worker's
        // capture ends at seqlen - 1, omitting the final prompt row with no output.
        0 => prompt.truncate(prompt.len() - row_bytes),
        1 => {}
        count => {
            let generated_rows = usize::try_from(count - 1).ok()?;
            let suffix_bytes = generated_rows.checked_mul(row_bytes)?;
            let decode_routes = BASE64_STANDARD
                .decode(meta.get("routed_experts")?.as_str()?)
                .ok()?;
            // Older workers also read back prompt slots on decode. Only prefill owns those rows.
            let offset = decode_routes.len().checked_sub(suffix_bytes)?;
            if offset != 0 && offset != prompt.len() {
                return None;
            }
            prompt.extend_from_slice(&decode_routes[offset..]);
        }
    }
    meta.insert(
        "routed_experts".into(),
        BASE64_STANDARD.encode(prompt).into(),
    );
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn leaves_decode_metadata_intact_if_rows_cannot_be_paired() {
        let flags = PrefillMetadata {
            routed_experts_start: Some(0),
            ..Default::default()
        };
        let p = json!({"meta_info": {"routed_experts": BASE64_STANDARD.encode([1_u8; 8])}});
        let d = json!({"meta_info": {"prompt_tokens": 2, "completion_tokens": 2,
                                      "routed_experts": BASE64_STANDARD.encode([8_u8; 8])}});
        let merge = |p: Value, d: Value| {
            flags.merge(
                &serde_json::to_vec(&p).unwrap(),
                &serde_json::to_vec(&d).unwrap(),
            )
        };
        // Four-byte rows: decode supplied two rows, while only one or three are valid.
        assert!(merge(p.clone(), d.clone()).is_none());
        assert!(merge(json!([p.clone()]), json!([d.clone(), d.clone()])).is_none());
        assert!(merge(json!({"meta_info": {"routed_experts": "?"}}), d.clone()).is_none());
        assert!(merge(json!({}), d).is_none());
    }
}
