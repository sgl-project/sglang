// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Chat reply rewrites: `reasoning_format: general` renames `reasoning_content`
//! to `reasoning` in messages and stream deltas.

use serde_json::Value;

use super::{LineBuffer, SseTransducer};

const FROM: &str = "reasoning_content";
const TO: &str = "reasoning";

/// Rename the field in every `choices[*].{message,delta}`; `true` if any changed.
pub fn rename_reasoning(v: &mut Value) -> bool {
    let mut changed = false;
    for choice in v
        .get_mut("choices")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
    {
        for key in ["message", "delta"] {
            if let Some(Value::Object(m)) = choice.get_mut(key) {
                if let Some(r) = m.remove(FROM) {
                    m.insert(TO.into(), r);
                    changed = true;
                }
            }
        }
    }
    changed
}

/// Line-level SSE rewrite; lines without the field pass through byte-for-byte.
#[derive(Default)]
pub struct RenameReasoning {
    lines: LineBuffer,
}

impl RenameReasoning {
    fn line(line: &[u8], out: &mut Vec<u8>) {
        let rewritten = line
            .strip_prefix(b"data:")
            .filter(|p| p.windows(FROM.len()).any(|w| w == FROM.as_bytes()))
            .and_then(|p| serde_json::from_slice::<Value>(p).ok())
            .and_then(|mut v| rename_reasoning(&mut v).then_some(v));
        match rewritten {
            Some(v) => {
                out.extend_from_slice(b"data: ");
                serde_json::to_writer(&mut *out, &v).expect("serialize chunk");
            }
            None => out.extend_from_slice(line),
        }
        out.push(b'\n');
    }
}

impl SseTransducer for RenameReasoning {
    fn feed(&mut self, chunk: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(chunk.len());
        self.lines.push(chunk, |l| Self::line(l, &mut out));
        out
    }

    fn finish(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        self.lines.flush(|l| Self::line(l, &mut out));
        out
    }

    fn fail(&mut self, _: &str) -> Vec<u8> {
        Vec::new()
    }

    fn is_terminal(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renames_in_messages_and_stream_deltas() {
        let mut v = json!({"choices": [{"message": {"content": "a", "reasoning_content": "r"}}]});
        assert!(rename_reasoning(&mut v));
        assert_eq!(
            v["choices"][0]["message"],
            json!({"content": "a", "reasoning": "r"})
        );

        let mut s = RenameReasoning::default();
        let input = b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hm\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\ndata: [DONE]\n\n";
        let (a, b) = input.split_at(20);
        let mut out = s.feed(a);
        out.extend(s.feed(b));
        out.extend(s.finish());
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.starts_with("data: {\"choices\":[{\"delta\":{\"reasoning\":\"hm\"}}]}\n\n"),
            "{text}"
        );
        assert!(
            text.ends_with(
                "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\ndata: [DONE]\n\n"
            ),
            "{text}"
        );
    }
}
