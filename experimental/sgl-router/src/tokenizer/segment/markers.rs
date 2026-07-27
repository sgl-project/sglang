// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Derive segment **markers** by probe-rendering a chat encoder.
//!
//! Encoder-agnostic: the caller passes a `render` closure (the model's actual
//! chat encoder — a Jinja template, DeepSeek-V4's built-in code encoder, etc.),
//! and we probe-render a conversation whose message contents are unique
//! sentinels (containing no special tokens). Every special token that appears in
//! the output must therefore be template *framing* → a marker. Splitting the
//! rendered prompt at markers yields the cache reuse unit; markers are
//! `AddedToken`s = HuggingFace hard split boundaries, so splitting there is
//! byte-exact. [`super::SegmentCache`] then runs a startup self-check
//! (concat-of-segments == whole) with the real tokenizer before trusting them.
//!
//! Role-agnostic: we do NOT categorize turns / tool-calls / roles — a marker is
//! just any structural special token the encoder emits.

use serde_json::Value;

/// Result of analyzing an encoder.
pub struct Segmenter {
    /// Turn-boundary marker strings (special tokens), longest first.
    pub markers: Vec<String>,
    /// False if no structural markers were found (caller must not segment).
    pub segmentable: bool,
}

/// Derive segment markers: probe-render (via `render`) with sentinel contents,
/// then keep every special token from `specials` that shows up in the output (it
/// must be framing, since the sentinels contain none). `render` takes a chat
/// `messages` array and returns the rendered prompt, or `None` if it can't
/// render that shape (e.g. a text-only template handed multimodal content — that
/// probe is simply skipped).
pub fn derive_markers<F>(render: F, specials: &[String]) -> Result<Segmenter, String>
where
    F: Fn(&Value) -> Option<String>,
{
    let s = |i: usize| format!("\u{E000}MT{i}\u{E001}");
    // Probe conversation covering system/user/assistant AND a tool interaction
    // (assistant tool_calls + a tool-output message) so tool framing special
    // tokens are emitted and captured as markers. Encoders that ignore tools
    // simply emit nothing extra.
    let probe = serde_json::json!([
        { "role": "system", "content": s(0) },
        { "role": "user", "content": s(1) },
        { "role": "assistant", "content": s(2) },
        { "role": "user", "content": s(3) },
        { "role": "assistant", "content": "",
          "tool_calls": [ { "type": "function",
              "function": { "name": s(4), "arguments": "{}" } } ] },
        { "role": "tool", "content": s(5) },
        { "role": "assistant", "content": s(6) },
    ]);
    let mut rendered = render(&probe).ok_or_else(|| "probe render failed".to_string())?;

    // Best-effort multimodal probe: a user message whose content is a list of
    // content-parts (image + text), the common HF multimodal format. Makes the
    // encoder emit vision/image framing tokens (<|vision_start|>, <|image_pad|>,
    // ...) so they're captured too. Encoders that don't handle list content
    // return None here and are skipped, leaving the text markers unaffected.
    let mm = serde_json::json!([
        { "role": "user", "content": [
            { "type": "image" },
            { "type": "text", "text": s(7) },
        ] },
    ]);
    if let Some(r) = render(&mm) {
        rendered.push_str(&r);
    }

    // Validate the plain-content sentinels survived (0..3 always present).
    for i in 0..4 {
        if !rendered.contains(&s(i)) {
            return Err(format!("sentinel {i} missing from probe render"));
        }
    }

    let mut markers: Vec<String> = specials
        .iter()
        .filter(|t| !t.is_empty() && rendered.contains(t.as_str()))
        .cloned()
        .collect();
    // longest first so split matches the longest marker at a position.
    markers.sort_by_key(|m| std::cmp::Reverse(m.len()));
    markers.dedup();
    let segmentable = !markers.is_empty();
    Ok(Segmenter {
        markers,
        segmentable,
    })
}

/// Split `text` at marker boundaries; each segment starts with its leading
/// marker. Markers must be BPE hard boundaries (special tokens).
pub fn split_at_markers<'a>(text: &'a str, markers: &[String]) -> Vec<&'a str> {
    if markers.is_empty() {
        return if text.is_empty() { vec![] } else { vec![text] };
    }
    let b = text.as_bytes();
    let mut cuts = vec![0usize];
    let mut i = 0;
    let mut prev_end = usize::MAX; // byte end of the previously matched marker
    let mut prev_marker = usize::MAX; // its index in `markers`
    while i < b.len() {
        let mut matched = false;
        for (mi, m) in markers.iter().enumerate() {
            let mb = m.as_bytes();
            if !mb.is_empty() && b[i..].starts_with(mb) {
                // Coalesce a run of the SAME marker (e.g. multimodal placeholders
                // like <|image_pad|> repeated thousands of times) into ONE
                // segment: only cut when this occurrence isn't contiguous with an
                // identical one. Byte-exact (the run still encodes token-per-
                // token via added-vocab); avoids pathological over-segmentation.
                let coalesce = i == prev_end && mi == prev_marker;
                if i != 0 && !coalesce {
                    cuts.push(i);
                }
                i += mb.len();
                prev_end = i;
                prev_marker = mi;
                matched = true;
                break;
            }
        }
        if !matched {
            // advance one UTF-8 char
            let step = match b[i] {
                0x00..=0x7F => 1,
                0xC0..=0xDF => 2,
                0xE0..=0xEF => 3,
                _ => 4,
            };
            i += step.min(b.len() - i);
        }
    }
    cuts.dedup();
    let mut out = Vec::with_capacity(cuts.len());
    for w in cuts.windows(2) {
        out.push(&text[w[0]..w[1]]);
    }
    let last = *cuts.last().unwrap();
    if last < text.len() {
        out.push(&text[last..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in Qwen-style encoder: wraps each message in `<|im_start|>` /
    /// `<|im_end|>` framing. Exercises `derive_markers` without a real template.
    fn qwen_render(msgs: &Value) -> Option<String> {
        let mut s = String::new();
        for m in msgs.as_array()? {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let c = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
            s.push_str(&format!("<|im_start|>{role}\n{c}<|im_end|>\n"));
        }
        Some(s)
    }

    #[test]
    fn derives_only_emitted_markers() {
        let specials = vec![
            "<|im_start|>".to_string(),
            "<|im_end|>".to_string(),
            "<|endoftext|>".to_string(), // in vocab but never emitted → not a marker
        ];
        let seg = derive_markers(qwen_render, &specials).unwrap();
        assert!(seg.segmentable);
        assert!(seg.markers.contains(&"<|im_start|>".to_string()));
        assert!(seg.markers.contains(&"<|im_end|>".to_string()));
        assert!(!seg.markers.contains(&"<|endoftext|>".to_string()));
    }

    #[test]
    fn dsv4_style_markers_derived() {
        // DeepSeek-V4 code encoder emits these; all are added tokens.
        fn dsv4_render(msgs: &Value) -> Option<String> {
            let mut s = String::from("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>");
            for m in msgs.as_array()? {
                let c = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
                s.push_str("<\u{ff5c}User\u{ff5c}>");
                s.push_str(c);
                s.push_str("<\u{ff5c}Assistant\u{ff5c}></think>");
            }
            Some(s)
        }
        let specials = vec![
            "<\u{ff5c}User\u{ff5c}>".to_string(),
            "<\u{ff5c}Assistant\u{ff5c}>".to_string(),
            "</think>".to_string(),
        ];
        let seg = derive_markers(dsv4_render, &specials).unwrap();
        assert!(seg.segmentable);
        assert!(seg.markers.contains(&"<\u{ff5c}User\u{ff5c}>".to_string()));
        assert!(seg.markers.contains(&"</think>".to_string()));
    }

    #[test]
    fn no_markers_when_encoder_emits_none() {
        fn plain(msgs: &Value) -> Option<String> {
            let mut s = String::new();
            for m in msgs.as_array()? {
                s.push_str(m.get("content").and_then(|c| c.as_str()).unwrap_or(""));
                s.push('\n');
            }
            Some(s)
        }
        let seg = derive_markers(plain, &["<|im_start|>".to_string()]).unwrap();
        assert!(!seg.segmentable);
        assert!(seg.markers.is_empty());
    }

    #[test]
    fn split_starts_each_segment_with_its_marker() {
        let markers = vec!["<|im_start|>".to_string(), "<|im_end|>".to_string()];
        let text = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
        let segs = split_at_markers(text, &markers);
        assert_eq!(segs.concat(), text);
        assert!(segs[0].starts_with("<|im_start|>"));
        assert!(segs.iter().skip(1).all(|s| s.starts_with("<|im")));
    }

    #[test]
    fn split_no_markers_returns_whole() {
        assert_eq!(split_at_markers("hello", &[]), vec!["hello"]);
        assert!(split_at_markers("", &[]).is_empty());
    }

    #[test]
    fn split_coalesces_same_marker_run() {
        let markers = vec!["<|image_pad|>".to_string()];
        let text = "a<|image_pad|><|image_pad|><|image_pad|>b";
        let segs = split_at_markers(text, &markers);
        assert_eq!(segs.concat(), text);
        assert_eq!(segs, vec!["a", "<|image_pad|><|image_pad|><|image_pad|>b"]);
    }
}
