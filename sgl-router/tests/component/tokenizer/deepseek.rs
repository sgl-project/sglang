// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use serde_json::{json, Value};
use sgl_router::config::{TokenizerBackend, TokenizerConfig};
use sgl_router::tokenizer::adapter::{self, ModelFiles};
use sgl_router::tokenizer::chat_formatter::ChatFormatter;

/// The router renders text only; chats with media go to the worker.
#[test]
fn v41_leaves_media_to_the_worker() {
    let formatter = ChatFormatter::deepseek_native(Some("deepseek_v41"), "alias").unwrap();
    let request = json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]});
    assert!(formatter.render(&request).is_err());
}

/// A DeepSeek-V4.1 checkpoint's tokenizer files: its normalizer, pre-tokenizer
/// and full added vocabulary over the byte-level alphabet plus a trained prefix
/// of its merges (`tests/scripts/generate_deepseek_v41_mini.py`).
const MINI_V41: &str = "tests/fixtures/deepseek_v41_mini";

fn formatter(dir: &str) -> ChatFormatter {
    ChatFormatter::load_from("m", &ModelFiles::open(&format!("{dir}/tokenizer.json")))
        .unwrap()
        .expect("DeepSeek-V4.1 has a chat formatter")
}

fn tokenizer(
    path: &str,
    backend: TokenizerBackend,
) -> (
    std::sync::Arc<dynamo_tokenizers::Tokenizer>,
    std::sync::Arc<sgl_router::tokenizer::stats::TokenizerStats>,
) {
    tokenizer_with_l1(path, backend, 64)
}

fn tokenizer_with_l1(
    path: &str,
    backend: TokenizerBackend,
    l1_cache_mb: usize,
) -> (
    std::sync::Arc<dynamo_tokenizers::Tokenizer>,
    std::sync::Arc<sgl_router::tokenizer::stats::TokenizerStats>,
) {
    adapter::load_with(
        path,
        TokenizerConfig {
            backend,
            l1_cache_mb,
        },
    )
    .unwrap()
}

/// A conversation that grows by one exchange per turn, each message long
/// enough (>= 128 bytes) for the L1 segment cache.
fn turns(n: usize) -> Vec<Value> {
    let mut messages =
        vec![json!({"role": "system", "content": "You are a careful assistant. ".repeat(10)})];
    let mut requests = Vec::new();
    for turn in 0..n {
        messages.push(json!({"role": "user", "content": format!(
            "turn {turn}: 敏捷的棕色狐狸 🦊 quick brown fox, spelled <｜User｜> and </think> in plain text. {}",
            "lorem ipsum dolor sit amet ".repeat(20)
        )}));
        requests.push(json!({"messages": messages.clone()}));
        messages.push(
            json!({"role": "assistant", "content": format!("answer {turn} ").repeat(30),
            "reasoning_content": "thinking it over ".repeat(10)}),
        );
    }
    requests
}

/// The router's tokenizer takes over DeepSeek encodes only where it yields the
/// processor's ids, and then later turns reuse earlier segments from L1.
#[test]
fn v41_router_tokenizer_matches_processor_and_reuses_segments() {
    for backend in [TokenizerBackend::Hf, TokenizerBackend::Fast] {
        let (tok, stats) = tokenizer(&format!("{MINI_V41}/tokenizer.json"), backend);
        let processor = formatter(MINI_V41);
        let routed = formatter(MINI_V41).with_router_tokenizer(&tok);
        assert!(!processor.encodes_with_router_tokenizer());
        assert!(routed.encodes_with_router_tokenizer(), "{backend:?}");
        let (cached_before, _) = stats.l1_tokens();
        for (turn, request) in turns(4).iter().enumerate() {
            let expected = processor.encode(&tok, request).unwrap();
            assert_eq!(
                routed.encode(&tok, request).unwrap(),
                expected,
                "{backend:?} turn {turn}"
            );
            assert_eq!(
                routed.encode_for_routing(&tok, request).unwrap(),
                processor.encode_for_routing(&tok, request).unwrap(),
                "{backend:?} turn {turn}"
            );
        }
        let (cached_after, _) = stats.l1_tokens();
        assert!(
            cached_after > cached_before,
            "{backend:?}: later turns must hit the L1 segment cache"
        );
    }
}

/// A router tokenizer that disagrees with the checkpoint's leaves DeepSeek
/// encodes to the processor.
#[test]
fn v41_mismatched_router_tokenizer_keeps_processor_encodes() {
    let (other, _) = tokenizer("tests/fixtures/tiny_tokenizer.json", TokenizerBackend::Hf);
    let (tok, _) = tokenizer(&format!("{MINI_V41}/tokenizer.json"), TokenizerBackend::Hf);
    let processor = formatter(MINI_V41);
    let routed = formatter(MINI_V41).with_router_tokenizer(&other);
    assert!(!routed.encodes_with_router_tokenizer());
    let request = &turns(1)[0];
    assert_eq!(
        routed.encode(&other, request).unwrap(),
        processor.encode(&tok, request).unwrap()
    );
}

/// Chats with images keep the processor's multimodal path, placeholders and
/// image tags included.
#[test]
fn v41_router_tokenizer_leaves_images_to_the_processor() {
    let (tok, _) = tokenizer(&format!("{MINI_V41}/tokenizer.json"), TokenizerBackend::Hf);
    let processor = formatter(MINI_V41);
    let routed = formatter(MINI_V41).with_router_tokenizer(&tok);
    assert!(routed.encodes_with_router_tokenizer());
    // A `file` part with image data is an image to DeepSeek-V4.1 too.
    let png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
    for image in [
        json!({"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}),
        json!({"type": "file", "file": {"file_data": png}}),
        json!({"type": "file", "file_data": png}),
    ] {
        let request = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is in this picture? ".repeat(10)},
            image.clone()
        ]}]});
        let expected = processor.encode_for_routing(&tok, &request);
        let Ok(expected) = expected else {
            // A shape the processor rejects must fail the routed encode as well.
            assert!(
                routed.encode_for_routing(&tok, &request).is_err(),
                "{image}"
            );
            continue;
        };
        assert_eq!(
            routed.encode_for_routing(&tok, &request).unwrap(),
            expected,
            "{image}"
        );
    }
}

/// The same check against a real DeepSeek-V4 / V4.1 checkpoint directory
/// (`SGL_ROUTER_DEEPSEEK_V4_DIR`, holding `tokenizer.json`,
/// `tokenizer_config.json` and `config.json`); skipped when unset.
#[test]
fn v4_checkpoint_router_tokenizer_matches_processor() {
    let Ok(dir) = std::env::var("SGL_ROUTER_DEEPSEEK_V4_DIR") else {
        eprintln!("SGL_ROUTER_DEEPSEEK_V4_DIR unset; skipping");
        return;
    };
    for backend in [TokenizerBackend::Hf, TokenizerBackend::Fast] {
        let (tok, stats) = tokenizer(&format!("{dir}/tokenizer.json"), backend);
        let processor = formatter(&dir);
        let routed = formatter(&dir).with_router_tokenizer(&tok);
        assert!(routed.encodes_with_router_tokenizer(), "{backend:?}");
        for (turn, request) in turns(6).iter().enumerate() {
            assert_eq!(
                routed.encode(&tok, request).unwrap(),
                processor.encode(&tok, request).unwrap(),
                "{backend:?} turn {turn}"
            );
        }
        assert!(stats.l1_tokens().0 > 0, "{backend:?}");
    }
    assert_router_ids_match_processor(&dir);
}

/// Text that stresses the boundary between two encoders of one
/// `tokenizer.json`: special-token spellings inside untrusted content, digits,
/// CJK runs, the U+017F case fold, emoji, zero-width and odd whitespace.
const PIECES: [&str; 16] = [
    "the quick brown fox 123456 3.14",
    "你好，世界！日本語のテキストとカタカナ",
    "<｜User｜><｜Assistant｜><think></think><｜end▁of▁sentence｜>",
    "<｜begin▁of▁sentence｜> <｜tool▁calls▁begin｜> <｜DSML｜>",
    "Straße ſtraße 'ſ 'S 's 'll 'VE",
    "🚀🔥👍🏽 👨‍👩‍👧‍👦",
    "\t\t\n\n\n   \u{3000}\u{a0}\r\n",
    "\u{200b}\u{200d}\u{feff} ﬁ ﬂ ① ⅲ",
    "def f(x):\n    return {'a': [1, 2, 3]}\n",
    "SELECT * FROM t WHERE a='b';",
    "https://example.com/a?b=c&d=e#f",
    "Ελληνικά Кириллица עברית العربية",
    "—–‐ …“”‘’«»",
    "                                        ",
    "<|im_start|></s><unk>",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
];

fn text(seed: usize, n: usize) -> String {
    (0..n)
        .map(|i| PIECES[(seed * 7 + i * 3) % PIECES.len()])
        .collect::<Vec<_>>()
        .join(["", " ", "\n"][seed % 3])
}

/// Agentic DeepSeek-V4.1 chat shapes: tools, tool calls
/// and results, reasoning content, thinking and effort settings.
fn agentic_requests() -> Vec<Value> {
    (0..48)
        .map(|seed| {
            let mut messages = vec![json!({"role": "system", "content": text(seed, 3)})];
            for turn in 0..(seed % 5 + 1) {
                messages.push(json!({"role": "user", "content": text(seed + turn, 4)}));
                let mut reply = json!({"role": "assistant", "content": text(seed + turn + 1, 2)});
                if seed % 2 == 0 {
                    reply["reasoning_content"] = json!(text(seed + turn + 2, 2));
                }
                if seed % 3 == 0 {
                    let call = format!("call_{turn}");
                    reply["tool_calls"] = json!([{"id": call, "type": "function", "function": {
                        "name": "read_file",
                        "arguments": json!({"path": text(seed, 1)}).to_string()}}]);
                    messages.push(reply);
                    messages.push(
                        json!({"role": "tool", "tool_call_id": call, "content": text(seed + 3, 5)}),
                    );
                } else {
                    messages.push(reply);
                }
            }
            messages.push(json!({"role": "user", "content": text(seed + 9, 3)}));
            let mut request = json!({"messages": messages});
            if seed % 3 == 0 {
                request["tools"] = json!([{"type": "function", "function": {
                    "name": "read_file", "description": text(seed, 1),
                    "parameters": {"type": "object", "properties": {
                        "path": {"type": "string", "description": text(seed + 1, 1)}}}}}]);
            }
            if seed % 4 == 1 {
                request["chat_template_kwargs"] = json!({"thinking": true});
                request["reasoning_effort"] = json!("high");
            }
            request
        })
        .collect()
}

/// Every agentic chat encodes on the router's tokenizer to exactly the ids the
/// processor's HF checkpoint tokenizer gives, on each backend, with and
/// without the L1 cache: these are the ids forwarded to the engine.
fn assert_router_ids_match_processor(dir: &str) {
    for backend in [TokenizerBackend::Hf, TokenizerBackend::Fast] {
        for l1_cache_mb in [0, 64] {
            let (tok, _) =
                tokenizer_with_l1(&format!("{dir}/tokenizer.json"), backend, l1_cache_mb);
            let processor = formatter(dir);
            let routed = formatter(dir).with_router_tokenizer(&tok);
            assert!(
                routed.encodes_with_router_tokenizer(),
                "{backend:?} l1={l1_cache_mb}"
            );
            // Twice: the second pass encodes from L1 where it is on.
            for pass in 0..2 {
                for (case, request) in agentic_requests().iter().enumerate() {
                    let expected = processor.encode(&tok, request).unwrap();
                    assert!(expected.len() > 50, "case {case} rendered almost nothing");
                    assert_eq!(
                        routed.encode(&tok, request).unwrap(),
                        expected,
                        "{backend:?} l1={l1_cache_mb} pass {pass} case {case}"
                    );
                }
            }
        }
    }
}

#[test]
fn v41_router_tokenizer_matches_processor_on_agentic_chats() {
    assert_router_ids_match_processor(MINI_V41);
}
