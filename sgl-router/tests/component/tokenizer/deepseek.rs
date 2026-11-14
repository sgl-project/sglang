// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;
use sgl_router::tokenizer::chat_formatter::ChatFormatter;

/// The router renders text only; chats with media go to the worker.
#[test]
fn v41_leaves_media_to_the_worker() {
    let formatter = ChatFormatter::deepseek_native(Some("deepseek_v41"), "alias").unwrap();
    let request = json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]});
    assert!(formatter.render(&request).is_err());
}

use sgl_router::config::{TokenizerBackend, TokenizerConfig};
use sgl_router::tokenizer::adapter;

/// The checkpoint's pre-tokenizer, normalizer and full added vocabulary over a
/// byte-level vocab: small enough to commit, faithful where encoders can differ.
const MINI: &str = "tests/fixtures/deepseek_v41_mini/tokenizer.json";

/// Text that stresses the boundary between the two encoders: special-token
/// spellings inside untrusted content, digits, CJK runs, the U+017F case fold,
/// emoji, zero-width and odd whitespace.
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

/// Chats in the shapes DeepSeek-V4.1 agentic traffic takes: tools, tool calls
/// and results, reasoning content, a final user turn or an assistant to continue.
fn requests() -> Vec<serde_json::Value> {
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

fn tokenizer(
    path: &str,
    backend: TokenizerBackend,
) -> std::sync::Arc<dynamo_tokenizers::Tokenizer> {
    let cfg = TokenizerConfig {
        backend,
        l1_cache_mb: 0,
    };
    adapter::load_with(path, cfg).unwrap().0
}

/// fastokens encodes every request to exactly the ids the processor's HF
/// tokenizer does.
fn assert_router_ids_match_processor(path: &str) {
    let model = "deepseek-ai/DeepSeek-V4.1-Flash";
    let processor = ChatFormatter::load(model, path).unwrap().unwrap();
    let router = ChatFormatter::load(model, path)
        .unwrap()
        .unwrap()
        .with_router_segment_encoding(true);
    let fast = tokenizer(path, TokenizerBackend::Fast);
    for (i, request) in requests().iter().enumerate() {
        let expected = processor.encode(&fast, request).unwrap();
        assert!(expected.len() > 50, "case {i} rendered almost nothing");
        assert_eq!(router.encode(&fast, request).unwrap(), expected, "case {i}");
    }
}

#[test]
fn v41_router_tokenizer_matches_processor_ids() {
    assert_router_ids_match_processor(MINI);
}

/// The same check over a real DeepSeek-V4/V4.1 `tokenizer.json`, whose full
/// merge table the mini fixture does not carry. Set
/// `SGL_ROUTER_DSV4_TOKENIZER_JSON` to the checkpoint's file to run it.
#[test]
fn v41_router_tokenizer_matches_processor_ids_on_the_checkpoint() {
    let Ok(path) = std::env::var("SGL_ROUTER_DSV4_TOKENIZER_JSON") else {
        eprintln!("skip: SGL_ROUTER_DSV4_TOKENIZER_JSON is not set");
        return;
    };
    assert_router_ids_match_processor(&path);
}

/// dynamo-tokenizers' HF backend has no segmented encode, so router encoding
/// over it fails instead of silently producing different ids. The registry
/// enables it only for `fast` (see `deepseek_encodes_on_the_router_tokenizer_only_with_fast`).
#[test]
fn v41_router_segment_encoding_fails_on_the_hf_backend() {
    let router = ChatFormatter::load("deepseek-ai/DeepSeek-V4.1-Flash", MINI)
        .unwrap()
        .unwrap()
        .with_router_segment_encoding(true);
    let hf = tokenizer(MINI, TokenizerBackend::Hf);
    assert!(router.encode(&hf, &requests()[0]).is_err());
}
