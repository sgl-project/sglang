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
