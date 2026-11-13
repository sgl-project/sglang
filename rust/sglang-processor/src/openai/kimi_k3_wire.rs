//! The Kimi-K3 shape of `/v1/chat/completions` responses (`kimi_k3_wire.py`),
//! which a server with the `kimi_k3` chat encoding answers chat in. Its usage
//! headers (`SGLANG_USAGE_HEADER_PREFIX`, unset by default) are not sent here,
//! and a stream asking for `internal_content` goes to the engine.

use std::collections::HashSet;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::wire::Reply;

/// `KimiK3WireResponse`'s state for one response.
#[derive(Debug, Default)]
pub(super) struct KimiK3Wire {
    id: Option<String>,
    /// Choices that streamed reasoning, and those whose reasoning end went out.
    reasoning_seen: HashSet<u64>,
    boundary_sent: HashSet<u64>,
}

impl KimiK3Wire {
    /// `response_id`: fixed by the first rid, since n>1 gives each choice its
    /// own; the leading 4 bytes of Unix seconds make ids sort by creation time.
    pub(super) fn response_id(&mut self, created: u64, rid: &str) -> &str {
        self.id.get_or_insert_with(|| {
            let digest = Sha256::digest(rid.as_bytes());
            let hex: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
            format!("chatcmpl-{:08x}{hex}", created & 0xFFFF_FFFF)
        })
    }

    /// The id once a frame set it.
    pub(super) fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// `ends_reasoning`: one `reasoning_content: ""` frame per choice, once
    /// its detector has left the think channel or the stream is finishing.
    pub(super) fn ends_reasoning(
        &mut self,
        index: u64,
        reasoning: &str,
        in_reasoning: bool,
        finishing: bool,
    ) -> bool {
        if !reasoning.is_empty() {
            self.reasoning_seen.insert(index);
        }
        if !self.reasoning_seen.contains(&index)
            || self.boundary_sent.contains(&index)
            || (in_reasoning && !finishing)
        {
            return false;
        }
        self.boundary_sent.insert(index);
        true
    }
}

/// `finish_reason`: the K3 wire has no `abort`; `server_interrupted` lets a
/// client tell retryable from terminal.
pub(super) fn finish_reason(reason: &str) -> &str {
    match reason {
        "abort" => "server_interrupted",
        reason => reason,
    }
}

/// `finalize_usage`: both token-detail objects are on every K3 usage, and
/// `cached_tokens` ignores `--enable-cache-report`.
pub(super) fn finalize_usage(usage: &mut Value, cached: u64) {
    let reasoning = usage["reasoning_tokens"].clone();
    match usage.get_mut("prompt_tokens_details") {
        Some(details @ Value::Object(_)) => details["cached_tokens"] = json!(cached),
        _ => usage["prompt_tokens_details"] = json!({"cached_tokens": cached}),
    }
    usage["completion_tokens_details"] = json!({"reasoning_tokens": reasoning});
}

/// `error_body`: the error under an `error` key, where a strict client looks
/// for it; the stock body puts its fields at the top level.
pub(super) fn error_reply(reply: Reply) -> Reply {
    let error: Value = serde_json::from_str(&reply.body).expect("error replies are JSON");
    Reply {
        status: reply.status,
        body: json!({ "error": error }).to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python: `f"chatcmpl-{created & 0xFFFFFFFF:08x}{sha256(rid).hexdigest()[:16]}"`.
    #[test]
    fn response_ids_hash_the_first_rid() {
        let mut wire = KimiK3Wire::default();
        assert_eq!(
            wire.response_id(0x1_2345_6789, "r"),
            "chatcmpl-23456789454349e422f05297"
        );
        assert_eq!(
            wire.response_id(0, "other"),
            "chatcmpl-23456789454349e422f05297"
        );
    }

    #[test]
    fn usage_always_has_both_details() {
        let mut usage = json!({"prompt_tokens": 3, "prompt_tokens_details": null, "reasoning_tokens": 2, "completion_tokens_details": null});
        finalize_usage(&mut usage, 1);
        assert_eq!(usage["prompt_tokens_details"], json!({"cached_tokens": 1}));
        assert_eq!(
            usage["completion_tokens_details"],
            json!({"reasoning_tokens": 2})
        );
    }
}
