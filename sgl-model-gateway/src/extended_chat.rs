//! SGLang-specific chat-completion fields that aren't modeled in the upstream
//! `openai-protocol` crate.
//!
//! **HTTP-only by design.** These extras reach the sglang backend only through
//! the HTTP transports (`http/router.rs`, `http/pd_router.rs`) because those
//! paths re-serialize the whole wrapper to JSON and let the Python endpoint
//! (`python/sglang/srt/entrypoints/openai/protocol.py` +
//! `serving_chat.py::_convert_to_internal_format`) turn them into
//! `GenerateReqInput` for the engine.
//!
//! The gRPC path forwards `&body.inner` and drops these fields intentionally:
//! the scheduler protobuf `GenerateRequest`
//! (`python/sglang/srt/grpc/sglang_scheduler_pb2.pyi:80-116`) has no slots for
//! `return_routed_experts`, `return_cached_tokens_details`,
//! `return_prompt_token_ids`, `return_meta_info`, and treats `input_ids` with
//! different semantics (gateway-side tokenization rather than a client bypass).
//! Supporting gRPC here would require protobuf schema changes and a tokenize
//! pipeline rework, which is out of scope.
//!
//! Scope discipline: **only fields that `serving_chat._convert_to_internal_format`
//! already consumes end-to-end** belong here. Do not add a new field without
//! walking the Python path first and weighing the PD implications.

use std::ops::{Deref, DerefMut};

use openai_protocol::validated::Normalizable;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::protocols::chat::ChatCompletionRequest;
use crate::protocols::common::GenerationRequest;

/// `ChatCompletionRequest` augmented with SGLang-specific fields that the
/// Python endpoint understands but the upstream `openai-protocol` crate does
/// not model.  See the module doc for the transport-level guarantees.
///
/// `return_hidden_states` is **not** redeclared here because
/// `openai_protocol::ChatCompletionRequest` already defines it
/// (`src/chat.rs:357`); adding it here would create a duplicate-key clash
/// under `#[serde(flatten)]`.
#[derive(Debug, Clone, Default, Deserialize, Serialize, Validate)]
pub struct ExtendedChatCompletionRequest {
    #[serde(flatten)]
    #[validate(nested)]
    pub inner: ChatCompletionRequest,

    // Python protocol mirror: python/sglang/srt/entrypoints/openai/protocol.py:566-569.
    // All five are forwarded to GenerateReqInput (serving_chat.py:307/318) or
    // used at response-formatting time (serving_chat.py:985-990). PD-safe:
    // prefill and decode receive the same JSON body, so the engines agree on
    // behavior.
    #[serde(default, skip_serializing_if = "is_false")]
    pub return_routed_experts: bool,

    #[serde(default, skip_serializing_if = "is_false")]
    pub return_cached_tokens_details: bool,

    #[serde(default, skip_serializing_if = "is_false")]
    pub return_prompt_token_ids: bool,

    #[serde(default, skip_serializing_if = "is_false")]
    pub return_meta_info: bool,

    /// Pre-tokenized prompt. When set, `serving_chat.py:361-364` bypasses chat
    /// template tokenization and uses these ids as `prompt_ids` directly.
    /// Python types it as `Optional[List[int]]` (single sequence only); batched
    /// tokenization goes through `/generate`, not chat.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_ids: Option<Vec<i64>>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl Deref for ExtendedChatCompletionRequest {
    type Target = ChatCompletionRequest;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for ExtendedChatCompletionRequest {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Normalizable for ExtendedChatCompletionRequest {
    fn normalize(&mut self) {
        self.inner.normalize();
    }
}

impl GenerationRequest for ExtendedChatCompletionRequest {
    fn is_stream(&self) -> bool {
        self.inner.is_stream()
    }

    fn get_model(&self) -> Option<&str> {
        self.inner.get_model()
    }

    fn extract_text_for_routing(&self) -> String {
        self.inner.extract_text_for_routing()
    }
}
