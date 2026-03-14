//! Anthropic Messages API routing.
//!
//! The SGLang inference engine natively supports the Anthropic Messages API
//! at `/v1/messages` (added in PR #18630). This module lets the gateway
//! accept those requests and proxy them **directly** to backend workers
//! without any protocol conversion.
//!
//! # Request flow
//!
//! ```text
//! Client  ──POST /v1/messages──►  Gateway
//!                                   │  extract `model` for worker selection
//!                                   │  forward raw body unchanged
//!                                   ▼
//!                               Backend worker
//!                               POST /v1/messages   (SGLang native)
//!                                   │
//!                                   ▼  Anthropic response (unchanged)
//! Client  ◄────────────────────  Gateway
//! ```

pub mod protocol;
pub mod router;

pub use protocol::{AnthropicCountTokensRequest, AnthropicMessagesRequest};
