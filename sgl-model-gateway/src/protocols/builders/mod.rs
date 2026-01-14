//! Builder patterns for protocol response types
//!
//! This module provides ergonomic builders for response types with many optional fields.
//! Builders help avoid telescoping constructors and make construction intent clear.
//!
//! # Organization
//!
//! Builders are organized by API:
//! - `chat/` - Chat Completion API builders (response, stream_response)
//! - `responses/` - Responses API builder (response)
//!
//! # Optional Fields
//!
//! For optional fields, builders provide `maybe_*` methods that handle `Option<T>` directly:
//! ```ignore
//! builder
//!     .field(value)
//!     .maybe_optional_field(optional_value)  // Accepts Option<T>
//!     .build()
//! ```

pub mod chat;
pub mod responses;

// Re-export all builders for convenient access
pub use chat::{ChatCompletionResponseBuilder, ChatCompletionStreamResponseBuilder};
pub use responses::ResponsesResponseBuilder;
