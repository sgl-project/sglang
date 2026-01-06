//! Harmony pipeline implementation
//!
//! This module provides support for GPT-OSS models that use Harmony encoding/parsing.
//! The Harmony protocol uses a channel-based approach with three channels:
//! - **analysis**: Reasoning/thinking content (optional)
//! - **commentary**: Tool calls (optional)
//! - **final**: Final response text (required)
//!
//! ## Architecture
//!
//! The Harmony implementation is structured as follows:
//!
//! - **detector**: Model detection (is this a Harmony-capable model?)
//! - **builder**: Request encoding (convert Chat/Responses → input_ids)
//! - **parser**: Response parsing (output_ids → channels)
//! - **types**: Shared type definitions
//!
//! ## Usage
//!
//! ```ignore
//! use smg::routers::grpc::harmony::{HarmonyDetector, HarmonyBuilder};
//!
//! // Detect if model supports Harmony
//! if HarmonyDetector::is_harmony_model("gpt-4o") {
//!     // Build Harmony request
//!     let builder = HarmonyBuilder::new();
//!     let output = builder.build_from_chat(&request)?;
//!     // ... use output.input_ids for gRPC request
//! }
//! ```

pub(crate) mod builder;
pub(crate) mod detector;
pub(crate) mod parser;
pub(crate) mod processor;
pub(crate) mod responses;
pub(crate) mod stages;
pub(crate) mod streaming;
pub(crate) mod types;

// Re-export types that are accessed via harmony::TypeName
pub(crate) use builder::HarmonyBuilder;
pub(crate) use detector::HarmonyDetector;
pub(crate) use parser::HarmonyParserAdapter;
pub(crate) use processor::{HarmonyResponseProcessor, ResponsesIterationResult};
pub(crate) use responses::{
    serve_harmony_responses, serve_harmony_responses_stream, HarmonyResponsesContext,
};
pub(crate) use streaming::HarmonyStreamingProcessor;
pub(crate) use types::HarmonyMessage;
