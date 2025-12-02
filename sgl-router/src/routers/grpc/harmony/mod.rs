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
//! use sgl_model_gateway::routers::grpc::harmony::{HarmonyDetector, HarmonyBuilder};
//!
//! // Detect if model supports Harmony
//! if HarmonyDetector::is_harmony_model("gpt-4o") {
//!     // Build Harmony request
//!     let builder = HarmonyBuilder::new();
//!     let output = builder.build_from_chat(&request)?;
//!     // ... use output.input_ids for gRPC request
//! }
//! ```

pub mod builder;
pub mod detector;
pub mod parser;
pub mod processor;
pub mod responses;
pub mod stages;
pub mod streaming;
pub mod types;

// Re-export main types for convenience
pub use builder::HarmonyBuilder;
pub use detector::HarmonyDetector;
pub use parser::HarmonyParserAdapter;
pub use processor::{HarmonyResponseProcessor, ResponsesIterationResult};
pub use responses::{
    serve_harmony_responses, serve_harmony_responses_stream, HarmonyResponsesContext,
};
pub use stages::{
    HarmonyPreparationStage, HarmonyRequestBuildingStage, HarmonyResponseProcessingStage,
};
pub use streaming::HarmonyStreamingProcessor;
pub use types::{
    FunctionDelta, HarmonyBuildOutput, HarmonyChannelDelta, HarmonyChannelOutput, HarmonyMessage,
    ToolCallDelta,
};
