//! gRPC router implementations

use crate::{grpc_client::proto, protocols::common::StringOrArray};

pub mod common;
pub mod context;
pub mod error;
pub mod harmony;
pub mod pd_router;
pub mod pipeline;
pub mod regular;
pub mod router;
pub mod utils;

/// Processed chat messages ready for gRPC generation
#[derive(Debug)]
pub struct ProcessedMessages {
    pub text: String,
    pub multimodal_inputs: Option<proto::MultimodalInputs>,
    pub stop_sequences: Option<StringOrArray>,
}
