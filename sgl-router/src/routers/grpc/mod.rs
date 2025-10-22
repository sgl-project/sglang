//! gRPC router implementations

use crate::{grpc_client::proto, protocols::common::StringOrArray};

pub mod context;
pub mod pd_router;
pub mod pipeline;
pub mod processing;
pub mod responses;
pub mod router;
pub mod streaming;
pub mod utils;

/// Processed chat messages ready for gRPC generation
#[derive(Debug)]
pub struct ProcessedMessages {
    pub text: String,
    pub multimodal_inputs: Option<proto::MultimodalInputs>,
    pub stop_sequences: Option<StringOrArray>,
}
