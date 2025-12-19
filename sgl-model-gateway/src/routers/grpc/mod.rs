//! gRPC router implementations

use crate::{grpc_client::sglang_proto::MultimodalInputs, protocols::common::StringOrArray};

pub mod client;
pub mod common;
pub mod context;
pub mod harmony;
pub mod pd_router;
pub mod pipeline;
pub mod proto_wrapper;
pub mod regular;
pub mod router;
pub mod utils;

/// Processed chat messages ready for gRPC generation
#[derive(Debug)]
pub struct ProcessedMessages {
    pub text: String,
    pub multimodal_inputs: Option<MultimodalInputs>,
    pub stop_sequences: Option<StringOrArray>,
}
