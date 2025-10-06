//! gRPC router implementations

use crate::grpc_client::proto;
use crate::protocols::spec::StringOrArray;

pub mod pd_router;
pub mod router;
pub mod utils;

/// Processed chat messages ready for gRPC generation
#[derive(Debug)]
pub struct ProcessedMessages {
    pub text: String,
    pub multimodal_inputs: Option<proto::MultimodalInputs>,
    pub stop_sequences: Option<StringOrArray>,
}
