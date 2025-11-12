pub mod common;
pub mod sglang_scheduler;
pub mod vllm_engine;

// Export common utilities
pub use common::{
    convert_grpc_to_http,
    create_grpc_channel,
    AbortOnDropStream,
    AbortableClient,
    // Unified types for enum dispatch
    ChunkData,
    // Unified proto types (router should use these instead of sglang_proto::*)
    Complete,
    CompleteData,
    GenerateResponse,
    GrpcClient,
    GrpcStream,
    HiddenStates,
    InputLogProbs,
    MatchedStop,
    MultimodalInputs,
    OutputLogProbs,
    Request,
    ResponseType,
};
// Export both clients
// Re-export proto modules with explicit names
pub use sglang_scheduler::{proto as sglang_proto, SglangSchedulerClient};
pub use vllm_engine::{proto as vllm_proto, VllmEngineClient};
