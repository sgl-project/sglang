pub mod sglang_scheduler;
pub mod vllm_engine;

// Export both clients
// Re-export proto modules with explicit names
pub use sglang_scheduler::proto as sglang_proto;
// Keep backward compatibility: proto = sglang_proto (default)
pub use sglang_scheduler::proto;
pub use sglang_scheduler::SglangSchedulerClient;
pub use vllm_engine::{proto as vllm_proto, VllmEngineClient};
