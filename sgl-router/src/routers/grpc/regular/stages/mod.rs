//! Pipeline stages for regular (non-harmony) model processing
//!
//! This module defines stages specific to regular tokenizer-based models.

pub mod chat;
pub mod generate;
mod preparation;
mod request_building;
mod response_processing;

// ============================================================================
// Public Exports
// ============================================================================

// Export endpoint-aware stages (used by pipeline.rs)
pub use preparation::PreparationStage;
pub use request_building::RequestBuildingStage;
pub use response_processing::ResponseProcessingStage;

// Export endpoint-specific stages (available for direct use if needed)
pub use chat::{ChatPreparationStage, ChatRequestBuildingStage, ChatResponseProcessingStage};
pub use generate::{
    GeneratePreparationStage, GenerateRequestBuildingStage, GenerateResponseProcessingStage,
};
