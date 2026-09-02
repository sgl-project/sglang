//! Pipeline stages for regular (non-harmony) model processing
//!
//! This module defines stages specific to regular tokenizer-based models.

pub(crate) mod chat;
pub(crate) mod classify;
pub(crate) mod embedding;
pub(crate) mod generate;
pub(crate) mod preparation;
pub(crate) mod request_building;
pub(crate) mod response_processing;

// Re-export main stages used by pipeline
pub(crate) use preparation::PreparationStage;
pub(crate) use request_building::RequestBuildingStage;
pub(crate) use response_processing::ResponseProcessingStage;
