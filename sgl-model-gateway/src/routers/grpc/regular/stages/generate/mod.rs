//! Generate endpoint pipeline stages
//!
//! These stages handle generate-specific preprocessing, request building, and response processing.
//! They work with any model type by using injected model adapters.

mod preparation;
mod request_building;
mod response_processing;

pub(crate) use preparation::GeneratePreparationStage;
pub(crate) use request_building::GenerateRequestBuildingStage;
pub(crate) use response_processing::GenerateResponseProcessingStage;
