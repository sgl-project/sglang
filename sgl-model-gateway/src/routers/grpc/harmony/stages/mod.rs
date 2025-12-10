//! Harmony-specific pipeline stages
//!
//! These stages replace their regular counterparts in the Harmony pipeline:
//! - HarmonyPreparationStage: Harmony encoding instead of chat template + tokenization
//! - HarmonyRequestBuildingStage: Token-based request building
//! - HarmonyResponseProcessingStage: Harmony channel parsing

pub mod preparation;
pub mod request_building;
pub mod response_processing;

pub use preparation::HarmonyPreparationStage;
pub use request_building::HarmonyRequestBuildingStage;
pub use response_processing::HarmonyResponseProcessingStage;
