//! Harmony-specific pipeline stages
//!
//! These stages replace their regular counterparts in the Harmony pipeline:
//! - HarmonyPreparationStage: Harmony encoding instead of chat template + tokenization
//! - HarmonyRequestBuildingStage: Token-based request building
//! - HarmonyResponseProcessingStage: Harmony channel parsing

pub(crate) mod preparation;
pub(crate) mod request_building;
pub(crate) mod response_processing;

pub(crate) use preparation::HarmonyPreparationStage;
pub(crate) use request_building::HarmonyRequestBuildingStage;
pub(crate) use response_processing::HarmonyResponseProcessingStage;
