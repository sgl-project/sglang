//! Pipeline stages for classify requests.
//!
//! Classify reuses embedding stages for preparation and request building,
//! as the scheduler treats classify as an embedding request and returns logits.
//! Only response processing is classify-specific (softmax + label mapping).

pub mod response_processing;

pub use response_processing::ClassifyResponseProcessingStage;
