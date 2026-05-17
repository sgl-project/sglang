//! Pipeline stages for classify requests.
//!
//! Classify reuses embedding stages for preparation and request building,
//! as the scheduler treats classify as an embedding request and returns logits.
//! Only response processing is classify-specific (softmax + label mapping).

pub(crate) mod response_processing;

pub(crate) use response_processing::ClassifyResponseProcessingStage;
