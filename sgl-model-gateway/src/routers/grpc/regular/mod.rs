//! Regular (non-harmony) model processing
//!
//! This module contains all code specific to regular tokenizer-based models,
//! including pipeline stages, response processing, and streaming.

pub(crate) mod processor;
pub(crate) mod responses;
pub(crate) mod stages;
pub(crate) mod streaming;
