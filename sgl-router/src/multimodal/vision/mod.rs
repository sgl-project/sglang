//! Pure Rust vision processing module for multimodal models.
//!
//! This module provides image preprocessing pipelines that match HuggingFace processor
//! outputs without requiring Python dependencies.
//!
//! # Architecture
//!
//! The vision module is structured as follows:
//!
//! - `transforms`: Core image transformations (resize, normalize, crop, etc.)
//! - `preprocessor_config`: HuggingFace config parsing
//! - `image_processor`: Trait and output types for processors
//!
//! Model-specific processors will be added in Phase 2.
//!
//! # Usage
//!
//! ```rust,ignore
//! use sgl_model_gateway::multimodal::vision::{
//!     PreProcessorConfig,
//!     transforms,
//! };
//!
//! // Load config from HuggingFace
//! let config = PreProcessorConfig::from_json(config_json)?;
//!
//! // Use transforms directly
//! let tensor = transforms::to_tensor(&image);
//! transforms::normalize(&mut tensor, &mean, &std);
//! ```

pub mod image_processor;
pub mod preprocessor_config;
pub mod transforms;

// Re-export commonly used types
pub use image_processor::{ImagePreProcessor, ModelSpecificValue, PreprocessedImages};
pub use preprocessor_config::PreProcessorConfig;
pub use transforms::TransformError;
