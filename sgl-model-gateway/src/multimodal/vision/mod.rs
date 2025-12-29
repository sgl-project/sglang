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
//! - `processors`: Model-specific implementations (LLaVA, Qwen-VL, etc.)
//!
//! # Usage
//!
//! ```rust,ignore
//! use sgl_model_gateway::multimodal::vision::{
//!     PreProcessorConfig,
//!     processors::LlavaProcessor,
//!     ImagePreProcessor,
//! };
//!
//! // Load config from HuggingFace
//! let config = PreProcessorConfig::from_json(config_json)?;
//!
//! // Create processor and preprocess images
//! let processor = LlavaProcessor::new();
//! let result = processor.preprocess(&images, &config)?;
//! ```

pub mod image_processor;
pub mod preprocessor_config;
pub mod processors;
pub mod transforms;

// Re-export commonly used types
pub use image_processor::{
    ImagePreProcessor, ImageProcessorRegistry, ModelSpecificValue, PreprocessedImages,
};
pub use preprocessor_config::PreProcessorConfig;
pub use processors::{
    Llama4VisionProcessor, LlavaNextProcessor, LlavaProcessor, Phi3VisionProcessor,
    Phi4VisionProcessor, PixtralProcessor, Qwen2VLProcessor, Qwen3VLProcessor,
};
pub use transforms::TransformError;
