//! Model-specific image processors.
//!
//! This module contains implementations of `ImagePreProcessor` for various
//! vision-language model families.
//!
//! # Supported Models
//!
//! - **LLaVA 1.5** (`llava`): CLIP-based preprocessing with configurable aspect ratio
//! - **LLaVA-NeXT** (`llava`): Multi-crop anyres processing

pub mod llava;

pub use llava::{ImageAspectRatio, LlavaNextProcessor, LlavaProcessor};
