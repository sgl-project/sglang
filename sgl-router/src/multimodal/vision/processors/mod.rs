//! Model-specific image processors.
//!
//! This module contains implementations of `ImagePreProcessor` for various
//! vision-language model families.
//!
//! # Supported Models
//!
//! - **LLaVA 1.5** (`llava`): CLIP-based preprocessing with configurable aspect ratio
//! - **LLaVA-NeXT** (`llava`): Multi-crop anyres processing
//! - **Qwen2-VL** (`qwen2_vl`): Dynamic resolution with smart resizing
//! - **Qwen2.5-VL** (`qwen2_vl`): Same processor as Qwen2-VL (identical preprocessing)

pub mod llava;
pub mod qwen2_vl;

pub use llava::{ImageAspectRatio, LlavaNextProcessor, LlavaProcessor};
pub use qwen2_vl::Qwen2VLProcessor;
