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
//! - **Qwen3-VL** (`qwen3_vl`): Similar to Qwen2-VL but with patch_size=16 and [0.5,0.5,0.5] normalization
//! - **Phi3-Vision** (`phi3_vision`): Dynamic HD transform with 336x336 tiles

pub mod llava;
pub mod phi3_vision;
pub mod qwen2_vl;
pub mod qwen3_vl;
pub mod qwen_vl_base;

pub use llava::{ImageAspectRatio, LlavaNextProcessor, LlavaProcessor};
pub use phi3_vision::Phi3VisionProcessor;
pub use qwen2_vl::Qwen2VLProcessor;
pub use qwen3_vl::Qwen3VLProcessor;
