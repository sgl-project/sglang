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
//! - **Phi4-Vision** (`phi4_vision`): Dynamic HD transform with 448x448 tiles and SiGLIP encoder
//! - **LLaMA 4 Vision** (`llama4_vision`): Tile-based processing with 336x336 tiles and global tile
//! - **Pixtral/Mistral3** (`pixtral`): CLIP-based preprocessing with dynamic resolution

pub mod llama4_vision;
pub mod llava;
pub mod phi3_vision;
pub mod phi4_vision;
pub mod pixtral;
pub mod qwen2_vl;
pub mod qwen3_vl;
pub mod qwen_vl_base;

pub use llama4_vision::Llama4VisionProcessor;
pub use llava::{ImageAspectRatio, LlavaNextProcessor, LlavaProcessor};
pub use phi3_vision::Phi3VisionProcessor;
pub use phi4_vision::Phi4VisionProcessor;
pub use pixtral::PixtralProcessor;
pub use qwen2_vl::Qwen2VLProcessor;
pub use qwen3_vl::Qwen3VLProcessor;
