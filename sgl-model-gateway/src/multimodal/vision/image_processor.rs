//! Image processor trait and output types.
//!
//! This module defines the interface for model-specific image processors
//! and the common output format for preprocessed images.

use std::collections::HashMap;

use image::DynamicImage;
use ndarray::{Array4, ArrayD};

use super::{preprocessor_config::PreProcessorConfig, transforms::TransformError};

/// Model-specific output values that vary by architecture.
///
/// Different vision models require different auxiliary outputs beyond pixel_values.
/// This enum captures the common types of such outputs.
#[derive(Debug, Clone)]
pub enum ModelSpecificValue {
    /// A tensor with shape information (data as flat vec, shape as dims)
    Tensor { data: Vec<f32>, shape: Vec<usize> },

    /// A tensor of integers (e.g., aspect_ratio_ids)
    IntTensor { data: Vec<i64>, shape: Vec<usize> },

    /// A tensor of unsigned integers (e.g., image_grid_thw)
    UintTensor { data: Vec<u32>, shape: Vec<usize> },

    /// Simple integer value
    Int(i64),

    /// Simple float value
    Float(f64),

    /// List of integers
    IntVec(Vec<i64>),

    /// List of unsigned integers
    UintVec(Vec<u32>),

    /// List of floats
    FloatVec(Vec<f32>),

    /// List of tuples (e.g., image sizes)
    TupleVec(Vec<(u32, u32)>),

    /// Boolean flag
    Bool(bool),
}

impl ModelSpecificValue {
    /// Create a 1D uint tensor from a vector.
    pub fn uint_1d(data: Vec<u32>) -> Self {
        let len = data.len();
        Self::UintTensor {
            data,
            shape: vec![len],
        }
    }

    /// Create a 2D uint tensor.
    pub fn uint_2d(data: Vec<u32>, rows: usize, cols: usize) -> Self {
        Self::UintTensor {
            data,
            shape: vec![rows, cols],
        }
    }

    /// Create a 1D int tensor from a vector.
    pub fn int_1d(data: Vec<i64>) -> Self {
        let len = data.len();
        Self::IntTensor {
            data,
            shape: vec![len],
        }
    }
}

/// Preprocessed images ready for model consumption.
///
/// This struct contains all the outputs needed by the SGLang scheduler
/// to construct `MultimodalInputs` for the model.
#[derive(Debug, Clone)]
pub struct PreprocessedImages {
    /// Pixel values as a dynamic-dimensional float32 tensor.
    ///
    /// This is the primary input to the vision encoder.
    /// Shape varies by model:
    /// - Standard: [B, C, H, W] (4D)
    /// - Phi3-Vision: [B, num_crops+1, C, H, W] (5D)
    pub pixel_values: ArrayD<f32>,

    /// Number of image tokens per image in the batch.
    ///
    /// Used to expand placeholder tokens in the text input.
    /// For example, LLaVA with 336x336 and patch_size=14 produces 576 tokens.
    pub num_img_tokens: Vec<usize>,

    /// Original image sizes as (width, height) before preprocessing.
    ///
    /// Some models need this for proper attention masking or position encoding.
    pub image_sizes: Vec<(u32, u32)>,

    /// Model-specific auxiliary outputs.
    ///
    /// Examples:
    /// - Qwen-VL: `image_grid_thw` for rotary position encoding
    /// - LLaMA-Vision: `aspect_ratio_ids`, `aspect_ratio_mask`
    /// - Phi3-Vision: `num_img_tokens` per crop
    pub model_specific: HashMap<String, ModelSpecificValue>,
}

impl PreprocessedImages {
    /// Create a new PreprocessedImages with required fields (4D pixel values).
    pub fn new(
        pixel_values: Array4<f32>,
        num_img_tokens: Vec<usize>,
        image_sizes: Vec<(u32, u32)>,
    ) -> Self {
        Self {
            pixel_values: pixel_values.into_dyn(),
            num_img_tokens,
            image_sizes,
            model_specific: HashMap::new(),
        }
    }

    /// Create a new PreprocessedImages with dynamic-dimensional pixel values.
    ///
    /// Use this for models like Phi3-Vision that have 5D tensors.
    pub fn new_dynamic(
        pixel_values: ArrayD<f32>,
        num_img_tokens: Vec<usize>,
        image_sizes: Vec<(u32, u32)>,
    ) -> Self {
        Self {
            pixel_values,
            num_img_tokens,
            image_sizes,
            model_specific: HashMap::new(),
        }
    }

    /// Add a model-specific value.
    pub fn with_extra(mut self, key: impl Into<String>, value: ModelSpecificValue) -> Self {
        self.model_specific.insert(key.into(), value);
        self
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.pixel_values.shape()[0]
    }

    /// Get the number of channels.
    ///
    /// For 4D tensors [B, C, H, W], returns shape[1].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[2].
    pub fn channels(&self) -> usize {
        match self.pixel_values.ndim() {
            4 => self.pixel_values.shape()[1],
            5 => self.pixel_values.shape()[2],
            ndim => panic!(
                "Unsupported pixel_values dimension: {}, expected 4 or 5",
                ndim
            ),
        }
    }

    /// Get the height of processed images.
    ///
    /// For 4D tensors [B, C, H, W], returns shape[2].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[3].
    pub fn height(&self) -> usize {
        match self.pixel_values.ndim() {
            4 => self.pixel_values.shape()[2],
            5 => self.pixel_values.shape()[3],
            ndim => panic!(
                "Unsupported pixel_values dimension: {}, expected 4 or 5",
                ndim
            ),
        }
    }

    /// Get the width of processed images.
    ///
    /// For 4D tensors [B, C, H, W], returns shape[3].
    /// For 5D tensors [B, N, C, H, W] (Phi3-Vision), returns shape[4].
    pub fn width(&self) -> usize {
        match self.pixel_values.ndim() {
            4 => self.pixel_values.shape()[3],
            5 => self.pixel_values.shape()[4],
            ndim => panic!(
                "Unsupported pixel_values dimension: {}, expected 4 or 5",
                ndim
            ),
        }
    }

    /// Get the number of dimensions of pixel_values.
    pub fn ndim(&self) -> usize {
        self.pixel_values.ndim()
    }

    /// Get total number of image tokens across all images.
    pub fn total_tokens(&self) -> usize {
        self.num_img_tokens.iter().sum()
    }

    /// Get pixel values as a flat f32 slice (row-major order).
    pub fn pixel_values_flat(&self) -> Vec<f32> {
        self.pixel_values.iter().copied().collect()
    }

    /// Get the shape of pixel values as a vector.
    pub fn pixel_values_shape(&self) -> Vec<usize> {
        self.pixel_values.shape().to_vec()
    }
}

/// Trait for model-specific image preprocessors.
///
/// Each vision model (LLaVA, Qwen-VL, Phi3-Vision, etc.) implements this trait
/// to provide the correct preprocessing pipeline.
pub trait ImagePreProcessor: Send + Sync {
    /// Default normalization mean for this model family.
    fn default_mean(&self) -> [f64; 3];

    /// Default normalization std for this model family.
    fn default_std(&self) -> [f64; 3];

    /// Preprocess a batch of images.
    ///
    /// # Arguments
    /// * `images` - Input images to preprocess
    /// * `config` - Preprocessor configuration from HuggingFace
    ///
    /// # Returns
    /// Preprocessed images ready for the model, or an error.
    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedImages, TransformError>;

    /// Calculate the number of image tokens for a given image size.
    ///
    /// This is used to determine how many placeholder tokens to insert
    /// in the text input before the image has been fully processed.
    ///
    /// # Arguments
    /// * `width` - Image width after preprocessing
    /// * `height` - Image height after preprocessing
    /// * `config` - Preprocessor configuration
    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize;

    /// Get the model family name for identification.
    fn model_name(&self) -> &'static str;

    /// Get the expected image size after preprocessing.
    ///
    /// Some models have fixed sizes, others are dynamic.
    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        config.get_target_size()
    }
}

/// Registry of available image processors.
pub struct ImageProcessorRegistry {
    processors: HashMap<String, Box<dyn ImagePreProcessor>>,
}

impl ImageProcessorRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            processors: HashMap::new(),
        }
    }

    /// Register a processor for a model pattern.
    pub fn register(&mut self, pattern: impl Into<String>, processor: Box<dyn ImagePreProcessor>) {
        self.processors.insert(pattern.into(), processor);
    }

    /// Find a processor for the given model ID.
    ///
    /// Matches by substring containment (case-insensitive).
    pub fn find(&self, model_id: &str) -> Option<&dyn ImagePreProcessor> {
        let model_lower = model_id.to_lowercase();
        for (pattern, processor) in &self.processors {
            if model_lower.contains(&pattern.to_lowercase()) {
                return Some(processor.as_ref());
            }
        }
        None
    }

    /// Check if a model has a registered processor.
    pub fn has_processor(&self, model_id: &str) -> bool {
        self.find(model_id).is_some()
    }

    /// Get list of supported model patterns.
    pub fn supported_patterns(&self) -> Vec<&str> {
        self.processors.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for ImageProcessorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageProcessorRegistry {
    /// Create a registry with all built-in processors registered.
    ///
    /// Currently registers:
    /// - `llava-next` -> LlavaNextProcessor
    /// - `llava` -> LlavaProcessor (also matches llava-1.5, etc.)
    /// - `qwen2-vl` -> Qwen2VLProcessor
    /// - `qwen2.5-vl` -> Qwen2VLProcessor (same preprocessing as Qwen2-VL)
    /// - `qwen3-vl` -> Qwen3VLProcessor (patch_size=16, [0.5,0.5,0.5] normalization)
    /// - `phi-3-vision` -> Phi3VisionProcessor (HD transform with 336x336 tiles)
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();

        // Register LLaVA-NeXT first (more specific pattern)
        registry.register(
            "llava-next",
            Box::new(super::processors::LlavaNextProcessor::new()),
        );
        registry.register(
            "llava-v1.6",
            Box::new(super::processors::LlavaNextProcessor::new()),
        );

        // Register standard LLaVA (matches llava-1.5, llava-v1.5, etc.)
        registry.register("llava", Box::new(super::processors::LlavaProcessor::new()));

        // Register Qwen3-VL first (more specific pattern - must match before qwen2)
        registry.register(
            "qwen3-vl",
            Box::new(super::processors::Qwen3VLProcessor::new()),
        );
        registry.register(
            "qwen3_vl",
            Box::new(super::processors::Qwen3VLProcessor::new()),
        );

        // Register Qwen2-VL (matches Qwen/Qwen2-VL-*, etc.)
        registry.register(
            "qwen2-vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );
        registry.register(
            "qwen2_vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );

        // Register Qwen2.5-VL (uses identical preprocessing to Qwen2-VL)
        registry.register(
            "qwen2.5-vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );
        registry.register(
            "qwen2_5-vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );
        registry.register(
            "qwen2_5_vl",
            Box::new(super::processors::Qwen2VLProcessor::new()),
        );

        // Register Phi3-Vision
        registry.register(
            "phi-3-vision",
            Box::new(super::processors::Phi3VisionProcessor::new()),
        );
        registry.register(
            "phi3-vision",
            Box::new(super::processors::Phi3VisionProcessor::new()),
        );

        registry
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array4;

    use super::*;

    #[test]
    fn test_preprocessed_images_accessors() {
        let pixel_values = Array4::<f32>::zeros((2, 3, 336, 336));
        let images =
            PreprocessedImages::new(pixel_values, vec![576, 576], vec![(640, 480), (800, 600)]);

        assert_eq!(images.batch_size(), 2);
        assert_eq!(images.channels(), 3);
        assert_eq!(images.height(), 336);
        assert_eq!(images.width(), 336);
        assert_eq!(images.total_tokens(), 1152);
    }

    #[test]
    fn test_preprocessed_images_with_extra() {
        let pixel_values = Array4::<f32>::zeros((1, 3, 224, 224));
        let images = PreprocessedImages::new(pixel_values, vec![196], vec![(224, 224)])
            .with_extra(
                "image_grid_thw",
                ModelSpecificValue::uint_1d(vec![1, 16, 16]),
            )
            .with_extra("aspect_ratio_id", ModelSpecificValue::Int(0));

        assert!(images.model_specific.contains_key("image_grid_thw"));
        assert!(images.model_specific.contains_key("aspect_ratio_id"));
    }

    #[test]
    fn test_model_specific_value_constructors() {
        let uint_1d = ModelSpecificValue::uint_1d(vec![1, 2, 3]);
        match uint_1d {
            ModelSpecificValue::UintTensor { data, shape } => {
                assert_eq!(data, vec![1, 2, 3]);
                assert_eq!(shape, vec![3]);
            }
            _ => panic!("Expected UintTensor"),
        }

        let uint_2d = ModelSpecificValue::uint_2d(vec![1, 2, 3, 4], 2, 2);
        match uint_2d {
            ModelSpecificValue::UintTensor { data, shape } => {
                assert_eq!(data, vec![1, 2, 3, 4]);
                assert_eq!(shape, vec![2, 2]);
            }
            _ => panic!("Expected UintTensor"),
        }
    }

    #[test]
    fn test_pixel_values_flat() {
        let mut pixel_values = Array4::<f32>::zeros((1, 1, 2, 2));
        pixel_values[[0, 0, 0, 0]] = 1.0;
        pixel_values[[0, 0, 0, 1]] = 2.0;
        pixel_values[[0, 0, 1, 0]] = 3.0;
        pixel_values[[0, 0, 1, 1]] = 4.0;

        let images = PreprocessedImages::new(pixel_values, vec![4], vec![(2, 2)]);
        let flat = images.pixel_values_flat();

        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_registry_with_defaults() {
        let registry = ImageProcessorRegistry::with_defaults();

        // Should find LLaVA processor
        assert!(registry.has_processor("llava-hf/llava-1.5-7b-hf"));
        assert!(registry.has_processor("liuhaotian/llava-v1.5-7b"));

        // Should find LLaVA-NeXT processor
        assert!(registry.has_processor("llava-hf/llava-v1.6-mistral-7b-hf"));
        assert!(registry.has_processor("lmms-lab/llava-next-interleave-qwen-7b"));

        // Get the processor and check model name
        let processor = registry.find("llava-hf/llava-1.5-7b-hf").unwrap();
        assert_eq!(processor.model_name(), "llava");
    }

    #[test]
    fn test_registry_find() {
        let mut registry = ImageProcessorRegistry::new();

        // Create a mock processor using LlavaProcessor
        registry.register(
            "test-model",
            Box::new(crate::multimodal::vision::processors::LlavaProcessor::new()),
        );

        assert!(registry.has_processor("test-model-7b"));
        assert!(registry.has_processor("TEST-MODEL"));
        assert!(!registry.has_processor("other-model"));
    }
}
