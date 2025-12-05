//! Qwen2-VL family image processors.
//!
//! This module provides the Qwen2-VL processor which wraps the shared
//! `QwenVLProcessorBase` with Qwen2-VL specific default parameters.
//!
//! # Key Features
//!
//! - **Smart Resize**: Resizes images to fit within min/max pixel bounds while
//!   preserving aspect ratio and aligning to patch boundaries
//! - **Dynamic Token Count**: Token count depends on actual image dimensions
//! - **image_grid_thw**: Returns (T, H, W) grid dimensions for position encoding
//!
//! # Qwen2-VL Parameters
//!
//! - patch_size: 14
//! - merge_size: 2
//! - factor: 28 (patch_size * merge_size)
//! - normalization: CLIP mean/std

use std::ops::Deref;

use image::DynamicImage;
use ndarray::Array3;

use super::qwen_vl_base::{QwenVLConfig, QwenVLProcessorBase};
use crate::multimodal::vision::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    transforms::TransformError,
};

/// CLIP normalization mean values used by Qwen2-VL models.
pub const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];

/// CLIP normalization std values used by Qwen2-VL models.
pub const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Default minimum pixels (256 * 28 * 28 = 200,704)
pub const DEFAULT_MIN_PIXELS: usize = 256 * 28 * 28;

/// Default maximum pixels (1280 * 28 * 28 = 1,003,520)
pub const DEFAULT_MAX_PIXELS: usize = 1280 * 28 * 28;

/// Default patch size
pub const DEFAULT_PATCH_SIZE: usize = 14;

/// Default merge size for token reduction
pub const DEFAULT_MERGE_SIZE: usize = 2;

/// Default temporal patch size (for video frames)
pub const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;

/// Qwen2-VL image processor.
///
/// This is a thin wrapper around `QwenVLProcessorBase` with Qwen2-VL
/// specific default parameters:
/// - patch_size: 14
/// - merge_size: 2
/// - CLIP normalization mean/std
#[derive(Debug, Clone)]
pub struct Qwen2VLProcessor {
    inner: QwenVLProcessorBase,
}

impl Default for Qwen2VLProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen2VLProcessor {
    /// Create a new Qwen2-VL processor with default settings.
    ///
    /// Defaults:
    /// - patch_size: 14
    /// - merge_size: 2
    /// - min_pixels: 200,704 (256 * 28 * 28)
    /// - max_pixels: 1,003,520 (1280 * 28 * 28)
    /// - temporal_patch_size: 2
    /// - normalization: CLIP mean/std
    pub fn new() -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size: DEFAULT_PATCH_SIZE,
                merge_size: DEFAULT_MERGE_SIZE,
                min_pixels: DEFAULT_MIN_PIXELS,
                max_pixels: DEFAULT_MAX_PIXELS,
                temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
                mean: CLIP_MEAN,
                std: CLIP_STD,
                model_name: "qwen2-vl",
            }),
        }
    }

    /// Create a processor with custom settings.
    pub fn with_config(
        patch_size: usize,
        merge_size: usize,
        min_pixels: usize,
        max_pixels: usize,
        temporal_patch_size: usize,
    ) -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size,
                merge_size,
                min_pixels,
                max_pixels,
                temporal_patch_size,
                mean: CLIP_MEAN,
                std: CLIP_STD,
                model_name: "qwen2-vl",
            }),
        }
    }

    /// Create a processor from preprocessor config.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size: config.get_patch_size(DEFAULT_PATCH_SIZE),
                merge_size: config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
                min_pixels: config.min_pixels.unwrap_or(DEFAULT_MIN_PIXELS),
                max_pixels: config.max_pixels.unwrap_or(DEFAULT_MAX_PIXELS),
                temporal_patch_size: config
                    .temporal_patch_size
                    .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE),
                mean: CLIP_MEAN,
                std: CLIP_STD,
                model_name: "qwen2-vl",
            }),
        }
    }

    /// Get the patch size.
    pub fn patch_size(&self) -> usize {
        self.inner.patch_size()
    }

    /// Get the merge size.
    pub fn merge_size(&self) -> usize {
        self.inner.merge_size()
    }

    /// Get the minimum pixels.
    pub fn min_pixels(&self) -> usize {
        self.inner.min_pixels()
    }

    /// Get the maximum pixels.
    pub fn max_pixels(&self) -> usize {
        self.inner.max_pixels()
    }

    /// Get the temporal patch size.
    pub fn temporal_patch_size(&self) -> usize {
        self.inner.temporal_patch_size()
    }

    /// Get the factor for dimension alignment.
    #[inline]
    pub fn get_factor(&self) -> usize {
        self.inner.get_factor()
    }

    /// Smart resize algorithm for Qwen2-VL.
    pub fn smart_resize(
        &self,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize), TransformError> {
        self.inner.smart_resize(height, width)
    }

    /// Calculate the grid dimensions (T, H, W) for an image.
    pub fn calculate_grid_thw(
        &self,
        height: usize,
        width: usize,
        num_frames: usize,
    ) -> (usize, usize, usize) {
        self.inner.calculate_grid_thw(height, width, num_frames)
    }

    /// Calculate the number of image tokens after merge.
    pub fn calculate_tokens_from_grid(&self, grid_t: usize, grid_h: usize, grid_w: usize) -> usize {
        self.inner
            .calculate_tokens_from_grid(grid_t, grid_h, grid_w)
    }

    /// Reshape pixel values from [C, H, W] to flattened patches format.
    pub fn reshape_to_patches(
        &self,
        tensor: &Array3<f32>,
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
    ) -> Vec<f32> {
        self.inner
            .reshape_to_patches(tensor, grid_t, grid_h, grid_w)
    }
}

impl Deref for Qwen2VLProcessor {
    type Target = QwenVLProcessorBase;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl ImagePreProcessor for Qwen2VLProcessor {
    fn default_mean(&self) -> [f64; 3] {
        self.inner.default_mean()
    }

    fn default_std(&self) -> [f64; 3] {
        self.inner.default_std()
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedImages, TransformError> {
        self.inner.preprocess(images, config)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        self.inner.calculate_num_tokens(width, height, config)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        self.inner.get_processed_size(config)
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;
    use crate::multimodal::vision::{
        image_processor::ModelSpecificValue, preprocessor_config::PatchSize,
    };

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_qwen2_vl_processor_default() {
        let processor = Qwen2VLProcessor::new();
        assert_eq!(processor.patch_size(), 14);
        assert_eq!(processor.merge_size(), 2);
        assert_eq!(processor.min_pixels(), DEFAULT_MIN_PIXELS);
        assert_eq!(processor.max_pixels(), DEFAULT_MAX_PIXELS);
        assert_eq!(processor.get_factor(), 28); // 14 * 2
    }

    #[test]
    fn test_smart_resize_within_bounds() {
        let processor = Qwen2VLProcessor::new();

        // Image that's already within bounds
        let (h, w) = processor.smart_resize(500, 500).unwrap();

        // Should be aligned to factor (28)
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);

        // Should be within bounds
        assert!(h * w >= processor.min_pixels());
        assert!(h * w <= processor.max_pixels());
    }

    #[test]
    fn test_smart_resize_too_large() {
        let processor = Qwen2VLProcessor::new();

        // Very large image
        let (h, w) = processor.smart_resize(3000, 3000).unwrap();

        // Should be scaled down
        assert!(h * w <= processor.max_pixels());
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_smart_resize_too_small() {
        let processor = Qwen2VLProcessor::new();

        // Small image (but above minimum dimension)
        let (h, w) = processor.smart_resize(100, 100).unwrap();

        // Should be scaled up to min_pixels
        assert!(h * w >= processor.min_pixels());
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_smart_resize_aspect_ratio_preserved() {
        let processor = Qwen2VLProcessor::new();

        // 2:1 aspect ratio
        let (h, w) = processor.smart_resize(400, 800).unwrap();

        // Aspect ratio should be approximately preserved
        let original_ratio = 800.0 / 400.0;
        let new_ratio = w as f64 / h as f64;
        assert!((new_ratio - original_ratio).abs() < 0.5);
    }

    #[test]
    fn test_smart_resize_extreme_aspect_ratio_error() {
        let processor = Qwen2VLProcessor::new();

        // 300:1 aspect ratio - should fail
        let result = processor.smart_resize(100, 30000);
        assert!(result.is_err());
    }

    #[test]
    fn test_smart_resize_too_small_dimension_error() {
        let processor = Qwen2VLProcessor::new();

        // Dimension smaller than factor
        let result = processor.smart_resize(10, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_grid_thw_image() {
        let processor = Qwen2VLProcessor::new();

        // 448x448 image (16x16 grid patches)
        let (t, h, w) = processor.calculate_grid_thw(448, 448, 1);

        assert_eq!(t, 1); // Single image
        assert_eq!(h, 448 / 14); // 32
        assert_eq!(w, 448 / 14); // 32
    }

    #[test]
    fn test_calculate_tokens() {
        let processor = Qwen2VLProcessor::new();

        // With merge_size=2, tokens = (t * h * w) / 4
        let tokens = processor.calculate_tokens_from_grid(1, 32, 32);
        assert_eq!(tokens, (32 * 32) / 4); // 256
    }

    #[test]
    fn test_qwen2_vl_preprocess() {
        let processor = Qwen2VLProcessor::new();
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_normalize: Some(true),
            image_mean: Some(CLIP_MEAN.to_vec()),
            image_std: Some(CLIP_STD.to_vec()),
            patch_size: Some(PatchSize {
                height: Some(14),
                width: Some(14),
            }),
            merge_size: Some(2),
            min_pixels: Some(DEFAULT_MIN_PIXELS),
            max_pixels: Some(DEFAULT_MAX_PIXELS),
            ..Default::default()
        };

        let image = create_test_image(600, 400, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[image], &config).unwrap();

        assert_eq!(result.batch_size(), 1);

        // Check pixel values are normalized
        let flat = result.pixel_values_flat();
        // After normalization with CLIP mean/std, gray (0.5) should be near 0
        // (0.5 - 0.48) / 0.27 ≈ 0.07
        assert!(flat.iter().all(|&v| v.abs() < 1.0)); // Should be normalized

        // Check image_grid_thw is present
        assert!(result.model_specific.contains_key("image_grid_thw"));

        // Verify token count is reasonable
        assert!(result.num_img_tokens[0] > 0);
    }

    #[test]
    fn test_qwen2_vl_preprocess_multiple() {
        let processor = Qwen2VLProcessor::new();
        let config = PreProcessorConfig::default();

        let images = vec![
            create_test_image(600, 400, Rgb([100, 100, 100])),
            create_test_image(400, 600, Rgb([150, 150, 150])),
        ];

        let result = processor.preprocess(&images, &config).unwrap();

        // Both images processed
        assert_eq!(result.image_sizes.len(), 2);
        assert_eq!(result.num_img_tokens.len(), 2);

        // Check grid_thw shape
        if let Some(ModelSpecificValue::UintTensor { data, shape }) =
            result.model_specific.get("image_grid_thw")
        {
            assert_eq!(shape, &[2, 3]); // 2 images, 3 values (T, H, W) each
            assert_eq!(data.len(), 6);
        } else {
            panic!("Expected image_grid_thw to be UintTensor");
        }
    }

    #[test]
    fn test_qwen2_vl_from_config() {
        let config = PreProcessorConfig {
            patch_size: Some(PatchSize {
                height: Some(16),
                width: Some(16),
            }),
            merge_size: Some(4),
            min_pixels: Some(100000),
            max_pixels: Some(500000),
            temporal_patch_size: Some(4),
            ..Default::default()
        };

        let processor = Qwen2VLProcessor::from_preprocessor_config(&config);

        assert_eq!(processor.patch_size(), 16);
        assert_eq!(processor.merge_size(), 4);
        assert_eq!(processor.min_pixels(), 100000);
        assert_eq!(processor.max_pixels(), 500000);
        assert_eq!(processor.temporal_patch_size(), 4);
    }

    #[test]
    fn test_model_name() {
        let processor = Qwen2VLProcessor::new();
        assert_eq!(processor.model_name(), "qwen2-vl");
    }

    #[test]
    fn test_default_mean_std() {
        let processor = Qwen2VLProcessor::new();
        assert_eq!(processor.default_mean(), CLIP_MEAN);
        assert_eq!(processor.default_std(), CLIP_STD);
    }
}
