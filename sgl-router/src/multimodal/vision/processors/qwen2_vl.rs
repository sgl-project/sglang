//! Qwen2-VL family image processors.
//!
//! This module implements preprocessing for Qwen2-VL models, which use dynamic
//! resolution with smart resizing to maintain aspect ratio within pixel bounds.
//!
//! # Key Features
//!
//! - **Smart Resize**: Resizes images to fit within min/max pixel bounds while
//!   preserving aspect ratio and aligning to patch boundaries
//! - **Dynamic Token Count**: Token count depends on actual image dimensions
//! - **image_grid_thw**: Returns (T, H, W) grid dimensions for position encoding
//!
//! # Processing Pipeline
//!
//! 1. Validate aspect ratio (must be < 200:1)
//! 2. Smart resize to fit within min/max pixel bounds
//! 3. Align dimensions to (patch_size * merge_size) boundary
//! 4. Convert to tensor and normalize with CLIP mean/std
//! 5. Reshape into patches for the vision encoder
//!
//! # Token Calculation
//!
//! For Qwen2-VL, the number of image tokens is:
//! ```text
//! grid_t = 1  (for images, temporal dimension is 1)
//! grid_h = resized_height / patch_size
//! grid_w = resized_width / patch_size
//! num_tokens = (grid_t * grid_h * grid_w) / merge_size²
//! ```

use image::{DynamicImage, GenericImageView};
use ndarray::Array3;

use crate::multimodal::vision::{
    image_processor::{ImagePreProcessor, ModelSpecificValue, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    transforms::{normalize, pil_to_filter, resize, stack_batch, to_tensor, TransformError},
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
/// Implements dynamic resolution preprocessing with smart resizing that:
/// - Maintains aspect ratio
/// - Fits within configurable min/max pixel bounds
/// - Aligns to patch boundaries for efficient vision encoding
///
///
/// The processor returns `image_grid_thw` in the model-specific outputs,
/// which contains the (T, H, W) grid dimensions needed for rotary position
/// encoding in the Qwen2-VL model.
#[derive(Debug, Clone)]
pub struct Qwen2VLProcessor {
    /// Vision encoder patch size (typically 14)
    pub patch_size: usize,
    /// Merge size for token reduction (typically 2)
    pub merge_size: usize,
    /// Minimum total pixels allowed
    pub min_pixels: usize,
    /// Maximum total pixels allowed
    pub max_pixels: usize,
    /// Temporal patch size for video (typically 2)
    pub temporal_patch_size: usize,
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
    pub fn new() -> Self {
        Self {
            patch_size: DEFAULT_PATCH_SIZE,
            merge_size: DEFAULT_MERGE_SIZE,
            min_pixels: DEFAULT_MIN_PIXELS,
            max_pixels: DEFAULT_MAX_PIXELS,
            temporal_patch_size: DEFAULT_TEMPORAL_PATCH_SIZE,
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
            patch_size,
            merge_size,
            min_pixels,
            max_pixels,
            temporal_patch_size,
        }
    }

    /// Create a processor from preprocessor config.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            patch_size: config.patch_size.unwrap_or(DEFAULT_PATCH_SIZE),
            merge_size: config.merge_size.unwrap_or(DEFAULT_MERGE_SIZE),
            min_pixels: config.min_pixels.unwrap_or(DEFAULT_MIN_PIXELS),
            max_pixels: config.max_pixels.unwrap_or(DEFAULT_MAX_PIXELS),
            temporal_patch_size: config
                .temporal_patch_size
                .unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE),
        }
    }

    /// Get the factor for dimension alignment.
    ///
    /// Dimensions must be divisible by (patch_size * merge_size).
    #[inline]
    pub fn get_factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// Smart resize algorithm for Qwen2-VL.
    ///
    /// Resizes image dimensions to fit within min/max pixel bounds while:
    /// - Preserving aspect ratio
    /// - Aligning to (patch_size * merge_size) boundaries
    ///
    /// # Arguments
    /// * `height` - Original image height
    /// * `width` - Original image width
    ///
    /// # Returns
    /// (new_height, new_width) or error if aspect ratio is too extreme
    ///
    /// # Errors
    /// - If height or width is smaller than the factor
    /// - If aspect ratio exceeds 200:1
    pub fn smart_resize(
        &self,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize), TransformError> {
        let factor = self.get_factor();

        // Validate minimum dimensions
        if height < factor || width < factor {
            return Err(TransformError::InvalidShape {
                expected: format!("dimensions >= {} (patch_size * merge_size)", factor),
                actual: vec![height, width],
            });
        }

        // Validate aspect ratio
        let max_dim = height.max(width) as f64;
        let min_dim = height.min(width) as f64;
        let aspect_ratio = max_dim / min_dim;
        if aspect_ratio > 200.0 {
            return Err(TransformError::InvalidShape {
                expected: "aspect ratio < 200:1".to_string(),
                actual: vec![height, width],
            });
        }

        // Round to nearest factor multiple
        let mut h_bar = (height as f64 / factor as f64).round() as usize * factor;
        let mut w_bar = (width as f64 / factor as f64).round() as usize * factor;

        // Ensure minimum size
        h_bar = h_bar.max(factor);
        w_bar = w_bar.max(factor);

        // Scale down if exceeding max_pixels
        if h_bar * w_bar > self.max_pixels {
            let beta = ((height * width) as f64 / self.max_pixels as f64).sqrt();
            h_bar = ((height as f64 / beta / factor as f64).floor() as usize) * factor;
            w_bar = ((width as f64 / beta / factor as f64).floor() as usize) * factor;
            // Ensure minimum size after scaling down
            h_bar = h_bar.max(factor);
            w_bar = w_bar.max(factor);
        }
        // Scale up if below min_pixels
        else if h_bar * w_bar < self.min_pixels {
            let beta = (self.min_pixels as f64 / (height * width) as f64).sqrt();
            h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
            w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
        }

        Ok((h_bar, w_bar))
    }

    /// Calculate the grid dimensions (T, H, W) for an image.
    ///
    /// For single images, T=1. For video, T = num_frames / temporal_patch_size.
    ///
    /// # Arguments
    /// * `height` - Resized image height
    /// * `width` - Resized image width
    /// * `num_frames` - Number of frames (1 for images)
    ///
    /// # Returns
    /// (grid_t, grid_h, grid_w)
    pub fn calculate_grid_thw(
        &self,
        height: usize,
        width: usize,
        num_frames: usize,
    ) -> (usize, usize, usize) {
        let grid_t = num_frames.max(self.temporal_patch_size) / self.temporal_patch_size;
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        (grid_t, grid_h, grid_w)
    }

    /// Calculate the number of image tokens after merge.
    ///
    /// tokens = (grid_t * grid_h * grid_w) / merge_size²
    pub fn calculate_tokens_from_grid(&self, grid_t: usize, grid_h: usize, grid_w: usize) -> usize {
        (grid_t * grid_h * grid_w) / (self.merge_size * self.merge_size)
    }

    /// Reshape pixel values from [C, H, W] to flattened patches format.
    ///
    /// This matches the HuggingFace Qwen2VLImageProcessor output format:
    /// `(num_patches, patch_features)` where:
    /// - num_patches = grid_t * grid_h * grid_w
    /// - patch_features = C * temporal_patch_size * patch_size * patch_size
    ///
    /// The transformation follows these steps (matching HuggingFace exactly):
    /// 1. Start with [C, H, W] tensor, expand to [temporal, C, H, W]
    /// 2. Reshape to [grid_t, temporal, C, grid_h/merge, merge, patch, grid_w/merge, merge, patch]
    /// 3. Permute to [grid_t, grid_h/merge, grid_w/merge, merge, merge, C, temporal, patch, patch]
    /// 4. Flatten to [num_patches, patch_features]
    ///
    /// # Arguments
    /// * `tensor` - Input tensor of shape [C, H, W]
    /// * `grid_t` - Temporal grid size (1 for images)
    /// * `grid_h` - Height grid size (H / patch_size)
    /// * `grid_w` - Width grid size (W / patch_size)
    ///
    /// # Returns
    /// Flattened patches as Vec<f32> with shape semantics (num_patches, patch_features)
    pub fn reshape_to_patches(
        &self,
        tensor: &Array3<f32>,
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
    ) -> Vec<f32> {
        use ndarray::IxDyn;

        let channel = tensor.shape()[0];
        let height = tensor.shape()[1];
        let width = tensor.shape()[2];

        let patch_size = self.patch_size;
        let merge_size = self.merge_size;
        let temporal_patch_size = self.temporal_patch_size;

        // Verify dimensions match expected grid
        debug_assert_eq!(
            height,
            grid_h * patch_size,
            "Height must match grid_h * patch_size"
        );
        debug_assert_eq!(
            width,
            grid_w * patch_size,
            "Width must match grid_w * patch_size"
        );

        // Step 1: Expand temporal dimension by replicating the frame
        // [C, H, W] -> [temporal_patch_size, C, H, W]
        let expanded = tensor
            .view()
            .insert_axis(ndarray::Axis(0))
            .broadcast((temporal_patch_size, channel, height, width))
            .expect("Broadcast failed")
            .to_owned();

        // Step 2: Reshape to split spatial dimensions into grid and patch components
        // [temporal, C, H, W] -> [grid_t, temporal, C, grid_h/merge, merge, patch, grid_w/merge, merge, patch]
        //
        // Note: For images, grid_t=1 and we have temporal_patch_size frames (replicated)
        // HF reshape: [grid_t, temporal, C, grid_h/merge, merge, patch, grid_w/merge, merge, patch]
        let grid_h_merged = grid_h / merge_size;
        let grid_w_merged = grid_w / merge_size;

        // Use IxDyn for 9-dimensional reshape (ndarray only supports up to Ix6 for fixed dims)
        let shape_9d = IxDyn(&[
            grid_t,
            temporal_patch_size,
            channel,
            grid_h_merged,
            merge_size,
            patch_size,
            grid_w_merged,
            merge_size,
            patch_size,
        ]);

        let reshaped = expanded
            .into_shape_with_order(shape_9d)
            .expect("Reshape failed");

        // Step 3: Permute axes to match HuggingFace output order
        // From: [grid_t, temporal, C, grid_h/merge, merge, patch, grid_w/merge, merge, patch]
        //       [  0   ,    1    , 2,      3      ,   4  ,   5  ,      6      ,   7  ,   8  ]
        // To:   [grid_t, grid_h/merge, grid_w/merge, merge, merge, C, temporal, patch, patch]
        //       [  0   ,      3      ,      6      ,   4  ,   7  , 2,    1    ,   5  ,   8  ]
        let permuted = reshaped.permuted_axes(&[0, 3, 6, 4, 7, 2, 1, 5, 8][..]);

        // Step 4: Flatten to [num_patches, patch_features]
        // num_patches = grid_t * grid_h * grid_w = grid_t * (grid_h/merge * merge) * (grid_w/merge * merge)
        // patch_features = C * temporal * patch * patch
        let num_patches = grid_t * grid_h * grid_w;
        let patch_features = channel * temporal_patch_size * patch_size * patch_size;

        // Make contiguous and flatten
        let contiguous = permuted.as_standard_layout().into_owned();
        let flat = contiguous
            .into_shape_with_order(IxDyn(&[num_patches, patch_features]))
            .expect("Final reshape failed");

        let (vec, _offset) = flat.into_raw_vec_and_offset();
        vec
    }
}

impl ImagePreProcessor for Qwen2VLProcessor {
    fn default_mean(&self) -> [f64; 3] {
        CLIP_MEAN
    }

    fn default_std(&self) -> [f64; 3] {
        CLIP_STD
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedImages, TransformError> {
        if images.is_empty() {
            return Err(TransformError::EmptyBatch);
        }

        // Store original sizes
        let image_sizes: Vec<(u32, u32)> = images.iter().map(|img| img.dimensions()).collect();

        // For Qwen2-VL, when batching multiple images, they may have different
        // resized dimensions. We need to either:
        // 1. Process each image separately and return individual tensors
        // 2. Pad all images to the max dimensions in the batch
        //
        // Following mistral.rs approach: find max dimensions and resize all to that

        // First pass: calculate target dimensions for each image
        let mut target_sizes = Vec::with_capacity(images.len());
        for image in images {
            let (w, h) = image.dimensions();
            let (new_h, new_w) = self.smart_resize(h as usize, w as usize)?;
            target_sizes.push((new_h, new_w));
        }

        // Find max height and width across all images
        let max_height = target_sizes.iter().map(|(h, _)| *h).max().unwrap_or(0);
        let max_width = target_sizes.iter().map(|(_, w)| *w).max().unwrap_or(0);

        // Process each image with uniform max dimensions
        let mean = config.get_image_mean();
        let std = config.get_image_std();
        let filter = pil_to_filter(config.resampling);

        let mut tensors = Vec::with_capacity(images.len());
        let mut grid_thw_data = Vec::with_capacity(images.len() * 3);
        let mut num_img_tokens = Vec::with_capacity(images.len());

        for (i, image) in images.iter().enumerate() {
            let (target_h, target_w) = target_sizes[i];

            // Resize to the target size for this image
            let resized = if config.do_resize.unwrap_or(true) {
                // For batching: resize to max dimensions to enable stacking
                // The actual grid dimensions are based on individual target sizes
                resize(image, max_width as u32, max_height as u32, filter)
            } else {
                image.clone()
            };

            // Convert to tensor
            let mut tensor = to_tensor(&resized);

            // Normalize
            if config.do_normalize.unwrap_or(true) {
                normalize(&mut tensor, &mean, &std);
            }

            tensors.push(tensor);

            // Grid dimensions are based on the individual image's target size
            let (grid_t, grid_h, grid_w) = self.calculate_grid_thw(target_h, target_w, 1);
            grid_thw_data.push(grid_t as u32);
            grid_thw_data.push(grid_h as u32);
            grid_thw_data.push(grid_w as u32);

            // Token count is based on individual grid
            let tokens = self.calculate_tokens_from_grid(grid_t, grid_h, grid_w);
            num_img_tokens.push(tokens);
        }

        // Stack tensors into batch (now all same size)
        let pixel_values = stack_batch(&tensors)?;

        // Create result with model-specific image_grid_thw
        let result = PreprocessedImages::new(pixel_values, num_img_tokens, image_sizes).with_extra(
            "image_grid_thw",
            ModelSpecificValue::uint_2d(grid_thw_data, images.len(), 3),
        );

        Ok(result)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, _config: &PreProcessorConfig) -> usize {
        // Calculate resized dimensions
        let (new_height, new_width) = match self.smart_resize(height as usize, width as usize) {
            Ok((h, w)) => (h, w),
            Err(_) => {
                // Fallback: use minimum size
                let factor = self.get_factor();
                (factor, factor)
            }
        };

        // Calculate grid and tokens
        let (grid_t, grid_h, grid_w) = self.calculate_grid_thw(new_height, new_width, 1);
        self.calculate_tokens_from_grid(grid_t, grid_h, grid_w)
    }

    fn model_name(&self) -> &'static str {
        "qwen2-vl"
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        // Qwen2-VL has dynamic sizing, no fixed output size
        None
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_qwen2_vl_processor_default() {
        let processor = Qwen2VLProcessor::new();
        assert_eq!(processor.patch_size, 14);
        assert_eq!(processor.merge_size, 2);
        assert_eq!(processor.min_pixels, DEFAULT_MIN_PIXELS);
        assert_eq!(processor.max_pixels, DEFAULT_MAX_PIXELS);
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
        assert!(h * w >= processor.min_pixels);
        assert!(h * w <= processor.max_pixels);
    }

    #[test]
    fn test_smart_resize_too_large() {
        let processor = Qwen2VLProcessor::new();

        // Very large image
        let (h, w) = processor.smart_resize(3000, 3000).unwrap();

        // Should be scaled down
        assert!(h * w <= processor.max_pixels);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_smart_resize_too_small() {
        let processor = Qwen2VLProcessor::new();

        // Small image (but above minimum dimension)
        let (h, w) = processor.smart_resize(100, 100).unwrap();

        // Should be scaled up to min_pixels
        assert!(h * w >= processor.min_pixels);
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
            patch_size: Some(14),
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
            patch_size: Some(16),
            merge_size: Some(4),
            min_pixels: Some(100000),
            max_pixels: Some(500000),
            temporal_patch_size: Some(4),
            ..Default::default()
        };

        let processor = Qwen2VLProcessor::from_preprocessor_config(&config);

        assert_eq!(processor.patch_size, 16);
        assert_eq!(processor.merge_size, 4);
        assert_eq!(processor.min_pixels, 100000);
        assert_eq!(processor.max_pixels, 500000);
        assert_eq!(processor.temporal_patch_size, 4);
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
