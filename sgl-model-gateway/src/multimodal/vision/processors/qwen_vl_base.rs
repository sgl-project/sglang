//! Shared base implementation for Qwen VL family image processors.
//!
//! This module provides a generic processor that handles the common logic
//! for Qwen2-VL, Qwen2.5-VL, and Qwen3-VL models. The specific variants
//! differ only in their default parameters (patch_size, normalization values).
//!
//! # Processing Pipeline
//!
//! 1. Validate aspect ratio (must be < 200:1)
//! 2. Smart resize to fit within min/max pixel bounds
//! 3. Align dimensions to (patch_size * merge_size) boundary
//! 4. Convert to tensor and normalize
//! 5. Reshape into patches for the vision encoder
//!
//! # Token Calculation
//!
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

/// Python-compatible rounding (banker's rounding / round half to even).
///
/// This matches Python's `round()` behavior where 0.5 is rounded to the nearest
/// even number, unlike Rust's `f64::round()` which rounds half away from zero.
///
/// Examples:
/// - round_half_to_even(12.5) = 12 (not 13)
/// - round_half_to_even(13.5) = 14 (not 14)
/// - round_half_to_even(12.4) = 12
/// - round_half_to_even(12.6) = 13
#[inline]
fn round_half_to_even(x: f64) -> f64 {
    let rounded = x.round();
    // Check if we're exactly at a .5 case
    if (x - x.floor() - 0.5).abs() < 1e-9 {
        // Round to nearest even
        if rounded as i64 % 2 != 0 {
            return rounded - 1.0;
        }
    }
    rounded
}

/// Configuration for a Qwen VL processor variant.
#[derive(Debug, Clone)]
pub struct QwenVLConfig {
    /// Vision encoder patch size
    pub patch_size: usize,
    /// Merge size for token reduction
    pub merge_size: usize,
    /// Minimum total pixels allowed
    pub min_pixels: usize,
    /// Maximum total pixels allowed
    pub max_pixels: usize,
    /// Temporal patch size for video
    pub temporal_patch_size: usize,
    /// Normalization mean values
    pub mean: [f64; 3],
    /// Normalization std values
    pub std: [f64; 3],
    /// Model name for identification
    pub model_name: &'static str,
}

/// Generic Qwen VL image processor.
///
/// This struct implements the shared preprocessing logic for all Qwen VL
/// model variants. Each variant (Qwen2-VL, Qwen3-VL, etc.) uses this with
/// different configuration values.
#[derive(Debug, Clone)]
pub struct QwenVLProcessorBase {
    config: QwenVLConfig,
}

impl QwenVLProcessorBase {
    /// Create a new processor with the given configuration.
    pub fn new(config: QwenVLConfig) -> Self {
        Self { config }
    }

    /// Get the patch size.
    pub fn patch_size(&self) -> usize {
        self.config.patch_size
    }

    /// Get the merge size.
    pub fn merge_size(&self) -> usize {
        self.config.merge_size
    }

    /// Get the minimum pixels.
    pub fn min_pixels(&self) -> usize {
        self.config.min_pixels
    }

    /// Get the maximum pixels.
    pub fn max_pixels(&self) -> usize {
        self.config.max_pixels
    }

    /// Get the temporal patch size.
    pub fn temporal_patch_size(&self) -> usize {
        self.config.temporal_patch_size
    }

    /// Get the factor for dimension alignment.
    ///
    /// Dimensions must be divisible by (patch_size * merge_size).
    #[inline]
    pub fn get_factor(&self) -> usize {
        self.config.patch_size * self.config.merge_size
    }

    /// Smart resize algorithm for Qwen VL models.
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

        // Round to nearest factor multiple using Python-compatible rounding
        // Python uses banker's rounding (round half to even), which affects
        // edge cases like 400/32 = 12.5 -> 12 (not 13)
        let mut h_bar = round_half_to_even(height as f64 / factor as f64) as usize * factor;
        let mut w_bar = round_half_to_even(width as f64 / factor as f64) as usize * factor;

        // Ensure minimum size
        h_bar = h_bar.max(factor);
        w_bar = w_bar.max(factor);

        // Scale down if exceeding max_pixels
        if h_bar * w_bar > self.config.max_pixels {
            let beta = ((height * width) as f64 / self.config.max_pixels as f64).sqrt();
            h_bar = ((height as f64 / beta / factor as f64).floor() as usize) * factor;
            w_bar = ((width as f64 / beta / factor as f64).floor() as usize) * factor;
            // Ensure minimum size after scaling down
            h_bar = h_bar.max(factor);
            w_bar = w_bar.max(factor);
        }
        // Scale up if below min_pixels
        else if h_bar * w_bar < self.config.min_pixels {
            let beta = (self.config.min_pixels as f64 / (height * width) as f64).sqrt();
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
        let grid_t =
            num_frames.max(self.config.temporal_patch_size) / self.config.temporal_patch_size;
        let grid_h = height / self.config.patch_size;
        let grid_w = width / self.config.patch_size;
        (grid_t, grid_h, grid_w)
    }

    /// Calculate the number of image tokens after merge.
    ///
    /// tokens = (grid_t * grid_h * grid_w) / merge_size²
    pub fn calculate_tokens_from_grid(&self, grid_t: usize, grid_h: usize, grid_w: usize) -> usize {
        (grid_t * grid_h * grid_w) / (self.config.merge_size * self.config.merge_size)
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

        let patch_size = self.config.patch_size;
        let merge_size = self.config.merge_size;
        let temporal_patch_size = self.config.temporal_patch_size;

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

impl ImagePreProcessor for QwenVLProcessorBase {
    fn default_mean(&self) -> [f64; 3] {
        self.config.mean
    }

    fn default_std(&self) -> [f64; 3] {
        self.config.std
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
        self.config.model_name
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        // Qwen VL models have dynamic sizing, no fixed output size
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> QwenVLConfig {
        QwenVLConfig {
            patch_size: 14,
            merge_size: 2,
            min_pixels: 256 * 28 * 28,
            max_pixels: 1280 * 28 * 28,
            temporal_patch_size: 2,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            model_name: "test-qwen-vl",
        }
    }

    #[test]
    fn test_qwen_vl_base_factor() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        assert_eq!(processor.get_factor(), 28); // 14 * 2
    }

    #[test]
    fn test_smart_resize_within_bounds() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        let (h, w) = processor.smart_resize(500, 500).unwrap();

        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!(h * w >= processor.min_pixels());
        assert!(h * w <= processor.max_pixels());
    }

    #[test]
    fn test_smart_resize_extreme_aspect_ratio_error() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        let result = processor.smart_resize(100, 30000);
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_grid_thw() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        let (t, h, w) = processor.calculate_grid_thw(448, 448, 1);

        assert_eq!(t, 1);
        assert_eq!(h, 448 / 14);
        assert_eq!(w, 448 / 14);
    }

    #[test]
    fn test_calculate_tokens() {
        let processor = QwenVLProcessorBase::new(create_test_config());
        let tokens = processor.calculate_tokens_from_grid(1, 32, 32);
        assert_eq!(tokens, (32 * 32) / 4);
    }
}
