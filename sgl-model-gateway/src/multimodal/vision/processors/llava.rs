//! LLaVA family image processors.
//!
//! This module implements preprocessing for:
//! - LLaVA 1.5: CLIP-based preprocessing with configurable aspect ratio handling
//! - LLaVA-NeXT: Multi-crop anyres processing for higher resolution
//!
//! # Image Aspect Ratio Modes
//!
//! The processing behavior depends on the `image_aspect_ratio` config:
//!
//! - **None/Square**: Standard CLIP processing (resize shortest edge, center crop)
//! - **"pad"**: Expand to square with mean color padding, then resize
//! - **"anyres"**: Multi-crop processing for higher resolution (LLaVA-NeXT)
//!
//! # Processing Pipeline
//!
//! ## LLaVA 1.5 (Standard - no expand_to_square)
//! Used for `llava-hf/*` models where `image_aspect_ratio` is not set:
//! 1. Resize so shortest edge = target_size (preserving aspect ratio)
//! 2. Center crop to target_size x target_size
//! 3. Rescale by 1/255
//! 4. Normalize with CLIP mean/std
//!
//! ## LLaVA 1.5 (Pad mode - with expand_to_square)
//! Used for `liuhaotian/llava-*` models where `image_aspect_ratio = "pad"`:
//! 1. Expand image to square by padding with mean color
//! 2. Resize to target size (typically 336x336)
//! 3. Normalize with CLIP mean/std
//!
//! ## LLaVA-NeXT
//! 1. Select best resolution from grid pinpoints
//! 2. Resize and pad to best resolution
//! 3. Divide into crops
//! 4. Process each crop + original resized image
//! 5. Stack all processed patches

use image::{DynamicImage, GenericImageView};
use ndarray::Array3;

use crate::multimodal::vision::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    transforms::{
        center_crop, expand_to_square, mean_to_rgb, normalize, pil_to_filter, resize, stack_batch,
        to_tensor, TransformError,
    },
};

/// CLIP normalization mean values used by LLaVA models.
pub const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];

/// CLIP normalization std values used by LLaVA models.
pub const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Image aspect ratio handling mode.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ImageAspectRatio {
    /// Standard CLIP processing: resize shortest edge, center crop.
    /// Used for llava-hf/* models where image_aspect_ratio is not set.
    #[default]
    Square,
    /// Expand to square with mean color padding, then resize.
    /// Used for liuhaotian/llava-* models where image_aspect_ratio = "pad".
    Pad,
    /// Multi-crop anyres processing (handled by LlavaNextProcessor).
    Anyres,
}

impl std::str::FromStr for ImageAspectRatio {
    type Err = std::convert::Infallible;

    /// Parse from string value (e.g., from config).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "pad" => Self::Pad,
            "anyres" => Self::Anyres,
            _ if s.contains("anyres") => Self::Anyres, // anyres_max_12, etc.
            _ => Self::Square,
        })
    }
}

/// LLaVA 1.5 image processor.
///
/// Implements CLIP-based preprocessing with configurable aspect ratio handling.
/// This processor is used for LLaVA 1.5 and similar models that expect fixed-size
/// square inputs.
///
/// # Aspect Ratio Modes
///
/// - `Square` (default): Standard CLIP processing (llava-hf/*)
/// - `Pad`: Expand to square with mean padding (liuhaotian/llava-*)
///
/// # Token Calculation
///
/// For LLaVA 1.5, the number of image tokens is:
/// ```text
/// num_tokens = (image_size / patch_size)²
/// ```
/// With default settings (336x336, patch_size=14): 576 tokens
#[derive(Debug, Clone)]
pub struct LlavaProcessor {
    /// Patch size for token calculation (typically 14)
    pub patch_size: u32,
    /// Target image size after processing (typically 336)
    pub image_size: u32,
    /// Image aspect ratio handling mode
    pub aspect_ratio: ImageAspectRatio,
}

impl Default for LlavaProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl LlavaProcessor {
    /// Create a new LLaVA 1.5 processor with default settings.
    ///
    /// Default: patch_size=14, image_size=336, aspect_ratio=Square
    /// This matches the llava-hf/* model behavior.
    pub fn new() -> Self {
        Self {
            patch_size: 14,
            image_size: 336,
            aspect_ratio: ImageAspectRatio::Square,
        }
    }

    /// Create a processor with "pad" aspect ratio mode.
    ///
    /// This matches the liuhaotian/llava-* model behavior where
    /// images are expanded to square before processing.
    pub fn new_with_pad() -> Self {
        Self {
            patch_size: 14,
            image_size: 336,
            aspect_ratio: ImageAspectRatio::Pad,
        }
    }

    /// Create a processor with custom settings.
    pub fn with_config(patch_size: u32, image_size: u32, aspect_ratio: ImageAspectRatio) -> Self {
        Self {
            patch_size,
            image_size,
            aspect_ratio,
        }
    }

    /// Create a processor from model config JSON.
    ///
    /// Extracts patch_size, image_size, and image_aspect_ratio from config.
    pub fn from_config(config: &serde_json::Value) -> Self {
        let patch_size = config
            .get("vision_config")
            .and_then(|v| v.get("patch_size"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(14);

        let image_size = config
            .get("vision_config")
            .and_then(|v| v.get("image_size"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(336);

        let aspect_ratio = config
            .get("image_aspect_ratio")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse().ok())
            .unwrap_or_default();

        Self {
            patch_size,
            image_size,
            aspect_ratio,
        }
    }

    /// Process a single image through the LLaVA 1.5 pipeline.
    ///
    /// The processing flow depends on `self.aspect_ratio`:
    /// - `Square`: Standard CLIP (resize shortest edge, center crop)
    /// - `Pad`: Expand to square with mean padding, then resize
    fn process_one_image(
        &self,
        image: &DynamicImage,
        config: &PreProcessorConfig,
    ) -> Result<Array3<f32>, TransformError> {
        let mean = config.get_image_mean();
        let std = config.get_image_std();
        let filter = pil_to_filter(config.resampling);

        // Get target size from config or use default
        let target_size = config
            .get_target_size()
            .map(|(h, _w)| h)
            .unwrap_or(self.image_size);

        // Get crop size (may be different from target_size)
        let crop_size = config
            .get_crop_size()
            .map(|(h, _w)| h)
            .unwrap_or(target_size);

        let processed = match self.aspect_ratio {
            ImageAspectRatio::Pad => {
                // Pad mode: expand to square with mean color padding, then resize
                let mean_color = mean_to_rgb(&mean);
                let squared = expand_to_square(image, mean_color);

                // Resize to target size (maintaining square)
                if config.do_resize.unwrap_or(true) {
                    resize(&squared, target_size, target_size, filter)
                } else {
                    squared
                }
            }
            ImageAspectRatio::Square | ImageAspectRatio::Anyres => {
                // Square mode: Standard CLIP processing
                // 1. Resize so shortest edge = target_size (preserving aspect ratio)
                // 2. Center crop to crop_size x crop_size
                let resized = if config.do_resize.unwrap_or(true) {
                    // Resize so shortest edge = target_size
                    let (w, h) = image.dimensions();
                    let scale = if w < h {
                        target_size as f32 / w as f32
                    } else {
                        target_size as f32 / h as f32
                    };
                    let new_w = (w as f32 * scale).round() as u32;
                    let new_h = (h as f32 * scale).round() as u32;
                    resize(image, new_w, new_h, filter)
                } else {
                    image.clone()
                };

                // Center crop to crop_size
                if config.do_center_crop.unwrap_or(true) {
                    center_crop(&resized, crop_size, crop_size)
                } else {
                    resized
                }
            }
        };

        // Convert to tensor [C, H, W] normalized to [0, 1]
        let mut tensor = to_tensor(&processed);

        // Normalize with mean/std
        if config.do_normalize.unwrap_or(true) {
            normalize(&mut tensor, &mean, &std);
        }

        Ok(tensor)
    }
}

impl ImagePreProcessor for LlavaProcessor {
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

        // Process each image
        let tensors: Vec<Array3<f32>> = images
            .iter()
            .map(|img| self.process_one_image(img, config))
            .collect::<Result<Vec<_>, _>>()?;

        // Stack into batch
        let pixel_values = stack_batch(&tensors)?;

        // Calculate token counts
        let num_img_tokens: Vec<usize> = images
            .iter()
            .map(|_| self.calculate_num_tokens(self.image_size, self.image_size, config))
            .collect();

        Ok(PreprocessedImages::new(
            pixel_values,
            num_img_tokens,
            image_sizes,
        ))
    }

    fn calculate_num_tokens(
        &self,
        _width: u32,
        _height: u32,
        config: &PreProcessorConfig,
    ) -> usize {
        // For LLaVA 1.5, token count is based on processed image size and patch size
        let patch_size = config.get_patch_size(self.patch_size as usize) as u32;
        let image_size = config
            .get_target_size()
            .map(|(h, _w)| h)
            .unwrap_or(self.image_size);

        let patches_per_side = image_size / patch_size;
        (patches_per_side * patches_per_side) as usize
    }

    fn model_name(&self) -> &'static str {
        "llava"
    }

    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        let size = config
            .get_target_size()
            .map(|(h, _w)| h)
            .unwrap_or(self.image_size);
        Some((size, size))
    }
}

// ============================================================================
// LLaVA-NeXT (Anyres) Support
// ============================================================================

/// LLaVA-NeXT image processor with anyres (multi-crop) support.
///
/// LLaVA-NeXT processes high-resolution images by:
/// 1. Selecting the best resolution from predefined grid pinpoints
/// 2. Resizing and padding the image to that resolution
/// 3. Dividing into crops
/// 4. Processing each crop plus the original resized image
///
/// # Token Calculation
///
/// For LLaVA-NeXT, the number of tokens depends on the selected resolution:
/// ```text
/// base_tokens = (image_size / patch_size)²
/// grid_shape = (best_width / patch_size, best_height / patch_size)
/// unpad_shape = adjusted for aspect ratio
/// total_tokens = base_tokens + (unpad_w + 1) * unpad_h
/// ```
#[derive(Debug, Clone)]
pub struct LlavaNextProcessor {
    /// Base processor for individual patches
    pub base: LlavaProcessor,
    /// Grid pinpoints for resolution selection [(width, height), ...]
    pub image_grid_pinpoints: Vec<(u32, u32)>,
}

impl Default for LlavaNextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl LlavaNextProcessor {
    /// Create a new LLaVA-NeXT processor with default settings.
    ///
    /// Default grid pinpoints are common LLaVA-NeXT resolutions.
    pub fn new() -> Self {
        Self {
            base: LlavaProcessor::new(),
            // Common LLaVA-NeXT grid pinpoints
            image_grid_pinpoints: vec![
                (336, 672),
                (672, 336),
                (672, 672),
                (1008, 336),
                (336, 1008),
            ],
        }
    }

    /// Create a processor with custom grid pinpoints.
    pub fn with_grid_pinpoints(grid_pinpoints: Vec<(u32, u32)>) -> Self {
        Self {
            base: LlavaProcessor::new(),
            image_grid_pinpoints: grid_pinpoints,
        }
    }

    /// Create a processor from model config.
    pub fn from_config(config: &serde_json::Value) -> Self {
        let base = LlavaProcessor::from_config(config);

        let grid_pinpoints = config
            .get("image_grid_pinpoints")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|p| {
                        let pair = p.as_array()?;
                        let w = pair.first()?.as_u64()? as u32;
                        let h = pair.get(1)?.as_u64()? as u32;
                        Some((w, h))
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]);

        Self {
            base,
            image_grid_pinpoints: grid_pinpoints,
        }
    }

    /// Select the best resolution from grid pinpoints for the given image.
    ///
    /// Minimizes wasted pixels while maximizing effective resolution.
    pub fn select_best_resolution(&self, original_size: (u32, u32)) -> (u32, u32) {
        select_best_resolution(original_size, &self.image_grid_pinpoints)
    }

    /// Get the grid shape (in patches) for anyres processing.
    pub fn get_anyres_grid_shape(&self, image_size: (u32, u32)) -> (u32, u32) {
        let (width, height) = self.select_best_resolution(image_size);
        (width / self.base.patch_size, height / self.base.patch_size)
    }

    /// Calculate unpad dimensions based on original aspect ratio.
    pub fn calculate_unpad(&self, grid_shape: (u32, u32), original_size: (u32, u32)) -> (u32, u32) {
        calculate_unpad(grid_shape, original_size)
    }

    /// Resize and pad image to target resolution, maintaining aspect ratio.
    fn resize_and_pad_image(&self, image: &DynamicImage, target: (u32, u32)) -> DynamicImage {
        resize_and_pad_image(image, target)
    }

    /// Divide image into crops of specified size.
    fn divide_to_samples(&self, image: &DynamicImage, crop_size: (u32, u32)) -> Vec<DynamicImage> {
        divide_to_samples(image, crop_size)
    }

    /// Process a single patch/crop.
    fn process_patch(
        &self,
        image: &DynamicImage,
        config: &PreProcessorConfig,
    ) -> Result<Array3<f32>, TransformError> {
        let mean = config.get_image_mean();
        let std = config.get_image_std();
        let filter = pil_to_filter(config.resampling);

        // Get target size for patches
        let target_size = config
            .get_target_size()
            .map(|(h, _w)| h)
            .unwrap_or(self.base.image_size);

        // Resize patch to target size
        let resized = if config.do_resize.unwrap_or(true) {
            resize(image, target_size, target_size, filter)
        } else {
            image.clone()
        };

        // Center crop if configured
        let cropped = if config.do_center_crop.unwrap_or(true) {
            if let Some((crop_h, crop_w)) = config.get_crop_size() {
                center_crop(&resized, crop_w, crop_h)
            } else {
                resized
            }
        } else {
            resized
        };

        // Convert to tensor
        let mut tensor = to_tensor(&cropped);

        // Normalize
        if config.do_normalize.unwrap_or(true) {
            normalize(&mut tensor, &mean, &std);
        }

        Ok(tensor)
    }
}

impl ImagePreProcessor for LlavaNextProcessor {
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

        let mut all_patches = Vec::new();
        let mut num_img_tokens = Vec::with_capacity(images.len());
        let mut image_sizes = Vec::with_capacity(images.len());

        let filter = pil_to_filter(config.resampling);
        let target_size = config
            .get_target_size()
            .map(|(h, _w)| h)
            .unwrap_or(self.base.image_size);
        let crop_size = config.get_crop_size().unwrap_or((target_size, target_size));

        for image in images {
            let original_size = image.dimensions();
            image_sizes.push(original_size);

            let best_resolution = self.select_best_resolution(original_size);
            let image_padded = self.resize_and_pad_image(image, best_resolution);
            let image_original_resize = resize(image, target_size, target_size, filter);

            let mut samples = vec![image_original_resize];
            samples.extend(self.divide_to_samples(&image_padded, crop_size));

            for sample in samples {
                all_patches.push(self.process_patch(&sample, config)?);
            }

            num_img_tokens.push(self.calculate_num_tokens(
                original_size.0,
                original_size.1,
                config,
            ));
        }

        let pixel_values = stack_batch(&all_patches)?;

        Ok(PreprocessedImages::new(
            pixel_values,
            num_img_tokens,
            image_sizes,
        ))
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, _config: &PreProcessorConfig) -> usize {
        let original_size = (width, height);

        // Base tokens (from original resized image)
        let patches_per_side = self.base.image_size / self.base.patch_size;
        let base_tokens = (patches_per_side * patches_per_side) as usize;

        // Grid tokens (from crops)
        let grid_shape = self.get_anyres_grid_shape(original_size);
        let unpad_shape = self.calculate_unpad(grid_shape, original_size);

        // Total: base + unpadded area
        base_tokens + (unpad_shape.0 as usize + 1) * unpad_shape.1 as usize
    }

    fn model_name(&self) -> &'static str {
        "llava-next"
    }

    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        // LLaVA-NeXT has variable output size based on crops
        // Return the base patch size
        let size = config
            .get_target_size()
            .map(|(h, _w)| h)
            .unwrap_or(self.base.image_size);
        Some((size, size))
    }
}

// ============================================================================
// Helper Functions (ported from mistral.rs)
// ============================================================================

/// Select the best resolution from possible resolutions for the given image size.
///
/// Minimizes wasted pixels while maximizing effective resolution.
fn select_best_resolution(
    original_size: (u32, u32),
    possible_resolutions: &[(u32, u32)],
) -> (u32, u32) {
    let (original_width, original_height) = original_size;
    let mut best_fit = (0, 0);
    let original_width_f = original_width as f32;
    let original_height_f = original_height as f32;
    let mut max_effective_resolution = 0_u32;
    let mut min_wasted_resolution = u32::MAX;

    for &(width, height) in possible_resolutions {
        let width_f = width as f32;
        let height_f = height as f32;
        let scale = (width_f / original_width_f).min(height_f / original_height_f);
        let (downscaled_width, downscaled_height) = (
            (original_width_f * scale) as u32,
            (original_height_f * scale) as u32,
        );
        let effective_resolution =
            std::cmp::min(width * height, downscaled_width * downscaled_height);
        let wasted_resolution = width * height - effective_resolution;

        if effective_resolution > max_effective_resolution
            || (effective_resolution == max_effective_resolution
                && wasted_resolution < min_wasted_resolution)
        {
            best_fit = (width, height);
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
        }
    }
    best_fit
}

/// Calculate unpad dimensions based on aspect ratio.
fn calculate_unpad(size: (u32, u32), original_size: (u32, u32)) -> (u32, u32) {
    let (original_width, original_height) = original_size;
    let (current_width, current_height) = size;
    let original_aspect_ratio = original_width as f32 / original_height as f32;
    let current_aspect_ratio = current_width as f32 / current_height as f32;

    if original_aspect_ratio > current_aspect_ratio {
        let scale_factor = current_width as f32 / original_width as f32;
        let new_height = (original_height as f32 * scale_factor).floor() as u32;
        let padding = (current_height - new_height) / 2;
        (current_width, current_height - 2 * padding)
    } else {
        let scale_factor = current_height as f32 / original_height as f32;
        let new_width = (original_width as f32 * scale_factor).floor() as u32;
        let padding = (current_width - new_width) / 2;
        (current_width - 2 * padding, current_height)
    }
}

/// Resize and pad image to target resolution, centering the image.
fn resize_and_pad_image(image: &DynamicImage, target: (u32, u32)) -> DynamicImage {
    let (original_width, original_height) = image.dimensions();
    let (target_width, target_height) = target;

    let scale_w = target_width as f32 / original_width as f32;
    let scale_h = target_height as f32 / original_height as f32;

    let (new_width, new_height) = if scale_w < scale_h {
        (
            target_width,
            std::cmp::min(
                (original_height as f32 * scale_w).ceil() as u32,
                target_height,
            ),
        )
    } else {
        (
            std::cmp::min(
                (original_width as f32 * scale_h).ceil() as u32,
                target_width,
            ),
            target_height,
        )
    };

    let resized = image.resize_exact(
        new_width,
        new_height,
        image::imageops::FilterType::CatmullRom,
    );

    let mut new_image = DynamicImage::new_rgb8(target_width, target_height);
    let paste_x = (target_width - new_width) as i64 / 2;
    let paste_y = (target_height - new_height) as i64 / 2;

    image::imageops::overlay(&mut new_image, &resized, paste_x, paste_y);
    new_image
}

/// Divide image into crops of specified size.
fn divide_to_samples(image: &DynamicImage, crop_size: (u32, u32)) -> Vec<DynamicImage> {
    let (width, height) = image.dimensions();
    let mut samples = Vec::new();

    for y in (0..height).step_by(crop_size.1 as usize) {
        for x in (0..width).step_by(crop_size.0 as usize) {
            let patch = image.crop_imm(x, y, crop_size.0, crop_size.1);
            samples.push(patch);
        }
    }
    samples
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_llava_processor_default() {
        let processor = LlavaProcessor::new();
        assert_eq!(processor.patch_size, 14);
        assert_eq!(processor.image_size, 336);
        assert_eq!(processor.aspect_ratio, ImageAspectRatio::Square);
    }

    #[test]
    fn test_llava_processor_with_pad() {
        let processor = LlavaProcessor::new_with_pad();
        assert_eq!(processor.patch_size, 14);
        assert_eq!(processor.image_size, 336);
        assert_eq!(processor.aspect_ratio, ImageAspectRatio::Pad);
    }

    #[test]
    fn test_llava_token_calculation() {
        let processor = LlavaProcessor::new();
        let config = PreProcessorConfig::default();

        // 336 / 14 = 24, 24 * 24 = 576
        let tokens = processor.calculate_num_tokens(336, 336, &config);
        assert_eq!(tokens, 576);
    }

    #[test]
    fn test_llava_preprocess_square() {
        let processor = LlavaProcessor::new();
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_center_crop: Some(true),
            do_normalize: Some(true),
            image_mean: Some(CLIP_MEAN.to_vec()),
            image_std: Some(CLIP_STD.to_vec()),
            ..Default::default()
        };

        let image = create_test_image(336, 336, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[image], &config).unwrap();

        assert_eq!(result.batch_size(), 1);
        assert_eq!(result.height(), 336);
        assert_eq!(result.width(), 336);
        assert_eq!(result.num_img_tokens[0], 576);
    }

    #[test]
    fn test_llava_preprocess_rectangular_square_mode() {
        // Square mode (default): resize shortest edge, center crop
        let processor = LlavaProcessor::new();
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_center_crop: Some(true),
            do_normalize: Some(true),
            size: Some([("shortest_edge".to_string(), 336)].into_iter().collect()),
            crop_size: Some(
                [("height".to_string(), 336), ("width".to_string(), 336)]
                    .into_iter()
                    .collect(),
            ),
            ..Default::default()
        };

        // Tall image - should be resized so shortest edge = 336, then center cropped
        let image = create_test_image(200, 400, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[image], &config).unwrap();

        assert_eq!(result.batch_size(), 1);
        assert_eq!(result.height(), 336);
        assert_eq!(result.width(), 336);
    }

    #[test]
    fn test_llava_preprocess_rectangular_pad_mode() {
        // Pad mode: expand to square with mean padding, then resize
        let processor = LlavaProcessor::new_with_pad();
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_center_crop: Some(false),
            do_normalize: Some(true),
            ..Default::default()
        };

        // Tall image should be padded to square first
        let image = create_test_image(200, 400, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[image], &config).unwrap();

        assert_eq!(result.batch_size(), 1);
        // After expand_to_square: 400x400, then resize to 336x336
        assert_eq!(result.height(), 336);
        assert_eq!(result.width(), 336);
    }

    #[test]
    fn test_select_best_resolution() {
        let pinpoints = vec![(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)];

        // Square image should pick square resolution
        let best = select_best_resolution((500, 500), &pinpoints);
        assert_eq!(best, (672, 672));

        // Wide image should pick wide resolution
        let best = select_best_resolution((800, 400), &pinpoints);
        assert_eq!(best, (672, 336));

        // Tall image should pick tall resolution
        let best = select_best_resolution((400, 800), &pinpoints);
        assert_eq!(best, (336, 672));
    }

    #[test]
    fn test_calculate_unpad() {
        // Square grid, wide original -> should reduce width padding
        let unpad = calculate_unpad((24, 24), (800, 400));
        assert!(unpad.0 >= unpad.1); // Width should be >= height

        // Square grid, tall original -> should reduce height padding
        let unpad = calculate_unpad((24, 24), (400, 800));
        assert!(unpad.1 >= unpad.0); // Height should be >= width
    }

    #[test]
    fn test_llava_next_processor_default() {
        let processor = LlavaNextProcessor::new();
        assert!(!processor.image_grid_pinpoints.is_empty());
        assert_eq!(processor.base.patch_size, 14);
    }

    #[test]
    fn test_llava_next_preprocess() {
        let processor = LlavaNextProcessor::new();
        let config = PreProcessorConfig {
            do_resize: Some(true),
            do_center_crop: Some(false),
            do_normalize: Some(true),
            ..Default::default()
        };

        let image = create_test_image(500, 500, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[image], &config).unwrap();

        // Should have multiple patches (original + crops)
        assert!(result.batch_size() > 1);
    }

    #[test]
    fn test_divide_to_samples() {
        let image = create_test_image(672, 672, Rgb([128, 128, 128]));
        let samples = divide_to_samples(&image, (336, 336));

        // 672x672 / 336x336 = 2x2 = 4 patches
        assert_eq!(samples.len(), 4);

        for sample in &samples {
            assert_eq!(sample.width(), 336);
            assert_eq!(sample.height(), 336);
        }
    }
}
