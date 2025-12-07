//! LLaMA 4 Vision image processor.
//!
//! This module implements the LLaMA 4 Vision (Llama-4-Scout, Llama-4-Maverick) image preprocessing
//! pipeline with tile-based processing similar to other dynamic resolution models.
//!
//! # Key Features
//!
//! | Feature | Value |
//! |---------|-------|
//! | Tile size | 336x336 |
//! | Default max_patches | 16 |
//! | Normalization | [0.5, 0.5, 0.5] mean/std |
//! | Interpolation | Bilinear |
//! | Global tile | Added when num_tiles > 1 |
//!
//! # Processing Pipeline
//!
//! 1. **Find supported resolutions**: Calculate valid tile configurations
//! 2. **Get best fit**: Find optimal resolution without distortion
//! 3. **Resize**: Scale to target resolution maintaining aspect ratio
//! 4. **Pad**: Add black padding (0) to reach target dimensions
//! 5. **Normalize**: Apply [0.5, 0.5, 0.5] mean/std normalization
//! 6. **Tile**: Split into (num_tiles_h * num_tiles_w, 3, 336, 336) tiles
//! 7. **Global tile**: If multiple tiles, add global view at the end
//!
//! # Token Count
//!
//! For LLaMA 4, tokens = num_tiles * (tile_size / patch_size)²
//! where patch_size is typically 14, giving 576 tokens per tile.

use std::collections::HashSet;

use image::{imageops::FilterType, DynamicImage, GenericImageView, Rgb, RgbImage};
use ndarray::{s, Array3, Array4, IxDyn};

use crate::multimodal::vision::{
    image_processor::{ImagePreProcessor, ModelSpecificValue, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    transforms::{self, TransformError},
};

/// Default normalization mean for LLaMA 4 Vision.
pub const LLAMA4_MEAN: [f64; 3] = [0.5, 0.5, 0.5];

/// Default normalization std for LLaMA 4 Vision.
pub const LLAMA4_STD: [f64; 3] = [0.5, 0.5, 0.5];

/// Default tile size for LLaMA 4 Vision.
pub const TILE_SIZE: u32 = 336;

/// Default maximum number of patches/tiles.
pub const DEFAULT_MAX_PATCHES: usize = 16;

/// Patch size used in vision encoder.
pub const PATCH_SIZE: usize = 14;

/// LLaMA 4 Vision image processor.
///
/// Implements tile-based processing with dynamic resolution selection.
#[derive(Debug, Clone)]
pub struct Llama4VisionProcessor {
    /// Tile size (both height and width).
    tile_size: u32,
    /// Maximum number of tiles/patches.
    max_patches: usize,
    /// Whether to resize to max canvas (upscale aggressively).
    resize_to_max_canvas: bool,
    /// Normalization mean.
    mean: [f64; 3],
    /// Normalization std.
    std: [f64; 3],
}

impl Default for Llama4VisionProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Llama4VisionProcessor {
    /// Create a new LLaMA 4 Vision processor with default settings.
    pub fn new() -> Self {
        Self {
            tile_size: TILE_SIZE,
            max_patches: DEFAULT_MAX_PATCHES,
            resize_to_max_canvas: false,
            mean: LLAMA4_MEAN,
            std: LLAMA4_STD,
        }
    }

    /// Create a processor with custom max_patches setting.
    pub fn with_max_patches(max_patches: usize) -> Self {
        Self {
            tile_size: TILE_SIZE,
            max_patches,
            resize_to_max_canvas: false,
            mean: LLAMA4_MEAN,
            std: LLAMA4_STD,
        }
    }

    /// Create a processor from preprocessor config.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            tile_size: config
                .size
                .as_ref()
                .and_then(|s| s.get("height").copied())
                .unwrap_or(TILE_SIZE),
            max_patches: config.max_image_tiles.unwrap_or(DEFAULT_MAX_PATCHES),
            resize_to_max_canvas: false,
            mean: config
                .image_mean
                .as_ref()
                .map(|v| [v[0], v[1], v[2]])
                .unwrap_or(LLAMA4_MEAN),
            std: config
                .image_std
                .as_ref()
                .map(|v| [v[0], v[1], v[2]])
                .unwrap_or(LLAMA4_STD),
        }
    }

    /// Get the tile size.
    pub fn tile_size(&self) -> u32 {
        self.tile_size
    }

    /// Get the max patches setting.
    pub fn max_patches(&self) -> usize {
        self.max_patches
    }

    /// Get all factors of a number.
    fn get_factors(n: usize) -> HashSet<usize> {
        let mut factors = HashSet::new();
        for i in 1..=(n as f64).sqrt() as usize {
            if n.is_multiple_of(i) {
                factors.insert(i);
                factors.insert(n / i);
            }
        }
        factors
    }

    /// Find all supported resolutions for the given max_patches.
    ///
    /// Returns list of (height, width) in pixels.
    fn find_supported_resolutions(&self) -> Vec<(u32, u32)> {
        let mut resolutions = Vec::new();
        let tile = self.tile_size;

        // For each possible number of chunks from max_patches down to 1
        for chunk_size in (1..=self.max_patches).rev() {
            let factors = Self::get_factors(chunk_size);
            for &factor in &factors {
                let h_tiles = factor;
                let w_tiles = chunk_size / factor;
                resolutions.push((h_tiles as u32 * tile, w_tiles as u32 * tile));
            }
        }

        resolutions
    }

    /// Get the maximum resolution without distortion.
    ///
    /// Given an image size and target size, compute the largest size
    /// that fits within target while maintaining aspect ratio.
    fn get_max_res_without_distortion(
        image_size: (u32, u32),
        target_size: (u32, u32),
    ) -> (u32, u32) {
        let (orig_h, orig_w) = image_size;
        let (target_h, target_w) = target_size;

        let scale_w = target_w as f64 / orig_w as f64;
        let scale_h = target_h as f64 / orig_h as f64;

        if scale_w < scale_h {
            let new_w = target_w;
            let new_h = (orig_h as f64 * scale_w).floor() as u32;
            (new_h.min(target_h), new_w)
        } else {
            let new_h = target_h;
            let new_w = (orig_w as f64 * scale_h).floor() as u32;
            (new_h, new_w.min(target_w))
        }
    }

    /// Find the best fitting resolution from supported resolutions.
    ///
    /// Selects resolution that:
    /// - Minimizes upscaling if possible (unless resize_to_max_canvas)
    /// - Minimizes downscaling if no upscaling possible
    /// - Minimizes padding area when tied
    fn get_best_fit(&self, image_size: (u32, u32)) -> (u32, u32) {
        let resolutions = self.find_supported_resolutions();
        let (orig_h, orig_w) = image_size;

        // Calculate scaling factors for each resolution
        let scales_and_resolutions: Vec<(f64, (u32, u32))> = resolutions
            .iter()
            .map(|&(target_h, target_w)| {
                let scale_w = target_w as f64 / orig_w as f64;
                let scale_h = target_h as f64 / orig_h as f64;
                // Limiting scale is the minimum (the side that constrains)
                let scale = scale_w.min(scale_h);
                (scale, (target_h, target_w))
            })
            .collect();

        // Separate upscaling and downscaling options
        let upscaling: Vec<_> = scales_and_resolutions
            .iter()
            .filter(|(s, _)| *s >= 1.0)
            .cloned()
            .collect();

        let selected_scale = if !upscaling.is_empty() {
            if self.resize_to_max_canvas {
                // Pick largest upscaling
                upscaling
                    .iter()
                    .map(|(s, _)| *s)
                    .fold(f64::NEG_INFINITY, f64::max)
            } else {
                // Pick smallest upscaling (minimum distortion)
                upscaling
                    .iter()
                    .map(|(s, _)| *s)
                    .fold(f64::INFINITY, f64::min)
            }
        } else {
            // No upscaling possible, pick largest downscaling (minimum reduction)
            scales_and_resolutions
                .iter()
                .filter(|(s, _)| *s < 1.0)
                .map(|(s, _)| *s)
                .fold(f64::NEG_INFINITY, f64::max)
        };

        // Get all resolutions with the selected scale
        let candidates: Vec<_> = scales_and_resolutions
            .iter()
            .filter(|(s, _)| (*s - selected_scale).abs() < 1e-9)
            .map(|(_, res)| *res)
            .collect();

        // If multiple candidates, pick the one with minimum area (less padding)
        if candidates.len() > 1 {
            *candidates
                .iter()
                .min_by_key(|(h, w)| h * w)
                .unwrap_or(&candidates[0])
        } else {
            candidates[0]
        }
    }

    /// Pad image to target dimensions with black padding.
    fn pad_image(&self, image: &DynamicImage, target_w: u32, target_h: u32) -> DynamicImage {
        let (w, h) = image.dimensions();
        if w == target_w && h == target_h {
            return image.clone();
        }

        // Create black background (LLaMA 4 uses 0 for padding)
        let black = Rgb([0u8, 0, 0]);
        let mut padded = RgbImage::from_pixel(target_w, target_h, black);

        // Copy image to top-left using efficient overlay
        image::imageops::overlay(&mut padded, &image.to_rgb8(), 0, 0);

        DynamicImage::ImageRgb8(padded)
    }

    /// Split image tensor into tiles.
    fn split_to_tiles(
        &self,
        tensor: &Array3<f32>,
        num_tiles_h: usize,
        num_tiles_w: usize,
    ) -> Array4<f32> {
        let tile = self.tile_size as usize;
        let num_tiles = num_tiles_h * num_tiles_w;

        let mut tiles = Array4::<f32>::zeros((num_tiles, 3, tile, tile));

        for h_idx in 0..num_tiles_h {
            for w_idx in 0..num_tiles_w {
                let tile_idx = h_idx * num_tiles_w + w_idx;
                let y_start = h_idx * tile;
                let x_start = w_idx * tile;

                let tile_view =
                    tensor.slice(s![.., y_start..y_start + tile, x_start..x_start + tile]);
                tiles.slice_mut(s![tile_idx, .., .., ..]).assign(&tile_view);
            }
        }

        tiles
    }

    /// Create global image by bilinear interpolation to tile size.
    fn create_global_image(&self, image: &DynamicImage) -> Array3<f32> {
        let tile = self.tile_size;
        let resized = image.resize_exact(tile, tile, FilterType::Triangle);
        let mut tensor = transforms::to_tensor(&resized);
        transforms::normalize(&mut tensor, &self.mean, &self.std);
        tensor
    }

    /// Process a single image.
    fn process_single_image(
        &self,
        image: &DynamicImage,
    ) -> Result<(Array4<f32>, (usize, usize)), TransformError> {
        let (orig_w, orig_h) = image.dimensions();
        let image_size = (orig_h, orig_w);

        // Step 1: Find best fit resolution (canvas size for padding/tiling)
        let target_size = self.get_best_fit(image_size);
        let (target_h, target_w) = target_size;

        // Step 2: Compute resize target - limit upscaling if not resize_to_max_canvas
        // This limits how much we resize the image, but we still pad to target_size
        let resize_target = if !self.resize_to_max_canvas {
            let tile = self.tile_size;
            let new_target_h = target_h.min(orig_h.max(tile));
            let new_target_w = target_w.min(orig_w.max(tile));
            (new_target_h, new_target_w)
        } else {
            target_size
        };

        // Step 3: Resize preserving aspect ratio to fit within resize_target
        let new_size = Self::get_max_res_without_distortion(image_size, resize_target);
        let (new_h, new_w) = (new_size.0.max(1), new_size.1.max(1));

        let resized = image.resize_exact(new_w, new_h, FilterType::Triangle);

        // Step 4: Pad to target_size (the canvas from get_best_fit, not resize_target)
        let padded = self.pad_image(&resized, target_w, target_h);

        // Step 5: Convert to tensor and normalize
        let mut tensor = transforms::to_tensor(&padded);
        transforms::normalize(&mut tensor, &self.mean, &self.std);

        // Step 6: Calculate tile counts based on target_size (canvas size)
        let tile = self.tile_size as usize;
        let num_tiles_h = target_h as usize / tile;
        let num_tiles_w = target_w as usize / tile;

        // Step 7: Split into tiles
        let tiles = self.split_to_tiles(&tensor, num_tiles_h, num_tiles_w);
        let num_tiles = num_tiles_h * num_tiles_w;

        // Step 8: Add global tile if there are multiple tiles
        let output = if num_tiles > 1 {
            let global_tile = self.create_global_image(image);
            let mut combined = Array4::<f32>::zeros((num_tiles + 1, 3, tile, tile));
            combined
                .slice_mut(s![..num_tiles, .., .., ..])
                .assign(&tiles);
            combined
                .slice_mut(s![num_tiles, .., .., ..])
                .assign(&global_tile);
            combined
        } else {
            tiles
        };

        Ok((output, (num_tiles_h, num_tiles_w)))
    }

    /// Calculate number of image tokens for a given aspect ratio.
    pub fn calculate_num_tokens_for_aspect_ratio(&self, aspect_ratio: (usize, usize)) -> usize {
        let (h_tiles, w_tiles) = aspect_ratio;
        let num_tiles = h_tiles * w_tiles;
        // Add 1 for global tile if num_tiles > 1
        let total_tiles = if num_tiles > 1 {
            num_tiles + 1
        } else {
            num_tiles
        };
        let tokens_per_tile = (self.tile_size as usize / PATCH_SIZE).pow(2);
        total_tiles * tokens_per_tile
    }
}

impl ImagePreProcessor for Llama4VisionProcessor {
    fn default_mean(&self) -> [f64; 3] {
        self.mean
    }

    fn default_std(&self) -> [f64; 3] {
        self.std
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
        config: &PreProcessorConfig,
    ) -> Result<PreprocessedImages, TransformError> {
        if images.is_empty() {
            return Err(TransformError::InvalidShape {
                expected: "non-empty image batch".to_string(),
                actual: vec![0],
            });
        }

        let processor = if config.max_image_tiles.is_some()
            || config.image_mean.is_some()
            || config.image_std.is_some()
            || config.size.is_some()
        {
            Self::from_preprocessor_config(config)
        } else {
            self.clone()
        };

        let mut all_outputs = Vec::new();
        let mut all_aspect_ratios = Vec::new();
        let mut image_sizes = Vec::new();
        let mut num_img_tokens = Vec::new();

        for image in images {
            let (output, aspect_ratio) = processor.process_single_image(image)?;
            let tokens = processor.calculate_num_tokens_for_aspect_ratio(aspect_ratio);

            all_outputs.push(output);
            all_aspect_ratios.push(aspect_ratio);
            image_sizes.push((image.height(), image.width()));
            num_img_tokens.push(tokens);
        }

        // Find max tiles across batch for padding
        let max_tiles = all_outputs.iter().map(|o| o.shape()[0]).max().unwrap();
        let tile = self.tile_size as usize;

        // Pad all outputs to max_tiles
        let batch_size = images.len();
        let mut pixel_values =
            ndarray::ArrayD::<f32>::zeros(IxDyn(&[batch_size, max_tiles, 3, tile, tile]));

        for (b, output) in all_outputs.iter().enumerate() {
            let num_tiles = output.shape()[0];
            for t in 0..num_tiles {
                pixel_values
                    .slice_mut(s![b, t, .., .., ..])
                    .assign(&output.slice(s![t, .., .., ..]));
            }
            // Remaining tiles stay as zeros (padding)
        }

        // Store aspect ratios as model-specific data
        let mut model_specific = std::collections::HashMap::new();

        let aspect_ratios_flat: Vec<u32> = all_aspect_ratios
            .iter()
            .flat_map(|&(h, w)| vec![h as u32, w as u32])
            .collect();
        model_specific.insert(
            "aspect_ratios".to_string(),
            ModelSpecificValue::UintTensor {
                data: aspect_ratios_flat,
                shape: vec![batch_size, 2],
            },
        );

        Ok(PreprocessedImages {
            pixel_values: pixel_values.into_dyn(),
            num_img_tokens,
            image_sizes,
            model_specific,
        })
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        let processor = Self::from_preprocessor_config(config);
        let image_size = (height, width);
        // target_size from get_best_fit determines the canvas and tile count
        let target_size = processor.get_best_fit(image_size);

        let tile = processor.tile_size as usize;
        let num_tiles_h = target_size.0 as usize / tile;
        let num_tiles_w = target_size.1 as usize / tile;

        processor.calculate_num_tokens_for_aspect_ratio((num_tiles_h, num_tiles_w))
    }

    fn model_name(&self) -> &'static str {
        "llama4-vision"
    }

    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        // For LLaMA 4, the size depends on the input image
        let _ = config;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_llama4_vision_processor_default() {
        let processor = Llama4VisionProcessor::new();
        assert_eq!(processor.tile_size(), TILE_SIZE);
        assert_eq!(processor.max_patches(), DEFAULT_MAX_PATCHES);
        assert_eq!(processor.mean, LLAMA4_MEAN);
        assert_eq!(processor.std, LLAMA4_STD);
    }

    #[test]
    fn test_get_factors() {
        let factors = Llama4VisionProcessor::get_factors(12);
        assert!(factors.contains(&1));
        assert!(factors.contains(&2));
        assert!(factors.contains(&3));
        assert!(factors.contains(&4));
        assert!(factors.contains(&6));
        assert!(factors.contains(&12));
        assert_eq!(factors.len(), 6);
    }

    #[test]
    fn test_find_supported_resolutions() {
        let processor = Llama4VisionProcessor::with_max_patches(4);
        let resolutions = processor.find_supported_resolutions();

        // Should include 1x1, 1x2, 2x1, 1x3, 3x1, 2x2, 1x4, 4x1
        let expected: Vec<(u32, u32)> = vec![
            (336, 336),  // 1x1
            (336, 672),  // 1x2
            (672, 336),  // 2x1
            (336, 1008), // 1x3
            (1008, 336), // 3x1
            (672, 672),  // 2x2
            (336, 1344), // 1x4
            (1344, 336), // 4x1
        ];

        for exp in expected {
            assert!(
                resolutions.contains(&exp),
                "Expected resolution {:?} not found",
                exp
            );
        }
    }

    #[test]
    fn test_get_best_fit_square() {
        let processor = Llama4VisionProcessor::new();
        let best = processor.get_best_fit((500, 500));
        // Square image should get a square or near-square resolution
        assert!(best.0 == best.1 || (best.0 as i32 - best.1 as i32).abs() <= 336);
    }

    #[test]
    fn test_get_best_fit_wide() {
        let processor = Llama4VisionProcessor::new();
        let best = processor.get_best_fit((300, 900));
        // Wide image should get wider resolution
        assert!(best.1 >= best.0);
    }

    #[test]
    fn test_get_best_fit_tall() {
        let processor = Llama4VisionProcessor::new();
        let best = processor.get_best_fit((900, 300));
        // Tall image should get taller resolution
        assert!(best.0 >= best.1);
    }

    #[test]
    fn test_preprocess_square_image() {
        let processor = Llama4VisionProcessor::new();
        let config = PreProcessorConfig::default();

        let image = create_test_image(500, 500, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[image], &config).unwrap();

        assert_eq!(result.batch_size(), 1);
        assert!(result.num_img_tokens[0] > 0);

        // Check pixel values are normalized
        let flat = result.pixel_values_flat();
        assert!(flat.iter().all(|&v| (-1.5..=1.5).contains(&v)));
    }

    #[test]
    fn test_preprocess_wide_image() {
        let processor = Llama4VisionProcessor::new();
        let config = PreProcessorConfig::default();

        let image = create_test_image(1000, 300, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[image], &config).unwrap();

        assert_eq!(result.batch_size(), 1);
        // Wide image should have more tiles in width direction
        let aspect_ratios = result.model_specific.get("aspect_ratios").unwrap();
        if let ModelSpecificValue::UintTensor { data, .. } = aspect_ratios {
            let h_tiles = data[0];
            let w_tiles = data[1];
            assert!(w_tiles >= h_tiles);
        }
    }

    #[test]
    fn test_preprocess_multiple_images() {
        let processor = Llama4VisionProcessor::new();
        let config = PreProcessorConfig::default();

        let images = vec![
            create_test_image(500, 500, Rgb([100, 100, 100])),
            create_test_image(800, 400, Rgb([150, 150, 150])),
        ];

        let result = processor.preprocess(&images, &config).unwrap();

        assert_eq!(result.batch_size(), 2);
        assert_eq!(result.image_sizes.len(), 2);
        assert_eq!(result.num_img_tokens.len(), 2);
    }

    #[test]
    fn test_global_tile_added_for_multiple_tiles() {
        let processor = Llama4VisionProcessor::new();
        let config = PreProcessorConfig::default();

        // Large image that will require multiple tiles
        let image = create_test_image(1000, 1000, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[image], &config).unwrap();

        let aspect_ratios = result.model_specific.get("aspect_ratios").unwrap();
        if let ModelSpecificValue::UintTensor { data, .. } = aspect_ratios {
            let h_tiles = data[0] as usize;
            let w_tiles = data[1] as usize;
            let num_tiles = h_tiles * w_tiles;

            if num_tiles > 1 {
                // Output should have num_tiles + 1 (for global tile)
                let shape = result.pixel_values.shape();
                assert_eq!(shape[1], num_tiles + 1);
            }
        }
    }

    #[test]
    fn test_model_name() {
        let processor = Llama4VisionProcessor::new();
        assert_eq!(processor.model_name(), "llama4-vision");
    }

    #[test]
    fn test_normalization_values() {
        let processor = Llama4VisionProcessor::new();
        assert_eq!(processor.default_mean(), [0.5, 0.5, 0.5]);
        assert_eq!(processor.default_std(), [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_token_count_calculation() {
        let processor = Llama4VisionProcessor::new();
        // 1x1 tile: 576 tokens
        assert_eq!(processor.calculate_num_tokens_for_aspect_ratio((1, 1)), 576);
        // 2x2 tiles + 1 global: 5 * 576 = 2880 tokens
        assert_eq!(
            processor.calculate_num_tokens_for_aspect_ratio((2, 2)),
            2880
        );
        // 1x2 tiles + 1 global: 3 * 576 = 1728 tokens
        assert_eq!(
            processor.calculate_num_tokens_for_aspect_ratio((1, 2)),
            1728
        );
    }
}
