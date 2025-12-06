//! Phi4-Vision (Phi4-Multimodal) image processor.
//!
//! This module implements the Phi4-Vision image preprocessing pipeline with
//! Dynamic HD transform using a 448x448 base resolution.
//!
//! # Key Differences from Phi3-Vision
//!
//! | Feature | Phi3-Vision | Phi4-Vision |
//! |---------|-------------|-------------|
//! | Base resolution | 336 | 448 |
//! | Normalization | CLIP | [0.5, 0.5, 0.5] |
//! | Default max crops | 16 | 36 |
//! | Aspect ratio selection | Simple scale | Target ratio matching |
//! | Attention mask | No | Yes |
//!
//! # Processing Pipeline
//!
//! 1. **Dynamic Preprocess**: Calculate target resolution based on aspect ratio
//! 2. **Resize**: Scale to target resolution maintaining aspect ratio
//! 3. **Pad**: Add white padding to reach exact target dimensions
//! 4. **Normalize**: Apply [0.5, 0.5, 0.5] mean/std normalization
//! 5. **Create Global Image**: Bilinear interpolate to 448x448
//! 6. **Tile**: Reshape HD image into (h_crops * w_crops, 3, 448, 448) tiles
//! 7. **Concatenate**: [global_image, tiles...]
//! 8. **Generate Attention Mask**: Track valid (non-padding) regions
//!
//! # Token Count Formula
//!
//! `num_tokens = 256 + 1 + mask_sum + mask_col0_sum + 16`
//!
//! Where:
//! - 256: base global tokens
//! - mask_sum: sum of downsampled attention mask
//! - mask_col0_sum: sum of first column of mask (row separators)

use std::collections::HashSet;

use image::{imageops::FilterType, DynamicImage, GenericImageView, Rgb, RgbImage};
use ndarray::{s, Array2, Array3, Array4, IxDyn};

use crate::multimodal::vision::{
    image_processor::{ImagePreProcessor, ModelSpecificValue, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    transforms::{self, TransformError},
};

/// Simple normalization mean for Phi4-Vision.
pub const PHI4_MEAN: [f64; 3] = [0.5, 0.5, 0.5];

/// Simple normalization std for Phi4-Vision.
pub const PHI4_STD: [f64; 3] = [0.5, 0.5, 0.5];

/// Default dynamic_hd value (max crops) for Phi4-Vision.
pub const DEFAULT_DYNAMIC_HD: usize = 36;

/// Base resolution for Phi4-Vision (448x448).
pub const BASE_RESOLUTION: u32 = 448;

/// Mask resolution (base_resolution / patch_size = 448 / 14).
pub const MASK_RESOLUTION: usize = 32;

/// Patch size used in Phi4-Vision.
pub const PATCH_SIZE: usize = 14;

/// Result type for single image processing.
/// Contains (pixel_values, attention_mask, (height, width), num_tokens).
type SingleImageResult = (Array4<f32>, Array3<u32>, (u32, u32), usize);

/// Phi4-Vision image processor.
///
/// Implements Dynamic HD transform with aspect ratio matching.
#[derive(Debug, Clone)]
pub struct Phi4VisionProcessor {
    /// Maximum number of crops (dynamic_hd parameter).
    dynamic_hd: usize,
    /// Base resolution for tiles.
    base_resolution: u32,
    /// Mask resolution (base_resolution / patch_size).
    mask_resolution: usize,
    /// Normalization mean.
    mean: [f64; 3],
    /// Normalization std.
    std: [f64; 3],
}

impl Default for Phi4VisionProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Phi4VisionProcessor {
    /// Create a new Phi4-Vision processor with default settings.
    pub fn new() -> Self {
        Self {
            dynamic_hd: DEFAULT_DYNAMIC_HD,
            base_resolution: BASE_RESOLUTION,
            mask_resolution: MASK_RESOLUTION,
            mean: PHI4_MEAN,
            std: PHI4_STD,
        }
    }

    /// Create a processor with custom dynamic_hd setting.
    pub fn with_dynamic_hd(dynamic_hd: usize) -> Self {
        Self {
            dynamic_hd,
            base_resolution: BASE_RESOLUTION,
            mask_resolution: MASK_RESOLUTION,
            mean: PHI4_MEAN,
            std: PHI4_STD,
        }
    }

    /// Create a processor from preprocessor config.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            dynamic_hd: config.dynamic_hd.unwrap_or(DEFAULT_DYNAMIC_HD),
            base_resolution: BASE_RESOLUTION,
            mask_resolution: MASK_RESOLUTION,
            mean: config
                .image_mean
                .as_ref()
                .map(|v| [v[0], v[1], v[2]])
                .unwrap_or(PHI4_MEAN),
            std: config
                .image_std
                .as_ref()
                .map(|v| [v[0], v[1], v[2]])
                .unwrap_or(PHI4_STD),
        }
    }

    /// Get the dynamic_hd (max crops) value.
    pub fn dynamic_hd(&self) -> usize {
        self.dynamic_hd
    }

    /// Get the base resolution.
    pub fn base_resolution(&self) -> u32 {
        self.base_resolution
    }

    /// Compute valid target aspect ratios for the given crop range.
    ///
    /// Returns sorted list of (width_crops, height_crops) tuples where
    /// min_num <= w * h <= max_num.
    fn compute_target_ratios(&self, min_num: usize, max_num: usize) -> Vec<(usize, usize)> {
        let mut ratios: HashSet<(usize, usize)> = HashSet::new();
        for n in min_num..=max_num {
            // Find factor pairs by iterating up to sqrt(n)
            for i in 1..=(n as f64).sqrt() as usize {
                if n % i == 0 {
                    ratios.insert((i, n / i));
                    ratios.insert((n / i, i));
                }
            }
        }
        let mut sorted_ratios: Vec<(usize, usize)> = ratios.into_iter().collect();
        sorted_ratios.sort_by_key(|&(i, j)| i * j);
        sorted_ratios
    }

    /// Find the closest aspect ratio from the target ratios.
    ///
    /// Selects the ratio that minimizes the difference from the original
    /// aspect ratio. When tied, prefers ratios where the resized area
    /// exceeds half the base area product.
    fn find_closest_aspect_ratio(
        &self,
        aspect_ratio: f64,
        target_ratios: &[(usize, usize)],
        width: u32,
        height: u32,
    ) -> (usize, usize) {
        let mut best_ratio_diff = f64::INFINITY;
        let mut best_ratio = (1, 1);
        let area = (width * height) as f64;
        let base_area = (self.base_resolution * self.base_resolution) as f64;

        for &(w_ratio, h_ratio) in target_ratios {
            let target_aspect_ratio = w_ratio as f64 / h_ratio as f64;
            let ratio_diff = (aspect_ratio - target_aspect_ratio).abs();

            if ratio_diff < best_ratio_diff {
                best_ratio_diff = ratio_diff;
                best_ratio = (w_ratio, h_ratio);
            } else if (ratio_diff - best_ratio_diff).abs() < 1e-6 {
                // Tie-breaker: prefer ratio if area > 0.5 * base_area * w * h
                if area > 0.5 * base_area * (w_ratio * h_ratio) as f64 {
                    best_ratio = (w_ratio, h_ratio);
                }
            }
        }
        best_ratio
    }

    /// Dynamic preprocess: calculate target dimensions and create attention mask.
    ///
    /// Returns (processed_image, attention_mask, target_h_crops, target_w_crops)
    fn dynamic_preprocess(
        &self,
        image: &DynamicImage,
    ) -> Result<(DynamicImage, Array2<u32>, usize, usize), TransformError> {
        let (orig_w, orig_h) = image.dimensions();
        let base_res = self.base_resolution as f64;

        // Calculate natural crop numbers
        let w_crop_num = (orig_w as f64 / base_res).ceil() as usize;
        let h_crop_num = (orig_h as f64 / base_res).ceil() as usize;

        let (target_w_crops, target_h_crops, target_width, target_height) =
            if w_crop_num * h_crop_num > self.dynamic_hd {
                // Image exceeds max crops, need to find best aspect ratio
                let aspect_ratio = orig_w as f64 / orig_h as f64;
                let target_ratios = self.compute_target_ratios(1, self.dynamic_hd);
                let (w_ratio, h_ratio) =
                    self.find_closest_aspect_ratio(aspect_ratio, &target_ratios, orig_w, orig_h);

                let target_width = self.base_resolution * w_ratio as u32;
                let target_height = self.base_resolution * h_ratio as u32;
                (w_ratio, h_ratio, target_width, target_height)
            } else {
                // Image fits within max crops
                let target_width = self.base_resolution * w_crop_num as u32;
                let target_height = self.base_resolution * h_crop_num as u32;
                (w_crop_num, h_crop_num, target_width, target_height)
            };

        // Calculate resize ratios
        let ratio_width = target_width as f64 / orig_w as f64;
        let ratio_height = target_height as f64 / orig_h as f64;

        let (new_w, new_h, padding_width, padding_height) = if ratio_width < ratio_height {
            // Width is the limiting factor
            let new_w = target_width;
            let new_h = (orig_h as f64 * ratio_width) as u32;
            (new_w, new_h, 0u32, target_height - new_h)
        } else {
            // Height is the limiting factor
            let new_h = target_height;
            let new_w = (orig_w as f64 * ratio_height) as u32;
            (new_w, new_h, target_width - new_w, 0u32)
        };

        // Create attention mask (tracks valid regions)
        let mask_h = self.mask_resolution * target_h_crops;
        let mask_w = self.mask_resolution * target_w_crops;
        let mut attention_mask = Array2::<u32>::ones((mask_h, mask_w));

        // Mark padding regions as 0 in mask
        if padding_width >= PATCH_SIZE as u32 {
            let padding_mask_cols = (padding_width as usize) / PATCH_SIZE;
            for row in 0..mask_h {
                for col in (mask_w - padding_mask_cols)..mask_w {
                    attention_mask[[row, col]] = 0;
                }
            }
        }
        if padding_height >= PATCH_SIZE as u32 {
            let padding_mask_rows = (padding_height as usize) / PATCH_SIZE;
            for row in (mask_h - padding_mask_rows)..mask_h {
                for col in 0..mask_w {
                    attention_mask[[row, col]] = 0;
                }
            }
        }

        // Resize image with bilinear interpolation (matching HuggingFace torchvision)
        // HuggingFace uses torchvision.transforms.functional.resize with BILINEAR + antialias=True.
        // FilterType::Triangle (bilinear) closely matches this behavior.
        let resized = image.resize_exact(new_w, new_h, FilterType::Triangle);

        // Pad to target dimensions (white padding on right/bottom)
        let padded = self.pad_image(&resized, target_width, target_height);

        Ok((padded, attention_mask, target_h_crops, target_w_crops))
    }

    /// Pad image to target dimensions with white padding.
    fn pad_image(&self, image: &DynamicImage, target_w: u32, target_h: u32) -> DynamicImage {
        let (w, h) = image.dimensions();
        if w == target_w && h == target_h {
            return image.clone();
        }

        // Create white background
        let white = Rgb([255u8, 255, 255]);
        let mut padded = RgbImage::from_pixel(target_w, target_h, white);

        // Copy image to top-left using efficient overlay
        image::imageops::overlay(&mut padded, &image.to_rgb8(), 0, 0);

        DynamicImage::ImageRgb8(padded)
    }

    /// Create global image by bicubic interpolation to base resolution.
    ///
    /// Uses the shared `bicubic_resize` which matches PyTorch's
    /// `torch.nn.functional.interpolate(mode='bicubic', align_corners=False)`.
    fn create_global_image(&self, tensor: &Array3<f32>) -> Array3<f32> {
        let target = self.base_resolution as usize;
        transforms::bicubic_resize(tensor, target, target)
    }

    /// Tile the HD image into crops of base_resolution x base_resolution.
    fn tile_image(&self, tensor: &Array3<f32>, h_crops: usize, w_crops: usize) -> Array4<f32> {
        let base = self.base_resolution as usize;
        let num_tiles = h_crops * w_crops;

        let mut tiles = Array4::<f32>::zeros((num_tiles, 3, base, base));

        for h_idx in 0..h_crops {
            for w_idx in 0..w_crops {
                let tile_idx = h_idx * w_crops + w_idx;
                let y_start = h_idx * base;
                let x_start = w_idx * base;

                for c in 0..3 {
                    for y in 0..base {
                        for x in 0..base {
                            tiles[[tile_idx, c, y, x]] = tensor[[c, y_start + y, x_start + x]];
                        }
                    }
                }
            }
        }

        tiles
    }

    /// Downsample attention mask by factor of 2.
    fn downsample_mask(&self, mask: &Array2<u32>, h_crops: usize, w_crops: usize) -> Array2<u32> {
        let half_res = self.mask_resolution / 2;
        let out_h = h_crops * half_res;
        let out_w = w_crops * half_res;

        let mut downsampled = Array2::<u32>::zeros((out_h, out_w));

        for y in 0..out_h {
            for x in 0..out_w {
                // Sample every other pixel
                let src_y = y * 2;
                let src_x = x * 2;
                if src_y < mask.shape()[0] && src_x < mask.shape()[1] {
                    downsampled[[y, x]] = mask[[src_y, src_x]];
                }
            }
        }

        downsampled
    }

    /// Calculate number of image tokens.
    ///
    /// Formula: 256 + 1 + mask_sum + mask_col0_sum + 16
    /// - 256: global image tokens
    /// - 1: separator
    /// - mask_sum: sum of downsampled attention mask (valid HD tokens)
    /// - mask_col0_sum: sum of first column (row separators)
    /// - 16: additional fixed tokens
    fn calculate_num_tokens(&self, downsampled_mask: &Array2<u32>) -> usize {
        let mask_sum: u32 = downsampled_mask.iter().sum();
        let mask_col0_sum: u32 = downsampled_mask.column(0).iter().sum();
        256 + 1 + mask_sum as usize + mask_col0_sum as usize + 16
    }

    /// Process a single image.
    fn process_single_image(
        &self,
        image: &DynamicImage,
    ) -> Result<SingleImageResult, TransformError> {
        // Step 1: Dynamic preprocess (resize, pad, create attention mask)
        let (hd_image, attention_mask, h_crops, w_crops) = self.dynamic_preprocess(image)?;

        let hd_h = hd_image.height();
        let hd_w = hd_image.width();

        // Step 2: Convert to tensor and normalize
        let mut hd_tensor = transforms::to_tensor(&hd_image);
        transforms::normalize(&mut hd_tensor, &self.mean, &self.std);

        // Step 3: Create global image
        let global_tensor = self.create_global_image(&hd_tensor);

        // Step 4: Tile HD image
        let tiles = self.tile_image(&hd_tensor, h_crops, w_crops);
        let num_hd_tiles = h_crops * w_crops;

        // Step 5: Concatenate global + tiles
        // Output shape: [num_hd_tiles + 1, 3, base_resolution, base_resolution]
        let base = self.base_resolution as usize;
        let total_crops = num_hd_tiles + 1;
        let mut output = Array4::<f32>::zeros((total_crops, 3, base, base));

        // First slot is global image
        output.slice_mut(s![0, .., .., ..]).assign(&global_tensor);

        // Remaining slots are HD tiles
        if num_hd_tiles > 0 {
            output.slice_mut(s![1.., .., .., ..]).assign(&tiles);
        }

        // Step 6: Create combined attention mask [total_crops, mask_resolution, mask_resolution]
        let mask_res = self.mask_resolution;
        let mut combined_mask = Array3::<u32>::zeros((total_crops, mask_res, mask_res));

        // Global mask is all ones
        combined_mask.slice_mut(s![0, .., ..]).fill(1);

        // Tile attention masks
        for h_idx in 0..h_crops {
            for w_idx in 0..w_crops {
                let tile_idx = h_idx * w_crops + w_idx + 1; // +1 for global
                let mask_y_start = h_idx * mask_res;
                let mask_x_start = w_idx * mask_res;

                let tile_mask = attention_mask.slice(s![
                    mask_y_start..mask_y_start + mask_res,
                    mask_x_start..mask_x_start + mask_res
                ]);
                combined_mask
                    .slice_mut(s![tile_idx, .., ..])
                    .assign(&tile_mask);
            }
        }

        // Step 7: Calculate token count
        let downsampled = self.downsample_mask(&attention_mask, h_crops, w_crops);
        let num_tokens = self.calculate_num_tokens(&downsampled);

        Ok((output, combined_mask, (hd_h, hd_w), num_tokens))
    }
}

impl ImagePreProcessor for Phi4VisionProcessor {
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

        let processor = if config.dynamic_hd.is_some() || config.image_mean.is_some() {
            Self::from_preprocessor_config(config)
        } else {
            self.clone()
        };

        let mut all_outputs = Vec::new();
        let mut all_masks = Vec::new();
        let mut image_sizes = Vec::new();
        let mut num_img_tokens = Vec::new();

        for image in images {
            let (output, mask, size, tokens) = processor.process_single_image(image)?;
            all_outputs.push(output);
            all_masks.push(mask);
            image_sizes.push(size);
            num_img_tokens.push(tokens);
        }

        // Find max crops across batch for padding
        let max_crops = all_outputs.iter().map(|o| o.shape()[0]).max().unwrap();
        let base = self.base_resolution as usize;
        let mask_res = self.mask_resolution;

        // Pad all outputs to max_crops
        let batch_size = images.len();
        let mut pixel_values =
            ndarray::ArrayD::<f32>::zeros(IxDyn(&[batch_size, max_crops, 3, base, base]));
        let mut attention_masks =
            ndarray::ArrayD::<u32>::zeros(IxDyn(&[batch_size, max_crops, mask_res, mask_res]));

        for (b, (output, mask)) in all_outputs.iter().zip(all_masks.iter()).enumerate() {
            let num_crops = output.shape()[0];
            for t in 0..num_crops {
                for c in 0..3 {
                    for y in 0..base {
                        for x in 0..base {
                            pixel_values[[b, t, c, y, x]] = output[[t, c, y, x]];
                        }
                    }
                }
                for y in 0..mask_res {
                    for x in 0..mask_res {
                        attention_masks[[b, t, y, x]] = mask[[t, y, x]];
                    }
                }
            }
            // Remaining crops stay as zeros (padding)
        }

        // Convert to standard format
        let mut model_specific = std::collections::HashMap::new();

        // Store attention mask as model-specific data
        let mask_flat: Vec<u32> = attention_masks.iter().copied().collect();
        model_specific.insert(
            "pixel_attention_mask".to_string(),
            ModelSpecificValue::UintTensor {
                data: mask_flat,
                shape: vec![batch_size, max_crops, mask_res, mask_res],
            },
        );

        // Store image sizes (H, W after HD transform)
        let sizes_flat: Vec<u32> = image_sizes.iter().flat_map(|&(h, w)| vec![h, w]).collect();
        model_specific.insert(
            "image_sizes".to_string(),
            ModelSpecificValue::UintTensor {
                data: sizes_flat,
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
        let base_res = processor.base_resolution as f64;

        let w_crop_num = (width as f64 / base_res).ceil() as usize;
        let h_crop_num = (height as f64 / base_res).ceil() as usize;

        let (target_w_crops, target_h_crops) = if w_crop_num * h_crop_num > processor.dynamic_hd {
            let aspect_ratio = width as f64 / height as f64;
            let target_ratios = processor.compute_target_ratios(1, processor.dynamic_hd);
            processor.find_closest_aspect_ratio(aspect_ratio, &target_ratios, width, height)
        } else {
            (w_crop_num, h_crop_num)
        };

        // Approximate token count (without actual mask)
        // Full mask would have target_w_crops * target_h_crops * (mask_res/2)^2 tokens
        let half_res = processor.mask_resolution / 2;
        let mask_area = target_h_crops * target_w_crops * half_res * half_res;
        let mask_col0 = target_h_crops * half_res;

        256 + 1 + mask_area + mask_col0 + 16
    }

    fn model_name(&self) -> &'static str {
        "phi4-vision"
    }

    fn get_processed_size(&self, config: &PreProcessorConfig) -> Option<(u32, u32)> {
        // For Phi4, the size depends on the input image
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
    fn test_phi4_vision_processor_default() {
        let processor = Phi4VisionProcessor::new();
        assert_eq!(processor.dynamic_hd(), DEFAULT_DYNAMIC_HD);
        assert_eq!(processor.base_resolution(), BASE_RESOLUTION);
        assert_eq!(processor.mean, PHI4_MEAN);
        assert_eq!(processor.std, PHI4_STD);
    }

    #[test]
    fn test_compute_target_ratios() {
        let processor = Phi4VisionProcessor::new();

        let ratios = processor.compute_target_ratios(1, 4);
        // Should include (1,1), (1,2), (2,1), (1,3), (3,1), (2,2), (1,4), (4,1)
        assert!(ratios.contains(&(1, 1)));
        assert!(ratios.contains(&(2, 2)));
        assert!(ratios.contains(&(1, 4)));
        assert!(ratios.contains(&(4, 1)));
    }

    #[test]
    fn test_find_closest_aspect_ratio_square() {
        let processor = Phi4VisionProcessor::new();
        let ratios = processor.compute_target_ratios(1, 36);

        // Square image should get close to (1,1) or similar square ratio
        let result = processor.find_closest_aspect_ratio(1.0, &ratios, 500, 500);
        assert_eq!(result.0, result.1); // Should be square
    }

    #[test]
    fn test_find_closest_aspect_ratio_wide() {
        let processor = Phi4VisionProcessor::new();
        let ratios = processor.compute_target_ratios(1, 36);

        // Wide image (2:1 aspect ratio)
        let result = processor.find_closest_aspect_ratio(2.0, &ratios, 1000, 500);
        assert!(result.0 > result.1); // Width crops > height crops
    }

    #[test]
    fn test_find_closest_aspect_ratio_tall() {
        let processor = Phi4VisionProcessor::new();
        let ratios = processor.compute_target_ratios(1, 36);

        // Tall image (1:2 aspect ratio)
        let result = processor.find_closest_aspect_ratio(0.5, &ratios, 500, 1000);
        assert!(result.0 < result.1); // Width crops < height crops
    }

    #[test]
    fn test_pad_image() {
        let processor = Phi4VisionProcessor::new();
        let image = create_test_image(300, 200, Rgb([100, 100, 100]));

        let padded = processor.pad_image(&image, 448, 448);
        assert_eq!(padded.width(), 448);
        assert_eq!(padded.height(), 448);

        // Check original content is preserved
        let p = padded.get_pixel(100, 100);
        assert_eq!(p.0[0], 100);

        // Check padding is white
        let p = padded.get_pixel(400, 400);
        assert_eq!(p.0[0], 255);
    }

    #[test]
    fn test_preprocess_square_image() {
        let processor = Phi4VisionProcessor::new();
        let config = PreProcessorConfig::default();

        let image = create_test_image(500, 500, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[image], &config).unwrap();

        assert_eq!(result.batch_size(), 1);
        assert!(result.num_img_tokens[0] > 256); // At least global tokens

        // Check pixel values are normalized
        let flat = result.pixel_values_flat();
        assert!(flat.iter().all(|&v| (-1.5..=1.5).contains(&v)));
    }

    #[test]
    fn test_preprocess_wide_image() {
        let processor = Phi4VisionProcessor::new();
        let config = PreProcessorConfig::default();

        let image = create_test_image(1000, 500, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[image], &config).unwrap();

        assert_eq!(result.batch_size(), 1);
        // Wide image should have more crops in width direction
        assert!(result.image_sizes[0].1 >= result.image_sizes[0].0);
    }

    #[test]
    fn test_preprocess_multiple_images() {
        let processor = Phi4VisionProcessor::new();
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
    fn test_model_name() {
        let processor = Phi4VisionProcessor::new();
        assert_eq!(processor.model_name(), "phi4-vision");
    }

    #[test]
    fn test_normalization_values() {
        let processor = Phi4VisionProcessor::new();
        assert_eq!(processor.default_mean(), [0.5, 0.5, 0.5]);
        assert_eq!(processor.default_std(), [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_phi4_vs_phi3_differences() {
        // Verify key differences from Phi3
        let processor = Phi4VisionProcessor::new();

        // Phi4 uses 448 base resolution (vs 336 in Phi3)
        assert_eq!(processor.base_resolution(), 448);

        // Phi4 uses simple 0.5 normalization (vs CLIP in Phi3)
        assert_eq!(processor.mean, [0.5, 0.5, 0.5]);
        assert_eq!(processor.std, [0.5, 0.5, 0.5]);

        // Phi4 default dynamic_hd is 36 (vs 16 num_crops in Phi3)
        assert_eq!(processor.dynamic_hd(), 36);
    }
}
