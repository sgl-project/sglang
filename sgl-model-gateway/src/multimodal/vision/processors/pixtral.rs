//! Pixtral/Mistral3 Vision image processor implementation.
//!
//! This module implements the image preprocessing for Pixtral/Mistral3 models,
//! matching the behavior of HuggingFace's `PixtralImageProcessor`.
//!
//! Key characteristics:
//! - CLIP normalization: mean [0.48145466, 0.4578275, 0.40821073], std [0.26862954, 0.26130258, 0.27577711]
//! - Bicubic resampling for resize
//! - Images resized to fit within longest_edge (default 1024)
//! - Output dimensions are multiples of patch_size (default 16)
//! - No tiling - single image output per input

use std::collections::HashMap;

use image::{imageops::FilterType, DynamicImage};
use ndarray::{Array4, IxDyn};

use crate::multimodal::vision::{
    image_processor::{ImagePreProcessor, ModelSpecificValue, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    transforms::{self, TransformError},
};

/// Default normalization mean values (CLIP)
const DEFAULT_IMAGE_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];

/// Default normalization std values (CLIP)
const DEFAULT_IMAGE_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Default longest edge for resize
const DEFAULT_LONGEST_EDGE: u32 = 1024;

/// Default patch size
const DEFAULT_PATCH_SIZE: u32 = 16;

/// Pixtral/Mistral3 Vision image processor.
///
/// This processor handles image preprocessing for Pixtral and Mistral3 vision models.
/// Unlike tile-based processors (Phi3, LLaMA4), Pixtral processes images at their
/// natural resolution (up to a maximum), preserving aspect ratio.
#[derive(Debug, Clone)]
pub struct PixtralProcessor {
    /// Maximum dimension for the longest edge
    longest_edge: u32,
    /// Patch size for calculating output dimensions
    patch_size: u32,
    /// Normalization mean values
    image_mean: [f64; 3],
    /// Normalization std values
    image_std: [f64; 3],
}

impl Default for PixtralProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl PixtralProcessor {
    /// Creates a new Pixtral processor with default settings.
    pub fn new() -> Self {
        Self {
            longest_edge: DEFAULT_LONGEST_EDGE,
            patch_size: DEFAULT_PATCH_SIZE,
            image_mean: DEFAULT_IMAGE_MEAN,
            image_std: DEFAULT_IMAGE_STD,
        }
    }

    /// Creates a processor from a HuggingFace preprocessor config.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        let longest_edge = config
            .size
            .as_ref()
            .and_then(|s| s.get("longest_edge").copied())
            .unwrap_or(DEFAULT_LONGEST_EDGE);

        // Patch size uses the new PatchSize type from config
        let patch_size = config.get_patch_size(DEFAULT_PATCH_SIZE as usize) as u32;

        let image_mean = config
            .image_mean
            .as_ref()
            .filter(|m| m.len() >= 3)
            .map(|m| [m[0], m[1], m[2]])
            .unwrap_or(DEFAULT_IMAGE_MEAN);

        let image_std = config
            .image_std
            .as_ref()
            .filter(|s| s.len() >= 3)
            .map(|s| [s[0], s[1], s[2]])
            .unwrap_or(DEFAULT_IMAGE_STD);

        Self {
            longest_edge,
            patch_size,
            image_mean,
            image_std,
        }
    }

    /// Calculates the target output size for an image.
    ///
    /// The image is resized to fit within `longest_edge` while preserving aspect ratio.
    /// The output dimensions are then adjusted to be multiples of `patch_size`.
    fn get_resize_output_size(&self, height: u32, width: u32) -> (u32, u32) {
        let max_size = self.longest_edge;
        let patch_size = self.patch_size;

        // Calculate ratio for scaling down (only if larger than max_size)
        let ratio = f64::max(
            height as f64 / max_size as f64,
            width as f64 / max_size as f64,
        );

        let (new_height, new_width) = if ratio > 1.0 {
            // Scale down using floor to ensure we don't exceed max_size
            let new_height = (height as f64 / ratio).floor() as u32;
            let new_width = (width as f64 / ratio).floor() as u32;
            (new_height, new_width)
        } else {
            (height, width)
        };

        // Calculate number of patches in each dimension
        // Using: num_tokens = (dim - 1) / patch_size + 1 (i.e., ceiling division)
        let num_height_tokens = (new_height.max(1) - 1) / patch_size + 1;
        let num_width_tokens = (new_width.max(1) - 1) / patch_size + 1;

        // Final size is patches * patch_size
        (
            num_height_tokens * patch_size,
            num_width_tokens * patch_size,
        )
    }

    /// Processes a single image through the Pixtral pipeline.
    fn process_single_image(
        &self,
        image: &DynamicImage,
    ) -> Result<(Array4<f32>, (usize, usize)), TransformError> {
        let (orig_width, orig_height) = (image.width(), image.height());

        // Step 1: Calculate output size
        let (target_h, target_w) = self.get_resize_output_size(orig_height, orig_width);

        // Step 2: Resize image using bicubic interpolation
        let resized = image.resize_exact(target_w, target_h, FilterType::CatmullRom);

        // Step 3: Convert to tensor (0-1 range) and normalize
        let mut tensor = transforms::to_tensor(&resized);
        transforms::normalize(&mut tensor, &self.image_mean, &self.image_std);

        // Step 4: Reshape to (1, C, H, W)
        let (c, h, w) = (tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);
        let output = tensor
            .into_shape_with_order((1, c, h, w))
            .map_err(|e| TransformError::ShapeError(e.to_string()))?;

        Ok((output, (target_h as usize, target_w as usize)))
    }
}

impl ImagePreProcessor for PixtralProcessor {
    fn default_mean(&self) -> [f64; 3] {
        self.image_mean
    }

    fn default_std(&self) -> [f64; 3] {
        self.image_std
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

        // Apply config overrides if present
        let processor = if config.size.is_some()
            || config.patch_size.is_some()
            || config.image_mean.is_some()
            || config.image_std.is_some()
        {
            Self::from_preprocessor_config(config)
        } else {
            self.clone()
        };

        let mut all_pixel_values = Vec::new();
        let mut all_image_sizes = Vec::new();
        let mut original_sizes = Vec::new();
        let mut num_img_tokens = Vec::new();

        for image in images {
            let (pixels, size) = processor.process_single_image(image)?;
            let tokens = processor.calculate_num_tokens(image.width(), image.height(), config);

            all_pixel_values.push(pixels);
            all_image_sizes.push(size);
            original_sizes.push((image.height(), image.width()));
            num_img_tokens.push(tokens);
        }

        // Pad images to the same size for batching
        let max_height = all_image_sizes.iter().map(|(h, _)| *h).max().unwrap_or(0);
        let max_width = all_image_sizes.iter().map(|(_, w)| *w).max().unwrap_or(0);

        // Create batch tensor with padding
        let batch_size = all_pixel_values.len();
        let channels = 3;
        let mut batch_tensor =
            ndarray::ArrayD::<f32>::zeros(IxDyn(&[batch_size, channels, max_height, max_width]));

        for (i, (pixels, (h, w))) in all_pixel_values
            .iter()
            .zip(all_image_sizes.iter())
            .enumerate()
        {
            // Copy the image data into the batch (top-left aligned, zero-padded)
            for c in 0..channels {
                for y in 0..*h {
                    for x in 0..*w {
                        batch_tensor[[i, c, y, x]] = pixels[[0, c, y, x]];
                    }
                }
            }
        }

        // Store image sizes as model-specific data
        let mut model_specific = HashMap::new();
        let image_sizes_flat: Vec<i64> = all_image_sizes
            .iter()
            .flat_map(|&(h, w)| vec![h as i64, w as i64])
            .collect();
        model_specific.insert(
            "image_sizes".to_string(),
            ModelSpecificValue::IntTensor {
                data: image_sizes_flat,
                shape: vec![batch_size, 2],
            },
        );

        Ok(PreprocessedImages {
            pixel_values: batch_tensor,
            num_img_tokens,
            image_sizes: original_sizes,
            model_specific,
        })
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, config: &PreProcessorConfig) -> usize {
        let processor = Self::from_preprocessor_config(config);
        let (target_h, target_w) = processor.get_resize_output_size(height, width);
        let patch_size = processor.patch_size;

        // Number of tokens = num_patches_h * num_patches_w
        let num_patches_h = target_h / patch_size;
        let num_patches_w = target_w / patch_size;
        (num_patches_h * num_patches_w) as usize
    }

    fn model_name(&self) -> &'static str {
        "pixtral"
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        // Pixtral has dynamic size based on input
        None
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::*;

    fn create_test_image(width: u32, height: u32) -> DynamicImage {
        let mut img = RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r = ((x * 255) / width.max(1)) as u8;
                let g = ((y * 255) / height.max(1)) as u8;
                let b = (((x + y) * 128) / (width + height).max(1)) as u8;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_resize_output_size_small_image() {
        let processor = PixtralProcessor::new();

        // Small image that doesn't need resizing - just pad to patch boundary
        // 100x100 -> patches: ceil(100/16) = 7, output: 7*16 = 112
        let (h, w) = processor.get_resize_output_size(100, 100);
        assert_eq!((h, w), (112, 112));
    }

    #[test]
    fn test_resize_output_size_large_image() {
        let processor = PixtralProcessor::new();

        // Large image that needs resizing
        // 2048x1024: ratio = 2048/1024 = 2.0
        // scaled: 2048/2 = 1024, 1024/2 = 512
        // patches h: ceil(1024/16) = 64, patches w: ceil(512/16) = 32
        // output: 64*16 = 1024, 32*16 = 512
        let (h, w) = processor.get_resize_output_size(2048, 1024);
        assert_eq!((h, w), (1024, 512));
    }

    #[test]
    fn test_resize_output_size_at_limit() {
        let processor = PixtralProcessor::new();

        // Image exactly at limit
        // 1024x768: ratio = max(1024/1024, 768/1024) = 1.0
        // No resize needed
        // patches h: ceil(1024/16) = 64, patches w: ceil(768/16) = 48
        // output: 64*16 = 1024, 48*16 = 768
        let (h, w) = processor.get_resize_output_size(1024, 768);
        assert_eq!((h, w), (1024, 768));
    }

    #[test]
    fn test_process_single_image() {
        let processor = PixtralProcessor::new();
        let image = create_test_image(200, 150);

        let (tensor, size) = processor.process_single_image(&image).unwrap();

        // 200x150 -> patches h: ceil(150/16) = 10, patches w: ceil(200/16) = 13
        // output: 10*16 = 160, 13*16 = 208
        assert_eq!(size, (160, 208));
        assert_eq!(tensor.shape(), &[1, 3, 160, 208]);
    }

    #[test]
    fn test_preprocess_batch() {
        let processor = PixtralProcessor::new();
        let config = PreProcessorConfig::default();

        let images = vec![create_test_image(200, 150), create_test_image(300, 100)];

        let result = processor.preprocess(&images, &config).unwrap();

        // First image: 150x200 -> 160x208
        // Second image: 100x300 -> 112x304 (ceil(100/16)=7, ceil(300/16)=19)
        // Batch padded to max: 160x304
        assert_eq!(result.pixel_values.shape()[0], 2); // batch size
        assert_eq!(result.pixel_values.shape()[1], 3); // channels
    }

    #[test]
    fn test_normalization_values() {
        let processor = PixtralProcessor::new();

        // Verify CLIP normalization values
        assert!((processor.image_mean[0] - 0.48145466).abs() < 1e-6);
        assert!((processor.image_mean[1] - 0.4578275).abs() < 1e-6);
        assert!((processor.image_mean[2] - 0.40821073).abs() < 1e-6);

        assert!((processor.image_std[0] - 0.26862954).abs() < 1e-6);
        assert!((processor.image_std[1] - 0.26130258).abs() < 1e-6);
        assert!((processor.image_std[2] - 0.27577711).abs() < 1e-6);
    }

    #[test]
    fn test_from_config() {
        let mut size = HashMap::new();
        size.insert("longest_edge".to_string(), 2048u32);

        let config = PreProcessorConfig {
            size: Some(size),
            patch_size: Some(crate::multimodal::vision::preprocessor_config::PatchSize {
                height: Some(14),
                width: Some(14),
            }),
            image_mean: Some(vec![0.5, 0.5, 0.5]),
            image_std: Some(vec![0.5, 0.5, 0.5]),
            ..Default::default()
        };

        let processor = PixtralProcessor::from_preprocessor_config(&config);

        assert_eq!(processor.longest_edge, 2048);
        assert_eq!(processor.patch_size, 14);
        assert_eq!(processor.image_mean, [0.5, 0.5, 0.5]);
        assert_eq!(processor.image_std, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_calculate_num_tokens() {
        let processor = PixtralProcessor::new();
        let config = PreProcessorConfig::default();

        // 200x150 -> 208x160 -> 13*10 = 130 patches
        let tokens = processor.calculate_num_tokens(200, 150, &config);
        assert_eq!(tokens, 130);
    }
}
