//! Phi3-Vision image processor.
//!
//! This module implements the Phi3-Vision image preprocessing pipeline with
//! Dynamic High Definition (HD) transform, which tiles images into 336x336 crops.
//!
//! # Processing Pipeline
//!
//! 1. **HD Transform**: Resize and pad image to multiples of 336
//! 2. **Normalize**: Apply CLIP normalization
//! 3. **Create Global Image**: Bicubic interpolate to 336x336
//! 4. **Tile**: Reshape into (num_tiles, 3, 336, 336)
//! 5. **Concatenate**: [global_image, tiles...]
//! 6. **Pad**: Zero-pad to (num_crops+1, 3, 336, 336)
//!
//! # Key Features
//!
//! - Dynamic resolution via HD transform
//! - Default num_crops: 16
//! - CLIP normalization: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
//! - Token count formula: `((h//336)*(w//336)+1)*144 + 1 + (h//336+1)*12`

use image::{imageops::FilterType, DynamicImage, GenericImageView, Rgb, RgbImage};
use ndarray::{s, Array3, Array4, IxDyn};

use crate::multimodal::vision::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    transforms::{self, TransformError},
};

/// CLIP normalization mean values.
pub const CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];

/// CLIP normalization std values.
pub const CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Default number of crops for HD transform.
pub const DEFAULT_NUM_CROPS: usize = 16;

/// Default number of image tokens per crop (144 per tile + base).
pub const DEFAULT_NUM_IMG_TOKENS: usize = 144;

/// Tile size used in Phi3-Vision (336x336).
pub const TILE_SIZE: u32 = 336;

/// Phi3-Vision image processor.
///
/// Implements Dynamic HD transform with tile-based processing.
#[derive(Debug, Clone)]
pub struct Phi3VisionProcessor {
    /// Maximum number of HD crops (not including global).
    num_crops: usize,
    /// Normalization mean.
    mean: [f64; 3],
    /// Normalization std.
    std: [f64; 3],
}

impl Default for Phi3VisionProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Phi3VisionProcessor {
    /// Create a new Phi3-Vision processor with default settings.
    pub fn new() -> Self {
        Self {
            num_crops: DEFAULT_NUM_CROPS,
            mean: CLIP_MEAN,
            std: CLIP_STD,
        }
    }

    /// Create a processor with custom settings.
    pub fn with_config(num_crops: usize) -> Self {
        Self {
            num_crops,
            mean: CLIP_MEAN,
            std: CLIP_STD,
        }
    }

    /// Create a processor from preprocessor config.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Self {
        Self {
            num_crops: config.num_crops.unwrap_or(DEFAULT_NUM_CROPS),
            mean: config
                .image_mean
                .as_ref()
                .map(|v| [v[0], v[1], v[2]])
                .unwrap_or(CLIP_MEAN),
            std: config
                .image_std
                .as_ref()
                .map(|v| [v[0], v[1], v[2]])
                .unwrap_or(CLIP_STD),
        }
    }

    /// Get the number of crops.
    pub fn num_crops(&self) -> usize {
        self.num_crops
    }

    /// HD transform: resize and pad image to multiples of 336.
    ///
    /// Algorithm:
    /// 1. If width < height, transpose (flip over main diagonal)
    /// 2. Calculate scale: while scale * ceil(scale/ratio) <= hd_num: scale++
    /// 3. Resize to new_w = scale * 336, new_h = new_w / ratio
    /// 4. Pad height to multiple of 336 (centered, white padding)
    /// 5. If transposed, transpose back
    pub fn hd_transform(&self, image: &DynamicImage) -> DynamicImage {
        let (width, height) = image.dimensions();

        let (img, transposed) = if width < height {
            // Transpose (PIL's Image.TRANSPOSE): equivalent to fliph + rotate270 (ccw 90°)
            // This swaps x and y coordinates: pixel at (x, y) goes to (y, x)
            (image.fliph().rotate270(), true)
        } else {
            (image.clone(), false)
        };

        let (width, height) = img.dimensions();
        let ratio = width as f64 / height as f64;

        // Calculate scale factor
        let mut scale = 1.0f64;
        while scale * (scale / ratio).ceil() <= self.num_crops as f64 {
            scale += 1.0;
        }
        scale -= 1.0;

        let new_w = (scale * TILE_SIZE as f64) as u32;
        let new_h = (new_w as f64 / ratio) as u32;

        // Resize using bilinear filter (matching torchvision's bilinear+antialias)
        // HuggingFace uses torchvision.transforms.functional.resize with
        // BILINEAR interpolation and antialias=True. PIL's BILINEAR includes
        // implicit antialiasing that closely matches torchvision.
        let resized = img.resize_exact(new_w, new_h, FilterType::Triangle);

        // Pad height to multiple of 336
        let padded = self.padding_336(&resized);

        // Transpose back if needed (transpose is self-inverse)
        if transposed {
            padded.fliph().rotate270()
        } else {
            padded
        }
    }

    /// Pad image height to multiple of 336 (centered, white padding).
    fn padding_336(&self, image: &DynamicImage) -> DynamicImage {
        let (width, height) = image.dimensions();
        let target_h = ((height as f64 / TILE_SIZE as f64).ceil() * TILE_SIZE as f64) as u32;

        if height == target_h {
            return image.clone();
        }

        let top_padding = (target_h - height) / 2;

        // Create white-padded image
        let mut new_image =
            DynamicImage::from(RgbImage::from_pixel(width, target_h, Rgb([255, 255, 255])));

        // Copy original image to center
        image::imageops::overlay(&mut new_image, image, 0, top_padding as i64);

        new_image
    }

    /// Create global image by bicubic interpolation to 336x336.
    ///
    /// Uses the shared `bicubic_resize` which matches PyTorch's
    /// `torch.nn.functional.interpolate(mode='bicubic', align_corners=False)`.
    fn create_global_image(&self, tensor: &Array3<f32>) -> Array3<f32> {
        transforms::bicubic_resize(tensor, TILE_SIZE as usize, TILE_SIZE as usize)
    }

    /// Reshape HD image into tiles.
    ///
    /// Transforms [3, H, W] -> [num_tiles, 3, 336, 336]
    /// where H and W are multiples of 336.
    fn reshape_to_tiles(&self, tensor: &Array3<f32>) -> Vec<Array3<f32>> {
        let (_c, h, w) = (tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);
        let grid_h = h / TILE_SIZE as usize;
        let grid_w = w / TILE_SIZE as usize;

        let mut tiles = Vec::with_capacity(grid_h * grid_w);

        for gh in 0..grid_h {
            for gw in 0..grid_w {
                let y_start = gh * TILE_SIZE as usize;
                let x_start = gw * TILE_SIZE as usize;
                let y_end = y_start + TILE_SIZE as usize;
                let x_end = x_start + TILE_SIZE as usize;

                let tile_view = tensor.slice(s![.., y_start..y_end, x_start..x_end]);
                tiles.push(tile_view.to_owned());
            }
        }

        tiles
    }

    /// Calculate number of image tokens for given HD size.
    ///
    /// Formula: `((h//336)*(w//336)+1)*144 + 1 + (h//336+1)*12`
    pub fn calculate_num_tokens(&self, h: usize, w: usize) -> usize {
        let grid_h = h / TILE_SIZE as usize;
        let grid_w = w / TILE_SIZE as usize;

        // ((h//336)*(w//336)+1)*144 + 1 + (h//336+1)*12
        (grid_h * grid_w + 1) * 144 + 1 + (grid_h + 1) * 12
    }

    /// Process a single image through the full pipeline.
    #[allow(clippy::type_complexity)]
    fn process_single_image(
        &self,
        image: &DynamicImage,
        config: &PreProcessorConfig,
    ) -> Result<(Array4<f32>, (usize, usize), usize), TransformError> {
        // 1. Convert to RGB
        let image = DynamicImage::ImageRgb8(image.to_rgb8());

        // 2. HD transform
        let hd_image = self.hd_transform(&image);
        let (hd_w, hd_h) = hd_image.dimensions();

        // 3. To tensor [0, 1] and normalize
        let mut tensor = transforms::to_tensor(&hd_image);
        let mean = config
            .image_mean
            .as_ref()
            .map(|v| [v[0], v[1], v[2]])
            .unwrap_or(self.mean);
        let std = config
            .image_std
            .as_ref()
            .map(|v| [v[0], v[1], v[2]])
            .unwrap_or(self.std);
        transforms::normalize(&mut tensor, &mean, &std);

        // 4. Create global image (336x336)
        let global_image = self.create_global_image(&tensor);

        // 5. Reshape HD image into tiles
        let tiles = self.reshape_to_tiles(&tensor);

        // 6. Concatenate global + tiles
        let max_crops = self.num_crops + 1; // num_crops + 1 for global

        // Create output tensor [max_crops, 3, 336, 336]
        let mut output =
            Array4::<f32>::zeros((max_crops, 3, TILE_SIZE as usize, TILE_SIZE as usize));

        // Copy global image (first position)
        output.slice_mut(s![0, .., .., ..]).assign(&global_image);

        // Copy tiles (positions 1..num_actual_crops)
        for (i, tile) in tiles.iter().enumerate() {
            if i + 1 < max_crops {
                output.slice_mut(s![i + 1, .., .., ..]).assign(tile);
            }
        }

        // Calculate token count
        let num_tokens = self.calculate_num_tokens(hd_h as usize, hd_w as usize);

        // image_sizes is the HD-transformed size
        Ok((output, (hd_h as usize, hd_w as usize), num_tokens))
    }
}

impl ImagePreProcessor for Phi3VisionProcessor {
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
                expected: "at least one image".to_string(),
                actual: vec![0],
            });
        }

        let mut all_pixel_values = Vec::with_capacity(images.len());
        let mut all_image_sizes = Vec::with_capacity(images.len());
        let mut all_num_tokens = Vec::with_capacity(images.len());

        for image in images {
            let (pixel_values, image_size, num_tokens) =
                self.process_single_image(image, config)?;
            all_pixel_values.push(pixel_values);
            all_image_sizes.push((image_size.1 as u32, image_size.0 as u32)); // (width, height)
            all_num_tokens.push(num_tokens);
        }

        // Stack into batch [B, num_crops+1, 3, 336, 336]
        let max_crops = self.num_crops + 1;
        let batch_size = images.len();
        let mut batch_tensor = ndarray::Array5::<f32>::zeros((
            batch_size,
            max_crops,
            3,
            TILE_SIZE as usize,
            TILE_SIZE as usize,
        ));

        for (i, pv) in all_pixel_values.iter().enumerate() {
            batch_tensor.slice_mut(s![i, .., .., .., ..]).assign(pv);
        }

        // Convert to dynamic array for storage
        let shape = batch_tensor.shape().to_vec();
        let (flat_data, _offset) = batch_tensor.into_raw_vec_and_offset();

        // Store image_sizes as model-specific data
        let mut model_specific = std::collections::HashMap::new();

        // image_sizes as [batch, 2] tensor (h, w for each image)
        let image_sizes_data: Vec<u32> = all_image_sizes
            .iter()
            .flat_map(|(w, h)| [*h, *w]) // [h, w] for each image
            .collect();
        model_specific.insert(
            "image_sizes".to_string(),
            crate::multimodal::vision::image_processor::ModelSpecificValue::UintTensor {
                data: image_sizes_data,
                shape: vec![batch_size, 2],
            },
        );

        // num_img_tokens as list
        model_specific.insert(
            "num_img_tokens".to_string(),
            crate::multimodal::vision::image_processor::ModelSpecificValue::UintVec(
                all_num_tokens.iter().map(|&t| t as u32).collect(),
            ),
        );

        // Convert 5D tensor to appropriate format
        // Phi3-Vision expects [B, num_crops+1, C, H, W]
        let pixel_values = ndarray::ArrayD::<f32>::from_shape_vec(IxDyn(&shape), flat_data)
            .map_err(|e| TransformError::InvalidShape {
                expected: format!("valid 5D shape, but failed with error: {}", e),
                actual: shape.clone(),
            })?;

        Ok(PreprocessedImages {
            pixel_values,
            num_img_tokens: all_num_tokens,
            image_sizes: all_image_sizes,
            model_specific,
        })
    }

    fn calculate_num_tokens(&self, width: u32, height: u32, _config: &PreProcessorConfig) -> usize {
        // First apply HD transform to get the actual size
        let image = DynamicImage::new_rgb8(width, height);
        let hd_image = self.hd_transform(&image);
        let (_, hd_h) = hd_image.dimensions();
        let hd_w = hd_image.width();

        self.calculate_num_tokens(hd_h as usize, hd_w as usize)
    }

    fn model_name(&self) -> &'static str {
        "phi3-vision"
    }

    fn get_processed_size(&self, _config: &PreProcessorConfig) -> Option<(u32, u32)> {
        // Phi3-Vision has dynamic size based on HD transform
        None
    }
}

#[cfg(test)]
mod tests {
    use image::RgbImage;

    use super::*;

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_phi3_vision_processor_default() {
        let processor = Phi3VisionProcessor::new();
        assert_eq!(processor.num_crops(), 16);
        assert_eq!(processor.default_mean(), CLIP_MEAN);
        assert_eq!(processor.default_std(), CLIP_STD);
    }

    #[test]
    fn test_hd_transform_square() {
        let processor = Phi3VisionProcessor::new();
        let image = create_test_image(504, 504, Rgb([128, 128, 128]));

        let hd_image = processor.hd_transform(&image);
        let (w, h) = hd_image.dimensions();

        // Should be multiple of 336
        assert_eq!(h % 336, 0);
        assert_eq!(w % 336, 0);

        // Should respect num_crops limit
        let num_tiles = (h / 336) * (w / 336);
        assert!(num_tiles <= processor.num_crops() as u32);
    }

    #[test]
    fn test_hd_transform_tall() {
        let processor = Phi3VisionProcessor::new();
        let image = create_test_image(400, 600, Rgb([100, 100, 100]));

        let hd_image = processor.hd_transform(&image);
        let (w, h) = hd_image.dimensions();

        // Should be multiple of 336
        assert_eq!(h % 336, 0);
        assert_eq!(w % 336, 0);
    }

    #[test]
    fn test_hd_transform_wide() {
        let processor = Phi3VisionProcessor::new();
        let image = create_test_image(600, 400, Rgb([150, 150, 150]));

        let hd_image = processor.hd_transform(&image);
        let (w, h) = hd_image.dimensions();

        // Should be multiple of 336
        assert_eq!(h % 336, 0);
        assert_eq!(w % 336, 0);
    }

    #[test]
    fn test_calculate_num_tokens() {
        let processor = Phi3VisionProcessor::new();

        // 1344x1344 -> 4x4 grid -> (16+1)*144 + 1 + (4+1)*12 = 2448 + 1 + 60 = 2509
        let tokens = processor.calculate_num_tokens(1344, 1344);
        assert_eq!(tokens, 2509);

        // 1008x1344 -> 3x4 grid -> (12+1)*144 + 1 + (3+1)*12 = 1872 + 1 + 48 = 1921
        let tokens = processor.calculate_num_tokens(1008, 1344);
        assert_eq!(tokens, 1921);

        // 1344x1008 -> 4x3 grid -> (12+1)*144 + 1 + (4+1)*12 = 1872 + 1 + 60 = 1933
        let tokens = processor.calculate_num_tokens(1344, 1008);
        assert_eq!(tokens, 1933);
    }

    #[test]
    fn test_phi3_vision_preprocess() {
        let processor = Phi3VisionProcessor::new();
        let config = PreProcessorConfig::default();

        let image = create_test_image(504, 504, Rgb([128, 128, 128]));
        let result = processor.preprocess(&[image], &config).unwrap();

        assert_eq!(result.batch_size(), 1);

        // Check output shape is [1, num_crops+1, 3, 336, 336]
        let shape = result.pixel_values.shape();
        assert_eq!(shape.len(), 5);
        assert_eq!(shape[0], 1); // batch
        assert_eq!(shape[1], 17); // num_crops + 1
        assert_eq!(shape[2], 3); // channels
        assert_eq!(shape[3], 336); // height
        assert_eq!(shape[4], 336); // width

        // Check model-specific outputs
        assert!(result.model_specific.contains_key("image_sizes"));
        assert!(result.model_specific.contains_key("num_img_tokens"));
    }

    #[test]
    fn test_phi3_vision_preprocess_multiple() {
        let processor = Phi3VisionProcessor::new();
        let config = PreProcessorConfig::default();

        let images = vec![
            create_test_image(504, 504, Rgb([100, 100, 100])),
            create_test_image(400, 600, Rgb([150, 150, 150])),
        ];

        let result = processor.preprocess(&images, &config).unwrap();

        assert_eq!(result.batch_size(), 2);
        assert_eq!(result.image_sizes.len(), 2);
        assert_eq!(result.num_img_tokens.len(), 2);
    }

    #[test]
    fn test_model_name() {
        let processor = Phi3VisionProcessor::new();
        assert_eq!(processor.model_name(), "phi3-vision");
    }

    #[test]
    fn test_from_config() {
        let config = PreProcessorConfig {
            num_crops: Some(8),
            image_mean: Some(vec![0.5, 0.5, 0.5]),
            image_std: Some(vec![0.5, 0.5, 0.5]),
            ..Default::default()
        };

        let processor = Phi3VisionProcessor::from_preprocessor_config(&config);
        assert_eq!(processor.num_crops(), 8);
    }

    #[test]
    fn test_transpose_equivalence() {
        // Test that fliph().rotate270() correctly implements PIL's Image.TRANSPOSE
        // TRANSPOSE swaps x and y coordinates: pixel at (x, y) goes to (y, x)
        use image::{GenericImageView, Rgb, RgbImage};

        let mut img = RgbImage::new(100, 200);
        img.put_pixel(0, 0, Rgb([255, 0, 0])); // Top-left = red
        img.put_pixel(99, 0, Rgb([0, 255, 0])); // Top-right = green
        img.put_pixel(0, 199, Rgb([0, 0, 255])); // Bottom-left = blue
        img.put_pixel(99, 199, Rgb([255, 255, 0])); // Bottom-right = yellow

        let img = DynamicImage::ImageRgb8(img);
        let transposed = img.fliph().rotate270();

        // After TRANSPOSE: (x, y) -> (y, x)
        assert_eq!(transposed.get_pixel(0, 0).0[0..3], [255, 0, 0]); // (0,0) -> (0,0)
        assert_eq!(transposed.get_pixel(0, 99).0[0..3], [0, 255, 0]); // (99,0) -> (0,99)
        assert_eq!(transposed.get_pixel(199, 0).0[0..3], [0, 0, 255]); // (0,199) -> (199,0)
        assert_eq!(transposed.get_pixel(199, 99).0[0..3], [255, 255, 0]); // (99,199) -> (199,99)
    }
}
