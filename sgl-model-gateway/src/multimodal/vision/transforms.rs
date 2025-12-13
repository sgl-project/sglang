//! Image transformation functions for vision preprocessing.
//!
//! This module provides composable transforms that match HuggingFace image processor
//! behavior, enabling pure Rust preprocessing without Python dependencies.

use image::{imageops::FilterType, DynamicImage, GenericImageView, Rgb, RgbImage};
use ndarray::{s, Array3, Array4};
use thiserror::Error;

/// Errors that can occur during image transformations.
#[derive(Error, Debug)]
pub enum TransformError {
    #[error("Invalid tensor shape: expected {expected}, got {actual:?}")]
    InvalidShape {
        expected: String,
        actual: Vec<usize>,
    },

    #[error("Image operation failed: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("Empty batch: cannot stack zero tensors")]
    EmptyBatch,

    #[error("Inconsistent tensor shapes in batch")]
    InconsistentShapes,

    #[error("Shape error: {0}")]
    ShapeError(String),
}

pub type Result<T> = std::result::Result<T, TransformError>;

/// Convert image to tensor [C, H, W] normalized to [0, 1].
///
/// This matches the default behavior of `torchvision.transforms.ToTensor()`.
pub fn to_tensor(image: &DynamicImage) -> Array3<f32> {
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut arr = Array3::<f32>::zeros((3, h, w));

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        arr[[0, y, x]] = pixel[0] as f32 / 255.0;
        arr[[1, y, x]] = pixel[1] as f32 / 255.0;
        arr[[2, y, x]] = pixel[2] as f32 / 255.0;
    }
    arr
}

/// Convert image to tensor [C, H, W] without normalization (keeps [0, 255]).
///
/// Some models expect unnormalized pixel values.
pub fn to_tensor_no_norm(image: &DynamicImage) -> Array3<f32> {
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut arr = Array3::<f32>::zeros((3, h, w));

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        arr[[0, y, x]] = pixel[0] as f32;
        arr[[1, y, x]] = pixel[1] as f32;
        arr[[2, y, x]] = pixel[2] as f32;
    }
    arr
}

/// Normalize tensor per channel: (x - mean) / std.
///
/// This matches `torchvision.transforms.Normalize(mean, std)`.
///
/// # Arguments
/// * `tensor` - Input tensor of shape [C, H, W]
/// * `mean` - Per-channel mean values
/// * `std` - Per-channel standard deviation values
pub fn normalize(tensor: &mut Array3<f32>, mean: &[f64; 3], std: &[f64; 3]) {
    for c in 0..3 {
        let mean_c = mean[c] as f32;
        let std_c = std[c] as f32;
        tensor
            .slice_mut(s![c, .., ..])
            .mapv_inplace(|v| (v - mean_c) / std_c);
    }
}

/// Rescale tensor by a constant factor.
///
/// Used when `do_rescale=True` in HuggingFace configs (typically 1/255).
pub fn rescale(tensor: &mut Array3<f32>, factor: f64) {
    let factor = factor as f32;
    tensor.mapv_inplace(|v| v * factor);
}

/// Resize image to exact dimensions.
///
/// # Arguments
/// * `image` - Input image
/// * `width` - Target width
/// * `height` - Target height
/// * `filter` - Interpolation filter (Nearest, Triangle/Bilinear, CatmullRom/Bicubic, Lanczos3)
pub fn resize(image: &DynamicImage, width: u32, height: u32, filter: FilterType) -> DynamicImage {
    image.resize_exact(width, height, filter)
}

/// Resize image preserving aspect ratio, fitting within max dimensions.
pub fn resize_to_fit(
    image: &DynamicImage,
    max_width: u32,
    max_height: u32,
    filter: FilterType,
) -> DynamicImage {
    image.resize(max_width, max_height, filter)
}

/// Center crop image to specified dimensions.
///
/// If the crop size is larger than the image, the image is returned unchanged.
pub fn center_crop(image: &DynamicImage, crop_w: u32, crop_h: u32) -> DynamicImage {
    let (w, h) = image.dimensions();
    if crop_w >= w && crop_h >= h {
        return image.clone();
    }
    let left = (w.saturating_sub(crop_w)) / 2;
    let top = (h.saturating_sub(crop_h)) / 2;
    let actual_w = crop_w.min(w);
    let actual_h = crop_h.min(h);
    image.crop_imm(left, top, actual_w, actual_h)
}

/// Expand image to square by padding with background color.
///
/// This is used by LLaVA models which expect square inputs. The image is
/// centered and padded with the mean color on the shorter dimension.
pub fn expand_to_square(image: &DynamicImage, background: Rgb<u8>) -> DynamicImage {
    let (w, h) = image.dimensions();
    match w.cmp(&h) {
        std::cmp::Ordering::Equal => image.clone(),
        std::cmp::Ordering::Less => {
            // Height > Width: pad horizontally
            let mut new_image = DynamicImage::from(RgbImage::from_pixel(h, h, background));
            image::imageops::overlay(&mut new_image, image, ((h - w) / 2) as i64, 0);
            new_image
        }
        std::cmp::Ordering::Greater => {
            // Width > Height: pad vertically
            let mut new_image = DynamicImage::from(RgbImage::from_pixel(w, w, background));
            image::imageops::overlay(&mut new_image, image, 0, ((w - h) / 2) as i64);
            new_image
        }
    }
}

/// Pad image to specified dimensions with background color.
///
/// Image is placed at top-left corner.
pub fn pad_to_size(
    image: &DynamicImage,
    target_w: u32,
    target_h: u32,
    background: Rgb<u8>,
) -> DynamicImage {
    let (w, h) = image.dimensions();
    if w >= target_w && h >= target_h {
        return image.clone();
    }
    let new_w = w.max(target_w);
    let new_h = h.max(target_h);
    let mut new_image = DynamicImage::from(RgbImage::from_pixel(new_w, new_h, background));
    image::imageops::overlay(&mut new_image, image, 0, 0);
    new_image
}

/// Stack multiple [C, H, W] tensors into [B, C, H, W].
///
/// All tensors must have the same shape.
pub fn stack_batch(tensors: &[Array3<f32>]) -> Result<Array4<f32>> {
    if tensors.is_empty() {
        return Err(TransformError::EmptyBatch);
    }

    let shape = tensors[0].shape();
    let (c, h, w) = (shape[0], shape[1], shape[2]);

    // Verify all tensors have the same shape
    for tensor in tensors.iter().skip(1) {
        if tensor.shape() != shape {
            return Err(TransformError::InvalidShape {
                expected: format!("[{}, {}, {}]", c, h, w),
                actual: tensor.shape().to_vec(),
            });
        }
    }

    let mut batch = Array4::<f32>::zeros((tensors.len(), c, h, w));
    for (i, tensor) in tensors.iter().enumerate() {
        batch.slice_mut(s![i, .., .., ..]).assign(tensor);
    }

    Ok(batch)
}

/// Convert PIL/HuggingFace resampling enum to image crate filter.
///
/// PIL resampling constants:
/// - 0: NEAREST
/// - 1: LANCZOS (also ANTIALIAS)
/// - 2: BILINEAR
/// - 3: BICUBIC
/// - 4: BOX
/// - 5: HAMMING
pub fn pil_to_filter(resampling: Option<usize>) -> FilterType {
    match resampling {
        Some(0) => FilterType::Nearest,
        Some(1) => FilterType::Lanczos3,
        Some(2) | None => FilterType::Triangle, // Bilinear (default)
        Some(3) => FilterType::CatmullRom,      // Bicubic
        // Box and Hamming don't have direct equivalents, use Triangle
        Some(4) | Some(5) => FilterType::Triangle,
        _ => FilterType::Triangle,
    }
}

/// Calculate mean color of an image as RGB.
pub fn calculate_mean_color(image: &DynamicImage) -> Rgb<u8> {
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width() as u64, rgb.height() as u64);
    let total_pixels = w * h;

    if total_pixels == 0 {
        return Rgb([128, 128, 128]);
    }

    let (mut r_sum, mut g_sum, mut b_sum) = (0u64, 0u64, 0u64);
    for pixel in rgb.pixels() {
        r_sum += pixel[0] as u64;
        g_sum += pixel[1] as u64;
        b_sum += pixel[2] as u64;
    }

    Rgb([
        (r_sum / total_pixels) as u8,
        (g_sum / total_pixels) as u8,
        (b_sum / total_pixels) as u8,
    ])
}

/// Convert normalized mean values [0, 1] to RGB bytes.
pub fn mean_to_rgb(mean: &[f64; 3]) -> Rgb<u8> {
    Rgb([
        (mean[0] * 255.0).round() as u8,
        (mean[1] * 255.0).round() as u8,
        (mean[2] * 255.0).round() as u8,
    ])
}

/// Cubic interpolation weight function (Keys bicubic kernel with a=-0.5).
///
/// This matches PyTorch's bicubic interpolation used in
/// `torch.nn.functional.interpolate(mode='bicubic')`.
#[inline]
pub fn cubic_weight(x: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

/// Perform bicubic interpolation at a single point in a tensor.
///
/// Uses a 4x4 kernel with Keys bicubic weights (a=-0.5) to match PyTorch's
/// `torch.nn.functional.interpolate(mode='bicubic')`.
///
/// # Arguments
/// * `tensor` - Input tensor of shape [C, H, W]
/// * `c` - Channel index
/// * `src_y` - Source Y coordinate (can be fractional)
/// * `src_x` - Source X coordinate (can be fractional)
/// * `h` - Height of the tensor
/// * `w` - Width of the tensor
///
/// # Returns
/// The interpolated value at the specified position.
pub fn bicubic_interpolate(
    tensor: &Array3<f32>,
    c: usize,
    src_y: f32,
    src_x: f32,
    h: usize,
    w: usize,
) -> f32 {
    let y_int = src_y.floor() as i32;
    let x_int = src_x.floor() as i32;
    let y_frac = src_y - y_int as f32;
    let x_frac = src_x - x_int as f32;

    let mut result = 0.0f32;

    // Sample 4x4 neighborhood
    for dy in -1..=2 {
        let y_idx = (y_int + dy).clamp(0, h as i32 - 1) as usize;
        let y_weight = cubic_weight(y_frac - dy as f32);

        for dx in -1..=2 {
            let x_idx = (x_int + dx).clamp(0, w as i32 - 1) as usize;
            let x_weight = cubic_weight(x_frac - dx as f32);

            result += tensor[[c, y_idx, x_idx]] * y_weight * x_weight;
        }
    }

    result
}

/// Resize a tensor using bicubic interpolation.
///
/// This matches PyTorch's `torch.nn.functional.interpolate(mode='bicubic', align_corners=False)`.
///
/// # Arguments
/// * `tensor` - Input tensor of shape [C, H, W]
/// * `target_h` - Target height
/// * `target_w` - Target width
///
/// # Returns
/// Resized tensor of shape [C, target_h, target_w].
pub fn bicubic_resize(tensor: &Array3<f32>, target_h: usize, target_w: usize) -> Array3<f32> {
    let (c, h, w) = (tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);

    if h == target_h && w == target_w {
        return tensor.clone();
    }

    let mut result = Array3::<f32>::zeros((c, target_h, target_w));

    // PyTorch align_corners=False coordinate mapping
    let scale_h = h as f32 / target_h as f32;
    let scale_w = w as f32 / target_w as f32;

    for ch in 0..c {
        for y in 0..target_h {
            for x in 0..target_w {
                // PyTorch align_corners=False: src = (dst + 0.5) * scale - 0.5
                let src_y = (y as f32 + 0.5) * scale_h - 0.5;
                let src_x = (x as f32 + 0.5) * scale_w - 0.5;

                result[[ch, y, x]] = bicubic_interpolate(tensor, ch, src_y, src_x, h, w);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        DynamicImage::from(RgbImage::from_pixel(width, height, color))
    }

    #[test]
    fn test_to_tensor_shape() {
        let img = create_test_image(10, 20, Rgb([255, 128, 0]));
        let tensor = to_tensor(&img);
        assert_eq!(tensor.shape(), &[3, 20, 10]); // [C, H, W]
    }

    #[test]
    fn test_to_tensor_values() {
        let img = create_test_image(2, 2, Rgb([255, 128, 0]));
        let tensor = to_tensor(&img);

        // Check normalization to [0, 1]
        assert!((tensor[[0, 0, 0]] - 1.0).abs() < 1e-6); // R=255 -> 1.0
        assert!((tensor[[1, 0, 0]] - 0.502).abs() < 0.01); // G=128 -> ~0.5
        assert!((tensor[[2, 0, 0]] - 0.0).abs() < 1e-6); // B=0 -> 0.0
    }

    #[test]
    fn test_to_tensor_no_norm() {
        let img = create_test_image(2, 2, Rgb([255, 128, 64]));
        let tensor = to_tensor_no_norm(&img);

        assert!((tensor[[0, 0, 0]] - 255.0).abs() < 1e-6);
        assert!((tensor[[1, 0, 0]] - 128.0).abs() < 1e-6);
        assert!((tensor[[2, 0, 0]] - 64.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut tensor = Array3::<f32>::from_elem((3, 2, 2), 0.5);
        let mean = [0.5, 0.5, 0.5];
        let std = [0.5, 0.5, 0.5];

        normalize(&mut tensor, &mean, &std);

        // (0.5 - 0.5) / 0.5 = 0.0
        for val in tensor.iter() {
            assert!(val.abs() < 1e-6);
        }
    }

    #[test]
    fn test_rescale() {
        let mut tensor = Array3::<f32>::from_elem((3, 2, 2), 255.0);
        rescale(&mut tensor, 1.0 / 255.0);

        for val in tensor.iter() {
            assert!((val - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_resize() {
        let img = create_test_image(100, 50, Rgb([128, 128, 128]));
        let resized = resize(&img, 50, 25, FilterType::Triangle);

        assert_eq!(resized.width(), 50);
        assert_eq!(resized.height(), 25);
    }

    #[test]
    fn test_center_crop() {
        let img = create_test_image(100, 100, Rgb([128, 128, 128]));
        let cropped = center_crop(&img, 50, 50);

        assert_eq!(cropped.width(), 50);
        assert_eq!(cropped.height(), 50);
    }

    #[test]
    fn test_expand_to_square_horizontal() {
        let img = create_test_image(100, 50, Rgb([255, 0, 0]));
        let background = Rgb([0, 0, 0]);
        let squared = expand_to_square(&img, background);

        assert_eq!(squared.width(), 100);
        assert_eq!(squared.height(), 100);
    }

    #[test]
    fn test_expand_to_square_vertical() {
        let img = create_test_image(50, 100, Rgb([255, 0, 0]));
        let background = Rgb([0, 0, 0]);
        let squared = expand_to_square(&img, background);

        assert_eq!(squared.width(), 100);
        assert_eq!(squared.height(), 100);
    }

    #[test]
    fn test_expand_to_square_already_square() {
        let img = create_test_image(100, 100, Rgb([255, 0, 0]));
        let background = Rgb([0, 0, 0]);
        let squared = expand_to_square(&img, background);

        assert_eq!(squared.width(), 100);
        assert_eq!(squared.height(), 100);
    }

    #[test]
    fn test_stack_batch() {
        let t1 = Array3::<f32>::zeros((3, 10, 10));
        let t2 = Array3::<f32>::ones((3, 10, 10));

        let batch = stack_batch(&[t1, t2]).unwrap();

        assert_eq!(batch.shape(), &[2, 3, 10, 10]);
    }

    #[test]
    fn test_stack_batch_empty() {
        let result = stack_batch(&[]);
        assert!(matches!(result, Err(TransformError::EmptyBatch)));
    }

    #[test]
    fn test_pil_to_filter() {
        assert!(matches!(pil_to_filter(Some(0)), FilterType::Nearest));
        assert!(matches!(pil_to_filter(Some(1)), FilterType::Lanczos3));
        assert!(matches!(pil_to_filter(Some(2)), FilterType::Triangle));
        assert!(matches!(pil_to_filter(Some(3)), FilterType::CatmullRom));
        assert!(matches!(pil_to_filter(None), FilterType::Triangle));
    }

    #[test]
    fn test_mean_to_rgb() {
        let mean = [0.5, 0.25, 1.0];
        let rgb = mean_to_rgb(&mean);

        assert_eq!(rgb[0], 128);
        assert_eq!(rgb[1], 64);
        assert_eq!(rgb[2], 255);
    }
}
