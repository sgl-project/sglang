use super::{ImageInfo, ImagePlan};
use crate::error::{MmError, Result};
use crate::qwen_vl::smart_resize;

const MAX_IMAGE_PIXELS: u64 = 12845056;
const MIN_SHORT_SIDE: u32 = 112;
const MIN_LONG_SIDE_PIXEL: i64 = 112;
const MAX_LONG_SIDE_PIXEL: i64 = 3584;
const MAX_ASPECT_RATIO: f64 = 200.0;
const DEFAULT_IMAGE_LONG_SIDE: u32 = 2016;
const SMART_RESIZE_MIN_PIXELS: usize = 4 * 28 * 28;

#[derive(Clone, Debug, serde::Deserialize)]
pub struct MinimaxM3ImageParams {
    pub image_token_id: i64,
    pub image_start_token_id: i64,
    pub image_end_token_id: i64,
    pub patch_size: u32,
    pub merge_size: u32,
    pub max_pixels: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
pub struct MinimaxM3ImageOptions {
    pub max_long_side_pixel: Option<i64>,
}

fn check_max_long_side_pixel(value: Option<i64>, factor: u32) -> Result<Option<u32>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let in_range = (MIN_LONG_SIDE_PIXEL..=MAX_LONG_SIDE_PIXEL).contains(&value);
    if !in_range || value % i64::from(factor) != 0 {
        return Err(MmError::invalid(format!(
            "max_long_side_pixel must be a multiple of {factor} in \
             [{MIN_LONG_SIDE_PIXEL}, {MAX_LONG_SIDE_PIXEL}], got {value}"
        )));
    }
    Ok(Some(value as u32))
}

fn align_side(side: u32, beta: f64, factor: u32) -> u64 {
    let steps = (side as f64 * beta / factor as f64).round_ties_even() as u64;
    (steps * factor as u64).max(factor as u64)
}

fn compute_aligned_size(
    height: u32,
    width: u32,
    factor: u32,
    max_long_side_pixel: u32,
) -> Result<(u64, u64)> {
    if height == 0 || width == 0 {
        return Err(MmError::invalid("empty image"));
    }
    let long = height.max(width);
    let short = height.min(width);
    let ratio = long as f64 / short as f64;
    if ratio > MAX_ASPECT_RATIO {
        return Err(MmError::invalid(format!(
            "media aspect ratio must be <= {MAX_ASPECT_RATIO}, got {ratio:.1}"
        )));
    }
    let cap = max_long_side_pixel - max_long_side_pixel % factor;
    let beta = if long > cap {
        cap as f64 / long as f64
    } else if short < MIN_SHORT_SIDE {
        (MIN_SHORT_SIDE as f64 / short as f64).min(cap as f64 / long as f64)
    } else {
        1.0
    };
    Ok((
        align_side(height, beta, factor),
        align_side(width, beta, factor),
    ))
}

fn resize_to_long_side(info: ImageInfo, factor: u32, long_side: u32) -> Result<(u32, u32)> {
    let (new_h, new_w) = compute_aligned_size(info.height, info.width, factor, long_side)?;
    if new_h * new_w > MAX_IMAGE_PIXELS {
        return Err(MmError::invalid(format!(
            "image pixels exceed {MAX_IMAGE_PIXELS}, got {} ({new_w}x{new_h})",
            new_h * new_w
        )));
    }
    Ok((new_h as u32, new_w as u32))
}

fn resize_to_max_pixels(info: ImageInfo, factor: u32, max_pixels: u64) -> Result<(u32, u32)> {
    let max_pixels = usize::try_from(max_pixels)
        .map_err(|_| MmError::internal("minimax_m3 plan: max_pixels does not fit in usize"))?;
    let (new_h, new_w) = smart_resize(
        info.height as usize,
        info.width as usize,
        factor as usize,
        SMART_RESIZE_MIN_PIXELS,
        max_pixels,
    )
    .map_err(MmError::invalid)?;
    match (u32::try_from(new_h), u32::try_from(new_w)) {
        (Ok(new_h), Ok(new_w)) => Ok((new_h, new_w)),
        _ => Err(MmError::invalid(format!(
            "resized image {new_w}x{new_h} is too large"
        ))),
    }
}

pub fn plan_image(
    params: &MinimaxM3ImageParams,
    info: ImageInfo,
    options: &MinimaxM3ImageOptions,
) -> Result<ImagePlan> {
    if params.patch_size == 0 || params.merge_size == 0 {
        return Err(MmError::internal(
            "minimax_m3 plan: patch_size and merge_size must be positive",
        ));
    }
    let factor = params.patch_size * params.merge_size;
    let limit = check_max_long_side_pixel(options.max_long_side_pixel, factor)?;
    let (resized_height, resized_width) = match (limit, params.max_pixels) {
        (None, Some(max_pixels)) => resize_to_max_pixels(info, factor, max_pixels)?,
        (None, None) => resize_to_long_side(info, factor, DEFAULT_IMAGE_LONG_SIDE)?,
        (Some(limit), _) => resize_to_long_side(info, factor, limit)?,
    };
    let grid = [
        1,
        resized_height / params.patch_size,
        resized_width / params.patch_size,
    ];
    let merge_area = params.merge_size as u64 * params.merge_size as u64;
    let tokens = (grid[1] as u64 * grid[2] as u64 / merge_area) as usize;
    let mut expansion = Vec::with_capacity(tokens + 2);
    expansion.push(params.image_start_token_id);
    expansion.resize(tokens + 1, params.image_token_id);
    expansion.push(params.image_end_token_id);
    Ok(ImagePlan {
        resized_height,
        resized_width,
        grid,
        expansion,
    })
}

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    use super::{MinimaxM3ImageOptions, MinimaxM3ImageParams, plan_image};
    use crate::plan::ImageInfo;

    type PyImagePlan = (u32, u32, (u32, u32, u32), Vec<i64>);

    #[pyfunction]
    #[pyo3(signature = (height, width, *, image_token_id, image_start_token_id, image_end_token_id, patch_size, merge_size, max_pixels=None, max_long_side_pixel=None))]
    #[allow(clippy::too_many_arguments)]
    fn minimax_m3_image(
        py: Python<'_>,
        height: u32,
        width: u32,
        image_token_id: i64,
        image_start_token_id: i64,
        image_end_token_id: i64,
        patch_size: u32,
        merge_size: u32,
        max_pixels: Option<u64>,
        max_long_side_pixel: Option<i64>,
    ) -> PyResult<PyImagePlan> {
        let params = MinimaxM3ImageParams {
            image_token_id,
            image_start_token_id,
            image_end_token_id,
            patch_size,
            merge_size,
            max_pixels,
        };
        let options = MinimaxM3ImageOptions {
            max_long_side_pixel,
        };
        let info = ImageInfo {
            height,
            width,
            orientation: 1,
        };
        let plan = py.detach(|| plan_image(&params, info, &options))?;
        let [t, h, w] = plan.grid;
        Ok((
            plan.resized_height,
            plan.resized_width,
            (t, h, w),
            plan.expansion,
        ))
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(minimax_m3_image, m)?)
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;

    fn params(max_pixels: Option<u64>) -> MinimaxM3ImageParams {
        MinimaxM3ImageParams {
            image_token_id: 10,
            image_start_token_id: 12,
            image_end_token_id: 13,
            patch_size: 14,
            merge_size: 2,
            max_pixels,
        }
    }

    fn plan(
        height: u32,
        width: u32,
        max_pixels: Option<u64>,
        max_long_side_pixel: Option<i64>,
    ) -> Result<ImagePlan> {
        let info = ImageInfo {
            height,
            width,
            orientation: 1,
        };
        let options = MinimaxM3ImageOptions {
            max_long_side_pixel,
        };
        plan_image(&params(max_pixels), info, &options)
    }

    fn size(height: u32, width: u32, max_long_side_pixel: Option<i64>) -> (u32, u32) {
        let planned = plan(height, width, None, max_long_side_pixel).unwrap();
        (planned.resized_height, planned.resized_width)
    }

    #[test]
    fn expansion_wraps_one_token_per_merged_patch_in_start_and_end() {
        let planned = plan(224, 336, None, None).unwrap();
        assert_eq!((planned.resized_height, planned.resized_width), (224, 336));
        assert_eq!(planned.grid, [1, 16, 24]);
        assert_eq!(planned.expansion.len(), 8 * 12 + 2);
        assert_eq!(planned.expansion[0], 12);
        assert_eq!(planned.expansion[8 * 12 + 1], 13);
        assert!(planned.expansion[1..8 * 12 + 1].iter().all(|&id| id == 10));
    }

    #[test]
    fn long_side_cap_wins_over_the_short_side_floor() {
        assert_eq!(size(2000, 2000, Some(476)), (476, 476));
        assert_eq!(size(50, 100, Some(1008)), (112, 224));
        assert_eq!(size(50, 2000, Some(1008)), (28, 1008));
        assert_eq!(size(50, 1000, Some(1008)), (56, 1008));
        assert_eq!(size(4000, 3000, None), (2016, 1512));
    }

    #[test]
    fn sides_round_half_to_even_multiples_of_the_factor() {
        assert_eq!(size(126, 126, None), (112, 112));
        assert_eq!(size(154, 154, None), (168, 168));
        assert_eq!(size(345, 1000, Some(1008)), (336, 1008));
    }

    #[test]
    fn aspect_ratio_boundary_is_inclusive_at_200() {
        assert_eq!(size(10, 2000, Some(1008)), (28, 1008));
        assert!(matches!(
            plan(10, 2001, None, Some(1008)),
            Err(MmError::InvalidRequest(_))
        ));
    }

    #[test]
    fn max_long_side_pixel_outside_the_spec_is_a_request_error() {
        for value in [0, -28, 84, 111, 113, 3585, 3612] {
            assert!(matches!(
                plan(500, 500, None, Some(value)),
                Err(MmError::InvalidRequest(_))
            ));
        }
        assert_eq!(size(5000, 5000, Some(3584)), (3584, 3584));
        assert_eq!(size(500, 500, Some(112)), (112, 112));
    }

    #[test]
    fn server_max_pixels_applies_only_without_a_request_limit() {
        let capped = plan(1080, 1920, Some(200704), None).unwrap();
        assert!(capped.resized_height as u64 * capped.resized_width as u64 <= 200704);
        assert_eq!(capped.resized_height % 28, 0);
        assert_eq!(capped.resized_width % 28, 0);
        let limited = plan(1000, 1500, Some(200704), Some(252)).unwrap();
        assert_eq!((limited.resized_height, limited.resized_width), (168, 252));
        let unset = plan(1080, 1920, None, None).unwrap();
        assert_eq!((unset.resized_height, unset.resized_width), (1092, 1932));
    }

    #[test]
    fn zero_patch_size_is_an_internal_error() {
        let mut zero_patch = params(None);
        zero_patch.patch_size = 0;
        let info = ImageInfo {
            height: 100,
            width: 100,
            orientation: 1,
        };
        assert!(matches!(
            plan_image(&zero_patch, info, &MinimaxM3ImageOptions::default()),
            Err(MmError::Internal(_))
        ));
    }
}
