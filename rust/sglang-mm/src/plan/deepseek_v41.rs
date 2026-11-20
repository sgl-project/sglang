use std::hint::black_box;

use super::ImageInfo;
use crate::error::{MmError, Result};

const MAX_EXACT_INT: u64 = 1 << 53;

#[derive(Clone, Debug, serde::Deserialize)]
pub struct DeepseekV41ImageParams {
    pub image_token_id: i64,
    pub patch_size: u32,
    pub downsample_ratio: u32,
    pub max_n_token: u32,
    pub min_pixels: u64,
    pub max_wh_ratio: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeepseekV41ImagePlan {
    pub resized_height: u32,
    pub resized_width: u32,
    pub grid: [u32; 3],
    pub expansion: Vec<i64>,
    pub output_height: u32,
    pub output_width: u32,
    pub padding_top: u32,
    pub padding_left: u32,
    pub patch_size: u32,
    pub llm_grid: [u32; 2],
}

fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| MmError::internal("deepseek_v41 plan: size overflow"))
}

fn to_u32(value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| MmError::internal("deepseek_v41 plan: size overflow"))
}

fn llm_grid(best_height: u64, best_width: u64, patch: u64, downsample: u64) -> (u64, u64) {
    let llm_h = ((best_height / patch) as f64 / downsample as f64).ceil() as u64;
    let llm_w = ((best_width / patch) as f64 / downsample as f64).ceil() as u64;
    (llm_h, llm_w)
}

fn num_image_tokens(llm_h: u64, llm_w: u64) -> u128 {
    llm_h as u128 * (llm_w as u128 + 1) + 2
}

fn solve_resize_ratio(
    height: u64,
    width: u64,
    patch: u64,
    downsample: u64,
    max_n_token: u64,
) -> Result<(u64, u64)> {
    let r = height as f64 / width as f64;
    let max_w_float = ((max_n_token - 2) as f64 / r + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * r;
    let cell = mul(patch, downsample)?;
    if max_w_float < 1.0 {
        return Ok((mul((max_n_token - 2) / 2, cell)?, cell));
    }
    if max_h_float < 1.0 {
        return Ok((cell, mul(max_n_token - 3, cell)?));
    }
    let by_width = mul(max_w_float.floor() as u64, cell)? as f64 / width as f64;
    let by_height = mul(max_h_float.floor() as u64, cell)? as f64 / height as f64;
    let beta = if by_height < by_width {
        by_height
    } else {
        by_width
    };
    let best_height = mul((height as f64 * beta / patch as f64).floor() as u64, patch)?;
    let best_width = mul((width as f64 * beta / patch as f64).floor() as u64, patch)?;
    Ok((best_height, best_width))
}

pub fn plan_image(
    params: &DeepseekV41ImageParams,
    info: ImageInfo,
) -> Result<DeepseekV41ImagePlan> {
    if params.patch_size == 0 || params.downsample_ratio == 0 {
        return Err(MmError::internal(
            "deepseek_v41 plan: patch_size and downsample_ratio must be positive",
        ));
    }
    if params.max_n_token < 4 {
        return Err(MmError::internal(
            "deepseek_v41 plan: max_n_token must be at least 4",
        ));
    }
    if params.min_pixels > MAX_EXACT_INT {
        return Err(MmError::internal(
            "deepseek_v41 plan: min_pixels is too large",
        ));
    }
    if params.max_wh_ratio == Some(0) {
        return Err(MmError::internal(
            "deepseek_v41 plan: max_wh_ratio must be positive",
        ));
    }
    if info.height == 0 || info.width == 0 {
        return Err(MmError::invalid(
            "deepseek_v41 plan: image height and width must be positive",
        ));
    }
    let patch = params.patch_size as u64;
    let downsample = params.downsample_ratio as u64;
    let max_n_token = params.max_n_token as u64;
    let image_height = info.height as u64;
    let image_width = info.width as u64;

    let mut height = image_height;
    let mut width = image_width;
    if let Some(ratio) = params.max_wh_ratio {
        let limit = height * ratio as u64;
        if width > limit {
            width = limit;
        }
    }
    let area = width * height;
    if area < params.min_pixels {
        let ratio = (params.min_pixels as f64 / area as f64).powf(black_box(0.5));
        width = (width as f64 * ratio) as u64;
        height = (height as f64 * ratio) as u64;
    }
    if height == 0 || width == 0 {
        return Err(MmError::internal(
            "deepseek_v41 plan: scaled image size is empty",
        ));
    }

    let mut best_width = mul((width as f64 / patch as f64).ceil() as u64, patch)?;
    let mut best_height = mul((height as f64 / patch as f64).ceil() as u64, patch)?;
    let (mut llm_h, mut llm_w) = llm_grid(best_height, best_width, patch, downsample);
    if num_image_tokens(llm_h, llm_w) > max_n_token as u128 {
        let (solved_height, solved_width) =
            solve_resize_ratio(height, width, patch, downsample, max_n_token)?;
        best_height = solved_height;
        best_width = solved_width;
        let (solved_llm_h, solved_llm_w) = llm_grid(best_height, best_width, patch, downsample);
        llm_h = solved_llm_h;
        llm_w = solved_llm_w;
        if num_image_tokens(llm_h, llm_w) > max_n_token as u128 {
            return Err(MmError::internal(
                "deepseek_v41 plan: resized image exceeds max_n_token",
            ));
        }
    }
    if best_height == 0 || best_width == 0 {
        return Err(MmError::invalid(
            "deepseek_v41 plan: image does not fit the token budget",
        ));
    }

    let stretch = match params.max_wh_ratio {
        Some(ratio) => image_width >= ratio as u64 * image_height,
        None => false,
    };
    let mut resize_height = best_height;
    let mut resize_width = best_width;
    if !stretch {
        let image_ratio = image_width as f64 / image_height as f64;
        let canvas_ratio = best_width as f64 / best_height as f64;
        if image_ratio > canvas_ratio {
            resize_height = (image_height as f64 / image_width as f64 * best_width as f64)
                .round_ties_even() as u64;
        } else if image_ratio < canvas_ratio {
            resize_width = (image_width as f64 / image_height as f64 * best_height as f64)
                .round_ties_even() as u64;
        }
    }
    if resize_height == 0 || resize_width == 0 {
        return Err(MmError::invalid(
            "deepseek_v41 plan: image aspect ratio is too extreme",
        ));
    }
    if resize_height > best_height || resize_width > best_width {
        return Err(MmError::internal(
            "deepseek_v41 plan: resize size exceeds the output size",
        ));
    }
    let top = ((best_height - resize_height) as f64 / 2.0).round_ties_even() as u64;
    let left = ((best_width - resize_width) as f64 / 2.0).round_ties_even() as u64;

    let tokens = usize::try_from(num_image_tokens(llm_h, llm_w))
        .map_err(|_| MmError::internal("deepseek_v41 plan: size overflow"))?;
    Ok(DeepseekV41ImagePlan {
        resized_height: to_u32(resize_height)?,
        resized_width: to_u32(resize_width)?,
        grid: [1, to_u32(best_height / patch)?, to_u32(best_width / patch)?],
        expansion: vec![params.image_token_id; tokens],
        output_height: to_u32(best_height)?,
        output_width: to_u32(best_width)?,
        padding_top: to_u32(top)?,
        padding_left: to_u32(left)?,
        patch_size: params.patch_size,
        llm_grid: [to_u32(llm_h)?, to_u32(llm_w)?],
    })
}

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    use super::{DeepseekV41ImageParams, plan_image};
    use crate::plan::ImageInfo;

    type PyImagePlan = (
        u32,
        u32,
        (u32, u32, u32),
        Vec<i64>,
        (u32, u32),
        (u32, u32),
        u32,
        (u32, u32),
    );

    #[pyfunction]
    #[pyo3(signature = (height, width, *, image_token_id, patch_size, downsample_ratio, max_n_token, min_pixels, max_wh_ratio=None))]
    #[allow(clippy::too_many_arguments)]
    fn deepseek_v41_image(
        py: Python<'_>,
        height: u32,
        width: u32,
        image_token_id: i64,
        patch_size: u32,
        downsample_ratio: u32,
        max_n_token: u32,
        min_pixels: u64,
        max_wh_ratio: Option<u32>,
    ) -> PyResult<PyImagePlan> {
        let params = DeepseekV41ImageParams {
            image_token_id,
            patch_size,
            downsample_ratio,
            max_n_token,
            min_pixels,
            max_wh_ratio,
        };
        let info = ImageInfo {
            height,
            width,
            orientation: 1,
        };
        let plan = py.detach(|| plan_image(&params, info))?;
        let [t, h, w] = plan.grid;
        let [llm_h, llm_w] = plan.llm_grid;
        Ok((
            plan.resized_height,
            plan.resized_width,
            (t, h, w),
            plan.expansion,
            (plan.output_height, plan.output_width),
            (plan.padding_top, plan.padding_left),
            plan.patch_size,
            (llm_h, llm_w),
        ))
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(deepseek_v41_image, m)?)
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> DeepseekV41ImageParams {
        DeepseekV41ImageParams {
            image_token_id: 129264,
            patch_size: 14,
            downsample_ratio: 3,
            max_n_token: 1024,
            min_pixels: 295936,
            max_wh_ratio: None,
        }
    }

    fn unscaled() -> DeepseekV41ImageParams {
        DeepseekV41ImageParams {
            min_pixels: 0,
            ..params()
        }
    }

    fn info(height: u32, width: u32) -> ImageInfo {
        ImageInfo {
            height,
            width,
            orientation: 1,
        }
    }

    #[test]
    fn small_image_is_scaled_up_to_min_pixels() {
        let plan = plan_image(&params(), info(100, 100)).unwrap();
        assert_eq!((plan.output_height, plan.output_width), (546, 546));
        assert_eq!((plan.resized_height, plan.resized_width), (546, 546));
        assert_eq!(plan.grid, [1, 39, 39]);
        assert_eq!(plan.llm_grid, [13, 13]);
        assert_eq!(plan.expansion, vec![129264; 13 * 14 + 2]);
    }

    #[test]
    fn large_image_shrinks_into_the_token_budget() {
        let plan = plan_image(&params(), info(3024, 4032)).unwrap();
        assert_eq!((plan.output_height, plan.output_width), (1134, 1512));
        assert_eq!((plan.resized_height, plan.resized_width), (1134, 1512));
        assert_eq!((plan.padding_top, plan.padding_left), (0, 0));
        assert_eq!(plan.grid, [1, 81, 108]);
        assert_eq!(plan.expansion.len(), 27 * 37 + 2);
    }

    #[test]
    fn token_budget_boundary_is_inclusive() {
        let fits = plan_image(&params(), info(588, 3024)).unwrap();
        assert_eq!((fits.output_height, fits.output_width), (588, 3024));
        assert_eq!(fits.expansion.len(), 1024);
        let shrunk = plan_image(&params(), info(589, 3024)).unwrap();
        assert!(shrunk.output_width < 3024);
        assert!(shrunk.expansion.len() <= 1024);
    }

    #[test]
    fn resize_and_padding_round_half_to_even() {
        let plan = plan_image(&unscaled(), info(6, 8)).unwrap();
        assert_eq!((plan.output_height, plan.output_width), (14, 14));
        assert_eq!((plan.resized_height, plan.resized_width), (10, 14));
        assert_eq!((plan.padding_top, plan.padding_left), (2, 0));

        let plan = plan_image(&unscaled(), info(51, 112)).unwrap();
        assert_eq!((plan.resized_height, plan.output_height), (51, 56));
        assert_eq!(plan.padding_top, 2);

        let plan = plan_image(&unscaled(), info(49, 112)).unwrap();
        assert_eq!((plan.resized_height, plan.output_height), (49, 56));
        assert_eq!(plan.padding_top, 4);
    }

    #[test]
    fn max_wh_ratio_clamps_the_width_and_stretches() {
        let params = DeepseekV41ImageParams {
            max_wh_ratio: Some(2),
            ..unscaled()
        };
        let plan = plan_image(&params, info(100, 500)).unwrap();
        assert_eq!((plan.output_height, plan.output_width), (112, 210));
        assert_eq!((plan.resized_height, plan.resized_width), (112, 210));
        assert_eq!((plan.padding_top, plan.padding_left), (0, 0));
        assert_eq!(plan.expansion.len(), 3 * 6 + 2);
    }

    #[test]
    fn plans_keep_the_kernel_geometry_contract() {
        let sizes = [1, 13, 14, 15, 543, 544, 545, 1000, 1302, 1303, 4096, 10000];
        for height in sizes {
            for width in sizes {
                let plan = match plan_image(&params(), info(height, width)) {
                    Ok(plan) => plan,
                    Err(error) => {
                        assert!(matches!(error, MmError::InvalidRequest(_)));
                        continue;
                    }
                };
                let [llm_h, llm_w] = plan.llm_grid;
                assert_eq!(plan.expansion.len() as u32, llm_h * (llm_w + 1) + 2);
                assert!(plan.expansion.len() <= 1024);
                assert!(plan.expansion.iter().all(|&id| id == 129264));
                assert_eq!(plan.grid[1] * 14, plan.output_height);
                assert_eq!(plan.grid[2] * 14, plan.output_width);
                assert!(plan.resized_height >= 1 && plan.resized_width >= 1);
                assert!(plan.padding_top + plan.resized_height <= plan.output_height);
                assert!(plan.padding_left + plan.resized_width <= plan.output_width);
                let slack = plan.output_height - plan.resized_height;
                assert!(plan.padding_top * 2 <= slack + 1 && slack <= plan.padding_top * 2 + 1);
            }
        }
    }

    #[test]
    fn unusable_image_sizes_are_request_errors() {
        for (height, width) in [(0, 10), (10, 0), (1, 100000), (100000, 1)] {
            assert!(matches!(
                plan_image(&params(), info(height, width)),
                Err(MmError::InvalidRequest(_))
            ));
        }
    }

    #[test]
    fn bad_params_are_internal_errors() {
        let params = DeepseekV41ImageParams {
            patch_size: 0,
            ..params()
        };
        assert!(matches!(
            plan_image(&params, info(100, 100)),
            Err(MmError::Internal(_))
        ));
    }
}
