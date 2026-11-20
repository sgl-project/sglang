use super::ImageInfo;
use crate::error::{MmError, Result};

const EXACT_F64_INT: u64 = 1 << 53;

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Glm5NextImageParams {
    pub image_token_id: i64,
    pub patch_size: u32,
    pub merge_size: u32,
    pub temporal_patch_size: u32,
    pub patch_expand_factor: u32,
    pub min_image_tokens: u64,
    pub max_image_tokens: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glm5NextImagePlan {
    pub canvas_height: u32,
    pub canvas_width: u32,
    pub content_height: u32,
    pub content_width: u32,
    pub grid: [u32; 3],
    pub expansion: Vec<i64>,
}

struct Budget {
    factor: u64,
    frames: u64,
    min_pixels: u64,
    max_pixels: u64,
}

fn budget_overflow() -> MmError {
    MmError::internal("glm5_next plan: pixel budget must not exceed 2^53")
}

fn resolve_budget(params: &Glm5NextImageParams) -> Result<Budget> {
    if params.patch_size == 0
        || params.merge_size == 0
        || params.temporal_patch_size == 0
        || params.patch_expand_factor == 0
    {
        return Err(MmError::internal(
            "glm5_next plan: patch_size, merge_size, temporal_patch_size and patch_expand_factor must be positive",
        ));
    }
    if params.max_image_tokens == 0 {
        return Err(MmError::internal(
            "glm5_next plan: max_image_tokens must be positive",
        ));
    }
    let frames = u64::from(params.temporal_patch_size);
    let factor = u64::from(params.patch_size)
        .checked_mul(u64::from(params.merge_size))
        .and_then(|value| value.checked_mul(u64::from(params.patch_expand_factor)))
        .ok_or_else(budget_overflow)?;
    let pixels_per_token = factor
        .checked_mul(factor)
        .and_then(|value| value.checked_mul(frames))
        .ok_or_else(budget_overflow)?;
    let min_pixels = params
        .min_image_tokens
        .checked_mul(pixels_per_token)
        .ok_or_else(budget_overflow)?;
    let max_pixels = params
        .max_image_tokens
        .checked_mul(pixels_per_token)
        .ok_or_else(budget_overflow)?;
    if min_pixels > EXACT_F64_INT || max_pixels > EXACT_F64_INT {
        return Err(budget_overflow());
    }
    Ok(Budget {
        factor,
        frames,
        min_pixels,
        max_pixels,
    })
}

fn align(value: u64, factor: u64) -> u64 {
    let blocks = (value as f64 / factor as f64).ceil();
    (blocks as u64).saturating_mul(factor)
}

fn volume(frames: u64, height: u64, width: u64) -> u128 {
    u128::from(frames) * u128::from(height) * u128::from(width)
}

fn fit_within_budget(budget: &Budget, height: u64, width: u64) -> (u64, u64) {
    let mut low: u64 = 1;
    let mut high: u64 = height;
    let mut best = (budget.factor, budget.factor);
    while low <= high {
        let content_height = (low + high) / 2;
        let scaled_width = ((width * content_height) as f64 / height as f64).floor() as u64;
        let content_width = scaled_width.max(1);
        let candidate_height = align(content_height, budget.factor);
        let candidate_width = align(content_width, budget.factor);
        let candidate = volume(budget.frames, candidate_height, candidate_width);
        if candidate <= u128::from(budget.max_pixels) {
            best = (candidate_height, candidate_width);
            low = content_height + 1;
        } else {
            high = content_height - 1;
        }
    }
    best
}

fn canvas_size(budget: &Budget, height: u64, width: u64, source_pixels: u64) -> (u64, u64) {
    let mut canvas_height = align(height, budget.factor);
    let mut canvas_width = align(width, budget.factor);
    if volume(budget.frames, canvas_height, canvas_width) < u128::from(budget.min_pixels) {
        let scale = (budget.min_pixels as f64 / source_pixels as f64).sqrt();
        let scaled_height = (height as f64 * scale).ceil() as u64;
        let scaled_width = (width as f64 * scale).ceil() as u64;
        canvas_height = align(scaled_height.max(1), budget.factor);
        canvas_width = align(scaled_width.max(1), budget.factor);
    }
    if volume(budget.frames, canvas_height, canvas_width) > u128::from(budget.max_pixels) {
        return fit_within_budget(budget, height, width);
    }
    (canvas_height, canvas_width)
}

fn to_u32(value: u64) -> Result<u32> {
    match u32::try_from(value) {
        Ok(side) => Ok(side),
        Err(_) => Err(MmError::invalid("glm5_next plan: resized side overflows")),
    }
}

pub fn plan_image(params: &Glm5NextImageParams, info: ImageInfo) -> Result<Glm5NextImagePlan> {
    let budget = resolve_budget(params)?;
    let (height, width) = if (5..=8).contains(&info.orientation) {
        (u64::from(info.width), u64::from(info.height))
    } else {
        (u64::from(info.height), u64::from(info.width))
    };
    if height == 0 || width == 0 {
        return Err(MmError::invalid(
            "glm5_next plan: image height and width must be positive",
        ));
    }
    let source_volume = volume(budget.frames, height, width);
    if source_volume > u128::from(EXACT_F64_INT) {
        return Err(MmError::invalid(
            "glm5_next plan: image has too many pixels",
        ));
    }
    let source_pixels = source_volume as u64;

    let (canvas_height, canvas_width) = canvas_size(&budget, height, width, source_pixels);

    let height_scale = canvas_height as f64 / height as f64;
    let width_scale = canvas_width as f64 / width as f64;
    let mut scale = height_scale.min(width_scale);
    if source_pixels >= budget.min_pixels {
        scale = scale.min(1.0);
    }
    let content_height = ((height as f64 * scale).floor() as u64)
        .min(canvas_height)
        .max(1);
    let content_width = ((width as f64 * scale).floor() as u64)
        .min(canvas_width)
        .max(1);

    let canvas_height = to_u32(canvas_height)?;
    let canvas_width = to_u32(canvas_width)?;
    let content_height = to_u32(content_height)?;
    let content_width = to_u32(content_width)?;
    let grid = [
        1,
        canvas_height / params.patch_size,
        canvas_width / params.patch_size,
    ];
    let merge_area = u64::from(params.merge_size) * u64::from(params.merge_size);
    let tokens = (u64::from(grid[1]) * u64::from(grid[2]) / merge_area) as usize;
    Ok(Glm5NextImagePlan {
        canvas_height,
        canvas_width,
        content_height,
        content_width,
        grid,
        expansion: vec![params.image_token_id; tokens],
    })
}

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    use super::{Glm5NextImageParams, plan_image};
    use crate::plan::ImageInfo;

    type PyGlm5NextImagePlan = (u32, u32, u32, u32, (u32, u32, u32), Vec<i64>);

    #[pyfunction]
    #[pyo3(signature = (height, width, *, orientation=1, image_token_id, patch_size, merge_size, temporal_patch_size, patch_expand_factor, min_image_tokens, max_image_tokens))]
    #[allow(clippy::too_many_arguments)]
    fn glm5_next_image(
        py: Python<'_>,
        height: u32,
        width: u32,
        orientation: u8,
        image_token_id: i64,
        patch_size: u32,
        merge_size: u32,
        temporal_patch_size: u32,
        patch_expand_factor: u32,
        min_image_tokens: u64,
        max_image_tokens: u64,
    ) -> PyResult<PyGlm5NextImagePlan> {
        let params = Glm5NextImageParams {
            image_token_id,
            patch_size,
            merge_size,
            temporal_patch_size,
            patch_expand_factor,
            min_image_tokens,
            max_image_tokens,
        };
        let info = ImageInfo {
            height,
            width,
            orientation,
        };
        let plan = py.detach(|| plan_image(&params, info))?;
        let [t, h, w] = plan.grid;
        Ok((
            plan.canvas_height,
            plan.canvas_width,
            plan.content_height,
            plan.content_width,
            (t, h, w),
            plan.expansion,
        ))
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(glm5_next_image, m)?)
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Glm5NextImageParams {
        Glm5NextImageParams {
            image_token_id: 59280,
            patch_size: 14,
            merge_size: 2,
            temporal_patch_size: 2,
            patch_expand_factor: 1,
            min_image_tokens: 16,
            max_image_tokens: 8000,
        }
    }

    fn plan(height: u32, width: u32, orientation: u8) -> Glm5NextImagePlan {
        let info = ImageInfo {
            height,
            width,
            orientation,
        };
        plan_image(&params(), info).unwrap()
    }

    #[test]
    fn image_at_the_maximum_budget_is_kept_and_one_row_more_is_searched_back() {
        let exact = plan(2800, 2240, 1);
        assert_eq!((exact.canvas_height, exact.canvas_width), (2800, 2240));
        assert_eq!((exact.content_height, exact.content_width), (2800, 2240));
        assert_eq!(exact.expansion.len(), 8000);

        let over = plan(2801, 2240, 1);
        assert_eq!((over.canvas_height, over.canvas_width), (2800, 2240));
        assert_eq!(over.grid, [1, 200, 160]);
        assert_eq!(over.expansion, vec![59280; 8000]);
        assert!(over.content_height <= 2800 && over.content_width <= 2240);
    }

    #[test]
    fn image_below_the_minimum_budget_is_upscaled_then_padded() {
        let at_minimum = plan(112, 112, 1);
        assert_eq!(
            (at_minimum.canvas_height, at_minimum.canvas_width),
            (112, 112)
        );
        assert_eq!(at_minimum.expansion.len(), 16);

        let below = plan(84, 112, 1);
        assert_eq!((below.canvas_height, below.canvas_width), (112, 140));
        assert_eq!((below.content_height, below.content_width), (105, 140));
        assert_eq!(below.expansion.len(), 20);

        let tiny = plan(1, 1, 1);
        assert_eq!((tiny.canvas_height, tiny.canvas_width), (112, 112));
        assert_eq!((tiny.content_height, tiny.content_width), (112, 112));
    }

    #[test]
    fn image_inside_the_budget_keeps_its_pixels_on_a_canvas_aligned_up() {
        let inside = plan(100, 300, 1);
        assert_eq!((inside.canvas_height, inside.canvas_width), (112, 308));
        assert_eq!((inside.content_height, inside.content_width), (100, 300));
        assert_eq!(inside.grid, [1, 8, 22]);
        assert_eq!(inside.expansion.len(), 44);

        let expanded = Glm5NextImageParams {
            patch_expand_factor: 4,
            ..params()
        };
        let info = ImageInfo {
            height: 100,
            width: 300,
            orientation: 1,
        };
        let wide = plan_image(&expanded, info).unwrap();
        assert_eq!((wide.canvas_height, wide.canvas_width), (336, 784));
        assert_eq!(wide.grid, [1, 24, 56]);
    }

    #[test]
    fn transposing_exif_orientations_swap_height_and_width() {
        for orientation in 1..=4 {
            assert_eq!(plan(100, 300, orientation), plan(100, 300, 1));
        }
        for orientation in 5..=8 {
            let rotated = plan(100, 300, orientation);
            assert_eq!((rotated.canvas_height, rotated.canvas_width), (308, 112));
            assert_eq!((rotated.content_height, rotated.content_width), (300, 100));
        }
    }

    #[test]
    fn empty_image_is_a_request_error_and_zero_budget_is_an_internal_error() {
        let empty = ImageInfo {
            height: 0,
            width: 64,
            orientation: 1,
        };
        assert!(matches!(
            plan_image(&params(), empty),
            Err(MmError::InvalidRequest(_))
        ));

        let no_budget = Glm5NextImageParams {
            max_image_tokens: 0,
            ..params()
        };
        let info = ImageInfo {
            height: 64,
            width: 64,
            orientation: 1,
        };
        assert!(matches!(
            plan_image(&no_budget, info),
            Err(MmError::Internal(_))
        ));
    }
}
