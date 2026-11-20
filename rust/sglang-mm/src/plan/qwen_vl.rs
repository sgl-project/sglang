use super::{ImageInfo, ImagePlan};
use crate::error::{MmError, Result};
use crate::qwen_vl::smart_resize;

#[derive(Clone, Debug, serde::Deserialize)]
pub struct QwenVlImageParams {
    pub image_token_id: i64,
    pub patch_size: u32,
    pub merge_size: u32,
    pub min_pixels: u64,
    pub max_pixels: u64,
}

pub fn plan_image(params: &QwenVlImageParams, info: ImageInfo) -> Result<ImagePlan> {
    if params.patch_size == 0 || params.merge_size == 0 {
        return Err(MmError::internal(
            "qwen_vl plan: patch_size and merge_size must be positive",
        ));
    }
    let factor = params.patch_size * params.merge_size;
    let (resized_height, resized_width) = smart_resize(
        info.height as usize,
        info.width as usize,
        factor as usize,
        params.min_pixels as usize,
        params.max_pixels as usize,
    )
    .map_err(MmError::invalid)?;
    let (resized_height, resized_width) = (resized_height as u32, resized_width as u32);
    let grid = [
        1,
        resized_height / params.patch_size,
        resized_width / params.patch_size,
    ];
    let tokens = (grid[1] / params.merge_size) as usize * (grid[2] / params.merge_size) as usize;
    Ok(ImagePlan {
        resized_height,
        resized_width,
        grid,
        expansion: vec![params.image_token_id; tokens],
    })
}

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    use super::{QwenVlImageParams, plan_image};
    use crate::plan::ImageInfo;

    type PyImagePlan = (u32, u32, (u32, u32, u32), Vec<i64>);

    #[pyfunction]
    #[pyo3(signature = (height, width, *, image_token_id, patch_size, merge_size, min_pixels, max_pixels))]
    #[allow(clippy::too_many_arguments)]
    fn qwen_vl_image(
        py: Python<'_>,
        height: u32,
        width: u32,
        image_token_id: i64,
        patch_size: u32,
        merge_size: u32,
        min_pixels: u64,
        max_pixels: u64,
    ) -> PyResult<PyImagePlan> {
        let params = QwenVlImageParams {
            image_token_id,
            patch_size,
            merge_size,
            min_pixels,
            max_pixels,
        };
        let info = ImageInfo {
            height,
            width,
            orientation: 1,
        };
        let plan = py.detach(|| plan_image(&params, info))?;
        let [t, h, w] = plan.grid;
        Ok((
            plan.resized_height,
            plan.resized_width,
            (t, h, w),
            plan.expansion,
        ))
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(qwen_vl_image, m)?)
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> QwenVlImageParams {
        QwenVlImageParams {
            image_token_id: 151655,
            patch_size: 14,
            merge_size: 2,
            min_pixels: 56 * 56,
            max_pixels: 14 * 14 * 4 * 1280,
        }
    }

    #[test]
    fn plans_the_grid_and_one_token_per_merged_patch() {
        let info = ImageInfo {
            height: 1024,
            width: 1024,
            orientation: 1,
        };
        let plan = plan_image(&params(), info).unwrap();
        assert_eq!((plan.resized_height, plan.resized_width), (980, 980));
        assert_eq!(plan.grid, [1, 70, 70]);
        assert_eq!(plan.expansion, vec![151655; 35 * 35]);
    }

    #[test]
    fn extreme_aspect_ratio_is_a_request_error() {
        let info = ImageInfo {
            height: 10,
            width: 4000,
            orientation: 1,
        };
        assert!(matches!(
            plan_image(&params(), info),
            Err(MmError::InvalidRequest(_))
        ));
    }
}
