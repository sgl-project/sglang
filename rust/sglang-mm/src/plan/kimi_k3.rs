use super::ImageInfo;
use crate::error::{MmError, Result};

pub const MEDIA_END_TEXT: &str = "<|media_end|>";

#[derive(Clone, Debug, serde::Deserialize)]
pub struct KimiK3ImageParams {
    pub image_token_id: i64,
    pub patch_size: u32,
    pub merge_kernel_size: u32,
    pub in_patch_limit: u64,
    pub patch_limit_on_one_side: u32,
    #[serde(default)]
    pub fixed_output_tokens: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KimiK3ImageGeometry {
    pub resized_height: u32,
    pub resized_width: u32,
    pub pad_height: u32,
    pub pad_width: u32,
    pub grid: [u32; 3],
    pub num_image_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KimiK3ImagePlan {
    pub resized_height: u32,
    pub resized_width: u32,
    pub grid: [u32; 3],
    pub expansion: Vec<i64>,
    pub pad_height: u32,
    pub pad_width: u32,
    pub placeholder_start: usize,
    pub num_image_tokens: usize,
}

fn to_u32(value: u64, name: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| MmError::internal(format!("kimi_k3 plan: {name} does not fit in u32")))
}

pub fn plan_geometry(params: &KimiK3ImageParams, info: ImageInfo) -> Result<KimiK3ImageGeometry> {
    if params.patch_size == 0 || params.merge_kernel_size == 0 {
        return Err(MmError::internal(
            "kimi_k3 plan: patch_size and merge_kernel_size must be positive",
        ));
    }
    if params.patch_limit_on_one_side == 0 {
        return Err(MmError::internal(
            "kimi_k3 plan: patch_limit_on_one_side must be positive",
        ));
    }
    if info.height == 0 || info.width == 0 {
        return Err(MmError::invalid(format!(
            "kimi_k3 plan: image size must be positive, got {}x{}",
            info.width, info.height
        )));
    }
    let patch = u64::from(params.patch_size);
    let width = u64::from(info.width);
    let height = u64::from(info.height);
    let side_limit = u64::from(params.patch_limit_on_one_side) * patch;

    let width_patches = ((width / patch) as f64).max(1.0);
    let height_patches = ((height / patch) as f64).max(1.0);
    let s1 = (params.in_patch_limit as f64 / (width_patches * height_patches)).sqrt();
    let s2 = side_limit as f64 / width as f64;
    let s3 = side_limit as f64 / height as f64;
    let scale = 1.0_f64.min(s1).min(s2).min(s3);
    let new_width = ((width as f64 * scale) as u64).max(1).min(side_limit);
    let new_height = ((height as f64 * scale) as u64).max(1).min(side_limit);

    let factor = u64::from(params.merge_kernel_size) * patch;
    let pad_height = (factor - new_height % factor) % factor;
    let pad_width = (factor - new_width % factor) % factor;
    let padded_height = new_height + pad_height;
    let padded_width = new_width + pad_width;

    let num_tokens = match params.fixed_output_tokens {
        Some(fixed) => fixed,
        None => (padded_height / factor) * (padded_width / factor),
    };
    let num_image_tokens = usize::try_from(num_tokens)
        .map_err(|_| MmError::internal("kimi_k3 plan: token count does not fit in usize"))?;

    Ok(KimiK3ImageGeometry {
        resized_height: to_u32(new_height, "resized height")?,
        resized_width: to_u32(new_width, "resized width")?,
        pad_height: to_u32(pad_height, "pad height")?,
        pad_width: to_u32(pad_width, "pad width")?,
        grid: [
            1,
            to_u32(padded_height / patch, "grid height")?,
            to_u32(padded_width / patch, "grid width")?,
        ],
        num_image_tokens,
    })
}

pub fn media_begin_text(info: ImageInfo) -> String {
    format!(
        "<|media_begin|>image {}x{}<|media_content|>",
        info.width, info.height
    )
}

pub fn assemble(
    params: &KimiK3ImageParams,
    geometry: KimiK3ImageGeometry,
    begin_ids: Vec<i64>,
    end_ids: Vec<i64>,
) -> KimiK3ImagePlan {
    let placeholder_start = begin_ids.len();
    let mut expansion = begin_ids;
    expansion.reserve(geometry.num_image_tokens + end_ids.len());
    expansion.extend(std::iter::repeat_n(
        params.image_token_id,
        geometry.num_image_tokens,
    ));
    expansion.extend(end_ids);
    KimiK3ImagePlan {
        resized_height: geometry.resized_height,
        resized_width: geometry.resized_width,
        grid: geometry.grid,
        expansion,
        pad_height: geometry.pad_height,
        pad_width: geometry.pad_width,
        placeholder_start,
        num_image_tokens: geometry.num_image_tokens,
    }
}

pub fn plan_image(
    params: &KimiK3ImageParams,
    info: ImageInfo,
    encode: &mut dyn FnMut(&str) -> Result<Vec<i64>>,
) -> Result<KimiK3ImagePlan> {
    let geometry = plan_geometry(params, info)?;
    let begin_text = media_begin_text(info);
    let begin_ids = encode(begin_text.as_str())?;
    let end_ids = encode(MEDIA_END_TEXT)?;
    Ok(assemble(params, geometry, begin_ids, end_ids))
}

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    use super::{KimiK3ImageParams, MEDIA_END_TEXT, assemble, media_begin_text, plan_geometry};
    use crate::plan::ImageInfo;

    type PyKimiK3ImagePlan = (
        u32,
        u32,
        (u32, u32, u32),
        Vec<i64>,
        (u32, u32),
        (usize, usize),
    );

    fn encode_text(encode: &Bound<'_, PyAny>, text: &str) -> PyResult<Vec<i64>> {
        let ids = encode.call1((text,))?;
        ids.extract::<Vec<i64>>()
    }

    #[pyfunction]
    #[pyo3(signature = (height, width, encode, *, image_token_id, patch_size, merge_kernel_size, in_patch_limit, patch_limit_on_one_side, fixed_output_tokens=None))]
    #[allow(clippy::too_many_arguments)]
    fn kimi_k3_image(
        py: Python<'_>,
        height: u32,
        width: u32,
        encode: &Bound<'_, PyAny>,
        image_token_id: i64,
        patch_size: u32,
        merge_kernel_size: u32,
        in_patch_limit: u64,
        patch_limit_on_one_side: u32,
        fixed_output_tokens: Option<u64>,
    ) -> PyResult<PyKimiK3ImagePlan> {
        let params = KimiK3ImageParams {
            image_token_id,
            patch_size,
            merge_kernel_size,
            in_patch_limit,
            patch_limit_on_one_side,
            fixed_output_tokens,
        };
        let info = ImageInfo {
            height,
            width,
            orientation: 1,
        };
        let geometry = py.detach(|| plan_geometry(&params, info))?;
        let begin_text = media_begin_text(info);
        let begin_ids = encode_text(encode, begin_text.as_str())?;
        let end_ids = encode_text(encode, MEDIA_END_TEXT)?;
        let plan = assemble(&params, geometry, begin_ids, end_ids);
        let [t, h, w] = plan.grid;
        Ok((
            plan.resized_height,
            plan.resized_width,
            (t, h, w),
            plan.expansion,
            (plan.pad_height, plan.pad_width),
            (plan.placeholder_start, plan.num_image_tokens),
        ))
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(kimi_k3_image, m)?)
    }
}

#[cfg(feature = "python")]
pub use python::register;

#[cfg(test)]
mod tests {
    use super::*;

    const PAD: i64 = 7;

    fn params() -> KimiK3ImageParams {
        KimiK3ImageParams {
            image_token_id: PAD,
            patch_size: 14,
            merge_kernel_size: 2,
            in_patch_limit: 16384,
            patch_limit_on_one_side: 512,
            fixed_output_tokens: None,
        }
    }

    fn info(height: u32, width: u32) -> ImageInfo {
        ImageInfo {
            height,
            width,
            orientation: 1,
        }
    }

    fn plan_with_texts(
        params: &KimiK3ImageParams,
        info: ImageInfo,
    ) -> (Result<KimiK3ImagePlan>, Vec<String>) {
        let mut texts: Vec<String> = Vec::new();
        let plan = {
            let mut encode = |text: &str| -> Result<Vec<i64>> {
                texts.push(text.to_string());
                Ok(vec![1000 + texts.len() as i64; 3])
            };
            plan_image(params, info, &mut encode)
        };
        (plan, texts)
    }

    #[test]
    fn header_carries_the_original_width_then_height() {
        let (plan, texts) = plan_with_texts(&params(), info(3000, 5000));
        let plan = plan.unwrap();
        assert!(plan.resized_width < 5000 && plan.resized_height < 3000);
        assert_eq!(
            texts,
            vec![
                "<|media_begin|>image 5000x3000<|media_content|>".to_string(),
                "<|media_end|>".to_string(),
            ]
        );
    }

    #[test]
    fn placeholders_sit_between_the_encoded_texts() {
        let (plan, _) = plan_with_texts(&params(), info(100, 60));
        let plan = plan.unwrap();
        assert_eq!((plan.resized_height, plan.resized_width), (100, 60));
        assert_eq!((plan.pad_height, plan.pad_width), (12, 24));
        assert_eq!(plan.grid, [1, 8, 6]);
        assert_eq!((plan.placeholder_start, plan.num_image_tokens), (3, 12));
        let mut expected = vec![1001; 3];
        expected.extend(vec![PAD; 12]);
        expected.extend(vec![1002; 3]);
        assert_eq!(plan.expansion, expected);
    }

    #[test]
    fn aligned_size_has_no_padding_and_one_more_pixel_adds_a_row() {
        let aligned = plan_geometry(&params(), info(56, 28)).unwrap();
        assert_eq!((aligned.pad_height, aligned.pad_width), (0, 0));
        assert_eq!(aligned.num_image_tokens, 2);
        let next = plan_geometry(&params(), info(57, 28)).unwrap();
        assert_eq!((next.resized_height, next.pad_height), (57, 27));
        assert_eq!(next.num_image_tokens, 3);
    }

    #[test]
    fn large_image_is_scaled_down_with_truncation_and_never_up() {
        let geometry = plan_geometry(&params(), info(3000, 5000)).unwrap();
        assert_eq!(
            (geometry.resized_height, geometry.resized_width),
            (1389, 2315)
        );
        assert_eq!(geometry.grid, [1, 100, 166]);
        assert_eq!(geometry.num_image_tokens, 50 * 83);
        let tiny = plan_geometry(&params(), info(1, 1)).unwrap();
        assert_eq!((tiny.resized_height, tiny.resized_width), (1, 1));
        assert_eq!(tiny.grid, [1, 2, 2]);
        assert_eq!(tiny.num_image_tokens, 1);
    }

    #[test]
    fn one_side_limit_caps_a_long_strip() {
        let geometry = plan_geometry(&params(), info(10, 20000)).unwrap();
        assert_eq!(geometry.resized_width, 512 * 14);
        assert_eq!(geometry.resized_height, 3);
    }

    #[test]
    fn fixed_output_tokens_overrides_the_count_but_not_the_grid() {
        let mut fixed = params();
        fixed.fixed_output_tokens = Some(5);
        let geometry = plan_geometry(&fixed, info(100, 60)).unwrap();
        assert_eq!(geometry.grid, [1, 8, 6]);
        assert_eq!(geometry.num_image_tokens, 5);
    }

    #[test]
    fn zero_size_is_a_request_error_and_no_text_is_encoded() {
        let (plan, texts) = plan_with_texts(&params(), info(0, 10));
        assert!(matches!(plan, Err(MmError::InvalidRequest(_))));
        assert!(texts.is_empty());
    }

    #[test]
    fn zero_patch_size_is_an_internal_error() {
        let mut broken = params();
        broken.patch_size = 0;
        assert!(matches!(
            plan_geometry(&broken, info(10, 10)),
            Err(MmError::Internal(_))
        ));
    }

    #[test]
    fn encode_failure_is_returned_unchanged() {
        let mut encode =
            |_: &str| -> Result<Vec<i64>> { Err(MmError::internal("tokenizer failed")) };
        assert_eq!(
            plan_image(&params(), info(10, 10), &mut encode),
            Err(MmError::internal("tokenizer failed"))
        );
    }
}
