//! DeepSeek-V4.1 images through deepseek-recipe-image before `sglang_processor::mm`:
//! decode, alpha blend, resize and pad to the token budget, re-encode as WebP.

use bytes::Bytes;
use sglang_processor::dynamo_multimodal::models::deepseek_v41::DeepseekV41GeometrySpec;

#[cfg(feature = "deepseek-image")]
pub(super) struct RecipeImages(tokio::runtime::Runtime);

#[cfg(feature = "deepseek-image")]
impl RecipeImages {
    pub(super) fn new(geometry: &DeepseekV41GeometrySpec) -> Result<Self, String> {
        use deepseek_recipe_core::multimodal::ImageTokenSpec;

        // Recipe resizes to `ImageTokenSpec::v41()` alone, whose minimum area
        // (544 * 544) has no getter.
        let recipe = ImageTokenSpec::v41();
        let served = (
            recipe.patch_size(),
            recipe.downsample_ratio(),
            544 * 544,
            recipe.max_token_count(),
            None,
        );
        let requested = (
            geometry.vision_patch_size as usize,
            geometry.vision_downsample_ratio as usize,
            geometry.vision_min_pixels as usize,
            geometry.vision_max_n_token as usize,
            geometry.vision_max_wh_ratio,
        );
        if requested != served {
            return Err(format!(
                "deepseek-recipe-image serves only DeepSeek-V4.1's default image geometry, not {geometry:?}"
            ));
        }
        // `preprocess` runs OpenCV through `spawn_blocking`, so it needs a runtime.
        let runtime = tokio::runtime::Builder::new_current_thread().build();
        runtime.map(Self).map_err(|error| error.to_string())
    }

    pub(super) fn preprocess(&self, images: &[Bytes]) -> Result<Vec<Vec<u8>>, String> {
        use deepseek_recipe_core::multimodal::ImageDetail;
        use deepseek_recipe_image::{ImageLimits, ImagePreprocessor, OpenCvImagePreprocessor};

        let options =
            ImageLimits::default().preprocess_options(ImageDetail::default(), images.len());
        (images.iter())
            .map(|image| {
                let info = self
                    .0
                    .block_on(OpenCvImagePreprocessor.preprocess(image.to_vec(), options));
                info.map(|info| info.data)
                    .map_err(|error| error.to_string())
            })
            .collect()
    }
}

#[cfg(not(feature = "deepseek-image"))]
pub(super) struct RecipeImages(std::convert::Infallible);

#[cfg(not(feature = "deepseek-image"))]
impl RecipeImages {
    pub(super) fn new(_geometry: &DeepseekV41GeometrySpec) -> Result<Self, String> {
        Err("DeepSeek-V4.1 images need sglang-server built with the deepseek-image feature".into())
    }

    pub(super) fn preprocess(&self, _images: &[Bytes]) -> Result<Vec<Vec<u8>>, String> {
        match self.0 {}
    }
}
