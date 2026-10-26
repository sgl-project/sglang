//! DeepSeek-V4.1 images through deepseek-recipe-image before `sglang_processor::mm`:
//! decode, alpha blend, resize and pad to the token budget, re-encode as WebP.

use bytes::Bytes;

#[cfg(feature = "deepseek-image")]
pub(super) struct RecipeImages(tokio::runtime::Runtime);

#[cfg(feature = "deepseek-image")]
impl RecipeImages {
    pub(super) fn new() -> Result<Self, String> {
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
    pub(super) fn new() -> Result<Self, String> {
        Err("DeepSeek-V4.1 images need sglang-server built with the deepseek-image feature".into())
    }

    pub(super) fn preprocess(&self, _images: &[Bytes]) -> Result<Vec<Vec<u8>>, String> {
        match self.0 {}
    }
}
