//! DeepSeek-V4.1 images after deepseek-recipe-image, which the host runs first:
//! the decoded image is already resized and padded to its patch grid, so this
//! normalizes and patchifies as `deepseek_v41_image_processing.patchify_image`.

use dynamo_multimodal::MmError;
use dynamo_multimodal::models::deepseek_v41::DeepseekV41Spec;
use dynamo_multimodal::processor::{
    DecodedMedia, Geometry, MediaMetadata, MmFamilyProcessor, Modality, ProcessedItem, Tensor,
    TensorData, TokenLayout,
};
use dynamo_multimodal::token_layout::layout_by_placeholder;

pub(super) struct DeepseekV41 {
    spec: DeepseekV41Spec,
}

impl DeepseekV41 {
    pub(super) fn new(spec: DeepseekV41Spec) -> Result<Self, MmError> {
        let geometry = &spec.geometry;
        if geometry.vision_patch_size == 0 || geometry.vision_downsample_ratio == 0 {
            return Err(MmError::invalid_input(
                "deepseek_v41: zero patch size or downsample ratio",
            ));
        }
        Ok(Self { spec })
    }
}

impl MmFamilyProcessor for DeepseekV41 {
    fn num_media_tokens(&self, _media: &MediaMetadata) -> Result<usize, MmError> {
        Err(MmError::unsupported(
            "deepseek_v41: tokens follow deepseek-recipe-image's resize",
        ))
    }

    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, MmError> {
        let DecodedMedia::Image { rgb, height, width } = media else {
            return Err(MmError::unsupported("deepseek_v41 supports only images"));
        };
        let p = self.spec.geometry.vision_patch_size as usize;
        let ratio = self.spec.geometry.vision_downsample_ratio as usize;
        if height % p != 0 || width % p != 0 {
            return Err(MmError::invalid_input(format!(
                "deepseek_v41: {width}x{height} is not on the {p}-pixel patch grid"
            )));
        }
        let (vit_h, vit_w) = (height / p, width / p);
        let (llm_h, llm_w) = (vit_h.div_ceil(ratio), vit_w.div_ceil(ratio));
        // `image_token_types`: start, each row plus its separator, end.
        let feature_token_count = llm_h * (llm_w + 1) + 2;
        let max_tokens = self.spec.geometry.vision_max_n_token as usize;
        if feature_token_count > max_tokens {
            return Err(MmError::invalid_input(format!(
                "deepseek_v41: {width}x{height} takes {feature_token_count} tokens, over the {max_tokens}-token budget"
            )));
        }
        // [vit_h * vit_w, 3, p, p], patches in reading order.
        let mut feature = Vec::with_capacity(rgb.len());
        for (row, col) in (0..vit_h).flat_map(|row| (0..vit_w).map(move |col| (row, col))) {
            for channel in 0..3 {
                for y in row * p..(row + 1) * p {
                    let pixels = &rgb[(y * width + col * p) * 3..(y * width + (col + 1) * p) * 3];
                    feature.extend(
                        (pixels.iter().skip(channel).step_by(3))
                            .map(|&value| (value as f32 / 255.0 - 0.5) / 0.5),
                    );
                }
            }
        }
        let scalar = |value: usize| Tensor {
            shape: Vec::new(),
            data: TensorData::I64(vec![value as i64]),
        };
        Ok(ProcessedItem {
            modality: Modality::Image,
            feature_token_count,
            feature: Tensor {
                shape: vec![vit_h * vit_w, 3, p, p],
                data: TensorData::F32(feature),
            },
            aux: vec![
                ("n_vit_h".into(), scalar(vit_h)),
                ("n_vit_w".into(), scalar(vit_w)),
            ],
            geometry: Some(Geometry::Grid([1, vit_h as u32, vit_w as u32])),
        })
    }

    fn layout(&self, input_ids: &[i32], items: &[ProcessedItem]) -> Result<TokenLayout, MmError> {
        let counts: Vec<usize> = items.iter().map(|item| item.feature_token_count).collect();
        layout_by_placeholder(input_ids, self.spec.image_token_id, &counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patchifies_a_presized_image() {
        let spec = serde_json::from_value(serde_json::json!({
            "image_token_id": 9, "vision_patch_size": 2, "vision_downsample_ratio": 3,
            "vision_min_pixels": 0, "vision_max_n_token": 1024,
        }))
        .unwrap();
        let family = DeepseekV41::new(spec).unwrap();
        // 2 x 8 pixels: a 1 x 4 patch grid, a 1 x 2 token grid.
        let rgb: Vec<u8> = (0..2 * 8 * 3).map(|i| i as u8).collect();
        let item = (family.process_item(&DecodedMedia::Image {
            rgb,
            height: 2,
            width: 8,
        }))
        .unwrap();
        // One row: two tokens and its separator, then start and end.
        assert_eq!(item.feature_token_count, 5);
        assert_eq!(item.feature.shape, [4, 3, 2, 2]);
        let TensorData::F32(feature) = &item.feature.data else {
            unreachable!()
        };
        // Patch 1, green: pixels (0, 2), (0, 3), (1, 2), (1, 3).
        let green: Vec<f32> = [7u8, 10, 31, 34]
            .iter()
            .map(|&v| (v as f32 / 255.0 - 0.5) / 0.5)
            .collect();
        assert_eq!(feature[12 + 4..12 + 8], green);
        assert_eq!(
            family.layout(&[1, 9, 2], &[item]).unwrap().segments.len(),
            3
        );
    }

    #[test]
    fn rejects_an_image_over_the_token_budget() {
        let spec = serde_json::from_value(serde_json::json!({
            "image_token_id": 9, "vision_patch_size": 2, "vision_downsample_ratio": 3,
            "vision_min_pixels": 0, "vision_max_n_token": 4,
        }))
        .unwrap();
        let family = DeepseekV41::new(spec).unwrap();
        // A 1 x 2 token grid takes 5 tokens.
        let image = DecodedMedia::Image {
            rgb: vec![0; 2 * 8 * 3],
            height: 2,
            width: 8,
        };
        assert!(family.process_item(&image).is_err());
    }
}
