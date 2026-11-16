//! Public, model-profiled CPU inputs. No fetching or preprocessing occurs here;
//! the result joins the same encoded-item path as the native image processor.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use sglang_api_types::api::v1 as api;
use sglang_mm::pipeline::{Tensor, TensorData};
use sglang_processor::dynamo_multimodal::models::qwen_vl::{
    MropeItem, QwenVlSpec, mrope_image_only,
};
use sglang_processor::dynamo_multimodal::registry::ProcessorSpec;

use super::encoded::{MRope, MmEncodedEntry, MmEncodedItem, MmMetaValue, MmModality};
use super::worker::MmProcessOutput;
use crate::message::types::TokenIds;

pub(crate) const QWEN_VL_IMAGE_V1: &str = "qwen_vl_image_v1";
const MAX_INLINE_BYTES: u64 = 64 << 20;

pub(crate) struct PreparedProfile {
    spec: QwenVlSpec,
    fingerprint: String,
}

impl PreparedProfile {
    fn qwen(spec: QwenVlSpec) -> Self {
        // Domain and fixed-width LE fields define this profile's fingerprint.
        // Never hash a Debug rendering, a local path or an internal protocol.
        let mut hash = Sha256::new();
        hash.update(b"sglang-prepared-qwen-vl-image-v1\0");
        hash.update(spec.image_token_id.to_le_bytes());
        for size in [
            spec.patch_size,
            spec.merge_size,
            spec.temporal_patch_size,
            spec.min_pixels,
            spec.max_pixels,
        ] {
            hash.update((size as u64).to_le_bytes());
        }
        for value in spec.image_mean.into_iter().chain(spec.image_std) {
            hash.update(value.to_bits().to_le_bytes());
        }
        use sglang_processor::dynamo_multimodal::models::qwen_vl::Resampler;
        hash.update(match spec.resample {
            Resampler::AtenU8 => b"aten_u8".as_slice(),
            Resampler::Pil => b"pil".as_slice(),
        });
        Self {
            spec,
            fingerprint: format!("{:x}", hash.finalize()),
        }
    }

    pub(crate) fn from_spec(spec: &ProcessorSpec) -> Option<Self> {
        match spec {
            ProcessorSpec::QwenVl(spec) => Some(Self::qwen(spec.clone())),
            _ => None,
        }
    }

    pub(crate) fn capability(&self) -> Option<api::PreparedInputCapability> {
        Some(api::PreparedInputCapability {
            profile: QWEN_VL_IMAGE_V1.into(),
            image_token_id: self.spec.image_token_id.try_into().ok()?,
            patch_size: self.spec.patch_size.try_into().ok()?,
            merge_size: self.spec.merge_size.try_into().ok()?,
            temporal_patch_size: self.spec.temporal_patch_size.try_into().ok()?,
            max_inline_bytes: MAX_INLINE_BYTES,
            processor_fingerprint: self.fingerprint.clone(),
        })
    }

    /// Cheap admission checks: sizes and byte arithmetic before allocating
    /// native tensors. The gRPC/HTTP body limit applies before this layer.
    pub(crate) fn check(&self, input: &api::PreparedMultimodalInput) -> Result<(), String> {
        if input.profile != QWEN_VL_IMAGE_V1 || self.capability().is_none() {
            return Err("unsupported prepared multimodal profile".into());
        }
        if input.processor_fingerprint != self.fingerprint {
            return Err(
                "prepared processor_fingerprint does not match the served processor".into(),
            );
        }
        if input.items.is_empty() {
            return Err("prepared_multimodal requires at least one image".into());
        }
        let mut total = 0u64;
        let patch_dim = 3usize
            .checked_mul(self.spec.temporal_patch_size)
            .and_then(|n| n.checked_mul(self.spec.patch_size))
            .and_then(|n| n.checked_mul(self.spec.patch_size))
            .ok_or("invalid processor patch dimension")?;
        let merge = self.spec.merge_size;
        if patch_dim == 0 || merge == 0 {
            return Err("invalid processor geometry".into());
        }
        let mut previous_end = 0;
        for image in &input.items {
            let (tensor, grid) = qwen_item(image)?;
            let (h, w) = (grid[1] as usize, grid[2] as usize);
            if h == 0 || w == 0 || !h.is_multiple_of(merge) || !w.is_multiple_of(merge) {
                return Err(
                    "prepared image grid must be positive and divisible by merge_size".into(),
                );
            }
            let patches = h.checked_mul(w).ok_or("prepared grid overflow")?;
            let tokens = (h / merge)
                .checked_mul(w / merge)
                .ok_or("prepared span overflow")?;
            let (start, end) = qwen_span(image)?;
            if start < previous_end || end <= start || u64::from(end - start) != tokens as u64 {
                return Err(
                    "prepared image spans must be ordered, disjoint and match the grid".into(),
                );
            }
            previous_end = end;
            if tensor.dtype != "float32"
                || tensor.shape.as_slice() != [patches as u64, patch_dim as u64]
            {
                return Err("prepared pixel_values must be float32 [grid_h * grid_w, 3 * temporal_patch_size * patch_size ** 2]".into());
            }
            let bytes = patches
                .checked_mul(patch_dim)
                .and_then(|n| n.checked_mul(4))
                .ok_or("prepared tensor byte size overflow")?;
            if bytes != tensor.data.len() {
                return Err("prepared tensor shape does not match its byte length".into());
            }
            total = total
                .checked_add(bytes as u64)
                .ok_or("prepared byte budget overflow")?;
            if total > MAX_INLINE_BYTES {
                return Err("prepared tensors exceed the request media byte budget".into());
            }
        }
        Ok(())
    }

    pub(crate) fn encode(
        &self,
        input_ids: TokenIds,
        input: api::PreparedMultimodalInput,
    ) -> Result<MmProcessOutput, String> {
        self.check(&input)?;
        // Every image-pad token belongs to exactly one declared item. Also
        // reject extra pads outside spans rather than silently losing alignment.
        let image_token_id = i64::from(self.spec.image_token_id);
        let mut cursor = 0usize;
        let mut geometry = Vec::with_capacity(input.items.len());
        for image in &input.items {
            let (start, end) = qwen_span(image)?;
            let (start, end) = (start as usize, end as usize);
            if end > input_ids.len()
                || input_ids[cursor..start].contains(&image_token_id)
                || input_ids[start..end].iter().any(|&id| id != image_token_id)
            {
                return Err("prepared image spans do not match expanded input_ids".into());
            }
            geometry.push(MropeItem {
                start: start as u32,
                end: (end - 1) as u32,
                grid: qwen_item(image)?.1,
            });
            cursor = end;
        }
        if input_ids[cursor..].contains(&image_token_id) {
            return Err("image-pad tokens outside prepared image spans".into());
        }
        // Reuse the native model-family implementation, never caller-supplied
        // unchecked M-RoPE positions or a duplicate position algorithm.
        let (positions, delta) = mrope_image_only(input_ids.len(), &geometry, self.spec.merge_size)
            .map_err(|e| e.to_string())?;
        let items = input
            .items
            .into_iter()
            .map(|image| {
                let grid = qwen_item(&image)?.1;
                let (start, end) = qwen_span(&image)?;
                let tensor = image
                    .tensors
                    .into_iter()
                    .next()
                    .and_then(|t| t.tensor)
                    .ok_or("missing prepared pixel_values")?;
                let values = tensor
                    .data
                    .chunks_exact(4)
                    .map(|chunk| {
                        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        if value.is_finite() {
                            Ok(value)
                        } else {
                            Err("prepared pixel_values must be finite")
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut item = MmEncodedItem {
                    modality: MmModality::Image,
                    feature: Tensor {
                        shape: tensor.shape.into_iter().map(|d| d as usize).collect(),
                        data: TensorData::F32(values),
                    },
                    // No caller-provided cache identity. Use the same typed-content
                    // identity as the raw processor, including grid metadata.
                    hash: 0,
                    identity: None,
                    offsets: vec![(start, end - 1)],
                    model_specific_data: BTreeMap::from([(
                        "image_grid_thw".into(),
                        MmMetaValue::Ints(grid.into_iter().map(i64::from).collect()),
                    )]),
                };
                let identity = item.cache_identity()?;
                item.hash = u64::from_be_bytes(
                    identity[..8]
                        .try_into()
                        .map_err(|_| "invalid item identity")?,
                );
                item.identity = Some(identity);
                Ok(item)
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(MmProcessOutput {
            input_ids,
            result: MmEncodedEntry {
                items,
                token_ids: None,
                mrope: Some(MRope { positions, delta }),
            },
            extension: None,
        })
    }
}

// Decode the profile's vocabulary strictly. The envelope stays usable by other
// processors without treating Qwen grid geometry as universal metadata.
fn qwen_span(item: &api::PreparedMediaItem) -> Result<(u32, u32), String> {
    let [span] = item.spans.as_slice() else {
        return Err("qwen_vl_image_v1 requires exactly one span per image".into());
    };
    Ok((span.start, span.end))
}

fn qwen_item(item: &api::PreparedMediaItem) -> Result<(&api::InlineTensor, [u32; 3]), String> {
    if item.modality != "image" || item.tensors.len() != 1 || item.metadata.len() != 1 {
        return Err(
            "qwen_vl_image_v1 requires image items with pixel_values and image_grid_thw only"
                .into(),
        );
    }
    let tensor = &item.tensors[0];
    let metadata = &item.metadata[0];
    if tensor.name != "pixel_values" || metadata.name != "image_grid_thw" {
        return Err("unsupported tensor or metadata name for qwen_vl_image_v1".into());
    }
    let [1, h, w] = metadata.values.as_slice() else {
        return Err("prepared image_grid_thw must be [1, h, w]".into());
    };
    let grid = [
        1,
        (*h).try_into()
            .map_err(|_| "prepared grid dimension outside uint32")?,
        (*w).try_into()
            .map_err(|_| "prepared grid dimension outside uint32")?,
    ];
    Ok((
        tensor
            .tensor
            .as_ref()
            .ok_or("missing prepared pixel_values")?,
        grid,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sglang_processor::dynamo_multimodal::models::qwen_vl::Resampler;

    fn profile() -> PreparedProfile {
        PreparedProfile::qwen(QwenVlSpec {
            image_token_id: 99,
            patch_size: 1,
            merge_size: 2,
            temporal_patch_size: 1,
            min_pixels: 4,
            max_pixels: 1024,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            resample: Resampler::AtenU8,
        })
    }

    fn input() -> api::PreparedMultimodalInput {
        api::PreparedMultimodalInput {
            profile: QWEN_VL_IMAGE_V1.into(),
            processor_fingerprint: profile().fingerprint,
            items: vec![api::PreparedMediaItem {
                modality: "image".into(),
                spans: vec![api::PreparedTokenSpan { start: 1, end: 2 }],
                tensors: vec![api::PreparedTensor {
                    name: "pixel_values".into(),
                    tensor: Some(api::InlineTensor {
                        dtype: "float32".into(),
                        shape: vec![4, 3],
                        data: (0..12).flat_map(|n| (n as f32).to_le_bytes()).collect(),
                    }),
                }],
                metadata: vec![api::PreparedMetadata {
                    name: "image_grid_thw".into(),
                    values: vec![1, 2, 2],
                }],
            }],
        }
    }

    #[test]
    fn prepared_items_preserve_ids_features_and_native_positions() {
        let output = profile().encode(vec![10, 99, 20], input()).unwrap();
        assert_eq!(output.input_ids, vec![10, 99, 20]);
        assert_eq!(output.result.items[0].offsets, vec![(1, 1)]);
        assert_eq!(output.result.items[0].feature.shape, vec![4, 3]);
        let TensorData::F32(values) = &output.result.items[0].feature.data else {
            panic!("wrong dtype");
        };
        assert_eq!(*values, (0..12).map(|n| n as f32).collect::<Vec<_>>());
        let mrope = output.result.mrope.unwrap();
        assert_eq!(mrope.positions, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert_eq!(mrope.delta, 0);
    }

    #[test]
    fn rejects_incompatible_profiles_names_and_tensor_descriptions() {
        let mut cases = Vec::new();
        let mut request = input();
        request.profile = "another_model_v1".into();
        cases.push(request);
        let mut request = input();
        request.processor_fingerprint = "stale-processor-spec".into();
        cases.push(request);
        let mut request = input();
        request.items[0]
            .spans
            .push(api::PreparedTokenSpan { start: 2, end: 3 });
        cases.push(request);
        let mut request = input();
        let duplicate = request.items[0].tensors[0].clone();
        request.items[0].tensors.push(duplicate);
        cases.push(request);
        let mut request = input();
        request.items[0].metadata[0].name = "crop_counts".into();
        cases.push(request);
        let mut request = input();
        request.items[0].tensors[0].tensor.as_mut().unwrap().dtype = "float16".into();
        cases.push(request);
        let mut request = input();
        request.items[0].tensors[0]
            .tensor
            .as_mut()
            .unwrap()
            .data
            .pop();
        cases.push(request);
        let mut request = input();
        request.items[0].metadata[0].values = vec![1, -2, 2];
        cases.push(request);
        let mut request = input();
        request.items[0].metadata[0].values = vec![1, i64::MAX, 2];
        cases.push(request);
        let mut request = input();
        request.items[0].spans[0].start = 2;
        cases.push(request);
        for request in cases {
            assert!(profile().check(&request).is_err());
        }
    }

    #[test]
    fn rejects_nonfinite_pixels_and_unclaimed_or_misaligned_pads() {
        for ids in [vec![99, 99, 20], vec![10, 20, 99], vec![10]] {
            assert!(profile().encode(ids, input()).is_err());
        }
        let mut request = input();
        request.items[0].tensors[0].tensor.as_mut().unwrap().data[..4]
            .copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(profile().encode(vec![10, 99, 20], request).is_err());
    }

    #[test]
    fn different_prepared_content_has_different_cache_identity() {
        let first = profile().encode(vec![10, 99, 20], input()).unwrap();
        let mut request = input();
        request.items[0].tensors[0].tensor.as_mut().unwrap().data[..4]
            .copy_from_slice(&100.0f32.to_le_bytes());
        let second = profile().encode(vec![10, 99, 20], request).unwrap();
        assert_ne!(
            first.result.items[0].identity,
            second.result.items[0].identity
        );
    }

    #[test]
    fn multiple_images_keep_their_own_geometry_and_spans() {
        let mut request = input();
        let mut second = request.items[0].clone();
        second.spans = vec![api::PreparedTokenSpan { start: 3, end: 7 }];
        second.metadata[0].values = vec![1, 4, 4];
        second.tensors[0].tensor = Some(api::InlineTensor {
            dtype: "float32".into(),
            shape: vec![16, 3],
            data: (0..48).flat_map(|n| (n as f32).to_le_bytes()).collect(),
        });
        request.items.push(second);
        let output = profile()
            .encode(vec![10, 99, 20, 99, 99, 99, 99, 30], request)
            .unwrap();
        let items = &output.result.items;
        assert_eq!(items[0].offsets, vec![(1, 1)]);
        assert_eq!(items[1].offsets, vec![(3, 6)]);
        assert_eq!(items[0].feature.shape, vec![4, 3]);
        assert_eq!(items[1].feature.shape, vec![16, 3]);
        assert_eq!(
            items[0].model_specific_data["image_grid_thw"],
            MmMetaValue::Ints(vec![1, 2, 2])
        );
        assert_eq!(
            items[1].model_specific_data["image_grid_thw"],
            MmMetaValue::Ints(vec![1, 4, 4])
        );
    }
}
