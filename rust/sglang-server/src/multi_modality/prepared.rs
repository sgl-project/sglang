//! Public, model-profiled CPU inputs. No fetching or preprocessing occurs here;
//! the result joins the same encoded-item path as the native image processor.

use sha2::{Digest, Sha256};

use sglang_api_types::api::v1 as api;
use sglang_processor::dynamo_multimodal::models::qwen_vl::QwenVlSpec;
use sglang_processor::dynamo_multimodal::processor::{
    Geometry, Modality, ProcessedItem, Tensor as NativeTensor, TensorData as NativeTensorData,
};
use sglang_processor::dynamo_multimodal::registry::ProcessorSpec;
use sglang_processor::prepared::{
    PreparedInput, PreparedItem, PreparedLimits, PreparedTensor, TensorDtype, TokenSpan,
};

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
        if self.spec.image_token_id < 0 {
            return None;
        }
        Some(api::PreparedInputCapability {
            profile: QWEN_VL_IMAGE_V1.into(),
            max_inline_bytes: MAX_INLINE_BYTES,
            processor_fingerprint: self.fingerprint.clone(),
            transports: vec!["inline".into()],
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
        pipeline: &sglang_processor::MmProcessor,
        input_ids: TokenIds,
        input: api::PreparedMultimodalInput,
    ) -> Result<sglang_processor::MmOutput, sglang_processor::ProcessorError> {
        self.check(&input)?;
        let ids = input_ids
            .into_iter()
            .map(|id| {
                i32::try_from(id).map_err(|_| format!("prepared token id {id} is outside int32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Borrow the public envelope into the common structural contract. No
        // protobuf type or model registry is added to sglang-processor.
        let tensors = input
            .items
            .iter()
            .map(|item| {
                item.tensors
                    .iter()
                    .map(|named| {
                        let tensor = named.tensor.as_ref().ok_or("missing prepared tensor")?;
                        let dtype = match tensor.dtype.as_str() {
                            "float32" => TensorDtype::Float32,
                            "int64" => TensorDtype::Int64,
                            _ => return Err("unsupported prepared tensor dtype"),
                        };
                        Ok(PreparedTensor {
                            name: &named.name,
                            dtype,
                            shape: &tensor.shape,
                            data: &tensor.data,
                        })
                    })
                    .collect::<Result<Vec<_>, &str>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let spans = input
            .items
            .iter()
            .map(|item| {
                item.spans
                    .iter()
                    .map(|span| TokenSpan {
                        start: span.start,
                        end: span.end,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let items = input
            .items
            .iter()
            .zip(&tensors)
            .zip(&spans)
            .map(|((item, tensors), spans)| {
                let modality = match item.modality.as_str() {
                    "image" => Modality::Image,
                    "video" => Modality::Video,
                    "audio" => Modality::Audio,
                    _ => return Err("unsupported prepared modality"),
                };
                Ok(PreparedItem {
                    modality,
                    tensors,
                    spans,
                })
            })
            .collect::<Result<Vec<_>, &str>>()?;
        let checked = PreparedInput {
            profile: &input.profile,
            processor_fingerprint: &input.processor_fingerprint,
            input_ids: &ids,
            items: &items,
        }
        .check(
            QWEN_VL_IMAGE_V1,
            &self.fingerprint,
            &PreparedLimits {
                max_tensor_bytes: MAX_INLINE_BYTES as usize,
                // Each admitted image occupies at least one expanded token.
                max_items: ids.len(),
                max_tensors_per_item: 1,
                max_rank: 2,
                max_spans_per_item: 1,
                max_input_tokens: ids.len(),
            },
        )?;

        // The active profile still validates required names, grid geometry,
        // finite pixels, and complete pad coverage. Structural acceptance is
        // not semantic admission. Reuse canonical community carriers and the
        // installed processor's positions instead of assembling scheduler data.
        let image_token_id = self.spec.image_token_id;
        let mut cursor = 0usize;
        let mut canonical = Vec::with_capacity(input.items.len());
        for (index, image) in input.items.iter().enumerate() {
            let (start, end) = qwen_span(image)?;
            let (start, end) = (start as usize, end as usize);
            if ids[cursor..start].contains(&image_token_id)
                || ids[start..end].iter().any(|&id| id != image_token_id)
            {
                return Err("prepared image spans do not match expanded input_ids".into());
            }
            let grid = qwen_item(image)?.1;
            let mut decoded = checked.decode_tensors(index)?.into_iter();
            let (_, feature) = decoded.next().ok_or("missing prepared pixel_values")?;
            match &feature.data {
                NativeTensorData::F32(values) if values.iter().all(|value| value.is_finite()) => {}
                _ => return Err("prepared pixel_values must be finite float32".into()),
            }
            canonical.push(ProcessedItem {
                modality: Modality::Image,
                feature_token_count: end - start,
                feature,
                aux: vec![(
                    "image_grid_thw".into(),
                    NativeTensor {
                        shape: vec![3],
                        data: NativeTensorData::I64(grid.into_iter().map(i64::from).collect()),
                    },
                )],
                geometry: Some(Geometry::Grid(grid)),
            });
            cursor = end;
        }
        if ids[cursor..].contains(&image_token_id) {
            return Err("image-pad tokens outside prepared image spans".into());
        }
        pipeline.finish_prepared(&checked, canonical)
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
    use super::super::encoded::MmMetaValue;
    use super::super::worker::{MmProcessOutput, encode_prepared_output};
    use super::*;
    use sglang_mm::pipeline::TensorData;
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

    fn encode(
        input_ids: TokenIds,
        input: api::PreparedMultimodalInput,
    ) -> Result<MmProcessOutput, String> {
        let profile = profile();
        let pipeline =
            sglang_processor::MmProcessor::from_spec(ProcessorSpec::QwenVl(profile.spec.clone()))
                .map_err(|error| error.to_string())?;
        let output = profile
            .encode(&pipeline, input_ids, input)
            .map_err(|error| error.to_string())?;
        encode_prepared_output(output)
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
        let output = encode(vec![10, 99, 20], input()).unwrap();
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
        for ids in [
            vec![99, 99, 20],
            vec![10, 20, 99],
            vec![10],
            vec![-1, 99, 20],
            vec![i64::MAX, 99, 20],
        ] {
            assert!(encode(ids, input()).is_err());
        }
        let mut request = input();
        request.items[0].tensors[0].tensor.as_mut().unwrap().data[..4]
            .copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(encode(vec![10, 99, 20], request).is_err());
    }

    #[test]
    fn different_prepared_content_has_different_cache_identity() {
        let first = encode(vec![10, 99, 20], input()).unwrap();
        let mut request = input();
        request.items[0].tensors[0].tensor.as_mut().unwrap().data[..4]
            .copy_from_slice(&100.0f32.to_le_bytes());
        let second = encode(vec![10, 99, 20], request).unwrap();
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
        let output = encode(vec![10, 99, 20, 99, 99, 99, 99, 30], request).unwrap();
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

    #[test]
    fn raw_and_wire_prepared_unequal_images_share_exact_scheduler_inputs() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../sglang-processor/tests/fixtures/mm_parity/qwen2.5-vl-3b-instruct.json"
        ))
        .unwrap();
        let spec: ProcessorSpec =
            serde_json::from_value(fixture["cases"][0]["spec"].clone()).unwrap();
        let profile = PreparedProfile::from_spec(&spec).unwrap();
        let pipeline = sglang_processor::MmProcessor::from_spec(spec).unwrap();
        let first = include_bytes!(
            "../../../sglang-processor/tests/fixtures/mm_parity/images/tiny_10x10.png"
        );
        let second = include_bytes!(
            "../../../sglang-processor/tests/fixtures/mm_parity/images/rgba_80x60.png"
        );
        let pad = profile.spec.image_token_id;
        let raw = pipeline
            .process(&[7, pad, 8, pad, 9], &[first.as_slice(), second.as_slice()])
            .unwrap();
        assert_ne!(raw.items[0].feature.shape, raw.items[1].feature.shape);
        let wire_items = raw
            .items
            .iter()
            .zip(&raw.offsets)
            .map(|(item, &(start, end))| {
                let NativeTensorData::F32(values) = &item.feature.data else {
                    panic!("expected pixel values");
                };
                let metadata = item
                    .aux
                    .iter()
                    .map(|(name, tensor)| {
                        let NativeTensorData::I64(values) = &tensor.data else {
                            panic!("expected integer metadata");
                        };
                        api::PreparedMetadata {
                            name: name.clone(),
                            values: values.clone(),
                        }
                    })
                    .collect();
                api::PreparedMediaItem {
                    modality: "image".into(),
                    tensors: vec![api::PreparedTensor {
                        name: "pixel_values".into(),
                        tensor: Some(api::InlineTensor {
                            dtype: "float32".into(),
                            shape: item.feature.shape.iter().map(|&d| d as u64).collect(),
                            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                        }),
                    }],
                    metadata,
                    spans: vec![api::PreparedTokenSpan {
                        start,
                        end: end + 1,
                    }],
                }
            })
            .collect();
        let ids = raw.input_ids.iter().copied().map(i64::from).collect();
        let imported = profile
            .encode(
                &pipeline,
                ids,
                api::PreparedMultimodalInput {
                    profile: QWEN_VL_IMAGE_V1.into(),
                    processor_fingerprint: profile.fingerprint.clone(),
                    items: wire_items,
                },
            )
            .unwrap();
        let actual = encode_prepared_output(imported).unwrap();
        let expected = super::super::worker::encode_output(raw).unwrap();
        assert_eq!(actual.input_ids, expected.input_ids);
        assert_eq!(actual.result.items.len(), expected.result.items.len());
        for (a, b) in actual.result.items.iter().zip(&expected.result.items) {
            assert_eq!(a.offsets, b.offsets);
            assert_eq!(a.model_specific_data, b.model_specific_data);
            assert_eq!(a.feature.shape, b.feature.shape);
            let (TensorData::F32(a_values), TensorData::F32(b_values)) =
                (&a.feature.data, &b.feature.data)
            else {
                panic!("expected f32 pixels");
            };
            assert!(
                a_values
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(b_values.iter().map(|v| v.to_bits()))
            );
            assert_eq!(a.cache_identity().unwrap(), b.cache_identity().unwrap());
        }
        let a = actual.result.mrope.unwrap();
        let b = expected.result.mrope.unwrap();
        assert_eq!(a.positions, b.positions);
        assert_eq!(a.delta, b.delta);
    }
}
