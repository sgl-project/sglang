//! Structural checks and the generic finish_prepared bridge: synthetic tensors
//! plus tiny fixture images, no model downloads or adapters.
#![cfg(feature = "mm")]

use std::path::{Path, PathBuf};

use serde_json::json;
use sglang_processor::ProcessorError;
use sglang_processor::dynamo_multimodal::processor::{
    Geometry, Modality, NamedTensors, PositionOutput, ProcessedItem, Tensor, TensorData,
};
use sglang_processor::dynamo_multimodal::registry::ProcessorSpec;
use sglang_processor::prepared::{
    PreparedInput, PreparedItem, PreparedLimits, PreparedTensor, TensorDtype, TokenSpan,
};
use sglang_processor::{MmOutput, MmProcessor};

const PAD_ID: i32 = 151655;
const PROFILE: &str = "test-profile";
const FINGERPRINT: &str = "test-fingerprint";

fn limits() -> PreparedLimits {
    PreparedLimits {
        max_tensor_bytes: 4096,
        max_items: 8,
        max_tensors_per_item: 8,
        max_rank: 4,
        max_spans_per_item: 8,
        max_input_tokens: 64,
    }
}

fn input<'a>(input_ids: &'a [i32], items: &'a [PreparedItem<'a>]) -> PreparedInput<'a> {
    PreparedInput {
        profile: PROFILE,
        processor_fingerprint: FINGERPRINT,
        input_ids,
        items,
    }
}

fn tensor<'a>(name: &'a str, shape: &'a [u64], data: &'a [u8]) -> PreparedTensor<'a> {
    PreparedTensor {
        name,
        dtype: TensorDtype::Float32,
        shape,
        data,
    }
}

fn item<'a>(tensors: &'a [PreparedTensor<'a>], spans: &'a [TokenSpan]) -> PreparedItem<'a> {
    PreparedItem {
        modality: Modality::Image,
        tensors,
        spans,
    }
}

fn rejected(result: Result<impl Sized, ProcessorError>, expected: &str) {
    match result {
        Err(ProcessorError::InvalidRequest(message)) => assert!(
            message.contains(expected),
            "expected {expected:?} in {message:?}"
        ),
        Err(other) => panic!("expected InvalidRequest, got {other:?}"),
        Ok(_) => panic!("expected rejection: {expected}"),
    }
}

#[test]
fn adjacent_items_keep_names_order_shapes_and_exact_little_endian_values() {
    let a: Vec<_> = [1.25f32, -2.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let b: Vec<_> = [1i64, -2, i64::MAX]
        .into_iter()
        .flat_map(i64::to_le_bytes)
        .collect();
    let a_tensors = [tensor("x", &[1, 2], &a)];
    let b_tensors = [PreparedTensor {
        dtype: TensorDtype::Int64,
        ..tensor("x", &[3], &b)
    }];
    let items = [
        item(&a_tensors, &[TokenSpan { start: 1, end: 3 }]),
        item(&b_tensors, &[TokenSpan { start: 3, end: 5 }]),
    ];
    let checked = input(&[7, 8, 8, 9, 9], &items)
        .check(PROFILE, FINGERPRINT, &limits())
        .unwrap();
    assert_eq!(checked.input().profile, PROFILE);
    assert_eq!(checked.input().processor_fingerprint, FINGERPRINT);
    assert_eq!(checked.input().items[0].spans, items[0].spans);
    assert_eq!(checked.total_tensor_bytes(), a.len() + b.len());
    let a = checked.decode_tensors(0).unwrap();
    let b = checked.decode_tensors(1).unwrap();
    assert_eq!(a[0].0, "x");
    assert_eq!(b[0].0, "x"); // Names are unique per item, not per request.
    assert_eq!(a[0].1.shape, vec![1, 2]);
    assert_eq!(b[0].1.shape, vec![3]);
    assert!(matches!(&a[0].1.data, TensorData::F32(v) if v == &[1.25, -2.5]));
    assert!(matches!(&b[0].1.data, TensorData::I64(v) if v == &[1, -2, i64::MAX]));
    rejected(checked.decode_tensors(2), "out of range");
}

#[test]
fn scalar_and_empty_tensor_shapes_decode_without_model_assumptions() {
    let scalar = 42.5f32.to_le_bytes();
    let tensors = [
        tensor("scalar", &[], &scalar),
        tensor("empty", &[0, 3], &[]),
        PreparedTensor {
            dtype: TensorDtype::Int64,
            ..tensor("empty_i64", &[2, 0], &[])
        },
        // A zero dimension makes the element count zero even after a prefix
        // whose nonzero product would overflow. Every dimension still fits usize.
        tensor("zero_after_large_prefix", &[usize::MAX as u64, 2, 0], &[]),
    ];
    let items = [item(&tensors, &[TokenSpan { start: 0, end: 1 }])];
    let checked = input(&[1], &items)
        .check(PROFILE, FINGERPRINT, &limits())
        .unwrap();
    let decoded = checked.decode_tensors(0).unwrap();
    assert!(decoded[0].1.shape.is_empty());
    assert!(matches!(&decoded[0].1.data, TensorData::F32(v) if v == &[42.5]));
    assert!(matches!(&decoded[1].1.data, TensorData::F32(v) if v.is_empty()));
    assert!(matches!(&decoded[2].1.data, TensorData::I64(v) if v.is_empty()));
    assert_eq!(decoded[3].1.shape, vec![usize::MAX, 2, 0]);
}

#[test]
fn checks_every_budget_before_tensor_conversion() {
    let bytes = 1f32.to_le_bytes();
    let tensors = [tensor("x", &[1], &bytes)];
    let items = [item(&tensors, &[TokenSpan { start: 0, end: 1 }])];
    let request = input(&[1, 2], &items);
    for (budget, error) in [
        (
            PreparedLimits {
                max_tensor_bytes: 3,
                ..limits()
            },
            "max_tensor_bytes",
        ),
        (
            PreparedLimits {
                max_items: 0,
                ..limits()
            },
            "max_items",
        ),
        (
            PreparedLimits {
                max_tensors_per_item: 0,
                ..limits()
            },
            "max_tensors_per_item",
        ),
        (
            PreparedLimits {
                max_rank: 0,
                ..limits()
            },
            "max_rank",
        ),
        (
            PreparedLimits {
                max_spans_per_item: 0,
                ..limits()
            },
            "max_spans_per_item",
        ),
        (
            PreparedLimits {
                max_input_tokens: 1,
                ..limits()
            },
            "max_input_tokens",
        ),
    ] {
        rejected(request.check(PROFILE, FINGERPRINT, &budget), error);
    }
    let second = [tensor("y", &[1], &bytes)];
    let two_items = [items[0], item(&second, &[TokenSpan { start: 1, end: 2 }])];
    rejected(
        input(&[1, 2], &two_items).check(
            PROFILE,
            FINGERPRINT,
            &PreparedLimits {
                max_tensor_bytes: 7,
                ..limits()
            },
        ),
        "max_tensor_bytes",
    );
}

#[test]
fn rejects_missing_headers_tokens_items_tensors_and_spans() {
    let bytes = 1f32.to_le_bytes();
    let tensors = [tensor("x", &[1], &bytes)];
    let spans = [TokenSpan { start: 0, end: 1 }];
    let items = [item(&tensors, &spans)];
    let request = input(&[1], &items);
    rejected(request.check("", FINGERPRINT, &limits()), "nonempty");
    rejected(request.check(PROFILE, "", &limits()), "nonempty");
    rejected(
        request.check("other", FINGERPRINT, &limits()),
        "profile mismatch",
    );
    rejected(
        request.check(PROFILE, "other", &limits()),
        "fingerprint mismatch",
    );
    rejected(
        input(&[], &items).check(PROFILE, FINGERPRINT, &limits()),
        "input_ids must be nonempty",
    );
    rejected(
        input(&[-1], &items).check(PROFILE, FINGERPRINT, &limits()),
        "negative",
    );
    rejected(
        input(&[1], &[]).check(PROFILE, FINGERPRINT, &limits()),
        "at least one item",
    );
    rejected(
        input(&[1], &[item(&[], &spans)]).check(PROFILE, FINGERPRINT, &limits()),
        "no tensors",
    );
    rejected(
        input(&[1], &[item(&tensors, &[])]).check(PROFILE, FINGERPRINT, &limits()),
        "no spans",
    );
}

#[test]
fn rejects_inconsistent_shapes_overflows_and_invalid_names() {
    let bytes = 1f32.to_le_bytes();
    let x = tensor("x", &[1], &bytes);
    let spans = [TokenSpan { start: 0, end: 1 }];
    let overflow = [usize::MAX as u64, 2];
    let byte_overflow = [usize::MAX as u64];
    for (tensors, error) in [
        (vec![tensor("x", &[2], &bytes)], "does not match"),
        (vec![tensor("x", &overflow, &[])], "element count overflows"),
        (
            vec![tensor("x", &byte_overflow, &[])],
            "byte length overflow",
        ),
        (vec![x, x], "duplicate tensor name"),
        (vec![tensor("", &[1], &bytes)], "empty name"),
    ] {
        rejected(
            input(&[1], &[item(&tensors, &spans)]).check(PROFILE, FINGERPRINT, &limits()),
            error,
        );
    }
    #[cfg(target_pointer_width = "32")]
    rejected(
        input(&[1], &[item(&[tensor("x", &[u64::MAX, 0], &[])], &spans)]).check(
            PROFILE,
            FINGERPRINT,
            &limits(),
        ),
        "does not fit in usize",
    );
}

#[test]
fn rejects_empty_reversed_out_of_bounds_unordered_and_overlapping_spans() {
    let bytes = 1f32.to_le_bytes();
    let tensors = [tensor("x", &[1], &bytes)];
    for (spans, error) in [
        (vec![TokenSpan { start: 1, end: 1 }], "empty or reversed"),
        (vec![TokenSpan { start: 2, end: 1 }], "empty or reversed"),
        (vec![TokenSpan { start: 0, end: 4 }], "exceeds input_ids"),
        (
            vec![
                TokenSpan { start: 2, end: 3 },
                TokenSpan { start: 0, end: 1 },
            ],
            "increasing start",
        ),
        (
            vec![
                TokenSpan { start: 0, end: 2 },
                TokenSpan { start: 1, end: 3 },
            ],
            "overlap",
        ),
    ] {
        rejected(
            input(&[1, 2, 3], &[item(&tensors, &spans)]).check(PROFILE, FINGERPRINT, &limits()),
            error,
        );
    }
    let overlap = [
        item(&tensors, &[TokenSpan { start: 0, end: 2 }]),
        item(&tensors, &[TokenSpan { start: 1, end: 3 }]),
    ];
    rejected(
        input(&[1, 2, 3], &overlap).check(PROFILE, FINGERPRINT, &limits()),
        "overlap",
    );
    let reversed_items = [
        item(&tensors, &[TokenSpan { start: 2, end: 3 }]),
        item(&tensors, &[TokenSpan { start: 0, end: 1 }]),
    ];
    rejected(
        input(&[1, 2, 3], &reversed_items).check(PROFILE, FINGERPRINT, &limits()),
        "first-span",
    );
}

#[test]
fn nonoverlapping_spans_can_interleave_across_items() {
    let bytes = 1f32.to_le_bytes();
    let tensors = [tensor("arbitrary_name", &[1], &bytes)];
    let items = [
        item(
            &tensors,
            &[
                TokenSpan { start: 0, end: 1 },
                TokenSpan { start: 3, end: 4 },
            ],
        ),
        PreparedItem {
            modality: Modality::Video,
            ..item(&tensors, &[TokenSpan { start: 1, end: 3 }])
        },
    ];
    let checked = input(&[10, 11, 12, 13], &items)
        .check(PROFILE, FINGERPRINT, &limits())
        .unwrap();
    assert_eq!(checked.input().items[0].spans.len(), 2);
    assert_eq!(checked.input().items[1].modality, Modality::Video);
    // Accepting unknown names/modality structurally is deliberately not a model
    // capability, semantic validation, or serving admission.
    assert_eq!(checked.decode_tensors(1).unwrap()[0].0, "arbitrary_name");
}

/// Actual Qwen processor knobs, used only by finish_prepared tests below.
fn qwen_spec() -> ProcessorSpec {
    serde_json::from_value(json!({
        "family": "qwen_vl",
        "image_token_id": 151655,
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "min_pixels": 3136,
        "max_pixels": 12845056,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "resample": "aten_u8"
    }))
    .unwrap()
}

fn mm_fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mm_parity/images")
}

fn clone_tensor(tensor: &Tensor) -> Tensor {
    Tensor {
        shape: tensor.shape.clone(),
        data: match &tensor.data {
            TensorData::F32(values) => TensorData::F32(values.clone()),
            TensorData::I64(values) => TensorData::I64(values.clone()),
            _ => panic!("unexpected tensor dtype in fixture"),
        },
    }
}

fn clone_aux(aux: &NamedTensors) -> NamedTensors {
    aux.iter()
        .map(|(name, tensor)| (name.clone(), clone_tensor(tensor)))
        .collect()
}

fn assert_tensor_eq(actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.shape, expected.shape);
    match (&actual.data, &expected.data) {
        (TensorData::F32(a), TensorData::F32(b)) => {
            assert!(
                a.iter()
                    .map(|v| v.to_bits())
                    .eq(b.iter().map(|v| v.to_bits()))
            );
        }
        (TensorData::I64(a), TensorData::I64(b)) => assert_eq!(a, b),
        _ => panic!("tensor dtype mismatch"),
    }
}

fn assert_aux_eq(actual: &NamedTensors, expected: &[(String, Tensor)]) {
    assert_eq!(actual.len(), expected.len());
    for ((a_name, a_tensor), (e_name, e_tensor)) in actual.iter().zip(expected) {
        assert_eq!(a_name, e_name);
        assert_tensor_eq(a_tensor, e_tensor);
    }
}

fn assert_positions_eq(actual: &PositionOutput, expected: &PositionOutput) {
    match (actual, expected) {
        (
            PositionOutput::MRope {
                positions: a,
                delta: da,
            },
            PositionOutput::MRope {
                positions: b,
                delta: db,
            },
        ) => {
            assert_eq!(a, b);
            assert_eq!(da, db);
        }
        (PositionOutput::Rope1D, PositionOutput::Rope1D) => {}
        _ => panic!("position scheme mismatch"),
    }
}

fn assert_output_payload_eq(actual: &MmOutput, expected: &MmOutput) {
    assert_eq!(actual.input_ids, expected.input_ids);
    assert_eq!(actual.offsets, expected.offsets);
    assert_eq!(actual.items.len(), expected.items.len());
    for (a, e) in actual.items.iter().zip(&expected.items) {
        assert_tensor_eq(&a.feature, &e.feature);
        assert_aux_eq(&a.aux, &e.aux);
    }
    assert_positions_eq(&actual.positions, &expected.positions);
}

fn grid_from_aux(aux: &NamedTensors) -> [u32; 3] {
    let (_, grid) = aux
        .iter()
        .find(|(name, _)| name == "image_grid_thw")
        .expect("missing image_grid_thw");
    let TensorData::I64(values) = &grid.data else {
        panic!("image_grid_thw must be i64");
    };
    assert_eq!(values.len(), 3);
    [values[0] as u32, values[1] as u32, values[2] as u32]
}

fn image_item(feature_token_count: usize, grid: [u32; 3], feature: Tensor) -> ProcessedItem {
    ProcessedItem {
        modality: Modality::Image,
        feature_token_count,
        feature,
        aux: vec![(
            "image_grid_thw".to_string(),
            Tensor {
                shape: vec![3],
                data: TensorData::I64(grid.iter().map(|&v| v as i64).collect()),
            },
        )],
        geometry: Some(Geometry::Grid(grid)),
    }
}

#[test]
fn finish_prepared_rejects_item_count_span_count_modality_and_multi_span() {
    let processor = MmProcessor::from_spec(qwen_spec()).unwrap();
    let bytes = 1f32.to_le_bytes();
    let tensors = [tensor("pixel_values", &[1], &bytes)];
    let feature = Tensor {
        shape: vec![1, 1],
        data: TensorData::F32(vec![0.5]),
    };

    let single = [item(&tensors, &[TokenSpan { start: 0, end: 1 }])];
    let checked = input(&[PAD_ID], &single)
        .check(PROFILE, FINGERPRINT, &limits())
        .unwrap();
    rejected(
        processor.finish_prepared(&checked, vec![]),
        "canonical item count",
    );
    rejected(
        processor.finish_prepared(
            &checked,
            vec![
                image_item(1, [1, 2, 2], clone_tensor(&feature)),
                image_item(1, [1, 2, 2], clone_tensor(&feature)),
            ],
        ),
        "canonical item count",
    );
    rejected(
        processor.finish_prepared(
            &checked,
            vec![ProcessedItem {
                modality: Modality::Video,
                ..image_item(1, [1, 2, 2], clone_tensor(&feature))
            }],
        ),
        "Image only",
    );
    let video_input = [PreparedItem {
        modality: Modality::Video,
        ..item(&tensors, &[TokenSpan { start: 0, end: 1 }])
    }];
    let video_checked = input(&[PAD_ID], &video_input)
        .check(PROFILE, FINGERPRINT, &limits())
        .unwrap();
    rejected(
        processor.finish_prepared(
            &video_checked,
            vec![image_item(1, [1, 2, 2], clone_tensor(&feature))],
        ),
        "does not match input",
    );
    rejected(
        processor.finish_prepared(
            &checked,
            vec![image_item(2, [1, 2, 2], clone_tensor(&feature))],
        ),
        "feature_token_count",
    );

    // Structural check permits multiple spans; finish must reject them.
    let multi_spans = [item(
        &tensors,
        &[
            TokenSpan { start: 0, end: 1 },
            TokenSpan { start: 2, end: 3 },
        ],
    )];
    let multi_checked = input(&[PAD_ID, 7, PAD_ID], &multi_spans)
        .check(PROFILE, FINGERPRINT, &limits())
        .unwrap();
    assert_eq!(multi_checked.input().items[0].spans.len(), 2);
    rejected(
        processor.finish_prepared(
            &multi_checked,
            vec![image_item(1, [1, 2, 2], clone_tensor(&feature))],
        ),
        "exactly one feature span",
    );
}

#[test]
fn finish_prepared_raw_round_trip_two_unequal_images_matches_except_hash() {
    let processor = MmProcessor::from_spec(qwen_spec()).unwrap();
    let tiny = std::fs::read(mm_fixtures().join("tiny_10x10.png")).unwrap();
    let rgba = std::fs::read(mm_fixtures().join("rgba_80x60.png")).unwrap();
    assert_ne!(tiny, rgba);
    let pad = PAD_ID;
    let placeholders = [151652, pad, 151653, 151652, pad, 151653, 74785, 13];
    let raw = processor
        .process(&placeholders, &[&tiny, &rgba])
        .expect("raw process");
    assert_eq!(raw.items.len(), 2);
    assert_ne!(
        raw.items[0].feature.shape, raw.items[1].feature.shape,
        "fixtures must stay unequal so dropped-item bugs are visible"
    );
    assert_ne!(raw.items[0].hash, 0);
    assert_ne!(raw.items[1].hash, 0);

    let payloads: Vec<Vec<u8>> = raw
        .items
        .iter()
        .map(|item| match &item.feature.data {
            TensorData::F32(values) => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            _ => panic!("fixture pixels must be float32"),
        })
        .collect();
    let shapes: Vec<Vec<u64>> = raw
        .items
        .iter()
        .map(|item| item.feature.shape.iter().map(|&dim| dim as u64).collect())
        .collect();
    let tensors: Vec<_> = payloads
        .iter()
        .zip(&shapes)
        .map(|(data, shape)| [tensor("pixel_values", shape, data)])
        .collect();
    let spans: Vec<_> = raw
        .offsets
        .iter()
        .map(|&(start, end)| {
            [TokenSpan {
                start,
                end: end + 1,
            }]
        })
        .collect();
    let prepared_items: Vec<_> = tensors
        .iter()
        .zip(&spans)
        .map(|(tensors, spans)| item(tensors, spans))
        .collect();
    let checked = PreparedInput {
        profile: PROFILE,
        processor_fingerprint: FINGERPRINT,
        input_ids: &raw.input_ids,
        items: &prepared_items,
    }
    .check(
        PROFILE,
        FINGERPRINT,
        &PreparedLimits {
            max_input_tokens: raw.input_ids.len(),
            max_tensor_bytes: payloads.iter().map(Vec::len).sum(),
            ..limits()
        },
    )
    .unwrap();

    // The bridge receives tensors decoded from the actual checked wire bytes,
    // not clones that bypass transport. Model geometry stays with the fixture.
    let canonical = raw
        .items
        .iter()
        .zip(&raw.offsets)
        .enumerate()
        .map(|(index, (item, &(start, end)))| {
            let (_, feature) = checked
                .decode_tensors(index)
                .unwrap()
                .into_iter()
                .next()
                .unwrap();
            ProcessedItem {
                modality: Modality::Image,
                feature_token_count: (end - start + 1) as usize,
                feature,
                aux: clone_aux(&item.aux),
                geometry: Some(Geometry::Grid(grid_from_aux(&item.aux))),
            }
        })
        .collect();
    let finished = processor
        .finish_prepared(&checked, canonical)
        .expect("finish_prepared");
    assert_output_payload_eq(&finished, &raw);
    assert_eq!(finished.items[0].hash, 0);
    assert_eq!(finished.items[1].hash, 0);
    assert_ne!(finished.items[0].hash, raw.items[0].hash);
    assert_ne!(finished.items[1].hash, raw.items[1].hash);
}
