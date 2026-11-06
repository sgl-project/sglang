//! Every `tests/fixtures/mm_parity/*.json` preprocesses as SGLang's Python CPU
//! path does, bit for bit. Fixtures come from
//! `tests/scripts/generate_mm_parity.py` (images) and
//! `tests/scripts/generate_mm_video_parity.py` (video, every stage).

#![cfg(feature = "mm")]

use std::path::{Path, PathBuf};

use std::collections::HashMap;

use serde_json::{Value, json};
use sglang_processor::dynamo_multimodal::image::decode::{DecodeLimits, decode_rgb};
use sglang_processor::dynamo_multimodal::image::resize::Resample;
use sglang_processor::dynamo_multimodal::processor::Modality;
use sglang_processor::dynamo_multimodal::processor::{
    DecodedMedia, PositionOutput, TensorData, TextEncoder,
};
use sglang_processor::dynamo_multimodal::registry::ProcessorSpec;
use sglang_processor::dynamo_multimodal::video_decode::{
    FrameResize, RgbConversion, Video, VideoDecodeLimits,
};
use sglang_processor::dynamo_multimodal::{MmError, content_hash_bytes};
use sglang_processor::{
    MmDecoded, MmOutput, MmProcessor, MmVideo, VideoHints, VideoSampling, frame_indices,
};
use sha2::{Digest, Sha256};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mm_parity")
}

#[test]
fn fixtures_match_sglang() {
    let mut failures = Vec::new();
    for entry in std::fs::read_dir(fixtures()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "json") {
            let fixture = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            failures.extend(check(&fixture));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

fn check(fixture: &Value) -> Vec<String> {
    let model = fixture["model"].as_str().unwrap();
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let name = format!("{model} {}", case["name"]);
        // Same contract as tests/parity.rs: a known gap must keep failing.
        let result = check_case(case);
        if let Err(error) = &result
            && error.starts_with(NO_DECODER)
            && case["nvdec"] == true
        {
            eprintln!("{name}: {error}");
            continue;
        }
        match (result, case["known_gap"].as_str()) {
            (Ok(()), None) => {}
            (Err(error), None) => failures.push(format!("{name}: {error}")),
            (Err(_), Some(gap)) => eprintln!("{name}: known gap: {gap}"),
            (Ok(()), Some(_)) => {
                failures.push(format!("{name} now matches SGLang; drop its known_gap"))
            }
        }
    }
    failures
}

fn check_case(case: &Value) -> Result<(), String> {
    if case.get("videos").is_some() {
        return check_video_case(case);
    }
    let spec: ProcessorSpec = serde_json::from_value(case["spec"].clone()).unwrap();
    let images = fixture_bytes("images", &case["images"]);
    let images: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
    let input_ids: Vec<i32> = (case["input_ids"].as_array().unwrap().iter())
        .map(|id| id.as_i64().unwrap() as i32)
        .collect();

    let output = MmProcessor::from_spec(spec, None)
        .and_then(|processor| processor.process(&input_ids, &images));
    match (output, case.get("error")) {
        (Ok(output), None) => compare(&summary(&output), &case["expected"]),
        (Err(_), Some(_)) => Ok(()),
        (Ok(_), Some(error)) => Err(format!("SGLang rejects this case: {error}")),
        (Err(error), None) => Err(error.to_string()),
    }
}

/// H.264, HEVC and AV1 decode only on NVDEC, which a CPU-only host lacks.
const NO_DECODER: &str = "not run, no decoder here";

/// The generator's timestamp texts and their ids under the real tokenizer.
struct Timestamps(HashMap<String, Vec<i32>>);

impl TextEncoder for Timestamps {
    fn encode(&self, text: &str) -> sglang_processor::dynamo_multimodal::Result<Vec<i32>> {
        (self.0.get(text).cloned())
            .ok_or_else(|| MmError::invalid_input(format!("no recorded ids for {text:?}")))
    }
}

fn fixture_bytes(dir: &str, names: &Value) -> Vec<Vec<u8>> {
    (names.as_array().unwrap().iter())
        .map(|name| std::fs::read(fixtures().join(dir).join(name.as_str().unwrap())).unwrap())
        .collect()
}

/// Every stage of one case, in order: the frames each video samples, its
/// decoded and pre-resized frames, then the processor's output.
fn check_video_case(case: &Value) -> Result<(), String> {
    let spec: ProcessorSpec = serde_json::from_value(case["spec"].clone()).unwrap();
    let sampling: Option<VideoSampling> =
        serde_json::from_value(case["spec"]["video_sampling"].clone()).unwrap();
    let videos = fixture_bytes("videos", &case["videos"]);
    let images = fixture_bytes("images", &case["images"]);
    let input_ids: Vec<i32> = (case["input_ids"].as_array().unwrap().iter())
        .map(|id| id.as_i64().unwrap() as i32)
        .collect();
    let hints: VideoHints = serde_json::from_value(case["video_hints"].clone()).unwrap();
    let media: Vec<MmVideo> = videos
        .iter()
        .map(|bytes| MmVideo { bytes, hints })
        .collect();
    let images: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
    let processor = MmProcessor::from_spec(spec, sampling.clone()).map_err(|e| e.to_string())?;
    let Some(expected) = case.get("expected") else {
        let empty = Timestamps(HashMap::new());
        return match processor.process_media(&input_ids, &images, &media, &empty) {
            Ok(_) => Err(format!("SGLang rejects this case: {}", case["error"])),
            Err(_) => Ok(()),
        };
    };
    let sampling = sampling.expect("a video case's spec samples video");
    let mut decoded: Vec<MmDecoded> =
        (images.iter().map(|bytes| decode_image(bytes))).collect::<Result<_, String>>()?;
    for (bytes, stages) in videos
        .iter()
        .zip(expected["video_stages"].as_array().unwrap())
    {
        decoded.push(check_video_stages(bytes, &sampling, &hints, stages)?);
    }
    let text = Timestamps(serde_json::from_value(expected["timestamp_ids"].clone()).unwrap());
    let output =
        (processor.process_media(&input_ids, &images, &media, &text)).map_err(|e| e.to_string())?;
    let actual = case_summary(&output);
    let from_decoded = (processor.process_decoded(&input_ids, decoded, &text))
        .map_err(|e| format!("process_decoded: {e}"))?;
    if case_summary(&from_decoded) != actual {
        return Err("process_decoded differs from process_media on the same frames".into());
    }
    let fields = [
        "input_ids_sha256",
        "num_input_ids",
        "modalities",
        "offsets",
        "grids",
        "feature_shapes",
        "feature_sha256",
        "mrope_sha256",
        "mrope_delta",
    ];
    let expected: serde_json::Map<String, Value> = fields
        .iter()
        .map(|&f| (f.to_owned(), expected[f].clone()))
        .collect();
    compare(&actual, &Value::Object(expected))
}

/// The stages Python records for one video, checked in order; returns the
/// pre-resized frames as `process_media` hands them to the model's family.
fn check_video_stages(
    bytes: &[u8],
    sampling: &VideoSampling,
    hints: &VideoHints,
    expected: &Value,
) -> Result<MmDecoded, String> {
    let limits = VideoDecodeLimits::default();
    let config = sampling
        .config
        .with_hints(hints)
        .map_err(|e| e.to_string())?;
    let video = Video::open(&bytes, &limits).map_err(|e| e.to_string())?;
    let info = video.info();
    let num_frames = (sampling.num_frames(&config, info.frame_count, info.average_fps))
        .map_err(|e| e.to_string())?;
    let indices = frame_indices(info.frame_count, num_frames);
    let decoded =
        (video.decode_with(&indices, None, &limits, RgbConversion::Swscale)).map_err(|error| {
            match error {
                MmError::Unsupported { .. } => format!("{NO_DECODER}: {error}"),
                _ => error.to_string(),
            }
        })?;
    let (height, width) = sampling
        .presize(
            &config,
            indices.len(),
            info.height as usize,
            info.width as usize,
        )
        .map_err(|e| e.to_string())?;
    let presize = FrameResize {
        width: width as u32,
        height: height as u32,
        filter: Resample::AtenF32Bilinear,
    };
    let presized = (video.decode_with(&indices, Some(&presize), &limits, RgbConversion::Swscale))
        .map_err(|e| e.to_string())?;
    let frames = |frames: &sglang_processor::dynamo_multimodal::frames::VideoFrames| {
        json!([frames.frame_count(), frames.height(), frames.width(), 3])
    };
    let actual = json!({
        "total_frames": info.frame_count,
        "fps": info.average_fps,
        "num_frames": num_frames,
        "indices": indices,
        "decoded_shape": frames(&decoded),
        "decoded_sha256": sha256(decoded.rgb().iter().copied()),
        "presized_shape": frames(&presized),
        "presized_sha256": sha256(presized.rgb().iter().copied()),
    });
    // JSON keeps Python's int-valued fps as an int; compare it as a number.
    let mut expected = expected.clone();
    expected["fps"] = json!(expected["fps"].as_f64());
    compare(&actual, &expected).map_err(|error| format!("video stages: {error}"))?;
    let (height, width) = (presized.height(), presized.width());
    Ok(MmDecoded {
        media: DecodedMedia::Video {
            rgb: presized.into_rgb(),
            height,
            width,
            frame_indices: indices,
            fps: info.average_fps,
        },
        hash: content_hash_bytes(bytes),
    })
}

/// An image as `process_media` hands it to the model's family.
fn decode_image(bytes: &[u8]) -> Result<MmDecoded, String> {
    let (rgb, height, width) =
        decode_rgb(bytes, &DecodeLimits::default()).map_err(|e| e.to_string())?;
    Ok(MmDecoded {
        media: DecodedMedia::Image { rgb, height, width },
        hash: content_hash_bytes(bytes),
    })
}

fn video_case(name: &str) -> Value {
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(fixtures().join("qwen3-vl-2b-instruct.video.json")).unwrap(),
    )
    .unwrap();
    (fixture["cases"].as_array().unwrap().iter())
        .find(|case| case["name"] == name)
        .unwrap()
        .clone()
}

/// A video placeholder before an image one: items come back images first, then
/// videos, as Python's `mm_items`, each with its own prompt span.
#[test]
fn items_follow_python_order_when_a_video_comes_first() {
    let case = video_case("image then video");
    let spec: ProcessorSpec = serde_json::from_value(case["spec"].clone()).unwrap();
    let sampling = serde_json::from_value(case["spec"]["video_sampling"].clone()).unwrap();
    let processor = MmProcessor::from_spec(spec, sampling).unwrap();
    let image = fixture_bytes("images", &case["images"]).remove(0);
    let video = fixture_bytes("videos", &case["videos"]).remove(0);
    // `image then video`'s prompt is `[image block, video block, text..]`.
    let ids: Vec<i32> = (case["input_ids"].as_array().unwrap().iter())
        .map(|id| id.as_i64().unwrap() as i32)
        .collect();
    let swapped: Vec<i32> = [&ids[3..6], &ids[0..3], &ids[6..]].concat();
    let videos = [MmVideo {
        bytes: &video,
        hints: VideoHints::default(),
    }];
    let text =
        Timestamps(serde_json::from_value(case["expected"]["timestamp_ids"].clone()).unwrap());
    let output = processor
        .process_media(&swapped, &[&image], &videos, &text)
        .unwrap();
    let modalities: Vec<Modality> = output.items.iter().map(|item| item.modality).collect();
    assert_eq!(modalities, [Modality::Image, Modality::Video]);
    let (image, video) = (&output.items[0].offsets, &output.items[1].offsets);
    assert!(
        video.last().unwrap().1 < image[0].0,
        "{video:?} then {image:?}"
    );
}

/// Decoded items come back in the caller's order, each in its own modality's
/// placeholder, with the same spans, features and hashes as `process_media`.
#[test]
fn decoded_items_keep_the_callers_order() {
    let case = video_case("image then video");
    let spec: ProcessorSpec = serde_json::from_value(case["spec"].clone()).unwrap();
    let sampling: VideoSampling =
        serde_json::from_value(case["spec"]["video_sampling"].clone()).unwrap();
    let processor = MmProcessor::from_spec(spec, Some(sampling.clone())).unwrap();
    let image = fixture_bytes("images", &case["images"]).remove(0);
    let video = fixture_bytes("videos", &case["videos"]).remove(0);
    let ids: Vec<i32> = (case["input_ids"].as_array().unwrap().iter())
        .map(|id| id.as_i64().unwrap() as i32)
        .collect();
    let text =
        Timestamps(serde_json::from_value(case["expected"]["timestamp_ids"].clone()).unwrap());
    let videos = [MmVideo {
        bytes: &video,
        hints: VideoHints::default(),
    }];
    let media = processor
        .process_media(&ids, &[&image], &videos, &text)
        .unwrap();
    let stages = &case["expected"]["video_stages"][0];
    let decoded = vec![
        check_video_stages(&video, &sampling, &VideoHints::default(), stages).unwrap(),
        decode_image(&image).unwrap(),
    ];
    let output = processor.process_decoded(&ids, decoded, &text).unwrap();
    let modalities: Vec<Modality> = output.items.iter().map(|item| item.modality).collect();
    assert_eq!(modalities, [Modality::Video, Modality::Image]);
    assert_eq!(output.input_ids, media.input_ids);
    for (decoded, encoded) in output.items.iter().zip([&media.items[1], &media.items[0]]) {
        assert_eq!(decoded.offsets, encoded.offsets);
        assert_eq!(decoded.hash, encoded.hash);
        assert_eq!(decoded.feature.shape, encoded.feature.shape);
        let (TensorData::F32(decoded), TensorData::F32(encoded)) =
            (&decoded.feature.data, &encoded.feature.data)
        else {
            panic!("Qwen-VL features are f32");
        };
        assert_eq!(decoded, encoded);
    }
}

#[test]
fn decoded_items_must_fill_the_placeholders() {
    let case = video_case("image then video");
    let spec: ProcessorSpec = serde_json::from_value(case["spec"].clone()).unwrap();
    let processor = MmProcessor::from_spec(spec, None).unwrap();
    let image = fixture_bytes("images", &case["images"]).remove(0);
    let ids: Vec<i32> = (case["input_ids"].as_array().unwrap().iter())
        .map(|id| id.as_i64().unwrap() as i32)
        .collect();
    let video = MmDecoded {
        media: DecodedMedia::Video {
            rgb: vec![0; 2 * 32 * 32 * 3],
            height: 32,
            width: 32,
            frame_indices: vec![0, 1],
            fps: 1.0,
        },
        hash: 0,
    };
    let items = vec![
        video,
        decode_image(&image).unwrap(),
        decode_image(&image).unwrap(),
    ];
    let empty = Timestamps(HashMap::new());
    let error = match processor.process_decoded(&ids, items, &empty) {
        Ok(_) => panic!("three items for two placeholders"),
        Err(error) => error.to_string(),
    };
    assert_eq!(
        error,
        "prompt has 1 image, 1 video placeholder(s) but the request has 1 video, 2 image item(s)"
    );
}

/// Every field a video case compares, from one output.
fn case_summary(output: &MmOutput) -> Value {
    let mut actual = summary(output);
    actual["offsets"] = json!(
        output
            .items
            .iter()
            .map(|item| &item.offsets)
            .collect::<Vec<_>>()
    );
    actual["grids"] = actual["image_grid_thw"].take();
    actual["modalities"] = json!(
        (output.items.iter())
            .map(|item| format!("{:?}", item.modality).to_lowercase())
            .collect::<Vec<_>>()
    );
    actual["hashes"] = json!(
        output
            .items
            .iter()
            .map(|item| item.hash)
            .collect::<Vec<_>>()
    );
    actual
}

/// The fields `generate_mm_parity.py` records, computed from the Rust output.
fn summary(output: &MmOutput) -> Value {
    let (mrope, delta) = match &output.positions {
        PositionOutput::MRope { positions, delta } => (positions.clone(), *delta),
        _ => panic!("Qwen-VL uses M-RoPE"),
    };
    let grids: Vec<Value> = (output.items.iter())
        .map(|item| {
            let (_, grid) = (item.aux.iter())
                .find(|(name, _)| name.ends_with("_grid_thw"))
                .expect("missing a grid");
            let TensorData::I64(grid) = &grid.data else {
                panic!("image_grid_thw is i64");
            };
            json!(grid)
        })
        .collect();
    let features = output.items.iter().map(|item| match &item.feature.data {
        TensorData::F32(values) => values,
        _ => panic!("Qwen-VL features are f32"),
    });
    json!({
        "input_ids_sha256": sha256(output.input_ids.iter().flat_map(|id| id.to_le_bytes())),
        "num_input_ids": output.input_ids.len(),
        "offsets": output.items.iter().map(|item| item.offsets[0]).collect::<Vec<_>>(),
        "image_grid_thw": grids,
        "feature_shapes": output.items.iter().map(|item| &item.feature.shape).collect::<Vec<_>>(),
        "feature_sha256": features
            .map(|values| sha256(values.iter().flat_map(|v| v.to_le_bytes())))
            .collect::<Vec<_>>(),
        "mrope_sha256": sha256(mrope.iter().flat_map(|p| p.to_le_bytes())),
        "mrope_delta": delta,
    })
}

fn compare(actual: &Value, expected: &Value) -> Result<(), String> {
    let mismatches: Vec<String> = (expected.as_object().unwrap().iter())
        .filter(|(field, value)| actual[field.as_str()] != **value)
        .map(|(field, value)| format!("{field}: rust {} != python {value}", actual[field.as_str()]))
        .collect();
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(mismatches.join("; "))
    }
}

fn sha256(bytes: impl Iterator<Item = u8>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes.collect::<Vec<u8>>());
    format!("{:x}", hasher.finalize())
}
