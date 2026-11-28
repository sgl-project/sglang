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
use sglang_processor::dynamo_multimodal::models::qwen_vl::Resampler;
use sglang_processor::dynamo_multimodal::processor::Modality;
use sglang_processor::dynamo_multimodal::processor::{
    DecodedMedia, Geometry, PositionOutput, TensorData, TextEncoder,
};
use sglang_processor::dynamo_multimodal::registry::ProcessorSpec;
use sglang_processor::dynamo_multimodal::video::sample::glm_frame_indices;
use sglang_processor::dynamo_multimodal::video_decode::{
    FrameResize, RgbConversion, Video, VideoDecodeLimits, VideoInfo,
};
use sglang_processor::dynamo_multimodal::{MmError, content_hash_bytes};
use sglang_processor::{
    MmDecoded, MmDirFamily, MmOutput, MmProcessor, MmVideo, VideoHints, VideoPolicy, VideoSampling,
    frame_indices, sample_video_frames, spec_from_model_dir,
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
    let filter = recorded_presize_filter(fixture);
    // A fixture with checkpoint files resolves its spec from them, as the server does.
    let checkpoint = fixture.get("checkpoint").map(|_| checkpoint_spec(fixture));
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let name = format!("{model} {}", case["name"]);
        // Same contract as tests/parity.rs: a known gap must keep failing.
        let result = match &checkpoint {
            Some(resolved) => (resolved.clone())
                .and_then(|(spec, video)| check_checkpoint_case(case, spec, video)),
            None => check_case(case, filter),
        };
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

fn recorded_presize_filter(fixture: &Value) -> Option<Resample> {
    let reference = &fixture["reference"];
    let capability = reference["torch_cpu_capability"].as_str()?;
    assert_eq!(
        reference["machine"], "x86_64",
        "presize kernels mirror x86 torch only"
    );
    Some(match capability {
        "DEFAULT" => Resample::AtenF32Bilinear,
        "AVX2" | "AVX512" => Resample::AtenF32BilinearAvx2,
        other => panic!("no presize kernel mirrors torch's {other} dispatch"),
    })
}

fn check_case(case: &Value, filter: Option<Resample>) -> Result<(), String> {
    if case.get("videos").is_some() {
        return check_video_case(case, filter.expect("a video fixture records its dispatch"));
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

fn checkpoint_spec(fixture: &Value) -> Result<(ProcessorSpec, Option<VideoPolicy>), String> {
    let server = &fixture["server"];
    let family = match fixture["family"].as_str() {
        Some("glm5_next") => MmDirFamily::Glm5Next,
        Some("nemotron_vl") => MmDirFamily::NemotronVl {
            max_model_len: server["max_model_len"].as_u64().unwrap() as u32,
            video_target_num_patches: server["video_target_num_patches"].as_u64().unwrap() as u32,
            video_maintain_aspect_ratio: server["video_maintain_aspect_ratio"].as_bool().unwrap(),
            video_pruning_rate: server["video_pruning_rate"].as_f64().unwrap(),
        },
        other => panic!("no model-dir family {other:?}"),
    };
    let dir = fixtures().join(fixture["checkpoint"].as_str().unwrap());
    // The generator records the class SGLang loads by default, `Glm5NextImageProcessor`.
    spec_from_model_dir(family, &dir, Resampler::AtenU8).map_err(|e| e.to_string())
}

/// A case of a model-dir fixture, stage by stage: the frames each video
/// samples and decodes, then the processor's output.
fn check_checkpoint_case(
    case: &Value,
    spec: ProcessorSpec,
    video: Option<VideoPolicy>,
) -> Result<(), String> {
    let images = fixture_bytes("images", &case["images"]);
    let images: Vec<&[u8]> = images.iter().map(Vec::as_slice).collect();
    let videos = fixture_bytes("videos", &case["videos"]);
    let media: Vec<MmVideo> = (videos.iter())
        .map(|bytes| MmVideo {
            bytes,
            hints: VideoHints::default(),
        })
        .collect();
    let input_ids: Vec<i32> = (case["input_ids"].as_array().unwrap().iter())
        .map(|id| id.as_i64().unwrap() as i32)
        .collect();
    let processor = MmProcessor::from_spec(spec, video.clone()).map_err(|e| e.to_string())?;
    let Some(expected) = case.get("expected") else {
        let empty = Timestamps(HashMap::new());
        return match processor.process_media(&input_ids, &images, &media, &empty) {
            Ok(_) => Err(format!("SGLang rejects this case: {}", case["error"])),
            Err(_) => Ok(()),
        };
    };
    for (bytes, stages) in videos
        .iter()
        .zip(expected["video_stages"].as_array().unwrap())
    {
        let sample = |info: &VideoInfo| match &video {
            Some(VideoPolicy::Glm { fps, max_frames }) => {
                glm_frame_indices(info.frame_count, info.average_fps, *fps, *max_frames)
            }
            Some(VideoPolicy::Nemotron) => {
                sample_video_frames(info.frame_count, info.average_fps, 2.0, 128)
            }
            _ => Err(MmError::unsupported(
                "the checkpoint resolved without video",
            )),
        };
        check_source_video_stages(bytes, sample, stages)?;
    }
    let labels = expected.get("timestamp_ids").or(expected.get("label_ids"));
    let text = Timestamps(serde_json::from_value(labels.unwrap().clone()).unwrap());
    let output =
        (processor.process_media(&input_ids, &images, &media, &text)).map_err(|e| e.to_string())?;
    let (actual, fields) = match video {
        Some(VideoPolicy::Nemotron) => (nemotron_summary(&output), NEMOTRON_FIELDS),
        _ => (case_summary(&output), GLM_FIELDS),
    };
    let expected: serde_json::Map<String, Value> = fields
        .iter()
        .map(|&f| (f.to_owned(), expected[f].clone()))
        .collect();
    compare(&actual, &Value::Object(expected))
}

const GLM_FIELDS: &[&str] = &[
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

const NEMOTRON_FIELDS: &[&str] = &[
    "input_ids_sha256",
    "num_input_ids",
    "modalities",
    "offsets",
    "feature_shapes",
    "feature_sha256",
    "model_kwargs",
];

/// The frames `sample` picks for one video (`glm4v.py::glm_sample_and_decode_sync`,
/// `nano_nemotron_vl.py::parse_video`), decoded at the source size.
fn check_source_video_stages(
    bytes: &[u8],
    sample: impl FnOnce(&VideoInfo) -> Result<Vec<u64>, MmError>,
    expected: &Value,
) -> Result<(), String> {
    let limits = VideoDecodeLimits::default();
    let video = Video::open(&bytes, &limits).map_err(|e| e.to_string())?;
    let info = video.info();
    let indices = sample(&info).map_err(|e| e.to_string())?;
    let rgb = match info.height % 2 {
        0 => RgbConversion::Swscale,
        _ => RgbConversion::Float,
    };
    let decoded =
        (video.decode_with(&indices, None, &limits, rgb)).map_err(|error| match error {
            MmError::Unsupported { .. } => format!("{NO_DECODER}: {error}"),
            _ => error.to_string(),
        })?;
    let actual = json!({
        "total_frames": info.frame_count,
        "fps": info.average_fps,
        "indices": indices,
        "decoded_shape": [decoded.frame_count(), decoded.height(), decoded.width(), 3],
        "decoded_sha256": sha256(decoded.rgb().iter().copied()),
    });
    // JSON keeps Python's int-valued fps as an int; compare it as a number.
    let mut expected = expected.clone();
    expected["fps"] = json!(expected["fps"].as_f64());
    compare(&actual, &expected).map_err(|error| format!("video stages: {error}"))
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
fn check_video_case(case: &Value, filter: Resample) -> Result<(), String> {
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
    let processor = MmProcessor::from_spec(spec, sampling.clone().map(VideoPolicy::Qwen))
        .map_err(|e| e.to_string())?
        .with_presize_filter(filter);
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
        decoded.push(check_video_stages(
            bytes, &sampling, &hints, filter, stages,
        )?);
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
    filter: Resample,
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
        filter,
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

fn video_case(name: &str) -> (Value, Resample) {
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(fixtures().join("qwen3-vl-2b-instruct.video.json")).unwrap(),
    )
    .unwrap();
    let case = (fixture["cases"].as_array().unwrap().iter())
        .find(|case| case["name"] == name)
        .unwrap()
        .clone();
    (case, recorded_presize_filter(&fixture).unwrap())
}

/// A video placeholder before an image one: items come back images first, then
/// videos, as Python's `mm_items`, each with its own prompt span.
#[test]
fn items_follow_python_order_when_a_video_comes_first() {
    let (case, _) = video_case("image then video");
    let spec: ProcessorSpec = serde_json::from_value(case["spec"].clone()).unwrap();
    let sampling: VideoSampling =
        serde_json::from_value(case["spec"]["video_sampling"].clone()).unwrap();
    let processor = MmProcessor::from_spec(spec, Some(VideoPolicy::Qwen(sampling))).unwrap();
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
    let (case, filter) = video_case("image then video");
    let spec: ProcessorSpec = serde_json::from_value(case["spec"].clone()).unwrap();
    let sampling: VideoSampling =
        serde_json::from_value(case["spec"]["video_sampling"].clone()).unwrap();
    let processor = MmProcessor::from_spec(spec, Some(VideoPolicy::Qwen(sampling.clone())))
        .unwrap()
        .with_presize_filter(filter);
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
        check_video_stages(&video, &sampling, &VideoHints::default(), filter, stages).unwrap(),
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
    let (case, _) = video_case("image then video");
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

/// Python's GLM path honors a request's `fps` and `max_frames`, which the
/// crate does not take, so they are refused; it ignores `num_frames`.
#[test]
fn glm_video_refuses_the_hints_python_honors() {
    let fixture: Value =
        serde_json::from_slice(&std::fs::read(fixtures().join("glm-5.3-flash.json")).unwrap())
            .unwrap();
    let (spec, video) = checkpoint_spec(&fixture).unwrap();
    let processor = MmProcessor::from_spec(spec, video).unwrap();
    let case = (fixture["cases"].as_array().unwrap().iter())
        .find(|case| case["name"] == "vp9_64x48_30f.webm")
        .unwrap();
    let ids: Vec<i32> = (case["input_ids"].as_array().unwrap().iter())
        .map(|id| id.as_i64().unwrap() as i32)
        .collect();
    let bytes = fixture_bytes("videos", &case["videos"]).remove(0);
    let text =
        Timestamps(serde_json::from_value(case["expected"]["timestamp_ids"].clone()).unwrap());
    let run = |hints| {
        let videos = [MmVideo {
            bytes: &bytes,
            hints,
        }];
        processor.process_media(&ids, &[], &videos, &text)
    };
    for (hints, name) in [
        (
            VideoHints {
                fps: Some(1.0),
                ..Default::default()
            },
            "fps",
        ),
        (
            VideoHints {
                max_frames: Some(2),
                ..Default::default()
            },
            "max_frames",
        ),
    ] {
        let error = run(hints).err().map(|error| error.to_string());
        assert_eq!(
            error,
            Some(format!("video input does not take the `{name}` hint"))
        );
    }
    let ignored = run(VideoHints {
        num_frames: Some(4),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(
        case_summary(&ignored),
        case_summary(&run(VideoHints::default()).unwrap())
    );
}

/// `--image-processor-backend` can load another class than the checkpoint
/// declares; the class SGLang loaded picks the resampler.
#[test]
fn glm_resampler_follows_the_class_sglang_loaded() {
    let source = fixtures().join("glm-5.3-flash");
    let dir = std::env::temp_dir().join(format!("sglang-mm-glm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(source.join("config.json"), dir.join("config.json")).unwrap();
    let mut processor: Value =
        serde_json::from_slice(&std::fs::read(source.join("processor_config.json")).unwrap())
            .unwrap();
    processor["image_processor"]["image_processor_type"] = json!("Glm5NextImageProcessorPil");
    std::fs::write(dir.join("processor_config.json"), processor.to_string()).unwrap();
    let resolved = spec_from_model_dir(MmDirFamily::Glm5Next, &dir, Resampler::AtenU8);
    std::fs::remove_dir_all(&dir).unwrap();
    let Ok((ProcessorSpec::Glm5NextImage(spec), _)) = resolved else {
        panic!("{resolved:?}");
    };
    assert_eq!(spec.resample, Resampler::AtenU8);
}

/// Transformers loads a `processor_config.json` that nests the image processor
/// before `preprocessor_config.json`, and the flat file when that entry is null.
#[test]
fn glm_processor_file_follows_transformers_order() {
    let source = fixtures().join("glm-5.3-flash");
    let dir = std::env::temp_dir().join(format!("sglang-mm-glm-flat-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(source.join("config.json"), dir.join("config.json")).unwrap();
    let mut processor: Value =
        serde_json::from_slice(&std::fs::read(source.join("processor_config.json")).unwrap())
            .unwrap();
    let mut flat = processor["image_processor"].clone();
    flat["max_image_tokens"] = json!(16);
    let resolve = |nested: &Value, flat: &Value| {
        std::fs::write(dir.join("processor_config.json"), nested.to_string()).unwrap();
        std::fs::write(dir.join("preprocessor_config.json"), flat.to_string()).unwrap();
        spec_from_model_dir(MmDirFamily::Glm5Next, &dir, Resampler::AtenU8)
    };
    // A stale flat file is ignored, even one that config.json contradicts.
    let mut stale = flat.clone();
    stale["patch_size"] = json!(16);
    let ignored = resolve(&processor, &stale);
    processor["image_processor"] = Value::Null;
    let fallback = resolve(&processor, &flat);
    std::fs::remove_dir_all(&dir).unwrap();
    let expected = spec_from_model_dir(MmDirFamily::Glm5Next, &source, Resampler::AtenU8);
    assert_eq!(format!("{ignored:?}"), format!("{expected:?}"));
    let Ok((ProcessorSpec::Glm5NextImage(spec), _)) = fallback else {
        panic!("{fallback:?}");
    };
    assert_eq!(spec.planner.geometry.max_image_tokens, 16);
}

fn nemotron_fixture() -> Value {
    let path = fixtures().join("nemotron-3.5-super-vl.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// Python merges a request's videos into one item and serves images with
/// videos; the family takes neither, so both are refused before any decode.
#[test]
fn nemotron_refuses_two_videos_and_mixed_requests() {
    let fixture = nemotron_fixture();
    let (spec, video) = checkpoint_spec(&fixture).unwrap();
    assert_eq!(video, Some(VideoPolicy::Nemotron));
    let processor = MmProcessor::from_spec(spec, video).unwrap();
    let empty = Timestamps(HashMap::new());
    let video = |bytes| MmVideo {
        bytes,
        hints: VideoHints::default(),
    };
    let error = |images: &[&[u8]], videos: &[MmVideo]| {
        (processor.process_media(&[18, 18], images, videos, &empty))
            .err()
            .map(|error| error.to_string())
    };
    // Not media: the refusal comes before any decode.
    assert_eq!(
        error(&[], &[video(b"a"), video(b"b")]),
        Some("a request with more than one video is not supported".into())
    );
    assert_eq!(
        error(&[b"a"], &[video(b"b")]),
        Some("a request with both images and videos is not supported".into())
    );
}

/// A tokenizer with one id per space-separated word.
struct Words(&'static [&'static str]);

impl TextEncoder for Words {
    fn encode(&self, text: &str) -> sglang_processor::dynamo_multimodal::Result<Vec<i32>> {
        (text.split_whitespace())
            .map(|word| {
                (self.0.iter().position(|known| *known == word))
                    .map(|id| id as i32)
                    .ok_or_else(|| MmError::invalid_input(format!("no id for {word:?}")))
            })
            .collect()
    }

    fn decode(&self, ids: &[i32]) -> sglang_processor::dynamo_multimodal::Result<String> {
        let words: Vec<&str> = ids.iter().map(|&id| self.0[id as usize]).collect();
        Ok(words.join(" "))
    }
}

/// `nano_nemotron_vl.py`'s text path: each of the request's videos takes the
/// next `<video>` as the image token (18), and the text between encodes alone.
#[test]
fn nemotron_rewrites_one_video_placeholder_per_video() {
    let (spec, video) = checkpoint_spec(&nemotron_fixture()).unwrap();
    let processor = MmProcessor::from_spec(spec, video).unwrap();
    let words = Words(&["<s>", "hi", "<video>", "there"]);
    let rewrite =
        |videos| (processor.rewrite_prompt(&[0, 1, 2, 3, 2], (0, videos), &words)).unwrap();
    assert_eq!(rewrite(2), Some(vec![0, 1, 18, 3, 18]));
    // A `<video>` past the request's videos stays text, as in Python.
    assert_eq!(rewrite(1), Some(vec![0, 1, 18, 3, 2]));
    assert_eq!(rewrite(0), None);
}

/// A checkpoint whose video SGLang cannot mirror (here `vision_config` with
/// one frame per group, Python's per-frame path) resolves with video off and
/// still serves images.
#[test]
fn nemotron_checkpoint_without_mirrorable_video_serves_images() {
    let source = fixtures().join("nemotron-3.5-super-vl");
    let dir = std::env::temp_dir().join(format!("sglang-mm-nemotron-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for name in [
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
    ] {
        std::fs::copy(source.join(name), dir.join(name)).unwrap();
    }
    let mut config: Value =
        serde_json::from_slice(&std::fs::read(source.join("config.json")).unwrap()).unwrap();
    config["vision_config"]["video_temporal_patch_size"] = json!(1);
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let family = MmDirFamily::NemotronVl {
        max_model_len: 16384,
        video_target_num_patches: 0,
        video_maintain_aspect_ratio: true,
        video_pruning_rate: 0.7,
    };
    let resolved = spec_from_model_dir(family, &dir, Resampler::Pil);
    std::fs::remove_dir_all(&dir).unwrap();
    let (spec, video) = resolved.unwrap();
    assert_eq!(video, None);
    let ProcessorSpec::NemotronVl(spec) = spec else {
        panic!("a Nemotron spec");
    };
    assert_eq!((spec.video, spec.max_model_len), (None, 16384));
}

/// The server's values replace the checkpoint's (16384 tokens, a 1024-patch
/// frame target, pruning rate 0.7): a `--json-model-override-args` pruning rate
/// left behind gives Rust a different video token count than Python.
#[test]
fn nemotron_server_values_replace_the_checkpoint_values() {
    let family = MmDirFamily::NemotronVl {
        max_model_len: 8192,
        video_target_num_patches: 0,
        video_maintain_aspect_ratio: false,
        video_pruning_rate: 0.0,
    };
    let dir = fixtures().join("nemotron-3.5-super-vl");
    let (spec, video) = spec_from_model_dir(family, &dir, Resampler::Pil).unwrap();
    assert!(video.is_some());
    let ProcessorSpec::NemotronVl(spec) = spec else {
        panic!("a Nemotron spec");
    };
    let video = spec.video.unwrap();
    assert_eq!(
        (
            spec.max_model_len,
            video.target_num_patches,
            video.maintain_aspect_ratio,
            video.pruning_rate,
        ),
        (8192, 0, false, 0.0)
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

/// The fields `generate_mm_nemotron_parity.py` records. The model kwargs are
/// the ones the drain derives from each item's counts for this checkpoint
/// (dynamic images, EVS on, 32 pixels per token side).
fn nemotron_summary(output: &MmOutput) -> Value {
    assert!(matches!(output.positions, PositionOutput::Rope1D));
    let kwargs = output
        .items
        .iter()
        .map(|item| match (&item.geometry, &item.feature.shape[..]) {
            (Some(Geometry::FrameGroups { tokens, .. }), &[frames, _, height, width]) => json!({
                "thw_grids": [[tokens.len(), height / 32, width / 32]],
                "frames_per_video": [frames],
            }),
            _ => json!({"num_tokens": item.feature_token_count, "is_dynamic": true}),
        });
    let features = output.items.iter().map(|item| match &item.feature.data {
        TensorData::Bf16(bits) => sha256(bits.iter().flat_map(|bit| bit.to_le_bytes())),
        _ => panic!("Nemotron features are bfloat16"),
    });
    json!({
        "input_ids_sha256": sha256(output.input_ids.iter().flat_map(|id| id.to_le_bytes())),
        "num_input_ids": output.input_ids.len(),
        "modalities": (output.items.iter())
            .map(|item| format!("{:?}", item.modality).to_lowercase())
            .collect::<Vec<_>>(),
        "offsets": output.items.iter().map(|item| &item.offsets).collect::<Vec<_>>(),
        "feature_shapes": output.items.iter().map(|item| &item.feature.shape).collect::<Vec<_>>(),
        "feature_sha256": features.collect::<Vec<_>>(),
        "model_kwargs": kwargs.collect::<Vec<_>>(),
    })
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
