"""Record SGLang's Python CPU video path as `tests/fixtures/mm_parity/<model>.video.json`.

The reference is SGLang's Python CPU path on x86-64: torchcodec on the CPU
(FFmpeg's SIMD swscale converts YUV to RGB), the fp32 pre-resize of torch's
DEFAULT CPU dispatch, and the HF video processor on the CPU. Run it where
torchcodec finds FFmpeg 6.1 (Ubuntu 24.04's, as the SGLang image ships):

    ATEN_CPU_CAPABILITY=default \\
        python tests/scripts/generate_mm_video_parity.py Qwen/Qwen3-VL-2B-Instruct

The video path pins frames to page-locked memory, so a CUDA device must be
visible; every stage still runs on the CPU. `ATEN_CPU_CAPABILITY=default`
matters: torch's AVX2 build fuses part of the pre-resize sum into FMA. Each case records every stage as sha256: the frames
sampled, the decoded frames, the pre-resized frames, then the processor's
outputs. Writes the clips it needs into `tests/fixtures/mm_parity/videos/`.
"""

import argparse
import asyncio
import io
import platform
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import torch
import torchcodec
import transformers
from generate_mm_parity import (
    FIXTURES,
    PROCESSOR_FILES,
    PythonPath,
    corpus_bytes,
    dump,
    sha256,
)
from huggingface_hub import HfApi, snapshot_download

from sglang.srt.multimodal.processors.qwen_vl import (  # noqa: E402
    preprocess_video,
    smart_nframes,
)
from sglang.srt.utils.video_decoder import VideoDecoderWrapper  # noqa: E402

VIDEOS = FIXTURES / "videos"


def frame(index, height, width):
    """A colorful moving gradient: chroma varies across every 2x2 block."""
    y, x = np.mgrid[0:height, 0:width]
    rgb = np.stack(
        [
            (x * 7 + index * 5) % 256,
            (y * 11 + index * 3) % 256,
            ((x + y) * 5 + index * 9) % 256,
        ],
        axis=-1,
    )
    return rgb.astype(np.uint8)


def encode(codec, fmt, height, width, count, fps, options=None, color=None):
    buffer = io.BytesIO()
    with av.open(buffer, "w", format=fmt) as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = options or {}
        for key, value in (color or {}).items():
            setattr(stream.codec_context, key, value)
        for i in range(count):
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(frame(i, height, width), format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


VP9 = {"crf": "30", "b:v": "0"}
# name -> encoded bytes. Small and deterministic; each exercises one decode path.
CORPUS = {
    "vp9_64x48_30f.webm": lambda: encode("libvpx-vp9", "webm", 48, 64, 30, 24, VP9),
    "vp8_64x48_30f.webm": lambda: encode(
        "libvpx", "webm", 48, 64, 30, 25, {"crf": "10", "b:v": "1M"}
    ),
    "vp9_odd_66x50_20f.webm": lambda: encode("libvpx-vp9", "webm", 50, 66, 20, 30, VP9),
    "vp9_bt709_full_64x48_20f.webm": lambda: encode(
        "libvpx-vp9",
        "webm",
        48,
        64,
        20,
        24,
        VP9,
        color={"colorspace": 1, "color_range": 2},
    ),
    "vp9_64x48_300f.webm": lambda: encode("libvpx-vp9", "webm", 48, 64, 300, 30, VP9),
    "vp9_64x48_1f.webm": lambda: encode("libvpx-vp9", "webm", 48, 64, 1, 24, VP9),
    "h264_64x48_48f.mp4": lambda: encode(
        "libx264", "mp4", 48, 64, 48, 24, {"crf": "23", "bf": "2"}
    ),
    "h264_bt709_64x48_20f.mp4": lambda: encode(
        "libx264", "mp4", 48, 64, 20, 30, {"crf": "23"}, color={"colorspace": 1}
    ),
    # NVDEC decodes AV1 from 128x128 up, and Dynamo has no CPU AV1 decoder.
    "av1_160x128_20f.webm": lambda: encode(
        "libaom-av1", "webm", 128, 160, 20, 24, {"crf": "40", "cpu-used": "8"}
    ),
}

# Decoded by NVDEC in Rust; the CPU-only test reports these as not run.
NVDEC_ONLY = {"h264_64x48_48f.mp4", "h264_bt709_64x48_20f.mp4", "av1_160x128_20f.webm"}


def video_bytes(name):
    path = VIDEOS / name
    if not path.exists():
        VIDEOS.mkdir(parents=True, exist_ok=True)
        path.write_bytes(CORPUS[name]())
    return path.read_bytes()


def cases():
    """(name, videos, images, server video config, hints of the one video)."""
    for name in CORPUS:
        yield name, [name], [], {}, {}
    long = "vp9_64x48_300f.webm"
    yield "fps hint", [long], [], {}, {"fps": 1.0}
    yield "num_frames hint", [long], [], {}, {"num_frames": 6}
    yield "max_frames hint", [long], [], {}, {"max_frames": 8}
    yield "min_frames and fps hints", [long], [], {}, {"fps": 0.5, "min_frames": 10}
    yield "server nframes", [long], [], {"nframes": 10}, {}
    yield (
        "server pixel bounds",
        [long],
        [],
        {"max_pixels": 40000, "min_pixels": 20000},
        {},
    )
    yield "fps and num_frames hints", [long], [], {}, {"fps": 1.0, "num_frames": 6}
    yield "two videos", ["vp9_64x48_30f.webm", "vp8_64x48_30f.webm"], [], {}, {}
    yield "image then video", ["vp9_64x48_30f.webm"], ["rgb_96x80.png"], {}, {}


def reference_config(server, hints):
    """The server config Python needs to sample as Rust does with `hints`
    (`VideoConfig::with_hints`): `num_frames` replaces the fps mode, and the
    fps-mode hints replace `nframes`. Python reads no per-request hints."""
    config = dict(server)
    if "num_frames" in hints:
        for key in ("fps", "min_frames", "max_frames"):
            config.pop(key, None)
        config["nframes"] = hints["num_frames"]
        hints = {k: v for k, v in hints.items() if k != "num_frames"}
        config.update(hints)  # a combined fps hint fails as Python's assert does
    elif hints:
        config.pop("nframes", None)
        config.update(hints)
    return config


def video_stages(data, config):
    """Each stage of `preprocess_video` for one clip, on the CPU decoder."""
    decoder = VideoDecoderWrapper(data, device="cpu")
    total, fps = len(decoder), decoder.avg_fps
    nframes = smart_nframes(config, total_frames=total, video_fps=fps)
    indices = np.unique(np.linspace(0, total - 1, num=nframes, dtype=np.int64))
    decoded = decoder.get_frames_as_tensor(indices.tolist()).numpy()
    presized, _ = asyncio.run(
        preprocess_video(VideoDecoderWrapper(data, device="cpu"), video_config=config)
    )
    presized = presized.permute(0, 2, 3, 1).contiguous().numpy()
    return {
        "total_frames": total,
        "fps": fps,
        "num_frames": nframes,
        "indices": indices.tolist(),
        "decoded_shape": list(decoded.shape),
        "decoded_sha256": sha256(decoded),
        "presized_shape": list(presized.shape),
        "presized_sha256": sha256(presized),
    }


def timestamp_ids(path, stages):
    """The ids of every timestamp the expansions need, from the real tokenizer:
    Rust's test encodes them through a map, as the server does through its tokenizer."""
    tokenizer = path.processor._processor.tokenizer
    tps = path.processor._processor.video_processor.temporal_patch_size
    texts = {}
    for stage in stages:
        indices = list(stage["indices"])
        indices += [indices[-1]] * (-len(indices) % tps)
        seconds = [i / stage["fps"] for i in indices]
        for group in range(0, len(seconds), tps):
            text = f"<{(seconds[group] + seconds[group + tps - 1]) / 2:.1f} seconds>"
            texts[text] = tokenizer.encode(text, add_special_tokens=False)
    return texts


def run(path, input_ids, image_data, video_data):
    output = asyncio.run(
        path.processor.process_mm_data_async(
            image_data=image_data,
            input_text=input_ids,
            request_obj=SimpleNamespace(
                video_data=video_data, audio_data=None, rid="parity"
            ),
        )
    )
    features = [item.feature.detach().cpu().numpy() for item in output.mm_items]
    grids = [
        item.model_specific_data[
            "video_grid_thw" if item.is_video() else "image_grid_thw"
        ]
        for item in output.mm_items
    ]
    return {
        "input_ids_sha256": sha256(np.asarray(output.input_ids, dtype="<i4")),
        "num_input_ids": len(output.input_ids),
        "modalities": [item.modality.name.lower() for item in output.mm_items],
        "offsets": [[list(span) for span in item.offsets] for item in output.mm_items],
        "grids": [np.asarray(grid).flatten().tolist() for grid in grids],
        "feature_shapes": [list(feature.shape) for feature in features],
        "feature_sha256": [sha256(feature.astype("<f4")) for feature in features],
        "mrope_sha256": sha256(output.mrope_positions.cpu().numpy().astype("<i8")),
        "mrope_delta": int(output.mrope_position_delta.item()),
    }


def prompt(path, num_images, num_videos):
    config = path.hf_config
    image = [
        config.vision_start_token_id,
        config.image_token_id,
        config.vision_end_token_id,
    ]
    video = [
        config.vision_start_token_id,
        config.video_token_id,
        config.vision_end_token_id,
    ]
    text = path.processor._processor.tokenizer.encode(
        "Describe.", add_special_tokens=False
    )
    return image * num_images + video * num_videos + text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--revision")
    args = parser.parse_args()
    revision = args.revision or HfApi().model_info(args.model).sha
    model_dir = Path(
        snapshot_download(args.model, revision=revision, allow_patterns=PROCESSOR_FILES)
    )
    assert torch.backends.cpu.get_cpu_capability() == "DEFAULT", (
        "set ATEN_CPU_CAPABILITY=default"
    )

    fixture = {
        "model": args.model,
        "revision": revision,
        "reference": {
            "torch": torch.__version__,
            "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "torchcodec": torchcodec.__version__,
            "ffmpeg": torchcodec._core.get_ffmpeg_library_versions()["ffmpeg_version"],
            "transformers": transformers.__version__,
            "machine": platform.machine(),
        },
        "cases": [],
    }
    for name, video_names, image_names, server, hints in cases():
        # Rust gets the server config and the hints; Python, which reads no
        # hints, gets them merged into its server config.
        spec_path = PythonPath(model_dir, "auto", None, video_config=server)
        rust_spec = spec_path.spec
        spec_path.close()
        config = reference_config(server, hints)
        path = PythonPath(
            model_dir,
            "auto",
            None,
            video_config=config,
        )
        videos = [video_bytes(video) for video in video_names]
        input_ids = prompt(path, len(image_names), len(video_names))
        case = {
            "name": name,
            "videos": video_names,
            "video_hints": hints,
            "images": image_names,
            "spec": rust_spec,
            "input_ids": input_ids,
        }
        try:
            stages = [video_stages(data, config) for data in videos]
            case["expected"] = {
                "video_stages": stages,
                "timestamp_ids": timestamp_ids(path, stages),
                **run(path, input_ids, [corpus_bytes(i) for i in image_names], videos),
            }
        except Exception as error:  # noqa: BLE001 -- the rejection is the recorded result
            case["error"] = f"{type(error).__name__}: {error}"
        if NVDEC_ONLY & set(video_names):
            case["nvdec"] = True
        fixture["cases"].append(case)
        print(name, "error" if "error" in case else "ok")
        path.close()

    out = FIXTURES / f"{args.model.split('/')[-1].lower()}.video.json"
    out.write_text(dump(fixture))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
