"""Record SGLang's Python CPU path for Nemotron 3.5 Super VL images and video
as `tests/fixtures/mm_parity/nemotron-3.5-super-vl.json`.

The reference is `NanoNemotronVLImageProcessor` on the CPU: PIL decode, its own
PIL resize, and torchcodec on the CPU. A chat prompt reaches it as text, so it
takes its text path: a `<video>` placeholder, which is no single token, is
replaced in the text and the whole prompt tokenized again. Each case records the
ids the Rust server hands the pipeline for the same prompt: the text around each
`<video>` encoded alone, with the `<image>` id in its place. Run it where
torchcodec finds FFmpeg 6.1:

    python tests/scripts/generate_mm_nemotron_parity.py \
        nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 --revision <sha> --model-dir <dir>

The Rust test resolves the spec from the checkpoint files this script writes to
`tests/fixtures/mm_parity/nemotron-3.5-super-vl/`, with the server values the
fixture records, as the server does.
"""

import argparse
import asyncio
import json
import math
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
import torch
import torchcodec
import transformers
from generate_mm_parity import (
    CORPUS,
    FIXTURES,
    KNOWN_GAPS,
    PROCESSOR_FILES,
    corpus_bytes,
    dump,
    sha256,
)
from huggingface_hub import snapshot_download

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.hf_transformers.config import get_config  # noqa: E402
from sglang.srt.hf_transformers.processor import get_processor  # noqa: E402
from sglang.srt.managers.multimodal_processor import import_processors  # noqa: E402
from sglang.srt.multimodal.media_io import sample_video_frames  # noqa: E402
from sglang.srt.multimodal.processors.nano_nemotron_vl import (  # noqa: E402
    DESIRED_FPS,
    MAX_FRAMES,
    NanoNemotronVLImageProcessor,
)
from sglang.srt.runtime_context import publish, reset_context  # noqa: E402
from sglang.srt.rust_server.multimodal import (  # noqa: E402
    RustMmModelDirSpec,
    RustMmProcessor,
)
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.srt.utils.video_decoder import VideoDecoderWrapper  # noqa: E402

VIDEOS = FIXTURES / "videos"
CHECKPOINT = "nemotron-3.5-super-vl"
CONTEXT_LENGTH = 16384
# Decoded by NVDEC in Rust; the CPU-only test reports these as not run.
NVDEC_ONLY = {"h264_64x48_48f.mp4"}
BEFORE = "<|im_start|>user\n"
AFTER = "Describe this.<|im_end|>\n<|im_start|>assistant\n"
# Python serves these; the Rust path refuses them with a 400, or differs where Python errs.
GAPS = {
    "two videos": "Rust refuses a request with more than one video: Python merges them into one item",
    "image then video": "Rust refuses a request with both images and videos (dmm: images or videos, not both)",
    "two images": "SGLang's text path writes every image at the first <image> "
    "(prompt.replace(..., 1)) and keeps the later <image> as a stray token; Rust expands each",
}


def cases():
    """(name, images, videos, prompt order) per case."""
    for name in CORPUS:
        yield name, [name], [], ["image"]
    yield "two images", ["rgb_96x80.png", "rgb_333x251.jpg"], [], ["image"] * 2
    for name in (
        "vp9_64x48_30f.webm",
        "vp8_64x48_30f.webm",
        "vp9_odd_66x50_20f.webm",
        "vp9_bt709_full_64x48_20f.webm",
        "vp9_64x48_300f.webm",
        "vp9_64x48_1f.webm",
        "h264_64x48_48f.mp4",
    ):
        yield name, [], [name], ["video"]
    two = ["vp9_64x48_30f.webm", "vp8_64x48_30f.webm"]
    yield "two videos", [], two, ["video"] * 2
    yield "image then video", ["rgb_96x80.png"], two[:1], ["image", "video"]


class PythonPath:
    """SGLang's Nemotron VL processor over one checkpoint, on the CPU."""

    def __init__(self, model_dir, model, revision):
        server_args = SimpleNamespace(
            model_impl="sglang",
            mm_feature_transport="cpu",
            mm_enable_dp_encoder=False,
            image_processor_backend="auto",
            skip_tokenizer_init=False,
            mm_preprocess_cache_size_mb=0,
            trust_mm_content_hashes=False,
            tp_size=1,
            dist_init_addr=None,
            mm_process_config={},
            mm_io_worker_num=1,
            mm_processor_worker_num=1,
            tokenizer_worker_num=1,
            base_gpu_id=0,
            rl_on_policy_target=None,
            allowed_media_domains=[],
            media_url_max_file_size_mb=64,
        )
        publish(
            ServerArgs(
                model_path=str(model_dir),
                model_impl="sglang",
                mm_feature_transport="cpu",
                trust_remote_code=True,
                context_length=CONTEXT_LENGTH,
            ),
            role="tokenizer",
        )
        self.hf_config = get_config(str(model_dir), trust_remote_code=True)
        processor = get_processor(str(model_dir), trust_remote_code=True)
        # The CPU path: torchcodec on the CPU; images already decode with PIL.
        NanoNemotronVLImageProcessor.gpu_video_decode = False
        self.processor = NanoNemotronVLImageProcessor(
            self.hf_config, server_args, processor, None, skip_mm_pool=True
        )
        self.spec = self.check_gate(processor, model, revision)

    def check_gate(self, processor, model, revision):
        """SGLang's Rust MM gate hands this checkpoint to the model-dir path."""
        import_processors("sglang.srt.multimodal.processors")
        host = RustMmProcessor.__new__(RustMmProcessor)
        host.model_config = SimpleNamespace(hf_config=self.hf_config, model_path=model)
        host._processor = processor
        host.server_args = SimpleNamespace(revision=revision)
        spec = host.resolve_spec()
        assert isinstance(spec, RustMmModelDirSpec), spec
        assert spec.family == "nemotron_vl", spec
        return spec

    def close(self):
        self.processor.io_executor.shutdown()
        self.processor.cpu_executor.shutdown()
        reset_context()

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def rust_input_ids(self, prompt, videos):
        """The Rust server's ids: the text around each of the first `videos`
        `<video>` strings encoded alone."""
        ids = []
        placeholder = self.hf_config.video_context_token
        for index, text in enumerate(prompt.split(placeholder, videos)):
            if index:
                ids.append(self.spec.image_token_id)
            ids.extend(self.tokenizer.encode(text, add_special_tokens=False))
        return ids

    def run(self, prompt, images, videos):
        output = asyncio.run(
            self.processor.process_mm_data_async(
                image_data=images,
                audio_data=None,
                input_text=prompt,
                request_obj=SimpleNamespace(
                    video_data=videos or None, audio_data=None, rid="parity"
                ),
            )
        )
        assert output.mrope_positions is None
        features = [
            item.feature.detach().cpu().contiguous() for item in output.mm_items
        ]
        assert all(feature.dtype == torch.bfloat16 for feature in features)
        return {
            "input_ids_sha256": sha256(np.asarray(output.input_ids, dtype="<i4")),
            "num_input_ids": len(output.input_ids),
            "modalities": [item.modality.name.lower() for item in output.mm_items],
            "offsets": [
                [list(span) for span in item.offsets] for item in output.mm_items
            ],
            "feature_shapes": [list(feature.shape) for feature in features],
            "feature_sha256": [
                sha256(feature.view(torch.int16).numpy().astype("<i2"))
                for feature in features
            ],
            "model_kwargs": [
                self.kwargs(item, output.input_ids) for item in output.mm_items
            ],
        }

    @staticmethod
    def kwargs(item, input_ids):
        """`model_specific_data`, with `pre_chunked_input_ids` checked against the ids."""
        kwargs = {}
        for key, value in item.model_specific_data.items():
            if key == "pre_chunked_input_ids":
                assert list(value) == list(input_ids)
                continue
            kwargs[key] = json.loads(json.dumps(value))
        return kwargs

    def video_stages(self, data):
        """The frames `parse_video` samples, decoded at the source size."""
        decoder = VideoDecoderWrapper(data, device="cpu")
        total, fps = len(decoder), decoder.avg_fps
        indices = sample_video_frames(
            decoder, desired_fps=DESIRED_FPS, max_frames=MAX_FRAMES
        )
        decoded = VideoDecoderWrapper(data, device="cpu").get_frames_at(indices)
        return {
            "total_frames": total,
            "fps": fps,
            "indices": indices,
            "decoded_shape": list(decoded.shape),
            "decoded_sha256": sha256(decoded),
        }

    def label_ids(self, stages):
        """Ids of each frame-group label a clip's expansion holds, as
        `render_tubelet` writes them, from the real tokenizer."""
        group = self.hf_config.video_temporal_patch_size
        texts = {}
        for stage in stages:
            ms = int(1000 / stage["fps"]) if stage["fps"] > 0 else 0
            count = len(stage["indices"])
            for first in range(0, count, group):
                label = " and ".join(
                    f"{'Frame' if frame == first else 'frame'} {frame + 1} sampled at "
                    f"{stage['indices'][frame] * ms / 1000.0:.2f} seconds"
                    for frame in range(first, min(first + group, count))
                )
                for text in (f"{label}: ", f"\n{label}: "):
                    texts[text] = self.tokenizer.encode(text, add_special_tokens=False)
        return texts


def checkpoint_files(model_dir):
    """config.json, cut to the fields the Rust resolver reads; both processor
    files; tokenizer.json cut to its added tokens."""
    config = json.loads((model_dir / "config.json").read_text())
    keep = (
        "model_type",
        "downsample_ratio",
        "force_image_size",
        "norm_mean",
        "norm_std",
        "img_context_token",
        "img_context_token_id",
        "video_context_token",
        "video_context_token_id",
        "video_pruning_rate",
    )
    cut = {key: config[key] for key in keep if key in config}
    cut["vision_config"] = {
        key: config["vision_config"][key]
        for key in (
            "patch_size",
            "image_size",
            "min_num_patches",
            "max_num_patches",
            "video_temporal_patch_size",
        )
    }
    out = FIXTURES / CHECKPOINT
    out.mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(cut, indent=2) + "\n")
    for name in ("preprocessor_config.json", "processor_config.json"):
        processor = json.loads((model_dir / name).read_text())
        (out / name).write_text(json.dumps(processor, indent=2) + "\n")
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text())
    added = [
        token
        for token in tokenizer["added_tokens"]
        if token["content"] in ("<image>", "<img>", "</img>")
    ]
    (out / "tokenizer.json").write_text(
        json.dumps({"added_tokens": added}, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--model-dir", help="a local copy of the checkpoint's processor and tokenizer"
    )
    args = parser.parse_args()
    model_dir = Path(
        args.model_dir
        or snapshot_download(
            args.model,
            revision=args.revision,
            allow_patterns=PROCESSOR_FILES + ["*.py"],
        )
    )
    checkpoint_files(model_dir)
    path = PythonPath(model_dir, args.model, args.revision)
    nemotron = path.spec.nemotron
    fixture = {
        "model": args.model,
        "revision": args.revision,
        "checkpoint": CHECKPOINT,
        "family": "nemotron_vl",
        "server": {
            "max_model_len": nemotron.max_model_len,
            "video_target_num_patches": nemotron.video_target_num_patches,
            "video_maintain_aspect_ratio": nemotron.video_maintain_aspect_ratio,
            "video_pruning_rate": nemotron.video_pruning_rate,
        },
        "reference": {
            "torch": torch.__version__,
            "torchcodec": torchcodec.__version__,
            "ffmpeg": torchcodec._core.get_ffmpeg_library_versions()["ffmpeg_version"],
            "transformers": transformers.__version__,
            "pillow": PIL.__version__,
            "machine": platform.machine(),
        },
        "cases": [],
    }
    blocks = {"image": path.hf_config.img_context_token, "video": "<video>"}
    for name, image_names, video_names, order in cases():
        prompt = BEFORE + "".join(blocks[b] for b in order) + AFTER
        case = {
            "name": name,
            "images": image_names,
            "videos": video_names,
            "input_ids": path.rust_input_ids(prompt, videos=len(video_names)),
        }
        images = [corpus_bytes(image) for image in image_names]
        videos = [(VIDEOS / video).read_bytes() for video in video_names]
        try:
            stages = [path.video_stages(data) for data in videos]
            case["expected"] = {
                "video_stages": stages,
                "label_ids": path.label_ids(stages),
                **path.run(prompt, images, videos),
            }
        except Exception as error:  # noqa: BLE001 -- the rejection is the recorded result
            case["error"] = f"{type(error).__name__}: {error}"
        gaps = [KNOWN_GAPS[image] for image in image_names if image in KNOWN_GAPS]
        gaps += [GAPS[name]] if name in GAPS else []
        if gaps:
            case["known_gap"] = "; ".join(gaps)
        if NVDEC_ONLY & set(video_names):
            case["nvdec"] = True
        fixture["cases"].append(case)
        print(name, "error" if "error" in case else "ok")
    path.close()

    out = FIXTURES / f"{CHECKPOINT}.json"
    out.write_text(dump(fixture))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
