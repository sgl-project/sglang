"""Record SGLang's Python CPU path for GLM-5.3-Flash images and video as
`tests/fixtures/mm_parity/glm-5.3-flash.json`.

The reference is `Glm4vImageProcessor` on the CPU: PIL decode with its
`smart_rgb_conversion`, the torchvision image processor on the CPU, torchcodec
on the CPU, and the frame resize under torch's own CPU dispatch (Dynamo's
bicubic frame resize mirrors the AVX2 and AVX-512 builds, not DEFAULT). Run it
where torchcodec finds FFmpeg 6.1, with a CUDA device visible (the video path
pins frames to page-locked memory):

    python tests/scripts/generate_mm_glm_parity.py zai-org/GLM-5.3-Flash

The Rust test resolves the spec from the checkpoint files this script writes to
`tests/fixtures/mm_parity/glm-5.3-flash/`, as the server does. Each case records
the frames each video samples, the decoded frames, then the processor's outputs.
It also records `glm-5.3-flash-small-video-budget`, the same checkpoint with a
video token budget small enough that SGLang's presize shrinks the frames.
"""

import argparse
import asyncio
import json
import platform
import tempfile
from pathlib import Path
from types import SimpleNamespace

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
from generate_mm_video_parity import run
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.hf_transformers.processor import get_processor  # noqa: E402
from sglang.srt.managers.multimodal_processor import import_processors  # noqa: E402
from sglang.srt.multimodal.processors.glm4v import (  # noqa: E402
    Glm4vImageProcessor,
    glm_processor_video_config,
    glm_sample_and_decode_sync,
)
from sglang.srt.runtime_context import publish, reset_context  # noqa: E402
from sglang.srt.rust_server.multimodal import (  # noqa: E402
    RustMmModelDirSpec,
    RustMmProcessor,
)
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.srt.utils.video_decoder import VideoDecoderWrapper  # noqa: E402

VIDEOS = FIXTURES / "videos"
CHECKPOINT = "glm-5.3-flash"
# Decoded by NVDEC in Rust; the CPU-only test reports these as not run.
NVDEC_ONLY = {"h264_64x48_48f.mp4"}
IMAGE = "<|begin_of_image|><|image|><|end_of_image|>"
VIDEO = "<|begin_of_video|><|video|><|end_of_video|>"


def cases():
    """(name, images, videos, prompt order) per case; the prompt order lists
    "image" and "video" blocks."""
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
    yield "video then image", ["rgb_96x80.png"], two[:1], ["video", "image"]


class PythonPath:
    """SGLang's GLM-V processor over one checkpoint, on the CPU."""

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
                tokenizer_path=str(model_dir),
                model_impl="sglang",
                mm_feature_transport="cpu",
                trust_remote_code=True,
            ),
            role="tokenizer",
        )
        self.hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        processor = get_processor(str(model_dir), trust_remote_code=True)
        self.processor = Glm4vImageProcessor(
            self.hf_config, server_args, processor, None, skip_mm_pool=True
        )
        # The CPU path: PIL decode (read by a classmethod) and the CPU image
        # kernels; the class already pins video to the CPU.
        Glm4vImageProcessor.gpu_image_decode = False
        assert Glm4vImageProcessor.video_preprocessing_device == "cpu"
        self.processor._fast_image_processor_device = lambda _processor: "cpu"
        self.check_gate(processor, model, revision)

    def check_gate(self, processor, model, revision):
        """SGLang's Rust MM gate hands this checkpoint to the model-dir path."""
        import_processors("sglang.srt.multimodal.processors")
        host = RustMmProcessor.__new__(RustMmProcessor)
        host.model_config = SimpleNamespace(hf_config=self.hf_config, model_path=model)
        host._processor = processor
        host.server_args = SimpleNamespace(revision=revision)
        spec = host.resolve_spec()
        assert isinstance(spec, RustMmModelDirSpec), spec
        assert spec.family == "glm5_next", spec

    def close(self):
        self.processor.io_executor.shutdown()
        self.processor.cpu_executor.shutdown()
        reset_context()

    @property
    def tokenizer(self):
        return self.processor._processor.tokenizer

    def prompt(self, order):
        """The chat template's media blocks, then text."""
        blocks = {"image": IMAGE, "video": VIDEO}
        return "[gMASK]<sop><|user|>" + "".join(blocks[b] for b in order) + "Describe."

    def video_stages(self, data):
        """The frames `glm_sample_and_decode_sync` picks, decoded at the source size."""
        video_processor = self.processor._processor.video_processor
        decoder = VideoDecoderWrapper(data, device="cpu")
        total, fps = len(decoder), decoder.avg_fps
        _, metadata = glm_sample_and_decode_sync(
            decoder, glm_processor_video_config(video_processor), video_processor
        )
        indices = metadata["frames_indices"]
        decoded = VideoDecoderWrapper(data, device="cpu").get_frames_as_tensor(indices)
        decoded = decoded.numpy()
        return {
            "total_frames": total,
            "fps": fps,
            "indices": indices,
            "decoded_shape": list(decoded.shape),
            "decoded_sha256": sha256(decoded),
        }

    def timestamp_ids(self, stages):
        """Ids of every `"{t:.1f} seconds"` a clip's frames can take, from the
        real tokenizer: the Rust test encodes them through this map."""
        texts = {}
        for stage in stages:
            for index in range(stage["total_frames"]):
                text = f"{index / stage['fps']:.1f} seconds"
                texts[text] = self.tokenizer.encode(text, add_special_tokens=False)
        return texts


def checkpoint_files(model_dir, checkpoint):
    """config.json, cut to the fields the Rust resolver reads, and processor_config.json."""
    config = json.loads((model_dir / "config.json").read_text())
    ids = [
        key
        for key in config
        if key.endswith("_token_id") and key.split("_")[0] in ("image", "video")
    ]
    vision = {
        key: config["vision_config"][key]
        for key in ("patch_size", "spatial_merge_size", "temporal_patch_size")
    }
    cut = {"model_type": config["model_type"], **{k: config[k] for k in ids}}
    cut["vision_config"] = vision
    out = FIXTURES / checkpoint
    out.mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(cut, indent=2) + "\n")
    processor = json.loads((model_dir / "processor_config.json").read_text())
    (out / "processor_config.json").write_text(json.dumps(processor, indent=2) + "\n")


# Small enough that the presize shrinks the 300-frame clip's frames to fit the
# budget, alone and split across two videos.
SMALL_VIDEO_BUDGET = 32


def small_budget_cases():
    clips = ["vp9_64x48_300f.webm", "vp9_64x48_30f.webm"]
    yield f"small budget: {clips[0]}", [], clips[:1], ["video"]
    yield "small budget: two videos", [], clips, ["video"] * 2


def small_budget_dir(model_dir, out):
    """`model_dir` with the video processor's `max_image_tokens` cut."""
    for file in model_dir.iterdir():
        if file.name != "processor_config.json":
            (out / file.name).symlink_to(file.resolve())
    processor = json.loads((model_dir / "processor_config.json").read_text())
    processor["video_processor"]["max_image_tokens"] = SMALL_VIDEO_BUDGET
    (out / "processor_config.json").write_text(json.dumps(processor))
    return out


def record(checkpoint, model_dir, cases, model, revision):
    checkpoint_files(model_dir, checkpoint)
    path = PythonPath(model_dir, model, revision)
    fixture = {
        "model": model,
        "revision": revision,
        "checkpoint": checkpoint,
        "family": "glm5_next",
        "reference": {
            "torch": torch.__version__,
            "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "torchcodec": torchcodec.__version__,
            "ffmpeg": torchcodec._core.get_ffmpeg_library_versions()["ffmpeg_version"],
            "transformers": transformers.__version__,
            "pillow": PIL.__version__,
            "machine": platform.machine(),
        },
        "cases": [],
    }
    for name, image_names, video_names, order in cases:
        prompt = path.prompt(order)
        case = {
            "name": name,
            "images": image_names,
            "videos": video_names,
            "input_ids": path.tokenizer.encode(prompt, add_special_tokens=False),
        }
        images = [corpus_bytes(image) for image in image_names]
        videos = [(VIDEOS / video).read_bytes() for video in video_names]
        try:
            stages = [path.video_stages(data) for data in videos]
            case["expected"] = {
                "video_stages": stages,
                "timestamp_ids": path.timestamp_ids(stages),
                **run(path, prompt, images, videos or None),
            }
        except Exception as error:  # noqa: BLE001 -- the rejection is the recorded result
            case["error"] = f"{type(error).__name__}: {error}"
        gaps = [KNOWN_GAPS[image] for image in image_names if image in KNOWN_GAPS]
        if gaps:
            case["known_gap"] = "; ".join(gaps)
        if NVDEC_ONLY & set(video_names):
            case["nvdec"] = True
        fixture["cases"].append(case)
        print(name, "error" if "error" in case else "ok")
    path.close()

    out = FIXTURES / f"{checkpoint}.json"
    out.write_text(dump(fixture))
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--revision")
    parser.add_argument(
        "--model-dir", help="a local copy of the checkpoint's processor and tokenizer"
    )
    args = parser.parse_args()
    revision = args.revision or HfApi().model_info(args.model).sha
    model_dir = Path(
        args.model_dir
        or snapshot_download(
            args.model, revision=revision, allow_patterns=PROCESSOR_FILES
        )
    )
    record(CHECKPOINT, model_dir, cases(), args.model, revision)
    with tempfile.TemporaryDirectory() as tmp:
        record(
            f"{CHECKPOINT}-small-video-budget",
            small_budget_dir(model_dir, Path(tmp)),
            small_budget_cases(),
            args.model,
            revision,
        )


if __name__ == "__main__":
    main()
