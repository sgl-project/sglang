"""Record SGLang's Python CPU image path as `tests/fixtures/mm_parity/<model>.json`.

Usage:
    python tests/scripts/generate_mm_parity.py Qwen/Qwen2.5-VL-3B-Instruct --formats
    python tests/scripts/generate_mm_parity.py Qwen/Qwen3.5-0.8B
    python tests/scripts/generate_mm_parity.py zai-org/GLM-5.3-Flash --formats

Loads the checkpoint's processor through SGLang's own `get_processor` (so the
backend and SGLang's processor kwargs apply), records the spec SGLang's Rust
MM gate resolves from it, then runs each case through the model's SGLang
processor's `process_mm_data_async` with images decoded by PIL and
resized on CPU: the path `MmProcessor::process` must reproduce bit for bit.
Writes the images it needs into `tests/fixtures/mm_parity/images/` when missing.
"""

import argparse
import asyncio
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
from huggingface_hub import HfApi, snapshot_download
from PIL import Image
from transformers import AutoConfig

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.multimodal_processor import (  # noqa: E402
    get_mm_processor_cls,
    import_processors,
)
from sglang.srt.runtime_context import publish, reset_context  # noqa: E402
from sglang.srt.rust_server.multimodal import (  # noqa: E402
    RUST_MM_FAMILIES,
    RustMmProcessor,
)
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.srt.utils.hf_transformers.processor import get_processor  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mm_parity"
IMAGES = FIXTURES / "images"
PROCESSOR_FILES = [
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "tokenizer*",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template*",
]


def gradient(width, height, seed=0, channels=3, noise=24):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            x * 255 / max(width - 1, 1),
            y * 255 / max(height - 1, 1),
            (x + y) % 256,
            x % 200,
        ],
        axis=-1,
    )[..., :channels]
    return np.clip(base + rng.integers(0, noise + 1, base.shape), 0, 255).astype(
        np.uint8
    )


def encode(image, fmt, **options):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    return buffer.getvalue()


def exif_rotated_jpeg():
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW. Python's Qwen path ignores it.
    return encode(Image.fromarray(gradient(64, 48, 7)), "JPEG", quality=90, exif=exif)


def animated_gif():
    frames = [Image.fromarray(gradient(48, 40, seed)) for seed in (8, 9)]
    return encode(
        frames[0], "GIF", save_all=True, append_images=frames[1:], duration=100
    )


# name -> encoded bytes. Small, deterministic, and each exercising one decode path.
CORPUS = {
    "rgb_96x80.png": lambda: encode(Image.fromarray(gradient(96, 80)), "PNG"),
    "rgb_333x251.jpg": lambda: encode(
        Image.fromarray(gradient(333, 251, 1)), "JPEG", quality=90
    ),
    "rgb_444_120x90.jpg": lambda: encode(
        Image.fromarray(gradient(120, 90, 2)), "JPEG", quality=95, subsampling=0
    ),
    "progressive_200x150.jpg": lambda: encode(
        Image.fromarray(gradient(200, 150, 3)), "JPEG", quality=85, progressive=True
    ),
    "cmyk_80x64.jpg": lambda: encode(
        Image.fromarray(gradient(80, 64, 4)).convert("CMYK"), "JPEG"
    ),
    "exif_orientation6_64x48.jpg": exif_rotated_jpeg,
    "gray_64x64.png": lambda: encode(
        Image.fromarray(gradient(64, 64, 5)[..., 0]), "PNG"
    ),
    "rgba_80x60.png": lambda: encode(
        Image.fromarray(gradient(80, 60, 6, channels=4)), "PNG"
    ),
    "palette_transparency_70x50.png": lambda: encode(
        Image.fromarray(gradient(70, 50, 10)).convert("P"), "PNG", transparency=0
    ),
    "gray16_40x30.png": lambda: encode(
        Image.fromarray((gradient(40, 30, 11)[..., 0].astype(np.uint16) * 257)), "PNG"
    ),
    "animated_48x40.gif": animated_gif,
    "lossy_90x70.webp": lambda: encode(
        Image.fromarray(gradient(90, 70, 12)), "WEBP", quality=80
    ),
    "lossless_alpha_60x40.webp": lambda: encode(
        Image.fromarray(gradient(60, 40, 13, channels=4)), "WEBP", lossless=True
    ),
    "rgb_50x40.bmp": lambda: encode(Image.fromarray(gradient(50, 40, 14)), "BMP"),
    "tiny_10x10.png": lambda: encode(Image.fromarray(gradient(10, 10, 15)), "PNG"),
    "large_800x600.jpg": lambda: encode(
        Image.fromarray(gradient(800, 600, 16, noise=0)), "JPEG", quality=80
    ),
    "aspect_201_4020x20.png": lambda: encode(
        Image.fromarray(gradient(4020, 20, 17, noise=0)), "PNG"
    ),
}


# Divergences inside Dynamo; tests/mm_parity.rs fails once one is fixed.
KNOWN_GAPS = {
    "gray16_40x30.png": "Dynamo rejects 16-bit samples, which PIL's convert('RGB') clips",
}


def corpus_bytes(name):
    path = IMAGES / name
    if not path.exists():
        IMAGES.mkdir(parents=True, exist_ok=True)
        path.write_bytes(CORPUS[name]())
    return path.read_bytes()


# Shapes the resize kernels treat differently: upscale to min_pixels, an odd
# size, a non-RGB source, and a downscale.
RESIZE_SHAPES = [
    "tiny_10x10.png",
    "rgb_333x251.jpg",
    "rgba_80x60.png",
    "large_800x600.jpg",
]


# Per family: the chat template's ids around one image placeholder, and
# `--mm-process-config` image overrides it mirrors.
FAMILIES = {
    "qwen_vl": (
        ("vision_start_token_id", "vision_end_token_id"),
        {"min_pixels": 64 * 64, "max_pixels": 320 * 320},
    ),
    "glm5_next": (
        ("image_start_token_id", "image_end_token_id"),
        {"min_image_tokens": 4, "max_image_tokens": 64},
    ),
}


def cases(formats, family):
    """(name, images, backend, image_overrides) per case.

    The backend only picks the resize kernel and decoding is the same for a
    family's models, so the format sweep (`formats`) is needed for one model each.
    """
    for name in CORPUS if formats else RESIZE_SHAPES:
        yield name, [name], "auto", None
    if formats:
        for name in RESIZE_SHAPES:
            yield f"{name} pil", [name], "pil", None
    yield "two images", ["rgb_96x80.png", "rgb_333x251.jpg"], "auto", None
    yield "pixel overrides", ["large_800x600.jpg"], "auto", FAMILIES[family][1]


class PythonPath:
    """SGLang's processor over one checkpoint, one backend and one config."""

    def __init__(self, model_dir, backend, image_overrides):
        mm_process_config = {"image": image_overrides} if image_overrides else {}
        server_args = SimpleNamespace(
            model_impl="sglang",
            mm_feature_transport="cpu",
            mm_enable_dp_encoder=False,
            image_processor_backend=backend,
            disable_fast_image_processor=False,
            skip_tokenizer_init=False,
            mm_preprocess_cache_size_mb=0,
            trust_mm_content_hashes=False,
            tp_size=1,
            dist_init_addr=None,
            mm_process_config=mm_process_config,
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
                mm_process_config=mm_process_config,
                image_processor_backend=backend,
            ),
            role="tokenizer",
        )
        self.hf_config = AutoConfig.from_pretrained(model_dir)
        processor = get_processor(str(model_dir), image_processor_backend=backend)
        import_processors("sglang.srt.multimodal.processors")
        processor_cls = get_mm_processor_cls(self.hf_config)
        self.processor = processor_cls(
            self.hf_config, server_args, processor, None, skip_mm_pool=True
        )
        # The CPU path: PIL decode (read by a classmethod), and CPU image kernels.
        processor_cls.gpu_image_decode = False
        self.processor._fast_image_processor_device = lambda _processor: "cpu"
        self.spec = self.rust_spec(server_args, processor)

    def rust_spec(self, server_args, processor):
        """The Dynamo `ProcessorSpec` the Rust server gets, from SGLang's own gate."""
        # Skip __init__: it would load the processor again.
        host = RustMmProcessor.__new__(RustMmProcessor)
        host.model_config = SimpleNamespace(hf_config=self.hf_config)
        host._processor = processor
        host.server_args = server_args
        spec = host.resolve_spec()
        assert spec is not None, "SGLang's Rust MM gate rejects this checkpoint"
        return json.loads(spec.rust_json())

    def close(self):
        self.processor.io_executor.shutdown()
        self.processor.cpu_executor.shutdown()
        reset_context()

    def prompt(self, num_images):
        config = self.hf_config
        start, end = FAMILIES[self.spec["family"]][0]
        block = [getattr(config, start), config.image_token_id, getattr(config, end)]
        text = self.processor._processor.tokenizer.encode(
            "Describe.", add_special_tokens=False
        )
        return block * num_images + text

    def run(self, input_ids, images):
        output = asyncio.run(
            self.processor.process_mm_data_async(
                image_data=images,
                input_text=input_ids,
                request_obj=SimpleNamespace(
                    video_data=None, audio_data=None, rid="parity"
                ),
            )
        )
        features = [item.feature.detach().cpu().numpy() for item in output.mm_items]
        return {
            "input_ids_sha256": sha256(np.asarray(output.input_ids, dtype="<i4")),
            "num_input_ids": len(output.input_ids),
            "offsets": [list(item.offsets[0]) for item in output.mm_items],
            "image_grid_thw": [
                item.image_grid_thw.flatten().tolist() for item in output.mm_items
            ],
            "feature_shapes": [list(feature.shape) for feature in features],
            "feature_sha256": [sha256(feature.astype("<f4")) for feature in features],
            "mrope_sha256": sha256(output.mrope_positions.cpu().numpy().astype("<i8")),
            "mrope_delta": int(output.mrope_position_delta.item()),
        }


def dump(fixture):
    """The fixture's fields one per line, and each case on one line."""
    lines = [
        f"  {json.dumps(key)}: {json.dumps(value)}"
        for key, value in fixture.items()
        if key != "cases"
    ]
    cases = ",\n".join(f"    {json.dumps(case)}" for case in fixture["cases"])
    return "{\n" + ",\n".join(lines) + ',\n  "cases": [\n' + cases + "\n  ]\n}\n"


def sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def record(path, input_ids, images):
    try:
        return {"expected": path.run(input_ids, images)}
    except Exception as error:  # noqa: BLE001 -- the rejection is the recorded result
        return {"error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--revision")
    parser.add_argument(
        "--formats",
        action="store_true",
        help="also cover every image format and the PIL backend (one model is enough)",
    )
    args = parser.parse_args()
    revision = args.revision or HfApi().model_info(args.model).sha
    model_dir = Path(
        snapshot_download(args.model, revision=revision, allow_patterns=PROCESSOR_FILES)
    )

    fixture = {
        "model": args.model,
        "revision": revision,
        # The PIL backend's bicubic output changes across Pillow releases.
        "pillow": PIL.__version__,
        "cases": [],
    }
    model_type = AutoConfig.from_pretrained(model_dir).model_type
    family = next(f.name for f in RUST_MM_FAMILIES if model_type in f.model_types)
    paths = {}
    for name, image_names, backend, overrides in cases(args.formats, family):
        key = (backend, repr(overrides))
        if key not in paths:
            paths[key] = PythonPath(model_dir, backend, overrides)
        path = paths[key]
        input_ids = path.prompt(len(image_names))
        images = [corpus_bytes(image) for image in image_names]
        case = {
            "name": name,
            "images": image_names,
            "image_processor_backend": backend,
            "image_overrides": overrides,
            "spec": path.spec,
            "input_ids": input_ids,
            **record(path, input_ids, images),
        }
        gaps = [KNOWN_GAPS[image] for image in image_names if image in KNOWN_GAPS]
        if gaps:
            case["known_gap"] = "; ".join(gaps)
        fixture["cases"].append(case)
        print(name, "error" if "error" in case else "ok")
    for path in paths.values():
        path.close()

    out = FIXTURES / f"{args.model.split('/')[-1].lower()}.json"
    out.write_text(dump(fixture))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
