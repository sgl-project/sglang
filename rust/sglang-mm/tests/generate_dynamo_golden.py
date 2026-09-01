"""Generate golden fixtures for dynamo-mm-preprocessor's `qwen_vl_golden.rs`.

The dynamo crate's CI has no Python/HF, so its end-to-end test replays these
committed fixtures. Expected values come from the mirrored HF image processor
and SGLang's `MRotaryEmbedding.get_rope_index` — never from the crate under
test — and the script cross-checks the built Rust extension against them
before writing, so a bad fixture cannot land.

Usage (from the sglang repo root, extension built):
    PYTHONPATH=python python3 rust/sglang-mm/tests/generate_dynamo_golden.py \
        /personal/frontend-crates/mm-preprocessor/tests/fixtures/qwen_vl
"""

from __future__ import annotations

import io
import json
import os
import sys
from typing import TYPE_CHECKING

import msgspec
import numpy as np
import torch
from PIL import Image

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.rust_extensions import _multimodal

if TYPE_CHECKING:
    from transformers.image_processing_utils import BaseImageProcessor

Config = dict[str, int | list[float]]
Grid = tuple[int, int, int]

IMAGE_TOKEN_ID: int = 900
VISION_START_ID: int = 901
VISION_END_ID: int = 902

QWEN2_VL: Config = dict(
    patch_size=14,
    merge_size=2,
    temporal_patch_size=2,
    min_pixels=56 * 56,
    max_pixels=28 * 28 * 1280,
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711],
)
# A tight pixel budget so the downscale branch stays fixture-sized.
TINY_MAX: Config = dict(QWEN2_VL, max_pixels=112 * 112)

# (name, resample, config, image (width, height) list). One placeholder per
# image: [7, <vs>, <pad>, <ve>, ...] — the wrapper ids are plain text to the
# driver but let `get_rope_index` locate the spans.
CASES: list[tuple[str, str, Config, list[tuple[int, int]]]] = [
    ("aten_upscale", "aten_u8", QWEN2_VL, [(40, 40)]),
    ("aten_downscale", "aten_u8", TINY_MAX, [(300, 200)]),
    ("pil_round", "pil", QWEN2_VL, [(100, 76)]),
    ("aten_multi", "aten_u8", QWEN2_VL, [(40, 40), (56, 56)]),
]


class GoldenCase(msgspec.Struct, kw_only=True):
    prompt_ids: list[int]
    input_ids: list[int]
    grids: list[Grid]
    offsets: list[tuple[int, int]]
    # Decimal strings: JSON numbers cannot carry a full u64.
    hashes: list[str]
    features: np.ndarray
    mrope: np.ndarray
    mrope_delta: int


def make_image(width: int, height: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack(
        (x * 255 / max(width - 1, 1), y * 255 / max(height - 1, 1), (x + y) % 256),
        axis=-1,
    )
    arr = np.clip(base + rng.integers(0, 24, base.shape), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return buffer.getvalue()


def hf_image_processor(resample: str, config: Config) -> "BaseImageProcessor":
    if resample == "pil":
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
            Qwen2VLImageProcessorPil as cls,
        )
    else:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor as cls,
        )
    return cls(**config)


def expected_for(resample: str, config: Config, pngs: list[bytes]) -> GoldenCase:
    hf = hf_image_processor(resample, config)
    features: list[np.ndarray] = []
    grids: list[Grid] = []
    for png in pngs:
        out = hf(
            images=[Image.open(io.BytesIO(png)).convert("RGB")], return_tensors="pt"
        )
        features.append(out.pixel_values.numpy().astype(np.float32))
        grids.append(tuple(out.image_grid_thw[0].tolist()))

    prompt_ids: list[int] = [7]
    for _ in pngs:
        prompt_ids.extend((VISION_START_ID, IMAGE_TOKEN_ID, VISION_END_ID))
    prompt_ids.append(8)

    expanded: list[int] = []
    offsets: list[tuple[int, int]] = []
    merge: int = int(config["merge_size"])
    counts = iter([int(np.prod(g)) // merge**2 for g in grids])
    for tok in prompt_ids:
        if tok == IMAGE_TOKEN_ID:
            n = next(counts)
            offsets.append((len(expanded), len(expanded) + n - 1))
            expanded.extend([IMAGE_TOKEN_ID] * n)
        else:
            expanded.append(tok)

    mrope, delta = MRotaryEmbedding.get_rope_index(
        spatial_merge_size=merge,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=903,
        vision_start_token_id=VISION_START_ID,
        model_type="qwen2_vl",
        input_ids=torch.tensor(expanded).unsqueeze(0),
        image_grid_thw=torch.tensor(grids),
        video_grid_thw=None,
    )
    return GoldenCase(
        prompt_ids=prompt_ids,
        input_ids=expanded,
        grids=grids,
        offsets=offsets,
        # blake3 truncated to 8 BE bytes — pinned via the built extension so
        # the crate cannot drift the hash algorithm without a fixture diff.
        hashes=[str(_multimodal.common.content_hash(png)) for png in pngs],
        features=np.concatenate([f.reshape(-1) for f in features]),
        mrope=mrope.squeeze(1).numpy().astype(np.int64).reshape(-1),
        mrope_delta=int(delta.item()),
    )


def cross_check(spec: Config, pngs: list[bytes], expected: GoldenCase) -> None:
    ids, feats, grids, hashes, offsets, mrope, delta = _multimodal.qwen_vl.process_mm(
        expected.prompt_ids, [bytearray(p) for p in pngs], json.dumps(spec)
    )
    assert ids == expected.input_ids
    assert [tuple(g) for g in grids] == expected.grids
    assert offsets == expected.offsets
    assert [str(h) for h in hashes] == expected.hashes
    assert delta == expected.mrope_delta
    np.testing.assert_array_equal(np.asarray(feats), expected.features)
    np.testing.assert_array_equal(np.asarray(mrope), expected.mrope)


def main(out_root: str) -> None:
    for index, (name, resample, config, sizes) in enumerate(CASES):
        spec: Config = {
            "family": "qwen_vl",
            "image_token_id": IMAGE_TOKEN_ID,
            "resample": resample,
            **config,
        }
        pngs = [make_image(w, h, seed=10 + index * 8 + i) for i, (w, h) in enumerate(sizes)]
        expected = expected_for(resample, config, pngs)
        cross_check(spec, pngs, expected)

        case_dir = os.path.join(out_root, name)
        os.makedirs(case_dir, exist_ok=True)
        for i, png in enumerate(pngs):
            with open(os.path.join(case_dir, f"input_{i}.png"), "wb") as f:
                f.write(png)
        expected.features.astype("<f4").tofile(
            os.path.join(case_dir, "pixel_values.f32le")
        )
        expected.mrope.astype("<i8").tofile(os.path.join(case_dir, "mrope.i64le"))
        meta: dict[str, object] = {
            "spec": spec,
            "prompt_ids": expected.prompt_ids,
            "input_ids": expected.input_ids,
            "grids": [list(g) for g in expected.grids],
            "offsets": [list(o) for o in expected.offsets],
            "hashes": expected.hashes,
            "mrope_delta": expected.mrope_delta,
        }
        with open(os.path.join(case_dir, "case.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"  {case_dir}: {len(pngs)} image(s), {expected.features.size} f32")
    print("DYNAMO_GOLDEN_OK")


if __name__ == "__main__":
    main(sys.argv[1])
