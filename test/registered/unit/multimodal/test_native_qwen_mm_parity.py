"""Parity tests for the native (Rust) Qwen VL multimodal pipeline.

The Rust tokenizer-manager path processes image-only Qwen requests entirely in
Rust (``rust/sglang-mm``): smart_resize → bicubic resize → normalize →
patchify, plus placeholder expansion and image-only M-RoPE. These tests pin
that pipeline against the references it must match:

* HF ``Qwen2VLImageProcessor`` — exact ``image_grid_thw``, ``pixel_values``
  within a small tolerance (PIL-style vs torchvision bicubic).
* ``MRotaryEmbedding.get_rope_index`` — exact positions + delta.

Runs on CPU. Skipped when the ``sglang.srt.multimodal._core`` Rust extension
was not built (e.g. SGLANG_BUILD_RUST_EXTS excludes it).
"""

import io
import json
import unittest

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

try:
    from sglang.srt.multimodal import _core

    HAS_CORE = hasattr(_core, "qwen_vl")
except ImportError:
    HAS_CORE = False

# One processor config per Qwen generation (matching each family's
# preprocessor_config.json values).
PROCESSOR_CONFIGS = {
    "qwen2_vl": dict(
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=56 * 56,
        max_pixels=28 * 28 * 1280,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    ),
    # Qwen2.5-VL / Qwen3-VL / Qwen3.5 share the 0.5 normalization; patch size
    # differs (14 vs 16).
    "qwen2_5_vl": dict(
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=56 * 56,
        max_pixels=28 * 28 * 1280,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ),
    "qwen3_5": dict(
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        min_pixels=65536,
        max_pixels=16777216,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ),
}

IMAGE_SIZES = [(640, 480), (1024, 683), (50, 40), (300, 301)]


def make_image(w: int, h: int, seed: int) -> np.ndarray:
    """Gradient + noise: smooth enough to be realistic, noisy enough to
    exercise the resize kernel."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack(
        [
            x * 255 / max(w - 1, 1),
            y * 255 / max(h - 1, 1),
            (x + y) * 127 / max(w + h - 2, 1),
        ],
        axis=-1,
    )
    noise = rng.integers(0, 40, size=(h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def spec_json(cfg: dict) -> str:
    return json.dumps({"family": "qwen_vl", "image_token_id": 0, **cfg})


@unittest.skipUnless(HAS_CORE, "sglang-mm Rust extension not built")
class TestNativeQwenImageParity(CustomTestCase):
    """Rust preprocess vs HF Qwen2VLImageProcessor on synthetic images."""

    def test_pixel_values_and_grids_match_hf(self):
        from PIL import Image
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor,
        )

        for name, cfg in PROCESSOR_CONFIGS.items():
            hf = Qwen2VLImageProcessor(**cfg)
            spec = spec_json(cfg)
            for i, (w, h) in enumerate(IMAGE_SIZES):
                with self.subTest(config=name, size=(w, h)):
                    img = Image.fromarray(make_image(w, h, seed=i))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")

                    pv_rs, grid_rs = _core.qwen_vl.preprocess(buf.getvalue(), spec)
                    out = hf(images=[img], return_tensors="pt")
                    grid_hf = tuple(out["image_grid_thw"][0].tolist())
                    self.assertEqual(grid_rs, grid_hf)

                    pv_hf = out["pixel_values"].float().numpy()
                    pv_rs = np.asarray(pv_rs).reshape(pv_hf.shape)
                    diff = np.abs(pv_rs - pv_hf)
                    # PIL-style fixed-point bicubic vs torchvision float
                    # bicubic: bounded by ~2 u8 steps through normalization.
                    self.assertLess(diff.max(), 0.06)
                    self.assertLess(diff.mean(), 1e-3)

    def test_smart_resize_matches_python(self):
        from sglang.srt.multimodal.processors.qwen_vl import smart_resize

        cases = [
            (1365, 2048, 28, 3136, 12845056),
            (100, 100, 28, 3136, 12845056),
            (3000, 4000, 28, 3136, 1003520),  # downscale branch
            (20, 20, 28, 3136, 12845056),  # upscale branch
            (1365, 2048, 32, 65536, 16777216),  # Qwen3.5 factors
            (4000, 48, 32, 4, 1 << 30),  # banker's-rounding tie (48/32 = 1.5)
        ]
        for h, w, factor, min_px, max_px in cases:
            with self.subTest(case=(h, w, factor)):
                self.assertEqual(
                    _core.qwen_vl.smart_resize_py(h, w, factor, min_px, max_px),
                    smart_resize(h, w, factor, min_px, max_px),
                )


@unittest.skipUnless(HAS_CORE, "sglang-mm Rust extension not built")
class TestNativeQwenMropeParity(CustomTestCase):
    """Rust image-only M-RoPE vs MRotaryEmbedding.get_rope_index."""

    VS, PAD, VE = 900, 901, 902

    def _build_ids(self, grids, merge=2, n_text=5):
        ids, items = [], []
        for k, (t, h, w) in enumerate(grids):
            ids += [10 + k] * n_text
            ids.append(self.VS)
            start = len(ids)
            ids += [self.PAD] * (t * (h // merge) * (w // merge))
            items.append((start, len(ids) - 1, t, h, w))
            ids.append(self.VE)
        ids += [50] * 3
        return ids, items

    def test_positions_and_delta_match_get_rope_index(self):
        from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

        for model_type in ["qwen2_vl", "qwen3_vl", "qwen3_5"]:
            for grids in [[(1, 4, 6)], [(1, 8, 4), (1, 6, 6)]]:
                with self.subTest(model_type=model_type, grids=grids):
                    ids, items = self._build_ids(grids)
                    pos_rs, delta_rs = _core.qwen_vl.mrope_image_only_py(
                        len(ids), items, 2
                    )
                    pos_rs = np.asarray(pos_rs).reshape(3, -1)

                    ref_pos, ref_delta = MRotaryEmbedding.get_rope_index(
                        spatial_merge_size=2,
                        image_token_id=self.PAD,
                        video_token_id=903,
                        vision_start_token_id=self.VS,
                        model_type=model_type,
                        tokens_per_second=None,
                        input_ids=torch.tensor(ids, dtype=torch.long).unsqueeze(0),
                        image_grid_thw=torch.tensor(grids, dtype=torch.long),
                        video_grid_thw=None,
                    )
                    np.testing.assert_array_equal(pos_rs, ref_pos.squeeze(1).numpy())
                    self.assertEqual(delta_rs, int(ref_delta.flatten()[0]))


if __name__ == "__main__":
    unittest.main()
