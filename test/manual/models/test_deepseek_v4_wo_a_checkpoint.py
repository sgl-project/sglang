"""Unit tests for DeepSeek V4 `_checkpoint_has_fp8_wo_a` checkpoint inspection.

Covers the wo_a dtype-detection logic added for checkpoints (e.g.
DeepSeek-V4-Flash-FP8) that keep ``wo_a`` in BF16 while every other
projection is FP8-quantized:

- non-local / nonexistent model path keeps the historical "assume FP8"
- local dir with model.safetensors.index.json: BF16 wo_a -> False, FP8 -> True
- local dir without an index (flat safetensors scan): BF16 -> False, FP8 -> True
- no wo_a tensors at all -> True (assume FP8)
"""

import json
import os
import tempfile
import unittest

from safetensors.torch import save_file

from sglang.test.test_utils import CustomTestCase
from sglang.test_internal.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, num_gpus=1)

import torch

from sglang.srt.models.deepseek_v4 import _checkpoint_has_fp8_wo_a


def _shard(name: str, dtype: torch.dtype) -> dict:
    return {name: torch.zeros((8, 8), dtype=dtype)}


class TestCheckpointHasFp8WoA(CustomTestCase):
    def setUp(self):
        # The function is lru_cached; clear between tests so each tmpdir is
        # inspected fresh.
        _checkpoint_has_fp8_wo_a.cache_clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.model_path = self._tmp.name

    def tearDown(self):
        _checkpoint_has_fp8_wo_a.cache_clear()
        self._tmp.cleanup()

    # ------------------------------------------------------------------
    # Non-local / missing paths keep the historical "assume FP8" behavior.
    # ------------------------------------------------------------------
    def test_nonexistent_path_assumes_fp8(self):
        self.assertTrue(
            _checkpoint_has_fp8_wo_a(os.path.join(self.model_path, "no-such-dir"))
        )

    def test_hf_repo_id_assumes_fp8(self):
        # A repo id like "deepseek-ai/DeepSeek-V4" is not a directory path.
        self.assertTrue(_checkpoint_has_fp8_wo_a("deepseek-ai/DeepSeek-V4-Flash"))

    # ------------------------------------------------------------------
    # Local dir with model.safetensors.index.json (sharded checkpoint).
    # ------------------------------------------------------------------
    def test_index_bf16_wo_a_is_not_fp8(self):
        save_file(
            _shard("model.layers.0.self_attn.wo_a.weight", torch.bfloat16),
            os.path.join(self.model_path, "model-00001-of-00002.safetensors"),
        )
        save_file(
            _shard("model.layers.0.self_attn.wo_b.weight", torch.float8_e4m3fn),
            os.path.join(self.model_path, "model-00002-of-00002.safetensors"),
        )
        with open(
            os.path.join(self.model_path, "model.safetensors.index.json"), "w"
        ) as f:
            json.dump(
                {
                    "weight_map": {
                        "model.layers.0.self_attn.wo_a.weight": "model-00001-of-00002.safetensors",
                        "model.layers.0.self_attn.wo_b.weight": "model-00002-of-00002.safetensors",
                    }
                },
                f,
            )
        # The wo_a lives in shard 2 by weight_map order but is BF16 there.
        self.assertFalse(_checkpoint_has_fp8_wo_a(self.model_path))

    def test_index_fp8_wo_a_is_fp8(self):
        save_file(
            _shard("model.layers.0.self_attn.wo_a.weight", torch.float8_e4m3fn),
            os.path.join(self.model_path, "model-00001-of-00002.safetensors"),
        )
        save_file(
            _shard("model.layers.0.self_attn.wo_b.weight", torch.bfloat16),
            os.path.join(self.model_path, "model-00002-of-00002.safetensors"),
        )
        with open(
            os.path.join(self.model_path, "model.safetensors.index.json"), "w"
        ) as f:
            json.dump(
                {
                    "weight_map": {
                        "model.layers.0.self_attn.wo_a.weight": "model-00001-of-00002.safetensors",
                        "model.layers.0.self_attn.wo_b.weight": "model-00002-of-00002.safetensors",
                    }
                },
                f,
            )
        self.assertTrue(_checkpoint_has_fp8_wo_a(self.model_path))

    def test_index_without_wo_a_assumes_fp8(self):
        save_file(
            _shard("model.layers.0.self_attn.wo_b.weight", torch.bfloat16),
            os.path.join(self.model_path, "model-00001-of-00001.safetensors"),
        )
        with open(
            os.path.join(self.model_path, "model.safetensors.index.json"), "w"
        ) as f:
            json.dump(
                {
                    "weight_map": {
                        "model.layers.0.self_attn.wo_b.weight": "model-00001-of-00001.safetensors"
                    }
                },
                f,
            )
        self.assertTrue(_checkpoint_has_fp8_wo_a(self.model_path))

    # ------------------------------------------------------------------
    # Local dir without an index: flat scan over *.safetensors files.
    # ------------------------------------------------------------------
    def test_flat_scan_bf16_wo_a_is_not_fp8(self):
        # Sorted order: "b" shard (no wo_a) is scanned first, "a" shard holds
        # the BF16 wo_a — the scan must continue past the miss.
        save_file(
            {"model.norm.weight": torch.zeros((8,), dtype=torch.bfloat16)},
            os.path.join(self.model_path, "1-no-wo-a.safetensors"),
        )
        save_file(
            _shard("model.layers.0.self_attn.wo_a.weight", torch.bfloat16),
            os.path.join(self.model_path, "2-has-wo-a.safetensors"),
        )
        self.assertFalse(_checkpoint_has_fp8_wo_a(self.model_path))

    def test_flat_scan_fp8_wo_a_is_fp8(self):
        save_file(
            _shard("model.layers.0.self_attn.wo_a.weight", torch.float8_e4m3fn),
            os.path.join(self.model_path, "model-00001.safetensors"),
        )
        self.assertTrue(_checkpoint_has_fp8_wo_a(self.model_path))

    def test_flat_scan_without_wo_a_assumes_fp8(self):
        save_file(
            {
                "model.layers.0.self_attn.wo_b.weight": torch.zeros(
                    (8, 8), dtype=torch.bfloat16
                )
            },
            os.path.join(self.model_path, "model-00001.safetensors"),
        )
        self.assertTrue(_checkpoint_has_fp8_wo_a(self.model_path))

    def test_empty_dir_assumes_fp8(self):
        self.assertTrue(_checkpoint_has_fp8_wo_a(self.model_path))

    def test_lru_cache_keyed_by_path(self):
        # Same path inspected twice returns the cached (unchanged) result.
        save_file(
            _shard("model.layers.0.self_attn.wo_a.weight", torch.bfloat16),
            os.path.join(self.model_path, "model-00001.safetensors"),
        )
        self.assertFalse(_checkpoint_has_fp8_wo_a(self.model_path))
        self.assertFalse(_checkpoint_has_fp8_wo_a(self.model_path))


if __name__ == "__main__":
    unittest.main()
