import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _install_qkv_row_reorder,
)

QKV_ROWS = 12


def _install(reorder=lambda w: w.flip(0)):
    param = torch.zeros(QKV_ROWS)
    seen = []
    param.weight_loader = lambda p, loaded_weight: seen.append(loaded_weight)
    _install_qkv_row_reorder(param, reorder, QKV_ROWS)
    return param, seen


class TestMiniMaxH3QkvScaleReorder(unittest.TestCase):
    """MiniMax-H3 permutes the fused qkv weight's output rows on load.

    Quantization metadata indexed by output row (NVFP4 block scales, fp8
    per-channel scales) has to move with those rows, or every scale lands on the
    wrong row: the model loads and runs, and renders noise. Metadata that is not
    row-indexed must be left alone, so the row count is the gate.
    """

    def test_row_indexed_metadata_is_reordered(self):
        param, seen = _install()
        scales = torch.arange(QKV_ROWS * 2, dtype=torch.float32).reshape(QKV_ROWS, 2)
        param.weight_loader(param, scales)
        self.assertTrue(torch.equal(seen[0], scales.flip(0)))

    def test_metadata_with_another_row_count_is_passed_through(self):
        param, seen = _install()
        swizzled = torch.arange(QKV_ROWS, dtype=torch.float32).reshape(QKV_ROWS // 3, 3)
        param.weight_loader(param, swizzled)
        self.assertTrue(torch.equal(seen[0], swizzled))

    def test_per_tensor_scale_is_passed_through(self):
        param, seen = _install()
        scale = torch.tensor(0.5)
        param.weight_loader(param, scale)
        self.assertTrue(torch.equal(seen[0], scale))

    def test_rank_local_transform_follows_the_same_gate(self):
        # rank-local FSDP reorders before selecting its shard, so it needs the
        # transform rather than the wrapped loader.
        param, _ = _install()
        scales = torch.arange(QKV_ROWS * 2, dtype=torch.float32).reshape(QKV_ROWS, 2)
        self.assertTrue(
            torch.equal(param.rank_local_weight_transform(scales), scales.flip(0))
        )
        scale = torch.tensor(0.5)
        self.assertTrue(torch.equal(param.rank_local_weight_transform(scale), scale))


if __name__ == "__main__":
    unittest.main()
