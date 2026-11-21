import unittest

import torch
import triton

from sglang.kernels.ops.kvcache.pd_dcp_gather import (
    _copy_mla_rows_into_pack_kernel,
    copy_mla_rows_into_pack,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Production MLA shape: 2 layers x 1152 B/row.
ITEM_LEN = 1152
NUM_LAYERS = 2


def _reference_gather(src_layers, row_indices, item_len, pack):
    """Reference gather: pack[layer][i] = src_layers[layer][row_indices[i]]."""
    n = int(row_indices.numel())
    for layer, src in enumerate(src_layers):
        rows = src[row_indices.long()][:, :item_len]
        pack[layer * n * item_len : (layer + 1) * n * item_len] = rows.reshape(-1)


def _run_gather(n, item_len=ITEM_LEN, src_item_len=None, seed=0):
    """Run copy_mla_rows_into_pack and check byte-exactness vs the reference."""
    src_item_len = src_item_len or item_len
    g = torch.Generator(device="cuda").manual_seed(seed)
    src_layers = [
        torch.randint(
            0, 256, (n, src_item_len), dtype=torch.uint8, device="cuda", generator=g
        )
        for _ in range(NUM_LAYERS)
    ]
    row_indices = torch.randperm(n, device="cuda", generator=g).to(torch.int32)
    pack = torch.zeros(NUM_LAYERS * n * item_len, dtype=torch.uint8, device="cuda")
    copy_mla_rows_into_pack(
        [src.data_ptr() for src in src_layers],
        row_indices,
        pack,
        [item_len] * NUM_LAYERS,
        [src_item_len] * NUM_LAYERS,
    )
    expected = torch.zeros_like(pack)
    _reference_gather(src_layers, row_indices, item_len, expected)
    torch.testing.assert_close(pack, expected)


class TestPdDcpGather(CustomTestCase):
    def test_production_shape_small(self):
        _run_gather(1_000)

    def test_grid_y_limit_boundary(self):
        # 58,253 rows is the largest pack the upstream (layer, block) grid
        # ordering can launch; the swapped grid must handle it and beyond.
        _run_gather(58_253)

    def test_beyond_grid_y_limit(self):
        # The upstream kernel crashes here with "CUDA error: invalid argument".
        _run_gather(100_000)

    def test_large_pack(self):
        _run_gather(300_000)

    def test_strided_source_rows(self):
        # src rows wider than the copy width (src_row_stride != item_len).
        _run_gather(10_000, src_item_len=1536)

    def test_non_8_aligned_item_len_uses_byte_path(self):
        _run_gather(4_096, item_len=1001)

    def test_use_int64_branch_compiles_and_runs(self):
        # Direct launch of the int64-addressing branch on a small input.
        n = 2_000
        g = torch.Generator(device="cuda").manual_seed(1)
        src = torch.randint(
            0, 256, (n, ITEM_LEN), dtype=torch.uint8, device="cuda", generator=g
        )
        row_indices = torch.randperm(n, device="cuda", generator=g).to(torch.int32)
        pack = torch.zeros(n * ITEM_LEN, dtype=torch.uint8, device="cuda")
        metadata = torch.tensor(
            [src.data_ptr(), ITEM_LEN, 0, ITEM_LEN], dtype=torch.int64, device="cuda"
        )
        grid = (triton.cdiv(n * ITEM_LEN, 1024), 1)
        _copy_mla_rows_into_pack_kernel[grid](
            metadata, row_indices, pack, n, USE_INT64=True, BLOCK_SIZE=1024
        )
        expected = torch.zeros_like(pack)
        _reference_gather([src], row_indices, ITEM_LEN, expected)
        torch.testing.assert_close(pack, expected)


if __name__ == "__main__":
    unittest.main()
