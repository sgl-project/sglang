# SPDX-License-Identifier: Apache-2.0
"""SM90-specific invariants for SubBlock sparse attention."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_sm90 = unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "requires SM90 (Hopper)",
)


@requires_sm90
class TestSubBlockSparseSM90(CustomTestCase):
    def test_64x64_routing_mask_uses_matching_compute_tile(self):
        """A tile spanning routing rows would apply one row's mask to another row."""
        from sglang.kernels.ops.attention.flash_attn.cute.interface import (
            _tile_size_fwd_sm90,
        )

        config = _tile_size_fwd_sm90(
            head_dim=128,
            head_dim_v=128,
            is_causal=False,
            is_local=False,
            sparse_block_size_q=64,
            sparse_block_size_kv=64,
        )

        self.assertEqual(config.m_block_size, 64)
        self.assertEqual(config.n_block_size, 64)

    def test_64x64_special_case_is_limited_to_head_dim_128(self):
        from sglang.kernels.ops.attention.flash_attn.cute.interface import (
            _tile_size_fwd_sm90,
        )

        config = _tile_size_fwd_sm90(
            head_dim=96,
            head_dim_v=96,
            is_causal=False,
            is_local=False,
            sparse_block_size_q=64,
            sparse_block_size_kv=64,
        )

        self.assertEqual(config.m_block_size, 128)
        self.assertEqual(config.n_block_size, 128)


if __name__ == "__main__":
    unittest.main(verbosity=3)
