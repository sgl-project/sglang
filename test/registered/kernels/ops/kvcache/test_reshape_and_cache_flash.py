"""reshape_and_cache_flash checks, including K and V with different head sizes (MiMo-V2: K 192, V 128)."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")

SENTINEL = 7.0


def _reference(k, v, k_cache, v_cache, slots):
    """Scatter K and V by cache slot; negative slots are skipped."""
    k_ref, v_ref = k_cache.clone(), v_cache.clone()
    block_size = k_cache.shape[1]
    keep = slots >= 0
    slots = slots[keep].long()
    k_ref[slots // block_size, slots % block_size] = k[keep]
    v_ref[slots // block_size, slots % block_size] = v[keep]
    return k_ref, v_ref


@unittest.skipUnless(torch.cuda.is_available(), "Triton kernels require a GPU")
class TestReshapeAndCacheFlash(CustomTestCase):
    def _check(self, num_tokens, num_heads, k_dim, v_dim, block_size, swa, token_pad=0):
        from sglang.kernels.ops.kvcache.cache_ops import launch_reshape_and_cache_flash

        torch.manual_seed(0)
        dev = "cuda"
        dtype = torch.bfloat16
        full_slots = 2 * num_tokens

        k = torch.randn(num_tokens, num_heads, k_dim, dtype=dtype, device=dev)
        # V as a column slice of a wider row, as when it is split out of a fused QKV output.
        v_rows = torch.randn(
            num_tokens, num_heads * v_dim + token_pad, dtype=dtype, device=dev
        )
        v = v_rows[:, : num_heads * v_dim].view(num_tokens, num_heads, v_dim)

        loc = torch.randperm(full_slots, device=dev)[:num_tokens]
        loc[::5] = -1
        if swa:
            cache_slots = num_tokens + block_size
            table = torch.full((full_slots,), -1, dtype=torch.int64, device=dev)
            table[loc[loc >= 0]] = torch.randperm(cache_slots, device=dev)[
                : int((loc >= 0).sum())
            ]
            slots = torch.where(loc >= 0, table[loc.clamp_min(0)], loc)
        else:
            cache_slots, table, slots = full_slots, None, loc
        num_blocks = (cache_slots + block_size - 1) // block_size

        k_cache = torch.full(
            (num_blocks, block_size, num_heads, k_dim),
            SENTINEL,
            dtype=dtype,
            device=dev,
        )
        v_cache = torch.full(
            (num_blocks, block_size, num_heads, v_dim),
            SENTINEL,
            dtype=dtype,
            device=dev,
        )
        k_ref, v_ref = _reference(k, v, k_cache, v_cache, slots)

        launch_reshape_and_cache_flash(k, v, k_cache, v_cache, loc, table)

        torch.testing.assert_close(k_cache, k_ref, rtol=0, atol=0)
        torch.testing.assert_close(v_cache, v_ref, rtol=0, atol=0)

    def test_value_head_smaller_than_key(self):
        for num_heads in (1, 8):
            for block_size in (1, 16):
                with self.subTest(num_heads=num_heads, block_size=block_size):
                    self._check(64, num_heads, 192, 128, block_size, swa=False)

    def test_value_head_smaller_than_key_swa_remap(self):
        for num_heads in (1, 8):
            for block_size in (1, 16):
                with self.subTest(num_heads=num_heads, block_size=block_size):
                    self._check(64, num_heads, 192, 128, block_size, swa=True)

    def test_value_token_stride_wider_than_row(self):
        self._check(64, 4, 192, 128, 1, swa=True, token_pad=320)

    def test_equal_head_sizes(self):
        for swa in (False, True):
            with self.subTest(swa=swa):
                self._check(64, 8, 128, 128, 1, swa=swa)


if __name__ == "__main__":
    unittest.main()
