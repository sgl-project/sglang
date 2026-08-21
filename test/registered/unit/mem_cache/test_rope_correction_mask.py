"""Layer-mask semantics of copy_kv_with_rope_correction.

Masked layers must be zeroed in the destination slots (K and V), and
unmasked layers must carry position-corrected donor content, in the same
call. Guards the contract the mask documentation describes: a provider
flagging a layer gets attenuation there without disturbing the copy on
any other layer.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.mem_cache.fuzzy_match.rope_correction import (
    copy_kv_with_rope_correction,
)
from sglang.test.test_utils import CustomTestCase

HEAD_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 3
POOL_SIZE = 64
MAX_POS = 512


def _cos_sin_cache() -> torch.Tensor:
    inv_freq = 1.0 / (
        10_000 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    freqs = torch.einsum(
        "i,j->ij", torch.arange(MAX_POS, dtype=torch.float32), inv_freq
    )
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _fake_pool():
    pool = SimpleNamespace()
    pool.layer_num = NUM_LAYERS
    pool.k_buffer = [
        torch.zeros(POOL_SIZE, NUM_HEADS, HEAD_DIM) for _ in range(NUM_LAYERS)
    ]
    pool.v_buffer = [
        torch.zeros(POOL_SIZE, NUM_HEADS, HEAD_DIM) for _ in range(NUM_LAYERS)
    ]
    return pool


class TestLayerMaskZeroing(CustomTestCase):
    def test_masked_layers_zero_and_unmasked_layers_copy(self):
        torch.manual_seed(5)
        n = 8
        cache = _cos_sin_cache()
        pool = _fake_pool()
        old_locs = torch.arange(0, n, dtype=torch.long)
        new_locs = torch.arange(n, 2 * n, dtype=torch.long)
        old_pos = torch.arange(100, 100 + n, dtype=torch.long)
        new_pos = torch.arange(20, 20 + n, dtype=torch.long)

        k_raw = torch.randn(NUM_LAYERS, n, NUM_HEADS, HEAD_DIM)
        v_ref = torch.randn(NUM_LAYERS, n, NUM_HEADS, HEAD_DIM)
        cos_old, sin_old = cache.index_select(0, old_pos).chunk(2, dim=-1)
        for layer in range(NUM_LAYERS):
            pool.k_buffer[layer][old_locs] = apply_rotary_emb(
                k_raw[layer], cos_old, sin_old, True
            )
            pool.v_buffer[layer][old_locs] = v_ref[layer]

        rotary = SimpleNamespace(
            cos_sin_cache=cache, is_neox_style=True, rotary_dim=HEAD_DIM
        )
        copy_kv_with_rope_correction(
            pool=pool,
            rotary_emb=rotary,
            old_locs=old_locs,
            new_locs=new_locs,
            old_positions=old_pos,
            new_positions=new_pos,
            layer_zero_mask=[False, True, False],
        )

        self.assertTrue(torch.all(pool.k_buffer[1][new_locs] == 0))
        self.assertTrue(torch.all(pool.v_buffer[1][new_locs] == 0))

        cos_new, sin_new = cache.index_select(0, new_pos).chunk(2, dim=-1)
        for layer in (0, 2):
            expected_k = apply_rotary_emb(k_raw[layer], cos_new, sin_new, True)
            torch.testing.assert_close(
                pool.k_buffer[layer][new_locs], expected_k, atol=1e-4, rtol=1e-4
            )
            torch.testing.assert_close(pool.v_buffer[layer][new_locs], v_ref[layer])

    def test_short_mask_leaves_trailing_layers_copied(self):
        torch.manual_seed(6)
        n = 4
        cache = _cos_sin_cache()
        pool = _fake_pool()
        old_locs = torch.arange(0, n, dtype=torch.long)
        new_locs = torch.arange(n, 2 * n, dtype=torch.long)
        pos = torch.arange(10, 10 + n, dtype=torch.long)
        for layer in range(NUM_LAYERS):
            pool.v_buffer[layer][old_locs] = torch.full(
                (n, NUM_HEADS, HEAD_DIM), float(layer + 1)
            )

        copy_kv_with_rope_correction(
            pool=pool,
            rotary_emb=SimpleNamespace(
                cos_sin_cache=cache, is_neox_style=True, rotary_dim=HEAD_DIM
            ),
            old_locs=old_locs,
            new_locs=new_locs,
            old_positions=pos,
            new_positions=pos,
            layer_zero_mask=[True],
        )

        self.assertTrue(torch.all(pool.v_buffer[0][new_locs] == 0))
        for layer in (1, 2):
            self.assertTrue(
                torch.all(pool.v_buffer[layer][new_locs] == float(layer + 1))
            )


if __name__ == "__main__":
    unittest.main()
