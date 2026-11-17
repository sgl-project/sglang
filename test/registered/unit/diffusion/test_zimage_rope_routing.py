"""Z-Image RoPE routing: which platforms fold RoPE into the qk-norm call."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.layers.rotary_embedding import RotaryEmbedding
from sglang.multimodal_gen.runtime.models.dits import zimage as zimage_module
from sglang.multimodal_gen.runtime.models.dits.zimage import (
    ZImageAttention,
    ZImageRMSNorm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEADS = 2
HEAD_DIM = 16
SEQ_LEN = 6

# Every sequence-parallel accessor reads this one group object, so a single-rank
# stub gives the attention block the degrees it asks for without a process group.
SINGLE_RANK_SP_GROUP = SimpleNamespace(
    world_size=1,
    rank_in_group=0,
    ulysses_world_size=1,
    ulysses_rank=0,
    ring_world_size=1,
    ring_rank=0,
)


class _FusedQKV(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)

    def forward(self, hidden_states):
        return hidden_states @ self.weight, None


class _CaptureAttention(torch.nn.Module):
    """Stand-in for the attention kernel that keeps the Q/K it was handed."""

    def __init__(self):
        super().__init__()
        self.query = None
        self.key = None

    def forward(self, q, k, v, **kwargs):
        self.query, self.key = q, k
        return v


class _PassThrough(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states, None


def _build_attention() -> ZImageAttention:
    generator = torch.Generator().manual_seed(7)
    attention = ZImageAttention.__new__(ZImageAttention)
    torch.nn.Module.__init__(attention)
    attention.qk_norm = True
    attention.head_dim = HEAD_DIM
    attention.local_num_heads = HEADS
    attention.local_num_kv_heads = HEADS
    attention.enable_zimage_qk_fusion = True
    attention.use_fused_qkv = True
    attention.norm_q = ZImageRMSNorm(HEAD_DIM)
    attention.norm_k = ZImageRMSNorm(HEAD_DIM)
    for norm in (attention.norm_q, attention.norm_k):
        with torch.no_grad():
            norm.weight.copy_(torch.rand(HEAD_DIM, generator=generator) + 0.5)
    attention.rotary_emb = RotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=HEAD_DIM,
        is_neox_style=False,
        use_precomputed_cache=False,
    )
    dim = HEADS * HEAD_DIM
    attention.to_qkv = _FusedQKV(torch.randn(dim, 3 * dim, generator=generator))
    attention.attn = _CaptureAttention()
    attention.to_out = torch.nn.ModuleList([_PassThrough()])
    return attention


def _rope_tables():
    generator = torch.Generator().manual_seed(11)
    cos = torch.randn(SEQ_LEN, HEAD_DIM // 2, generator=generator)
    sin = torch.randn(SEQ_LEN, HEAD_DIM // 2, generator=generator)
    return torch.cat([cos, sin], dim=-1), torch.complex(cos, sin)


class TestZImageRopeRouting(CustomTestCase):
    def setUp(self):
        sp_group = patch.object(parallel_state, "_SP", SINGLE_RANK_SP_GROUP)
        sp_group.start()
        self.addCleanup(sp_group.stop)
        self.attention = _build_attention()
        self.cos_sin_cache, self.freqs_complex = _rope_tables()
        self.hidden_states = torch.randn(
            1, SEQ_LEN, HEADS * HEAD_DIM, generator=torch.Generator().manual_seed(3)
        )

    def _run(self):
        with torch.no_grad():
            self.attention(
                self.hidden_states,
                rope_cos_sin_cache=self.cos_sin_cache,
                rope_positions=None,
                rope_freqs_complex=self.freqs_complex,
            )
        return self.attention.attn.query, self.attention.attn.key

    def test_rocm_folds_a_2d_table_into_the_qk_norm_call(self):
        for is_rocm, folded in ((False, False), (True, True)):
            with (
                patch.object(zimage_module, "_is_rocm", is_rocm),
                patch.object(
                    zimage_module,
                    "apply_qk_norm_with_optional_rope",
                    wraps=zimage_module.apply_qk_norm_with_optional_rope,
                ) as qk_norm_rope,
            ):
                self._run()
            self.assertEqual(qk_norm_rope.called, folded)
            if folded:
                self.assertIs(
                    qk_norm_rope.call_args.kwargs["cos_sin_cache"], self.cos_sin_cache
                )

    def test_both_routes_apply_the_same_rotation(self):
        with patch.object(zimage_module, "_is_rocm", False):
            query, key = (tensor.clone() for tensor in self._run())
        with patch.object(zimage_module, "_is_rocm", True):
            folded_query, folded_key = self._run()
        torch.testing.assert_close(folded_query, query, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(folded_key, key, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
