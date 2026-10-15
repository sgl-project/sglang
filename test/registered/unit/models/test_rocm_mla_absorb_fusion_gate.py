"""The gate that picks the fused q-absorb + RoPE + KV-write kernel.

Every term names a branch that one of the two kernels it replaces would
otherwise have taken, so the fused path is only equivalent where all of them
hold. A term quietly dropped here would route a shape the kernel does not
implement, and the wrong answer would look like a plausible one.

The caller carries one further term, ``not q_replicate_active``, which needs a
ForwardBatch to evaluate and so is not part of this predicate.
"""

import functools
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models.bailing_moe_v3 import DsV3MLA
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_CAN_FUSE = forward_mla_rocm._can_fuse_bmm_rope_cat_and_cache
_ABSORB_CONFIG = forward_mla_rocm._absorb_bmm_config


def _attn(**overrides):
    """An attention module every term accepts, before the caller breaks one."""
    attn = SimpleNamespace(
        w_kc=SimpleNamespace(dtype=torch.float8_e4m3fn),
        rotary_emb=object(),
        current_attention_backend="dsa",
        use_deep_gemm_bmm=False,
        kv_b_proj=SimpleNamespace(set_lora=False),
        _skip_rope_for_dsa_tilelang_fused=lambda: True,
    )
    for k, v in overrides.items():
        setattr(attn, k, v)
    return attn


def _patch_dcp(case, *, dcp_enabled: bool):
    """Give the gate a parallel context; the real one needs an initialized runtime."""
    parallel = SimpleNamespace(dcp_enabled=dcp_enabled)
    saved = forward_mla_rocm.get_parallel
    forward_mla_rocm.get_parallel = lambda: parallel
    case.addCleanup(setattr, forward_mla_rocm, "get_parallel", saved)
    return parallel


class TestFusedAbsorbGate(CustomTestCase):
    def setUp(self):
        # The first term is the platform, and it short-circuits everything else;
        # patch it so the remaining terms are reachable off gfx95.
        self._saved = forward_mla_rocm._use_aiter_gfx95
        forward_mla_rocm._use_aiter_gfx95 = True
        self.addCleanup(setattr, forward_mla_rocm, "_use_aiter_gfx95", self._saved)
        self._parallel = _patch_dcp(self, dcp_enabled=False)

    def test_dcp_turns_it_off(self):
        """DCP decode all-gathers q_nope_out, which this path never produces.

        Without this term the fused branch sets q_nope_out to None and
        all_gather_q_for_mla_decode dereferences it.
        """
        self._parallel.dcp_enabled = True
        self.assertFalse(_CAN_FUSE(_attn()))

    def test_takes_the_fused_path_when_every_term_holds(self):
        self.assertTrue(_CAN_FUSE(_attn()))

    def test_off_gfx95_nothing_else_is_consulted(self):
        forward_mla_rocm._use_aiter_gfx95 = False
        # An attn that would otherwise qualify, so only the platform can decide.
        self.assertFalse(_CAN_FUSE(_attn()))

    def test_each_term_alone_turns_it_off(self):
        # The kernel absorbs an fp8 weight, needs a rope to fuse, writes the
        # tilelang DSA cache layout, and has no LoRA or deep-gemm variant.
        for name, broken in (
            ("w_kc", SimpleNamespace(dtype=torch.bfloat16)),
            ("rotary_emb", None),
            ("current_attention_backend", "triton"),
            ("use_deep_gemm_bmm", True),
            ("kv_b_proj", SimpleNamespace(set_lora=True)),
            ("_skip_rope_for_dsa_tilelang_fused", lambda: False),
        ):
            with self.subTest(term=name):
                self.assertFalse(_CAN_FUSE(_attn(**{name: broken})))


class TestAbsorbBmmConfig(CustomTestCase):
    def setUp(self):
        self._saved = forward_mla_rocm._absorb_bmm_num_cu
        forward_mla_rocm._absorb_bmm_num_cu = 256
        self.addCleanup(
            setattr,
            forward_mla_rocm,
            "_absorb_bmm_num_cu",
            self._saved,
        )

    def test_decode_tiles(self):
        cases = (
            ((16, 1, 512), (16, 32, 1)),
            ((16, 48, 512), (16, 128, 1)),
            ((16, 200, 512), (32, 128, 2)),
        )
        for args, expected in cases:
            with self.subTest(args=args):
                config = _ABSORB_CONFIG(*args)
                self.assertEqual(
                    (
                        config["BLOCK_SIZE_M"],
                        config["BLOCK_SIZE_N"],
                        config["waves_per_eu"],
                    ),
                    expected,
                )

    def test_large_row_count_falls_back(self):
        self.assertIsNone(_ABSORB_CONFIG(16, 0, 512))
        self.assertIsNone(_ABSORB_CONFIG(16, 257, 512))


class TestRocmAbsorbCoreAttentionOutputGate(CustomTestCase):
    def test_bailing_gate_after_rocm_prepare_state_reaches_o_proj(self):
        """BailingMoeV3 appends its output gate after the ROCm prepare state;
        the ROCm absorb core must accept it and gate the input of o_proj."""
        num_tokens, num_heads, kv_lora_rank, v_head_dim = 3, 2, 4, 5
        attn_output = torch.randn(num_tokens, num_heads * kv_lora_rank)
        bmm_output = torch.randn(num_tokens, num_heads * v_head_dim)
        gate = torch.rand(num_tokens, num_heads)
        o_proj_inputs = []
        attn = SimpleNamespace(
            current_attention_backend="aiter",
            _skip_rope_for_aiter_fused_mla=lambda: False,
            attn_mqa=lambda *args, **kwargs: attn_output,
            num_local_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            v_head_dim=v_head_dim,
            use_deep_gemm_bmm=False,
            o_proj=lambda x: (o_proj_inputs.append(x) or x, None),
            next_skip_topk=None,
            gated_attention_proj_granularity_type="head_wise",
        )
        attn._apply_gated = functools.partial(DsV3MLA._apply_gated, attn)
        rocm_prepare_state = (
            torch.zeros(num_tokens, num_heads, 1),
            torch.zeros(num_tokens, 1, 1),
            torch.zeros(num_tokens, num_heads, kv_lora_rank),
            torch.zeros(num_tokens, 1, kv_lora_rank),
            SimpleNamespace(),
            None,
            None,
            None,
            None,
            None,
        )

        with (
            mock.patch.object(
                forward_mla_rocm, "is_dcp_mla_decode_phase", return_value=False
            ),
            mock.patch.object(
                forward_mla_rocm, "rocm_absorb_v_bmm", return_value=bmm_output
            ),
        ):
            forward_mla_rocm.DeepseekMLARocmForwardMixin.forward_absorb_rocm_core(
                attn, *rocm_prepare_state, gate
            )

        expected = (
            bmm_output.view(num_tokens, num_heads, v_head_dim) * gate[:, :, None]
        ).view(num_tokens, num_heads * v_head_dim)
        self.assertEqual(len(o_proj_inputs), 1)
        torch.testing.assert_close(o_proj_inputs[0], expected)


if __name__ == "__main__":
    unittest.main()
