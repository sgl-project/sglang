"""Hermetic CPU unit tests for DeepGEMM MoE standard layout under deterministic inference.

In ``auto`` mode, ``_should_use_masked_standard_layout`` selects ``masked`` layout
when the estimated workspace fits inside ``_masked_standard_layout_memory_budget_bytes``
(small-M decode / CUDA graph) and falls back to ``compact`` layout on larger prefill
batches (or when ``SGLANG_OPT_DG_COMPACT_EAGER=1``). Because ``masked`` fuses
post-GEMM1 ``SiLU * mul`` with FP8 quantization in FP32 whereas ``compact`` rounds
``SiLU * mul`` to BF16 before quantizing, switching layouts across batch sizes
breaks batch invariance and prefill-vs-decode equivalence under
``--enable-deterministic-inference``.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner import deep_gemm as dg_mod
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_exec(deterministic: bool):
    return SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
    )


def _fake_flags():
    return SimpleNamespace(
        capture=SimpleNamespace(disable_dispose_tensor=False),
        moe=SimpleNamespace(in_speculative_scope=False),
    )


class TestDeepGemmMoeStandardLayoutDeterministic(CustomTestCase):
    def _inputs(self, num_tokens: int):
        cfg = MoeRunnerConfig(
            num_experts=64,
            num_local_experts=64,
            hidden_size=2048,
            intermediate_size_per_partition=512,
            top_k=8,
            inplace=False,
        )
        qinfo = DeepGemmMoeQuantInfo(
            w13_weight=torch.empty((1, 1, 1), dtype=torch.float8_e4m3fn),
            w2_weight=torch.empty((1, 1, 1), dtype=torch.float8_e4m3fn),
            use_fp8=True,
            block_shape=[128, 128],
        )
        hidden = torch.empty((num_tokens, 2048), dtype=torch.bfloat16)
        return cfg, qinfo, hidden

    def _eval(self, *, deterministic: bool, num_tokens: int, layout_env: str = "auto"):
        cfg, qinfo, hidden = self._inputs(num_tokens)
        # Budget chosen between M=16 and M=256 peak estimates so non-deterministic
        # auto mode switches from masked (True) at M=16 to compact (False) at M=256.
        small_bytes = dg_mod._estimate_masked_standard_layout_peak_bytes(
            cfg, qinfo, torch.empty((16, 2048), dtype=torch.bfloat16)
        )
        large_bytes = dg_mod._estimate_masked_standard_layout_peak_bytes(
            cfg, qinfo, torch.empty((256, 2048), dtype=torch.bfloat16)
        )
        budget = (small_bytes + large_bytes) // 2
        with (
            envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override(layout_env),
            envs.SGLANG_OPT_DG_COMPACT_EAGER.override(False),
            mock.patch.object(
                dg_mod, "get_exec", return_value=_fake_exec(deterministic)
            ),
            mock.patch.object(dg_mod, "get_flags", return_value=_fake_flags()),
            mock.patch.object(
                dg_mod, "_masked_activation_unsupported_reason", return_value=None
            ),
            mock.patch.object(
                dg_mod, "_masked_standard_layout_memory_budget_bytes", budget
            ),
        ):
            return dg_mod._should_use_masked_standard_layout(cfg, qinfo, hidden)

    def test_auto_mode_switches_by_budget_when_nondeterministic(self):
        self.assertTrue(self._eval(deterministic=False, num_tokens=16))
        self.assertFalse(self._eval(deterministic=False, num_tokens=256))

    def test_auto_mode_stays_compact_across_all_batch_sizes_when_deterministic(self):
        self.assertFalse(self._eval(deterministic=True, num_tokens=1))
        self.assertFalse(self._eval(deterministic=True, num_tokens=16))
        self.assertFalse(self._eval(deterministic=True, num_tokens=256))

    def test_explicit_layout_override_is_respected_in_deterministic_mode(self):
        self.assertTrue(
            self._eval(deterministic=True, num_tokens=16, layout_env="masked")
        )
        self.assertFalse(
            self._eval(deterministic=True, num_tokens=16, layout_env="compact")
        )


if __name__ == "__main__":
    unittest.main()
