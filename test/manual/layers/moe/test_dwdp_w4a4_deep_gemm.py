"""W4A4 (SGLANG_USE_DWDP_W4A4) DeepGEMM MoE path branch tests.

Covers the SGLANG_USE_DWDP_W4A4 additions in
python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:
  - pre_permute_standard_to_deep_gemm: packed e2m1 scatter branch
  - DeepGemmRunnerCore._run_contiguous_gemm: (1, 32) recipe selection, the
    int32-scale skip of tma_align_input_scale, and the fused
    silu_mul_quant_mxfp4 activation
GEMMs / scatter / quant helpers are mocked; only the branch logic is real.
"""

import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.kernels.ops.moe.ep_moe_kernels as ep_kernels
import sglang.kernels.ops.quantization.mxfp4_group_quant as mxfp4_group_quant
import sglang.srt.layers.deep_gemm_wrapper as deep_gemm_wrapper
from sglang.srt.layers.moe.moe_runner import deep_gemm
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.test_utils import CustomTestCase


@contextlib.contextmanager
def _dwdp_w4a4(enabled):
    if enabled:
        with patch.dict(os.environ, {"SGLANG_USE_DWDP_W4A4": "1"}):
            yield
    else:
        saved = os.environ.pop("SGLANG_USE_DWDP_W4A4", None)
        try:
            yield
        finally:
            if saved is not None:
                os.environ["SGLANG_USE_DWDP_W4A4"] = saved


class TestPrePermuteW4A4(CustomTestCase):
    def _run(self, inplace):
        E, K, N, T = 2, 128, 256, 3
        device = "cuda"
        hidden_states = torch.randn((T, K), device=device, dtype=torch.bfloat16)
        topk_weights = torch.ones((T, 1), device=device, dtype=torch.float32)
        topk_ids = torch.randint(0, E, (T, 1), device=device, dtype=torch.int32)
        dispatch_output = SimpleNamespace(
            hidden_states=hidden_states,
            topk_output=(topk_weights, topk_ids, None),
        )
        quant_info = SimpleNamespace(
            w13_weight=torch.empty((E, N, K), device=device, dtype=torch.float8_e4m3fn),
            block_shape=[1, 128],
            use_mxfp8=False,
        )
        runner_config = SimpleNamespace(
            num_local_experts=E, num_experts=E, top_k=1, inplace=inplace
        )

        scatter_calls = []
        quant_calls = []

        def fake_dispatch_index(topk_ids, num_experts, m_max, expert_start=0):
            counts = torch.zeros((num_experts,), device=device, dtype=torch.int32)
            for e in topk_ids.flatten().tolist():
                counts[e] += 1
            return counts, torch.empty(
                (topk_ids.numel(),), device=device, dtype=torch.int32
            )

        def fake_quant(x):
            quant_calls.append(x)
            M, Kx = x.shape
            return (
                torch.zeros((M, Kx // 2), device=x.device, dtype=torch.int8),
                torch.zeros((M, Kx // 128), device=x.device, dtype=torch.int32),
            )

        def fake_scatter(*args, **kwargs):
            scatter_calls.append((args, kwargs))

        dispose_calls = []

        with (
            _dwdp_w4a4(True),
            patch.object(
                deep_gemm, "_should_use_masked_standard_layout", return_value=False
            ),
            patch.object(
                deep_gemm,
                "get_exec",
                return_value=SimpleNamespace(
                    deterministic=SimpleNamespace(enable_deterministic_inference=False)
                ),
            ),
            patch.object(
                ep_kernels, "fused_moe_dispatch_index", side_effect=fake_dispatch_index
            ),
            patch.object(
                mxfp4_group_quant, "quant_mxfp4_group32_v2", side_effect=fake_quant
            ),
            patch.object(ep_kernels, "ep_scatter", side_effect=fake_scatter),
            patch.object(
                deep_gemm,
                "dispose_tensor",
                side_effect=lambda t: dispose_calls.append(t),
            ),
        ):
            running_state = {}
            runner_input = deep_gemm.pre_permute_standard_to_deep_gemm(
                dispatch_output, quant_info, runner_config, running_state
            )

        # all_tokens depends on the deep_gemm contiguous-layout alignment
        # (128 on the patch's base machine, 64 on SM100 here), so compute it
        # with the same helpers the production path uses instead of pinning it.
        block_e = deep_gemm_wrapper.get_contiguous_layout_alignment(topk_ids.numel(), E)
        all_tokens = deep_gemm._get_compact_all_tokens(topk_ids.numel(), E, block_e)
        self.assertFalse(runner_input.use_masked_gemm)
        self.assertEqual(runner_input.hidden_states.shape, (all_tokens, K // 2))
        self.assertEqual(runner_input.hidden_states.dtype, torch.int8)
        self.assertEqual(runner_input.hidden_states_scale.shape, (all_tokens, K // 128))
        self.assertEqual(runner_input.hidden_states_scale.dtype, torch.int32)
        self.assertEqual(runner_input.m_indices.shape, (all_tokens,))

        self.assertEqual(running_state["all_tokens"], all_tokens)
        self.assertEqual(running_state["mxfp8_act_gran_k"], 128)
        self.assertIs(running_state["topk_ids"], topk_ids)
        # hidden_states_device is the real torch.device (e.g. "cuda:0"), not
        # the bare backend string passed to the "cuda" fixtures above.
        self.assertEqual(running_state["hidden_states_device"], hidden_states.device)

        self.assertEqual(len(scatter_calls), 1)
        # The packed-e2m1 quantization must run exactly once, on the raw
        # (T, K) bf16 activations before the scatter.
        self.assertEqual(len(quant_calls), 1)
        self.assertEqual(quant_calls[0].shape, (T, K))
        self.assertEqual(quant_calls[0].dtype, torch.bfloat16)
        args, kwargs = scatter_calls[0]
        # source q/scale then the scattered destinations
        self.assertEqual(args[0].shape, (T, K // 2))
        self.assertEqual(args[1].shape, (T, K // 128))
        self.assertIs(args[2], topk_ids)
        self.assertEqual(args[6].shape, (all_tokens, K // 2))
        self.assertEqual(args[7].shape, (all_tokens, K // 128))
        self.assertEqual(args[8].shape, (all_tokens,))
        self.assertEqual(kwargs.get("quant_block_size"), 64)
        self.assertFalse(kwargs.get("scale_ue8m0"))

        # The caller's activation is released only in inplace mode.
        caller_disposals = [t for t in dispose_calls if t is hidden_states]
        self.assertEqual(len(caller_disposals), int(inplace))

    def test_w4a4_branch_inplace(self):
        self._run(inplace=True)

    def test_w4a4_branch_not_inplace(self):
        self._run(inplace=False)


class TestRunnerCoreW4A4(CustomTestCase):
    E, K, N, T = 2, 128, 256, 4

    def _make_core(self, activation="silu", **config_kwargs):
        config = MoeRunnerConfig(
            activation=activation,
            is_gated=True,
            swiglu_limit=config_kwargs.pop("swiglu_limit", None),
            **config_kwargs,
        )
        with patch.object(
            deep_gemm,
            "get_moe_a2a_backend",
            return_value=SimpleNamespace(is_megamoe=lambda: False),
        ):
            return deep_gemm.DeepGemmRunnerCore(config)

    def _make_inputs(self, scale_dtype):
        device = "cuda"
        hidden_states = torch.zeros(
            (self.T, self.K), device=device, dtype=torch.float8_e4m3fn
        )
        hidden_states_scale = torch.zeros(
            (self.T, self.K // 128), device=device, dtype=scale_dtype
        )
        m_indices = torch.zeros((self.T,), device=device, dtype=torch.int32)
        runner_input = deep_gemm.DeepGemmRunnerInput(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            use_masked_gemm=False,
            m_indices=m_indices,
        )
        running_state = {
            "all_tokens": self.T,
            "hidden_states_device": device,
            "hidden_states_dtype": torch.bfloat16,
            "hidden_states_shape": (self.T, self.K),
        }
        return runner_input, running_state

    def _make_quant_info(self, is_fp4_experts):
        device = "cuda"
        return SimpleNamespace(
            w13_weight=torch.empty(
                (self.E, self.N, self.K), device=device, dtype=torch.float8_e4m3fn
            ),
            w2_weight=torch.empty(
                (self.E, self.K, self.N // 2), device=device, dtype=torch.float8_e4m3fn
            ),
            w13_scale=torch.ones((self.E,), device=device, dtype=torch.float32),
            w2_scale=torch.ones((self.E,), device=device, dtype=torch.float32),
            block_shape=[1, 128],
            is_fp4_experts=is_fp4_experts,
            use_mxfp8=False,
        )

    def _run_contiguous(self, core, quant_info, scale_dtype):
        runner_input, running_state = self._make_inputs(scale_dtype)
        gemm_calls = []
        tma_calls = []
        silu_calls = []

        def fake_gemm(inputs, weights, out, m_indices, recipe_a=None, recipe_b=None):
            # Produce finite activations so downstream kernels see real data.
            if out.dtype == torch.bfloat16:
                out.normal_()
            gemm_calls.append({"recipe_a": recipe_a, "recipe_b": recipe_b})
            return out

        def fake_tma(scale):
            tma_calls.append(scale)
            return scale

        def fake_silu_mul_quant(gateup, swiglu_limit):
            silu_calls.append(swiglu_limit)
            return (
                torch.zeros(
                    (gateup.shape[0], gateup.shape[1] // 2),
                    device=gateup.device,
                    dtype=torch.float8_e4m3fn,
                ),
                torch.zeros(
                    (gateup.shape[0], 1),
                    device=gateup.device,
                    dtype=torch.int32,
                ),
            )

        @contextlib.contextmanager
        def fake_symmetric_memory(group=None, disabled=True):
            yield

        with (
            patch.object(
                deep_gemm_wrapper,
                "grouped_gemm_nt_f8f8bf16_contig",
                side_effect=fake_gemm,
            ),
            patch.object(ep_kernels, "tma_align_input_scale", side_effect=fake_tma),
            patch.object(
                mxfp4_group_quant,
                "silu_mul_quant_mxfp4",
                side_effect=fake_silu_mul_quant,
            ),
            patch.object(
                deep_gemm, "use_symmetric_memory", side_effect=fake_symmetric_memory
            ),
            patch.object(deep_gemm, "get_tp_group", return_value=None),
            patch.object(deep_gemm, "is_allocation_symmetric", return_value=False),
            patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False, create=True),
        ):
            output = core._run_contiguous_gemm(runner_input, quant_info, running_state)

        return output, gemm_calls, tma_calls, silu_calls

    def test_w4a4_selects_group32_recipe_and_skips_tma_align(self):
        core = self._make_core()
        quant_info = self._make_quant_info(is_fp4_experts=True)
        with (
            _dwdp_w4a4(True),
            patch.object(
                deep_gemm_wrapper, "DEEPGEMM_NEED_TMA_ALIGNED_SCALES", True, create=True
            ),
        ):
            output, gemm_calls, tma_calls, silu_calls = self._run_contiguous(
                core, quant_info, scale_dtype=torch.int32
            )
        self.assertEqual(output.shape, (self.T, self.K))
        self.assertEqual(len(gemm_calls), 2)
        for call in gemm_calls:
            self.assertEqual(call["recipe_a"], (1, 32))
            self.assertEqual(call["recipe_b"], (1, 32))
        # Packed int32 scales must bypass tma_align_input_scale entirely.
        self.assertEqual(tma_calls, [])
        self.assertEqual(len(silu_calls), 1)
        self.assertIsNone(silu_calls[0])  # swiglu_limit

    def test_w4a4_with_float_scale_still_aligns(self):
        core = self._make_core()
        quant_info = self._make_quant_info(is_fp4_experts=True)
        with (
            _dwdp_w4a4(True),
            patch.object(
                deep_gemm_wrapper, "DEEPGEMM_NEED_TMA_ALIGNED_SCALES", True, create=True
            ),
        ):
            _, gemm_calls, tma_calls, _ = self._run_contiguous(
                core, quant_info, scale_dtype=torch.float32
            )
        # Only gemm-1's hidden_states_scale is float here; gemm-2's
        # down_input_scale always comes back int32 (packed ue8m0) from the
        # fused silu_mul_quant_mxfp4, so only one call needs TMA alignment.
        self.assertEqual(len(tma_calls), 1)
        for call in gemm_calls:
            self.assertEqual(call["recipe_a"], (1, 32))

    def test_w4a4_passes_swiglu_limit_to_fused_quant(self):
        # The runner must forward config.swiglu_limit verbatim (10.5, not
        # int-truncated to 10) into the fused silu_mul_quant_mxfp4.
        core = self._make_core(swiglu_limit=10.5)
        quant_info = self._make_quant_info(is_fp4_experts=True)
        with (
            _dwdp_w4a4(True),
            patch.object(
                deep_gemm_wrapper,
                "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
                False,
                create=True,
            ),
        ):
            _, _, _, silu_calls = self._run_contiguous(
                core, quant_info, scale_dtype=torch.int32
            )
        self.assertEqual(silu_calls, [10.5])

    def test_w4a4_env_set_but_fp8_experts_uses_default_branch(self):
        # SGLANG_USE_DWDP_W4A4=1 alone must not switch fp8 weights onto the
        # packed-e2m1 path; only is_fp4_experts gates that.
        core = self._make_core()
        quant_info = self._make_quant_info(is_fp4_experts=False)
        with (
            _dwdp_w4a4(True),
            patch.object(
                deep_gemm_wrapper,
                "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
                False,
                create=True,
            ),
        ):
            _, gemm_calls, tma_calls, silu_calls = self._run_contiguous(
                core, quant_info, scale_dtype=torch.float32
            )
        for call in gemm_calls:
            self.assertIsNone(call["recipe_a"])
            self.assertIsNone(call["recipe_b"])
        self.assertEqual(tma_calls, [])
        self.assertEqual(silu_calls, [])  # not the fused mxfp4 activation

    def test_fp4_without_w4a4_keeps_default_recipe(self):
        core = self._make_core()
        quant_info = self._make_quant_info(is_fp4_experts=True)
        with (
            _dwdp_w4a4(False),
            patch.object(
                deep_gemm_wrapper, "DEEPGEMM_NEED_TMA_ALIGNED_SCALES", True, create=True
            ),
        ):
            _, gemm_calls, tma_calls, silu_calls = self._run_contiguous(
                core, quant_info, scale_dtype=torch.float32
            )
        for call in gemm_calls:
            self.assertEqual(call["recipe_a"], (1, 128))
            self.assertEqual(call["recipe_b"], (1, 32))
        self.assertEqual(len(tma_calls), 2)
        self.assertEqual(silu_calls, [])  # not the fused mxfp4 activation

    def test_fp8_without_w4a4_uses_null_recipe(self):
        core = self._make_core()
        quant_info = self._make_quant_info(is_fp4_experts=False)
        with (
            _dwdp_w4a4(False),
            patch.object(
                deep_gemm_wrapper,
                "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
                False,
                create=True,
            ),
        ):
            _, gemm_calls, tma_calls, _ = self._run_contiguous(
                core, quant_info, scale_dtype=torch.float32
            )
        for call in gemm_calls:
            self.assertIsNone(call["recipe_a"])
            self.assertIsNone(call["recipe_b"])
        self.assertEqual(tma_calls, [])

    def test_situ_activation_without_w4a4(self):
        core = self._make_core(
            activation="situ", gemm1_alpha=1.0, gemm1_clamp_limit=1.0
        )
        quant_info = self._make_quant_info(is_fp4_experts=False)
        with _dwdp_w4a4(False):
            output, gemm_calls, _, silu_calls = self._run_contiguous(
                core, quant_info, scale_dtype=torch.float32
            )
        self.assertEqual(output.shape, (self.T, self.K))
        self.assertEqual(silu_calls, [])


if __name__ == "__main__":
    unittest.main()
