"""W4A4 (SGLANG_USE_DEEPGEMM_W4A4) DeepGEMM MoE path branch tests.

Covers the SGLANG_USE_DEEPGEMM_W4A4 additions in
python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:
  - pre_permute_standard_to_deep_gemm: packed e2m1 scatter branch
  - DeepGemmRunnerCore._run_contiguous_gemm: (1, 32) recipe selection, the
    int32-scale skip of tma_align_input_scale, and the fused
    silu_mul_quant_mxfp4 activation
GEMMs / scatter / quant helpers are mocked; only the branch logic is real.
"""

import contextlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import sglang.kernels.ops.moe.ep_moe_kernels as ep_kernels
import sglang.kernels.ops.quantization.mxfp4_group_quant as mxfp4_group_quant
import sglang.srt.layers.deep_gemm_wrapper as deep_gemm_wrapper
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner import deep_gemm
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@contextlib.contextmanager
def _deepgemm_w4a4(enabled):
    with envs.SGLANG_USE_DEEPGEMM_W4A4.override(enabled):
        yield


@contextlib.contextmanager
def _fused_scatter(enabled):
    with envs.SGLANG_USE_DEEPGEMM_W4A4_FUSED_SCATTER.override(enabled):
        yield


class TestPrePermuteW4A4(CustomTestCase):
    def _run(self, inplace, fused=False):
        E, K, N, T = 2, 128, 256, 3
        device = "cuda"
        hidden_states = torch.randn((T, K), device=device, dtype=torch.bfloat16)
        topk_weights = torch.ones((T, 1), device=device, dtype=torch.float32)
        topk_ids = torch.randint(0, E, (T, 1), device=device, dtype=torch.int32)
        dispatch_output = SimpleNamespace(
            hidden_states=hidden_states,
            topk_output=(topk_weights, topk_ids, None),
        )
        # The real dataclass, not a namespace: upstream's scale_recipes() is a
        # method on it.
        quant_info = deep_gemm.DeepGemmMoeQuantInfo(
            w13_weight=torch.empty((E, N, K), device=device, dtype=torch.float8_e4m3fn),
            w2_weight=torch.empty(
                (E, K, N // 2), device=device, dtype=torch.float8_e4m3fn
            ),
            use_fp8=True,
            block_shape=[1, 128],
            use_mxfp8=False,
            is_fp4_experts=True,
        )
        runner_config = SimpleNamespace(
            num_local_experts=E,
            num_experts=E,
            top_k=1,
            inplace=inplace,
            activation="silu",
            is_gated=True,
            # _masked_activation_unsupported_reason reads both.
            swiglu_limit=None,
            intermediate_size_per_partition=None,
        )

        scatter_calls = []
        quant_calls = []

        def fake_dispatch_index(topk_ids, num_experts, block, expert_start=0):
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
            _deepgemm_w4a4(True),
            _fused_scatter(fused),
            patch.object(
                deep_gemm, "_should_use_masked_standard_layout", return_value=False
            ),
            patch.object(
                ep_kernels, "fused_moe_dispatch_index", side_effect=fake_dispatch_index
            ),
            patch.object(
                mxfp4_group_quant, "quant_mxfp4_group32_v2", side_effect=fake_quant
            ),
            patch.object(ep_kernels, "ep_scatter", side_effect=fake_scatter),
            patch.object(
                ep_kernels, "ep_scatter_quant_mxfp4", side_effect=fake_scatter
            ),
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

        # The compact layout's block_e comes from DeepGEMM
        # (get_contiguous_layout_alignment) and shrinks with the row count, so
        # derive the padded row count rather than hard-coding it. topk is 1 in
        # this fixture.
        num_assignments = T
        block_e = deep_gemm_wrapper.get_contiguous_layout_alignment(num_assignments, E)
        all_tokens = deep_gemm._get_compact_all_tokens(num_assignments, E, block_e)
        self.assertFalse(runner_input.use_masked_gemm)
        self.assertEqual(runner_input.hidden_states.shape, (all_tokens, K // 2))
        self.assertEqual(runner_input.hidden_states.dtype, torch.int8)
        self.assertEqual(runner_input.hidden_states_scale.shape, (all_tokens, K // 128))
        self.assertEqual(runner_input.hidden_states_scale.dtype, torch.int32)
        self.assertEqual(runner_input.m_indices.shape, (all_tokens,))

        self.assertEqual(running_state["all_tokens"], all_tokens)
        # One ue8m0 scale per 32 activation elements along K. This used to be
        # published as running_state["mxfp8_act_gran_k"].
        self.assertEqual(runner_input.activation_scale_block_size, 32)
        self.assertIs(running_state["topk_ids"], topk_ids)
        # hidden_states_device is the real torch.device (e.g. "cuda:0"), not
        # the bare backend string passed to the "cuda" fixtures above.
        self.assertEqual(running_state["hidden_states_device"], hidden_states.device)

        self.assertEqual(len(scatter_calls), 1)
        # The packed-e2m1 quantization must run exactly once, on the raw
        # (T, K) bf16 activations before the scatter.
        if fused:
            self.assertEqual(quant_calls, [])
            self.assertIs(scatter_calls[0][0][0], hidden_states)
            self.assertIs(scatter_calls[0][0][1], topk_ids)
            self.assertEqual(
                len([t for t in dispose_calls if t is hidden_states]), int(inplace)
            )
            return
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

    def test_fused_scatter(self):
        for inplace in (False, True):
            with self.subTest(inplace=inplace):
                self._run(inplace=inplace, fused=True)

    def _layout_fixtures(self, is_fp4_experts=True):
        E, K, N, T = 2, 128, 256, 3
        device = "cuda"
        quant_info = deep_gemm.DeepGemmMoeQuantInfo(
            w13_weight=torch.empty((E, N, K), device=device, dtype=torch.int8),
            w2_weight=torch.empty((E, K, N // 4), device=device, dtype=torch.int8),
            use_fp8=True,
            block_shape=[1, 32],
            use_mxfp8=False,
            is_fp4_experts=is_fp4_experts,
        )
        runner_config = SimpleNamespace(
            num_local_experts=E,
            num_experts=E,
            top_k=1,
            inplace=False,
            activation="silu",
            is_gated=True,
            # _masked_activation_unsupported_reason reads both.
            swiglu_limit=None,
            intermediate_size_per_partition=None,
        )
        hidden_states = torch.zeros((T, K), device=device, dtype=torch.bfloat16)
        return quant_info, runner_config, hidden_states

    def test_w4a4_follows_an_explicit_layout_pin(self):
        # Both layouts implement W4A4, so pinning is a plain override again.
        quant_info, runner_config, hidden_states = self._layout_fixtures()
        with _deepgemm_w4a4(True):
            for mode, expected in (("masked", True), ("compact", False)):
                with (
                    self.subTest(mode=mode),
                    envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override(mode),
                ):
                    self.assertIs(
                        deep_gemm._should_use_masked_standard_layout(
                            runner_config, quant_info, hidden_states
                        ),
                        expected,
                    )

    def test_w4a4_auto_layout_still_follows_the_memory_budget(self):
        # The pin used to force W4A4 onto compact whatever the budget said; now
        # a batch that fits keeps the masked layout. Without this the flag would
        # silently re-pin the layout the moment a W4A4 run starts.
        quant_info, runner_config, hidden_states = self._layout_fixtures()
        with (
            _deepgemm_w4a4(True),
            envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override("auto"),
        ):
            for budget, expected in ((1 << 30, True), (1, False)):
                with (
                    self.subTest(budget=budget),
                    patch.object(
                        deep_gemm,
                        "_masked_standard_layout_memory_budget_bytes",
                        budget,
                    ),
                ):
                    self.assertIs(
                        deep_gemm._should_use_masked_standard_layout(
                            runner_config, quant_info, hidden_states
                        ),
                        expected,
                    )

    def test_masked_layout_is_untouched_without_w4a4(self):
        quant_info, runner_config, hidden_states = self._layout_fixtures(
            is_fp4_experts=False
        )
        with (
            _deepgemm_w4a4(False),
            envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override("masked"),
        ):
            self.assertTrue(
                deep_gemm._should_use_masked_standard_layout(
                    runner_config, quant_info, hidden_states
                )
            )

    def _masked_preprocess(self, output_dtype, fused=True):
        E, K, T, topk, num_experts = 2, 256, 7, 2, 2
        device = "cuda"
        torch.manual_seed(4321)
        x = torch.randn((T, K), device=device, dtype=torch.bfloat16)
        topk_ids = torch.stack(
            [
                (torch.arange(topk, device=device, dtype=torch.int32) + token)
                % num_experts
                for token in range(T)
            ]
        )
        from sglang.kernels.ops.moe.ep_moe_kernels import moe_ep_deepgemm_preprocess

        with _fused_scatter(fused):
            return (
                K,
                topk,
                x,
                moe_ep_deepgemm_preprocess(
                    topk_ids=topk_ids,
                    num_local_experts=num_experts,
                    hidden_states=x,
                    top_k=topk,
                    block_shape=[1, 32],
                    output_dtype=output_dtype,
                    use_mxfp8=False,
                ),
            )

    def test_masked_preprocess_packs_e2m1_for_w4a4(self):
        # The masked twin of the contiguous packed-e2m1 scatter: the same
        # (1, 32) quantization, scattered into the padded (E, m_max, ...) layout
        # with the scale written MN-major in place. Both implementations behind
        # SGLANG_USE_DEEPGEMM_W4A4_FUSED_SCATTER have to land on the same bytes.
        from sglang.kernels.ops.quantization.mxfp4_group_quant import (
            quant_mxfp4_group32_v2,
        )

        seen = []
        for fused in (True, False):
            with self.subTest(fused=fused):
                (
                    K,
                    topk,
                    x,
                    (
                        _,
                        _,
                        src2dst,
                        grouped_x,
                        grouped_scale,
                    ),
                ) = self._masked_preprocess(torch.int8, fused=fused)
                direct_x, direct_scale = quant_mxfp4_group32_v2(x)

                self.assertEqual(grouped_x.dtype, torch.int8)
                self.assertEqual(grouped_x.shape[-1], K // 2)
                self.assertEqual(grouped_scale.dtype, torch.int32)
                self.assertEqual(grouped_scale.shape[-1], K // 128)
                self.assertFalse(grouped_scale.is_contiguous())  # MN-major view
                self.assertEqual(src2dst.numel(), x.shape[0] * topk)

                per_run = {}
                for token in range(x.shape[0]):
                    for slot in range(topk):
                        dst = int(src2dst[token * topk + slot])
                        expert, row = divmod(dst, grouped_x.shape[1])
                        q_row = grouped_x[expert, row].view(torch.uint8)
                        s_row = grouped_scale[expert, row]
                        self.assertTrue(
                            torch.equal(q_row, direct_x[token].view(torch.uint8))
                        )
                        self.assertTrue(torch.equal(s_row, direct_scale[token]))
                        per_run[dst] = (q_row.clone(), s_row.clone())
                # Rows no slot claimed keep whatever `torch.empty` handed over, so
                # comparing the two runs means comparing the claimed rows only.
                seen.append(per_run)

        self.assertEqual(len(seen[0]), len(seen[1]))
        for dst, (q_fused, s_fused) in seen[0].items():
            q_two_step, s_two_step = seen[1][dst]
            self.assertTrue(torch.equal(q_fused, q_two_step))
            self.assertTrue(torch.equal(s_fused, s_two_step))

    def test_masked_preprocess_keeps_the_fp8_layout_for_fp8_output(self):
        # The int8 branch must not capture the fp8 dtype: an fp8 output still
        # carries one byte per element. (Its ue8m0 word count coincides with the
        # packed-e2m1 one -- both pack four bytes per 32-element group -- so the
        # dtype and the value width are what tell the two apart.)
        K, _topk, _x, (_, _, _, grouped_x, grouped_scale) = self._masked_preprocess(
            torch.float8_e4m3fn
        )
        self.assertEqual(grouped_x.dtype, torch.float8_e4m3fn)
        self.assertEqual(grouped_x.shape[-1], K)
        self.assertEqual(grouped_scale.dtype, torch.int32)


class TestRunnerCoreW4A4(CustomTestCase):
    E, K, N, T = 2, 128, 256, 4

    def _make_core(self, activation="silu", **config_kwargs):
        config = MoeRunnerConfig(
            activation=activation,
            is_gated=True,
            swiglu_limit=config_kwargs.pop("swiglu_limit", None),
            **config_kwargs,
        )
        return deep_gemm.DeepGemmRunnerCore(config)

    def _make_inputs(self, packed):
        """Build the pre-permute output _run_contiguous_gemm consumes.

        packed=True is what pre_permute_standard_to_deep_gemm emits on the W4A4
        path: int8 packed e2m1 rows plus int32 packed ue8m0 scale words.
        packed=False is the ordinary float8_e4m3fn + float32-scale contiguous
        input every other caller of this helper produces.
        """
        device = "cuda"
        hidden_states = torch.zeros(
            (self.T, self.K // 2 if packed else self.K),
            device=device,
            dtype=torch.int8 if packed else torch.float8_e4m3fn,
        )
        hidden_states_scale = torch.zeros(
            (self.T, self.K // 128),
            device=device,
            dtype=torch.int32 if packed else torch.float32,
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
        return deep_gemm.DeepGemmMoeQuantInfo(
            w13_weight=torch.empty(
                (self.E, self.N, self.K), device=device, dtype=torch.float8_e4m3fn
            ),
            w2_weight=torch.empty(
                (self.E, self.K, self.N // 2), device=device, dtype=torch.float8_e4m3fn
            ),
            use_fp8=True,
            w13_scale=torch.ones((self.E,), device=device, dtype=torch.float32),
            w2_scale=torch.ones((self.E,), device=device, dtype=torch.float32),
            block_shape=[1, 128],
            is_fp4_experts=is_fp4_experts,
            use_mxfp8=False,
        )

    def _run_contiguous(self, core, quant_info, packed, break_scale=False):
        runner_input, running_state = self._make_inputs(packed)
        if break_scale:
            # int32 is what marks the scale as packed ue8m0; a float scale next
            # to int8 activations has to be rejected rather than mis-decoded.
            runner_input.hidden_states_scale = runner_input.hidden_states_scale.float()
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
            # Mirror the real contract - (T, H // 2) int8 packed e2m1 plus
            # (T, H // 128) int32 ue8m0 words - rather than an fp8-shaped stand-in.
            silu_calls.append(swiglu_limit)
            return (
                torch.zeros(
                    (gateup.shape[0], gateup.shape[1] // 4),
                    device=gateup.device,
                    dtype=torch.int8,
                ),
                torch.zeros(
                    (gateup.shape[0], gateup.shape[1] // 256),
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
            patch.object(
                deep_gemm,
                "get_parallel",
                return_value=SimpleNamespace(tp_group=None),
            ),
            patch.object(deep_gemm, "is_allocation_symmetric", return_value=False),
            patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False, create=True),
        ):
            output = core._run_contiguous_gemm(runner_input, quant_info, running_state)

        return output, gemm_calls, tma_calls, silu_calls

    def test_w4a4_selects_group32_recipe_and_skips_tma_align(self):
        core = self._make_core()
        quant_info = self._make_quant_info(is_fp4_experts=True)
        with (
            _deepgemm_w4a4(True),
            patch.object(
                deep_gemm_wrapper, "DEEPGEMM_NEED_TMA_ALIGNED_SCALES", True, create=True
            ),
        ):
            output, gemm_calls, tma_calls, silu_calls = self._run_contiguous(
                core, quant_info, packed=True
            )
        self.assertEqual(output.shape, (self.T, self.K))
        self.assertEqual(len(gemm_calls), 2)
        for call in gemm_calls:
            self.assertEqual(call["recipe_a"], (1, 32))
            self.assertEqual(call["recipe_b"], (1, 32))
        # Packed e2m1 input must bypass tma_align_input_scale entirely.
        self.assertEqual(tma_calls, [])
        self.assertEqual(len(silu_calls), 1)
        self.assertIsNone(silu_calls[0])  # swiglu_limit

    def test_w4a4_flag_does_not_hijack_fp8_contiguous_input(self):
        # The global flag must not move an ordinary DeepEP FP8 activation onto
        # the packed-e2m1 recipe; only int8 packed input selects W4A4.
        core = self._make_core()
        quant_info = self._make_quant_info(is_fp4_experts=True)
        with (
            _deepgemm_w4a4(True),
            patch.object(
                deep_gemm_wrapper, "DEEPGEMM_NEED_TMA_ALIGNED_SCALES", True, create=True
            ),
        ):
            _, gemm_calls, tma_calls, silu_calls = self._run_contiguous(
                core, quant_info, packed=False
            )
        for call in gemm_calls:
            self.assertEqual(call["recipe_a"], (1, 128))
            self.assertEqual(call["recipe_b"], (1, 32))
        # The legacy path keeps both of its TMA alignments.
        self.assertEqual(len(tma_calls), 2)
        self.assertEqual(silu_calls, [])

    def test_packed_input_without_w4a4_env_is_rejected(self):
        # A packed e2m1 tensor may only reach the runner when W4A4 produced it.
        core = self._make_core()
        quant_info = self._make_quant_info(is_fp4_experts=True)
        with (
            _deepgemm_w4a4(False),
            patch.object(
                deep_gemm_wrapper,
                "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
                False,
                create=True,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "Packed MXFP4 input requires"):
                self._run_contiguous(core, quant_info, packed=True)

    def test_packed_input_with_fp8_experts_is_rejected(self):
        core = self._make_core()
        quant_info = self._make_quant_info(is_fp4_experts=False)
        with (
            _deepgemm_w4a4(True),
            patch.object(
                deep_gemm_wrapper,
                "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
                False,
                create=True,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "Packed MXFP4 input requires"):
                self._run_contiguous(core, quant_info, packed=True)

    def test_packed_input_with_a_float_scale_is_rejected(self):
        # int8 activations only mean "packed e2m1" together with an int32 scale;
        # a float scale there is a layout mismatch, not a fp8 input.
        core = self._make_core()
        quant_info = self._make_quant_info(is_fp4_experts=True)
        with (
            _deepgemm_w4a4(True),
            patch.object(
                deep_gemm_wrapper,
                "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
                False,
                create=True,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "Invalid packed MXFP4 input"):
                self._run_contiguous(core, quant_info, packed=True, break_scale=True)

    def test_w4a4_passes_swiglu_limit_to_fused_quant(self):
        # The runner must forward config.swiglu_limit verbatim (10.5, not
        # int-truncated to 10) into the fused silu_mul_quant_mxfp4.
        core = self._make_core(swiglu_limit=10.5)
        quant_info = self._make_quant_info(is_fp4_experts=True)
        with (
            _deepgemm_w4a4(True),
            patch.object(
                deep_gemm_wrapper,
                "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
                False,
                create=True,
            ),
        ):
            _, _, _, silu_calls = self._run_contiguous(core, quant_info, packed=True)
        self.assertEqual(silu_calls, [10.5])

    def test_w4a4_helper_rejects_non_silu_activation(self):
        # There is no MXFP4 activation kernel for SiTU/others, so the helper
        # must fail loudly instead of silently computing SiLU.
        quant_info = self._make_quant_info(is_fp4_experts=True)
        config = MoeRunnerConfig(activation="situ", is_gated=True)
        with _deepgemm_w4a4(True):
            with self.assertRaisesRegex(ValueError, "only supports gated SiLU"):
                deep_gemm._use_deepgemm_w4a4(quant_info, config)

    def test_w4a4_helper_requires_env_and_fp4_experts(self):
        config = MoeRunnerConfig(activation="silu", is_gated=True)
        fp4_quant_info = self._make_quant_info(is_fp4_experts=True)
        with _deepgemm_w4a4(False):
            self.assertFalse(deep_gemm._use_deepgemm_w4a4(fp4_quant_info, config))
        with _deepgemm_w4a4(True):
            self.assertTrue(deep_gemm._use_deepgemm_w4a4(fp4_quant_info, config))
            self.assertFalse(
                deep_gemm._use_deepgemm_w4a4(
                    self._make_quant_info(is_fp4_experts=False), config
                )
            )

    def test_w4a4_env_set_but_fp8_experts_uses_default_branch(self):
        # SGLANG_USE_DEEPGEMM_W4A4=1 alone must not switch fp8 weights onto the
        # packed-e2m1 path; only is_fp4_experts gates that.
        core = self._make_core()
        quant_info = self._make_quant_info(is_fp4_experts=False)
        with (
            _deepgemm_w4a4(True),
            patch.object(
                deep_gemm_wrapper,
                "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
                False,
                create=True,
            ),
        ):
            _, gemm_calls, tma_calls, silu_calls = self._run_contiguous(
                core, quant_info, packed=False
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
            _deepgemm_w4a4(False),
            patch.object(
                deep_gemm_wrapper, "DEEPGEMM_NEED_TMA_ALIGNED_SCALES", True, create=True
            ),
        ):
            _, gemm_calls, tma_calls, silu_calls = self._run_contiguous(
                core, quant_info, packed=False
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
            _deepgemm_w4a4(False),
            patch.object(
                deep_gemm_wrapper,
                "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
                False,
                create=True,
            ),
        ):
            _, gemm_calls, tma_calls, _ = self._run_contiguous(
                core, quant_info, packed=False
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
        with _deepgemm_w4a4(False):
            output, gemm_calls, _, silu_calls = self._run_contiguous(
                core, quant_info, packed=False
            )
        self.assertEqual(output.shape, (self.T, self.K))
        self.assertEqual(silu_calls, [])


class TestRunnerCoreMaskedW4A4(CustomTestCase):
    """The masked twin of `_run_contiguous_gemm`'s W4A4 branch.

    The masked grouped GEMMs are mocked; the packed-e2m1 activation quantizer is
    real, so the two padded (E, m_max, ...) buffers it hands over are checked
    against the shapes the masked DeepGEMM call consumes.
    """

    E, K, N, M_MAX = 2, 128, 256, 256
    TOPK, T = 1, 17
    MASKED_M = (10, 7)

    def _make_core(self, **config_kwargs):
        config = MoeRunnerConfig(
            activation=config_kwargs.pop("activation", "silu"),
            is_gated=True,
            top_k=config_kwargs.pop("top_k", self.TOPK),
            swiglu_limit=config_kwargs.pop("swiglu_limit", None),
            **config_kwargs,
        )
        return deep_gemm.DeepGemmRunnerCore(config)

    def _make_quant_info(self, is_fp4_experts=True):
        device = "cuda"
        return deep_gemm.DeepGemmMoeQuantInfo(
            w13_weight=torch.empty(
                (self.E, self.N, self.K // 2), device=device, dtype=torch.int8
            ),
            w2_weight=torch.empty(
                (self.E, self.K, self.N // 4), device=device, dtype=torch.int8
            ),
            use_fp8=True,
            w13_scale=torch.ones((self.E,), device=device, dtype=torch.int32),
            w2_scale=torch.ones((self.E,), device=device, dtype=torch.int32),
            block_shape=[1, 32],
            is_fp4_experts=is_fp4_experts,
            use_mxfp8=False,
        )

    def _make_inputs(self, packed=True, break_scale=False):
        device = "cuda"
        if packed:
            hidden_states = torch.zeros(
                (self.E, self.M_MAX, self.K // 2), device=device, dtype=torch.int8
            )
            scale_storage = torch.zeros(
                (self.E, self.K // 128, self.M_MAX), device=device, dtype=torch.int32
            )
        else:
            hidden_states = torch.zeros(
                (self.E, self.M_MAX, self.K), device=device, dtype=torch.float8_e4m3fn
            )
            scale_storage = torch.zeros(
                (self.E, self.K // 32, self.M_MAX), device=device, dtype=torch.float32
            )
        if break_scale:
            scale_storage = scale_storage.float()
        runner_input = deep_gemm.DeepGemmRunnerInput(
            hidden_states=hidden_states,
            hidden_states_scale=scale_storage.transpose(1, 2),
            use_masked_gemm=True,
            masked_m=torch.tensor(self.MASKED_M, device=device, dtype=torch.int32),
            expected_m=4,
        )
        running_state = {
            "hidden_states_device": device,
            "topk_ids": torch.zeros(
                (self.T, self.TOPK), device=device, dtype=torch.int32
            ),
        }
        return runner_input, running_state

    def _run_masked(
        self,
        core,
        quant_info,
        packed=True,
        break_scale=False,
        drop_topk_ids=False,
    ):
        runner_input, running_state = self._make_inputs(packed, break_scale)
        if drop_topk_ids:
            del running_state["topk_ids"]
        gemm_calls = []
        tma_calls = []
        varlen_calls = []

        def fake_masked_gemm(
            lhs, rhs, out, masked_m, expected_m, recipe_a=None, recipe_b=None, **kwargs
        ):
            out.normal_()
            # Shapes/dtypes are recorded, not the tensors: the runner disposes
            # the gateup activation right after its GEMM.
            gemm_calls.append(
                {
                    "lhs_q_shape": tuple(lhs[0].shape),
                    "lhs_q_dtype": lhs[0].dtype,
                    "lhs_sf_shape": None if lhs[1] is None else tuple(lhs[1].shape),
                    "lhs_sf_dtype": None if lhs[1] is None else lhs[1].dtype,
                    "lhs_sf_stride": None if lhs[1] is None else lhs[1].stride(),
                    "rhs_q": rhs[0],
                    "masked_m": masked_m,
                    "expected_m": expected_m,
                    "recipe_a": recipe_a,
                    "recipe_b": recipe_b,
                }
            )
            return None

        def fake_tma(scale):
            tma_calls.append(scale)
            return scale

        def fake_varlen(*args, **kwargs):
            varlen_calls.append((args, kwargs))
            device = "cuda"
            return (
                torch.zeros(
                    (self.E, self.M_MAX, self.N // 2),
                    device=device,
                    dtype=torch.float8_e4m3fn,
                ),
                torch.zeros(
                    (self.E, self.M_MAX, 1), device=device, dtype=torch.float32
                ),
            )

        @contextlib.contextmanager
        def fake_symmetric_memory(group=None, disabled=True):
            yield

        with (
            patch.object(
                deep_gemm_wrapper,
                "grouped_gemm_nt_f8f8bf16_masked",
                side_effect=fake_masked_gemm,
            ),
            patch.object(
                deep_gemm_wrapper,
                "get_mn_major_tma_aligned_tensor",
                side_effect=fake_tma,
            ),
            patch.object(
                deep_gemm, "_varlen_deep_gemm_silu_mul_quant", side_effect=fake_varlen
            ),
            patch.object(
                deep_gemm, "use_symmetric_memory", side_effect=fake_symmetric_memory
            ),
            patch.object(
                deep_gemm,
                "get_parallel",
                return_value=SimpleNamespace(tp_group=None),
            ),
            patch.object(deep_gemm, "is_allocation_symmetric", return_value=False),
        ):
            output = core._run_masked_gemm(runner_input, quant_info, running_state)

        return output, gemm_calls, tma_calls, varlen_calls

    def test_w4a4_masked_selects_group32_recipes_and_fp4_activation(self):
        core = self._make_core()
        quant_info = self._make_quant_info()
        with (
            _deepgemm_w4a4(True),
            patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False, create=True),
            patch.object(
                deep_gemm_wrapper, "DEEPGEMM_NEED_TMA_ALIGNED_SCALES", True, create=True
            ),
        ):
            output, gemm_calls, tma_calls, varlen_calls = self._run_masked(
                core, quant_info
            )

        self.assertEqual(output.shape, (self.E, self.M_MAX, self.K))
        self.assertEqual(varlen_calls, [])  # not the fp8 masked activation
        # int32 scales stay untouched on every machine.
        self.assertEqual(tma_calls, [])

        self.assertEqual(len(gemm_calls), 2)
        for call in gemm_calls:
            self.assertEqual(call["recipe_a"], (1, 32))
            self.assertEqual(call["recipe_b"], (1, 32))
        gateup, down = gemm_calls
        self.assertEqual(gateup["lhs_q_shape"], (self.E, self.M_MAX, self.K // 2))
        self.assertEqual(gateup["lhs_q_dtype"], torch.int8)
        self.assertEqual(gateup["lhs_sf_shape"], (self.E, self.M_MAX, self.K // 128))
        self.assertEqual(gateup["lhs_sf_dtype"], torch.int32)
        # The scale is a view over (E, K // 128, m_max): stride(-2) == 1 is what
        # "MN-major" means to DeepGEMM.
        self.assertEqual(gateup["lhs_sf_stride"][-2], 1)
        self.assertIs(gateup["rhs_q"], quant_info.w13_weight)
        self.assertEqual(gateup["masked_m"].tolist(), list(self.MASKED_M))
        self.assertEqual(gateup["expected_m"], 4)
        # The down activation is group-32 e2m1 too, but half way narrower.
        self.assertEqual(down["lhs_q_shape"], (self.E, self.M_MAX, self.N // 4))
        self.assertEqual(down["lhs_q_dtype"], torch.int8)
        self.assertEqual(down["lhs_sf_shape"], (self.E, self.M_MAX, self.N // 256))
        self.assertEqual(down["lhs_sf_dtype"], torch.int32)
        self.assertIs(down["rhs_q"], quant_info.w2_weight)

    def test_packed_masked_input_without_w4a4_is_rejected(self):
        core = self._make_core()
        quant_info = self._make_quant_info()
        with (
            _deepgemm_w4a4(False),
            patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False, create=True),
        ):
            with self.assertRaisesRegex(ValueError, "Packed MXFP4 input requires"):
                self._run_masked(core, quant_info)

    def test_masked_fp8_input_without_w4a4_keeps_its_recipe(self):
        # The existing masked fp8 path (DSV4 W4A8) must not be re-routed by the
        # new branch: recipe_a stays the 1x128 activation one.
        core = self._make_core()
        quant_info = self._make_quant_info()
        with (
            _deepgemm_w4a4(False),
            patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False, create=True),
            patch.object(
                deep_gemm_wrapper,
                "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
                False,
                create=True,
            ),
        ):
            _, gemm_calls, _, varlen_calls = self._run_masked(
                core, quant_info, packed=False
            )
        self.assertEqual(len(varlen_calls), 1)
        for call in gemm_calls:
            self.assertEqual(call["recipe_a"], (1, 128))
            self.assertEqual(call["recipe_b"], (1, 32))

    def test_packed_masked_input_with_a_float_scale_is_rejected(self):
        core = self._make_core()
        quant_info = self._make_quant_info()
        with (
            _deepgemm_w4a4(True),
            patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False, create=True),
        ):
            with self.assertRaisesRegex(ValueError, "Invalid packed MXFP4 masked"):
                self._run_masked(core, quant_info, break_scale=True)

    def test_masked_w4a4_needs_running_state_topk_ids(self):
        # The activation kernel sizes its grid from topk_ids, so a caller that
        # forgot to publish it must fail here rather than quantize nothing.
        core = self._make_core()
        quant_info = self._make_quant_info()
        with (
            _deepgemm_w4a4(True),
            patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False, create=True),
        ):
            with self.assertRaisesRegex(AssertionError, "topk_ids"):
                self._run_masked(core, quant_info, drop_topk_ids=True)

    def test_masked_w4a4_forwards_swiglu_limit(self):
        core = self._make_core(swiglu_limit=10.5)
        quant_info = self._make_quant_info()
        calls = []
        real = mxfp4_group_quant.silu_mul_quant_mxfp4_masked

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return real(*args, **kwargs)

        with (
            _deepgemm_w4a4(True),
            patch.object(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False, create=True),
            patch.object(
                mxfp4_group_quant, "silu_mul_quant_mxfp4_masked", side_effect=spy
            ),
        ):
            self._run_masked(core, quant_info)
        self.assertEqual([call["swiglu_limit"] for call in calls], [10.5])
        self.assertEqual([call["topk"] for call in calls], [self.TOPK])
        self.assertEqual([call["num_real_tokens"] for call in calls], [self.T])


class TestW4A4MaskedMatchesCompact(CustomTestCase):
    """W4A4 must give the same answer on both standard layouts.

    Real DeepGEMM (packed-fp4 x packed-fp4 masked and contiguous), real
    packed-e2m1 expert weights, real quantizers. Only the process-global
    parallel groups are stubbed out, since this runs outside a serving process.
    """

    E, HIDDEN, INTERMEDIATE, TOPK, T = 2, 512, 512, 2, 7
    NUM_EXPERTS = 8

    @staticmethod
    def _fp4_weight(weight_bf16):
        """Per-expert packed e2m1 weights + the MN-major ue8m0 scales DeepGEMM wants."""
        from deep_gemm import transform_sf_into_required_layout
        from deep_gemm.utils import per_token_cast_to_fp4

        E, n, k = weight_bf16.shape
        q = torch.empty((E, n, k // 2), dtype=torch.int8, device=weight_bf16.device)
        sf = torch.empty(
            (E, n, k // 32), dtype=torch.float32, device=weight_bf16.device
        )
        for e in range(E):
            q[e], sf[e] = per_token_cast_to_fp4(
                weight_bf16[e], use_ue8m0=True, gran_k=32
            )
        return q, transform_sf_into_required_layout(
            sf, mn=n, k=k, recipe=(1, 32), num_groups=E, disable_ue8m0_cast=False
        )

    def _fixtures(self):
        device = "cuda"
        torch.manual_seed(20260921)
        hidden_states = torch.randn(
            (self.T, self.HIDDEN), device=device, dtype=torch.bfloat16
        )
        topk_ids = torch.tensor(
            [[0, -1], [1, -1], [0, 1], [-1, 1], [0, -1], [1, 0], [-1, 1]],
            device=device,
            dtype=torch.int32,
        )
        topk_weights = torch.rand(
            (self.T, self.TOPK), device=device, dtype=torch.float32
        )

        weight_std = self.HIDDEN**-0.5
        w13_bf16 = (
            torch.randn(
                (self.E, 2 * self.INTERMEDIATE, self.HIDDEN),
                device=device,
                dtype=torch.bfloat16,
            )
            * weight_std
        )
        w2_bf16 = (
            torch.randn(
                (self.E, self.HIDDEN, self.INTERMEDIATE),
                device=device,
                dtype=torch.bfloat16,
            )
            * weight_std
        )
        w13, w13_scale = self._fp4_weight(w13_bf16)
        w2, w2_scale = self._fp4_weight(w2_bf16)

        quant_info = deep_gemm.DeepGemmMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            use_fp8=True,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=[1, 32],
            is_fp4_experts=True,
        )
        dispatch_output = SimpleNamespace(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=(topk_weights, topk_ids, None),
        )
        return dispatch_output, quant_info, topk_ids

    def _run_layout(self, layout, dispatch_output, quant_info):
        config = MoeRunnerConfig(
            num_experts=self.NUM_EXPERTS,
            num_local_experts=self.E,
            hidden_size=self.HIDDEN,
            intermediate_size_per_partition=self.INTERMEDIATE,
            top_k=self.TOPK,
            activation="silu",
            is_gated=True,
            inplace=False,
        )
        running_state = {}
        with (
            _deepgemm_w4a4(True),
            envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override(layout),
        ):
            runner_input = deep_gemm.pre_permute_standard_to_deep_gemm(
                dispatch_output, quant_info, config, running_state
            )
            # Probe now: the runner disposes the packed activation, which resets
            # this reference to an empty tensor.
            probe = SimpleNamespace(
                use_masked_gemm=runner_input.use_masked_gemm,
                q_shape=tuple(runner_input.hidden_states.shape),
                q_dtype=runner_input.hidden_states.dtype,
                sf_shape=tuple(runner_input.hidden_states_scale.shape),
                sf_dtype=runner_input.hidden_states_scale.dtype,
                masked_m=(
                    None
                    if runner_input.masked_m is None
                    else int(runner_input.masked_m.sum())
                ),
                m_indices=(
                    None
                    if runner_input.m_indices is None
                    else runner_input.m_indices.clone()
                ),
            )
            runner_output = deep_gemm.DeepGemmRunnerCore(config).run(
                runner_input, quant_info, running_state
            )
            out = deep_gemm.post_permute_deep_gemm_to_standard(
                runner_output, quant_info, config, running_state
            ).hidden_states
        return probe, out, running_state

    def test_masked_and_compact_agree(self):
        dispatch_output, quant_info, topk_ids = self._fixtures()

        with (
            patch.object(
                deep_gemm,
                "get_parallel",
                return_value=SimpleNamespace(tp_group=None),
            ),
            patch.object(
                deep_gemm,
                "use_symmetric_memory",
                lambda *a, **k: contextlib.nullcontext(),
            ),
        ):
            masked, masked_out, masked_state = self._run_layout(
                "masked", dispatch_output, quant_info
            )
            compact, compact_out, compact_state = self._run_layout(
                "compact", dispatch_output, quant_info
            )
        torch.cuda.synchronize()

        # Both layouts really did quantize to packed e2m1, and to the same
        # padding semantics the two GEMMs expect.
        self.assertTrue(masked.use_masked_gemm)
        self.assertFalse(compact.use_masked_gemm)
        self.assertEqual(masked.q_dtype, torch.int8)
        self.assertEqual(compact.q_dtype, torch.int8)
        self.assertEqual(masked.q_shape, (self.E, 256, self.HIDDEN // 2))
        self.assertEqual(masked.sf_shape, (self.E, 256, self.HIDDEN // 128))
        self.assertEqual(masked.sf_dtype, torch.int32)
        self.assertEqual(compact.q_shape[1], self.HIDDEN // 2)

        # All 9 valid (token, slot) pairs are routed exactly once per layout.
        self.assertEqual(masked.masked_m, 9)
        self.assertIn("src2dst", masked_state)
        num_assignments = self.T * self.TOPK
        block_e = deep_gemm_wrapper.get_contiguous_layout_alignment(
            num_assignments, self.E
        )
        self.assertEqual(
            compact_state["all_tokens"],
            deep_gemm._get_compact_all_tokens(num_assignments, self.E, block_e),
        )
        valid = topk_ids[topk_ids >= 0]
        self.assertTrue(
            torch.equal(
                torch.bincount(
                    compact.m_indices[compact.m_indices >= 0], minlength=self.E
                ),
                torch.bincount(valid, minlength=self.E),
            )
        )

        # Same kernel family, same quantization, so this is currently exact;
        # the tolerance only keeps the assertion from pinning that.
        torch.testing.assert_close(masked_out, compact_out, rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
    unittest.main()
