# SPDX-License-Identifier: Apache-2.0

import gc
import math
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch
from flashinfer import fp4_quantize
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType

from sglang.kernels.ops.moe.nvfp4_moe_sm120 import (
    NVFP4_MOE_SM120_MAX_TOKENS,
    Nvfp4MoeWorkspace,
    nvfp4_moe_sm120,
    nvfp4_moe_sm120_enabled,
)
from sglang.kernels.registry import registry
from sglang.srt.environ import envs
from sglang.srt.utils import is_sm120_supported, is_sm121
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.quant_ref_utils import dequantize_nvfp4_to_dtype
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

HIDDEN = 256
INTERMEDIATE = 320
LOCAL_EXPERTS = 8
GLOBAL_EXPERTS = 16
TOP_K = 4
MAX_TOKENS = NVFP4_MOE_SM120_MAX_TOKENS


def _quantize_weight(weight: torch.Tensor):
    global_scale = torch.tensor(
        2688.0 / weight.abs().max().item(), dtype=torch.float32, device="cuda"
    )
    packed, scale = fp4_quantize(weight, global_scale)
    dequant = dequantize_nvfp4_to_dtype(packed, scale, global_scale, torch.float32)
    return packed, scale.view(torch.float8_e4m3fn), 1.0 / global_scale, dequant


def _quant_roundtrip(value: torch.Tensor, global_scale: torch.Tensor):
    packed, scale = fp4_quantize(value.to(torch.bfloat16), global_scale)
    return dequantize_nvfp4_to_dtype(packed, scale, global_scale, torch.float32)


class _Fixture:
    def __init__(self):
        torch.manual_seed(20260827)
        w13_q = []
        w2_q = []
        w13_sf = []
        w2_sf = []
        w13_alpha = []
        w2_alpha = []
        w13_ref = []
        w2_ref = []
        for _ in range(LOCAL_EXPERTS):
            up_row_scale = torch.linspace(
                0.25, 2.0, INTERMEDIATE, dtype=torch.float32, device="cuda"
            )[:, None]
            gate_row_scale = torch.linspace(
                2.0, 0.25, INTERMEDIATE, dtype=torch.float32, device="cuda"
            )[:, None]
            down_row_scale = torch.linspace(
                0.5, 1.75, HIDDEN, dtype=torch.float32, device="cuda"
            )[:, None]
            up = (
                torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.float32, device="cuda")
                * up_row_scale
                / 8
            ).to(torch.bfloat16)
            gate = (
                torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.float32, device="cuda")
                * gate_row_scale
                / 8
            ).to(torch.bfloat16)
            down = (
                torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.float32, device="cuda")
                * down_row_scale
                / 8
            ).to(torch.bfloat16)
            q13, sf13, alpha13, ref13 = _quantize_weight(torch.cat((up, gate), dim=0))
            q2, sf2, alpha2, ref2 = _quantize_weight(down)
            w13_q.append(q13)
            w2_q.append(q2)
            w13_sf.append(sf13)
            w2_sf.append(sf2)
            w13_alpha.append(alpha13)
            w2_alpha.append(alpha2)
            w13_ref.append(ref13)
            w2_ref.append(ref2)

        self.w13 = torch.stack(w13_q)
        self.w2 = torch.stack(w2_q)
        self.w13_sf = torch.stack(w13_sf)
        self.w2_sf = torch.stack(w2_sf)
        self.w13_weight_scale = torch.stack(w13_alpha).float()
        self.w2_weight_scale = torch.stack(w2_alpha).float()
        self.w13_ref = torch.stack(w13_ref)
        self.w2_ref = torch.stack(w2_ref)
        self.input_scale_1 = torch.tensor(1024.0, dtype=torch.float32, device="cuda")
        self.input_scale_2 = torch.tensor(64.0, dtype=torch.float32, device="cuda")
        self.g1_alpha = self.w13_weight_scale / self.input_scale_1
        self.g2_alpha = self.w2_weight_scale / self.input_scale_2
        self.workspace = Nvfp4MoeWorkspace.allocate(
            max_tokens=MAX_TOKENS,
            top_k=TOP_K,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            device=torch.device("cuda"),
        )

    def launch(
        self,
        x,
        ids,
        weights,
        *,
        workspace=None,
        input_scale_1=None,
        g1_alpha=None,
        g1_alpha_up=None,
        global_routed_experts=GLOBAL_EXPERTS,
        local_routed_experts=LOCAL_EXPERTS,
        local_expert_start=0,
    ):
        output = torch.empty_like(x)
        launched = nvfp4_moe_sm120(
            x=x,
            topk_ids=ids,
            topk_weights=weights,
            w13_weight=self.w13,
            w2_weight=self.w2,
            w13_scale=self.w13_sf,
            w2_scale=self.w2_sf,
            input_scale_1=(
                self.input_scale_1 if input_scale_1 is None else input_scale_1
            ),
            input_scale_2=self.input_scale_2,
            g1_alpha=self.g1_alpha if g1_alpha is None else g1_alpha,
            g1_alpha_up=self.g1_alpha if g1_alpha_up is None else g1_alpha_up,
            g2_alpha=self.g2_alpha,
            global_routed_experts=global_routed_experts,
            local_routed_experts=local_routed_experts,
            local_expert_start=local_expert_start,
            output=output,
            workspace=self.workspace if workspace is None else workspace,
        )
        return output, launched

    def run(self, x, ids, weights, **kwargs):
        output, launched = self.launch(x, ids, weights, **kwargs)
        if not launched:
            raise RuntimeError("cooperative launch is unavailable")
        return output

    def reference(
        self,
        x,
        ids,
        weights,
        *,
        up_multiplier=None,
        gate_multiplier=None,
        global_routed_experts=GLOBAL_EXPERTS,
        local_routed_experts=LOCAL_EXPERTS,
        local_expert_start=0,
    ):
        x_dequant = _quant_roundtrip(x, self.input_scale_1)
        out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device="cuda")
        for token in range(x.shape[0]):
            for slot in range(TOP_K):
                global_expert = int(ids[token, slot])
                if global_expert < 0:
                    continue
                if global_expert < global_routed_experts:
                    local_expert = global_expert - local_expert_start
                else:
                    local_expert = (
                        global_expert - global_routed_experts + local_routed_experts
                    )
                if local_expert < 0 or local_expert >= LOCAL_EXPERTS:
                    continue
                fc1 = x_dequant[token] @ self.w13_ref[local_expert].T
                up, gate = fc1.split(INTERMEDIATE)
                if up_multiplier is not None:
                    up = up * up_multiplier[local_expert]
                if gate_multiplier is not None:
                    gate = gate * gate_multiplier[local_expert]
                act = torch.nn.functional.silu(gate) * up
                act = _quant_roundtrip(act[None], self.input_scale_2)[0]
                out[token] += float(weights[token, slot]) * (
                    act @ self.w2_ref[local_expert].T
                )
        return out


class TestNvfp4MoeSm120Metadata(unittest.TestCase):
    def test_registered_for_sm120(self):
        specs = registry.get("moe.nvfp4_fused_experts")
        self.assertEqual(len(specs), 1)
        requirement = next(iter(specs[0].capabilities))
        self.assertEqual(requirement.min_cuda_arch, (12, 0))
        self.assertEqual(requirement.max_cuda_arch, (12, 0))
        self.assertIn("returns whether", specs[0].format_signature.description)

    def test_kill_switch(self):
        self.assertTrue(hasattr(envs, "SGLANG_NVFP4_MOE_SM120"))
        with envs.SGLANG_NVFP4_MOE_SM120.override(False):
            self.assertFalse(nvfp4_moe_sm120_enabled())
        with envs.SGLANG_NVFP4_MOE_SM120.override(True):
            self.assertTrue(nvfp4_moe_sm120_enabled())

    def test_cuda_launch_errors_do_not_use_panicking_checks(self):
        source = (
            Path(__file__).resolve().parents[5]
            / "python/sglang/kernels/jit/csrc/moe/nvfp4_moe_sm120.cuh"
        ).read_text()
        launch_path = source[source.index("constexpr int kSharedBytes") :]
        for cuda_call in (
            "cudaFuncSetAttribute",
            "cudaDeviceGetAttribute",
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
            "cudaLaunchKernelEx",
        ):
            self.assertNotIn(f"RuntimeDeviceCheck({cuda_call}", launch_path)
        self.assertIn("return false;", launch_path)


@unittest.skipUnless(is_sm120_supported() and not is_sm121(), "requires SM120")
class TestNvfp4MoeSm120(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = _Fixture()

    @classmethod
    def tearDownClass(cls):
        del cls.fixture
        torch.cuda.empty_cache()

    @staticmethod
    def _routes(tokens: int, case: str):
        base = torch.tensor(
            [
                [0, 2, 4, 6],
                [1, 3, 5, 7],
                [0, 4, 2, 6],
                [1, 5, 3, 7],
            ],
            dtype=torch.int32,
            device="cuda",
        ).repeat(math.ceil(tokens / 4), 1)[:tokens]
        if case == "duplicate":
            base[:, 1] = base[:, 0]
        elif case == "skewed":
            base.fill_(0)
        elif case == "nonlocal_masked":
            base[:, 1] = LOCAL_EXPERTS
            base[:, 2] = -1
            base[:, 3] = GLOBAL_EXPERTS - 1
        weights = torch.rand(tokens, TOP_K, dtype=torch.float32, device="cuda")
        weights /= weights.sum(dim=-1, keepdim=True)
        return base, weights

    def test_numerics_and_layouts(self):
        torch.manual_seed(17)
        for tokens in (1, 4, 7, 8, 16):
            x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
            for case in ("balanced", "duplicate", "skewed", "nonlocal_masked"):
                with self.subTest(tokens=tokens, case=case):
                    ids, weights = self._routes(tokens, case)
                    actual = self.fixture.run(x, ids, weights).float()
                    reference = self.fixture.reference(x, ids, weights)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, reference, rtol=0.20, atol=0.025)

    def test_no_worse_than_cutlass(self):
        torch.manual_seed(29)
        for tokens in (1, 4, 16):
            x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
            ids, weights = self._routes(tokens, "balanced")
            identity_ids = ids
            current = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
            cutlass_fused_moe(
                input=x,
                token_selected_experts=identity_ids,
                token_final_scales=weights,
                fc1_expert_weights=self.fixture.w13.view(torch.long),
                fc2_expert_weights=self.fixture.w2.view(torch.long),
                output_dtype=torch.bfloat16,
                quant_scales=[
                    self.fixture.input_scale_1,
                    self.fixture.w13_sf.view(torch.int32),
                    self.fixture.g1_alpha,
                    self.fixture.input_scale_2,
                    self.fixture.w2_sf.view(torch.int32),
                    self.fixture.g2_alpha,
                ],
                output=current,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                activation_type=ActivationType.Swiglu,
                tune_max_num_tokens=1 << (tokens - 1).bit_length(),
                use_fused_finalize=True,
            )
            candidate = self.fixture.run(x, ids, weights).float()
            reference = self.fixture.reference(x, ids, weights)
            current_error = current.float() - reference
            candidate_error = candidate - reference
            self.assertLessEqual(
                candidate_error.norm() / reference.norm(),
                current_error.norm() / reference.norm() + 1e-3,
            )
            self.assertLessEqual(
                candidate_error.abs().max(), current_error.abs().max() + 1e-3
            )

    def test_graph_replay_keeps_addresses(self):
        torch.manual_seed(41)
        addresses = self.fixture.workspace.data_ptrs()
        for tokens in (1, 4, 16):
            x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
            ids, weights = self._routes(tokens, "balanced")
            self.fixture.run(x, ids, weights)
            self.assertTrue(self.fixture.workspace.graph_capture_supported)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.fixture.run(x, ids, weights)
            graph.replay()
            torch.cuda.synchronize()
            expected = output.clone()
            # Order-dependent GC of earlier tests' garbage can shrink the pool
            # mid-loop; settle Python-side garbage before the baseline so the
            # equality below measures only what replay itself allocates.
            gc.collect()
            allocated = torch.cuda.memory_allocated()
            for _ in range(10_000):
                graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(addresses, self.fixture.workspace.data_ptrs())
            self.assertEqual(allocated, torch.cuda.memory_allocated())
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_workspace_serializes_different_streams(self):
        workspace = self.fixture.workspace
        default_stream = torch.cuda.current_stream()
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        x_a = torch.randn(16, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
        x_b = torch.randn(16, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
        ids_a, weights_a = self._routes(16, "balanced")
        ids_b, weights_b = self._routes(16, "skewed")
        with torch.cuda.stream(stream_a):
            output_a, launched_a = self.fixture.launch(x_a, ids_a, weights_a)
        with torch.cuda.stream(stream_b):
            output_b, launched_b = self.fixture.launch(x_b, ids_b, weights_b)
        torch.cuda.synchronize()
        self.assertTrue(launched_a and launched_b)
        self.assertEqual(workspace._last_stream, stream_b.cuda_stream)
        torch.testing.assert_close(
            output_a.float(),
            self.fixture.reference(x_a, ids_a, weights_a),
            rtol=0.20,
            atol=0.025,
        )
        torch.testing.assert_close(
            output_b.float(),
            self.fixture.reference(x_b, ids_b, weights_b),
            rtol=0.20,
            atol=0.025,
        )
        workspace._bind_current_stream(default_stream)
        workspace._record_completion(default_stream)

    def test_underflowed_block_scale_quantizes_to_zero(self):
        x = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device="cuda")
        x[:, ::16] = 1e-6
        ids, weights = self._routes(1, "balanced")
        output = self.fixture.run(x, ids, weights)
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.count_nonzero(self.fixture.workspace.x_scale[:1]) == 0)
        self.assertTrue(torch.count_nonzero(self.fixture.workspace.x_q[:1]) == 0)

    def test_result_does_not_alias_workspace(self):
        x = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
        ids, weights = self._routes(1, "balanced")
        first = self.fixture.run(x, ids, weights)
        expected = first.clone()
        second = self.fixture.run(x, ids, weights)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertNotIn(first.data_ptr(), self.fixture.workspace.data_ptrs())
        torch.testing.assert_close(first, expected, rtol=0, atol=0)

    def test_gate_and_up_alphas_are_independent(self):
        x = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
        ids, weights = self._routes(4, "balanced")
        up_multiplier = torch.linspace(
            0.55, 1.45, LOCAL_EXPERTS, dtype=torch.float32, device="cuda"
        )
        gate_multiplier = torch.linspace(
            1.60, 0.70, LOCAL_EXPERTS, dtype=torch.float32, device="cuda"
        )
        actual = self.fixture.run(
            x,
            ids,
            weights,
            g1_alpha=self.fixture.g1_alpha * gate_multiplier,
            g1_alpha_up=self.fixture.g1_alpha * up_multiplier,
        ).float()
        reference = self.fixture.reference(
            x,
            ids,
            weights,
            up_multiplier=up_multiplier,
            gate_multiplier=gate_multiplier,
        )
        swapped = self.fixture.run(
            x,
            ids,
            weights,
            g1_alpha=self.fixture.g1_alpha * up_multiplier,
            g1_alpha_up=self.fixture.g1_alpha * gate_multiplier,
        ).float()
        actual_error = (actual - reference).norm()
        swapped_error = (swapped - reference).norm()
        torch.testing.assert_close(actual, reference, rtol=0.20, atol=0.025)
        self.assertLess(actual_error, swapped_error * 0.8)

    def test_row_distinct_block_scale_selectors(self):
        scale_bytes = self.fixture.w13_sf[0].view(torch.uint8)
        self.assertGreater(torch.unique(scale_bytes).numel(), 8)
        x = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
        ids = torch.zeros(1, TOP_K, dtype=torch.int32, device="cuda")
        weights = torch.zeros(1, TOP_K, dtype=torch.float32, device="cuda")
        weights[:, 0] = 1.0
        actual = self.fixture.run(x, ids, weights).float()
        reference = self.fixture.reference(x, ids, weights)
        torch.testing.assert_close(actual, reference, rtol=0.15, atol=0.025)

    def test_physical_expert_ids_map_from_current_topology(self):
        x = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
        ids, weights = self._routes(4, "balanced")
        ids = ids + LOCAL_EXPERTS
        actual = self.fixture.run(
            x,
            ids,
            weights,
            local_expert_start=LOCAL_EXPERTS,
        ).float()
        reference = self.fixture.reference(
            x,
            ids,
            weights,
            local_expert_start=LOCAL_EXPERTS,
        )
        torch.testing.assert_close(actual, reference, rtol=0.20, atol=0.025)

    def test_out_of_range_expert_is_not_silently_masked(self):
        x = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
        ids, weights = self._routes(1, "balanced")
        ids[0, 0] = GLOBAL_EXPERTS + LOCAL_EXPERTS
        output = self.fixture.run(x, ids, weights)
        self.assertTrue(torch.isnan(output[0]).all())

    def test_workspace_and_scale_validation(self):
        x = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
        ids, weights = self._routes(1, "balanced")
        bad_workspace = replace(
            self.fixture.workspace,
            x_q=torch.empty(1, dtype=torch.uint8, device="cuda"),
        )
        with self.assertRaisesRegex(Exception, "x_q workspace is too small"):
            self.fixture.run(x, ids, weights, workspace=bad_workspace)
        with self.assertRaisesRegex(Exception, "input_scale_1 must be scalar"):
            self.fixture.run(
                x,
                ids,
                weights,
                input_scale_1=torch.ones(2, dtype=torch.float32, device="cuda"),
            )

    def test_direct_api_rejects_above_max_tokens(self):
        x = torch.randn(MAX_TOKENS + 1, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
        ids, weights = self._routes(MAX_TOKENS + 1, "balanced")
        with self.assertRaisesRegex(ValueError, "workspace holds 16 tokens"):
            self.fixture.run(x, ids, weights)

    def test_flashinfer_cutlass_small_row_dispatch(self):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            FlashInferCutlassMoeQuantInfo,
            _run_flashinfer_cutlass,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        x = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
        ids, weights = self._routes(4, "balanced")
        dispatch = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(weights, ids, None),
        )
        quant_info = FlashInferCutlassMoeQuantInfo(
            quant_type="fp4",
            w13_weight=self.fixture.w13,
            w2_weight=self.fixture.w2,
            output_dtype=torch.bfloat16,
            quant_scales=[
                self.fixture.input_scale_1,
                self.fixture.w13_sf,
                self.fixture.g1_alpha,
                self.fixture.input_scale_2,
                self.fixture.w2_sf,
                self.fixture.g2_alpha,
            ],
            apply_routed_scaling_factor=False,
            g1_alpha_up=self.fixture.g1_alpha,
            moe_ep_size=2,
            smallm_workspace=self.fixture.workspace,
            smallm_global_routed_experts=GLOBAL_EXPERTS,
            smallm_local_routed_experts=LOCAL_EXPERTS,
            smallm_local_expert_start=0,
        )
        runner_config = MoeRunnerConfig(
            num_experts=GLOBAL_EXPERTS,
            num_local_experts=LOCAL_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            top_k=TOP_K,
            activation="silu",
            is_gated=True,
        )
        cutlass = mock.Mock(side_effect=AssertionError("unexpected cutlass fallback"))
        with (
            mock.patch(
                "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
                "_flashinfer_cutlass_fused_moe",
                return_value=(cutlass, object()),
            ),
            mock.patch(
                "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass.get_tp_group",
                return_value=None,
            ),
            mock.patch(
                "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
                "use_symmetric_memory",
                return_value=nullcontext(),
            ),
        ):
            actual = _run_flashinfer_cutlass(
                dispatch_output=dispatch,
                quant_info=quant_info,
                runner_config=runner_config,
            )
        self.assertFalse(cutlass.called)
        reference = self.fixture.reference(x, ids, weights)
        torch.testing.assert_close(actual.float(), reference, rtol=0.20, atol=0.025)

    def test_flashinfer_cutlass_fallback_above_max_tokens(self):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            FlashInferCutlassMoeQuantInfo,
            _run_flashinfer_cutlass,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        tokens = MAX_TOKENS + 1
        x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
        ids, weights = self._routes(tokens, "balanced")
        dispatch = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(weights, ids, None),
        )
        quant_info = FlashInferCutlassMoeQuantInfo(
            quant_type="fp4",
            w13_weight=self.fixture.w13,
            w2_weight=self.fixture.w2,
            output_dtype=torch.bfloat16,
            quant_scales=[
                self.fixture.input_scale_1,
                self.fixture.w13_sf,
                self.fixture.g1_alpha,
                self.fixture.input_scale_2,
                self.fixture.w2_sf,
                self.fixture.g2_alpha,
            ],
            apply_routed_scaling_factor=False,
            g1_alpha_up=self.fixture.g1_alpha,
            moe_ep_size=2,
            smallm_workspace=self.fixture.workspace,
            smallm_global_routed_experts=GLOBAL_EXPERTS,
            smallm_local_routed_experts=LOCAL_EXPERTS,
            smallm_local_expert_start=0,
        )
        runner_config = MoeRunnerConfig(
            num_experts=GLOBAL_EXPERTS,
            num_local_experts=LOCAL_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            top_k=TOP_K,
            activation="silu",
            is_gated=True,
        )
        cutlass_output = torch.empty_like(x)
        cutlass = mock.Mock(return_value=(cutlass_output,))
        with (
            mock.patch(
                "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
                "_flashinfer_cutlass_fused_moe",
                return_value=(cutlass, object()),
            ),
            mock.patch(
                "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
                "_activation_type",
                return_value=object(),
            ),
            mock.patch(
                "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass.get_tp_group",
                return_value=None,
            ),
            mock.patch(
                "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
                "use_symmetric_memory",
                return_value=nullcontext(),
            ),
        ):
            actual = _run_flashinfer_cutlass(
                dispatch_output=dispatch,
                quant_info=quant_info,
                runner_config=runner_config,
            )
        self.assertTrue(cutlass.called)
        self.assertEqual(actual.data_ptr(), cutlass_output.data_ptr())

    def test_flashinfer_cutlass_fallback_for_unsupported_config(self):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
            FlashInferCutlassMoeQuantInfo,
            _run_flashinfer_cutlass,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        x = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="cuda") / 8
        ids, weights = self._routes(4, "balanced")
        dispatch = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(weights, ids, None),
        )
        quant_info = FlashInferCutlassMoeQuantInfo(
            quant_type="fp4",
            w13_weight=self.fixture.w13,
            w2_weight=self.fixture.w2,
            output_dtype=torch.bfloat16,
            quant_scales=[
                self.fixture.input_scale_1,
                self.fixture.w13_sf,
                self.fixture.g1_alpha,
                self.fixture.input_scale_2,
                self.fixture.w2_sf,
                self.fixture.g2_alpha,
            ],
            apply_routed_scaling_factor=False,
            g1_alpha_up=self.fixture.g1_alpha,
            moe_ep_size=2,
            smallm_workspace=self.fixture.workspace,
            smallm_global_routed_experts=GLOBAL_EXPERTS,
            smallm_local_routed_experts=LOCAL_EXPERTS,
            smallm_local_expert_start=0,
        )
        cutlass_output = torch.empty_like(x)
        cutlass = mock.Mock(return_value=(cutlass_output,))
        for runner_config, local_start, capture_without_support in (
            (
                MoeRunnerConfig(
                    num_experts=GLOBAL_EXPERTS,
                    num_local_experts=LOCAL_EXPERTS,
                    hidden_size=HIDDEN,
                    intermediate_size_per_partition=INTERMEDIATE,
                    top_k=TOP_K,
                    activation="silu",
                    is_gated=True,
                    swiglu_limit=1.0,
                ),
                0,
                False,
            ),
            (
                MoeRunnerConfig(
                    num_experts=GLOBAL_EXPERTS,
                    num_local_experts=LOCAL_EXPERTS,
                    hidden_size=HIDDEN,
                    intermediate_size_per_partition=INTERMEDIATE,
                    top_k=TOP_K,
                    activation="silu",
                    is_gated=True,
                ),
                1,
                False,
            ),
            (
                MoeRunnerConfig(
                    num_experts=GLOBAL_EXPERTS,
                    num_local_experts=LOCAL_EXPERTS,
                    hidden_size=HIDDEN,
                    intermediate_size_per_partition=INTERMEDIATE,
                    top_k=TOP_K,
                    activation="silu",
                    is_gated=True,
                ),
                0,
                True,
            ),
        ):
            with self.subTest(
                swiglu_limit=runner_config.swiglu_limit,
                local_start=local_start,
                capture_without_support=capture_without_support,
            ):
                cutlass.reset_mock()
                quant_info.smallm_local_expert_start = local_start
                previous_graph_support = self.fixture.workspace.graph_capture_supported
                if capture_without_support:
                    self.fixture.workspace.graph_capture_supported = False
                try:
                    with (
                        mock.patch(
                            "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
                            "_flashinfer_cutlass_fused_moe",
                            return_value=(cutlass, object()),
                        ),
                        mock.patch(
                            "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
                            "_activation_type",
                            return_value=object(),
                        ),
                        mock.patch(
                            "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
                            "get_tp_group",
                            return_value=None,
                        ),
                        mock.patch(
                            "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
                            "use_symmetric_memory",
                            return_value=nullcontext(),
                        ),
                        mock.patch(
                            "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
                            "get_kernel",
                            side_effect=AssertionError("custom kernel must not launch"),
                        ),
                        mock.patch("torch.cuda.current_stream") as current_stream,
                        mock.patch(
                            "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
                            "_is_stream_capturing",
                            return_value=capture_without_support,
                        ),
                        mock.patch(
                            "torch.cuda.is_current_stream_capturing",
                            side_effect=AssertionError(
                                "the torch capture probe must not run"
                            ),
                        ),
                    ):
                        current_stream.return_value = object()
                        actual = _run_flashinfer_cutlass(
                            dispatch_output=dispatch,
                            quant_info=quant_info,
                            runner_config=runner_config,
                        )
                finally:
                    self.fixture.workspace.graph_capture_supported = (
                        previous_graph_support
                    )
                self.assertTrue(cutlass.called)
                self.assertEqual(actual.data_ptr(), cutlass_output.data_ptr())


if __name__ == "__main__":
    unittest.main()
