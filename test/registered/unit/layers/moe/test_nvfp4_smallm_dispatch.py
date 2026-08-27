"""CPU-reachable coverage for the NVFP4 small-row dispatch gates."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    _run_flashinfer_cutlass,
    _smallm_ineligibility_reason,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    _nvfp4_smallm_load_eligible,
    _prepare_nvfp4_smallm_workspace,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _runtime_call(**overrides):
    workspace = SimpleNamespace(
        graph_capture_supported=True,
        max_tokens=16,
        hidden_size=8,
        top_k=2,
    )
    quant_info = SimpleNamespace(
        quant_type="fp4",
        g1_alpha_up=object(),
        smallm_global_routed_experts=8,
        smallm_local_routed_experts=4,
        smallm_local_expert_start=0,
        moe_tp_size=1,
        moe_ep_size=2,
        moe_ep_rank=0,
        w13_weight=torch.zeros(4, 16, 8, dtype=torch.uint8),
    )
    runner_config = SimpleNamespace(
        is_gated=True,
        activation="silu",
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        swiglu_limit=None,
    )
    kwargs = dict(
        quant_info=quant_info,
        workspace=workspace,
        x=torch.zeros(1, 8, dtype=torch.bfloat16),
        x_sf=None,
        topk_ids=torch.zeros(1, 2, dtype=torch.int32),
        topk_weights=torch.zeros(1, 2, dtype=torch.float32),
        output_dtype=torch.bfloat16,
        quant_scales=[object()] * 6,
        runner_config=runner_config,
        output_supplied=False,
        enable_alltoall=False,
        capturing=False,
    )
    targets = {
        "quant_info": quant_info,
        "workspace": workspace,
        "runner_config": runner_config,
    }
    for key, value in overrides.items():
        target_name, separator, attribute = key.partition("__")
        if separator:
            setattr(targets[target_name], attribute, value)
        else:
            kwargs[key] = value
    return _smallm_ineligibility_reason(**kwargs)


def _load_call(**overrides):
    kwargs = dict(
        enable_flashinfer_cutlass_moe=True,
        kernel_supported=True,
        kernel_enabled=True,
        use_per_token_activation=False,
        is_gated=True,
        activation="silu",
        hidden_size=256,
        intermediate_size=64,
        num_local_experts=8,
        num_fused_shared_experts=0,
        moe_ep_rank=0,
        moe_ep_size=1,
        expert_storage_rank=0,
        num_global_routed_experts=8,
        top_k=4,
    )
    kwargs.update(overrides)
    return _nvfp4_smallm_load_eligible(**kwargs)


class TestSmallmLoadGate(unittest.TestCase):
    def test_supported_shape_is_eligible(self):
        self.assertTrue(_load_call())

    def test_each_unsupported_load_property_is_refused(self):
        cases = {
            "backend": {"enable_flashinfer_cutlass_moe": False},
            "architecture": {"kernel_supported": False},
            "kill_switch": {"kernel_enabled": False},
            "per_token_activation": {"use_per_token_activation": True},
            "ungated": {"is_gated": False},
            "activation": {"activation": "gelu"},
            "hidden_alignment": {"hidden_size": 384},
            "intermediate_alignment": {"intermediate_size": 96},
            "expert_limit": {"num_local_experts": 513},
            "no_routed_experts": {"num_fused_shared_experts": 8},
            "negative_ep_rank": {"moe_ep_rank": -1},
            "high_ep_rank": {"moe_ep_rank": 1},
            "parallel_ep": {"moe_ep_size": 2, "num_global_routed_experts": 16},
            "storage_rank": {"expert_storage_rank": 1},
            "nonuniform_ep": {"num_global_routed_experts": 7},
            "missing_top_k": {"top_k": None},
        }
        for name, case_overrides in cases.items():
            with self.subTest(name=name):
                self.assertFalse(_load_call(**case_overrides))

    def test_jit_prepare_failure_disables_workspace(self):
        layer = SimpleNamespace(nvfp4_smallm_workspace=object())
        with (
            mock.patch(
                "sglang.kernels.ops.moe.nvfp4_moe_sm120." "prepare_nvfp4_moe_sm120",
                side_effect=RuntimeError("nvcc failed"),
            ),
            mock.patch(
                "sglang.srt.layers.quantization.modelopt_quant.logger.warning_once"
            ) as warning_once,
        ):
            _prepare_nvfp4_smallm_workspace(
                layer=layer,
                max_tokens=16,
                top_k=4,
                hidden_size=256,
                intermediate_size=64,
                device=torch.device("cpu"),
            )
        self.assertIsNone(layer.nvfp4_smallm_workspace)
        warning_once.assert_called_once()
        self.assertIn("using CUTLASS", warning_once.call_args.args[0])


class TestSmallmIneligibilityReason(unittest.TestCase):
    def test_supported_runtime_inputs_are_eligible(self):
        self.assertIsNone(_runtime_call())

    def test_early_refusal_branches(self):
        cases = (
            ({"quant_info__quant_type": "fp8"}, "quantization is not NVFP4"),
            (
                {"workspace": None},
                "workspace is disabled or the layer shape is unsupported",
            ),
            (
                {"quant_info__g1_alpha_up": None},
                "the up-projection alpha is unavailable",
            ),
            (
                {"quant_info__smallm_local_routed_experts": None},
                "expert topology is unavailable",
            ),
            (
                {"output_supplied": True},
                "the all-to-all output contract requires CUTLASS",
            ),
            (
                {"enable_alltoall": True},
                "the all-to-all output contract requires CUTLASS",
            ),
            ({"x_sf": torch.zeros(1)}, "the input is already quantized"),
        )
        for case_overrides, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(_runtime_call(**case_overrides), expected)

    def test_capture_without_cooperative_support_is_refused(self):
        with (
            mock.patch(
                "torch.cuda.is_current_stream_capturing",
                side_effect=AssertionError("the torch capture probe must not run"),
            ),
        ):
            self.assertEqual(
                _runtime_call(workspace__graph_capture_supported=False, capturing=True),
                "cooperative graph capture is unavailable",
            )

    def test_token_range_is_refused(self):
        self.assertEqual(
            _runtime_call(x=torch.zeros(0, 8, dtype=torch.bfloat16)),
            "token count is outside the small-row range",
        )
        self.assertEqual(
            _runtime_call(x=torch.zeros(17, 8, dtype=torch.bfloat16)),
            "token count is outside the small-row range",
        )

    def test_input_and_routing_dtypes_are_refused(self):
        for override in (
            {"x": torch.zeros(1, 8)},
            {"topk_ids": torch.zeros(1, 2, dtype=torch.int64)},
            {"topk_weights": torch.zeros(1, 2, dtype=torch.bfloat16)},
        ):
            with self.subTest(override=next(iter(override))):
                self.assertEqual(
                    _runtime_call(**override),
                    "input or routing dtypes are unsupported",
                )

    def test_noncontiguous_inputs_are_refused(self):
        inputs = (
            {"x": torch.zeros(1, 16, dtype=torch.bfloat16)[:, ::2]},
            {"topk_ids": torch.zeros(1, 4, dtype=torch.int32)[:, ::2]},
            {"topk_weights": torch.zeros(1, 4, dtype=torch.float32)[:, ::2]},
        )
        for override in inputs:
            with self.subTest(override=next(iter(override))):
                self.assertEqual(
                    _runtime_call(**override),
                    "input or routing tensors are not contiguous",
                )

    def test_shape_mismatches_are_refused(self):
        inputs = (
            {"x": torch.zeros(1, 2, 4, dtype=torch.bfloat16)},
            {"topk_ids": torch.zeros(2, dtype=torch.int32)},
            {"topk_weights": torch.zeros(1, 3, dtype=torch.float32)},
            {"topk_ids": torch.zeros(2, 2, dtype=torch.int32)},
            {"x": torch.zeros(1, 16, dtype=torch.bfloat16)},
            {"topk_ids": torch.zeros(1, 3, dtype=torch.int32)},
        )
        for override in inputs:
            with self.subTest(override=next(iter(override))):
                self.assertEqual(
                    _runtime_call(**override),
                    "input or routing shapes do not match the workspace",
                )

    def test_activation_contract_is_refused(self):
        for override in (
            {"runner_config__is_gated": False},
            {"runner_config__activation": "gelu"},
        ):
            with self.subTest(override=next(iter(override))):
                self.assertEqual(
                    _runtime_call(**override), "the activation is unsupported"
                )

    def test_each_activation_modifier_is_refused(self):
        for attribute in (
            "gemm1_alpha",
            "gemm1_beta",
            "gemm1_clamp_limit",
            "swiglu_limit",
        ):
            with self.subTest(attribute=attribute):
                self.assertEqual(
                    _runtime_call(**{f"runner_config__{attribute}": 1.0}),
                    "the activation modifiers are unsupported",
                )

    def test_output_and_scale_contracts_are_refused(self):
        self.assertEqual(
            _runtime_call(output_dtype=torch.float16),
            "the output dtype is unsupported",
        )
        self.assertEqual(
            _runtime_call(quant_scales=None), "the NVFP4 scale set is incomplete"
        )
        self.assertEqual(
            _runtime_call(quant_scales=[object()] * 5),
            "the NVFP4 scale set is incomplete",
        )

    def test_parallel_topology_bounds_are_refused(self):
        for override in (
            {"quant_info__moe_tp_size": 0},
            {"quant_info__moe_ep_size": 0},
        ):
            with self.subTest(override=next(iter(override))):
                self.assertEqual(
                    _runtime_call(**override), "the MoE parallel topology is invalid"
                )

    def test_each_uniform_ep_shard_constraint_is_refused(self):
        cases = (
            {"quant_info__smallm_global_routed_experts": 0},
            {"quant_info__smallm_local_routed_experts": 0},
            {"quant_info__smallm_local_expert_start": -1},
            {"quant_info__moe_ep_rank": -1},
            {"quant_info__moe_ep_rank": 2},
            {"quant_info__smallm_global_routed_experts": 7},
            {"quant_info__smallm_local_expert_start": 1},
            {"quant_info__smallm_local_routed_experts": 5},
        )
        for override in cases:
            with self.subTest(override=next(iter(override))):
                self.assertEqual(
                    _runtime_call(**override),
                    "the expert topology is not a uniform contiguous EP shard",
                )


class TestSmallmRuntimeFallback(unittest.TestCase):
    def _run(self, kernel_effect, *, capturing=False, capture_failure=None):
        x = torch.zeros(1, 8, dtype=torch.bfloat16)
        ids = torch.zeros(1, 2, dtype=torch.int32)
        weights = torch.zeros(1, 2, dtype=torch.float32)
        output = torch.full_like(x, 7)
        workspace = SimpleNamespace(
            graph_capture_supported=True,
            max_tokens=16,
            hidden_size=8,
            top_k=2,
        )
        quant_info = SimpleNamespace(
            quant_type="fp4",
            g1_alpha_up=torch.ones(4),
            smallm_global_routed_experts=8,
            smallm_local_routed_experts=4,
            smallm_local_expert_start=0,
            moe_tp_size=1,
            moe_tp_rank=0,
            moe_ep_size=2,
            moe_ep_rank=0,
            apply_routed_scaling_factor=False,
            w13_weight=torch.zeros(4, 16, 8, dtype=torch.uint8),
            w2_weight=torch.zeros(4, 8, 8, dtype=torch.uint8),
            output_dtype=torch.bfloat16,
            quant_scales=[
                torch.ones(1),
                torch.zeros(4, dtype=torch.uint8),
                torch.ones(4),
                torch.ones(1),
                torch.zeros(4, dtype=torch.uint8),
                torch.ones(4),
            ],
            smallm_workspace=workspace,
        )
        runner_config = SimpleNamespace(
            is_gated=True,
            activation="silu",
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            swiglu_limit=None,
            routed_scaling_factor=None,
        )
        dispatch = SimpleNamespace(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=SimpleNamespace(topk_ids=ids, topk_weights=weights),
        )
        cutlass = mock.Mock(return_value=(output,))
        if isinstance(kernel_effect, Exception):
            custom_kernel = mock.Mock(side_effect=kernel_effect)
        else:
            custom_kernel = mock.Mock(return_value=False)
        stream = object()
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
            mock.patch(
                "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass.get_kernel",
                return_value=custom_kernel,
            ),
            mock.patch(
                "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
                "_log_smallm_decision"
            ) as log_decision,
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.current_stream", return_value=stream),
            mock.patch(
                "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
                "_is_stream_capturing",
                return_value=capturing,
            ),
            mock.patch(
                "torch.cuda.is_current_stream_capturing",
                side_effect=AssertionError("the torch capture probe must not run"),
            ),
        ):
            if capture_failure is None:
                actual = _run_flashinfer_cutlass(
                    dispatch_output=dispatch,
                    quant_info=quant_info,
                    runner_config=runner_config,
                )
            else:
                with self.assertRaisesRegex(RuntimeError, capture_failure):
                    _run_flashinfer_cutlass(
                        dispatch_output=dispatch,
                        quant_info=quant_info,
                        runner_config=runner_config,
                    )
                cutlass.assert_not_called()
                return ""
        self.assertEqual(actual.data_ptr(), output.data_ptr())
        cutlass.assert_called_once()
        log_decision.assert_called_once()
        return log_decision.call_args.args[0]

    def test_jit_or_launch_exception_falls_back_once(self):
        message = self._run(RuntimeError("ptxas failed"))
        self.assertIn("using CUTLASS", message)
        self.assertIn("JIT launch failed", message)

    def test_false_launch_result_falls_back_once(self):
        message = self._run(None)
        self.assertIn("using CUTLASS", message)
        self.assertIn("launch is unavailable", message)

    def test_capture_launch_exception_does_not_fall_back(self):
        self._run(
            RuntimeError("cooperative launch failed"),
            capturing=True,
            capture_failure="failed during CUDA graph capture",
        )

    def test_capture_false_launch_result_does_not_fall_back(self):
        self._run(
            None,
            capturing=True,
            capture_failure="failed during CUDA graph capture",
        )


class TestSm121Refusal(unittest.TestCase):
    """GB10 (SM121) shares the SM120 capability major but has no validated
    cubins; the JIT guard must refuse it even when SM120 support reads true."""

    def test_jit_module_refuses_sm121(self):
        from unittest import mock

        from sglang.kernels.ops.moe import nvfp4_moe_sm120 as ops_mod

        with (
            mock.patch.object(ops_mod, "is_sm120_supported", return_value=True),
            mock.patch.object(ops_mod, "is_sm121", return_value=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "SM121"):
                ops_mod._jit_nvfp4_moe_module.__wrapped__(2560, 320, 10)


class TestNvfp4WrapperCaptureProbe(unittest.TestCase):
    def test_wrapper_uses_cuda_runtime_capture_status(self):
        from sglang.kernels.ops.moe import nvfp4_moe_sm120 as ops_mod

        stream = object()
        workspace = SimpleNamespace(
            max_tokens=16,
            top_k=2,
            hidden_size=8,
            intermediate_size=16,
            graph_capture_supported=False,
        )
        x = torch.zeros(1, 8, dtype=torch.bfloat16)
        topk_ids = torch.zeros(1, 2, dtype=torch.int32)
        topk_weights = torch.zeros(1, 2, dtype=torch.float32)
        w2_weight = torch.zeros(1, 1, 8, dtype=torch.uint8)
        with (
            mock.patch("torch.cuda.current_stream", return_value=stream),
            mock.patch.object(
                ops_mod, "_is_stream_capturing", return_value=True
            ) as capture_probe,
            mock.patch(
                "torch.cuda.is_current_stream_capturing",
                side_effect=AssertionError("the torch capture probe must not run"),
            ),
        ):
            launched = ops_mod.nvfp4_moe_sm120(
                x=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                w13_weight=None,
                w2_weight=w2_weight,
                w13_scale=None,
                w2_scale=None,
                input_scale_1=None,
                input_scale_2=None,
                g1_alpha=None,
                g1_alpha_up=None,
                g2_alpha=None,
                global_routed_experts=8,
                local_routed_experts=8,
                local_expert_start=0,
                output=None,
                workspace=workspace,
            )

        self.assertFalse(launched)
        capture_probe.assert_called_once_with(stream)


if __name__ == "__main__":
    unittest.main()
