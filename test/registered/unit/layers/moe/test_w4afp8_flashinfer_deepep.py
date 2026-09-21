"""CPU checks for the FlashInfer W4AFP8 DeepEP-normal adapter contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLDispatchOutput,
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
)
from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend
from sglang.srt.layers.quantization.w4afp8 import W4AFp8MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestFlashInferW4AFP8DeepEPNormal(unittest.TestCase):
    def payload(self, rank):
        return runner.FlashInferCutlassMoeQuantInfo(
            quant_type="w4afp8",
            w13_weight=torch.empty(32, 0),
            w2_weight=torch.empty(32, 0),
            quant_scales=[torch.empty(0) for _ in range(8)],
            moe_tp_size=1,
            moe_tp_rank=0,
            moe_ep_size=8,
            moe_ep_rank=rank,
        )

    def dispatch(self, tokens=3):
        x = torch.ones(tokens, 128, dtype=torch.bfloat16)
        ids = torch.tensor([[0, -1], [31, 2], [-1, -1]])[:tokens]
        weights = torch.tensor(
            [[0.25, float("nan")], [0.75, 0.125], [float("nan"), 1.0]]
        )[:tokens]
        if tokens:
            x[-1].fill_(float("nan"))
        return DeepEPNormalDispatchOutput(x, None, ids, weights, [1] * 32)

    def test_local_ids_padding_and_external_routed_scaling(self):
        for rank in (0, 3, 7):
            with self.subTest(rank=rank):
                dispatch = self.dispatch()
                config = MoeRunnerConfig(routed_scaling_factor=2.5)
                q = self.payload(rank)

                def kernel(standard, payload, call_config, *, symmetric_output):
                    self.assertFalse(symmetric_output)
                    self.assertIs(payload, q)
                    self.assertIsNone(call_config.routed_scaling_factor)
                    offset = rank * 32
                    remote = 0 if rank else 32
                    torch.testing.assert_close(
                        standard.topk_output.topk_ids,
                        torch.tensor(
                            [
                                [offset, remote],
                                [offset + 31, offset + 2],
                                [remote, remote],
                            ],
                            dtype=torch.int32,
                        ),
                    )
                    torch.testing.assert_close(
                        standard.topk_output.topk_weights,
                        torch.tensor([[0.25, 0], [0.75, 0.125], [0, 0]]),
                    )
                    self.assertTrue(torch.isfinite(standard.hidden_states).all())
                    self.assertEqual(standard.hidden_states[-1].count_nonzero(), 0)
                    result = torch.full_like(standard.hidden_states, 4)
                    result[-1].fill_(float("nan"))
                    return result

                with patch.object(runner, "_run_flashinfer_w4afp8", side_effect=kernel):
                    output = runner.fused_experts_deepep_to_flashinfer_cutlass(
                        dispatch, q, config
                    )
                self.assertIsInstance(output, DeepEPNormalCombineInput)
                self.assertIs(output.topk_ids, dispatch.topk_ids)
                self.assertIs(output.topk_weights, dispatch.topk_weights)
                torch.testing.assert_close(
                    output.hidden_states[:2],
                    torch.full((2, 128), 4, dtype=torch.bfloat16),
                )
                self.assertEqual(output.hidden_states[-1].count_nonzero(), 0)
                self.assertEqual(config.routed_scaling_factor, 2.5)

    def test_empty_input_does_not_launch(self):
        with patch.object(runner, "_flashinfer_cutlass_fused_moe") as kernel:
            output = runner.fused_experts_deepep_to_flashinfer_cutlass(
                self.dispatch(0), self.payload(3), MoeRunnerConfig()
            )
        kernel.assert_not_called()
        self.assertEqual(output.hidden_states.shape, (0, 128))

    def test_rejects_unsupported_payloads_and_layouts(self):
        dispatch = self.dispatch()
        ll = DeepEPLLDispatchOutput(
            dispatch.hidden_states,
            None,
            dispatch.topk_ids,
            dispatch.topk_weights,
            torch.zeros(32),
            1,
        )
        cases = [
            (ll, self.payload(3), "DeepEP buffers"),
            (dispatch, SimpleNamespace(), "native W4AFP8"),
            (
                dispatch._replace(hidden_states_scale=torch.ones(1)),
                self.payload(3),
                "BF16",
            ),
            (
                dispatch._replace(hidden_states=dispatch.hidden_states.float()),
                self.payload(3),
                "BF16",
            ),
        ]
        for d, q, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                runner.fused_experts_deepep_to_flashinfer_cutlass(
                    d, q, MoeRunnerConfig()
                )

    def test_preprocessing_sets_bf16_dispatch(self):
        method = W4AFp8MoEMethod(SimpleNamespace())
        method.use_flashinfer = True
        layer = SimpleNamespace(dispatcher=Mock())
        with patch.object(method, "_process_flashinfer_weights"):
            method.process_weights_after_loading(layer)
        layer.dispatcher.set_quant_config.assert_called_once_with(
            {
                "normal_dispatcher_output_dtype": "bf16",
                "low_latency_dispatcher_output_dtype": "bf16",
            }
        )

    def test_layer_selects_registered_runner(self):
        from sglang.srt.layers.moe.ep_moe import layer as ep_layer
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config

        with (
            patch.object(ep_layer.FusedMoE, "__init__", return_value=None),
            patch.object(
                ep_layer,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_CUTLASS,
            ),
            patch.object(
                ep_layer, "get_moe_a2a_backend", return_value=MoeA2ABackend.DEEPEP
            ),
        ):
            layer = ep_layer.DeepEPMoE(
                256, 8, 6144, 2048, 0, quant_config=W4AFp8Config()
            )
        self.assertTrue(layer.deprecate_flag)
        dispatch = self.dispatch()
        with patch.object(
            ep_layer.FusedMoE, "run_moe_core", return_value="registered"
        ) as core:
            self.assertEqual(layer.run_moe_core(dispatch), "registered")
        core.assert_called_once_with(dispatch)

    def test_rejects_explicit_fp8_dispatch(self):
        method = W4AFp8MoEMethod(SimpleNamespace())
        with (
            patch(
                "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                return_value=MoeA2ABackend.DEEPEP,
            ),
            patch(
                "sglang.srt.layers.moe.utils.get_deepep_mode",
                return_value=DeepEPMode.NORMAL,
            ),
            patch(
                "sglang.srt.runtime_context.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(deepep_dispatcher_output_dtype="fp8")
                ),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "BF16"):
                method._validate_flashinfer_config()

    def test_low_latency_counts_padding_and_unweighted_combine(self):
        for rank in (0, 3, 7):
            for counts in ([0, 0], [3, 1], [1, 2]):
                with self.subTest(rank=rank, counts=counts):
                    q = self.payload(rank)
                    q.w13_weight = torch.empty(2, 0)
                    q.w2_weight = torch.empty(2, 0)
                    x = torch.full((2, 3, 128), float("nan"), dtype=torch.bfloat16)
                    for expert, count in enumerate(counts):
                        x[expert, :count] = expert + 1
                    topk_ids = torch.tensor([[rank * 2, rank * 2 + 1]])
                    topk_weights = torch.tensor([[0.2, 0.8]])
                    dispatch = DeepEPLLDispatchOutput(
                        x, None, topk_ids, topk_weights, torch.tensor(counts), 1
                    )

                    def kernel(standard, payload, config, *, symmetric_output):
                        self.assertFalse(symmetric_output)
                        self.assertIsNone(config.routed_scaling_factor)
                        self.assertTrue(torch.isfinite(standard.hidden_states).all())
                        live = torch.arange(3)[None, :] < torch.tensor(counts)[:, None]
                        expected_ids = (
                            torch.where(
                                live,
                                torch.arange(2)[:, None] + rank * 2,
                                0 if rank else 2,
                            )
                            .int()
                            .reshape(-1, 1)
                        )
                        torch.testing.assert_close(
                            standard.topk_output.topk_ids, expected_ids
                        )
                        torch.testing.assert_close(
                            standard.topk_output.topk_weights,
                            live.reshape(-1, 1).float(),
                        )
                        return standard.hidden_states * 3

                    with patch.object(
                        runner, "_run_flashinfer_w4afp8", side_effect=kernel
                    ):
                        result = runner.fused_experts_deepep_to_flashinfer_cutlass(
                            dispatch, q, MoeRunnerConfig(routed_scaling_factor=2.5)
                        )
                    self.assertEqual(result.hidden_states.shape, x.shape)
                    self.assertIs(result.topk_weights, topk_weights)
                    self.assertIs(result.topk_ids, topk_ids)
                    for expert, count in enumerate(counts):
                        torch.testing.assert_close(
                            result.hidden_states[expert, :count],
                            torch.full(
                                (count, 128), 3 * (expert + 1), dtype=torch.bfloat16
                            ),
                        )
                        self.assertEqual(
                            result.hidden_states[expert, count:].count_nonzero(), 0
                        )

    def test_deepep_skips_only_moe_autotuning(self):
        from sglang.srt.model_executor.runner import flashinfer_autotune as tuning

        for a2a in ("none", "deepep"):
            context = SimpleNamespace(
                kernel=SimpleNamespace(flashinfer_autotune_skip_ops=["user_op"]),
                moe=SimpleNamespace(
                    moe_runner_backend="flashinfer_cutlass", moe_a2a_backend=a2a
                ),
            )
            with patch.object(tuning, "get_exec", return_value=context):
                actual = tuning.get_flashinfer_autotune_skip_ops(SimpleNamespace())
            expected = {"user_op"}
            if a2a == "deepep":
                expected.update(
                    {"trtllm::fused_moe::gemm1", "trtllm::fused_moe::gemm2"}
                )
            self.assertEqual(actual, expected)

    def test_low_latency_zero_capacity_does_not_launch(self):
        q = self.payload(3)
        d = DeepEPLLDispatchOutput(
            torch.empty(32, 0, 128, dtype=torch.bfloat16),
            None,
            torch.empty(0, 8, dtype=torch.int64),
            torch.empty(0, 8),
            torch.zeros(32, dtype=torch.int32),
            0,
        )
        with patch.object(runner, "_flashinfer_cutlass_fused_moe") as kernel:
            result = runner.fused_experts_deepep_to_flashinfer_cutlass(
                d, q, MoeRunnerConfig()
            )
        kernel.assert_not_called()
        self.assertEqual(result.hidden_states.shape, (32, 0, 128))


if __name__ == "__main__":
    unittest.main()
