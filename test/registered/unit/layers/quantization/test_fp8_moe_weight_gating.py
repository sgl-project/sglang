"""w13 buffers must be sized by the layer's gating, not assumed to be gate+up fused.

A non-gated MoE (e.g. NemotronH: relu2, checkpoint carries up_proj/down_proj and
no gate_proj) fuses a single projection into w13. Sizing w13 as 2*intermediate
leaves the upper half as uninitialised ``torch.empty`` that no weight loader ever
writes, which silently corrupts quantized MoE weights. The weight scales must
follow the same shard count, or the two describe different tensors.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS = 4
HIDDEN = 256
INTERMEDIATE = 640
BLOCK_N = 128
BLOCK_K = 128


class _RecordingLayer:
    """Collects the parameters ``create_fp8_moe_weight_`` registers."""

    def __init__(self, is_gated: bool):
        self.moe_runner_config = MoeRunnerConfig(is_gated=is_gated)
        self.moe_tp_size = 1
        self.params = {}

    def register_parameter(self, name, param):
        self.params[name] = param


def _create_weights(is_gated: bool, block_quant: bool):
    from sglang.srt.layers.quantization import fp8 as fp8_quant

    layer = _RecordingLayer(is_gated)
    quant_config = MagicMock(
        weight_block_size=[BLOCK_N, BLOCK_K],
        activation_scheme="dynamic",
        is_checkpoint_fp8_serialized=False,
    )

    fp8_quant.Fp8MoEMethod.create_fp8_moe_weight_(
        layer=layer,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        block_quant=block_quant,
        quant_config=quant_config,
        use_mxfp8=False,
        is_checkpoint_fp8_serialized=False,
        is_fp4_expert=False,
        params_dtype=torch.bfloat16,
    )
    return layer.params


class TestFp8MoEWeightGating(CustomTestCase):
    def test_gated_fuses_gate_and_up(self):
        params = _create_weights(is_gated=True, block_quant=True)
        self.assertEqual(params["w13_weight"].shape[1], 2 * INTERMEDIATE)
        self.assertEqual(
            params["w13_weight_scale_inv"].shape[1], 2 * (INTERMEDIATE // BLOCK_N)
        )

    def test_non_gated_w13_holds_up_only(self):
        # Regression: w13 was always sized 2*intermediate, so the upper half
        # stayed uninitialised for NemotronH.
        params = _create_weights(is_gated=False, block_quant=True)
        self.assertEqual(params["w13_weight"].shape[1], INTERMEDIATE)

    def test_non_gated_block_scale_matches_weight(self):
        params = _create_weights(is_gated=False, block_quant=True)
        weight_rows = params["w13_weight"].shape[1]
        scale_rows = params["w13_weight_scale_inv"].shape[1]
        self.assertEqual(scale_rows * BLOCK_N, weight_rows)

    def test_non_gated_per_tensor_scale_is_single(self):
        # One shard means one scale per expert; nothing to fuse afterwards.
        params = _create_weights(is_gated=False, block_quant=False)
        self.assertEqual(params["w13_weight_scale"].shape, (NUM_EXPERTS, 1))


class TestFallbackWeightPadding(CustomTestCase):
    def setUp(self):
        from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
        from sglang.srt.runtime_context import get_context, get_flags

        self.enterContext(
            get_context().override_server_args(model_path="dummy", device="cpu")
        )
        self.enterContext(
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
                a2a_backend=MoeA2ABackend.NONE,
            )
        )
        # CPU's always-padded loader would hide a GPU fallback's uninitialized tail.
        self.enterContext(
            patch("sglang.srt.layers.moe.fused_moe_triton.layer._is_cpu", False)
        )
        self.enterContext(
            patch("sglang.srt.layers.moe.fused_moe_triton.layer._use_aiter", False)
        )
        self.enterContext(
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.cutlass_fp8_supported",
                return_value=False,
            )
        )

    @staticmethod
    def _expert_output(hidden, w13, w2, is_gated):
        projected = hidden @ w13.float().t()
        if is_gated:
            gate, up = projected.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
        else:
            activated = projected.relu().square()
        return activated @ w2.float().t()

    @staticmethod
    def _build_layer(logical_size, tp_size, rank, presharded, is_gated):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptFp8Config,
            ModelOptFp8MoEMethod,
        )
        from sglang.srt.runtime_context import get_parallel

        with get_parallel().override(
            tp_size=tp_size,
            tp_rank=rank,
            attn_tp_size=tp_size,
            attn_tp_rank=rank,
            moe_tp_size=tp_size,
            moe_tp_rank=rank,
            moe_ep_size=1,
            moe_ep_rank=0,
        ):
            config = ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
            return FusedMoE(
                num_experts=1,
                top_k=1,
                hidden_size=128,
                intermediate_size=logical_size * tp_size,
                layer_id=0,
                quant_config=config,
                quant_method=ModelOptFp8MoEMethod(config),
                use_presharded_weights=presharded,
                is_gated=is_gated,
                activation="silu" if is_gated else "relu2",
            )

    def _load_and_run(self, layer, checkpoint, hidden, logical_size):
        layer.w13_weight.data.fill_(float("nan"))
        layer.w2_weight.data.fill_(float("nan"))
        is_gated = layer.moe_runner_config.is_gated
        shard_ids = ("w3", "w1", "w2") if is_gated else ("w1", "w2")
        for shard_id in shard_ids:
            weight = checkpoint[shard_id]
            if layer.use_presharded_weights:
                dim = 1 if shard_id == "w2" else 0
                weight = weight.narrow(
                    dim, layer.moe_tp_rank * logical_size, logical_size
                )
            param = layer.w2_weight if shard_id == "w2" else layer.w13_weight
            layer.weight_loader(
                param=param,
                loaded_weight=weight,
                weight_name="weight",
                shard_id=shard_id,
                expert_id=0,
            )
        physical_size = layer.intermediate_size_per_partition
        for part in layer.w13_weight[0].float().split(physical_size):
            self.assertEqual(torch.count_nonzero(part[logical_size:]).item(), 0)
        self.assertEqual(
            torch.count_nonzero(layer.w2_weight[0].float()[:, logical_size:]).item(), 0
        )
        return self._expert_output(
            hidden, layer.w13_weight[0], layer.w2_weight[0], is_gated
        )

    def test_fallback_loading_initializes_padding_on_every_load(self):
        """Padded FP8 fallback weights must produce the unpadded expert's output."""
        from sglang.srt.layers.moe import MoeRunnerBackend

        hidden_size = 128
        hidden = torch.arange(3 * hidden_size).reshape(3, hidden_size).remainder(5)
        hidden = (hidden.float() - 2) / 8
        cases = (
            (192, 1, False, True),
            (192, 2, False, True),
            (192, 2, True, True),
            (256, 1, False, True),
            (192, 1, False, False),
        )
        for logical_size, tp_size, presharded, is_gated in cases:
            with self.subTest(
                intermediate=logical_size,
                tp_size=tp_size,
                presharded=presharded,
                is_gated=is_gated,
            ):
                global_size = logical_size * tp_size
                values = torch.arange(global_size * hidden_size).reshape(
                    global_size, hidden_size
                )
                gate = (values.remainder(7).float() - 3) / 32
                up = (values.remainder(5).float() - 2) / 16
                down = (values.t().remainder(3).float() - 1) / 16
                layers = [
                    self._build_layer(logical_size, tp_size, rank, presharded, is_gated)
                    for rank in range(tp_size)
                ]
                for layer in layers:
                    self.assertEqual(layer.runner_backend, MoeRunnerBackend.TRITON)
                    self.assertFalse(layer.use_padded_loading)

                for multiplier in (1, 2):
                    checkpoint = {
                        "w1": (gate * multiplier).to(torch.float8_e4m3fn),
                        "w2": (down * multiplier).to(torch.float8_e4m3fn),
                        "w3": (up * multiplier).to(torch.float8_e4m3fn),
                    }
                    outputs = [
                        self._load_and_run(layer, checkpoint, hidden, logical_size)
                        for layer in layers
                    ]
                    reference_w13 = checkpoint["w1"].float()
                    if is_gated:
                        reference_w13 = torch.cat(
                            (reference_w13, checkpoint["w3"].float())
                        )
                    expected = self._expert_output(
                        hidden, reference_w13, checkpoint["w2"], is_gated
                    )
                    actual = torch.stack(outputs).sum(dim=0)
                    self.assertTrue(torch.isfinite(actual).all().item())
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
