import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.compressed_tensors import compressed_tensors
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    XPUCompressedTensorsWNA16MoE,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="extra-test-1-gpu-xpu")

EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 8, 512, 256, 2


def _pack_int32_along_k(codes):
    """Pack [E, K, N] 4-bit codes as compressed-tensors does: [E, K // 8, N] int32."""
    experts, k, n = codes.shape
    words = torch.zeros(experts, k // 8, n, dtype=torch.int64, device=codes.device)
    for i in range(8):
        words |= codes[:, i::8, :].to(torch.int64) << (4 * i)
    words = torch.where(words >= 2**31, words - 2**32, words)
    return torch.nn.Parameter(words.to(torch.int32), requires_grad=False)


@unittest.skipUnless(torch.xpu.is_available(), "needs an Intel XPU")
class TestXPUNativeWNA16MoE(CustomTestCase):
    def _run(self, group_size, drop_slots=False):
        device = torch.device("xpu")
        torch.manual_seed(0)
        config = CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "targets": ["re:.*mlp.experts.*"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": True,
                            "strategy": "group",
                            "group_size": group_size,
                        },
                        "input_activations": None,
                    }
                },
                "ignore": [],
            }
        )
        with mock.patch.object(
            compressed_tensors,
            "get_moe_runner_backend",
            return_value=MoeRunnerBackend.AUTO,
        ):
            scheme = config.get_moe_scheme(
                torch.nn.Module(), layer_name="model.layers.0.mlp.experts"
            )
        self.assertIsInstance(scheme, XPUCompressedTensorsWNA16MoE)

        layer = torch.nn.Module()
        dense = {}
        for name, (k, n) in (
            ("w13", (HIDDEN, 2 * INTERMEDIATE)),
            ("w2", (INTERMEDIATE, HIDDEN)),
        ):
            codes = torch.randint(0, 16, (EXPERTS, k, n), device=device)
            scale = torch.rand(EXPERTS, k // group_size, n, device=device) * 0.02
            scale = (scale + 0.005).to(torch.bfloat16)
            setattr(layer, f"{name}_weight_packed", _pack_int32_along_k(codes))
            setattr(
                layer,
                f"{name}_weight_scale",
                torch.nn.Parameter(scale, requires_grad=False),
            )
            # Dequantized [E, N, K] reference weight.
            dense[name] = (
                (codes.float() - 8.0) * scale.float().repeat_interleave(group_size, 1)
            ).transpose(1, 2)

        scheme.process_weights_after_loading(layer)
        scheme.create_moe_runner(
            layer,
            MoeRunnerConfig(
                num_experts=EXPERTS,
                num_local_experts=EXPERTS,
                hidden_size=HIDDEN,
                intermediate_size_per_partition=INTERMEDIATE,
                top_k=TOP_K,
                params_dtype=torch.bfloat16,
                activation="silu",
                is_gated=True,
                inplace=False,
            ),
        )

        tokens = 16
        x = torch.randn(tokens, HIDDEN, device=device, dtype=torch.bfloat16) * 0.5
        logits = torch.randn(tokens, EXPERTS, device=device)
        weights, ids = torch.topk(torch.softmax(logits, -1), TOP_K, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        if drop_slots:
            # Expert parallelism marks experts owned by another rank with -1.
            ids[::3, 1] = -1
        output = scheme.apply_weights(
            layer,
            StandardDispatchOutput(
                hidden_states=x,
                hidden_states_scale=None,
                topk_output=StandardTopKOutput(weights, ids.to(torch.int32), logits),
            ),
        ).hidden_states

        expected = torch.zeros(tokens, HIDDEN, device=device)
        for t in range(tokens):
            for j in range(TOP_K):
                e = int(ids[t, j])
                if e < 0:
                    continue
                gate_up = dense["w13"][e] @ x[t].float()
                act = F.silu(gate_up[:INTERMEDIATE]) * gate_up[INTERMEDIATE:]
                expected[t] += float(weights[t, j]) * (dense["w2"][e] @ act)
        error = (output.float() - expected).norm() / expected.norm()
        self.assertLess(float(error), 2e-2)

    def test_matches_the_dequantized_reference(self):
        for group_size in (32, 64, 128, 256):
            with self.subTest(group_size=group_size):
                self._run(group_size)

    def test_skips_dropped_expert_slots(self):
        self._run(group_size=128, drop_slots=True)


if __name__ == "__main__":
    unittest.main()
