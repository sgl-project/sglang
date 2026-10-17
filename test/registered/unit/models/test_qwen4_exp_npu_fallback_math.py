"""Exercise the NPU eager GR math using real CPU tensors and CUDA sentinels."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from qwen4_exp_cpu_test_utils import forbidden, load

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen4NPUFallbackMath(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.alias = patch.object(torch.Tensor, "is_cuda", property(lambda self: True))
        self.alias.start()
        self.addCleanup(self.alias.stop)

    def test_gr_mix_combine_and_empty_rows(self):
        mix = load(
            "kernels/ops/gemm/hc_mix.py",
            {"fused_hc_mix_supported": None},
            {"_deterministic_inference": lambda: False, "_FUSED_MIX_MAX_ROWS": 16},
        )
        module = load(
            "srt/layers/hyperconnection.py",
            {
                "GroupedGemmaRMSNorm": None,
                "HyperConnectionBase": None,
                "GatedResidual": None,
            },
            {
                "_is_npu": True,
                "fused_hc_mix_supported": mix.fused_hc_mix_supported,
                "fused_hc_mix": forbidden,
                "_log_path_once": lambda *a: None,
            },
        )
        for per_branch in (False, True):
            config = NS(
                hc_count=4,
                hidden_size=512,
                params_dtype=torch.float32,
                mtp_hc=False,
                hc_lowrank=8,
                rms_norm_eps=1e-06,
                hc_per_branch_norm=per_branch,
            )
            with (
                patch.object(
                    torch,
                    "get_device_module",
                    return_value=NS(current_device=lambda: "cpu"),
                ),
                patch.object(torch, "compile", side_effect=forbidden),
            ):
                layer = module.GatedResidual(config)
            for rows in (0, 1, 17):
                with self.subTest(per_branch=per_branch, rows=rows), torch.no_grad():
                    x = torch.randn(rows, 2048)
                    block = torch.randn(rows, 512)
                    mixed, residual = layer.mix(x)
                    result = layer.combine(block, residual)
                    branches = x.reshape(rows, 4, 512)
                    normalized = branches / torch.sqrt(
                        branches.square().mean(-1, keepdim=True) + 1e-06
                    )
                    normalized = normalized.reshape(rows, 2048)
                    low = (
                        torch.einsum(
                            "td,ld->tl", normalized, layer.input_mix_weight_down.weight
                        )
                        / 4
                    )
                    gate = torch.sigmoid(
                        torch.einsum(
                            "tl,dl->td",
                            low * torch.sigmoid(low),
                            layer.input_mix_weight_up.weight,
                        )
                    )
                    expected_mix = (gate * normalized).reshape(rows, 4, 512).sum(1) / 4
                    inject = 2 * torch.sigmoid(
                        torch.einsum(
                            "td,hd->th", normalized, layer.block_inject_weight.weight
                        )
                        / 4
                    )
                    expected_result = (
                        branches + inject[..., None] * block[:, None, :]
                    ).flatten(1)
                    torch.testing.assert_close(mixed, expected_mix)
                    torch.testing.assert_close(result, expected_result)

    def test_attention_gate_keeps_native_math_with_cuda_alias(self):
        code = load(
            "srt/models/qwen4_exp.py",
            {"Qwen4ExpAttentionDecoderLayer": {"self_attention"}},
            {"fused_sigmoid_mul": forbidden},
        )
        layer = code.Qwen4ExpAttentionDecoderLayer()
        layer.is_qsa = False
        layer.alt_stream = None
        x = torch.randn(3, 8).to(torch.bfloat16)
        gate = torch.randn(3, 2, 8).to(torch.bfloat16)[..., ::2]
        layer._prepare_qkv_gate = lambda **kw: (None, None, None, gate)
        layer.attn = lambda *a, **kw: x
        layer.o_proj = lambda value: (value, None)
        actual = layer.self_attention(None, None, None)
        torch.testing.assert_close(actual, x * gate.reshape_as(x).sigmoid())


if __name__ == "__main__":
    unittest.main()
