# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for CPU GPTQ shards that split quantization groups."""

import unittest
from types import MethodType, SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.layers.quantization.gptq.gptq import (
    CPUGPTQConfig,
    GPTQConfig,
    GPTQLinearMethod,
)
from sglang.srt.layers.quantization.gptq.schemes.gptq_cpu import (
    GPTQIntelAMXLinearScheme,
)
from sglang.srt.models.gemma4_causal import Gemma4Router
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="stage-a-tp-test-cpu-intel")


def _pack(x, dim):
    x = x.movedim(dim, -1).long()
    shifts = torch.arange(8) * 4
    return (
        (x.unflatten(-1, (-1, 8)) << shifts).sum(-1).int().movedim(-1, dim).contiguous()
    )


def _unpack(x, dim):
    x = x.movedim(dim, -1).long()
    shifts = torch.arange(8) * 4
    return ((x.unsqueeze(-1) >> shifts) & 15).flatten(-2).movedim(-1, dim).contiguous()


def _checkpoint(k=2112, n=32, group_size=128):
    gen = torch.Generator().manual_seed(129)
    groups = (k + group_size - 1) // group_size
    weights = torch.randint(0, 16, (k, n), generator=gen)
    zeros = torch.randint(1, 16, (groups, n), generator=gen)

    return {
        "qweight": _pack(weights, 0),
        "qzeros": _pack(zeros - 1, 1),
        "scales": (torch.rand(groups, n, generator=gen) * 0.02 + 0.01).bfloat16(),
        "g_idx": torch.arange(k, dtype=torch.int32) // group_size,
    }


def _make_layer(checkpoint, tp_size, rank):
    config = GPTQConfig(
        weight_bits=4,
        group_size=128,
        desc_act=False,
        lm_head_quantized=False,
        dynamic={},
    )
    scheme = GPTQIntelAMXLinearScheme(config)
    layer = torch.nn.Module()
    layer.tp_group = SimpleNamespace(rank_in_group=rank, world_size=tp_size)
    layer.use_presharded_weights = False

    scheme.create_weights(
        layer=layer,
        input_size_per_partition=2112 // tp_size,
        output_partition_sizes=[32],
        input_size=2112,
        params_dtype=torch.bfloat16,
        weight_loader=MethodType(RowParallelLinear.weight_loader_v2, layer),
    )

    for name, value in checkpoint.items():
        param = layer.get_parameter(name)
        param.weight_loader(param, value)

    return layer, scheme


def _dequant(weights):
    q = _unpack(weights["qweight"], 0)
    z = _unpack(weights["qzeros"], 1) + 1
    g = weights["g_idx"].long()
    return (q - z[g]).float() * weights["scales"][g].float()


class TestGPTQCPUGroupPadding(CustomTestCase):
    def test_gemma4_router_loads_gptq_projection(self):
        """GPTQ Router tensors must not leave an uninitialized float projection."""
        quant_config = CPUGPTQConfig(
            weight_bits=4,
            group_size=128,
            desc_act=False,
            lm_head_quantized=False,
            dynamic={},
        )
        router = Gemma4Router(
            SimpleNamespace(hidden_size=2816, num_experts=128, rms_norm_eps=1e-6),
            quant_config=quant_config,
        )
        params = dict(router.proj.named_parameters())
        self.assertIsInstance(router.proj.quant_method, GPTQLinearMethod)
        self.assertIsInstance(router.proj.scheme, GPTQIntelAMXLinearScheme)
        self.assertNotIn("weight", params)
        self.assertEqual(tuple(params["qweight"].shape), (352, 128))
        self.assertEqual(tuple(params["qzeros"].shape), (22, 16))
        self.assertEqual(tuple(params["scales"].shape), (22, 128))
        self.assertEqual(tuple(params["g_idx"].shape), (2816,))

        checkpoint = {
            "qweight": torch.randint(0, 2**15, (352, 128), dtype=torch.int32),
            "qzeros": torch.randint(0, 2**15, (22, 16), dtype=torch.int32),
            "scales": torch.rand(22, 128, dtype=torch.float16),
        }
        for name, value in checkpoint.items():
            param = params[name]
            param.weight_loader(param, value)
            torch.testing.assert_close(param, value.to(param.dtype))

        expected_g_idx = torch.arange(2816, dtype=torch.int32) // 128
        torch.testing.assert_close(params["g_idx"], expected_g_idx)

    def test_tp_group_padding(self):
        checkpoint = _checkpoint()
        reference = _dequant(checkpoint)

        for tp in (1, 2, 4):
            shard = 2112 // tp

            for rank in range(tp):
                with self.subTest(tp=tp, rank=rank):
                    layer, _ = _make_layer(checkpoint, tp, rank)
                    padding = layer.gptq_cpu_padding

                    padding.fill_weight_padding(
                        qweight=layer.qweight.data,
                        qzeros=layer.qzeros.data,
                    )

                    actual = _dequant(dict(layer.named_parameters()))
                    expected = F.pad(
                        reference[rank * shard : (rank + 1) * shard],
                        (0, 0, padding.left, padding.right),
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_amx_gemm(self):
        from sglang.srt.utils import cpu_has_amx_support

        if not cpu_has_amx_support():
            self.skipTest("AMX hardware required")

        try:
            import sgl_kernel  # noqa: F401
        except ImportError:
            self.skipTest("CPU kernel wheel required")

        checkpoint = _checkpoint()
        x = torch.randn(
            4, 2112, generator=torch.Generator().manual_seed(130)
        ).bfloat16()
        expected = x.float() @ _dequant(checkpoint)

        for tp in (1, 2, 4):
            with self.subTest(tp=tp):
                shard = 2112 // tp
                outputs = []

                for rank in range(tp):
                    layer, scheme = _make_layer(checkpoint, tp, rank)
                    scheme.process_weights_after_loading(layer)

                    local_x = x[:, rank * shard : (rank + 1) * shard].contiguous()
                    outputs.append(scheme.apply_weights(layer, local_x, None).float())

                torch.testing.assert_close(sum(outputs), expected, atol=0.15, rtol=0.02)


if __name__ == "__main__":
    unittest.main()
