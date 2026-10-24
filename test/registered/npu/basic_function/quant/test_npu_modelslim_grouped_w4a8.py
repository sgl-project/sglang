"""Regression coverage for grouped R1 ModelSlim W4A8 loading."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="base-b-test-1-npu-a3")


class TestGroupedW4A8(unittest.TestCase):
    def schemes(self, group_size, tp_size=1):
        prefix = "model.layers.3.mlp.experts"
        description = {"group_size": group_size}
        for name in ["gate_proj", "up_proj", "down_proj"]:
            description[f"{prefix}.0.{name}.weight"] = "W4A8_DYNAMIC"
        layer = torch.nn.Module()
        layer.moe_tp_size = tp_size
        return ModelSlimConfig(description).get_moe_scheme(layer, prefix)

    def test_checkpoint_group_size_reaches_both_projections(self):
        for scheme in self.schemes(256):
            self.assertEqual(scheme.group_size, 256)
            self.assertFalse(scheme.is_per_channel_weight)

    def test_tensor_parallel_size_reaches_both_projections(self):
        for scheme in self.schemes(0, 4):
            self.assertEqual(scheme.tp_size, 4)

    def check_tp_scale_loading(self, group_size):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.modelslim.modelslim import (
            ModelSlimFusedMoEMethod,
        )

        hidden_size, intermediate_size = 32, 64
        for tp_size in (1, 4):
            for rank in range(tp_size):
                for presharded in (False, True):
                    schemes = self.schemes(group_size, tp_size)
                    # Exercise real parameter creation and loader dispatch without
                    # starting distributed workers or allocating NPU tensors.
                    layer = FusedMoE.__new__(FusedMoE)
                    torch.nn.Module.__init__(layer)
                    layer.quant_config = schemes[0].quant_config
                    layer.quant_method = ModelSlimFusedMoEMethod(layer.quant_config)
                    layer.scheme = None
                    layer._has_fused_shared = False
                    layer.use_flashinfer_trtllm_moe = False
                    layer.use_triton_kernels = False
                    layer.use_padded_loading = False
                    layer.use_presharded_weights = presharded
                    layer.moe_tp_size = tp_size
                    layer.moe_tp_rank = rank
                    layer.moe_runner_config = SimpleNamespace(is_gated=True)
                    for scheme in schemes:
                        scheme.create_weights(
                            layer,
                            num_experts=2,
                            hidden_size=hidden_size,
                            intermediate_size_per_partition=intermediate_size
                            // tp_size,
                            weight_loader=layer._weight_loader_impl,
                        )
                    for expert in range(2):
                        for shard in ("w1", "w3", "w2"):
                            prefix = "w2" if shard == "w2" else "w13"
                            n, k = (
                                (hidden_size, intermediate_size)
                                if shard == "w2"
                                else (intermediate_size, hidden_size)
                            )
                            widths = {
                                "weight_scale": 1,
                                "weight_offset": 1,
                                "scale_bias": 16 if shard == "w2" else 1,
                            }
                            if group_size:
                                widths.update(
                                    weight_scale_second=k // group_size,
                                    weight_offset_second=k // group_size,
                                )
                            for suffix, width in widths.items():
                                with self.subTest(
                                    tp_size=tp_size,
                                    rank=rank,
                                    presharded=presharded,
                                    expert=expert,
                                    shard=shard,
                                    suffix=suffix,
                                ):
                                    name = f"{prefix}_{suffix}"
                                    param = getattr(layer, name)
                                    full = torch.arange(
                                        n * width, dtype=torch.float32
                                    ).reshape(n, width)
                                    full += expert * 10000 + (
                                        1000 if shard == "w3" else 0
                                    )
                                    if shard != "w2":
                                        expected = full.chunk(tp_size, dim=0)[rank]
                                    elif suffix in ("weight_scale", "weight_offset"):
                                        expected = full
                                    else:
                                        expected = full.chunk(tp_size, dim=1)[rank]
                                    param.weight_loader(
                                        param,
                                        expected if presharded else full,
                                        name,
                                        shard,
                                        expert,
                                    )
                                    actual = param[expert]
                                    if shard != "w2":
                                        actual = actual.chunk(2, dim=0)[shard == "w3"]
                                    torch.testing.assert_close(actual, expected)

    def test_grouped_scales_load_tp_shards(self):
        self.check_tp_scale_loading(group_size=16)

    def test_per_channel_scale_bias_loads_tp_shards(self):
        self.check_tp_scale_loading(group_size=0)

    def test_scales_use_unpacked_output_dimension(self):
        # Initialize the public quantization package before its backend module.
        from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
            NPUW4A8Int8MoEMethod,
        )

        kernel = NPUW4A8Int8MoEMethod(is_per_channel_weight=False)
        weight = torch.empty((1, 16, 64), dtype=torch.int8)
        scale = torch.ones((1, 32, 1))
        group_scale = torch.full((1, 32, 4), 0.125)
        with patch.object(torch.Tensor, "npu", lambda tensor, *args, **kwargs: tensor):
            actual = kernel._process_scale(weight, scale, group_scale, False)
        self.assertEqual(tuple(actual.shape), (1, 4, 32))
        self.assertEqual(actual.dtype, torch.int64)
        self.assertTrue(torch.all(actual == 1040187392).item())


if __name__ == "__main__":
    unittest.main()
