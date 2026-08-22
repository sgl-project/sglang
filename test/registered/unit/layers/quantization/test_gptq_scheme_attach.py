"""GPTQ builds its per-layer scheme lazily in `create_weights`, so the layer has
to declare `scheme = None` for the `is None` probe to see it.

Regression: the probe used to be `hasattr(layer, "scheme")`, which degraded to
always-true once `LinearBase` grew that class default -- the scheme was never
built and every GPTQ model died with ``'NoneType' object has no attribute
'create_weights'``.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.linear import LinearBase, ReplicatedLinear
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig, GPTQMarlinConfig
from sglang.srt.layers.quantization.gptq.schemes.gptq_moe import (
    GPTQMarlinMoEScheme,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_GPTQ_CHECKPOINT_CONFIG = {
    "bits": 4,
    "group_size": 128,
    "desc_act": False,
    "lm_head": False,
    "dynamic": {},
    "checkpoint_format": "gptq",
    "true_sequential": True,
    "static_groups": False,
}


class TestGPTQSchemeAttach(CustomTestCase):
    def test_marlin_moe_scales_follow_params_dtype(self):
        quant_config = GPTQMarlinConfig(
            weight_bits=4,
            group_size=128,
            desc_act=False,
            is_sym=True,
            lm_head_quantized=False,
            dynamic={},
            full_config={},
        )

        for params_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(params_dtype=params_dtype):
                layer = torch.nn.Module()
                layer.moe_tp_size = 1
                GPTQMarlinMoEScheme(quant_config).create_weights(
                    layer=layer,
                    num_experts=2,
                    hidden_size=256,
                    intermediate_size_per_partition=128,
                    params_dtype=params_dtype,
                )

                self.assertEqual(layer.w13_scales.dtype, params_dtype)
                self.assertEqual(layer.w2_scales.dtype, params_dtype)

    def test_marlin_moe_w2_scale_shape_for_tp_and_act_order(self):
        for desc_act, expected_groups in ((False, 1), (True, 2)):
            with self.subTest(desc_act=desc_act):
                quant_config = GPTQMarlinConfig(
                    weight_bits=4,
                    group_size=128,
                    desc_act=desc_act,
                    is_sym=True,
                    lm_head_quantized=False,
                    dynamic={},
                    full_config={},
                )
                layer = torch.nn.Module()
                layer.moe_tp_size = 2

                GPTQMarlinMoEScheme(quant_config).create_weights(
                    layer=layer,
                    num_experts=2,
                    hidden_size=256,
                    intermediate_size_per_partition=128,
                    params_dtype=torch.bfloat16,
                )

                self.assertEqual(layer.w2_scales.shape, (2, expected_groups, 256))
                self.assertEqual(layer.w2_scales.load_full_w2, desc_act)

    def test_fused_moe_w2_loader_honors_load_full(self):
        loader = SimpleNamespace(
            quant_config=None,
            use_padded_loading=False,
            use_presharded_weights=False,
            use_triton_kernels=False,
            moe_tp_size=2,
        )
        loaded_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)

        sharded = torch.empty(2, 4)
        FusedMoE._load_w2(
            loader,
            expert_data=sharded,
            shard_dim=0,
            shard_id="w2",
            loaded_weight=loaded_weight,
            tp_rank=1,
        )
        self.assertTrue(torch.equal(sharded, loaded_weight[2:]))

        full = torch.empty_like(loaded_weight)
        FusedMoE._load_w2(
            loader,
            expert_data=full,
            shard_dim=0,
            shard_id="w2",
            loaded_weight=loaded_weight,
            tp_rank=1,
            load_full=True,
        )
        self.assertTrue(torch.equal(full, loaded_weight))

    def test_linear_layer_gets_a_scheme(self):
        layer = ReplicatedLinear(
            input_size=256,
            output_size=128,
            bias=False,
            params_dtype=torch.float16,
            quant_config=GPTQConfig.from_config(_GPTQ_CHECKPOINT_CONFIG),
            prefix="model.layers.0.mlp.down_proj",
        )
        self.assertIsNotNone(layer.scheme)
        self.assertTrue(hasattr(layer, "qweight"))

    def test_scheme_default_is_declared_on_every_quantizable_layer_base(self):
        """`get_linear_quant_method` hands a linear method a `LinearBase` or a
        quantized `ParallelLMHead`; `GPTQMarlinConfig` hands
        `GPTQMarlinMoEMethod` a bare `FusedMoE`. The MoE attach has no e2e
        coverage, so this is its only guard.
        """
        self.assertIsNone(LinearBase.scheme)
        self.assertIsNone(VocabParallelEmbedding.scheme)
        self.assertIsNone(FusedMoE.scheme)


if __name__ == "__main__":
    unittest.main()
