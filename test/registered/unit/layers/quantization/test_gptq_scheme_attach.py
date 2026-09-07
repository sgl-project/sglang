"""GPTQ builds its per-layer scheme lazily in `create_weights`, so the layer has
to declare `scheme = None` for the `is None` probe to see it.

Regression: the probe used to be `hasattr(layer, "scheme")`, which degraded to
always-true once `LinearBase` grew that class default -- the scheme was never
built and every GPTQ model died with ``'NoneType' object has no attribute
'create_weights'``.
"""

import unittest

import torch

from sglang.srt.layers.linear import LinearBase, ReplicatedLinear
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

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
