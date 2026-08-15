"""GPTQ attaches its per-layer scheme when the layer declares `scheme = None`.

`GPTQ{,Marlin}{Linear,MoE}Method.create_weights` builds the scheme lazily,
because the dynamic (per-module) rules are baked into the *cloned* config the
method was constructed with, not into the config that answered
`get_quant_method`. The "already attached?" probe used to be
`hasattr(layer, "scheme")`, which silently degraded to always-true once
`LinearBase` grew a class-level `scheme = None` default: the scheme was never
built and weight creation died with ``'NoneType' object has no attribute
'create_weights'`` for every GPTQ model.

The probe is a `None` test now, so these cases fail if the probe regresses to an
attribute-existence check, or if any of the three layer bases a GPTQ method can
be handed drops the default the probe reads.
"""

import unittest

import torch

from sglang.srt.layers.linear import LinearBase, ReplicatedLinear
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig
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
        """The `None` probe only works if the default is an attribute, not absent.

        Every layer type a GPTQ method can be handed has to carry it:
        `get_linear_quant_method` hands a linear method either a `LinearBase` or
        a quantized `ParallelLMHead`, and `GPTQMarlinConfig.get_quant_method`
        hands `GPTQMarlinMoEMethod` a bare `FusedMoE` — the lazy attach in
        `GPTQMarlinMoEMethod.create_weights` is the only scheme attach point for
        a gptq_marlin MoE checkpoint, and no e2e test in the repo covers it.
        """
        self.assertIsNone(LinearBase.scheme)
        self.assertIsNone(VocabParallelEmbedding.scheme)
        self.assertIsNone(FusedMoE.scheme)


if __name__ == "__main__":
    unittest.main()
