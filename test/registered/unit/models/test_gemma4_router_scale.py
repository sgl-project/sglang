"""Regression test for Gemma 4 router scale updates after the first forward."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.gemma4_causal import Gemma4ForCausalLM, Gemma4Router
from sglang.test.test_utils import CustomTestCase

_CONFIG = SimpleNamespace(
    hidden_size=8, rms_norm_eps=1e-6, num_experts=4, tie_word_embeddings=False
)
_SCALE_NAME = "model.layers.0.router.scale"


def _causal_lm_with_router(router):
    lm = Gemma4ForCausalLM.__new__(Gemma4ForCausalLM)
    torch.nn.Module.__init__(lm)
    lm.config = _CONFIG
    lm.pp_group = SimpleNamespace(world_size=1)
    lm.model = torch.nn.Module()
    lm.model.start_layer, lm.model.end_layer = 0, 1
    lm.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    lm.model.layers[0].router = router
    return lm


class TestGemma4RouterScale(CustomTestCase):
    def test_reloaded_scale_reaches_router_norm(self):
        """A scale loaded after the first forward must reach the router norm weight."""
        router = Gemma4Router(_CONFIG)
        lm = _causal_lm_with_router(router)
        norm_ptr = router.norm.weight.data_ptr()

        lm.load_weights([(_SCALE_NAME, torch.full((8,), 2.0))])
        router(torch.randn(3, 8))
        updated = torch.full((8,), 5.0)
        lm.load_weights([(_SCALE_NAME, updated)])

        self.assertEqual(router.norm.weight.data_ptr(), norm_ptr)
        torch.testing.assert_close(router.norm.weight, updated * router.root_size)


if __name__ == "__main__":
    unittest.main()
