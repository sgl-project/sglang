"""Gemma4Router folds its scale into the norm weight when the scale is loaded,
so that no forward writes a weight."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models.gemma4_causal import Gemma4Router, fold_router_scales
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CONFIG = SimpleNamespace(hidden_size=8, rms_norm_eps=1e-6, num_experts=4)


def folded(router: Gemma4Router) -> torch.Tensor:
    return (router.scale * router.root_size).to(router.norm.weight.dtype)


class TestGemma4RouterFold(CustomTestCase):
    def test_loading_the_scale_folds_it(self):
        router = Gemma4Router(CONFIG)
        router.scale.weight_loader(router.scale, torch.arange(8.0))
        self.assertTrue(router._scale_fused)
        self.assertTrue(torch.equal(router.scale, torch.arange(8.0)))
        self.assertTrue(torch.allclose(router.norm.weight, folded(router)))

    def test_a_later_load_of_the_scale_folds_again(self):
        router = Gemma4Router(CONFIG)
        router.scale.weight_loader(router.scale, torch.ones(8))
        router.scale.weight_loader(router.scale, torch.full((8,), 2.0))
        self.assertTrue(torch.allclose(router.norm.weight, folded(router)))

    def test_forward_writes_no_weight_once_loaded(self):
        router = Gemma4Router(CONFIG)
        router.scale.weight_loader(router.scale, torch.arange(8.0))
        before = router.norm.weight.clone()
        router(torch.randn(3, 8))
        self.assertTrue(torch.equal(router.norm.weight, before))

    def test_loaders_without_load_weights_fold_after_loading(self):
        model = nn.Module()
        model.router = Gemma4Router(CONFIG)
        with torch.no_grad():
            model.router.scale.copy_(torch.arange(8.0))  # as dummy weights do
        fold_router_scales(model)
        self.assertTrue(model.router._scale_fused)
        self.assertTrue(torch.allclose(model.router.norm.weight, folded(model.router)))


if __name__ == "__main__":
    unittest.main()
