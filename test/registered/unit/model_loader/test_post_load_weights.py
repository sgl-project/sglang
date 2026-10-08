"""Loaders that write params without `load_weights` must leave the buffers weight loaders derive from them current."""

import unittest

import torch

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.model_loader.loader import post_load_weights
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPostLoadWeights(CustomTestCase):
    def test_gemma_weight_follows_a_weight_written_without_its_loader(self):
        """p2p weight updates and the remote-instance loader write params by address; fused allreduce kernels read
        `gemma_weight`, so a stale one would run the old norm."""
        model = torch.nn.Module()
        model.norm = GemmaRMSNorm(4)
        gemma_weight_address = model.norm.gemma_weight.data_ptr()
        new_weight = torch.tensor([0.5, -0.25, 0.0, 2.0])

        model.norm.weight.data.copy_(new_weight)
        post_load_weights(model)

        torch.testing.assert_close(model.norm.gemma_weight, new_weight + 1)
        self.assertEqual(model.norm.gemma_weight.data_ptr(), gemma_weight_address)


if __name__ == "__main__":
    unittest.main()
