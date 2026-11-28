"""Startup prebuilds of FlashInfer's JIT sampling and top-k modules (CUDA).

Without flashinfer-jit-cache, FlashInfer compiles these modules on first use;
the top-k one takes minutes. Reached first from a request, the build runs
inside the scheduler loop and stalls every request, health checks included,
until it finishes. The prebuilds load both modules while the server starts.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestPrebuildFlashInferSampling(CustomTestCase):
    def test_radix_topk_prebuild_loads_the_topk_module(self):
        from flashinfer.topk import get_topk_module

        from sglang.kernels.ops.sampling.topk import prebuild_radix_topk

        prebuild_radix_topk.cache_clear()
        prebuild_radix_topk("cuda")
        self.assertEqual(get_topk_module.cache_info().currsize, 1)

    def test_sampling_prebuild_loads_the_module_and_keeps_the_rng(self):
        from flashinfer.sampling import get_sampling_module

        from sglang.srt.layers.sampler import prebuild_flashinfer_sampling

        prebuild_flashinfer_sampling.cache_clear()
        rng_before = torch.cuda.get_rng_state()
        prebuild_flashinfer_sampling()
        self.assertEqual(get_sampling_module.cache_info().currsize, 1)
        self.assertTrue(torch.equal(rng_before, torch.cuda.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
