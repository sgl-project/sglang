"""Which runners prebuild FlashInfer's sampling and top-k modules (CPU)."""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sglang.kernels.ops.sampling.topk as topk_module
import sglang.srt.layers.sampler as sampler_module
import sglang.srt.model_executor.model_runner as model_runner_module
from sglang.srt.environ import envs
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _exec(sampling_backend):
    return SimpleNamespace(kernel=SimpleNamespace(sampling_backend=sampling_backend))


class TestModelRunnerSamplingPrebuild(CustomTestCase):
    def _prebuilds(self, *, device, sampling_backend):
        prebuild = MagicMock()
        with (
            patch.object(
                model_runner_module, "get_exec", return_value=_exec(sampling_backend)
            ),
            patch.object(sampler_module, "prebuild_flashinfer_sampling", prebuild),
        ):
            ModelRunner.maybe_prebuild_flashinfer_sampling(
                SimpleNamespace(device=device)
            )
        return prebuild.call_count

    def test_flashinfer_backend_on_cuda_prebuilds(self):
        self.assertEqual(
            self._prebuilds(device="cuda", sampling_backend="flashinfer"), 1
        )

    def test_other_backends_and_devices_do_not(self):
        self.assertEqual(self._prebuilds(device="cuda", sampling_backend="pytorch"), 0)
        self.assertEqual(
            self._prebuilds(device="cpu", sampling_backend="flashinfer"), 0
        )


class TestRadixTopkPrebuild(CustomTestCase):
    def setUp(self):
        topk_module.prebuild_radix_topk.cache_clear()

    def tearDown(self):
        topk_module.prebuild_radix_topk.cache_clear()

    def _top_k_calls(self, device, *, enabled=True):
        flashinfer_top_k = MagicMock()
        with (
            patch.object(topk_module, "_flashinfer_top_k", flashinfer_top_k),
            envs.SGLANG_FLASHINFER_PREBUILD_SAMPLING.override(enabled),
        ):
            topk_module.prebuild_radix_topk(device)
        return flashinfer_top_k.call_count

    def test_cpu_device_skips(self):
        self.assertEqual(self._top_k_calls("cpu"), 0)

    def test_opt_out_skips(self):
        self.assertEqual(self._top_k_calls("cuda", enabled=False), 0)


class TestSamplingPrebuildOptOut(CustomTestCase):
    def test_opt_out_skips(self):
        sampler_module.prebuild_flashinfer_sampling.cache_clear()
        sample = MagicMock()
        with (
            patch.object(
                sampler_module, "top_k_top_p_sampling_from_probs", sample, create=True
            ),
            envs.SGLANG_FLASHINFER_PREBUILD_SAMPLING.override(False),
        ):
            sampler_module.prebuild_flashinfer_sampling()
        sampler_module.prebuild_flashinfer_sampling.cache_clear()
        self.assertEqual(sample.call_count, 0)


if __name__ == "__main__":
    unittest.main()
