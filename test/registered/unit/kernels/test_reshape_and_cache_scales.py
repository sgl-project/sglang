"""Host-side scale flags for ``launch_reshape_and_cache_flash``.

``k_scale`` and ``v_scale`` are independent optional arguments. The launcher
must not reuse one pointer for the other channel, nor enable scaling when only
one side is provided. These tests replace the Triton kernel with a recording
stub, so they do not need a GPU.
"""

import unittest

import torch

import sglang.kernels.ops.kvcache.cache_ops as cache_ops
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


class TestReshapeAndCacheScaleFlags(unittest.TestCase):
    def _launch(self, k_scale=None, v_scale=None):
        kernel = _RecordingKernel()
        original = cache_ops.reshape_and_cache_flash
        cache_ops.reshape_and_cache_flash = kernel
        try:
            cache_ops.launch_reshape_and_cache_flash(
                torch.zeros(1, 1, 2),
                torch.zeros(1, 1, 2),
                torch.zeros(2, 1, 1, 2),
                torch.zeros(2, 1, 1, 2),
                torch.tensor([1]),
                k_scale=None if k_scale is None else torch.tensor([k_scale]),
                v_scale=None if v_scale is None else torch.tensor([v_scale]),
            )
        finally:
            cache_ops.reshape_and_cache_flash = original
        self.assertEqual(len(kernel.calls), 1)
        return kernel.calls[0][2]

    def test_key_scale_only_enables_key_flag(self):
        kwargs = self._launch(k_scale=2.0)
        self.assertTrue(kwargs["USE_K_SCALE"])
        self.assertFalse(kwargs["USE_V_SCALE"])

    def test_value_scale_only_enables_value_flag(self):
        kwargs = self._launch(v_scale=4.0)
        self.assertFalse(kwargs["USE_K_SCALE"])
        self.assertTrue(kwargs["USE_V_SCALE"])

    def test_both_scales_enable_both_flags(self):
        kwargs = self._launch(k_scale=2.0, v_scale=4.0)
        self.assertTrue(kwargs["USE_K_SCALE"])
        self.assertTrue(kwargs["USE_V_SCALE"])

    def test_no_scales_disable_both_flags(self):
        kwargs = self._launch()
        self.assertFalse(kwargs["USE_K_SCALE"])
        self.assertFalse(kwargs["USE_V_SCALE"])


if __name__ == "__main__":
    unittest.main()
