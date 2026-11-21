# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0

"""Unit tests for the FP8 fused prefix-valid capability gate.

The fused kernel ``set_kv_buffer_prefix_valid_tiled_fp8`` compiles
``fp8e4nv`` in Triton, which only lowers on SM89+ (Ada/Hopper). The gate
``_fp8_fused_prefix_valid_supported`` must keep ROCm on the fused path
(fnuz/OCP fp8 types work there) and drop pre-SM89 NVIDIA GPUs (e.g. SM86 /
RTX 3090) to the eager quantization fallback. The check is evaluated
lazily (lru_cache) so importing the module never initializes CUDA.

These tests mock ``get_device_sm`` and ``_is_hip``, so they exercise the
gate logic on any architecture without per-arch hardware.
"""

import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFp8FusedPrefixValidGate(unittest.TestCase):
    def _evaluate_gate(self, is_hip: bool, device_sm: int) -> bool:
        from sglang.srt.mem_cache import memory_pool

        memory_pool._fp8_fused_prefix_valid_supported.cache_clear()
        with (
            mock.patch.object(memory_pool, "_is_hip", is_hip),
            mock.patch.object(memory_pool, "get_device_sm", return_value=device_sm),
        ):
            return memory_pool._fp8_fused_prefix_valid_supported()

    def tearDown(self):
        from sglang.srt.mem_cache import memory_pool

        # Drop the lru_cache entry so a mocked result never leaks.
        memory_pool._fp8_fused_prefix_valid_supported.cache_clear()

    def test_rocm_keeps_fused_path(self):
        # ROCm is reachable for this dispatch and runs the fused kernel
        # with fnuz/OCP fp8 types; the gate must stay True on any gfx arch.
        self.assertTrue(self._evaluate_gate(is_hip=True, device_sm=0))

    def test_sm86_ampere_falls_back(self):
        # fp8e4nv does not compile on SM86 (e.g. RTX 3090): the gate must
        # force the eager quantization fallback.
        self.assertFalse(self._evaluate_gate(is_hip=False, device_sm=86))

    def test_sm80_ampere_falls_back(self):
        self.assertFalse(self._evaluate_gate(is_hip=False, device_sm=80))

    def test_sm89_ada_keeps_fused_path(self):
        self.assertTrue(self._evaluate_gate(is_hip=False, device_sm=89))

    def test_sm90_hopper_keeps_fused_path(self):
        self.assertTrue(self._evaluate_gate(is_hip=False, device_sm=90))

    def test_sm120_blackwell_keeps_fused_path(self):
        self.assertTrue(self._evaluate_gate(is_hip=False, device_sm=120))

    def test_gate_result_is_cached(self):
        # The gate is lru_cache'd: the capability check runs once and the
        # per-dispatch cost after the first call is zero.
        from sglang.srt.mem_cache import memory_pool

        memory_pool._fp8_fused_prefix_valid_supported.cache_clear()
        with (
            mock.patch.object(memory_pool, "_is_hip", False),
            mock.patch.object(memory_pool, "get_device_sm", return_value=89) as get_sm,
        ):
            self.assertTrue(memory_pool._fp8_fused_prefix_valid_supported())
            self.assertTrue(memory_pool._fp8_fused_prefix_valid_supported())
        get_sm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
