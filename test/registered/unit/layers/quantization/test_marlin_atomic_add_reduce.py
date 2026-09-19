"""Tests for when Marlin uses the atomicAdd K-slice reduction."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import marlin_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CUDA = torch.device("cuda")
# n < 2048 and k >= 2048: the only shapes where the atomic path is considered.
NARROW_N, LARGE_K = 1536, 8960


class TestShouldUseAtomicAddReduce(CustomTestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        # Pretend to be on Hopper so the tests run without a GPU, and silence the
        # info_once hints, which need transformers' logger patch.
        stack.enter_context(
            patch.object(
                marlin_utils.torch.cuda, "get_device_capability", return_value=(9, 0)
            )
        )
        stack.enter_context(
            patch.object(marlin_utils, "maybe_warn_marlin_atomic_add_env")
        )
        stack.enter_context(patch.object(marlin_utils, "maybe_warn_marlin_atomic_add"))
        # Keep deterministic inference off unless a test turns it on.
        stack.enter_context(envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False))

    def check(self, n=NARROW_N, k=LARGE_K, device=CUDA, dtype=torch.float16):
        return marlin_utils.should_use_atomic_add_reduce(16, n, k, device, dtype)

    def test_off_by_default(self):
        with envs.SGLANG_MARLIN_USE_ATOMIC_ADD.override(False):
            self.assertFalse(self.check())
        marlin_utils.maybe_warn_marlin_atomic_add_env.assert_called_once()

    def test_opt_in_turns_it_on(self):
        with envs.SGLANG_MARLIN_USE_ATOMIC_ADD.override(True):
            self.assertTrue(self.check())

    def test_deterministic_inference_wins_over_opt_in(self):
        with (
            envs.SGLANG_MARLIN_USE_ATOMIC_ADD.override(True),
            envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(True),
        ):
            self.assertFalse(self.check())

    def test_shapes_outside_the_condition_never_use_it(self):
        with envs.SGLANG_MARLIN_USE_ATOMIC_ADD.override(True):
            self.assertFalse(self.check(n=2048))
            self.assertFalse(self.check(k=2047))
            self.assertFalse(self.check(device=torch.device("cpu")))

    def test_bf16_before_sm90_stays_off(self):
        with (
            envs.SGLANG_MARLIN_USE_ATOMIC_ADD.override(True),
            patch.object(
                marlin_utils.torch.cuda, "get_device_capability", return_value=(8, 0)
            ),
        ):
            self.assertFalse(self.check(dtype=torch.bfloat16))
            self.assertTrue(self.check(dtype=torch.float16))


if __name__ == "__main__":
    unittest.main(verbosity=3)
