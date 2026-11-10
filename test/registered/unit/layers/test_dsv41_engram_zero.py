"""Ablation: SGLANG_DSV41_ENGRAM_ZERO returns the residual unchanged."""

from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.engram import Engram
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEngramZeroAblation(CustomTestCase):
    def test_zero_returns_input_without_embed(self):
        x = torch.randn(3, 4, 8)
        dummy = Engram.__new__(Engram)
        with envs.SGLANG_DSV41_ENGRAM_ZERO.override(True):
            out = Engram.forward(dummy, x, torch.zeros(3, 2, dtype=torch.int64))
        self.assertIs(out, x)

    def test_default_is_off(self):
        with envs.SGLANG_DSV41_ENGRAM_ZERO.override(False):
            self.assertFalse(envs.SGLANG_DSV41_ENGRAM_ZERO.get())

    def test_overlap_default_on(self):
        with envs.SGLANG_DSV41_ENGRAM_OVERLAP.override(True):
            self.assertTrue(envs.SGLANG_DSV41_ENGRAM_OVERLAP.get())

    def test_finish_from_gathered_zero_skips(self):
        x = torch.randn(3, 4, 8)
        dummy = Engram.__new__(Engram)
        with envs.SGLANG_DSV41_ENGRAM_ZERO.override(True):
            out = Engram.finish_from_gathered(dummy, x, torch.zeros(3, 2, 8))
        self.assertIs(out, x)
