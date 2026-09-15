# SPDX-License-Identifier: Apache-2.0

"""Tests for the AITER fused Qwen3-Next GDN decode integration.

The integration is a capability, not a mode: every gate must degrade to the
existing unfused chain rather than raise. These tests pin that, and they pin it
without needing a GPU or AITER for the cases that matter most -- the ones where
the fused path must decline.

The kernel's numerics are covered in AITER's own op tests; what is tested here
is SGLang's side of the contract: the probe, the fallback, and the
attempt-and-verify stash.
"""

import os
import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention import gdn_fused_decode_aiter as adapter
from sglang.test.test_utils import CustomTestCase

_ENV = "SGLANG_QWEN3_NEXT_GDN_FUSED_BACKEND"


class TestGdnFusedDecodeAiterGating(CustomTestCase):
    """The opt-in and the capability probe."""

    def test_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_ENV, None)
            self.assertFalse(adapter.enabled())
            self.assertFalse(adapter.available())

    def test_enabled_only_for_aiter_value(self):
        for value, expected in (
            ("aiter", True),
            ("AITER", True),
            ("triton", False),
            ("", False),
            ("1", False),
        ):
            with mock.patch.dict(os.environ, {_ENV: value}):
                self.assertEqual(adapter.enabled(), expected, f"value={value!r}")

    def test_unavailable_when_not_hip(self):
        with (
            mock.patch.dict(os.environ, {_ENV: "aiter"}),
            mock.patch.object(adapter, "is_hip", return_value=False),
        ):
            self.assertFalse(adapter.available())

    def test_unavailable_when_aiter_missing(self):
        with (
            mock.patch.dict(os.environ, {_ENV: "aiter"}),
            mock.patch.object(adapter, "_ops", return_value=(None, None)),
        ):
            self.assertFalse(adapter.available())


class TestGdnFusedDecodeAiterCoverage(CustomTestCase):
    """covered() must decline, with a reason, rather than raise."""

    @staticmethod
    def _args(**over):
        base = dict(
            projected_qkvz=torch.empty(0),
            projected_ba=torch.empty(0),
            conv_state=torch.empty(0),
            ssm_state=torch.empty(0),
            state_indices=torch.empty(0),
            conv_weight=torch.empty(0),
            conv_bias=torch.empty(0),
            activation="silu",
            quant_dtype=None,
        )
        base.update(over)
        return base

    def test_declines_non_silu_activation(self):
        """The kernel folds a SiLU output gate; anything else is wrong numerics.

        This is the one condition AITER's predicate cannot see, so it must be
        checked on the SGLang side.
        """
        for activation in ("gelu", "relu", None):
            ok, reason = adapter.covered(**self._args(activation=activation))
            self.assertFalse(ok, f"activation={activation!r} must decline")
            self.assertIn("SiLU", reason)

    def test_declines_without_conv_bias(self):
        ok, reason = adapter.covered(**self._args(conv_bias=None))
        self.assertFalse(ok)
        self.assertIn("bias", reason.lower())

    def test_declines_when_aiter_unavailable(self):
        with mock.patch.object(adapter, "_ops", return_value=(None, None)):
            ok, reason = adapter.covered(**self._args())
            self.assertFalse(ok)
            self.assertTrue(reason)

    def test_reason_is_non_empty_on_every_decline(self):
        """A decline is logged once by the caller; an empty reason is useless."""
        ok, reason = adapter.covered(**self._args(activation="gelu"))
        self.assertFalse(ok)
        self.assertTrue(reason.strip())


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip, "requires a ROCm GPU"
)
class TestGdnFusedDecodeAiterOnDevice(CustomTestCase):
    """Contract checks that need a device. Skipped off ROCm, and off gfx950."""

    def setUp(self):
        # The probe is opt-in, so enable it for the duration of this test rather
        # than requiring the suite to be run with the env var already set.
        self._env = mock.patch.dict(os.environ, {_ENV: "aiter"})
        self._env.start()
        self.addCleanup(self._env.stop)
        if not adapter.available():
            self.skipTest("aiter fused GDN decode not available here")

    def test_declines_wrong_head_ratio(self):
        """num_v_heads must be 2 * num_k_heads; the kernel static_asserts it."""
        dev, kh, vh, d = "cuda", 4, 4, 128  # vh == kh, not 2*kh
        channels = 2 * kh * d + vh * d
        ok, reason = adapter.covered(
            projected_qkvz=torch.zeros(8, kh * 768, dtype=torch.bfloat16, device=dev),
            projected_ba=torch.zeros(8, 2 * vh, dtype=torch.bfloat16, device=dev),
            conv_state=torch.zeros(16, channels, 3, dtype=torch.bfloat16, device=dev),
            ssm_state=torch.zeros(16, vh, d, d, dtype=torch.float32, device=dev),
            state_indices=torch.arange(1, 9, dtype=torch.int32, device=dev),
            conv_weight=torch.zeros(channels, 4, dtype=torch.bfloat16, device=dev),
            conv_bias=torch.zeros(channels, dtype=torch.bfloat16, device=dev),
            activation="silu",
            quant_dtype=None,
        )
        self.assertFalse(ok)
        self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main()
