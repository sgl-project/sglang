"""Unit tests for HiCache draft budget adjustment — no server, no model loading.

When ``--hicache-size`` caps the target host pool and a SIDECAR draft model
adds a second host pool with the same token count but a larger per-token byte
cost (e.g. quantized target + bf16 draft), the combined allocation can exceed
the user-specified budget.  ``_adjust_hicache_size_for_draft`` shrinks the
budget so the *total* (target + draft) stays within the cap.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.kv_cache_builder import (
    _adjust_hicache_size_for_draft,
    _device_pool_size_per_token,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The function under test does a local import of _apply_fields, so we patch it
# at its source module to prevent real ServerArgs mutation.
_APPLY_FIELDS_PATH = "sglang.srt.arg_groups.overrides._apply_fields"


def _make_mha_pool(head_dim, head_num, layer_num, dtype, size=4096):
    """Create a minimal MHA-like pool with the attributes used by the sizing logic."""
    return SimpleNamespace(
        head_dim=head_dim,
        head_num=head_num,
        layer_num=layer_num,
        store_dtype=dtype,
        dtype=dtype,
        size=size,
    )


def _make_mla_pool(kv_cache_dim, layer_num, dtype, size=4096):
    """Create a minimal MLA-like pool with the attributes used by the sizing logic."""
    return SimpleNamespace(
        kv_cache_dim=kv_cache_dim,
        layer_num=layer_num,
        store_dtype=dtype,
        dtype=dtype,
        size=size,
    )


def _make_server_args(hicache_size=4, hicache_ratio=2.0):
    return SimpleNamespace(
        hicache_size=hicache_size,
        hicache_ratio=hicache_ratio,
    )


class TestDevicePoolSizePerToken(CustomTestCase):
    """Tests for _device_pool_size_per_token geometry-based computation."""

    def test_mha_pool_bf16(self):
        pool = _make_mha_pool(
            head_dim=128, head_num=8, layer_num=28, dtype=torch.bfloat16
        )
        # head_dim * head_num * layer_num * itemsize * 2 (K+V)
        self.assertEqual(_device_pool_size_per_token(pool), 128 * 8 * 28 * 2 * 2)

    def test_mha_pool_fp8(self):
        pool = _make_mha_pool(
            head_dim=128, head_num=8, layer_num=28, dtype=torch.float8_e4m3fn
        )
        # fp8 itemsize = 1
        self.assertEqual(_device_pool_size_per_token(pool), 128 * 8 * 28 * 1 * 2)

    def test_mla_pool(self):
        pool = _make_mla_pool(kv_cache_dim=512, layer_num=28, dtype=torch.bfloat16)
        # kv_cache_dim * layer_num * itemsize = 512 * 28 * 2
        self.assertEqual(_device_pool_size_per_token(pool), 512 * 28 * 2)

    def test_zero_size_pool_returns_zero(self):
        pool = SimpleNamespace(
            head_dim=None,
            head_num=None,
            kv_cache_dim=None,
            layer_num=None,
            store_dtype=None,
            dtype=None,
            size=0,
            get_kv_size_bytes=lambda: 0,
        )
        self.assertEqual(_device_pool_size_per_token(pool), 0.0)


class TestAdjustHicacheSizeForDraft(CustomTestCase):
    """Tests for _adjust_hicache_size_for_draft budget shrinking."""

    def test_no_adjustment_when_hicache_size_zero(self):
        """Ratio mode (hicache_size <= 0) should not be adjusted."""
        server_args = _make_server_args(hicache_size=0, hicache_ratio=2.0)
        target = _make_mha_pool(128, 8, 28, torch.bfloat16)
        draft = _make_mha_pool(128, 8, 28, torch.bfloat16)

        with mock.patch(_APPLY_FIELDS_PATH) as mock_apply:
            _adjust_hicache_size_for_draft(server_args, target, (draft,))
            mock_apply.assert_not_called()

    def test_no_adjustment_when_no_draft_pools(self):
        server_args = _make_server_args(hicache_size=4, hicache_ratio=2.0)
        target = _make_mha_pool(128, 8, 28, torch.bfloat16)

        with mock.patch(_APPLY_FIELDS_PATH) as mock_apply:
            _adjust_hicache_size_for_draft(server_args, target, ())
            mock_apply.assert_not_called()

    def test_no_adjustment_when_target_spt_zero(self):
        server_args = _make_server_args(hicache_size=4, hicache_ratio=2.0)
        target = SimpleNamespace(
            head_dim=None,
            head_num=None,
            kv_cache_dim=None,
            layer_num=None,
            store_dtype=None,
            dtype=None,
            size=0,
            get_kv_size_bytes=lambda: 0,
        )
        draft = _make_mha_pool(128, 8, 28, torch.bfloat16)

        with mock.patch(_APPLY_FIELDS_PATH) as mock_apply:
            _adjust_hicache_size_for_draft(server_args, target, (draft,))
            mock_apply.assert_not_called()

    def test_equal_per_token_cost_halves_budget(self):
        """When target and draft have equal per-token cost, each gets half."""
        server_args = _make_server_args(hicache_size=4, hicache_ratio=2.0)
        target = _make_mha_pool(128, 8, 28, torch.bfloat16)
        draft = _make_mha_pool(128, 8, 28, torch.bfloat16)

        with mock.patch(_APPLY_FIELDS_PATH) as mock_apply:
            _adjust_hicache_size_for_draft(server_args, target, (draft,))
            mock_apply.assert_called_once()
            fields = mock_apply.call_args[0][1]
            # effective = 4 * spt / (spt + spt) = 2
            self.assertEqual(fields["hicache_size"], 2)
            self.assertAlmostEqual(fields["hicache_ratio"], 1.0)

    def test_quantized_target_bf16_draft(self):
        """fp8 target + bf16 draft: draft is 2x the per-token cost."""
        server_args = _make_server_args(hicache_size=4, hicache_ratio=2.0)
        target = _make_mha_pool(128, 8, 28, torch.float8_e4m3fn)  # spt = 57344
        draft = _make_mha_pool(128, 8, 28, torch.bfloat16)  # spt = 114688

        with mock.patch(_APPLY_FIELDS_PATH) as mock_apply:
            _adjust_hicache_size_for_draft(server_args, target, (draft,))
            mock_apply.assert_called_once()
            fields = mock_apply.call_args[0][1]
            # effective = 4 * 57344 / (57344 + 114688) = 4/3
            expected_size = int(4 * 57344 / (57344 + 114688))
            self.assertEqual(fields["hicache_size"], expected_size)
            # ratio_factor = 1/3
            expected_ratio = 2.0 * 57344 / (57344 + 114688)
            self.assertAlmostEqual(fields["hicache_ratio"], expected_ratio)

    def test_bf16_target_fp8_draft(self):
        """bf16 target + fp8 draft: draft is 0.5x the per-token cost."""
        server_args = _make_server_args(hicache_size=4, hicache_ratio=2.0)
        target = _make_mha_pool(128, 8, 28, torch.bfloat16)  # spt = 114688
        draft = _make_mha_pool(128, 8, 28, torch.float8_e4m3fn)  # spt = 57344

        with mock.patch(_APPLY_FIELDS_PATH) as mock_apply:
            _adjust_hicache_size_for_draft(server_args, target, (draft,))
            mock_apply.assert_called_once()
            fields = mock_apply.call_args[0][1]
            # effective = 4 * 114688 / (114688 + 57344) = 8/3
            expected_size = int(4 * 114688 / (114688 + 57344))
            self.assertEqual(fields["hicache_size"], expected_size)

    def test_mla_target_mha_draft(self):
        """MLA target + MHA draft: different pool types are handled."""
        server_args = _make_server_args(hicache_size=4, hicache_ratio=2.0)
        target = _make_mla_pool(kv_cache_dim=512, layer_num=28, dtype=torch.bfloat16)
        draft = _make_mha_pool(128, 8, 28, torch.bfloat16)

        with mock.patch(_APPLY_FIELDS_PATH) as mock_apply:
            _adjust_hicache_size_for_draft(server_args, target, (draft,))
            mock_apply.assert_called_once()
            fields = mock_apply.call_args[0][1]
            # target_spt = 512 * 28 * 2 = 28672
            # draft_spt = 128 * 8 * 28 * 2 * 2 = 114688
            target_spt = 28672
            draft_spt = 114688
            expected_size = int(4 * target_spt / (target_spt + draft_spt))
            self.assertEqual(fields["hicache_size"], expected_size)
            expected_ratio = 2.0 * target_spt / (target_spt + draft_spt)
            self.assertAlmostEqual(fields["hicache_ratio"], expected_ratio)


if __name__ == "__main__":
    unittest.main()
