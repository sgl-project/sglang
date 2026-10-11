"""Hopper admission, load-time weight layout and split/verify compressor dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.dsv4 import dsv41_sparse
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCompressorAdmission(unittest.TestCase):
    def test_hopper_opt_in_preserves_other_devices(self):
        for enabled in (False, True):
            for major, expected in ((8, False), (9, enabled), (10, True), (12, True)):
                with (
                    self.subTest(enabled=enabled, major=major),
                    envs.SGLANG_OPT_USE_FUSED_DSV41_COMPRESS_SM90.override(enabled),
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.version.hip", None),
                    patch("torch.cuda.get_device_capability", return_value=(major, 0)),
                ):
                    self.assertEqual(
                        dsv41_sparse.fused_low_ratio_compress_supported(), expected
                    )
        with (
            envs.SGLANG_OPT_USE_FUSED_DSV41_COMPRESS_SM90.override(True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            self.assertFalse(dsv41_sparse.fused_low_ratio_compress_supported())
        for supported in (False, True):
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.version.hip", "test"),
                patch.object(
                    dsv41_sparse, "is_gfx95_supported", return_value=supported
                ),
            ):
                self.assertEqual(
                    dsv41_sparse.fused_low_ratio_compress_supported(), supported
                )

    def test_layout_is_fixed_before_capture(self):
        for ratio in (1, 2):
            for enabled in (False, True):
                with (
                    self.subTest(ratio=ratio, enabled=enabled),
                    envs.SGLANG_OPT_USE_FUSED_DSV41_COMPRESS_SM90.override(enabled),
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.version.hip", None),
                    patch("torch.cuda.get_device_capability", return_value=(9, 0)),
                ):
                    compressor = dsv41_sparse.DeepseekV41Compressor(8, 4, ratio, 1e-6)
                with envs.SGLANG_OPT_USE_FUSED_DSV41_COMPRESS_SM90.override(
                    not enabled
                ):
                    self.assertEqual(compressor.use_fused_compress, enabled)
                    self.assertEqual(
                        hasattr(compressor, "wkv_gate"), enabled and ratio == 2
                    )
                    self.assertEqual(
                        hasattr(compressor, "wkv"), not enabled or ratio == 1
                    )
                    self.assertEqual(
                        hasattr(compressor, "wgate"), not enabled and ratio == 2
                    )


class TestCompressorDispatch(unittest.TestCase):
    def setUp(self):
        self.backend = SimpleNamespace(
            is_dspark_draft=False,
            speculative_num_draft_tokens=6,
            _low_ratio_compress_fused=Mock(),
            _low_ratio_compress_torch=Mock(),
            _low_ratio_compress_decode=Mock(),
        )
        self.layer = SimpleNamespace(
            compress_ratio=2, compressor=SimpleNamespace(use_fused_compress=True)
        )

    def test_split_prefill_uses_request_offsets_without_padding_owner(self):
        starts = torch.tensor([0, 3], dtype=torch.int32)
        lengths = torch.tensor([3, 2, 99], dtype=torch.int32)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.SPLIT_PREFILL,
            extend_start_loc=starts,
            extend_seq_lens=lengths,
        )
        DeepseekV4AttnBackend._low_ratio_compress(
            self.backend, self.layer, torch.empty(8, 8), None, None, batch
        )
        self.backend._low_ratio_compress_torch.assert_not_called()
        offsets = self.backend._low_ratio_compress_fused.call_args.kwargs[
            "extend_offsets"
        ]
        torch.testing.assert_close(offsets[0], starts)
        torch.testing.assert_close(offsets[1], lengths[:2])

    def test_verify_requires_equal_request_blocks(self):
        from sglang.srt.layers.attention import deepseek_v4_backend as backend_module

        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY, batch_size=2)
        for rows, fused in ((12, True), (11, False)):
            self.backend._low_ratio_compress_fused.reset_mock()
            self.backend._low_ratio_compress_torch.reset_mock()
            with patch.object(
                backend_module, "read_ragged_verify_mode", return_value=None
            ):
                DeepseekV4AttnBackend._low_ratio_compress(
                    self.backend, self.layer, torch.empty(rows, 8), None, None, batch
                )
            self.assertEqual(self.backend._low_ratio_compress_fused.called, fused)
            self.assertEqual(self.backend._low_ratio_compress_torch.called, not fused)
            if fused:
                self.assertEqual(
                    self.backend._low_ratio_compress_fused.call_args.kwargs[
                        "draft_len"
                    ],
                    6,
                )


if __name__ == "__main__":
    unittest.main()
