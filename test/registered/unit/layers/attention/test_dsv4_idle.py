"""IDLE attention-DP ranks participate in MoE without planning attention."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSV4IdleForward(unittest.TestCase):
    def _make_backend(self):
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.mtp_enabled = False
        backend.online_c128_mtp = Mock()
        backend.forward_metadata = object()
        backend.tail_forward_metadata = object()
        backend.encoder_replay = True
        backend._build_forward_metadata = Mock(
            side_effect=AssertionError("IDLE must not build attention metadata")
        )
        backend.init_forward_metadata_in_graph = Mock()
        return backend

    def test_non_mtp_idle_clears_stale_metadata(self):
        backend = self._make_backend()
        backend.init_forward_metadata(SimpleNamespace(forward_mode=ForwardMode.IDLE))
        backend.online_c128_mtp.clear.assert_called_once_with()
        backend._build_forward_metadata.assert_not_called()
        backend.init_forward_metadata_in_graph.assert_not_called()
        self.assertIsNone(backend.forward_metadata)
        self.assertIsNone(backend.tail_forward_metadata)
        self.assertFalse(backend.encoder_replay)

    def test_idle_with_stale_verify_mode_clears_metadata(self):
        backend = self._make_backend()
        backend.init_forward_metadata(
            SimpleNamespace(
                forward_mode=ForwardMode.IDLE,
                _original_forward_mode=ForwardMode.TARGET_VERIFY,
            )
        )
        backend.online_c128_mtp.clear.assert_called_once_with()
        backend._build_forward_metadata.assert_not_called()

    def test_non_mtp_idle_skips_attention(self):
        backend = self._make_backend()
        q = torch.empty((0, 2, 4))
        output = backend._forward_attention(
            q,
            q,
            q,
            SimpleNamespace(v_head_dim=3),
            SimpleNamespace(forward_mode=ForwardMode.IDLE),
            compress_ratio=0,
        )
        self.assertEqual(output.shape, (0, 2, 3))


if __name__ == "__main__":
    unittest.main()
