"""Unit tests for NPU stream_management module (no real NPU required)."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.hardware_backend.npu import stream_management as sm


class TestNpuStreamManagement(unittest.TestCase):
    def setUp(self):
        sm.cmo_stream = None
        sm.share_stream = None
        sm.routed_stream = None
        sm.indexer_weight_stream = None

    def test_share_stream_lifecycle(self):
        fake = MagicMock(name="share_stream")
        sm.set_share_stream(fake)
        self.assertIs(sm.get_share_stream(), fake)
        sm.wait_share_stream()  # should not raise when stream set

    def test_routed_stream_lifecycle(self):
        fake = MagicMock(name="routed_stream")
        sm.set_routed_stream(fake)
        self.assertIs(sm.get_routed_stream(), fake)

    def test_cmo_stream_setters(self):
        fake = MagicMock(name="cmo_stream")
        sm.set_cmo_stream(fake)
        self.assertIs(sm.get_cmo_stream(), fake)

    @patch("sglang.srt.hardware_backend.npu.stream_management.torch.npu")
    def test_get_indexer_weight_stream_lazy_init(self, mock_npu):
        mock_npu.Stream.return_value = MagicMock(name="idx_stream")
        s1 = sm.get_indexer_weight_stream()
        s2 = sm.get_indexer_weight_stream()
        self.assertIs(s1, s2)
        mock_npu.Stream.assert_called_once()

    @patch("sglang.srt.hardware_backend.npu.stream_management._device_module")
    def test_process_shared_expert_creates_stream(self, mock_dev_mod):
        dev = MagicMock()
        stream = MagicMock()
        ctx = MagicMock()
        dev.Stream.return_value = stream
        dev.current_stream.return_value = MagicMock()
        dev.stream.return_value.__enter__ = MagicMock(return_value=None)
        dev.stream.return_value.__exit__ = MagicMock(return_value=False)
        mock_dev_mod.return_value = dev

        out = sm.process_shared_expert("hidden", lambda h: "ok")
        self.assertEqual(out, "ok")
        dev.Stream.assert_called_once()

    def test_shared_expert_alias(self):
        self.assertIs(sm.shared_expert_on_independent_stream, sm.process_shared_expert)


if __name__ == "__main__":
    unittest.main()
