# SPDX-License-Identifier: Apache-2.0
"""Unit tests for transfer engine abstraction."""

import ctypes
import unittest

from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    BaseTransferEngine,
    MockTransferEngine,
    create_transfer_engine,
)


class TestMockTransferEngine(unittest.TestCase):
    """Test MockTransferEngine simulated RDMA transfers."""

    def setUp(self):
        MockTransferEngine.reset()

    def tearDown(self):
        MockTransferEngine.reset()

    def test_session_id_unique(self):
        e1 = MockTransferEngine()
        e2 = MockTransferEngine()
        self.assertNotEqual(e1.session_id, e2.session_id)

    def test_custom_session_id(self):
        e = MockTransferEngine(session_id="test-123")
        self.assertEqual(e.session_id, "test-123")

    def test_register_and_deregister(self):
        e = MockTransferEngine()
        # Should not raise
        e.register_buffer(0x1000, 4096)
        e.deregister_buffer(0x1000)
        # Double deregister should be safe
        e.deregister_buffer(0x1000)

    def test_transfer_sync_copies_data(self):
        """Verify that transfer_sync copies data between memory regions."""
        e1 = MockTransferEngine(session_id="src")
        e2 = MockTransferEngine(session_id="dst")

        # Allocate source and destination buffers
        src_buf = (ctypes.c_byte * 1024)()
        dst_buf = (ctypes.c_byte * 1024)()

        # Fill source with pattern
        for i in range(1024):
            src_buf[i] = i % 128

        src_addr = ctypes.addressof(src_buf)
        dst_addr = ctypes.addressof(dst_buf)

        e1.register_buffer(src_addr, 1024)
        e2.register_buffer(dst_addr, 1024)

        # Transfer from src to dst
        ret = e1.transfer_sync("dst", src_addr, dst_addr, 1024)
        self.assertEqual(ret, 0)

        # Verify data was copied
        for i in range(1024):
            self.assertEqual(dst_buf[i], i % 128)

    def test_batch_transfer_sync(self):
        """Verify batch transfer copies multiple regions."""
        e1 = MockTransferEngine(session_id="src")
        e2 = MockTransferEngine(session_id="dst")

        buf1_src = (ctypes.c_byte * 256)()
        buf2_src = (ctypes.c_byte * 256)()
        buf1_dst = (ctypes.c_byte * 256)()
        buf2_dst = (ctypes.c_byte * 256)()

        for i in range(256):
            buf1_src[i] = 42
            buf2_src[i] = 99

        ret = e1.batch_transfer_sync(
            "dst",
            [ctypes.addressof(buf1_src), ctypes.addressof(buf2_src)],
            [ctypes.addressof(buf1_dst), ctypes.addressof(buf2_dst)],
            [256, 256],
        )
        self.assertEqual(ret, 0)
        self.assertEqual(buf1_dst[0], 42)
        self.assertEqual(buf2_dst[0], 99)

    def test_is_base_transfer_engine(self):
        e = MockTransferEngine()
        self.assertIsInstance(e, BaseTransferEngine)

    def test_reset_clears_state(self):
        MockTransferEngine(session_id="a")
        MockTransferEngine(session_id="b")
        self.assertEqual(len(MockTransferEngine._registry), 2)
        MockTransferEngine.reset()
        self.assertEqual(len(MockTransferEngine._registry), 0)


class TestCreateTransferEngine(unittest.TestCase):
    """Test factory function."""

    def setUp(self):
        MockTransferEngine.reset()

    def tearDown(self):
        MockTransferEngine.reset()

    def test_force_mock(self):
        engine = create_transfer_engine(force_mock=True)
        self.assertIsInstance(engine, MockTransferEngine)

    def test_returns_base_interface(self):
        engine = create_transfer_engine(force_mock=True)
        self.assertIsInstance(engine, BaseTransferEngine)


if __name__ == "__main__":
    unittest.main()
