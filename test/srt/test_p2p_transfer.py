"""
Unit tests for P2P weight transfer using Mooncake RDMA.

This test simulates the full workflow:
1. Sender (WeightUpdateEngine) starts and listens for requests
2. Receiver (P2PTransferEngine) connects to sender
3. Receiver sends buffer address
4. Sender transfers data via RDMA
5. Sender sends completion signal
6. Receiver's handle.wait() returns

Note: This test uses mock objects to avoid requiring actual InfiniBand hardware.
"""

import logging
import threading
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch
import zmq

from sglang.srt.weight_sync.p2p_transfer import P2PTransferEngine, TransferHandle

logger = logging.getLogger(__name__)


class MockMooncakeTransferEngine:
    """Mock version of MooncakeTransferEngine for testing without RDMA hardware"""

    def __init__(self, hostname: str, gpu_id: int, ib_device=None):
        self.hostname = hostname
        self.gpu_id = gpu_id
        self.ib_device = ib_device
        self.rpc_port = 50051  # Mock RPC port
        self.registered_buffers = {}

        # Mock the underlying engine
        self.engine = Mock()
        self.engine.get_rpc_port = Mock(return_value=self.rpc_port)

    def register(self, ptr, length):
        """Mock memory registration"""
        self.registered_buffers[ptr] = length
        logger.debug(f"Mock registered buffer: ptr={hex(ptr)}, length={length}")

    def deregister(self, ptr):
        """Mock memory deregistration"""
        if ptr in self.registered_buffers:
            del self.registered_buffers[ptr]
            logger.debug(f"Mock deregistered buffer: ptr={hex(ptr)}")

    def transfer_sync(self, session_id: str, src_ptr: int, dst_ptr: int, length: int):
        """Mock RDMA transfer"""
        logger.debug(
            f"Mock RDMA transfer: session={session_id}, "
            f"src={hex(src_ptr)}, dst={hex(dst_ptr)}, len={length}"
        )
        # Simulate successful transfer
        return 0

    def get_session_id(self):
        """Get mock session ID"""
        return f"{self.hostname}:{self.rpc_port}"


class MockWeightUpdateEngine:
    """
    Mock version of WeightUpdateEngine (the sender side).

    This simulates the behavior of the actual weight update engine:
    1. Listens for address requests from receivers
    2. Transfers data via RDMA (mocked)
    3. Sends completion signals back
    """

    def __init__(self, hostname: str, port: int):
        self.hostname = hostname
        self.port = port
        self.running = False

        # Mock transfer engine
        self.transfer_engine = MockMooncakeTransferEngine(hostname, gpu_id=0)

        # ZMQ socket for control messages
        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{hostname}:{port}")

        # Mock weight data
        self.weight_data = {}  # name -> (ptr, length)

        logger.info(f"Mock WeightUpdateEngine bound to {hostname}:{port}")

    def register_weight(self, name: str, ptr: int, length: int):
        """Register a weight buffer for sending"""
        self.transfer_engine.register(ptr, length)
        self.weight_data[name] = (ptr, length)

    def start(self):
        """Start the message handling loop"""
        if self.running:
            return

        self.running = True
        self.message_thread = threading.Thread(target=self._message_loop, daemon=True)
        self.message_thread.start()
        logger.info("Mock WeightUpdateEngine started")

    def stop(self):
        """Stop the engine"""
        self.running = False
        if hasattr(self, 'message_thread'):
            self.message_thread.join(timeout=2)
        self.router_socket.close()
        self.context.term()
        logger.info("Mock WeightUpdateEngine stopped")

    def _message_loop(self):
        """Handle incoming requests from receivers"""
        while self.running:
            try:
                if self.router_socket.poll(timeout=100):  # 100ms timeout
                    # Receive: [identity, empty_frame, json_data]
                    identity = self.router_socket.recv()
                    empty = self.router_socket.recv()
                    json_data = self.router_socket.recv_json()

                    logger.debug(f"Received request: {json_data}")

                    # Extract client info
                    client_ptr = json_data["ptr"]
                    client_length = json_data["length"]
                    client_session_id = json_data["session_id"]

                    # Simulate RDMA transfer
                    # In real implementation, this would call transfer_engine.transfer_sync()
                    # For now, we just simulate it
                    time.sleep(0.01)  # Simulate transfer time

                    # Send completion signal
                    self.router_socket.send_multipart(
                        [
                            identity,
                            b"",
                            zmq.utils.jsonapi.dumps({"status": "completed", "error": ""}),
                        ]
                    )
                    logger.debug(f"Sent completion signal to client")

            except zmq.ZMQError as e:
                if self.running:
                    logger.error(f"ZMQ error: {e}")
                break
            except Exception as e:
                logger.error(f"Error in message loop: {e}", exc_info=True)


class TestTransferHandle(unittest.TestCase):
    """Test the TransferHandle class"""

    def test_handle_wait_success(self):
        """Test that wait() blocks until marked done"""
        handle = TransferHandle()

        # Start a thread that marks done after a delay
        def mark_done_later():
            time.sleep(0.1)
            handle._mark_done(True)

        thread = threading.Thread(target=mark_done_later)
        thread.start()

        # wait() should block until _mark_done is called
        start_time = time.time()
        handle.wait()
        elapsed = time.time() - start_time

        self.assertTrue(handle.completed)
        self.assertTrue(handle.success)
        self.assertGreaterEqual(elapsed, 0.1)  # Should have waited at least 0.1s

        thread.join()

    def test_handle_wait_failure(self):
        """Test that wait() raises exception on failure"""
        handle = TransferHandle()

        def mark_failed():
            time.sleep(0.05)
            handle._mark_done(False)

        thread = threading.Thread(target=mark_failed)
        thread.start()

        # wait() should raise RuntimeError
        with self.assertRaises(RuntimeError):
            handle.wait()

        thread.join()


class TestP2PTransferEngine(unittest.TestCase):
    """Test P2PTransferEngine (receiver side)"""

    @patch('sglang.srt.weight_sync.p2p_transfer.MooncakeTransferEngine', MockMooncakeTransferEngine)
    def test_initialization(self):
        """Test that P2PTransferEngine initializes correctly"""
        engine = P2PTransferEngine(
            hostname="127.0.0.1",
            gpu_id=0,
            ib_device=None
        )

        # Check that engine is initialized
        self.assertIsNotNone(engine.engine)
        self.assertIsNotNone(engine.client_socket)
        self.assertIsNotNone(engine.p2p_port)

        # Clean up
        engine.client_socket.close()
        engine.context.term()

    @patch('sglang.srt.weight_sync.p2p_transfer.MooncakeTransferEngine', MockMooncakeTransferEngine)
    def test_register_deregister(self):
        """Test memory registration and deregistration"""
        engine = P2PTransferEngine(
            hostname="127.0.0.1",
            gpu_id=0,
            ib_device=None
        )

        # Test registration
        ptr = 0x12345678
        length = 4096
        engine.register(ptr, length)

        self.assertIn(ptr, engine.engine.registered_buffers)
        self.assertEqual(engine.engine.registered_buffers[ptr], length)

        # Test deregistration
        engine.deregister(ptr)
        self.assertNotIn(ptr, engine.engine.registered_buffers)

        # Clean up
        engine.client_socket.close()
        engine.context.term()


class TestP2PEndToEnd(unittest.TestCase):
    """End-to-end tests with sender and receiver"""

    @patch('sglang.srt.weight_sync.p2p_transfer.MooncakeTransferEngine', MockMooncakeTransferEngine)
    def test_full_transfer_flow(self):
        """Test complete transfer flow from sender to receiver"""
        # Start mock sender
        sender_port = 18000  # Use a fixed port for testing
        sender = MockWeightUpdateEngine(hostname="127.0.0.1", port=sender_port)
        sender.start()

        try:
            # Give sender time to start
            time.sleep(0.1)

            # Initialize receiver
            receiver = P2PTransferEngine(
                hostname="127.0.0.1",
                gpu_id=0,
                ib_device=None
            )

            # Connect receiver to sender
            receiver.connect("127.0.0.1", sender_port)

            # Simulate weight buffer
            mock_ptr = 0x7F000000
            mock_length = 1024

            # Register buffer
            receiver.register(mock_ptr, mock_length)

            # Send address to sender
            receiver.send_back_address(mock_ptr, mock_length)

            # Receive completion signal (non-blocking)
            handle = receiver.recv_complete_signal()

            # This should be a TransferHandle
            self.assertIsInstance(handle, TransferHandle)

            # Wait for completion (blocking)
            handle.wait()

            # Check that transfer completed successfully
            self.assertTrue(handle.completed)
            self.assertTrue(handle.success)

            # Clean up receiver
            receiver.deregister(mock_ptr)
            receiver.client_socket.close()
            receiver.context.term()

        finally:
            # Stop sender
            sender.stop()

    @patch('sglang.srt.weight_sync.p2p_transfer.MooncakeTransferEngine', MockMooncakeTransferEngine)
    def test_multiple_transfers(self):
        """Test multiple sequential transfers"""
        sender_port = 18001
        sender = MockWeightUpdateEngine(hostname="127.0.0.1", port=sender_port)
        sender.start()

        try:
            time.sleep(0.1)

            receiver = P2PTransferEngine(
                hostname="127.0.0.1",
                gpu_id=0,
                ib_device=None
            )
            receiver.connect("127.0.0.1", sender_port)

            # Perform multiple transfers
            for i in range(3):
                mock_ptr = 0x7F000000 + i * 1024
                mock_length = 1024

                receiver.register(mock_ptr, mock_length)
                receiver.send_back_address(mock_ptr, mock_length)
                handle = receiver.recv_complete_signal()
                handle.wait()

                self.assertTrue(handle.success, f"Transfer {i} failed")
                receiver.deregister(mock_ptr)

            # Clean up
            receiver.client_socket.close()
            receiver.context.term()

        finally:
            sender.stop()

    @patch('sglang.srt.weight_sync.p2p_transfer.MooncakeTransferEngine', MockMooncakeTransferEngine)
    def test_concurrent_transfers(self):
        """Test concurrent transfers (multiple handles waiting)"""
        sender_port = 18002
        sender = MockWeightUpdateEngine(hostname="127.0.0.1", port=sender_port)
        sender.start()

        try:
            time.sleep(0.1)

            receiver = P2PTransferEngine(
                hostname="127.0.0.1",
                gpu_id=0,
                ib_device=None
            )
            receiver.connect("127.0.0.1", sender_port)

            # Start multiple transfers
            handles = []
            ptrs = []
            for i in range(3):
                mock_ptr = 0x7F000000 + i * 1024
                mock_length = 1024
                ptrs.append(mock_ptr)

                receiver.register(mock_ptr, mock_length)
                receiver.send_back_address(mock_ptr, mock_length)
                handle = receiver.recv_complete_signal()
                handles.append(handle)

            # Wait for all transfers to complete
            for i, handle in enumerate(handles):
                handle.wait()
                self.assertTrue(handle.success, f"Transfer {i} failed")
                receiver.deregister(ptrs[i])

            # Clean up
            receiver.client_socket.close()
            receiver.context.term()

        finally:
            sender.stop()


class TestCompatibilityWithTorchDistributed(unittest.TestCase):
    """Test that P2PTransferEngine handle is compatible with torch.distributed pattern"""

    @patch('sglang.srt.weight_sync.p2p_transfer.MooncakeTransferEngine', MockMooncakeTransferEngine)
    def test_handle_interface_compatibility(self):
        """Test that handle has the same interface as torch.distributed handles"""
        sender_port = 18003
        sender = MockWeightUpdateEngine(hostname="127.0.0.1", port=sender_port)
        sender.start()

        try:
            time.sleep(0.1)

            receiver = P2PTransferEngine(
                hostname="127.0.0.1",
                gpu_id=0,
                ib_device=None
            )
            receiver.connect("127.0.0.1", sender_port)

            # Simulate the pattern used in model_runner.py
            handles = []

            # Simulating multiple weight tensors
            for i in range(2):
                mock_ptr = 0x7F000000 + i * 1024
                mock_length = 1024

                receiver.register(mock_ptr, mock_length)
                receiver.send_back_address(mock_ptr, mock_length)
                handles.append(receiver.recv_complete_signal())

            # This is exactly how it's used in model_runner.py
            for handle in handles:
                handle.wait()  # Should not raise

            # Clean up
            receiver.client_socket.close()
            receiver.context.term()

        finally:
            sender.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    unittest.main()
