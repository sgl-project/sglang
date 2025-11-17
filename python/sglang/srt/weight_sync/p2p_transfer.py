import logging
import threading
import zmq

from typing import Optional

from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.utils import get_free_port
from sglang.srt.utils.common import format_tcp_address

logger = logging.getLogger(__name__)


class TransferHandle:
    """A handle that mimics torch.distributed handle interface for compatibility"""

    def __init__(self):
        self.completed = False
        self.success = False
        self._event = threading.Event()

    def wait(self):
        """Wait for the transfer to complete"""
        self._event.wait()
        if not self.success:
            raise RuntimeError("P2P weight transfer failed")

    def _mark_done(self, success: bool):
        """Internal method to mark completion"""
        self.success = success
        self.completed = True
        self._event.set()


class P2PTransferEngine:

    def __init__(self, hostname: str, gpu_id: int, ib_device: Optional[str] = None):
        self._init_transfer_engine(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=ib_device,
        )

        # Use synchronous ZMQ socket for simplicity
        self.context = zmq.Context()
        self.client_socket = self.context.socket(zmq.DEALER)

        # Bind to a free port (different from Mooncake's RPC port)
        self.p2p_port = self._bind_server_socket()

    def register(self, ptr, length):
        """Register GPU memory buffer for RDMA Write"""
        self.engine.register(ptr, length)

    def deregister(self, ptr):
        """Deregister GPU memory buffer"""
        self.engine.deregister(ptr)

    def connect(self, remote_ip: str, remote_port: int):
        """Connect to the remote weight update engine"""
        # Disconnect from the previous client address if exists
        if hasattr(self, 'client_ip') and hasattr(self, 'client_port'):
            try:
                self.client_socket.disconnect(format_tcp_address(self.client_ip, self.client_port))
            except Exception as e:
                logger.warning(f"Failed to disconnect from previous address: {e}")

        self.client_ip = remote_ip
        self.client_port = remote_port

        self.client_socket.connect(format_tcp_address(remote_ip, remote_port))
        logger.info(f"P2P transfer engine connected to {remote_ip}:{remote_port}")

    def send_back_address(self, ptr, length):
        """Send buffer address to the sender (non-blocking)"""
        self.client_socket.send_json({
            "ptr": ptr,
            "length": length,
            "session_id": self.engine.get_session_id()
        })

    def recv_complete_signal(self):
        """
        Receive completion signal from sender.
        Returns a handle compatible with torch.distributed handle interface.

        This method returns immediately with a handle. The actual receiving
        happens in a background thread. Call handle.wait() to block until completion.
        """
        handle = TransferHandle()

        def _recv_thread():
            try:
                # Block until we receive the completion signal
                msg = self.client_socket.recv_json()
                success = msg.get("status", "failed") == "completed"
                handle._mark_done(success)
                if success:
                    logger.debug("P2P weight transfer completed successfully")
                else:
                    error_msg = msg.get('error', 'unknown error')
                    logger.error(f"P2P weight transfer failed: {error_msg}")
            except Exception as e:
                logger.error(f"Error receiving completion signal: {e}")
                handle._mark_done(False)

        # Start receiving in a background thread to make it non-blocking
        thread = threading.Thread(target=_recv_thread, daemon=True)
        thread.start()

        return handle

    def _init_transfer_engine(self, hostname: str, gpu_id: int, ib_device: Optional[str] = None):
        """Initialize the underlying Mooncake transfer engine"""
        self.engine = MooncakeTransferEngine(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=ib_device,
        )

    def _bind_server_socket(self):
        """Bind to a free port and return the port number"""
        p2p_port = get_free_port()
        self.client_socket.bind(format_tcp_address(self.engine.hostname, p2p_port))
        logger.info(f"P2P transfer engine bound to {self.engine.hostname}:{p2p_port}")
        return p2p_port

    def get_listen_address(self):
        """Get the address that this engine is listening on"""
        return self.engine.hostname, self.p2p_port