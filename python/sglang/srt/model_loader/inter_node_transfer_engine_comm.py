"""
Inter-Node Transfer Engine Info Communication

This module provides ZMQ-based communication between nodes for sharing
transfer engine information in multi-node setups.
"""

import json
import logging
import threading
from typing import Dict, Optional, Tuple

import zmq

logger = logging.getLogger(__name__)


class TransferEngineInfoServer:
    """
    ZMQ server that runs on non-head nodes to serve transfer engine info
    for their local TP ranks.
    """

    def __init__(self, port: int, node_rank: int):
        self.port = port
        self.node_rank = node_rank
        self.context = zmq.Context()
        self.socket = None
        self.transfer_engine_info: Dict[int, Tuple] = {}
        self.running = False
        self.server_thread = None

    def set_transfer_engine_info(self, transfer_engine_info: Dict[int, Tuple]):
        """
        Set the transfer engine info that this server will serve.

        Args:
            transfer_engine_info: Dictionary mapping TP ranks to
                                 (session_id, weights_info_dict) tuples
        """
        self.transfer_engine_info = transfer_engine_info
        logger.info(
            f"Node {self.node_rank}: Set transfer engine info for ranks {list(transfer_engine_info.keys())}"
        )

    def start(self):
        """Start the ZMQ server in a background thread."""
        if self.running:
            logger.warning(
                f"Node {self.node_rank}: Transfer engine info server already running"
            )
            return

        self.running = True
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        logger.info(
            f"Node {self.node_rank}: Started transfer engine info server on port {self.port}"
        )

    def stop(self):
        """Stop the ZMQ server."""
        if not self.running:
            return

        self.running = False
        if self.socket:
            self.socket.close()
        if self.server_thread:
            self.server_thread.join(timeout=5.0)
        self.context.term()
        logger.info(f"Node {self.node_rank}: Stopped transfer engine info server")

    def _run_server(self):
        """Main server loop running in background thread."""
        try:
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{self.port}")
            logger.info(
                f"Node {self.node_rank}: ZMQ server listening on port {self.port}"
            )

            while self.running:
                try:
                    # Set receive timeout to check self.running periodically
                    self.socket.RCVTIMEO = 1000  # 1 second timeout

                    # Receive request
                    message = self.socket.recv_string()
                    request = json.loads(message)

                    logger.debug(f"Node {self.node_rank}: Received request: {request}")

                    # Process request
                    response = self._handle_request(request)

                    # Send response
                    self.socket.send_string(json.dumps(response))

                except zmq.Again:
                    # Timeout - continue loop to check self.running
                    continue
                except Exception as e:
                    logger.error(f"Node {self.node_rank}: Error in server loop: {e}")
                    if self.running:  # Only send error response if still running
                        try:
                            error_response = {"success": False, "error": str(e)}
                            self.socket.send_string(json.dumps(error_response))
                        except Exception:
                            pass  # Ignore errors when sending error response

        except Exception as e:
            logger.error(f"Node {self.node_rank}: Failed to start ZMQ server: {e}")
        finally:
            if self.socket:
                self.socket.close()

    def _handle_request(self, request: dict) -> dict:
        """
        Handle incoming request for transfer engine info.

        Args:
            request: Dictionary with 'action' and 'rank' fields

        Returns:
            Dictionary with response data
        """
        action = request.get("action")
        rank = request.get("rank")

        if action == "get_transfer_engine_info":
            if rank is None:
                return {"success": False, "error": "Missing rank parameter"}

            if rank not in self.transfer_engine_info:
                return {
                    "success": False,
                    "error": f"Rank {rank} not available on node {self.node_rank}",
                }

            # Return the transfer engine info for the requested rank
            session_id, weights_info_dict = self.transfer_engine_info[rank]
            return {
                "success": True,
                "rank": rank,
                "remote_instance_transfer_engine_info": (session_id, weights_info_dict),
            }

        else:
            return {"success": False, "error": f"Unknown action: {action}"}


class TransferEngineInfoClient:
    """
    ZMQ client that runs on head node to request transfer engine info
    from other nodes.
    """

    def __init__(self, timeout_ms: int = 5000):
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()

    def get_transfer_engine_info(
        self, host: str, port: int, rank: int
    ) -> Optional[Tuple]:
        """
        Request transfer engine info for a specific rank from a remote node.

        Args:
            host: Target node hostname/IP
            port: Target node ZMQ server port
            rank: TP rank to query

        Returns:
            (session_id, weights_info_dict) tuple or None if failed
        """
        socket = None
        try:
            socket = self.context.socket(zmq.REQ)
            socket.RCVTIMEO = self.timeout_ms
            socket.SNDTIMEO = self.timeout_ms

            # Connect to remote node
            remote_address = f"tcp://{host}:{port}"
            socket.connect(remote_address)
            logger.debug(f"Connected to {remote_address} for rank {rank}")

            # Send request
            request = {"action": "get_transfer_engine_info", "rank": rank}
            socket.send_string(json.dumps(request))

            # Receive response
            response_str = socket.recv_string()
            response = json.loads(response_str)

            logger.debug(f"Received response from {remote_address}: {response}")

            if response.get("success"):
                return response.get("remote_instance_transfer_engine_info")
            else:
                logger.error(
                    f"Request to {remote_address} failed: {response.get('error')}"
                )
                return None

        except zmq.Again:
            logger.error(f"Timeout requesting rank {rank} from {host}:{port}")
            return None
        except Exception as e:
            logger.error(f"Error requesting rank {rank} from {host}:{port}: {e}")
            return None
        finally:
            if socket:
                socket.close()

    def close(self):
        """Close the ZMQ context."""
        self.context.term()


# Global instances for server and client
_transfer_engine_info_server: Optional[TransferEngineInfoServer] = None
_transfer_engine_info_client: Optional[TransferEngineInfoClient] = None


def get_transfer_engine_info_client() -> Optional[TransferEngineInfoClient]:
    """Get the global transfer engine info client instance."""
    return _transfer_engine_info_client


def init_transfer_engine_info_server(
    port: int, node_rank: int
) -> TransferEngineInfoServer:
    """Initialize the global transfer engine info server."""
    global _transfer_engine_info_server
    if _transfer_engine_info_server is not None:
        _transfer_engine_info_server.stop()

    _transfer_engine_info_server = TransferEngineInfoServer(port, node_rank)
    return _transfer_engine_info_server


def init_transfer_engine_info_client() -> TransferEngineInfoClient:
    """Initialize the global transfer engine info client."""
    global _transfer_engine_info_client
    if _transfer_engine_info_client is None:
        _transfer_engine_info_client = TransferEngineInfoClient()
    return _transfer_engine_info_client


def cleanup_transfer_engine_info_comm():
    """Cleanup global communication instances."""
    global _transfer_engine_info_server, _transfer_engine_info_client

    if _transfer_engine_info_server:
        _transfer_engine_info_server.stop()
        _transfer_engine_info_server = None

    if _transfer_engine_info_client:
        _transfer_engine_info_client.close()
        _transfer_engine_info_client = None
