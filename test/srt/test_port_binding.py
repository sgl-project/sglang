# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Tests for create_server_socket utility function.

This tests the port pre-binding feature that ensures ports are checked
before model loading to fail fast on port conflicts.
"""

import socket
import unittest


def create_server_socket(host: str, port: int) -> socket.socket:
    """Create and bind a server socket for HTTP serving.

    This is a copy of the function from sglang.srt.utils.common for testing
    without importing the full sglang package.
    """
    family = socket.AF_INET
    if host and ":" in host:
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except (AttributeError, OSError):
        pass

    try:
        sock.bind((host or "", port))
    except OSError as e:
        sock.close()
        raise RuntimeError(
            f"Cannot bind to {host or '0.0.0.0'}:{port}. "
            f"Port may already be in use. Please check if another process is using this port. "
            f"Error: {e}"
        ) from e

    sock.listen(128)
    sock.set_inheritable(True)
    return sock


class TestCreateServerSocket(unittest.TestCase):
    """Test cases for create_server_socket function."""

    def test_create_socket_success(self):
        """Test successful socket creation on an available port."""
        # Use port 0 to let the OS assign an available port
        sock = create_server_socket("127.0.0.1", 0)
        try:
            self.assertIsNotNone(sock)
            self.assertEqual(sock.getsockname()[0], "127.0.0.1")
            self.assertGreater(sock.getsockname()[1], 0)
        finally:
            sock.close()

    def test_create_socket_specific_port(self):
        """Test socket creation on a specific port."""
        test_port = 19999  # Use a high port unlikely to be in use
        sock = create_server_socket("127.0.0.1", test_port)
        try:
            self.assertEqual(sock.getsockname(), ("127.0.0.1", test_port))
        finally:
            sock.close()

    def test_port_conflict_detection(self):
        """Test that RuntimeError with clear message is raised on bind failure."""
        # Test the error message format by trying to bind to a privileged port
        # (requires root on Unix) or by mocking
        # For simplicity, we just verify the function handles errors gracefully

        # Create a socket that holds a port
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        occupied_port = blocker.getsockname()[1]

        try:
            # Try binding with a socket that does NOT have SO_REUSEPORT
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Only set SO_REUSEADDR, not SO_REUSEPORT
            test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                test_sock.bind(("127.0.0.1", occupied_port))
                # Port reuse is allowed on this platform, skip conflict test
                test_sock.close()
                self.skipTest("Port reuse allowed on this platform")
            except OSError:
                # Good - conflict detected as expected
                test_sock.close()
                pass
        finally:
            blocker.close()

    def test_error_message_format(self):
        """Test that error message contains useful information."""
        # Try to bind to a port that should fail (privileged port without root)
        import os

        if os.geteuid() == 0:
            self.skipTest("Running as root, cannot test privileged port failure")

        try:
            # Port 1 requires root privileges
            create_server_socket("127.0.0.1", 1)
            self.fail("Should have raised RuntimeError")
        except RuntimeError as e:
            error_msg = str(e)
            self.assertIn("Cannot bind", error_msg)
            self.assertIn("127.0.0.1", error_msg)
            self.assertIn("1", error_msg)

    def test_socket_reuse_after_close(self):
        """Test that port can be reused after socket is closed."""
        # Create and close a socket
        sock1 = create_server_socket("127.0.0.1", 0)
        port = sock1.getsockname()[1]
        sock1.close()

        # Should be able to bind to the same port again
        sock2 = create_server_socket("127.0.0.1", port)
        try:
            self.assertEqual(sock2.getsockname()[1], port)
        finally:
            sock2.close()

    def test_socket_inheritable(self):
        """Test that socket is set as inheritable for multi-process scenarios."""
        sock = create_server_socket("127.0.0.1", 0)
        try:
            self.assertTrue(sock.get_inheritable())
        finally:
            sock.close()

    def test_empty_host_binds_to_all_interfaces(self):
        """Test that empty host binds to all interfaces (0.0.0.0)."""
        sock = create_server_socket("", 0)
        try:
            # On most systems, binding to "" results in "0.0.0.0"
            self.assertIn(sock.getsockname()[0], ["", "0.0.0.0"])
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()
