"""Legacy port utilities.

DEPRECATED: This module will be removed during e2e_response_api migration.
Use infra.get_open_port() instead.
"""

import socket


def find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
