"""Print a free TCP port on 127.0.0.1 to stdout. Used by the runner shell
script to pick a server port without race-prone shell loops.
"""

import random
import socket
import sys


def find_free_port(low: int = 20000, high: int = 55000, attempts: int = 1000) -> int:
    for _ in range(attempts):
        port = random.randint(low, high)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"Could not find a free port in {low}..{high}")


if __name__ == "__main__":
    try:
        print(find_free_port())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
