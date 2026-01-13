#!/usr/bin/env python3
"""
Simple Terminal-based Attention Fingerprint Visualizer

No dependencies beyond pyzmq. Shows fingerprints in a formatted terminal display.

Usage:
    python scripts/attention_visualizer_simple.py
"""

import json
import sys
from collections import deque

try:
    import zmq
except ImportError:
    print("Please install: pip install pyzmq")
    sys.exit(1)


# ANSI colors
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


def create_bar(value: float, width: int = 20) -> str:
    """Create a simple ASCII bar chart."""
    filled = int(value * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    return bar


def format_vector(vector: list, width: int = 40) -> str:
    """Format vector as mini sparkline."""
    if not vector:
        return ""

    # Normalize to 0-1
    min_v = min(vector)
    max_v = max(vector)
    range_v = max_v - min_v if max_v > min_v else 1

    normalized = [(v - min_v) / range_v for v in vector]

    # Use block characters for sparkline
    blocks = " ▁▂▃▄▅▆▇█"
    sparkline = ""
    for v in normalized[:width]:
        idx = int(v * (len(blocks) - 1))
        sparkline += blocks[idx]

    return sparkline


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def main():
    print(f"{Colors.BOLD}🧠 Attention Fingerprint Visualizer{Colors.RESET}")
    print(f"{Colors.GRAY}Connecting to tcp://localhost:9001...{Colors.RESET}")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, 5000)

    try:
        socket.connect("tcp://localhost:9001")
    except Exception as e:
        print(f"{Colors.RED}Failed to connect: {e}{Colors.RESET}")
        sys.exit(1)

    print(f"{Colors.GREEN}Connected! Waiting for fingerprints...{Colors.RESET}")
    print(
        f"{Colors.GRAY}(Send a request with return_attention_tokens: true){Colors.RESET}\n"
    )

    # Stats
    total = 0
    requests = {}
    history = deque(maxlen=20)

    # Manifold colors
    manifold_colors = {
        "semantic_bridge": Colors.CYAN,
        "syntax_floor": Colors.YELLOW,
        "exploration": Colors.MAGENTA,
        "steering": Colors.GREEN,
    }

    try:
        while True:
            try:
                message = socket.recv()
                data = json.loads(message.decode())

                total += 1
                request_id = data.get("request_id", "unknown")[:8]
                vector = data.get("vector", [])
                manifold = data.get("manifold", "unknown")
                step = data.get("step", 0)
                think_phase = data.get("think_phase", "")

                # Track per-request
                if request_id not in requests:
                    requests[request_id] = 0
                requests[request_id] += 1

                # Add to history
                history.append(
                    {
                        "rid": request_id,
                        "step": step,
                        "manifold": manifold,
                        "vector": vector,
                        "phase": think_phase,
                    }
                )

                # Display
                color = manifold_colors.get(manifold, Colors.GRAY)
                sparkline = format_vector(vector)

                print(
                    f"{Colors.BOLD}[{total:04d}]{Colors.RESET} "
                    f"rid={Colors.BLUE}{request_id}{Colors.RESET} "
                    f"step={step:3d} "
                    f"zone={color}{manifold:16s}{Colors.RESET} "
                    f"phase={think_phase:8s} "
                    f"{Colors.GRAY}{sparkline}{Colors.RESET}"
                )

                # Every 10 fingerprints, show summary
                if total % 10 == 0:
                    print(
                        f"\n{Colors.GRAY}--- Stats: {total} total, {len(requests)} requests ---{Colors.RESET}\n"
                    )

            except zmq.Again:
                print(f"{Colors.GRAY}... waiting for fingerprints ...{Colors.RESET}")
                continue

    except KeyboardInterrupt:
        print(
            f"\n\n{Colors.YELLOW}Stopped. Received {total} fingerprints from {len(requests)} requests.{Colors.RESET}"
        )

    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
