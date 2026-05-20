"""
Mock ZMQ Receiver for Encoder Benchmark

A lightweight ZMQ PULL receiver that accepts embeddings from the encoder
and discards them. Used for testing zmq_to_tokenizer backend.

Usage:
    python mock_receiver.py --port 12345
"""

import argparse
import pickle
import signal
import time
from dataclasses import dataclass

import zmq


@dataclass
class ReceiverStats:
    """Statistics for the mock receiver."""

    messages_received: int = 0
    bytes_received: int = 0
    start_time: float = 0.0
    last_message_time: float = 0.0


def run_receiver(host: str, port: int, verbose: bool = False):
    """Run the mock ZMQ receiver."""
    context = zmq.Context()
    socket = context.socket(zmq.PULL)

    endpoint = f"tcp://{host}:{port}"
    socket.bind(endpoint)

    print(f"Mock Receiver started on {endpoint}")
    print("Waiting for embeddings... (Ctrl+C to stop)")
    print("-" * 50)

    stats = ReceiverStats(start_time=time.time())

    # Handle graceful shutdown
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\nShutting down...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while running:
            # Use poll to allow checking running flag
            if socket.poll(timeout=1000):  # 1 second timeout
                try:
                    # Receive multipart message (metadata + embedding)
                    parts = socket.recv_multipart(flags=zmq.NOBLOCK)
                    stats.messages_received += 1
                    stats.last_message_time = time.time()

                    total_bytes = sum(len(p) for p in parts)
                    stats.bytes_received += total_bytes

                    if verbose and len(parts) >= 1:
                        try:
                            metadata = pickle.loads(parts[0])
                            req_id = getattr(metadata, "req_id", "unknown")
                            print(
                                f"  [{stats.messages_received}] req_id={req_id}, "
                                f"parts={len(parts)}, bytes={total_bytes}"
                            )
                        except Exception:
                            print(
                                f"  [{stats.messages_received}] parts={len(parts)}, "
                                f"bytes={total_bytes}"
                            )
                    elif stats.messages_received % 10 == 0:
                        elapsed = time.time() - stats.start_time
                        rate = stats.messages_received / elapsed if elapsed > 0 else 0
                        throughput_mbps = (
                            stats.bytes_received / elapsed / 1024 / 1024
                            if elapsed > 0
                            else 0
                        )
                        print(
                            f"  Received: {stats.messages_received} messages "
                            f"({stats.bytes_received / 1024 / 1024:.2f} MB, "
                            f"{rate:.2f} msg/s, {throughput_mbps:.2f} MB/s)"
                        )

                except zmq.Again:
                    pass
                except Exception as e:
                    print(f"Error receiving message: {e}")

    finally:
        # Print final stats
        elapsed = time.time() - stats.start_time
        print("\n" + "=" * 50)
        print("Mock Receiver Statistics")
        print("=" * 50)
        print(f"Total messages: {stats.messages_received}")
        print(f"Total bytes: {stats.bytes_received / 1024 / 1024:.2f} MB")
        print(f"Run time: {elapsed:.2f}s")
        if elapsed > 0:
            print(f"Average rate: {stats.messages_received / elapsed:.2f} msg/s")
            print(
                f"Average throughput: {stats.bytes_received / elapsed / 1024 / 1024:.2f} MB/s"
            )

        socket.close()
        context.term()


def main():
    parser = argparse.ArgumentParser(
        description="Mock ZMQ Receiver for Encoder Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port to listen on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print details for each message",
    )

    args = parser.parse_args()

    run_receiver(
        host=args.host,
        port=args.port,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
