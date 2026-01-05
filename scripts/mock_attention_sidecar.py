#!/usr/bin/env python3
"""
Mock Attention Sidecar for testing the semantic routing loop.

This script simulates a sidecar that:
1. Receives fingerprints from the scheduler (ZMQ PULL on port 9001)
2. Sends feedback to the scheduler (ZMQ PUSH on port 9002)

Usage:
    python scripts/mock_attention_sidecar.py [--fingerprint-port 9001] [--feedback-port 9002]

Example with SGLang server:
    # Terminal 1: Start the sidecar
    python scripts/mock_attention_sidecar.py

    # Terminal 2: Start SGLang with sidecar URL
    python -m sglang.launch_server --model meta-llama/Llama-3.2-1B-Instruct \
        --attention-sidecar-url tcp://localhost:9001 \
        --return-attention-tokens \
        --attention-fingerprint-mode
"""

import argparse
import json
import time
import signal
import sys
from typing import Dict, Any, Optional

try:
    import zmq
except ImportError:
    print("ERROR: zmq not installed. Run: pip install pyzmq")
    sys.exit(1)


class MockAttentionSidecar:
    """Mock sidecar for testing attention steering."""

    def __init__(
        self,
        fingerprint_port: int = 9001,
        feedback_port: int = 9002,
        verbose: bool = True,
    ):
        self.fingerprint_port = fingerprint_port
        self.feedback_port = feedback_port
        self.verbose = verbose
        self.running = False

        # Statistics
        self.fingerprints_received = 0
        self.feedbacks_sent = 0

        # Request tracking
        self.request_history: Dict[str, list] = {}

        # Steering rules (can be customized)
        self.steering_rules = {
            # Example: After seeing 3 fingerprints, suggest probing layer 15
            "probe_after_n": 3,
            "probe_layers": [15, 16],
            # Example: Add attention bias to token 0 (BOS) on layer 10
            "bias_layer": 10,
            "bias_token": 0,
            "bias_value": 0.5,
        }

    def start(self):
        """Start the sidecar."""
        self.context = zmq.Context()

        # Fingerprint receiver (PULL from scheduler's PUSH)
        self.fingerprint_socket = self.context.socket(zmq.PULL)
        self.fingerprint_socket.bind(f"tcp://*:{self.fingerprint_port}")
        print(f"[Sidecar] Listening for fingerprints on port {self.fingerprint_port}")

        # Feedback sender (PUSH to scheduler's PULL)
        self.feedback_socket = self.context.socket(zmq.PUSH)
        self.feedback_socket.bind(f"tcp://*:{self.feedback_port}")
        print(f"[Sidecar] Feedback channel ready on port {self.feedback_port}")

        self.running = True
        print("[Sidecar] Ready. Press Ctrl+C to stop.\n")

        try:
            self._run_loop()
        except KeyboardInterrupt:
            print("\n[Sidecar] Shutting down...")
        finally:
            self.stop()

    def stop(self):
        """Stop the sidecar."""
        self.running = False
        self.fingerprint_socket.close()
        self.feedback_socket.close()
        self.context.term()
        print(f"[Sidecar] Stats: {self.fingerprints_received} fingerprints, {self.feedbacks_sent} feedbacks")

    def _run_loop(self):
        """Main processing loop."""
        poller = zmq.Poller()
        poller.register(self.fingerprint_socket, zmq.POLLIN)

        while self.running:
            try:
                socks = dict(poller.poll(timeout=100))  # 100ms timeout

                if self.fingerprint_socket in socks:
                    message = self.fingerprint_socket.recv()
                    self._handle_fingerprint(message)

            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break
                raise

    def _handle_fingerprint(self, message: bytes):
        """Process a received fingerprint and optionally send feedback."""
        try:
            data = json.loads(message.decode())
            request_id = data.get("request_id", "unknown")
            vector = data.get("vector", [])
            manifold = data.get("manifold", "unknown")

            self.fingerprints_received += 1

            # Track history per request
            if request_id not in self.request_history:
                self.request_history[request_id] = []
            self.request_history[request_id].append(data)

            if self.verbose:
                print(f"[Fingerprint] rid={request_id[:8]}... manifold={manifold} vector_len={len(vector)}")

            # Decide if we should send feedback
            feedback = self._compute_feedback(request_id, data)
            if feedback:
                self._send_feedback(feedback)

        except json.JSONDecodeError:
            print(f"[Sidecar] Invalid JSON received")
        except Exception as e:
            print(f"[Sidecar] Error processing fingerprint: {e}")

    def _compute_feedback(self, request_id: str, fingerprint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Compute feedback based on fingerprint analysis.

        This is where the "intelligence" of the sidecar lives.
        Override this method for custom steering logic.
        """
        history = self.request_history.get(request_id, [])
        step = len(history)

        # Example steering rules:

        # Rule 1: After N fingerprints, suggest probing different layers
        if step == self.steering_rules["probe_after_n"]:
            return {
                "request_id": request_id,
                "manifold_zone": "exploration",
                "manifold_confidence": 0.8,
                "next_capture_layers": self.steering_rules["probe_layers"],
            }

        # Rule 2: After more steps, add attention biases
        if step == self.steering_rules["probe_after_n"] + 2:
            return {
                "request_id": request_id,
                "manifold_zone": "steering",
                "manifold_confidence": 0.9,
                "suggested_biases": {  # Note: uses "suggested_biases" to match update_from_sidecar
                    str(self.steering_rules["bias_layer"]): {
                        str(self.steering_rules["bias_token"]): self.steering_rules["bias_value"]
                    }
                },
                "hub_tokens": [0, 1, 2],  # Mark first few tokens as hubs
            }

        return None

    def _send_feedback(self, feedback: Dict[str, Any]):
        """Send feedback to the scheduler."""
        try:
            message = json.dumps(feedback).encode()
            self.feedback_socket.send(message, flags=zmq.NOBLOCK)
            self.feedbacks_sent += 1

            if self.verbose:
                rid = feedback.get("request_id", "unknown")[:8]
                zone = feedback.get("manifold_zone", "?")
                print(f"[Feedback] -> rid={rid}... zone={zone}")

        except zmq.Again:
            print("[Sidecar] Feedback buffer full, dropping message")
        except Exception as e:
            print(f"[Sidecar] Error sending feedback: {e}")

    def inject_bias(self, request_id: str, layer_id: int, token_pos: int, bias: float):
        """Manually inject an attention bias for a specific request."""
        feedback = {
            "request_id": request_id,
            "attention_biases": {
                str(layer_id): {str(token_pos): bias}
            },
        }
        self._send_feedback(feedback)
        print(f"[Manual] Injected bias: layer={layer_id} token={token_pos} bias={bias}")


def main():
    parser = argparse.ArgumentParser(description="Mock Attention Sidecar")
    parser.add_argument("--fingerprint-port", type=int, default=9001,
                        help="Port to receive fingerprints (default: 9001)")
    parser.add_argument("--feedback-port", type=int, default=9002,
                        help="Port to send feedback (default: 9002)")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    args = parser.parse_args()

    sidecar = MockAttentionSidecar(
        fingerprint_port=args.fingerprint_port,
        feedback_port=args.feedback_port,
        verbose=not args.quiet,
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        sidecar.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    sidecar.start()


if __name__ == "__main__":
    main()
