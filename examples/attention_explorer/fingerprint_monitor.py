#!/usr/bin/env python3
"""
Attention Fingerprint Streaming Monitor

Real-time monitoring of attention fingerprints streamed from SGLang server.
Provides multiple output modes: console dashboard, JSON export, and metrics.

Usage:
    # Start SGLang server with fingerprint streaming:
    python -m sglang.launch_server \
        --model Qwen/Qwen3-1.7B \
        --attention-fingerprint-mode \
        --attention-sidecar-url tcp://127.0.0.1:9000 \
        --port 30000

    # Start this monitor to receive fingerprints:
    python examples/attention_explorer/fingerprint_monitor.py \
        --bind tcp://*:9000 \
        --mode dashboard

    # Or export to JSON Lines for analysis:
    python examples/attention_explorer/fingerprint_monitor.py \
        --bind tcp://*:9000 \
        --mode jsonl \
        --output fingerprints.jsonl

Available modes:
    - dashboard: Real-time terminal dashboard with stats
    - jsonl: Export fingerprints to JSON Lines file
    - prometheus: Expose metrics on HTTP port for Prometheus scraping
    - quiet: Just receive and count, no output

Author: SGLang Team
"""

import argparse
import json
import os
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from threading import Event, Lock, Thread
from typing import Dict, List, Optional

try:
    import zmq
except ImportError:
    print("Please install pyzmq: pip install pyzmq")
    sys.exit(1)


@dataclass
class StreamStats:
    """Statistics for fingerprint stream."""

    total_received: int = 0
    total_bytes: int = 0
    start_time: float = field(default_factory=time.time)
    last_received: float = 0
    requests_seen: set = field(default_factory=set)
    manifold_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    zone_history: List[str] = field(default_factory=list)
    fingerprint_buffer: List[dict] = field(default_factory=list)
    buffer_max_size: int = 1000

    def record(self, payload: dict):
        """Record a received fingerprint."""
        self.total_received += 1
        self.total_bytes += len(json.dumps(payload))
        self.last_received = time.time()

        # Track unique requests
        if "request_id" in payload:
            self.requests_seen.add(payload["request_id"])

        # Track manifold distribution
        manifold = payload.get("manifold", "unknown")
        self.manifold_counts[manifold] += 1
        self.zone_history.append(manifold)

        # Circular buffer for recent fingerprints
        self.fingerprint_buffer.append(payload)
        if len(self.fingerprint_buffer) > self.buffer_max_size:
            self.fingerprint_buffer.pop(0)

    @property
    def uptime(self) -> float:
        return time.time() - self.start_time

    @property
    def rate(self) -> float:
        if self.uptime > 0:
            return self.total_received / self.uptime
        return 0

    @property
    def active_requests(self) -> int:
        return len(self.requests_seen)


class FingerprintMonitor:
    """Main monitor class for receiving and processing fingerprint stream."""

    def __init__(
        self,
        bind_url: str = "tcp://*:9000",
        mode: str = "dashboard",
        output_file: Optional[str] = None,
        prometheus_port: int = 9090,
    ):
        self.bind_url = bind_url
        self.mode = mode
        self.output_file = output_file
        self.prometheus_port = prometheus_port

        self.stats = StreamStats()
        self.lock = Lock()
        self.stop_event = Event()

        self._output_file_handle = None
        self._prometheus_thread = None

    def start(self):
        """Start the monitor."""
        # Setup ZMQ receiver
        context = zmq.Context()
        receiver = context.socket(zmq.PULL)
        receiver.bind(self.bind_url)
        receiver.setsockopt(zmq.RCVHWM, 10000)

        print(f"[Monitor] Listening on {self.bind_url}")
        print(f"[Monitor] Mode: {self.mode}")

        # Setup output
        if self.mode == "jsonl" and self.output_file:
            self._output_file_handle = open(self.output_file, "a")
            print(f"[Monitor] Writing to {self.output_file}")

        if self.mode == "prometheus":
            self._start_prometheus_server()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        # Start dashboard thread if needed
        if self.mode == "dashboard":
            dashboard_thread = Thread(target=self._dashboard_loop, daemon=True)
            dashboard_thread.start()

        # Main receive loop
        try:
            while not self.stop_event.is_set():
                try:
                    # Poll with timeout for clean shutdown
                    if receiver.poll(100):  # 100ms timeout
                        message = receiver.recv(flags=zmq.NOBLOCK)
                        self._process_message(message)
                except zmq.Again:
                    continue
                except zmq.ZMQError as e:
                    print(f"[Monitor] ZMQ error: {e}")
                    break
        finally:
            receiver.close()
            context.term()
            self._cleanup()

    def _process_message(self, message: bytes):
        """Process a received fingerprint message."""
        try:
            payload = json.loads(message.decode())
        except json.JSONDecodeError:
            return

        with self.lock:
            self.stats.record(payload)

        # Mode-specific handling
        if self.mode == "jsonl" and self._output_file_handle:
            # Add timestamp and write
            payload["_received_at"] = datetime.now().isoformat()
            self._output_file_handle.write(json.dumps(payload) + "\n")
            self._output_file_handle.flush()

    def _dashboard_loop(self):
        """Render dashboard in terminal."""
        while not self.stop_event.is_set():
            self._render_dashboard()
            time.sleep(0.5)

    def _render_dashboard(self):
        """Render the dashboard to terminal."""
        # Clear screen
        os.system("clear" if os.name != "nt" else "cls")

        with self.lock:
            stats = self.stats

            # Header
            print("=" * 60)
            print(" SGLang Attention Fingerprint Monitor")
            print("=" * 60)
            print()

            # Connection info
            print(f"  Listening: {self.bind_url}")
            print(f"  Uptime:    {stats.uptime:.1f}s")
            print()

            # Statistics
            print("-" * 60)
            print(" STREAM STATISTICS")
            print("-" * 60)
            print(f"  Total Received:    {stats.total_received:,}")
            print(
                f"  Total Bytes:       {stats.total_bytes:,} ({stats.total_bytes / 1024:.1f} KB)"
            )
            print(f"  Receive Rate:      {stats.rate:.1f} msg/sec")
            print(f"  Unique Requests:   {stats.active_requests}")

            if stats.last_received > 0:
                since_last = time.time() - stats.last_received
                print(f"  Last Received:     {since_last:.1f}s ago")
            else:
                print("  Last Received:     (waiting...)")

            print()

            # Manifold distribution
            print("-" * 60)
            print(" MANIFOLD DISTRIBUTION")
            print("-" * 60)

            total = sum(stats.manifold_counts.values()) or 1
            for zone in [
                "semantic_bridge",
                "syntax_floor",
                "structure_ripple",
                "long_range",
                "diffuse",
                "unknown",
            ]:
                count = stats.manifold_counts.get(zone, 0)
                pct = count / total * 100
                bar = "█" * int(pct / 2)
                print(f"  {zone:18s}: {count:6d} ({pct:5.1f}%) {bar}")

            print()

            # Recent activity (last 10 fingerprints)
            print("-" * 60)
            print(" RECENT ACTIVITY (last 10)")
            print("-" * 60)

            recent = stats.fingerprint_buffer[-10:] if stats.fingerprint_buffer else []
            for i, fp in enumerate(reversed(recent)):
                req_id = fp.get("request_id", "?")[:8]
                manifold = fp.get("manifold", "?")
                moe = "MOE" if fp.get("moe") else "   "
                print(f"  {i+1:2d}. [{req_id}] {manifold:18s} {moe}")

            print()
            print("-" * 60)
            print(" Press Ctrl+C to stop")
            print("-" * 60)

    def _start_prometheus_server(self):
        """Start HTTP server for Prometheus metrics."""
        try:
            from http.server import BaseHTTPRequestHandler, HTTPServer
        except ImportError:
            print("[Monitor] http.server not available for Prometheus mode")
            return

        monitor = self

        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    with monitor.lock:
                        stats = monitor.stats
                        metrics = [
                            f"# HELP sglang_fingerprints_total Total fingerprints received",
                            f"# TYPE sglang_fingerprints_total counter",
                            f"sglang_fingerprints_total {stats.total_received}",
                            f"",
                            f"# HELP sglang_fingerprints_bytes_total Total bytes received",
                            f"# TYPE sglang_fingerprints_bytes_total counter",
                            f"sglang_fingerprints_bytes_total {stats.total_bytes}",
                            f"",
                            f"# HELP sglang_fingerprint_rate Fingerprints per second",
                            f"# TYPE sglang_fingerprint_rate gauge",
                            f"sglang_fingerprint_rate {stats.rate:.2f}",
                            f"",
                            f"# HELP sglang_active_requests Unique request IDs seen",
                            f"# TYPE sglang_active_requests gauge",
                            f"sglang_active_requests {stats.active_requests}",
                            f"",
                        ]

                        # Manifold counts
                        metrics.append(
                            "# HELP sglang_manifold_count Count by manifold zone"
                        )
                        metrics.append("# TYPE sglang_manifold_count counter")
                        for zone, count in stats.manifold_counts.items():
                            metrics.append(
                                f'sglang_manifold_count{{zone="{zone}"}} {count}'
                            )

                    self.send_response(200)
                    self.send_header("Content-type", "text/plain")
                    self.end_headers()
                    self.wfile.write("\n".join(metrics).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress HTTP logs

        server = HTTPServer(("", self.prometheus_port), MetricsHandler)
        print(
            f"[Monitor] Prometheus metrics at http://localhost:{self.prometheus_port}/metrics"
        )

        self._prometheus_thread = Thread(target=server.serve_forever, daemon=True)
        self._prometheus_thread.start()

    def _handle_signal(self, signum, frame):
        """Handle shutdown signal."""
        print("\n[Monitor] Shutting down...")
        self.stop_event.set()

    def _cleanup(self):
        """Cleanup resources."""
        if self._output_file_handle:
            self._output_file_handle.close()

        # Print final stats
        print("\n" + "=" * 60)
        print(" FINAL STATISTICS")
        print("=" * 60)
        print(f"  Total Fingerprints: {self.stats.total_received:,}")
        print(f"  Total Bytes:        {self.stats.total_bytes:,}")
        print(f"  Unique Requests:    {self.stats.active_requests}")
        print(f"  Uptime:             {self.stats.uptime:.1f}s")

        if self.stats.manifold_counts:
            print("\n  Manifold Distribution:")
            total = sum(self.stats.manifold_counts.values())
            for zone, count in sorted(
                self.stats.manifold_counts.items(), key=lambda x: -x[1]
            ):
                pct = count / total * 100
                print(f"    {zone:18s}: {count:6d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="SGLang Attention Fingerprint Monitor")
    parser.add_argument(
        "--bind",
        default="tcp://*:9000",
        help="ZMQ bind URL (default: tcp://*:9000)",
    )
    parser.add_argument(
        "--mode",
        choices=["dashboard", "jsonl", "prometheus", "quiet"],
        default="dashboard",
        help="Output mode (default: dashboard)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for jsonl mode",
    )
    parser.add_argument(
        "--prometheus-port",
        type=int,
        default=9090,
        help="Port for Prometheus metrics (default: 9090)",
    )
    args = parser.parse_args()

    # Validate args
    if args.mode == "jsonl" and not args.output:
        parser.error("--output is required for jsonl mode")

    monitor = FingerprintMonitor(
        bind_url=args.bind,
        mode=args.mode,
        output_file=args.output,
        prometheus_port=args.prometheus_port,
    )
    monitor.start()


if __name__ == "__main__":
    main()
