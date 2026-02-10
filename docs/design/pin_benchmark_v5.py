#!/usr/bin/env python3
"""PIN V5 Benchmark: depth-sweep showing PIN value scales with conversation length.

For each warmup depth D (number of turns cached before flood):
  Baseline: warmup D turns -> flood -> measure turn D+1 TTFT (full recompute)
  Pinned:   warmup D turns -> PIN -> flood -> measure turn D+1 TTFT (cache hit)

The gap between baseline and pinned TTFT grows with depth, proving PIN's value
scales with how much conversation context is preserved.

Usage:
    .venv/bin/python docs/design/pin_benchmark_v5.py
    .venv/bin/python docs/design/pin_benchmark_v5.py --depths 0 2 6 10 16
"""

import argparse
import json
import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
import zmq
from msgspec.msgpack import Decoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FLASHINFER_WORKSPACE = "/tmp/flashinfer_workspace"
AIPERF_DIR = Path.home() / "aiperf"


# ---------------------------------------------------------------------------
# KV Event Collector (reused from v4)
# ---------------------------------------------------------------------------

class KVEventCollector:
    def __init__(self, endpoint="tcp://localhost:5557", topic="kv-events"):
        self.endpoint = endpoint
        self.topic = topic
        self.block_hashes = []
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        from sglang.srt.disaggregation.kv_events import KVEventBatch, BlockStored
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.connect(self.endpoint)
        sub.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        sub.setsockopt(zmq.RCVTIMEO, 500)
        decoder = Decoder(type=KVEventBatch)
        while self._running:
            try:
                _, seq_bytes, payload = sub.recv_multipart()
                batch = decoder.decode(payload)
                for event in batch.events:
                    if isinstance(event, BlockStored):
                        for bh in event.block_hashes:
                            self.block_hashes.append(bh)
            except zmq.Again:
                continue
            except Exception as e:
                if self._running:
                    logger.warning(f"KV event error: {e}")
        sub.close()
        ctx.term()

    def get_unique_hashes(self):
        seen = set()
        result = []
        for bh in self.block_hashes:
            if bh not in seen:
                seen.add(bh)
                result.append(bh)
        return result


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def start_server(model, port, mem_fraction, tp_size=1, context_length=32768,
                 page_size=64, gpu_id=0, zmq_port=5557):
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--mem-fraction-static", str(mem_fraction),
        "--kv-events-config", json.dumps({
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": f"tcp://*:{zmq_port}",
        }),
        "--tp-size", str(tp_size),
        "--trust-remote-code",
        "--log-level", "info",
        "--watchdog-timeout", "10000",
        "--enable-metrics",
        "--enable-cache-report",
        "--json-model-override-args", json.dumps({
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
            "max_position_embeddings": 131072,
        }),
        "--context-length", str(context_length),
        "--page-size", str(page_size),
    ]
    logger.info(f"Starting server: port={port}, gpu={gpu_id}, zmq={zmq_port}")
    log_file = Path(f"/tmp/sglang_server_{port}.log")
    log_fh = open(log_file, "w")

    env = os.environ.copy()
    ws = f"/tmp/flashinfer_workspace_{gpu_id}"
    os.makedirs(ws, exist_ok=True)
    env["FLASHINFER_WORKSPACE_BASE"] = ws
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    proc._log_fh = log_fh
    return proc


def stop_server(proc):
    """Stop a specific server process group. Does NOT use global pkill."""
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:
            pass
    if hasattr(proc, '_log_fh'):
        proc._log_fh.close()
    time.sleep(3)
    logger.info("Server stopped")


def cleanup_all_servers():
    """Kill ALL sglang processes. Use only at start/end of benchmark."""
    for pattern in ["sglang.launch_server", "sglang.srt", "multiprocessing.spawn"]:
        try:
            subprocess.run(["pkill", "-9", "-f", pattern],
                           capture_output=True, timeout=5)
        except Exception:
            pass
    time.sleep(3)


def wait_for_server(base_url, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                logger.info("Server healthy")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server not ready in {timeout}s")


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def load_vip_turns(dataset_path: str) -> list[dict]:
    """Load all turns from VIP dataset, normalized to messages format."""
    with open(dataset_path) as f:
        data = json.loads(f.readline())
    turns = data["turns"]
    messages = []
    for t in turns:
        messages.append({"role": t["role"], "content": t["text"]})
    return messages


def send_warmup(base_url: str, model: str, messages: list[dict]) -> dict:
    """Send conversation prefix to populate cache. Returns usage info."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
    return r.json().get("usage", {})


def measure_ttft(base_url: str, model: str, messages: list[dict]) -> dict:
    """Send a streaming request and measure time to first token.
    Uses stream_options to get usage stats (including cached_tokens) from the
    same request, avoiding a second request that would contaminate the cache.
    Returns dict with ttft_ms, prompt_tokens, cached_tokens."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 16,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start = time.perf_counter()
    first_token_time = None
    usage = {}

    r = requests.post(f"{base_url}/v1/chat/completions", json=payload,
                      stream=True, timeout=120)
    for line in r.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8", errors="replace")
        if line_str.startswith("data: ") and line_str != "data: [DONE]":
            if first_token_time is None:
                first_token_time = time.perf_counter()
            try:
                chunk = json.loads(line_str[6:])
                if "usage" in chunk and chunk["usage"]:
                    usage = chunk["usage"]
            except (json.JSONDecodeError, KeyError):
                pass

    ttft_ms = (first_token_time - start) * 1000 if first_token_time else None

    prompt_tokens = usage.get("prompt_tokens", 0)
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = details.get("cached_tokens", usage.get("cached_tokens", 0))

    return {
        "ttft_ms": ttft_ms,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
    }


# ---------------------------------------------------------------------------
# PIN / UNPIN
# ---------------------------------------------------------------------------

def pin_blocks(base_url, block_hashes):
    r = requests.post(f"{base_url}/hicache/pin_blocks",
                      json={"block_hashes": block_hashes}, timeout=30)
    return r.json()


def unpin_blocks(base_url, block_hashes):
    r = requests.post(f"{base_url}/hicache/unpin_blocks",
                      json={"block_hashes": block_hashes}, timeout=30)
    return r.json()


# ---------------------------------------------------------------------------
# Flood (via aiperf)
# ---------------------------------------------------------------------------

def run_flood(model, aiperf_model, base_url, flood_dataset, concurrency,
              num_requests, output_dir, timeout=1800):
    """Run flood traffic via aiperf."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "aiperf", "profile",
        "--model", aiperf_model,
        "--url", base_url,
        "--endpoint-type", "chat",
        "--input-file", flood_dataset,
        "--custom-dataset-type", "multi-turn",
        "--use-server-token-count",
        "--use-legacy-max-tokens",
        "--concurrency", str(concurrency),
        "--request-count", str(num_requests),
        "--osl", "64",
        "--streaming",
        "--ui-type", "none",
        "--no-gpu-telemetry",
        "--no-server-metrics",
        "--output-artifact-dir", str(out_path),
        "--profile-export-prefix", "flood",
    ]

    logger.info(f"  Flood: {num_requests} requests, concurrency={concurrency}")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(AIPERF_DIR), timeout=timeout)
    if result.returncode != 0:
        logger.warning(f"  Flood aiperf exit code: {result.returncode}")
    return result.returncode


def get_cache_capacity(base_url) -> int:
    """Query server for max_total_num_tokens (KV cache capacity)."""
    try:
        r = requests.get(f"{base_url}/server_info", timeout=10)
        info = r.json()
        return info.get("max_total_num_tokens", 0)
    except Exception:
        return 0


# Average unique (non-shared) tokens each flood request adds to the radix tree.
# Multi-turn requests share long prefixes, so each request contributes far fewer
# unique tokens than its total prompt length.  Measured empirically on
# long_multiturn_opus.jsonl against a 42K-token cache.
FLOOD_UNIQUE_TOKENS_PER_REQ = 150
# We want the flood to cycle through the cache this many times to ensure
# full LRU eviction of unpinned blocks.
EVICTION_CYCLES = 3


def calc_flood_requests(cache_capacity: int, min_requests: int = 100) -> int:
    """Calculate flood requests needed to cycle through the full KV cache."""
    needed = math.ceil(cache_capacity * EVICTION_CYCLES / FLOOD_UNIQUE_TOKENS_PER_REQ)
    result = max(needed, min_requests)
    logger.info(f"    Cache capacity: {cache_capacity} tokens, "
                f"flood requests: {result} "
                f"(need {needed} to cycle {EVICTION_CYCLES}x)")
    return result


def drain_server(base_url):
    """Wait for in-flight requests to complete."""
    for _ in range(30):
        try:
            r = requests.get(f"{base_url}/get_server_info", timeout=5)
            info = r.json()
            if info.get("num_running_req", 0) == 0 and info.get("num_waiting_req", 0) == 0:
                break
        except Exception:
            pass
        time.sleep(2)
    time.sleep(2)


# ---------------------------------------------------------------------------
# Run one depth
# ---------------------------------------------------------------------------

def run_depth(depth: int, pin: bool, all_messages: list[dict],
              model: str, aiperf_model: str, port: int, mem_fraction: float,
              tp_size: int, flood_dataset: str, flood_concurrency: int,
              flood_requests: int, output_dir: str, context_length: int,
              gpu_id: int = 0, zmq_port: int = 5557) -> dict:
    """Run one (depth, phase) combination. Returns result dict."""
    label = f"{'pinned' if pin else 'baseline'}_d{depth}"
    base_url = f"http://localhost:{port}"

    # Build message arrays
    # Warmup: all turns up to depth (must end on assistant turn for clean prefix)
    warmup_messages = all_messages[:depth + 1]
    # Measurement: warmup + next user turn
    measure_messages = all_messages[:depth + 2] if depth + 2 <= len(all_messages) else all_messages

    result = {
        "depth": depth,
        "pin": pin,
        "warmup_turns": len(warmup_messages),
        "measure_turns": len(measure_messages),
    }

    proc = start_server(model, port, mem_fraction, tp_size, context_length,
                        gpu_id=gpu_id, zmq_port=zmq_port)
    try:
        wait_for_server(base_url)

        # Step 1: Warmup -- send conversation prefix to populate cache
        # For pinned runs, start KV event collector BEFORE warmup so we
        # capture BlockStored events from the initial cache population.
        collector = None
        if pin:
            collector = KVEventCollector(endpoint=f"tcp://localhost:{zmq_port}")
            collector.start()
            time.sleep(1)  # Let ZMQ subscriber connect

        logger.info(f"  [{label}] Warmup: sending {len(warmup_messages)} turns")
        warmup_usage = send_warmup(base_url, aiperf_model, warmup_messages)
        warmup_tokens = warmup_usage.get("prompt_tokens", 0)
        logger.info(f"    Warmup: {warmup_tokens} prompt tokens")
        result["warmup_tokens"] = warmup_tokens

        # Step 2: PIN (if enabled)
        pinned_hashes = []
        if pin and collector:
            time.sleep(1)  # Let events arrive
            pinned_hashes = collector.get_unique_hashes()
            collector.stop()

            if pinned_hashes:
                pin_result = pin_blocks(base_url, pinned_hashes)
                blocks_pinned = pin_result.get("pinned_count", 0)
                logger.info(f"    Pinned {blocks_pinned}/{len(pinned_hashes)} blocks")
                result["blocks_pinned"] = blocks_pinned
            else:
                logger.warning("    No block hashes collected!")
                result["blocks_pinned"] = 0

        # Step 3: Flood -- auto-calculate request count from cache capacity
        cache_capacity = get_cache_capacity(base_url)
        if cache_capacity > 0:
            actual_flood_requests = calc_flood_requests(cache_capacity,
                                                        min_requests=flood_requests)
        else:
            actual_flood_requests = flood_requests
        result["cache_capacity"] = cache_capacity
        result["flood_requests"] = actual_flood_requests

        logger.info(f"  [{label}] Flood: evicting unpinned blocks")
        flood_dir = f"{output_dir}/{label}_flood"
        run_flood(model, aiperf_model, base_url, flood_dataset,
                  flood_concurrency, actual_flood_requests, flood_dir)
        drain_server(base_url)

        # Step 4: Measure TTFT
        logger.info(f"  [{label}] Measure: sending {len(measure_messages)} turns")
        measurement = measure_ttft(base_url, aiperf_model, measure_messages)
        result["ttft_ms"] = measurement["ttft_ms"]
        result["prompt_tokens"] = measurement["prompt_tokens"]
        result["cached_tokens"] = measurement["cached_tokens"]
        cache_pct = (measurement["cached_tokens"] / measurement["prompt_tokens"] * 100
                     if measurement["prompt_tokens"] else 0)
        result["cache_hit_pct"] = cache_pct

        logger.info(f"    TTFT={measurement['ttft_ms']:.1f}ms, "
                    f"cached={measurement['cached_tokens']}/{measurement['prompt_tokens']} "
                    f"({cache_pct:.0f}%)")

        # Unpin
        if pinned_hashes:
            try:
                unpin_blocks(base_url, pinned_hashes)
            except Exception:
                pass

    finally:
        stop_server(proc)
        import shutil
        log_src = Path(f"/tmp/sglang_server_{port}.log")
        log_dst = Path(output_dir) / f"{label}_server.log"
        log_dst.parent.mkdir(parents=True, exist_ok=True)
        if log_src.exists():
            shutil.copy2(log_src, log_dst)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PIN V5: Depth Sweep Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3-14B-FP8")
    parser.add_argument("--aiperf-model", default="qwen3-14b-fp8")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction", type=float, default=0.65)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 2, 6, 10, 16],
                        help="Turn indices to test (must be even, ending on assistant turn)")
    parser.add_argument("--flood-concurrency", type=int, default=8)
    parser.add_argument("--flood-requests", type=int, default=300)
    parser.add_argument("--vip-dataset",
                        default=str(Path.home() / "datasets/claude_history_sonnet.jsonl"))
    parser.add_argument("--flood-dataset",
                        default=str(Path.home() / "datasets/long_multiturn_opus.jsonl"))
    parser.add_argument("--output-dir", default="/tmp/pin_benchmark_v5")
    parser.add_argument("--phase", choices=["baseline", "pinned", "both"], default="both",
                        help="Run only baseline, only pinned, or both")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id (CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--zmq-port", type=int, default=5557)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save params
    with open(output_dir / "params.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load VIP conversation
    all_messages = load_vip_turns(args.vip_dataset)
    logger.info(f"VIP conversation: {len(all_messages)} turns")

    logger.info(f"PIN V5 Depth Sweep Benchmark")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Depths: {args.depths}")
    logger.info(f"  Flood: {args.flood_requests} requests, concurrency={args.flood_concurrency}")

    run_baseline = args.phase in ("baseline", "both")
    run_pinned = args.phase in ("pinned", "both")

    results = []

    for depth in args.depths:
        if depth + 2 > len(all_messages):
            logger.warning(f"Depth {depth} exceeds conversation length, skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"DEPTH {depth} (turns 0..{depth}, measure turn {depth+1})")
        logger.info(f"{'='*60}")

        depth_dir = str(output_dir / f"depth_{depth}")
        common_kwargs = dict(
            depth=depth, all_messages=all_messages,
            model=args.model, aiperf_model=args.aiperf_model,
            port=args.port, mem_fraction=args.mem_fraction, tp_size=args.tp_size,
            flood_dataset=args.flood_dataset,
            flood_concurrency=args.flood_concurrency,
            flood_requests=args.flood_requests,
            output_dir=depth_dir, context_length=args.context_length,
            gpu_id=args.gpu, zmq_port=args.zmq_port,
        )

        depth_result = {"depth": depth}

        if run_baseline:
            logger.info(f"\n--- Baseline (depth={depth}) ---")
            baseline = run_depth(pin=False, **common_kwargs)
            depth_result["baseline"] = baseline
            logger.info(f"     Baseline: TTFT={baseline['ttft_ms']:.1f}ms, "
                        f"cached={baseline.get('cached_tokens', 0)}/{baseline.get('prompt_tokens', 0)}")

        if run_pinned:
            logger.info(f"\n--- Pinned (depth={depth}) ---")
            pinned = run_depth(pin=True, **common_kwargs)
            depth_result["pinned"] = pinned
            logger.info(f"     Pinned:   TTFT={pinned['ttft_ms']:.1f}ms, "
                        f"cached={pinned.get('cached_tokens', 0)}/{pinned.get('prompt_tokens', 0)}")

        if "baseline" in depth_result and "pinned" in depth_result:
            speedup = (depth_result["baseline"]["ttft_ms"] / depth_result["pinned"]["ttft_ms"]
                       if depth_result["pinned"]["ttft_ms"] > 0 else 0)
            depth_result["speedup"] = speedup
            logger.info(f"     Speedup:  {speedup:.1f}x")

        results.append(depth_result)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"V5 DEPTH SWEEP SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Depth':>6} {'BL TTFT':>10} {'PIN TTFT':>10} {'Speedup':>10} "
                f"{'BL cached':>12} {'PIN cached':>12}")
    logger.info(f"{'-'*60}")
    for r in results:
        bl = r.get("baseline", {})
        pin = r.get("pinned", {})
        bl_ttft = f"{bl['ttft_ms']:>9.1f}ms" if bl.get("ttft_ms") else "       N/A"
        pin_ttft = f"{pin['ttft_ms']:>9.1f}ms" if pin.get("ttft_ms") else "       N/A"
        speedup = f"{r['speedup']:>9.1f}x" if "speedup" in r else "       N/A"
        bl_cached = f"{bl.get('cached_tokens', 0):>5}/{bl.get('prompt_tokens', 0):<5}" if bl else "          N/A"
        pin_cached = f"{pin.get('cached_tokens', 0):>5}/{pin.get('prompt_tokens', 0):<5}" if pin else "          N/A"
        logger.info(f"{r['depth']:>6} {bl_ttft} {pin_ttft} {speedup} {bl_cached} {pin_cached}")

    # Only cleanup all servers when running both phases (not in parallel mode)
    if args.phase == "both":
        cleanup_all_servers()

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({"results": results, "params": vars(args)}, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
