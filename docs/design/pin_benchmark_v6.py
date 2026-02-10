#!/usr/bin/env python3
"""PIN V6 Benchmark: Tier-aware PIN with HiRadixCache.

Proves that pinned blocks survive GPU eviction pressure when HiCache is enabled.
Pinned blocks flow GPU -> CPU (host) during flood, then get restored via load_back
when the measurement request arrives.

Three configurations:
  baseline:  No PIN, no HiCache  -> everything evicted, full recompute
  gpu-pin:   PIN with lock_ref   -> blocks frozen on GPU, direct cache hit
  tier-pin:  PIN with pin_count  -> blocks demoted to CPU, load_back + cache hit
             + HiCache enabled

Usage:
    # Quick single-depth proof (tier-pin only)
    .venv/bin/python docs/design/pin_benchmark_v6.py --phase tier-pin --depths 0

    # Full three-way comparison
    .venv/bin/python docs/design/pin_benchmark_v6.py --phase all --depths 0 6 16

    # Parallel on two GPUs
    .venv/bin/python docs/design/pin_benchmark_v6.py \
        --phase baseline --gpu 0 --port 30000 --zmq-port 5557 &
    .venv/bin/python docs/design/pin_benchmark_v6.py \
        --phase tier-pin --gpu 1 --port 30001 --zmq-port 5558 &
    wait
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
# KV Event Collector
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
                 page_size=64, gpu_id=0, zmq_port=5557,
                 hicache=False, hicache_write_policy="write_through",
                 hicache_ratio=2.0):
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

    if hicache:
        cmd.extend([
            "--enable-hierarchical-cache",
            "--hicache-write-policy", hicache_write_policy,
            "--hicache-ratio", str(hicache_ratio),
        ])

    logger.info(f"Starting server: port={port}, gpu={gpu_id}, zmq={zmq_port}, "
                f"hicache={hicache}")
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
    with open(dataset_path) as f:
        data = json.loads(f.readline())
    turns = data["turns"]
    messages = []
    for t in turns:
        messages.append({"role": t["role"], "content": t["text"]})
    return messages


def send_warmup(base_url: str, model: str, messages: list[dict]) -> dict:
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
    """Measure TTFT via streaming, then get tier breakdown via non-streaming sglext."""
    # Streaming request for TTFT measurement
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


def get_cache_metrics(base_url: str) -> dict:
    """Scrape /metrics for device vs host cached token counters."""
    try:
        r = requests.get(f"{base_url}/metrics", timeout=10)
        m = {"device": 0, "host": 0}
        for line in r.text.split("\n"):
            if "cached_tokens_total" in line and not line.startswith("#"):
                val = float(line.split()[-1])
                if 'cache_source="device"' in line:
                    m["device"] = val
                elif 'cache_source="host"' in line:
                    m["host"] = val
        return m
    except Exception as e:
        logger.warning(f"Failed to get metrics: {e}")
        return {"device": 0, "host": 0}


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
# Flood
# ---------------------------------------------------------------------------

def run_flood(model, aiperf_model, base_url, flood_dataset, concurrency,
              num_requests, output_dir, timeout=1800):
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
    try:
        r = requests.get(f"{base_url}/server_info", timeout=10)
        info = r.json()
        return info.get("max_total_num_tokens", 0)
    except Exception:
        return 0


FLOOD_UNIQUE_TOKENS_PER_REQ = 150
EVICTION_CYCLES = 3


def calc_flood_requests(cache_capacity: int, min_requests: int = 100,
                        hicache: bool = False, hicache_ratio: float = 0.0) -> int:
    # With HiCache + write_through, every GPU entry also gets backed up to CPU.
    # To stress-test pin_count we need to fill AND cycle the CPU tier too.
    effective_capacity = cache_capacity
    if hicache and hicache_ratio > 0:
        effective_capacity = int(cache_capacity * (1 + hicache_ratio))
    needed = math.ceil(effective_capacity * EVICTION_CYCLES / FLOOD_UNIQUE_TOKENS_PER_REQ)
    result = max(needed, min_requests)
    logger.info(f"    Cache capacity: {cache_capacity} tokens "
                f"(effective={effective_capacity} with hicache={hicache}, ratio={hicache_ratio}), "
                f"flood requests: {result} "
                f"(need {needed} to cycle {EVICTION_CYCLES}x)")
    return result


def drain_server(base_url):
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
# Phase configurations
# ---------------------------------------------------------------------------

PHASE_CONFIGS = {
    "baseline": {
        "pin": False,
        "hicache": True,
        "description": "HiCache on, no PIN (VIP evictable from GPU+CPU)",
    },
    "gpu-pin": {
        "pin": True,
        "hicache": False,
        "description": "PIN with lock_ref (blocks frozen on GPU)",
    },
    "tier-pin": {
        "pin": True,
        "hicache": True,
        "description": "PIN with pin_count + HiCache (blocks can flow to CPU)",
    },
}


# ---------------------------------------------------------------------------
# Run one depth
# ---------------------------------------------------------------------------

def run_depth(depth: int, phase_name: str, all_messages: list[dict],
              model: str, aiperf_model: str, port: int, mem_fraction: float,
              tp_size: int, flood_dataset: str, flood_concurrency: int,
              flood_requests: int, output_dir: str, context_length: int,
              gpu_id: int = 0, zmq_port: int = 5557,
              hicache_write_policy: str = "write_through",
              hicache_ratio: float = 2.0,
              use_flush: bool = False) -> dict:
    """Run one (depth, phase) combination. Returns result dict."""
    config = PHASE_CONFIGS[phase_name]
    label = f"{phase_name}_d{depth}"
    base_url = f"http://localhost:{port}"

    warmup_messages = all_messages[:depth + 1]
    measure_messages = (all_messages[:depth + 2]
                        if depth + 2 <= len(all_messages) else all_messages)

    result = {
        "depth": depth,
        "phase": phase_name,
        "pin": config["pin"],
        "hicache": config["hicache"],
        "warmup_turns": len(warmup_messages),
        "measure_turns": len(measure_messages),
    }

    proc = start_server(model, port, mem_fraction, tp_size, context_length,
                        gpu_id=gpu_id, zmq_port=zmq_port,
                        hicache=config["hicache"],
                        hicache_write_policy=hicache_write_policy,
                        hicache_ratio=hicache_ratio)
    try:
        wait_for_server(base_url)

        # Collect metrics baseline
        metrics_before = get_cache_metrics(base_url)

        # Step 1: Warmup
        collector = None
        if config["pin"]:
            collector = KVEventCollector(endpoint=f"tcp://localhost:{zmq_port}")
            collector.start()
            time.sleep(1)

        logger.info(f"  [{label}] Warmup: sending {len(warmup_messages)} turns")
        warmup_usage = send_warmup(base_url, aiperf_model, warmup_messages)
        warmup_tokens = warmup_usage.get("prompt_tokens", 0)
        logger.info(f"    Warmup: {warmup_tokens} prompt tokens")
        result["warmup_tokens"] = warmup_tokens

        # Step 2: PIN
        pinned_hashes = []
        if config["pin"] and collector:
            time.sleep(1)
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

        # Step 3: Evict unpinned blocks (flush or flood)
        cache_capacity = get_cache_capacity(base_url)
        result["cache_capacity"] = cache_capacity

        if use_flush:
            logger.info(f"  [{label}] Flush: evicting unpinned blocks via /flush_cache")
            drain_server(base_url)
            # Retry flush until server accepts it (no pending requests).
            # HiCache write-through creates background operations that keep
            # running-req > 0 for a while after the API call returns.
            for attempt in range(60):
                r = requests.post(f"{base_url}/flush_cache", timeout=30)
                if r.status_code == 200:
                    break
                if attempt % 10 == 0:
                    logger.info(f"    flush attempt {attempt+1}: status={r.status_code}, "
                                f"waiting for write-through to settle...")
                time.sleep(1)
            logger.info(f"    flush_cache: status={r.status_code}, response={r.text.strip()}")
            result["eviction_method"] = "flush"
            result["flood_requests"] = 0
        else:
            if cache_capacity > 0:
                actual_flood_requests = calc_flood_requests(
                    cache_capacity, min_requests=flood_requests,
                    hicache=config["hicache"], hicache_ratio=hicache_ratio)
            else:
                actual_flood_requests = flood_requests
            result["flood_requests"] = actual_flood_requests
            result["eviction_method"] = "flood"

            logger.info(f"  [{label}] Flood: evicting unpinned blocks")
            flood_dir = f"{output_dir}/{label}_flood"
            run_flood(model, aiperf_model, base_url, flood_dataset,
                      flood_concurrency, actual_flood_requests, flood_dir)
            drain_server(base_url)

        # Collect metrics after flood (before measurement)
        metrics_after_flood = get_cache_metrics(base_url)
        result["metrics_after_flood"] = metrics_after_flood

        # Step 4: Measure TTFT
        logger.info(f"  [{label}] Measure: sending {len(measure_messages)} turns")
        measurement = measure_ttft(base_url, aiperf_model, measure_messages)
        result["ttft_ms"] = measurement["ttft_ms"]
        result["prompt_tokens"] = measurement["prompt_tokens"]
        result["cached_tokens"] = measurement["cached_tokens"]
        cache_pct = (measurement["cached_tokens"] / measurement["prompt_tokens"] * 100
                     if measurement["prompt_tokens"] else 0)
        result["cache_hit_pct"] = cache_pct

        # Collect metrics after measurement
        metrics_after_measure = get_cache_metrics(base_url)
        result["metrics_after_measure"] = metrics_after_measure

        # Compute host cache hit delta (tokens loaded from host during measurement)
        host_hits_during_measure = (metrics_after_measure["host"]
                                    - metrics_after_flood["host"])
        result["host_cache_hits"] = host_hits_during_measure

        logger.info(f"    TTFT={measurement['ttft_ms']:.1f}ms, "
                    f"cached={measurement['cached_tokens']}/{measurement['prompt_tokens']} "
                    f"({cache_pct:.0f}%), "
                    f"host_hits={host_hits_during_measure:.0f}")

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
    parser = argparse.ArgumentParser(description="PIN V6: Tier-Aware PIN Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen3-14B-FP8")
    parser.add_argument("--aiperf-model", default="qwen3-14b-fp8")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction", type=float, default=0.50)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 2, 6, 10, 16])
    parser.add_argument("--flood-concurrency", type=int, default=8)
    parser.add_argument("--flood-requests", type=int, default=300)
    parser.add_argument("--vip-dataset",
                        default=str(Path.home() / "datasets/claude_history_sonnet.jsonl"))
    parser.add_argument("--flood-dataset",
                        default=str(Path.home() / "datasets/long_multiturn_opus.jsonl"))
    parser.add_argument("--output-dir", default="/tmp/pin_benchmark_v6")
    parser.add_argument("--phase",
                        choices=["baseline", "gpu-pin", "tier-pin", "all"],
                        default="tier-pin",
                        help="Which configuration to run")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--zmq-port", type=int, default=5557)
    parser.add_argument("--hicache-write-policy", default="write_through",
                        choices=["write_back", "write_through", "write_through_selective"])
    parser.add_argument("--hicache-ratio", type=float, default=2.0,
                        help="Host memory as multiple of GPU KV cache")
    parser.add_argument("--flush", action="store_true",
                        help="Use /flush_cache instead of flood for eviction (much faster)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "params.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    all_messages = load_vip_turns(args.vip_dataset)
    logger.info(f"VIP conversation: {len(all_messages)} turns")

    if args.phase == "all":
        phases = ["baseline", "gpu-pin", "tier-pin"]
    else:
        phases = [args.phase]

    logger.info(f"PIN V6 Tier-Aware Benchmark")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Depths: {args.depths}")
    logger.info(f"  Phases: {phases}")
    logger.info(f"  HiCache write policy: {args.hicache_write_policy}")
    logger.info(f"  HiCache ratio: {args.hicache_ratio}")

    results = []

    for depth in args.depths:
        if depth + 2 > len(all_messages):
            logger.warning(f"Depth {depth} exceeds conversation length, skipping")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"DEPTH {depth} (turns 0..{depth}, measure turn {depth+1})")
        logger.info(f"{'='*70}")

        depth_result = {"depth": depth}

        for phase in phases:
            logger.info(f"\n--- {phase}: {PHASE_CONFIGS[phase]['description']} ---")
            depth_dir = str(output_dir / f"depth_{depth}")

            r = run_depth(
                depth=depth, phase_name=phase, all_messages=all_messages,
                model=args.model, aiperf_model=args.aiperf_model,
                port=args.port, mem_fraction=args.mem_fraction,
                tp_size=args.tp_size,
                flood_dataset=args.flood_dataset,
                flood_concurrency=args.flood_concurrency,
                flood_requests=args.flood_requests,
                output_dir=depth_dir, context_length=args.context_length,
                gpu_id=args.gpu, zmq_port=args.zmq_port,
                hicache_write_policy=args.hicache_write_policy,
                hicache_ratio=args.hicache_ratio,
                use_flush=args.flush,
            )
            depth_result[phase] = r

            logger.info(f"     {phase}: TTFT={r['ttft_ms']:.1f}ms, "
                        f"cached={r.get('cached_tokens', 0)}/{r.get('prompt_tokens', 0)}, "
                        f"host_hits={r.get('host_cache_hits', 0):.0f}")

        results.append(depth_result)

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"V6 TIER-AWARE PIN SUMMARY")
    logger.info(f"{'='*70}")

    header = f"{'Depth':>6}"
    for phase in phases:
        header += f" {'TTFT':>10} {'cached%':>8} {'host':>6}"
    logger.info(header)
    logger.info(f"{'-'*70}")

    for dr in results:
        line = f"{dr['depth']:>6}"
        for phase in phases:
            r = dr.get(phase, {})
            ttft = f"{r['ttft_ms']:>9.1f}ms" if r.get("ttft_ms") else "       N/A"
            cpct = f"{r.get('cache_hit_pct', 0):>7.0f}%" if r else "     N/A"
            host = f"{r.get('host_cache_hits', 0):>6.0f}" if r else "   N/A"
            line += f" {ttft} {cpct} {host}"
        logger.info(line)

    # Cross-phase speedup (if multiple phases)
    if len(phases) > 1 and "baseline" in phases:
        logger.info(f"\nSpeedup vs baseline:")
        for dr in results:
            bl = dr.get("baseline", {})
            bl_ttft = bl.get("ttft_ms", 0)
            if bl_ttft <= 0:
                continue
            for phase in phases:
                if phase == "baseline":
                    continue
                r = dr.get(phase, {})
                r_ttft = r.get("ttft_ms", 0)
                if r_ttft > 0:
                    speedup = bl_ttft / r_ttft
                    logger.info(f"  depth={dr['depth']}: {phase} {speedup:.1f}x")

    # Save
    with open(output_dir / "results.json", "w") as f:
        json.dump({"results": results, "params": vars(args)}, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
