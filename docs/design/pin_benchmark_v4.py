#!/usr/bin/env python3
"""PIN V4 Benchmark: aiperf-driven multi-turn eviction test.

Uses real multi-turn datasets from ~/datasets/ and delegates load generation
to aiperf.  The orchestrator handles server lifecycle, PIN/UNPIN API calls,
and KV event collection between aiperf phases.

Workflow (per rep):
  Phase A (baseline - no PIN):
    1. Fresh server start
    2. VIP warmup: replay VIP conversation (sonnet ~29.5K tokens)
    3. Flood: replay 10 opus sessions at high concurrency (natural eviction)
    4. VIP replay: replay VIP conversation again, measure TTFT
    5. Cache check: send system prompt, verify cached_tokens = 0 (evicted)
    6. Stop server

  Phase B (pinned):
    1. Fresh server start + KV event collector
    2. Send system prompt + first user msg to populate prefix cache
    3. PIN only those block hashes (system prompt only, not full conversation)
    4. VIP warmup: replay full VIP conversation
    5. Flood: replay 10 opus sessions at high concurrency (natural eviction)
    6. VIP replay: replay VIP conversation again, measure TTFT
    7. Cache check: send system prompt, verify cached_tokens > 0 (pinned!)
    8. UNPIN + stop server

  Compare Phase A vs Phase B post-flood TTFT.
  Cache check runs AFTER VIP replay to avoid re-warming the cache.

Usage:
    .venv/bin/python docs/design/pin_benchmark_v4.py
    .venv/bin/python docs/design/pin_benchmark_v4.py --num-reps 3 --flood-concurrency 32
"""

import argparse
import json
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
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

    def clear(self):
        self.block_hashes.clear()


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def start_server(model, port, mem_fraction, tp_size=1, context_length=32768,
                 page_size=64):
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--mem-fraction-static", str(mem_fraction),
        "--kv-events-config", json.dumps({
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": "tcp://*:5557",
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
    logger.info(f"Starting server: mem_fraction={mem_fraction}")
    log_file = Path(f"/tmp/sglang_server_{port}.log")
    log_fh = open(log_file, "w")

    env = os.environ.copy()
    os.makedirs(FLASHINFER_WORKSPACE, exist_ok=True)
    env["FLASHINFER_WORKSPACE_BASE"] = FLASHINFER_WORKSPACE

    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    proc._log_fh = log_fh
    return proc


def stop_server(proc):
    """Aggressively stop the server and all child processes."""
    if proc and proc.poll() is None:
        # Kill the entire process group to catch TP workers
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
    # Belt and suspenders: kill anything with sglang in the command line
    for pattern in ["sglang.launch_server", "sglang.srt", "multiprocessing.spawn"]:
        try:
            subprocess.run(["pkill", "-9", "-f", pattern],
                           capture_output=True, timeout=5)
        except Exception:
            pass
    # Wait for GPU memory to be released
    time.sleep(5)
    logger.info("Server stopped")


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


def flush_cache(base_url):
    # SGLang silently skips flush if requests are still running.
    # Retry a few times to ensure the flush actually happens.
    for attempt in range(5):
        r = requests.post(f"{base_url}/flush_cache", timeout=30)
        if r.status_code == 200:
            logger.info(f"    flush_cache: attempt {attempt+1} -> OK")
            time.sleep(1)
        else:
            logger.warning(f"    flush_cache: attempt {attempt+1} -> {r.status_code}")
            time.sleep(2)
    return True


# ---------------------------------------------------------------------------
# aiperf runner
# ---------------------------------------------------------------------------

def run_aiperf(
    model: str,
    aiperf_model: str,
    url: str,
    input_file: str,
    concurrency: int,
    num_requests: int | None = None,
    output_dir: str = "/tmp/aiperf_out",
    prefix: str = "phase",
    osl: int | None = None,
    streaming: bool = True,
    extra_args: list[str] | None = None,
    timeout: int = 600,
) -> dict:
    """Run aiperf profile and return parsed summary metrics.

    model: Full model name for server (e.g. Qwen/Qwen3-14B)
    aiperf_model: Short name for aiperf CLI (e.g. qwen3-14b) -- avoids slash parsing issues
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "aiperf", "profile",
        "--model", aiperf_model,
        "--url", url,
        "--endpoint-type", "chat",
        "--input-file", input_file,
        "--custom-dataset-type", "multi-turn",
        "--use-server-token-count",
        "--use-legacy-max-tokens",
        "--concurrency", str(concurrency),
        "--ui-type", "none",
        "--no-gpu-telemetry",
        "--no-server-metrics",
        "--output-artifact-dir", str(out_path),
        "--profile-export-prefix", prefix,
        "--export-level", "records",
    ]
    if streaming:
        cmd.append("--streaming")
    if num_requests is not None:
        cmd.extend(["--request-count", str(num_requests)])
    if osl is not None:
        cmd.extend(["--osl", str(osl)])
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"  aiperf: concurrency={concurrency}, input={Path(input_file).name}, "
                f"prefix={prefix}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(AIPERF_DIR),
        timeout=timeout,
    )

    if result.returncode != 0:
        logger.error(f"  aiperf failed (rc={result.returncode})")
        logger.error(f"  stderr: {result.stderr[-500:]}")
        logger.error(f"  stdout: {result.stdout[-500:]}")
        return {"error": f"aiperf exit {result.returncode}"}

    # Parse the summary JSON -- aiperf writes {prefix}_aiperf.json
    summary_file = out_path / f"{prefix}_aiperf.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        return summary

    # Fallback: try to find any JSON output
    for p in sorted(out_path.glob(f"*_aiperf.json")):
        with open(p) as f:
            return json.load(f)

    logger.warning(f"  No summary JSON found in {out_path}")
    return {"error": "no summary file"}


def extract_ttft_from_summary(summary: dict) -> dict:
    """Extract TTFT aggregate from aiperf summary JSON.
    aiperf structure: summary["time_to_first_token"] = {"unit":"ms","avg":...,"p50":...}
    """
    if "error" in summary:
        return summary
    ttft = summary.get("time_to_first_token", {})
    if not ttft:
        return {"error": "no time_to_first_token in summary"}
    return {
        "avg_ms": ttft.get("avg", 0),
        "p50_ms": ttft.get("p50", 0),
        "p90_ms": ttft.get("p90", 0),
        "p99_ms": ttft.get("p99", 0),
        "min_ms": ttft.get("min", 0),
        "max_ms": ttft.get("max", 0),
        "std_ms": ttft.get("std", 0),
    }


def extract_ttft_from_records(output_dir: str, prefix: str) -> dict:
    """Parse per-request TTFT from aiperf records JSONL.
    Record structure: {"metadata":{...},"metrics":{"time_to_first_token":{"value":X,"unit":"ms"},...}}

    Returns dict with aggregate stats plus turn_0_ttft_ms -- the TTFT for turn_index=0,
    which is the primary PIN metric. In multi-turn replays, only turn 0 has an identical
    context between warmup and probe (subsequent turns diverge because the model generates
    different assistant responses). This makes turn 0 TTFT the cleanest cold-vs-warm
    cache comparison.
    """
    records_dir = Path(output_dir)
    ttfts = []
    isls = []
    turn0_ttft = None

    for p in records_dir.glob(f"{prefix}*.jsonl"):
        if "raw" in p.name:
            continue
        with open(p) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    metadata = rec.get("metadata", {})
                    metrics = rec.get("metrics", {})
                    ttft_obj = metrics.get("time_to_first_token", {})
                    ttft = ttft_obj.get("value") if isinstance(ttft_obj, dict) else None
                    if ttft is not None and ttft > 0:
                        ttfts.append(ttft)
                        if metadata.get("turn_index") == 0:
                            turn0_ttft = ttft
                    isl_obj = metrics.get("input_sequence_length", {})
                    isl = isl_obj.get("value") if isinstance(isl_obj, dict) else None
                    if isl is not None:
                        isls.append(isl)
                except (json.JSONDecodeError, KeyError):
                    continue

    if not ttfts:
        return {"error": "no TTFT records found",
                "files": [str(p) for p in records_dir.glob(f"{prefix}*")]}

    ttfts.sort()
    n = len(ttfts)
    result = {
        "count": n,
        "avg_ms": sum(ttfts) / n,
        "p50_ms": ttfts[n // 2],
        "p90_ms": ttfts[int(n * 0.9)],
        "p99_ms": ttfts[int(n * 0.99)],
        "min_ms": ttfts[0],
        "max_ms": ttfts[-1],
    }
    if isls:
        result["avg_isl"] = sum(isls) / len(isls)
        result["max_isl"] = max(isls)
    if turn0_ttft is not None:
        result["turn_0_ttft_ms"] = turn0_ttft
    return result


# ---------------------------------------------------------------------------
# Benchmark phases
# ---------------------------------------------------------------------------

def _load_vip_system_prompt(vip_dataset: str) -> list[dict]:
    """Load system prompt + first user message from VIP dataset."""
    with open(vip_dataset) as f:
        data = json.loads(f.readline())
    turns = data["turns"]
    system_msg = next(t for t in turns if t["role"] == "system")
    user_msg = next(t for t in turns if t["role"] == "user")
    return [
        {"role": "system", "content": system_msg["text"]},
        {"role": "user", "content": user_msg["text"]},
    ]


def _send_system_prompt(base_url: str, aiperf_model: str, messages: list[dict]) -> dict:
    """Send system prompt request to populate cache. Returns usage info."""
    payload = {
        "model": aiperf_model,
        "messages": messages,
        "max_tokens": 1,
        "stream": False,
    }
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
    return r.json().get("usage", {})


def run_phase(
    label: str,
    model: str,
    aiperf_model: str,
    port: int,
    mem_fraction: float,
    tp_size: int,
    vip_dataset: str,
    flood_dataset: str,
    flood_concurrency: int,
    flood_requests: int | None,
    probe_concurrency: int,
    output_dir: str,
    pin: bool = False,
    flood_osl: int = 64,
    context_length: int | None = None,
) -> dict:
    """Run one complete phase (warmup -> [pin] -> flood -> probe)."""
    base_url = f"http://localhost:{port}"
    result = {"label": label, "pin": pin}

    proc = start_server(model, port, mem_fraction, tp_size, context_length)
    try:
        wait_for_server(base_url)

        # Step 1a: PIN system prompt only (if enabled)
        # Send a single request with just the system prompt to populate cache
        # and collect only those block hashes for pinning.
        blocks_pinned = 0
        pinned_hashes = []
        if pin:
            collector = KVEventCollector()
            collector.start()
            time.sleep(1)  # let ZMQ connect

            logger.info(f"  [{label}] Step 1a: Populate system prompt cache")
            sys_messages = _load_vip_system_prompt(vip_dataset)
            usage = _send_system_prompt(base_url, aiperf_model, sys_messages)
            prompt_tokens = usage.get("prompt_tokens", 0)
            logger.info(f"    System prompt: {prompt_tokens} tokens")

            time.sleep(1)  # let KV events arrive
            pinned_hashes = collector.get_unique_hashes()
            collector.stop()

            if pinned_hashes:
                pin_result = pin_blocks(base_url, pinned_hashes)
                blocks_pinned = pin_result.get("pinned_count", 0)
                logger.info(f"    Pinned {blocks_pinned}/{len(pinned_hashes)} blocks "
                            f"(system prompt only)")
            else:
                logger.warning("    No block hashes collected!")
        result["blocks_pinned"] = blocks_pinned

        # Step 1b: Warmup -- replay full VIP conversation to populate cache
        logger.info(f"  [{label}] Step 1b: Warmup with full VIP conversation")
        warmup_out = f"{output_dir}/{label}_warmup"
        warmup_summary = run_aiperf(
            model=model, aiperf_model=aiperf_model, url=base_url,
            input_file=vip_dataset,
            concurrency=1,
            output_dir=warmup_out,
            prefix="warmup",
            osl=64,  # short output for warmup
            streaming=True,
        )
        result["warmup_summary"] = extract_ttft_from_summary(warmup_summary)
        warmup_ttft = extract_ttft_from_records(warmup_out, "warmup")
        result["warmup_ttft"] = warmup_ttft
        if "error" not in warmup_ttft:
            logger.info(f"    Warmup TTFT: avg={warmup_ttft['avg_ms']:.1f}ms "
                        f"(n={warmup_ttft['count']}, max_isl={warmup_ttft.get('max_isl', '?')})")
        else:
            logger.warning(f"    Warmup TTFT extraction failed: {warmup_ttft}")

        # Step 3: Flood -- replay opus sessions at high concurrency
        logger.info(f"  [{label}] Step 3: Flood with multi-tenant traffic")
        flood_out = f"{output_dir}/{label}_flood"
        flood_extra = []
        if flood_requests:
            flood_extra.extend(["--request-count", str(flood_requests)])
        flood_summary = run_aiperf(
            model=model, aiperf_model=aiperf_model, url=base_url,
            input_file=flood_dataset,
            concurrency=flood_concurrency,
            output_dir=flood_out,
            prefix="flood",
            osl=flood_osl,
            streaming=True,
            extra_args=flood_extra if flood_extra else None,
            timeout=1200,
        )
        flood_ttft = extract_ttft_from_summary(flood_summary)
        result["flood_summary"] = flood_ttft
        flood_throughput = flood_summary.get("request_throughput", {}).get("avg", 0)
        logger.info(f"    Flood done: throughput={flood_throughput:.1f} req/s")

        # Wait for in-flight flood requests to fully drain on the server.
        # aiperf may exit before all server-side responses complete.
        logger.info(f"  [{label}] Draining in-flight requests...")
        for _ in range(30):
            try:
                r = requests.get(f"{base_url}/get_server_info", timeout=5)
                info = r.json()
                running = info.get("num_running_req", 0)
                waiting = info.get("num_waiting_req", 0)
                if running == 0 and waiting == 0:
                    break
                logger.info(f"    Still draining: {running} running, {waiting} waiting")
            except Exception:
                pass
            time.sleep(2)
        time.sleep(2)  # extra buffer

        # Health check after flood
        try:
            r = requests.get(f"{base_url}/health", timeout=10)
            logger.info(f"  [{label}] Health check after flood: {r.status_code}")
        except Exception as e:
            logger.warning(f"  [{label}] Health check failed: {e}")

        # Step 4: VIP replay -- replay VIP conversation again, measure TTFT.
        # This is the primary measurement. Must run BEFORE the cache check
        # to avoid re-warming the cache and contaminating the results.
        logger.info(f"  [{label}] Step 4: VIP replay (post-flood TTFT measurement)")
        vip_replay_out = f"{output_dir}/{label}_vip_replay"
        vip_replay_summary = run_aiperf(
            model=model, aiperf_model=aiperf_model, url=base_url,
            input_file=vip_dataset,
            concurrency=probe_concurrency,
            output_dir=vip_replay_out,
            prefix="vip_replay",
            osl=16,  # minimal output for measurement
            streaming=True,
            timeout=1200,
        )
        result["vip_replay_summary"] = extract_ttft_from_summary(vip_replay_summary)
        vip_replay_ttft = extract_ttft_from_records(vip_replay_out, "vip_replay")
        result["vip_replay_ttft"] = vip_replay_ttft
        if "error" not in vip_replay_ttft:
            logger.info(f"    VIP replay TTFT: avg={vip_replay_ttft['avg_ms']:.1f}ms "
                        f"(n={vip_replay_ttft['count']}, "
                        f"max_isl={vip_replay_ttft.get('max_isl', '?')})")
        else:
            logger.warning(f"    VIP replay TTFT extraction failed: {vip_replay_ttft}")

        # Step 5: Cache check -- send the system prompt to verify whether
        # the flood evicted it from cache. Runs AFTER the VIP replay so it
        # doesn't re-warm the cache and contaminate the measurement.
        logger.info(f"  [{label}] Step 5: Cache check (verify eviction status)")
        sys_messages = _load_vip_system_prompt(vip_dataset)
        try:
            check_start = time.perf_counter()
            check_usage = _send_system_prompt(base_url, aiperf_model, sys_messages)
            check_ttft_ms = (time.perf_counter() - check_start) * 1000
            check_prompt = check_usage.get("prompt_tokens", 0)
            details = check_usage.get("prompt_tokens_details") or {}
            check_cached = details.get("cached_tokens",
                                       check_usage.get("cached_tokens", 0))
            check_pct = (check_cached / check_prompt * 100) if check_prompt else 0
            evicted = check_pct < 50
            result["cache_check"] = {
                "ttft_ms": check_ttft_ms,
                "prompt_tokens": check_prompt,
                "cached_tokens": check_cached,
                "cache_hit_pct": check_pct,
                "evicted": evicted,
            }
            status = "EVICTED" if evicted else "CACHED"
            logger.info(f"    Cache check: {status} -- {check_cached}/{check_prompt} "
                        f"tokens cached ({check_pct:.0f}%), TTFT={check_ttft_ms:.1f}ms")
        except Exception as e:
            logger.warning(f"    Cache check failed: {e}")
            result["cache_check"] = {"error": str(e)}

        # Unpin
        if pinned_hashes:
            try:
                unpin_blocks(base_url, pinned_hashes)
            except Exception as e:
                logger.warning(f"    Unpin failed (server may have died): {e}")

    finally:
        stop_server(proc)
        # Copy server log for debugging
        import shutil
        log_src = Path(f"/tmp/sglang_server_{port}.log")
        log_dst = Path(output_dir) / f"{label}_server.log"
        log_dst.parent.mkdir(parents=True, exist_ok=True)
        if log_src.exists():
            shutil.copy2(log_src, log_dst)
        time.sleep(3)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PIN V4 Benchmark (aiperf-driven)")
    parser.add_argument("--model", default="Qwen/Qwen3-14B-FP8",
                        help="Full model path for SGLang server")
    parser.add_argument("--aiperf-model", default="qwen3-14b-fp8",
                        help="Short model name for aiperf CLI (no slashes)")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction", type=float, default=0.40)
    parser.add_argument("--num-reps", type=int, default=3)
    parser.add_argument("--flood-concurrency", type=int, default=8)
    parser.add_argument("--flood-requests", type=int, default=400,
                        help="Total flood requests (ensure all turns are processed)")
    parser.add_argument("--flood-osl", type=int, default=64,
                        help="Output sequence length for flood (keep short to control memory)")
    parser.add_argument("--context-length", type=int, default=32768,
                        help="Context length (with YaRN scaling)")
    parser.add_argument("--probe-concurrency", type=int, default=1)
    parser.add_argument("--vip-dataset", default=str(Path.home() / "datasets/claude_history_sonnet.jsonl"))
    parser.add_argument("--flood-dataset", default=str(Path.home() / "datasets/long_multiturn_opus.jsonl"))
    parser.add_argument("--output-dir", default="/tmp/pin_benchmark_v4")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save params
    params = vars(args)
    with open(output_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)
    logger.info("PIN V4 Benchmark (aiperf-driven)")
    logger.info(f"  Model: {args.model} (aiperf: {args.aiperf_model})")
    logger.info(f"  mem_fraction: {args.mem_fraction}")
    logger.info(f"  flood_concurrency: {args.flood_concurrency}")
    logger.info(f"  flood_osl: {args.flood_osl}")
    logger.info(f"  probe_concurrency: {args.probe_concurrency}")
    logger.info(f"  context_length: {args.context_length}")
    logger.info(f"  VIP dataset: {args.vip_dataset}")
    logger.info(f"  Flood dataset: {args.flood_dataset}")
    logger.info(f"  num_reps: {args.num_reps}")

    all_reps = []

    for rep in range(args.num_reps):
        logger.info(f"\n{'#'*70}")
        logger.info(f"# REP {rep+1}/{args.num_reps}")
        logger.info(f"{'#'*70}")

        rep_dir = str(output_dir / f"rep{rep}")

        # Phase A: Baseline (no PIN)
        logger.info(f"\n--- Baseline (no PIN) ---")
        baseline = run_phase(
            label="baseline",
            model=args.model,
            aiperf_model=args.aiperf_model,
            port=args.port,
            mem_fraction=args.mem_fraction,
            tp_size=args.tp_size,
            vip_dataset=args.vip_dataset,
            flood_dataset=args.flood_dataset,
            flood_concurrency=args.flood_concurrency,
            flood_requests=args.flood_requests,
            probe_concurrency=args.probe_concurrency,
            output_dir=rep_dir,
            pin=False,
            flood_osl=args.flood_osl,
            context_length=args.context_length,
        )

        # Phase B: Pinned
        logger.info(f"\n--- Pinned ---")
        pinned = run_phase(
            label="pinned",
            model=args.model,
            aiperf_model=args.aiperf_model,
            port=args.port,
            mem_fraction=args.mem_fraction,
            tp_size=args.tp_size,
            vip_dataset=args.vip_dataset,
            flood_dataset=args.flood_dataset,
            flood_concurrency=args.flood_concurrency,
            flood_requests=args.flood_requests,
            probe_concurrency=args.probe_concurrency,
            output_dir=rep_dir,
            pin=True,
            flood_osl=args.flood_osl,
            context_length=args.context_length,
        )

        # Compare -- all-turn average and turn 0 (primary PIN metric)
        bl_ttft = baseline.get("vip_replay_ttft", {}).get("avg_ms", 0)
        pin_ttft = pinned.get("vip_replay_ttft", {}).get("avg_ms", 0)
        delta_pct = ((pin_ttft - bl_ttft) / bl_ttft * 100) if bl_ttft else 0

        bl_t0 = baseline.get("vip_replay_ttft", {}).get("turn_0_ttft_ms", 0)
        pin_t0 = pinned.get("vip_replay_ttft", {}).get("turn_0_ttft_ms", 0)
        t0_delta_pct = ((pin_t0 - bl_t0) / bl_t0 * 100) if bl_t0 else 0

        bl_check = baseline.get("cache_check", {})
        pin_check = pinned.get("cache_check", {})

        rep_result = {
            "rep": rep,
            "baseline_warmup_ttft": baseline.get("warmup_ttft", {}),
            "baseline_vip_replay_ttft": baseline.get("vip_replay_ttft", {}),
            "baseline_cache_check": bl_check,
            "pinned_warmup_ttft": pinned.get("warmup_ttft", {}),
            "pinned_vip_replay_ttft": pinned.get("vip_replay_ttft", {}),
            "pinned_cache_check": pin_check,
            "blocks_pinned": pinned.get("blocks_pinned", 0),
            "delta_pct": delta_pct,
            "turn_0_baseline_ms": bl_t0,
            "turn_0_pinned_ms": pin_t0,
            "turn_0_delta_pct": t0_delta_pct,
        }

        def _fmt(v):
            return f"{v:.1f}" if isinstance(v, (int, float)) else str(v)

        logger.info(f"\n  ** Rep {rep+1} Results:")
        logger.info(f"     Baseline cache check: {'EVICTED' if bl_check.get('evicted') else 'CACHED'} "
                    f"({bl_check.get('cache_hit_pct', 0):.0f}% cached, "
                    f"TTFT={bl_check.get('ttft_ms', 0):.1f}ms)")
        logger.info(f"     Pinned cache check:   {'EVICTED' if pin_check.get('evicted') else 'CACHED'} "
                    f"({pin_check.get('cache_hit_pct', 0):.0f}% cached, "
                    f"TTFT={pin_check.get('ttft_ms', 0):.1f}ms)")
        logger.info(f"     Baseline VIP replay:  {_fmt(bl_ttft)}ms (all-turn avg)")
        logger.info(f"     Pinned VIP replay:    {_fmt(pin_ttft)}ms (all-turn avg)")
        logger.info(f"     All-turn delta:       {delta_pct:+.1f}%")
        logger.info(f"     Turn 0 baseline:      {_fmt(bl_t0)}ms")
        logger.info(f"     Turn 0 pinned:        {_fmt(pin_t0)}ms")
        logger.info(f"     Turn 0 delta:         {t0_delta_pct:+.1f}% *** PRIMARY ***")
        logger.info(f"     Blocks pinned:        {pinned.get('blocks_pinned', 0)}")

        all_reps.append(rep_result)

        with open(output_dir / f"rep{rep}.json", "w") as f:
            json.dump(rep_result, f, indent=2, default=str)

    # Summary
    deltas = [r["delta_pct"] for r in all_reps]
    t0_deltas = [r["turn_0_delta_pct"] for r in all_reps]
    if deltas:
        mean_delta = sum(deltas) / len(deltas)
        std_delta = (sum((d - mean_delta) ** 2 for d in deltas) / max(len(deltas) - 1, 1)) ** 0.5
        t0_mean = sum(t0_deltas) / len(t0_deltas)
        t0_std = (sum((d - t0_mean) ** 2 for d in t0_deltas) / max(len(t0_deltas) - 1, 1)) ** 0.5
        t0_bl_mean = sum(r["turn_0_baseline_ms"] for r in all_reps) / len(all_reps)
        t0_pin_mean = sum(r["turn_0_pinned_ms"] for r in all_reps) / len(all_reps)

        logger.info(f"\n{'='*70}")
        logger.info(f"V4 SUMMARY ({args.num_reps} reps)")
        logger.info(f"{'='*70}")
        logger.info(f"")
        logger.info(f"  PRIMARY: Turn 0 TTFT (first-request prefix cache hit)")
        logger.info(f"    Mean baseline: {t0_bl_mean:.1f}ms")
        logger.info(f"    Mean pinned:   {t0_pin_mean:.1f}ms")
        logger.info(f"    Delta:         {t0_mean:+.1f}%, std={t0_std:.1f}%")
        logger.info(f"    Speedup:       {t0_bl_mean/t0_pin_mean:.1f}x")
        logger.info(f"    Per-rep: {[f'{d:+.1f}%' for d in t0_deltas]}")
        logger.info(f"")
        logger.info(f"  SECONDARY: All-turn avg TTFT (diluted by self-caching)")
        logger.info(f"    Delta: {mean_delta:+.1f}%, std={std_delta:.1f}%")
        logger.info(f"    Per-rep: {[f'{d:+.1f}%' for d in deltas]}")

    summary = {"reps": all_reps, "params": params}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\nV4 Benchmark complete.")


if __name__ == "__main__":
    main()
