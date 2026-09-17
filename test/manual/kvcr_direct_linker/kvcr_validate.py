"""Real-GPU validation driver for the KVCR direct linker.

Launches SGLang servers, drives prompt sequences over HTTP, and writes a JSON
report with per-request ``cached_tokens``, greedy outputs, and the linker's
stats lines scraped from the server logs. Every scenario compares restored
runs against a control so byte movement is judged by output equality and
served tokens, not by counters alone.

Scenarios:

  roundtrip  One worker. Prompt A, enough distinct prompts to evict A from the
             GPU, then A again: with the linker the replay is served from KVCR
             (cached_tokens > 0) and reproduces the control output.
  peer       Two workers with separate KVCR tiers. The source serves A and
             offloads it; the cold target replays A with an explicit kv.fetch
             hint, then controls: no hint, stale hint, dead peer.

Example:
  python kvcr_validate.py roundtrip --model Qwen/Qwen3-0.6B --gpus 0 \
      --workdir /tmp/kvcr-val --report roundtrip_tp1.json
  python kvcr_validate.py peer --model Qwen/Qwen3-0.6B --gpus 0,1 --tp 1 \
      --workdir /tmp/kvcr-val --report peer_tp1.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from sglang.srt.mem_cache.utils import get_storage_hash_str, hash_str_to_int64

STATS_RE = re.compile(r"KVCRDirectLinker stats rank=(\d+): (.*)$")


def _post(url: str, payload: dict, timeout: float = 900.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _get(url: str, timeout: float = 10.0) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status


class Server:
    def __init__(
        self,
        *,
        name: str,
        model: str,
        port: int,
        gpus: str,
        tp: int,
        workdir: Path,
        page_size: int,
        max_total_tokens: int,
        linker_config: dict | None,
        extra_args: list[str],
    ) -> None:
        self.name = name
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.log_path = workdir / f"{name}.log"
        self.linker_config = linker_config
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tp-size",
            str(tp),
            "--page-size",
            str(page_size),
            "--max-total-tokens",
            str(max_total_tokens),
            "--log-level",
            "info",
            *extra_args,
        ]
        if linker_config is not None:
            command += [
                "--enable-unified-cache-external-linker",
                "--unified-cache-external-linker-backend",
                "kvcr",
                "--hicache-storage-backend-extra-config",
                json.dumps(linker_config),
            ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
        self.command = command
        self.log = open(self.log_path, "w")
        self.process = subprocess.Popen(
            command, stdout=self.log, stderr=subprocess.STDOUT, env=env
        )

    def wait_ready(self, timeout_s: float = 900.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name} exited; see {self.log_path}")
            try:
                if _get(f"{self.base}/health") == 200:
                    return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            time.sleep(1.0)
        raise TimeoutError(f"{self.name} did not become healthy")

    def generate(
        self, token_ids: list[int], max_new_tokens: int, kv_hints=None
    ) -> dict:
        payload = {
            "input_ids": token_ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
        }
        if kv_hints is not None:
            payload["kv_hints"] = kv_hints
        return _post(f"{self.base}/generate", payload)

    def flush(self) -> None:
        _post(f"{self.base}/flush_cache", {}, timeout=120.0)

    def stats(self) -> dict[str, dict[str, str]]:
        """Latest stats line per rank from the log."""
        latest: dict[str, dict[str, str]] = {}
        with open(self.log_path, errors="replace") as handle:
            for line in handle:
                match = STATS_RE.search(line)
                if match is None:
                    continue
                fields = dict(pair.split("=", 1) for pair in match.group(2).split())
                latest[match.group(1)] = fields
        return latest

    def log_matches(self, pattern: str) -> list[str]:
        regex = re.compile(pattern)
        with open(self.log_path, errors="replace") as handle:
            return [line.rstrip() for line in handle if regex.search(line)]

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)
        self.log.close()


def _prompt(rng: random.Random, length: int, vocab: int) -> list[int]:
    return [rng.randrange(1000, vocab) for _ in range(length)]


def _hint(source_control: str, token_ids: list[int], page_size: int) -> dict:
    pages = len(token_ids) // page_size * page_size
    hashes = get_storage_hash_str(token_ids[:pages], None, page_size=page_size)
    return {
        "protocol_version": "0.1",
        "message_id": uuid.uuid4().hex,
        "actions": [
            {
                "action_id": uuid.uuid4().hex,
                "action_type": "kv.fetch",
                "action_version": "1.0",
                "payload": {
                    "source_control_endpoint": source_control,
                    "block_hashes": [hash_str_to_int64(h) for h in hashes],
                },
            }
        ],
    }


def _summary(result: dict) -> dict:
    meta = result["meta_info"]
    return {
        "text": result["text"],
        "cached_tokens": meta.get("cached_tokens"),
        "prompt_tokens": meta.get("prompt_tokens"),
        "e2e_latency_s": meta.get("e2e_latency"),
    }


def _linker_config(args, *, control_port: int | None) -> dict:
    config = {
        "local_dram_bytes_per_worker": args.dram_gib << 30,
        "preparation_deadline_ms": args.deadline_ms,
        "fetch_chunk_pages": 64,
        # Scheduler processes may be killed before close() logs, so log often.
        "stats_log_interval_s": 2.0,
    }
    if control_port is not None:
        config.update(
            enable_remote_hint=True,
            control_port=control_port,
            control_advertise_host="127.0.0.1",
        )
    return config


def scenario_roundtrip(args, workdir: Path) -> dict:
    """Local offload, forced GPU eviction, restore; control run without linker."""
    rng = random.Random(args.seed)
    prompt = _prompt(rng, args.prompt_tokens, args.vocab)
    fillers = [
        _prompt(rng, args.filler_tokens, args.vocab)
        for _ in range(args.max_total_tokens // args.filler_tokens + 4)
    ]
    report: dict = {"scenario": "roundtrip", "tp": args.tp, "runs": {}}
    for label, linker in (
        ("control", None),
        ("linker", _linker_config(args, control_port=None)),
    ):
        server = Server(
            name=f"roundtrip_{label}",
            model=args.model,
            port=args.port,
            gpus=args.gpus,
            tp=args.tp,
            workdir=workdir,
            page_size=args.page_size,
            max_total_tokens=args.max_total_tokens,
            linker_config=linker,
            extra_args=args.extra,
        )
        try:
            server.wait_ready()
            first = server.generate(prompt, args.max_new_tokens)
            time.sleep(args.settle_s)
            for filler in fillers:
                server.generate(filler, 1)
            time.sleep(args.settle_s)
            replay = server.generate(prompt, args.max_new_tokens)
            time.sleep(args.settle_s)
            report["runs"][label] = {
                "first": _summary(first),
                "replay": _summary(replay),
                "stats": server.stats(),
                "host_pool_lines": server.log_matches(r"HiCache|host memory|hicache"),
                "linker_startup": server.log_matches(r"KVCRDirectLinker rank="),
                "command": server.command,
            }
        finally:
            server.stop()
    control, linker = report["runs"]["control"], report["runs"]["linker"]
    report["checks"] = {
        "control_replay_recomputed": control["replay"]["cached_tokens"] in (0, None),
        "linker_replay_restored": (linker["replay"]["cached_tokens"] or 0) > 0,
        "outputs_identical": control["first"]["text"]
        == control["replay"]["text"]
        == linker["first"]["text"]
        == linker["replay"]["text"],
        "no_hicache_host_pool": not any(
            "host" in line.lower() and "alloc" in line.lower()
            for line in linker["host_pool_lines"]
        ),
    }
    return report


def scenario_peer(args, workdir: Path) -> dict:
    """Peer reuse with cold target GPU and KVCR, plus no/stale/dead-peer controls."""
    rng = random.Random(args.seed)
    prompt = _prompt(rng, args.prompt_tokens, args.vocab)
    stale = _prompt(rng, args.prompt_tokens, args.vocab)
    gpus = args.gpus.split(",")
    per_worker = max(1, len(gpus) // 2)
    source_gpus = ",".join(gpus[:per_worker])
    target_gpus = ",".join(gpus[per_worker : 2 * per_worker])
    source_control = args.control_port
    target_control = args.control_port + 100
    report: dict = {"scenario": "peer", "tp": args.tp, "runs": {}}
    source = Server(
        name="peer_source",
        model=args.model,
        port=args.port,
        gpus=source_gpus,
        tp=args.tp,
        workdir=workdir,
        page_size=args.page_size,
        max_total_tokens=args.max_total_tokens,
        linker_config=_linker_config(args, control_port=source_control),
        extra_args=args.extra,
    )
    target = Server(
        name="peer_target",
        model=args.model,
        port=args.port + 1,
        gpus=target_gpus,
        tp=args.tp,
        workdir=workdir,
        page_size=args.page_size,
        max_total_tokens=args.max_total_tokens,
        linker_config=_linker_config(args, control_port=target_control),
        extra_args=args.extra,
    )
    try:
        source.wait_ready()
        target.wait_ready()
        control = source.generate(prompt, args.max_new_tokens)
        time.sleep(args.settle_s)
        source_endpoint = f"tcp://127.0.0.1:{source_control}"
        runs = {"control_source": _summary(control)}

        def cold_target_run(label: str, kv_hints, expect_hit: bool) -> None:
            target.flush()
            time.sleep(0.5)
            started = time.monotonic()
            result = target.generate(prompt, args.max_new_tokens, kv_hints=kv_hints)
            runs[label] = _summary(result)
            runs[label]["wall_s"] = time.monotonic() - started
            runs[label]["expect_hit"] = expect_hit

        cold_target_run("hinted", _hint(source_endpoint, prompt, args.page_size), True)
        # Warm target: a second hinted replay must be served from the local
        # tier, not presented as peer reuse.
        cold_target_run(
            "hinted_again", _hint(source_endpoint, prompt, args.page_size), True
        )
        cold_target_run("no_hint", None, False)
        cold_target_run(
            "stale_hint", _hint(source_endpoint, stale, args.page_size), False
        )
        cold_target_run(
            "dead_peer",
            _hint(f"tcp://127.0.0.1:{args.control_port + 500}", prompt, args.page_size),
            False,
        )
        time.sleep(args.settle_s)
        report["runs"] = runs
        report["source_stats"] = source.stats()
        report["target_stats"] = target.stats()
        report["commands"] = {"source": source.command, "target": target.command}
    finally:
        source.stop()
        target.stop()
    runs = report["runs"]
    texts = {label: run["text"] for label, run in runs.items()}
    report["checks"] = {
        "hinted_restored": (runs["hinted"]["cached_tokens"] or 0) > 0,
        "no_hint_recomputed": runs["no_hint"]["cached_tokens"] in (0, None),
        "stale_hint_recomputed": runs["stale_hint"]["cached_tokens"] in (0, None),
        "dead_peer_recomputed": runs["dead_peer"]["cached_tokens"] in (0, None),
        "dead_peer_bounded_wait_s": runs["dead_peer"]["wall_s"],
        "outputs_identical": len(set(texts.values())) == 1,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=["roundtrip", "peer"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--port", type=int, default=30100)
    parser.add_argument("--control-port", type=int, default=25100)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--prompt-tokens", type=int, default=3000)
    parser.add_argument("--filler-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--vocab", type=int, default=30000)
    parser.add_argument("--dram-gib", type=int, default=4)
    parser.add_argument("--deadline-ms", type=int, default=5000)
    parser.add_argument("--settle-s", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--report", required=True)
    # Unknown flags are forwarded to sglang.launch_server verbatim.
    args, args.extra = parser.parse_known_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    report = (scenario_roundtrip if args.scenario == "roundtrip" else scenario_peer)(
        args, workdir
    )
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["checks"], indent=2))
    return (
        0
        if all(v is True for k, v in report["checks"].items() if isinstance(v, bool))
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
