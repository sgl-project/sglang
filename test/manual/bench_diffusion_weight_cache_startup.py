# SPDX-License-Identifier: Apache-2.0
"""Paired DiT HTTP startup benchmark, separate from functional acceptance.

Uses the existing HTTP server, owner and generation helpers. A warm owner is present
for BOTH modes, with alternating AB/BA order to reduce drift and the same
resolved placement. Process-entry through HTTP readiness is measured, including
imports/preflight; owner cold start is reported separately. No profiler, page
cache eviction or storage throttling is used. This measures warm-file recovery,
not cold storage. A regression bound is never treated as a speedup assertion.

Example:
  python test/manual/bench_diffusion_weight_cache_startup.py \
    --model-path /path/to/pinned/Wan/snapshot --output-dir /tmp/wan-startup-run
"""

import argparse
import hashlib
import json
import re
import shlex
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import requests

from sglang.multimodal_gen.test.server.test_server_utils import ServerManager
from sglang.multimodal_gen.test.single_test_file.test_weight_cache_1_gpu import (
    _generate,
    _start_owner,
    _stop_owner,
)
from sglang.multimodal_gen.test.single_test_file.test_weight_cache_minimax_h3_1_gpu import (
    MINIMAX_FLAGS,
    generate_minimax_h3,
)
from sglang.multimodal_gen.test.single_test_file.test_weight_cache_qwen_image_1_gpu import (
    generate_qwen_image,
)
from sglang.srt.utils.network import get_free_port


class StartupServerManager(ServerManager):
    """Keep normal readiness/error handling; observe both endpoints at 50 ms."""

    def start(self):
        self.started = time.perf_counter()
        self.readiness = {}
        return super().start()

    def _wait_for_ready(self, process, stdout_path):
        stopped = threading.Event()

        def observe():
            with requests.Session() as session:
                session.trust_env = False
                while not stopped.is_set() and process.poll() is None:
                    for name in ("liveness", "health"):
                        if name in self.readiness:
                            continue
                        try:
                            response = session.get(
                                f"http://127.0.0.1:{self.port}/{name}", timeout=0.2
                            )
                            if response.status_code == 200:
                                self.readiness[name] = (
                                    time.perf_counter() - self.started
                                )
                        except requests.RequestException:
                            pass
                    if len(self.readiness) == 2:
                        return
                    stopped.wait(0.05)

        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        try:
            super()._wait_for_ready(process, stdout_path)
            # The normal 1 s health loop might win by a few milliseconds.
            observer.join(timeout=2)
            assert len(self.readiness) == 2, self.readiness
        finally:
            stopped.set()
            observer.join(timeout=2)


def summarize(records):
    summary = {}
    for warmup in sorted({r["warmup"] for r in records}):
        subset = [r for r in records if r["warmup"] == warmup]
        result = {}
        for mode in ("off", "client"):
            samples = [r for r in subset if r["mode"] == mode]
            result[mode] = {
                field: {
                    "median": float(np.median([r[field] for r in samples])),
                    "p90": float(np.percentile([r[field] for r in samples], 90)),
                }
                for field in ("liveness", "health", "component")
            }
        pairs = sorted({r["pair"] for r in subset})
        for field in ("liveness", "health"):
            saved = [
                next(r[field] for r in subset if r["pair"] == p and r["mode"] == "off")
                - next(
                    r[field] for r in subset if r["pair"] == p and r["mode"] == "client"
                )
                for p in pairs
            ]
            result[f"paired_{field}_seconds_saved"] = {
                "samples": saved,
                "median": float(np.median(saved)),
                "positive_pairs": sum(value > 0 for value in saved),
            }
        summary[warmup] = result
    return summary


def run(options):
    root = Path(options.output_dir)
    # Do not overwrite an earlier evidence directory.
    root.mkdir(parents=True, exist_ok=False)
    model = str(Path(options.model_path).resolve(strict=True))
    records, references, placements = [], {}, {}
    qwen = options.model_kind == "qwen-image"
    minimax = options.model_kind == "minimax-h3"
    with tempfile.TemporaryDirectory(prefix="sgl-wc-perf-") as runtime:
        socket_path = Path(runtime) / "owner.sock"
        env = {"SGLANG_DIFFUSION_WEIGHT_CACHE_DIR": runtime, "HF_HUB_OFFLINE": "1"}
        if qwen:
            # Sharing changes free VRAM. Do not let warmup calibration turn that
            # into a different text-encoder placement in the A/B comparison.
            # Keep normal initial auto placement and synthetic server warmup.
            env["SGLANG_DIFFUSION_DISABLE_AUTO_RESIDENCY"] = "1"
        with (root / "owner.log").open("w") as log:
            owner_start = time.perf_counter()
            owner = _start_owner(
                model,
                socket_path,
                env,
                log,
                extra_args=shlex.split(MINIMAX_FLAGS) if minimax else (),
            )
            owner_seconds = time.perf_counter() - owner_start
            try:
                for warmup in options.warmup:
                    flags = f"--num-gpus 1 --warmup-mode {warmup}"
                    if minimax:
                        flags = f"{MINIMAX_FLAGS} --warmup-mode {warmup}"
                    if qwen:
                        flags += " --attention-backend fa"
                    if warmup == "server" and minimax:
                        flags += " --warmup-resolutions 1344x768 --warmup-num-frames 96 --warmup-steps 2"
                    elif warmup == "server":
                        flags += (
                            " --warmup-resolutions 1024x1024 --warmup-steps 1"
                            if qwen
                            else " --warmup-resolutions 832x480 --warmup-num-frames 9 --warmup-steps 1"
                        )
                    for pair in range(options.count):
                        order = (
                            ("off", "client") if pair % 2 == 0 else ("client", "off")
                        )
                        for mode in order:
                            name = f"{warmup}-{pair}-{mode}"
                            extra = flags
                            if mode == "client":
                                extra += f" --weight-cache-mode client --weight-cache-socket {shlex.quote(str(socket_path))}"
                            manager = StartupServerManager(
                                model, get_free_port(), extra_args=extra, env_vars=env
                            )
                            context = manager.start()
                            try:
                                if minimax:
                                    content = generate_minimax_h3(context, model, name)
                                elif qwen:
                                    content = generate_qwen_image(context, model, name)
                                else:
                                    content, _ = _generate(context, model, name)
                                digest = hashlib.sha256(content).hexdigest()
                                assert references.setdefault(warmup, digest) == digest
                                extension = "png" if qwen else "mp4"
                                (root / f"{name}.{extension}").write_bytes(content)
                                text = context.stdout_file.read_text()
                                (root / f"{name}.log").write_text(text)
                                args = json.loads(
                                    re.findall(r"server_args: (\{[^\n]+\})", text)[0]
                                )
                                assert args["warmup_mode"] == warmup
                                placement = {
                                    k: v
                                    for k, v in args.items()
                                    if "offload" in k
                                    or "residen" in k
                                    or k
                                    in ("use_fsdp_inference", "num_gpus", "tp_size")
                                }
                                assert (
                                    placements.setdefault(warmup, placement)
                                    == placement
                                )
                                pattern = (
                                    r"\[WeightCache\] transformer imported in ([\d.]+)s"
                                    if mode == "client"
                                    else r"\[ComponentLoader\] transformer materialized in ([\d.]+)s"
                                )
                                found = re.findall(pattern, text)
                                assert len(found) == 1, found
                                stages = {
                                    kind: json.loads(payload)
                                    for kind, payload in re.findall(
                                        r"\[WeightCache\] (launcher admission|transformer import) stages: (\{[^\n]+\})",
                                        text,
                                    )
                                }
                                record = {
                                    "warmup": warmup,
                                    "pair": pair,
                                    "mode": mode,
                                    **manager.readiness,
                                    "component": float(found[0]),
                                    "sha256": digest,
                                    "stages": stages,
                                }
                                records.append(record)
                                result = {
                                    "model": model,
                                    "model_kind": options.model_kind,
                                    "owner_start_seconds": owner_seconds,
                                    "owner_present_for_both_modes": True,
                                    "post_warmup_auto_residency": not (qwen or minimax),
                                    "poll_interval_seconds": 0.05,
                                    "storage_condition": "warm file cache, no eviction or throttling",
                                    "placement": placements,
                                    "samples": records,
                                }
                                (root / "samples.json").write_text(
                                    json.dumps(result, indent=2)
                                )
                                print("STARTUP_SAMPLE", json.dumps(record), flush=True)
                            finally:
                                context.cleanup()
                                context.process.wait(timeout=20)
                            assert owner.poll() is None
            finally:
                _stop_owner(owner)
    result["summary"] = summarize(records)
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    print("STARTUP_SUMMARY", json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--model-kind", choices=("wan", "qwen-image", "minimax-h3"), default="wan"
    )
    parser.add_argument(
        "--warmup", choices=("off", "server"), nargs="+", default=["off", "server"]
    )
    options = parser.parse_args()
    if options.count < 5:
        parser.error("At least five pairs per warmup mode are required")
    run(options)
