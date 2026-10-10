"""Real-model A/B/C tool-gap benchmark, fresh cache per trial, no delay injection.

Example: python bench_proactive_prefetch.py --model-path /opt/model --work-dir /work/bench --results-dir /work/results
Requires one FULL-attention model/worker, CUDA, and the file backend.
"""

import argparse
import collections
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--gaps-ms", type=int, nargs="+", default=[0, 100, 500, 1000, 3000]
    )
    args = parser.parse_args()
    work, out = args.work_dir, args.results_dir
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    source = work / "source-l3"
    source.mkdir()
    trace = out / "storage-trace.jsonl"
    url = f"http://127.0.0.1:{args.port}"
    flags = [
        "--model-path",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tp-size",
        "1",
        "--pp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "triton",
        "--sampling-backend",
        "pytorch",
        "--enable-deterministic-inference",
        "--random-seed",
        "42",
        "--max-running-requests",
        "1",
        "--context-length",
        "8192",
        "--max-total-tokens",
        "8192",
        "--page-size",
        "16",
        "--enable-hierarchical-cache",
        "--hicache-host-memory-mode",
        "cache",
        "--hicache-size",
        "4",
        "--hicache-io-backend",
        "kernel",
        "--hicache-mem-layout",
        "page_first",
        "--hicache-write-policy",
        "write_through",
        "--hicache-storage-backend",
        "file",
        "--hicache-storage-prefetch-policy",
        "wait_complete",
        "--hicache-storage-backend-extra-config",
        '{"prefetch_threshold":64}',
        "--enable-metrics",
        "--stream-interval",
        "1",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
    ]
    (out / "server-command.json").write_text(
        json.dumps([sys.executable, "-m", "sglang.launch_server"] + flags, indent=2)
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    ids = tokenizer.encode(
        "The validation notebook records observations about the lifecycle of cached tokens. "
        * 400,
        add_special_tokens=False,
    )[:4096]
    assert len(ids) == 4096
    prefix = ids[:4080]
    payload = dict(
        input_ids=ids,
        sampling_params=dict(
            temperature=0, max_new_tokens=32, ignore_eos=True, sampling_seed=42
        ),
        return_logprob=True,
        stream=True,
    )
    (out / "request.json").write_text(json.dumps(payload))

    def request(path, data=None, timeout=180):
        req = urllib.request.Request(
            url + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode()

    def control(action, operation_id, **kwargs):
        result = json.loads(
            request(
                "/hicache/prefetch",
                dict(action=action, operation_id=operation_id, **kwargs),
            )
        )
        assert result["success"], result
        return result["result"]

    def generate():
        start = time.monotonic_ns()
        first = None
        last = None
        req = urllib.request.Request(
            url + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                last = json.loads(data)
                assert "error" not in last, last
                if last.get("output_ids") and first is None:
                    first = time.monotonic_ns()
        assert first is not None and len(last["output_ids"]) == 32, last
        return last, dict(
            arrival_ns=start, ttft_ms=(first - start) / 1e6, end_ns=time.monotonic_ns()
        )

    def stop(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)

    def start(label, directory):
        env = os.environ.copy()
        env.update(
            SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=str(directory),
            HICACHE_BENCH_TRACE=str(trace),
            HICACHE_BENCH_LABEL=label,
            SGLANG_PLUGINS="hicache_trace",
        )
        env["PYTHONPATH"] = (
            str(Path(__file__).parent / "trace") + ":" + env.get("PYTHONPATH", "")
        )
        with (out / f"server-{label}.log").open("w") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "sglang.launch_server"] + flags,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server {label} exited {proc.returncode}")
                try:
                    request("/health", timeout=2)
                    return proc
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.5)
            raise TimeoutError(f"Server {label} did not become ready in 3 minutes")
        except BaseException:
            stop(proc)
            raise

    def events(label):
        result = []
        if trace.exists():
            for line in trace.read_text().splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item["label"] == label:
                    result.append(item)
        return result

    def upload():
        # Optional adapter in the validated container; never includes KV/model files.
        script = os.environ.get("HICACHE_RESULTS_UPLOADER")
        if script:
            subprocess.run([sys.executable, script], check=True, timeout=120)

    producer = start("producer", source)
    try:
        baseline, _ = generate()
        expected = baseline["output_ids"]
        previous = None
        stable = 0
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            current = sorted((x.name, x.stat().st_size) for x in source.glob("*.bin"))
            stable = stable + 1 if current and current == previous else 0
            previous = current
            if stable >= 3:
                break
            time.sleep(1)
        else:
            raise TimeoutError("L3 writes did not settle")
    finally:
        stop(producer)
    (out / "l3-source-manifest.json").write_text(json.dumps(previous))
    expected_file = out / "expected-output-ids.json"
    if expected_file.exists():
        assert json.loads(expected_file.read_text()) == expected, (
            "Resume changed baseline output"
        )
    expected_file.write_text(json.dumps(expected))
    upload()
    rows_file = out / "trials.json"
    rows = json.loads(rows_file.read_text()) if rows_file.exists() else []
    completed = {(r["repetition"], r["gap_ms"], r["mode"]) for r in rows}
    plans = [
        (rep, gap, mode)
        for rep in range(args.repetitions)
        for gap in args.gaps_ms
        for mode in "ABC"
    ]
    random.Random(42).shuffle(plans)
    for rep, gap, mode in plans:
        if (rep, gap, mode) in completed:
            continue
        label = f"{mode}-{gap}-{rep}"
        directory = work / label
        directory.mkdir()
        if mode != "A":
            for page in source.glob("*.bin"):
                shutil.copyfile(page, directory / page.name)
        proc = start(label, directory)
        try:
            tool_start = time.monotonic_ns()
            accepted = None
            if mode == "C":
                accepted = control("submit", label, input_ids=prefix, ttl_ms=60000)
            time.sleep(
                max(0, (tool_start + gap * 1_000_000 - time.monotonic_ns()) / 1e9)
            )
            response, timing = generate()
            assert response["output_ids"] == expected, (
                label,
                response["output_ids"],
                expected,
            )
            status = control("status", label) if mode == "C" else None
            records = events(label)
            arrival = timing["arrival_ns"]
            io = [x for x in records if x["kind"] == "restore_io"]
            reads = [x for x in records if x["kind"] == "read"]
            keys = collections.Counter(k for r in reads for k in r["keys"])
            published = [x for x in records if x["kind"] == "publish"]
            stats = dict(
                mode=mode,
                gap_ms=gap,
                repetition=rep,
                **timing,
                actual_signal_to_arrival_ms=(arrival - tool_start) / 1e6,
                restore_io_ms=sum((x["end_ns"] - x["start_ns"]) / 1e6 for x in io),
                restore_hidden_ms=sum(
                    max(0, min(x["end_ns"], arrival) - x["start_ns"]) / 1e6 for x in io
                ),
                restored_bytes=sum(x["bytes"] for x in reads),
                backend_read_pages=sum(keys.values()),
                duplicate_backend_read_pages=sum(max(0, n - 1) for n in keys.values()),
                read_pages_completed_before_arrival=sum(
                    len(x["keys"]) for x in reads if x["end_ns"] <= arrival
                ),
                published_before_arrival=any(
                    x["at_ns"] <= arrival and x["restored_tokens"] > 0
                    for x in published
                ),
                restored_tokens=sum(x["restored_tokens"] for x in published),
                host_evicted_tokens=sum(
                    x["evicted_tokens"] for x in records if x["kind"] == "evict_host"
                ),
                host_occupancy_snapshots=[
                    {
                        k: x[k]
                        for k in [
                            "kind",
                            "at_ns",
                            "host_used_tokens",
                            "inflight_tokens",
                        ]
                    }
                    for x in records
                    if "host_used_tokens" in x
                ],
                status=status,
                cached_tokens_details=response["meta_info"].get(
                    "cached_tokens_details"
                ),
                server_first_token_latency_ms=response["meta_info"].get(
                    "first_token_latency", 0
                )
                * 1000,
                output_correct=True,
            )
            if mode == "A":
                assert not reads, stats
            if mode == "B":
                assert io and all(x["start_ns"] >= arrival for x in io), stats
            if mode == "C":
                assert status["state"] in ("SUCCESS", "CACHED"), status
                assert io, stats
                # Query/allocation can overlap a short gap even if file reads
                # start later. Validate submission of the entire restore, then
                # measure actual I/O overlap independently (which may be zero).
                assert all(x["operation_start_ns"] < arrival for x in io), stats
                assert not any(
                    x["kind"] == "h2d_submit" and tool_start <= x["at_ns"] < arrival
                    for x in records
                ), "Proactive H2D is out of scope"
            rows.append(stats)
            (out / "trials.json").write_text(json.dumps(rows, indent=2))
            print(
                "TRIAL",
                json.dumps(
                    {
                        k: stats[k]
                        for k in [
                            "mode",
                            "gap_ms",
                            "repetition",
                            "ttft_ms",
                            "restore_io_ms",
                            "restore_hidden_ms",
                            "duplicate_backend_read_pages",
                            "published_before_arrival",
                        ]
                    }
                ),
                flush=True,
            )
            upload()
        finally:
            stop(proc)
            shutil.rmtree(directory)
    directory = work / "wasted"
    shutil.copytree(source, directory)
    proc = start("wasted", directory)
    try:
        initial = control("submit", "no-continuation", input_ids=prefix, ttl_ms=10000)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = control("status", "no-continuation")
            if status["state"] != "RUNNING":
                break
            time.sleep(0.01)
        assert status["state"] == "SUCCESS", status
        assert status["inflight_tokens"] == 0 and not status["cleanup_pending"], status
        after_cancel = control("cancel", "no-continuation")
        waste = dict(
            restored_tokens=status["restored_tokens"],
            wasted_bytes=status["restored_bytes"],
            host_available_token_delta=initial["host_available_tokens"]
            - status["host_available_tokens"],
            resident_after_cancel=after_cancel["state"],
            policy="Completed restore is ordinary evictable L2, no pin or post-publication lease; entire staged prefix is wasted if continuation never arrives",
        )
        request("/flush_cache", {})
        after_flush = control("status", "no-continuation")
        waste["host_available_after_flush"] = after_flush["host_available_tokens"]
        assert (
            after_flush["host_available_tokens"] >= initial["host_available_tokens"]
        ), waste
        (out / "wasted-prefetch.json").write_text(json.dumps(waste, indent=2))
    finally:
        stop(proc)
    summary = []
    for gap in args.gaps_ms:
        bymode = {
            mode: [x for x in rows if x["mode"] == mode and x["gap_ms"] == gap]
            for mode in "ABC"
        }
        item = dict(
            gap_ms=gap,
            **{
                f"{mode}_ttft_median_ms": statistics.median(
                    x["ttft_ms"] for x in bymode[mode]
                )
                for mode in "ABC"
            },
            C_hidden_median_ms=statistics.median(
                x["restore_hidden_ms"] for x in bymode["C"]
            ),
            B_restore_median_ms=statistics.median(
                x["restore_io_ms"] for x in bymode["B"]
            ),
        )
        item["B_minus_C_ms"] = item["B_ttft_median_ms"] - item["C_ttft_median_ms"]
        summary.append(item)
    result = dict(
        trials=len(rows),
        repetitions=args.repetitions,
        summary=summary,
        all_outputs_correct=all(x["output_correct"] for x in rows),
        duplicate_backend_read_pages=sum(
            x["duplicate_backend_read_pages"] for x in rows
        ),
        host_evicted_tokens=sum(x["host_evicted_tokens"] for x in rows),
        wasted_prefetch=waste,
        scope="one L4, FULL resident file, TP1 PP1, no injected delay",
    )
    (out / "benchmark.json").write_text(json.dumps(result, indent=2))
    print("BENCHMARK_RESULT", json.dumps(result), flush=True)
    upload()


if __name__ == "__main__":
    main()
