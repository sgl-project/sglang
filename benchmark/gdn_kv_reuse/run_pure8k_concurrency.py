"""Frozen-source ABBA benchmark: identical pure-8k targets at concurrency 1/8/16."""

import argparse
import atexit
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
SOURCE = Path(__file__).resolve().parents[2]
OUT = args.output_dir.resolve()
PYTHON = sys.executable
BENCH = Path(__file__).with_name("bench_pure_concurrency.py")
MODELS = ["Qwen3.5-4B", "Qwen3.5-9B", "Olmo-Hybrid-7B"]
MODES = ["base", "abl", "abl", "base"]
CONCURRENCIES = [1, 8, 16]
COMMIT = subprocess.check_output(
    ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
).strip()
URL = "http://127.0.0.1:18110"
OUT.mkdir(parents=True, exist_ok=False)
(OUT / "suite.pid").write_text(str(os.getpid()) + "\n")
(OUT / "REPORT.md").write_text(
    "纯 8192 输入、输出 1，三模型并发 1/8/16 实验运行中。结果由脚本自动更新。\n"
)
complete = False


def stop(signum, frame):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, stop)


@atexit.register
def finish():
    if not complete:
        (OUT / "FAILED.json").write_text(
            json.dumps(
                dict(
                    message="Suite ended before completion; inspect run.log and completed blocks."
                ),
                indent=2,
            )
        )
        with (OUT / "REPORT.md").open("a") as f:
            f.write("\n实验未完整结束，请查看 run.log；已完成块和模型数据仍保留。\n")


head = COMMIT
paths = sorted(
    str(p.relative_to(SOURCE))
    for folder in ["python", "benchmark/gdn_kv_reuse"]
    for p in (SOURCE / folder).rglob("*")
    if p.is_file()
)
metadata = dict(
    source_commit=head,
    source_snapshot=str(SOURCE),
    snapshot_kind="checkout with file hashes",
    source_sha256={
        p: hashlib.sha256((SOURCE / p).read_bytes()).hexdigest() for p in paths
    },
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    bench_sha256=hashlib.sha256(BENCH.read_bytes()).hexdigest(),
    fixture_origin="benchmark/gdn_kv_reuse/bench_pure_replay.py at " + COMMIT,
    gpu=subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).splitlines()[0],
    packages={
        p: importlib.metadata.version(p)
        for p in ["torch", "transformers", "flashinfer-python", "sglang-kernel"]
    },
    checkpoints={
        m: json.loads((args.model_dir / m / "download_revision.json").read_text())
        for m in MODELS
    },
    protocol=dict(
        modes=MODES,
        concurrencies=CONCURRENCIES,
        targets_per_round=16,
        input_tokens=8192,
        output_tokens=1,
        C=4096,
        H=8192,
        D=4096,
        chunk=4096,
        page_size=256,
        tp=1,
        max_total_tokens=98304,
        max_running_requests=16,
        max_mamba_cache_size=64,
        mem_fraction_static=0.9,
        weights_dtype="bfloat16",
        ssm_dtype="bfloat16",
        conv_dtype="bfloat16",
        overlap=False,
        prefill_graph=False,
        decode_graph=False,
        batch_invariant_mm=False,
        warmup_rounds_per_block=3,
        timing_rounds_per_block=16,
        discard_first_timing_round=True,
        retained_rounds_per_mode=30,
        retained_requests_per_concurrency_mode=480,
        target_submission="Same sixteen target prompts and preparation at each concurrency. Groups of 1, 8 or 16 submitted sequentially within a round.",
        primary_metric="Per-request server ReqTimeStats.forward_duration p50; excludes initial queue and HTTP, includes interleaved server work.",
        throughput_metric="16 / mean client duration to finish the same sixteen-target set; grouping differs by concurrency.",
        note="Global chunk budget is 4096. Each replay is exactly 4096, so the long prefill batch is normally one request even at serving concurrency 16. No suffix is appended.",
    ),
)
(OUT / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
# Preserve the actual scripts alongside the results.
(OUT / "driver.py").write_bytes(Path(__file__).read_bytes())
(OUT / "bench.py").write_bytes(BENCH.read_bytes())
env = os.environ.copy()
env.update(
    CUDA_VISIBLE_DEVICES="0",
    PYTHON=PYTHON,
    PYTHONPATH=str(SOURCE / "python"),
    MAX_RUNNING_REQUESTS="16",
    MAX_TOTAL_TOKENS="98304",
    MAX_MAMBA_CACHE_SIZE="64",
    MEM_FRACTION_STATIC="0.9",
    MAMBA_SSM_DTYPE="bfloat16",
    SGLANG_MAMBA_SSM_DTYPE="bfloat16",
    SGLANG_MAMBA_CONV_DTYPE="bfloat16",
    ENABLE_OVERLAP="0",
    BATCH_INVARIANT_MM="0",
    PATH=str(Path(PYTHON).parent) + os.pathsep + env["PATH"],
)


@contextmanager
def server(model, reuse, log_path):
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [
                "bash",
                str(SOURCE / "benchmark/gdn_kv_reuse/launch_server.sh"),
                str(args.model_dir / model),
                str(reuse),
                "18110",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd="/tmp",
        )
        try:
            for _ in range(240):
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited: {log_path}")
                try:
                    if requests.get(URL + "/health", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("startup timeout")
            assert re.search(
                r"max_total_num_tokens=98304,.*max_running_requests=16",
                log_path.read_text(),
            ), log_path
            yield
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


input_hashes = {}


def bench(model, c, dest, stage, repeats):
    command = [
        PYTHON,
        str(OUT / "bench.py"),
        "--model-path",
        str(args.model_dir / model),
        "--url",
        URL,
        "--concurrency",
        str(c),
        "--output-dir",
        str(dest),
        "--rid-prefix",
        f"c{c}-{stage}-",
        "--new-tokens",
        "1",
        "--repeats",
        str(repeats),
    ]
    with Path(str(dest) + ".log").open("w") as log:
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd="/tmp"
        )
    inputs = json.loads((dest / "inputs.json").read_text())
    assert len(inputs["targets"]) == 16 and all(
        len(ids) == 8192 for ids in inputs["targets"]
    )
    digest = hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if model in input_hashes:
        assert input_hashes[model] == digest, (model, "request or preparation changed")
    else:
        input_hashes[model] = digest
    return json.loads((dest / "responses.json").read_text())


def events(text, c, stage):
    return re.findall(
        rf"GDN_KV_REUSE rid=c{c}-{stage}-target-(\d+)-(\d+) C=(\d+) H=(\d+) tokens=(\d+)",
        text,
    )


def compare(actual, reference):
    return dict(
        matching_tokens=sum(
            a["output_ids"] == b["output_ids"] for a, b in zip(actual, reference)
        ),
        exact_selected_logprobs=sum(
            a["meta_info"]["output_token_logprobs"]
            == b["meta_info"]["output_token_logprobs"]
            for a, b in zip(actual, reference)
        ),
        exact_top5=sum(
            a["meta_info"]["output_top_logprobs"]
            == b["meta_info"]["output_top_logprobs"]
            for a, b in zip(actual, reference)
        ),
        mismatched_tokens=[
            i
            for i, (a, b) in enumerate(zip(actual, reference))
            if a["output_ids"] != b["output_ids"]
        ],
    )


def report(summary):
    lines = [
        "# 纯 8k、输出 1：并发 1 / 8 / 16",
        "",
        f"源码固定为 {COMMIT[:9]}，三模型、三种并发的计算设置与容量相同：BF16 权重/SSM/conv、TP=1、chunk=4096、C=4096、H=8192、D=4096；关闭 overlap 和 CUDA Graph，普通 CUDA MM。没有追加新输入 token。",
        "",
        "| 模型 | 并发 | Base forward p50 | Abl forward p50 | 耗时降低 | D 区域 KV 复用率 | Base → Abl req/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, cases in summary.items():
        for c, row in cases.items():
            a, b = row["base"], row["abl"]
            lines.append(
                f"| {model} | {c} | {a['median_forward_ms']:.2f} ms | {b['median_forward_ms']:.2f} ms | {row['saving_percent']:.2f}% | {100 * b['kv_reuse_fraction']:.1f}% | {a['requests_per_second']:.3f} → {b['requests_per_second']:.3f} |"
            )
    lines += [
        "",
        "每轮固定同样的 16 条目标及相同缓存准备。并发 1/8/16 分别按每组 1/8/16 条提交。每块预热 3 轮，计时 16 轮丢弃第一轮，ABBA 后每方案保留 30 轮、480 请求。缓存准备和预热不计入主指标。",
        "",
        "主指标为每请求 server forward_duration 的 p50，排除初始排队和 HTTP，但包含执行期间的服务器工作；它不是纯 GPU kernel 时间。吞吐为 16/平均客户端总耗时，同时保存各组耗时与请求排队时间。",
        "",
        "KV 复用率的分母仅为每请求 4096-token 重放段；前 4096-token 的标准缓存命中不计入该比例。",
        "",
        "chunk=4096 是全局预算；每请求恰好重放 4096 token，因此服务并发较高时仍通常逐个执行 prefill，并不代表模型 prefill batch size 为 8 或 16。",
        "",
        "所有模式统一：98304 KV 槽、64 Mamba state 槽、mem_fraction_static=0.9。非计时 seed 为 8449 后延伸至 8705，再用分支请求建立 4096 GDN state。Qwen 使用原生模型；Olmo 使用明确的外部实验适配层。",
        "",
        "## 一 token 输出诊断（普通 CUDA；不将数值差异标为精确一致）",
        "",
    ]
    for model, cases in summary.items():
        for c, row in cases.items():
            different = sorted(
                {
                    i
                    for b in row["blocks"]
                    if b["mode"] == "abl"
                    for i in b["diagnostic"]["mismatched_tokens"]
                }
            )
            lines.append(
                f"- {model}，并发 {c}：Base/Abl 不同的请求索引 {different}。所选 logprob/top-5 和全部原始响应另存。"
            )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


# Check all requested settings before leaving the long experiment unattended.
checks = {}
preflight = OUT / "preflight"
preflight.mkdir()
for model in MODELS:
    dest = preflight / model
    dest.mkdir()
    with server(model, 1, dest / "server.log"):
        checks[model] = {}
        for c in CONCURRENCIES:
            bench(model, c, dest / f"c{c}", "preflight", 1)
            found = events((dest / "server.log").read_text(), c, "preflight")
            assert len(found) == 16 and {
                int(j) for i, j, start, end, n in found
            } == set(range(16))
            assert all(
                (int(start), int(end), int(n)) == (4096, 8192, 4096)
                for i, j, start, end, n in found
            ), (model, c, found)
            checks[model][str(c)] = dict(
                requests=16,
                input_tokens=8192,
                output_tokens=1,
                reused_token_rows=65536,
                kv_reuse_fraction=1.0,
            )
            (OUT / "PREFLIGHT_OK.json").write_text(json.dumps(checks, indent=2))
            print(
                json.dumps(
                    dict(event="PREFLIGHT_OK", model=model, concurrency=c, reuse=1.0)
                ),
                flush=True,
            )
print("ALL_PREFLIGHT_OK", flush=True)

summary = {}
for model in MODELS:
    model_out = OUT / model
    model_out.mkdir()
    blocks = []
    references = {}
    for block, mode in enumerate(MODES):
        name = f"block-{block}-{mode}"
        log_path = model_out / f"{name}-server.log"
        # Each arm sees both forward and reverse concurrency order once.
        order = CONCURRENCIES if block % 2 == 0 else list(reversed(CONCURRENCIES))
        diagnostics = {}
        with server(model, int(mode == "abl"), log_path):
            for c in order:
                for stage, repeats in [
                    ("diagnostic", 1),
                    ("warmup", 3),
                    ("timing", 16),
                ]:
                    rows = bench(
                        model, c, model_out / f"{name}-c{c}-{stage}", stage, repeats
                    )
                    if stage == "diagnostic":
                        target = rows[0]["targets"]
                        if c not in references:
                            references[c] = target
                        diagnostics[c] = compare(target, references[c])
                        if model == MODELS[0] and block == 0 and c == 1:
                            (OUT / "STARTED.json").write_text(
                                json.dumps(
                                    dict(
                                        source_commit=COMMIT,
                                        model=model,
                                        completed_sixteen_target_set=True,
                                        input_tokens=8192,
                                        output_tokens=1,
                                    ),
                                    indent=2,
                                )
                            )
                            print(
                                "SUITE_STARTED: first baseline pure-8k target set completed",
                                flush=True,
                            )
        text = log_path.read_text()
        assert not re.search(
            r"Traceback|AssertionError|memory leak|out of memory", text, re.I
        ), log_path
        for c in CONCURRENCIES:
            matches = re.findall(
                rf"ReqTimeStats\(rid=c{c}-timing-target-(\d+)-(\d+), input_len=8192, cached_input_len=4096, output_len=1,.*?queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms",
                text,
            )
            assert len(matches) == 16 * 16, (
                model,
                block,
                c,
                "timing requests",
                len(matches),
            )
            kept = [
                (int(i), int(j), float(q), float(f))
                for i, j, q, f in matches
                if int(i) > 0
            ]
            active = events(text, c, "timing")
            if mode == "abl":
                assert len(active) == 16 * 16 and all(
                    (int(start), int(end), int(n)) == (4096, 8192, 4096)
                    for i, j, start, end, n in active
                ), (model, block, c, "KV reuse")
                assert (
                    len({(int(i), int(j)) for i, j, start, end, n in active}) == 16 * 16
                )
            else:
                assert not active
            rows = json.loads(
                (model_out / f"{name}-c{c}-timing/responses.json").read_text()
            )[1:]
            row = dict(
                block=block,
                mode=mode,
                concurrency=c,
                forward_ms=[r[3] for r in kept],
                queue_ms=[r[2] for r in kept],
                sixteen_target_ms=[r["wave_ms"] for r in rows],
                group_ms=[r["group_ms"] for r in rows],
                reused_token_rows=sum(
                    int(n) for i, j, start, end, n in active if int(i) > 0
                ),
                replay_token_rows=15 * 16 * 4096,
                diagnostic=diagnostics[c],
            )
            blocks.append(row)
            print(
                json.dumps(
                    dict(
                        model=model,
                        block=block,
                        mode=mode,
                        concurrency=c,
                        median_forward_ms=statistics.median(row["forward_ms"]),
                        diagnostic=row["diagnostic"],
                    )
                ),
                flush=True,
            )
        (model_out / "blocks.json").write_text(json.dumps(blocks, indent=2))
        # Publish summaries only after a model has completed all ABBA blocks.
        if block == 3:
            cases = {}
            for c in CONCURRENCIES:
                result = {}
                for arm in ["base", "abl"]:
                    selected = [
                        b for b in blocks if b["mode"] == arm and b["concurrency"] == c
                    ]
                    times = [v for b in selected for v in b["forward_ms"]]
                    waves = [v for b in selected for v in b["sixteen_target_ms"]]
                    result[arm] = dict(
                        requests=len(times),
                        rounds=len(waves),
                        median_forward_ms=statistics.median(times),
                        mean_forward_ms=statistics.mean(times),
                        median_sixteen_target_ms=statistics.median(waves),
                        requests_per_second=16000 / statistics.mean(waves),
                        kv_reuse_fraction=sum(b["reused_token_rows"] for b in selected)
                        / sum(b["replay_token_rows"] for b in selected),
                    )
                result["saving_percent"] = 100 * (
                    1
                    - result["abl"]["median_forward_ms"]
                    / result["base"]["median_forward_ms"]
                )
                result["blocks"] = [b for b in blocks if b["concurrency"] == c]
                cases[str(c)] = result
            summary[model] = cases
            (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
            report(summary)
report(summary)
(OUT / "results.json").write_text(
    json.dumps(
        dict(
            environment=metadata,
            preflight=checks,
            input_hashes=input_hashes,
            models=summary,
        ),
        indent=2,
    )
    + "\n"
)
complete = True
(OUT / "COMPLETE.json").write_text(
    json.dumps(
        dict(source_commit=COMMIT, models=MODELS, concurrencies=CONCURRENCIES), indent=2
    )
)
print("SUITE_COMPLETE", flush=True)
