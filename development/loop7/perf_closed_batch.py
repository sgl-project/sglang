"""Loop-7 closed-batch decode-TPS + TTFT probe (AC-6 perf guardrail).

The trustworthy pure-decode-TPS method (per the loop's bench lessons): fire ``conc``
concurrent ``/generate`` requests with a SHORT prompt + ``ignore_eos`` + a fixed
output length, so the server runs a steady closed decode batch (no new arrivals,
``#queue-req: 0``). This avoids the GSP window-mode harness that can fabricate
throughput from empty streams (BL-20260531-bench-empty-stream-failclosed).

Two modes:
  * default (non-streaming): per-request decode TPS = output_tokens / e2e (prefill is
    negligible for a short prompt, so e2e ~= decode time). Kept for R19 reproducibility.
  * ``--stream`` (SSE): records TTFT (first-token arrival - submit) AND a clean
    post-first-token decode TPS = (completion_tokens - 1) / (t_last - t_first). This is
    the AC-6 TTFT guardrail. It mirrors the canonical SGLang streaming parser
    (``data: {"text": <cumulative>, "meta_info": {"completion_tokens": N}}``) and FAILS
    CLOSED on an HTTP-200 empty stream (never records a no-token response as a
    completion -- BL-20260531-bench-empty-stream-failclosed).

Usage:
    DS_BASE_URL=http://127.0.0.1:30000 python development/loop7/perf_closed_batch.py \
        --conc 16 --osl 256 --stream --label "DS-hybrid graph" --out development/loop7/perf_x.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import os
import subprocess
import time
import urllib.request


def _git_commit():
    """Best-effort short HEAD of the repo this file lives in (None on failure)."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return (
            subprocess.check_output(
                ["git", "-C", here, "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
            or None
        )
    except Exception:
        return None


def _git_tree_dirty():
    """Best-effort: True if the working tree has uncommitted changes (None on failure)."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.check_output(
            ["git", "-C", here, "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        return bool(out.strip())
    except Exception:
        return None


def _gpu_info():
    """Best-effort (gpu_name, gpu_count) via nvidia-smi ((None, None) on failure)."""
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
            .splitlines()
        )
        return (out[0].strip() if out else None), len(out)
    except Exception:
        return None, None


def build_run_provenance(**fields):
    """Assemble a run_provenance object for a measurement artifact (single schema source).

    All fields are caller-supplied so the SAME schema serves both a live `--stream` run
    (which auto-detects git/GPU and passes server-side facts via CLI) and a reconstructed
    backfill (which passes every field explicitly + ``reconstructed=True``). Keys with a
    None value are dropped so the object stays compact.
    """
    schema = [
        "reconstructed",
        "reconstructed_in_round",
        "note",
        "server_code_commit",
        "server_code_note",
        "measurement_tool_commit",
        "tree_dirty_during_run",
        "gpu",
        "gpu_count",
        "tp_size",
        "op_point",
        "launch_cmd",
        "server_config",
        "mem_fraction_static",
        "gpu_mem_per_gpu",
        "mem_source",
        "graph",
        "graph_evidence",
        "radix_cache",
        "overlap_schedule",
        "served",
        "artifact_path",
    ]
    out = {}
    for k in schema:
        v = fields.get(k)
        if v is not None:
            out[k] = v
    return out


def _pct(xs, q):
    """Linear-interpolation percentile (q in [0,1]); no numpy dependency."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _decode_tps(completion_tokens, e2e_s, ttft_s):
    """Per-request decode throughput in tokens/sec = tokens / (e2e - ttft).

    Decode-only throughput: output tokens over the post-first-token wall-time.
    Kept dependency-free so this probe stays stdlib-only; it is the same
    definition as ``sglang.bench_serving.decode_throughput_tps`` (cross-locked
    by a unit test). Deliberately NOT ``completion_tokens / e2e`` (that bills the
    decode rate for the prefill/TTFT time). Returns 0.0 for an unmeasurable
    sample (no decode window or no tokens).
    """
    decode_window = e2e_s - ttft_s
    if decode_window <= 0 or completion_tokens <= 0:
        return 0.0
    return completion_tokens / decode_window


def _one(base_url: str, osl: int, prompt: str = "The capital of France is"):
    """Non-streaming /generate: returns (e2e_seconds, completion_tokens)."""
    body = json.dumps(
        {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": osl,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        }
    ).encode()
    t0 = time.time()
    req = urllib.request.Request(
        base_url + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    e2e = time.time() - t0
    ct = int(d.get("meta_info", {}).get("completion_tokens", osl))
    return e2e, ct


def _one_stream(base_url: str, osl: int, prompt: str):
    """Streaming SSE /generate: returns dict with ttft, e2e, completion_tokens, decode_tps.

    Mirrors ``async_request_sglang_generate``: each SSE line is ``data: {json}``; the
    payload carries the *cumulative* ``text`` and ``meta_info.completion_tokens``. TTFT is
    the arrival time of the first chunk carrying non-empty text; pure decode TPS uses the
    post-first-token window only. FAILS CLOSED on an HTTP-200 stream that yields no token.
    """
    body = json.dumps(
        {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": osl,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        base_url + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    ttft = 0.0
    t_first = None
    t_last = t0
    completion_tokens = 0
    generated_text = ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            raw = raw.strip()
            if not raw:
                continue
            chunk = raw.decode("utf-8")
            if chunk.startswith("data: "):
                chunk = chunk[len("data: ") :]
            if chunk == "[DONE]":
                continue
            data = json.loads(chunk)
            if data.get("text"):
                now = time.perf_counter()
                generated_text = data["text"]
                completion_tokens = int(data["meta_info"]["completion_tokens"])
                if ttft == 0.0:
                    ttft = now - t0
                    t_first = now
                t_last = now
    e2e = time.perf_counter() - t0
    # Fail closed: an HTTP-200 stream that produced no token must NOT be recorded as a
    # completion (otherwise throughput/TTFT is fabricated). See the bench lesson.
    if ttft == 0.0 and not generated_text:
        raise RuntimeError(
            "HTTP 200 but streaming response produced no tokens (empty stream); "
            "refusing to record as a completed request"
        )
    decode_s = (t_last - t_first) if t_first is not None else 0.0
    decode_tps = (
        (completion_tokens - 1) / decode_s
        if (decode_s > 0 and completion_tokens > 1)
        else 0.0
    )
    return {
        "ttft": ttft,
        "e2e": e2e,
        "completion_tokens": completion_tokens,
        "decode_tps": decode_tps,
    }


def _run_concurrent(call, warmup, conc):
    """Warm up once (capture/JIT), then fire `conc` concurrent copies of `call()`.

    Returns ``(results, wall_seconds)``. ``call`` and ``warmup`` are zero-arg callables
    (e.g. ``functools.partial``) so the closed batch and its warmup can use different
    output lengths through one shared orchestration path.
    """
    warmup()
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(call) for _ in range(conc)]
        res = [f.result() for f in futs]
    return res, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--osl", type=int, default=256)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--stream",
        action="store_true",
        help="SSE streaming mode: record TTFT + clean post-first-token decode TPS",
    )
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help="repeat the prompt sentence N times to build a longer (prefill-bound) prompt",
    )
    # Optional run-provenance pass-through (server-side facts the probe can't introspect);
    # the git commit / tree-dirty / GPU are auto-detected. Recorded under run_provenance so
    # every `--stream` artifact is self-documenting.
    ap.add_argument("--launch-cmd", default=None, help="exact server launch command")
    ap.add_argument("--op-point", default=None, help="operating-point summary string")
    ap.add_argument("--mem-per-gpu", default=None, help="e.g. '125 GB'")
    ap.add_argument("--graph-evidence", default=None, help="a server-log decode line")
    args = ap.parse_args()
    base = os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000")
    prompt = (" ".join([args.prompt] * args.prompt_repeat)).strip()
    _gpu_name, _gpu_n = _gpu_info()

    if args.stream:
        res, wall = _run_concurrent(
            functools.partial(_one_stream, base, args.osl, prompt),
            functools.partial(_one_stream, base, 16, prompt),
            args.conc,
        )
        ttfts = [r["ttft"] for r in res]
        decode_tps = [r["decode_tps"] for r in res if r["decode_tps"] > 0]
        cts = [r["completion_tokens"] for r in res]
        # Canonical client-SLO decode throughput = out_tokens / (e2e - ttft) per
        # request (the new definition; distinct from the legacy decode_tps that
        # used the inter-arrival window, and from output_tokens / e2e).
        slo_tps = [
            t
            for t in (
                _decode_tps(r["completion_tokens"], r["e2e"], r["ttft"]) for r in res
            )
            if t > 0
        ]
        out = {
            "label": args.label,
            "mode": "stream",
            "conc": args.conc,
            "osl": args.osl,
            "prompt_repeat": args.prompt_repeat,
            "completed": len(res),
            "slo_decode_tps_mean": round(sum(slo_tps) / len(slo_tps), 2) if slo_tps else 0.0,
            "slo_decode_tps_p50": round(_pct(slo_tps, 0.50), 2) if slo_tps else 0.0,
            "slo_decode_tps_p10": round(_pct(slo_tps, 0.10), 2) if slo_tps else 0.0,
            "slo_decode_tps_min": round(min(slo_tps), 2) if slo_tps else 0.0,
            "ttft_ms_mean": round(1000 * sum(ttfts) / len(ttfts), 1),
            "ttft_ms_p50": round(1000 * _pct(ttfts, 0.50), 1),
            "ttft_ms_p99": round(1000 * _pct(ttfts, 0.99), 1),
            "ttft_ms_min": round(1000 * min(ttfts), 1),
            "ttft_ms_max": round(1000 * max(ttfts), 1),
            "ttft_ms_all": [round(1000 * t, 1) for t in ttfts],
            "per_req_decode_tps_mean": (
                round(sum(decode_tps) / len(decode_tps), 2) if decode_tps else 0.0
            ),
            "per_req_decode_tps_min": round(min(decode_tps), 2) if decode_tps else 0.0,
            "system_throughput_tok_s": round(sum(cts) / wall, 1),
            "total_out_tokens": sum(cts),
            "wall_s": round(wall, 1),
        }
        _tool_commit = _git_commit()
        out["run_provenance"] = build_run_provenance(
            measurement_tool_commit=_tool_commit,
            server_code_commit=_tool_commit,
            tree_dirty_during_run=_git_tree_dirty(),
            gpu=_gpu_name,
            gpu_count=_gpu_n,
            launch_cmd=args.launch_cmd,
            op_point=args.op_point,
            gpu_mem_per_gpu=args.mem_per_gpu,
            graph_evidence=args.graph_evidence,
            served=len(res),
            artifact_path=args.out,
        )
    else:
        res, wall = _run_concurrent(
            functools.partial(_one, base, args.osl, prompt),
            functools.partial(_one, base, 16, prompt),
            args.conc,
        )
        e2es = [e for e, _ in res]
        cts = [c for _, c in res]
        per_req_tps = [c / e for c, e in zip(cts, e2es) if e > 0]
        out = {
            "label": args.label,
            "mode": "e2e",
            "conc": args.conc,
            "osl": args.osl,
            "completed": len(res),
            "per_req_decode_tps_mean": round(sum(per_req_tps) / len(per_req_tps), 2),
            "per_req_decode_tps_min": round(min(per_req_tps), 2),
            "system_throughput_tok_s": round(sum(cts) / wall, 1),
            "mean_e2e_s": round(sum(e2es) / len(e2es), 2),
            "total_out_tokens": sum(cts),
            "wall_s": round(wall, 1),
        }

    print(json.dumps(out), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
