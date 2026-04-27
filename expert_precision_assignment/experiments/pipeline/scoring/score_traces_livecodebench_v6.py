"""Score a LiveCodeBench v6 trace JSONL against its meta + private-tests sidecars.

Inputs (paths relative to experiments/):
    --trace          data/results/livecodebench_v6/mc{mc}_{variant}.jsonl   (bench_serving)
    --meta           pipeline/prompt/livecodebench_v6.meta.jsonl            (prep output)
    --private-tests  pipeline/prompt/livecodebench_v6.private_tests.pkl     (prep output)
    --vendored       pipeline/scoring/vendored/lcb_runner                   (lcb checker)

The private-tests pickle is a dict {question_id: base64_zlib_pickle_string}.
Loaded once at startup and looked up by meta[i].question_id.

Extracts the Python code block from each generated_texts[i], reconstructs
the input/output test dict that lcb_runner expects, and calls
`lcb_runner.evaluation.testing_util.run_test(...)` per problem with a
hard timeout.  A problem passes iff every test case passes (pass@1
greedy; multi-sample pass@k is a TODO).

Emits overall accuracy plus breakdowns by platform, difficulty, and
contest_date window.

!!!  SECURITY WARNING  !!!
The checker EXECUTES MODEL-GENERATED CODE on this machine.  Only run
against models you trust.  The script uses a subprocess-per-problem
harness from lcb_runner which applies a CPU time limit; however, that
is not a security sandbox.  For untrusted models, run this scorer in
an isolated container / VM.

VENDORING (one-time, before running; paths relative to experiments/):

    mkdir -p pipeline/scoring/vendored
    cd pipeline/scoring/vendored
    git clone https://github.com/LiveCodeBench/LiveCodeBench.git lcb_runner_repo
    cd lcb_runner_repo && git checkout 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
    cd .. && ln -sfn lcb_runner_repo/lcb_runner lcb_runner
    cd ../../..
    pip install pebble datasets

Then run:
    python pipeline/scoring/score_traces_lcb_v6.py \
        --trace data/results/livecodebench_v6/mc128_thr128.jsonl \
        --meta pipeline/prompt/livecodebench_v6.meta.jsonl
"""
from __future__ import annotations

import argparse
import base64
import json
import pickle
import re
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_VENDORED = THIS_DIR / "vendored" / "lcb_runner"
DEFAULT_PRIVATE_TESTS = (
    THIS_DIR.parent / "prompts" / "livecodebench_v6.private_tests.pkl"
)

_RE_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_RE_PY_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_RE_ANY_FENCE = re.compile(r"```\s*\n(.*?)\n```", re.DOTALL)


def strip_thinking(response: str) -> str:
    """Remove Qwen3 <think>...</think> blocks before code extraction.

    Qwen3's reasoning often contains candidate solutions in fenced
    ```python``` blocks inside <think>.  extract_code() takes the LAST
    fence, so an unstripped response with a thinking candidate followed
    by the real answer would already work — but a response that rejects
    the thinking candidate and writes only prose outside would regress.
    Stripping first is the robust choice.

    If the response has an open <think> with no matching </think> (max_tokens
    truncation), we return "" — truncated reasoning has no real code answer.
    """
    if not response:
        return ""
    cleaned = _RE_THINK_BLOCK.sub("", response)
    if "<think>" in cleaned and "</think>" not in cleaned:
        return ""
    return cleaned.strip()


def extract_code(response: str) -> str:
    if not response:
        return ""
    m = _RE_PY_FENCE.findall(response)
    if m:
        return m[-1].strip()
    m = _RE_ANY_FENCE.findall(response)
    if m:
        return m[-1].strip()
    return response.strip()


def decode_private_tests(b64z: str | None) -> list[dict]:
    if not b64z:
        return []
    try:
        raw = base64.b64decode(b64z)
        decompressed = zlib.decompress(raw)
        obj = pickle.loads(decompressed)
    except Exception:
        return []
    if isinstance(obj, (bytes, bytearray)):
        try:
            obj = obj.decode("utf-8")
        except UnicodeDecodeError:
            return []
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        inputs = obj.get("inputs") or []
        outputs = obj.get("outputs") or []
        return [{"input": i, "output": o} for i, o in zip(inputs, outputs)]
    return []


def _build_input_output(meta: dict, private_b64z: str | None) -> str:
    public = meta.get("public_test_cases") or []
    private = decode_private_tests(private_b64z)

    inputs: list[Any] = []
    outputs: list[Any] = []
    for tc in (*public, *private):
        if "input" in tc and "output" in tc:
            inputs.append(tc["input"])
            outputs.append(tc["output"])

    io_dict: dict[str, Any] = {"inputs": inputs, "outputs": outputs}
    fn_name = meta.get("fn_name")
    if fn_name:
        io_dict["fn_name"] = fn_name
    return json.dumps(io_dict)


def _import_lcb(vendored_dir: Path):
    if not vendored_dir.exists():
        raise FileNotFoundError(
            f"Vendored lcb_runner dir not found: {vendored_dir}\n"
            f"See the header of this script for vendoring instructions."
        )
    repo_root = vendored_dir.parent
    sys.path.insert(0, str(repo_root))
    try:
        from lcb_runner.evaluation.testing_util import run_test
    except ImportError as e:
        raise ImportError(
            f"Failed to import vendored lcb_runner from {repo_root}. "
            f"Ensure lcb_runner/ is present (or symlinked) and all pip "
            f"deps (pebble, datasets) are installed. Original: {e}"
        ) from e
    return run_test


def _load_trace(path: Path) -> dict:
    with open(path) as f:
        text = f.read().lstrip()
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj


def _load_meta(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--private-tests", type=Path, default=DEFAULT_PRIVATE_TESTS,
                    help="Pickle sidecar {question_id: b64z_string} "
                         "(from prepare_prompts_lcb_v6.py).")
    ap.add_argument("--vendored", type=Path, default=DEFAULT_VENDORED)
    ap.add_argument("--out", type=Path,
                    help="Defaults to the trace file with .jsonl → .scores.json.")
    ap.add_argument("--timeout", type=int, default=6,
                    help="Per-test-case timeout in seconds (default 6, matches LCB).")
    ap.add_argument("--per-doc", action="store_true",
                    help="Include per-problem pass/fail in the output.")
    args = ap.parse_args()

    out = args.out or args.trace.with_suffix(".scores.json")
    run_test = _import_lcb(args.vendored)

    if not args.private_tests.exists():
        raise FileNotFoundError(
            f"private-tests sidecar not found: {args.private_tests}\n"
            f"Run `python pipeline/prompt/prepare_prompts_lcb_v6.py` first."
        )
    with open(args.private_tests, "rb") as f:
        private_tests: dict[str, str] = pickle.load(f)
    print(f"Loaded {len(private_tests)} private-test blobs from {args.private_tests}")

    trace = _load_trace(args.trace)
    meta = _load_meta(args.meta)

    generated = trace.get("generated_texts", [])
    errors = trace.get("errors", [None] * len(generated))

    n_trace = len(generated)
    n_meta = len(meta)
    n = min(n_trace, n_meta)
    if n_trace != n_meta:
        print(
            f"WARNING: trace has {n_trace} rows, meta has {n_meta}. "
            f"Scoring the first {n} rows only."
        )

    by_platform: dict[str, list[int]] = defaultdict(list)
    by_difficulty: dict[str, list[int]] = defaultdict(list)

    n_pass = 0
    n_failed_gen = 0
    n_failed_exec = 0
    records: list[dict] = []

    for i in range(n):
        m = meta[i]
        resp = generated[i] or ""
        err = errors[i] if i < len(errors) else None

        if err:
            n_failed_gen += 1
            passed = 0
            detail: dict[str, Any] = {"error": f"generation: {err}"}
        else:
            code = extract_code(strip_thinking(resp))
            if not code:
                n_failed_exec += 1
                passed = 0
                detail = {"error": "no_code_extracted"}
            else:
                qid = m.get("question_id") or f"row_{i}"
                sample = {"input_output": _build_input_output(m, private_tests.get(qid))}
                try:
                    result, metadata = run_test(sample, test=code, timeout=args.timeout)
                except Exception as e:
                    n_failed_exec += 1
                    passed = 0
                    detail = {"error": f"run_test_exception: {type(e).__name__}: {e}"}
                else:
                    all_pass = bool(result) and all(r is True for r in result)
                    passed = int(all_pass)
                    detail = {
                        "n_tests": len(result) if result else 0,
                        "n_pass": sum(1 for r in (result or []) if r is True),
                        "metadata": metadata if not all_pass else None,
                    }

        n_pass += passed
        by_platform[str(m.get("platform") or "unknown")].append(passed)
        by_difficulty[str(m.get("difficulty") or "unknown")].append(passed)

        records.append({
            "row_index": m.get("row_index"),
            "question_id": m.get("question_id"),
            "platform": m.get("platform"),
            "difficulty": m.get("difficulty"),
            "contest_date": m.get("contest_date"),
            "passed": passed,
            **detail,
        })

    def _summary(buckets: dict[str, list[int]]) -> dict:
        return {
            k: {"n": len(v), "pass@1": (sum(v) / len(v)) if v else 0.0}
            for k, v in sorted(buckets.items())
        }

    scores: dict[str, Any] = {
        "task": "livecodebench_v6",
        "trace": str(args.trace),
        "meta": str(args.meta),
        "n_total": n,
        "n_pass": n_pass,
        "n_failed_generation": n_failed_gen,
        "n_failed_execution": n_failed_exec,
        "pass@1": (n_pass / n) if n else 0.0,
        "pass@1_by_platform": _summary(by_platform),
        "pass@1_by_difficulty": _summary(by_difficulty),
        "timeout_s": args.timeout,
    }
    if args.per_doc:
        scores["per_doc"] = records

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(
        f"[{args.trace.name}] n={n} pass={n_pass} "
        f"fail_gen={n_failed_gen} fail_exec={n_failed_exec} "
        f"pass@1={scores['pass@1']:.4f} → {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
