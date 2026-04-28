"""Inspect a calib/sweep trace JSONL produced by bench_serving.

Works for any task — sharegpt, ifbench, supergpqa, livecodebench_v6, gsm8k, …

Six flags only:
    -n N         how many answers to show (default 1)
    --full       no truncation (default: each section capped ~500 chars)
    --row K      just that one row, in full (shortcut for `-n 1 --full` from row K)
    --errors     only rows that errored
    --raw        ZERO post-processing — dump the exact bytes bench_serving
                 received from the server. No truncation, no <think> split,
                 no framing. Pipe this into less/grep/editor.
    --rendered   additionally show the FULL rendered chat-template string
                 (with <|im_start|>…, and the pre-injected <think>\n\n</think>\n\n
                 when enable_thinking=False). Uses the model's own tokenizer.

Examples (paths relative to experiments/):
    python monitor/show_calib.py data/kv_calib/calib_ifbench_n16.jsonl            # summary + row 0
    python monitor/show_calib.py data/kv_calib/calib_ifbench_n16.jsonl -n 5       # first 5, truncated
    python monitor/show_calib.py data/kv_calib/calib_ifbench_n16.jsonl -n 5 --full   # first 5, full text
    python monitor/show_calib.py data/kv_calib/calib_ifbench_n16.jsonl --row 3    # row 3, full
    python monitor/show_calib.py data/kv_calib/calib_ifbench_n16.jsonl --errors   # only failures
    python monitor/show_calib.py data/kv_calib/calib_ifbench_n16.jsonl --row 1 --raw > row1.txt  # raw dump

Prompts + meta files are auto-joined from pipeline/prompt/<task>.jsonl and
pipeline/prompt/<task>.meta.jsonl when <task> can be inferred from the trace
filename (e.g. calib_ifbench_n16.jsonl → task=ifbench).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parent


def _load_trace(path: Path) -> dict:
    with open(path) as f:
        text = f.read().lstrip()
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _guess_task(trace_path: Path) -> tuple[str | None, str | None]:
    """Return (task, variant).  Variant is None when there's no suffix.

    calib_ifbench_n16          → ("ifbench", None)
    calib_ifbench_n16_think    → ("ifbench", "think")
    calib_ifbench_n16_nothink  → ("ifbench", "nothink")
    mc128_thr128               → ("thr128",  None)   (sweep-style name)
    """
    m = re.search(r"calib_([A-Za-z0-9_]+?)_n\d+(?:_([A-Za-z0-9]+))?$",
                  trace_path.stem)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"mc\d+_([A-Za-z0-9_]+)", trace_path.stem)
    if m:
        return m.group(1), None
    return None, None


def _auto_prompts(trace_path: Path) -> tuple[Path | None, Path | None]:
    task, variant = _guess_task(trace_path)
    if not task:
        return None, None
    pdir = EXPERIMENTS_DIR / "pipeline" / "prompt"
    # Prefer a variant-specific snapshot (e.g. ifbench_nothink.jsonl) when present.
    candidates: list[tuple[Path, Path]] = []
    if variant:
        candidates.append((pdir / f"{task}_{variant}.jsonl",
                           pdir / f"{task}_{variant}.meta.jsonl"))
    candidates.append((pdir / f"{task}.jsonl", pdir / f"{task}.meta.jsonl"))
    for p, m in candidates:
        if p.exists():
            return p, (m if m.exists() else None)
    return None, None


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    xs_s = sorted(xs)
    n = len(xs_s)

    def pct(q):
        k = max(0, min(n - 1, int(round((n - 1) * q))))
        return xs_s[k]

    return {
        "n": n,
        "min": xs_s[0],
        "mean": statistics.mean(xs_s),
        "std": statistics.pstdev(xs_s),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": xs_s[-1],
    }


def _fmt_stats(label: str, s: dict[str, float], unit: str = "") -> str:
    if not s:
        return f"  {label:18s} (empty)"
    return (
        f"  {label:18s} n={s['n']:<5d} min={s['min']:>8.1f}{unit} "
        f"mean={s['mean']:>9.1f}{unit}  std={s['std']:>8.1f}{unit}  "
        f"p50={s['p50']:>8.1f}{unit}  p95={s['p95']:>8.1f}{unit}  "
        f"p99={s['p99']:>8.1f}{unit}  max={s['max']:>8.1f}{unit}"
    )


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 20] + f"  …[+{len(s)-n} chars]"


def _print_summary(trace: dict):
    n_total = trace.get("completed") or len(trace.get("output_lens", []))
    errors = trace.get("errors") or []
    n_err = sum(1 for e in errors if e)
    print("=" * 72)
    print(f"  trace                {trace.get('output_file', '<inline>')}")
    print(f"  backend / dataset    {trace.get('backend')} / {trace.get('dataset_name')}")
    print(f"  completed / errors   {n_total} / {n_err}")
    dur = trace.get("duration")
    if dur:
        print(f"  duration             {dur:.1f}s  "
              f"({trace.get('request_throughput', 0):.2f} req/s, "
              f"{trace.get('output_throughput', 0):.1f} out tok/s)")
    print("=" * 72)

    input_lens = trace.get("input_lens") or []
    output_lens = trace.get("output_lens") or []
    ttfts = trace.get("ttfts") or []
    itls = trace.get("itls") or []
    if itls and isinstance(itls[0], list):
        # itls is a list-of-lists (per-request inter-token list)
        itls_flat = [x for seq in itls for x in seq]
    else:
        itls_flat = list(itls)

    print(_fmt_stats("input_len",  _stats(input_lens),  " tok"))
    print(_fmt_stats("output_len", _stats(output_lens), " tok"))
    if input_lens and output_lens:
        totals = [a + b for a, b in zip(input_lens, output_lens)]
        print(_fmt_stats("total_len",  _stats(totals),     " tok"))
    if ttfts:
        print(_fmt_stats("ttft",       _stats([t * 1000 for t in ttfts]), " ms"))
    if itls_flat:
        print(_fmt_stats("itl",        _stats([t * 1000 for t in itls_flat]), " ms"))
    print()


_TOKENIZER_CACHE: dict[str, object] = {}


def _render_prompt(prompt_row: dict, tokenizer) -> str:
    """Re-apply the chat template locally to reconstruct what the server saw.

    bench_serving doesn't save the rendered prompt — only `messages` +
    chat_template_kwargs — so we re-render it with the same tokenizer to
    show the exact string that got fed to the model (including any
    pre-injected <think>\\n\\n</think>\\n\\n when enable_thinking=False).
    """
    msgs = prompt_row.get("messages") or []
    tpl_kwargs = prompt_row.get("chat_template_kwargs") or {}
    return tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        **tpl_kwargs,
    )


def _load_tokenizer_from_trace(trace: dict):
    path = ((trace.get("server_info") or {}).get("tokenizer_path")
            or (trace.get("server_info") or {}).get("model_path"))
    if not path:
        raise RuntimeError(
            "Trace has no server_info.tokenizer_path/model_path — "
            "cannot auto-load a tokenizer for --rendered."
        )
    if path in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[path]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    _TOKENIZER_CACHE[path] = tok
    return tok


def _print_row(i: int, trace: dict, prompt_rows: list[dict],
               meta_rows: list[dict], preview_chars: int,
               tokenizer=None):
    n = len(trace.get("generated_texts") or [])
    if i < 0 or i >= n:
        print(f"(row {i} out of range 0..{n - 1})")
        return

    def _at(key: str, default=None):
        arr = trace.get(key)
        if isinstance(arr, list) and i < len(arr):
            return arr[i]
        return default

    gen = _at("generated_texts", "") or ""
    err = _at("errors")
    in_len = _at("input_lens")
    out_len = _at("output_lens")
    ttft = _at("ttfts")
    e2e = _at("e2e_latencies")
    if e2e is None and ttft is not None:
        itl_i = _at("itls") or []
        if isinstance(itl_i, list):
            e2e = ttft + sum(itl_i)

    print(f"----- row {i} -----")
    header = f"  in_len={in_len}  out_len={out_len}"
    if ttft is not None:
        header += f"  ttft={ttft * 1000:.0f}ms"
    if e2e is not None:
        header += f"  e2e={e2e:.1f}s"
    if err:
        header += f"  ERROR: {err}"
    print(header)

    if i < len(prompt_rows):
        prow = prompt_rows[i]
        msgs = prow.get("messages") or []
        for m in msgs:
            content = str(m.get("content", ""))
            tag = f"[{m.get('role','?')}]"
            snippet = _truncate(content, preview_chars)
            print(f"  prompt {tag:10s} {snippet}")
        if tokenizer is not None:
            rendered = _render_prompt(prow, tokenizer)
            print(f"  prompt [rendered] {_truncate(rendered, preview_chars)}")
    if i < len(meta_rows):
        meta = meta_rows[i]
        keep = {k: v for k, v in meta.items()
                if k not in {"prompt", "public_test_cases"}}
        print(f"  meta              {_truncate(json.dumps(keep, ensure_ascii=False), preview_chars)}")

    thinking, answer = _split_thinking(gen)
    if thinking is not None:
        print(f"  generated <think> {_truncate(thinking, preview_chars)}")
    print(f"  generated answer  {_truncate(answer, preview_chars)}")
    print()


def _split_thinking(text: str) -> tuple[str | None, str]:
    if "<think>" in text and "</think>" in text:
        start = text.index("<think>") + len("<think>")
        end = text.index("</think>")
        return text[start:end].strip(), text[end + len("</think>") :].strip()
    return None, text.strip()


DEFAULT_PREVIEW_CHARS = 500


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("trace", type=Path,
                    help="Path to a bench_serving trace JSONL.")
    ap.add_argument("-n", type=int, default=1, dest="n",
                    help="How many answers to show (default 1).")
    ap.add_argument("--full", action="store_true",
                    help="No truncation (default truncates each section to "
                         f"{DEFAULT_PREVIEW_CHARS} chars).")
    ap.add_argument("--row", type=int,
                    help="Show only this single row in full (shortcut).")
    ap.add_argument("--errors", action="store_true",
                    help="Show only rows that errored.")
    ap.add_argument("--raw", action="store_true",
                    help="Zero post-processing: dump the exact generated_texts "
                         "string(s) verbatim. No framing, no truncation, no "
                         "<think> split. Respects --row or -n.")
    ap.add_argument("--rendered", action="store_true",
                    help="Also show the FULL rendered chat-template string "
                         "(pre-injected <think>…</think> when thinking disabled). "
                         "Lazy-loads the model's tokenizer from server_info.")
    args = ap.parse_args()

    trace = _load_trace(args.trace)

    if args.raw:
        texts = trace.get("generated_texts") or []
        n_total = len(texts)
        if args.row is not None:
            idxs = [args.row] if 0 <= args.row < n_total else []
        else:
            idxs = list(range(min(args.n, n_total)))
        for j, i in enumerate(idxs):
            if j > 0:
                print(f"\n\x1e--- row {i} ---\x1e")  # RS control char as hard separator
            print(texts[i], end="")
        return 0

    prompts_path, meta_path = _auto_prompts(args.trace)
    prompt_rows = _load_jsonl(prompts_path) if prompts_path else []
    meta_rows = _load_jsonl(meta_path) if meta_path else []

    if prompts_path:
        print(f"(joined with prompts: {prompts_path})")
    if meta_path:
        print(f"(joined with meta:    {meta_path})")
    print()

    _print_summary(trace)

    tok = _load_tokenizer_from_trace(trace) if args.rendered else None

    if args.row is not None:
        _print_row(args.row, trace, prompt_rows, meta_rows, 10 ** 9,
                   tokenizer=tok)
        return 0

    preview = 10 ** 9 if args.full else DEFAULT_PREVIEW_CHARS
    n_total = len(trace.get("generated_texts") or [])

    if args.errors:
        errors = trace.get("errors") or []
        idxs = [i for i, e in enumerate(errors) if e]
        print(f"=== {len(idxs)} errored row(s) ===")
        for i in idxs[: args.n]:
            _print_row(i, trace, prompt_rows, meta_rows, preview, tokenizer=tok)
        return 0

    for i in range(min(args.n, n_total)):
        _print_row(i, trace, prompt_rows, meta_rows, preview, tokenizer=tok)
    if args.n < n_total:
        print(f"(showed {args.n} of {n_total}. pass `-n {n_total}` for all, "
              f"or `--row K` for one row full.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
