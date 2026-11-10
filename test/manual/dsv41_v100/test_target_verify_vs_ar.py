#!/usr/bin/env python3
"""Compare DSpark's 6-token main-model sign-off with the AR-matching next token.

Not CI. Talks to a live 8xV100 server (default http://127.0.0.1:11435). Import
and --self-test need no GPU and do not import sglang or torch.

Why this is not the stock KL helper
-----------------------------------
KLDivergenceMixin / kl_test_utils score the first *generated* token from the
last row of prompt prefill, then later tokens from decode, and compare both to
a fresh full prefill of prompt+generation. That split can look clean while
DSpark is still wrong.

DSpark drafts propose; the *full main model* signs off. That sign-off is one
ForwardMode.TARGET_VERIFY forward of T=speculative_num_draft_tokens (6 here,
classified as extend/prefill because T>1), not six one-token DECODE steps.
We keep that 6-token pass. This bench exists so its greedy token after the
bonus matches one-token AR.

Two independent conditions (see .claude/skills/kl-consistency-test/SKILL.md):
  1. Batch-invariance: a token's result must not depend on how many tokens
     share the forward (T=6 vs T=1).
  2. Same function: TARGET_VERIFY at position i given prefix+draft[:i] equals
     DECODE of that one next token.

This HTTP bench measures the product of both. It does not name Engram vs MoE
vs attention as the first disagreeing op.

Paths (same chat-tokenized prefix, main model only; draft dump is irrelevant)
---------------------------------------------------------------------------
Path A -- AR-matching teacher-force. POST /generate with
  input_ids = chat(prompt) + [bonus], max_new_tokens=1, temperature=0.
  The one new token is the last row of an EXTEND of that prefix. On this
  checkpoint that greedy token matches one-token AR (comma after No). It is
  *not* a DECODE step: HTTP max_new_tokens=1 never runs DECODE. True T=1
  DECODE of the token after No is the second output of a DSpark-off generate;
  we do not disable DSpark on the live unit.

Path B -- 6-token main-model TARGET_VERIFY. POST /generate with
  input_ids = chat(prompt), max_new_tokens=8, temperature=0, DSpark on.
  output_ids[0] is the prompt-prefill bonus (not verify).
  output_ids[1] / output_top_logprobs[1] is TARGET_VERIFY row 0 given that
  bonus -- the 6-token signer, not six AR decode steps.

HTTP limit: /generate returns logprobs for *accepted* tokens only. Rejected
draft positions in the T=6 window are not in the response. Alignment metric
is greedy fork + top-k / watched-id logprobs at the token after the bonus,
the same metric already used on this unit.

Golden unit: "Do old church windows flow downward because glass is a slow
liquid?" after token No (id 4484). PASS iff Path A and Path B agree on the
greedy next token and that token is comma (id 14), not emdash (id 965).

Example::

  PYTHONPATH=python python \\
    test/manual/dsv41_v100/test_target_verify_vs_ar.py \\
    --base http://127.0.0.1:11435
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:11435"
DEFAULT_OUT = "prefill-vs-ar-target-verify.json"

# Official DeepSeek-V4.1-Flash tokenizer ids used on this unit.
COMMA_ID = 14
PERIOD_ID = 16
EMDASH_ID = 965
NO_ID = 4484
WATCH_IDS = (COMMA_ID, PERIOD_ID, EMDASH_ID, NO_ID)
WATCH_NAME = {
    COMMA_ID: "comma",
    PERIOD_ID: "period",
    EMDASH_ID: "emdash",
    NO_ID: "No",
}

GLASS_WINDOWS = (
    "Do old church windows flow downward because glass is a slow liquid? "
    "Answer in a few short paragraphs."
)
BRAIN_10PCT = (
    "Is it true that humans only use 10% of their brain? "
    "Answer in a few short paragraphs."
)

PROMPT_SPECS: dict[str, dict[str, Any]] = {
    "glass-windows": {
        "prompt": GLASS_WINDOWS,
        "expected_bonus_id": NO_ID,
        "expected_next_id": COMMA_ID,
        "gates_exit": True,
    },
    "brain-10pct": {
        "prompt": BRAIN_10PCT,
        "expected_bonus_id": None,
        "expected_next_id": None,
        "gates_exit": False,
    },
}


def http_json(
    base: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    method: str | None = None,
    timeout: int = 600,
) -> Any:
    url = base.rstrip("/") + path
    if payload is None:
        req = urllib.request.Request(url, method=method or "GET")
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method=method or "POST",
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw.decode("utf-8", errors="replace")


def flush(base: str) -> None:
    http_json(base, "/flush_cache?timeout=30", method="POST", timeout=60)


def tokenize_chat(base: str, prompt: str) -> list[int]:
    out = http_json(
        base,
        "/v1/tokenize",
        {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    if not isinstance(out, dict):
        raise RuntimeError(f"tokenize failed: {out!r}")
    toks = out.get("tokens")
    if toks and isinstance(toks[0], list):
        toks = toks[0]
    if not toks:
        raise RuntimeError(f"tokenize returned no tokens: {out!r}")
    return [int(x) for x in toks]


def generate(
    base: str,
    input_ids: list[int],
    *,
    max_new_tokens: int,
    top_logprobs_num: int = 8,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "return_text_in_logprobs": True,
        "logprob_start_len": -1,
        "top_logprobs_num": top_logprobs_num,
        "token_ids_logprob": list(WATCH_IDS),
    }
    out = http_json(base, "/generate", payload)
    if isinstance(out, list):
        out = out[0]
    if not isinstance(out, dict):
        raise RuntimeError(f"generate failed: {out!r}")
    return out


def lp_triple(item: Any) -> tuple[float | None, int | None, str | None]:
    if item is None:
        return None, None, None
    lp = item[0] if len(item) > 0 else None
    tid = item[1] if len(item) > 1 else None
    txt = item[2] if len(item) > 2 else None
    return (
        None if lp is None else float(lp),
        None if tid is None else int(tid),
        None if txt is None else str(txt),
    )


def topk_rows(entries: Any, k: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in entries or []:
        lp, tid, txt = lp_triple(item)
        if lp is None or tid is None:
            continue
        rows.append({"logprob": lp, "id": tid, "text": txt})
    rows.sort(key=lambda r: r["logprob"], reverse=True)
    return rows[:k]


def logprob_for_id(entries: Any, token_id: int) -> float | None:
    for item in entries or []:
        lp, tid, _txt = lp_triple(item)
        if tid == token_id:
            return lp
    return None


def k3(log_p: float, log_q: float) -> float:
    logr = log_p - log_q
    return math.exp(logr) - 1.0 - logr


def piece_repr(text: str | None, token_id: int | None) -> str:
    if text:
        return json.dumps(text, ensure_ascii=False)
    if token_id is None:
        return "None"
    return f"id={token_id}"


def spec_snapshot(meta: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "spec_verify_ct",
        "spec_accept_length",
        "spec_num_proposed_drafts",
        "spec_num_correct_drafts",
        "spec_accepted_drafts",
        "completion_tokens",
        "cached_tokens",
    )
    return {k: meta.get(k) for k in keys}


def path_b_ran_verify(meta: dict[str, Any]) -> bool:
    ct = meta.get("spec_verify_ct")
    if isinstance(ct, (int, float)) and ct > 0:
        return True
    acc = meta.get("spec_accept_length")
    return isinstance(acc, (int, float))


def position_record(
    *,
    label: str,
    output_ids: list[int],
    meta: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    out_lp = meta.get("output_token_logprobs") or []
    out_top = meta.get("output_top_logprobs") or []
    out_watch = meta.get("output_token_ids_logprobs") or []
    tid = output_ids[index] if index < len(output_ids) else None
    sampled = lp_triple(out_lp[index] if index < len(out_lp) else None)
    top = topk_rows(out_top[index] if index < len(out_top) else None, 3)
    watch_src = out_watch[index] if index < len(out_watch) else None
    if not watch_src:
        watch_src = out_top[index] if index < len(out_top) else None
    watch = {WATCH_NAME[i]: logprob_for_id(watch_src, i) for i in WATCH_IDS}
    greedy_top = top[0] if top else None
    return {
        "label": label,
        "index": index,
        "greedy_id": tid,
        "greedy_text": sampled[2],
        "greedy_logprob": sampled[0],
        "top3": top,
        "watch": watch,
        "top1_id": None if greedy_top is None else greedy_top["id"],
    }


def compare_watch(
    a: dict[str, float | None], b: dict[str, float | None]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("comma", "period", "emdash"):
        lp_a, lp_b = a.get(name), b.get(name)
        row: dict[str, Any] = {"path_a": lp_a, "path_b": lp_b}
        if lp_a is not None and lp_b is not None:
            row["path_b_minus_path_a"] = lp_b - lp_a
            row["k3"] = k3(lp_a, lp_b)
        out[name] = row
    for side, watch in (("path_a", a), ("path_b", b)):
        c, e = watch.get("comma"), watch.get("emdash")
        out[f"{side}_comma_minus_emdash"] = (
            None if c is None or e is None else c - e
        )
    return out


def glass_verdict(
    *,
    bonus_id: int | None,
    path_a_id: int | None,
    path_b_id: int | None,
    expected_bonus_id: int | None,
    expected_next_id: int | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if expected_bonus_id is not None and bonus_id != expected_bonus_id:
        reasons.append(
            f"bonus id={bonus_id} expected {expected_bonus_id} ({WATCH_NAME.get(expected_bonus_id, '?')})"
        )
    if expected_next_id is not None and path_a_id != expected_next_id:
        reasons.append(
            f"Path A greedy id={path_a_id} expected {expected_next_id} "
            f"({WATCH_NAME.get(expected_next_id, '?')})"
        )
    if expected_next_id is not None and path_b_id != expected_next_id:
        reasons.append(
            f"Path B greedy id={path_b_id} expected {expected_next_id} "
            f"({WATCH_NAME.get(expected_next_id, '?')})"
        )
    if path_a_id is None or path_b_id is None:
        reasons.append("missing greedy id on Path A or Path B")
    elif path_a_id != path_b_id:
        reasons.append(
            f"Path A and Path B disagree: {piece_repr(None, path_a_id)} vs "
            f"{piece_repr(None, path_b_id)}"
        )
    # Dedupe while keeping order.
    seen: set[str] = set()
    uniq: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return (len(uniq) == 0), uniq


def server_snapshot(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "speculative_algorithm": info.get("speculative_algorithm"),
        "speculative_num_draft_tokens": info.get("speculative_num_draft_tokens"),
        "speculative_attention_mode": info.get("speculative_attention_mode"),
        "disable_radix_cache": info.get("disable_radix_cache"),
        "model_path": info.get("model_path"),
        "tp_size": info.get("tp_size"),
        "page_size": info.get("page_size"),
    }


def require_dspark_server(snap: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    algo = str(snap.get("speculative_algorithm") or "").upper()
    if algo != "DSPARK":
        errs.append(
            f"speculative_algorithm={snap.get('speculative_algorithm')!r}; "
            "Path B needs a DSpark-on server so token 1 is TARGET_VERIFY, "
            "not T=1 DECODE"
        )
    n = snap.get("speculative_num_draft_tokens")
    if n is not None and int(n) <= 1:
        errs.append(
            f"speculative_num_draft_tokens={n}; TARGET_VERIFY width must be T>1"
        )
    return errs


def run_one(
    base: str,
    name: str,
    spec: dict[str, Any],
    *,
    path_b_new_tokens: int,
) -> dict[str, Any]:
    prompt = spec["prompt"]
    print(f"\n===== {name} =====", flush=True)
    flush(base)
    prompt_ids = tokenize_chat(base, prompt)
    print(f"prompt_tokens={len(prompt_ids)}", flush=True)

    # Path B first so we can pin Path A's bonus to the live DSpark bonus token
    # when the spec does not hard-code it. Glass-windows still asserts No.
    t0 = time.time()
    path_b_raw = generate(base, prompt_ids, max_new_tokens=path_b_new_tokens)
    dt_b = time.time() - t0
    b_meta = path_b_raw.get("meta_info") or {}
    b_ids = [int(x) for x in (path_b_raw.get("output_ids") or [])]
    bonus_id = b_ids[0] if b_ids else None
    print(
        f"Path B generate n={path_b_new_tokens} {dt_b:.1f}s "
        f"text={path_b_raw.get('text', '')[:80]!r} ids={b_ids[:8]}",
        flush=True,
    )
    if not path_b_ran_verify(b_meta):
        raise RuntimeError(
            f"{name}: Path B meta has no spec_verify_ct / spec_accept_length; "
            "this request did not run TARGET_VERIFY. Is DSpark on?"
        )

    # Same prefix as Path B: teacher-force the live bonus, then generate 1.
    # Glass-windows still fails the verdict if that bonus is not No.
    teacher_bonus = bonus_id
    if teacher_bonus is None:
        raise RuntimeError(f"{name}: no bonus token for Path A teacher-force")

    flush(base)
    t0 = time.time()
    path_a_raw = generate(
        base, prompt_ids + [int(teacher_bonus)], max_new_tokens=1
    )
    dt_a = time.time() - t0
    a_meta = path_a_raw.get("meta_info") or {}
    a_ids = [int(x) for x in (path_a_raw.get("output_ids") or [])]
    print(
        f"Path A teacher-force bonus={teacher_bonus} n=1 {dt_a:.1f}s "
        f"text={path_a_raw.get('text', '')[:80]!r} ids={a_ids[:2]}",
        flush=True,
    )
    if path_b_ran_verify(a_meta):
        print(
            "warning: Path A unexpectedly has spec verify fields; "
            "max_new_tokens=1 should be extend last-row, not TARGET_VERIFY",
            flush=True,
        )

    a_pos = position_record(
        label="path_a_extend_last_row_after_bonus",
        output_ids=a_ids,
        meta=a_meta,
        index=0,
    )
    b_bonus = position_record(
        label="path_b_prompt_prefill_bonus",
        output_ids=b_ids,
        meta=b_meta,
        index=0,
    )
    b_verify_positions = []
    for i in range(1, min(7, len(b_ids))):
        label = (
            "path_b_target_verify_row0_after_bonus"
            if i == 1
            else f"path_b_accepted_token_{i}"
        )
        b_verify_positions.append(
            position_record(label=label, output_ids=b_ids, meta=b_meta, index=i)
        )
    b_after = b_verify_positions[0] if b_verify_positions else None
    path_a_id = a_pos.get("greedy_id")
    path_b_id = None if b_after is None else b_after.get("greedy_id")
    ok, reasons = glass_verdict(
        bonus_id=bonus_id,
        path_a_id=path_a_id,
        path_b_id=path_b_id,
        expected_bonus_id=spec.get("expected_bonus_id"),
        expected_next_id=spec.get("expected_next_id"),
    )
    # Optional prompts: agreement-only (no pinned next token).
    if spec.get("expected_next_id") is None:
        ok = path_a_id is not None and path_a_id == path_b_id
        reasons = [] if ok else [
            f"Path A and Path B disagree: {piece_repr(a_pos.get('greedy_text'), path_a_id)} vs "
            f"{piece_repr(None if b_after is None else b_after.get('greedy_text'), path_b_id)}"
        ]

    watch = compare_watch(
        a_pos.get("watch") or {},
        (b_after or {}).get("watch") or {},
    )
    row = {
        "name": name,
        "prompt": prompt,
        "prompt_tokens": len(prompt_ids),
        "gates_exit": bool(spec.get("gates_exit")),
        "pass": ok,
        "reasons": reasons,
        "path_a_s": round(dt_a, 2),
        "path_b_s": round(dt_b, 2),
        "path_a_text": path_a_raw.get("text") or "",
        "path_b_text": path_b_raw.get("text") or "",
        "bonus_id": bonus_id,
        "path_a": a_pos,
        "path_b_bonus": b_bonus,
        "path_b_after_bonus": b_after,
        "path_b_accepted_positions": b_verify_positions,
        "watch_compare": watch,
        "path_a_spec": spec_snapshot(a_meta),
        "path_b_spec": spec_snapshot(b_meta),
        "http_note": (
            "Path B logprobs are accepted tokens only. Index 0 is the "
            "prompt-prefill bonus; index 1 is TARGET_VERIFY row 0. Unused "
            "draft rows in the T=6 window are not returned by /generate."
        ),
    }
    status = "PASS" if ok else "FAIL"
    print(
        f"{status} {name} after bonus {piece_repr(b_bonus.get('greedy_text'), bonus_id)}: "
        f"path_a={piece_repr(a_pos.get('greedy_text'), path_a_id)} "
        f"path_b={piece_repr(None if b_after is None else b_after.get('greedy_text'), path_b_id)} "
        f"comma-emdash gap a={watch.get('path_a_comma_minus_emdash')} "
        f"b={watch.get('path_b_comma_minus_emdash')}",
        flush=True,
    )
    return row


def print_one_line(rows: list[dict[str, Any]], overall: bool) -> None:
    gated = [r for r in rows if r.get("gates_exit")]
    if not gated:
        gated = rows
    parts = ["PASS" if overall else "FAIL"]
    for r in gated:
        a = r.get("path_a") or {}
        b = r.get("path_b_after_bonus") or {}
        parts.append(
            f"{r['name']}: A={piece_repr(a.get('greedy_text'), a.get('greedy_id'))} "
            f"B={piece_repr(b.get('greedy_text'), b.get('greedy_id'))}"
        )
    print("ONE_LINE " + " | ".join(parts), flush=True)


def self_test() -> None:
    assert lp_triple([ -1.0, 14, "," ]) == (-1.0, 14, ",")
    assert abs(k3(-1.041, -1.041)) < 1e-12
    tops = topk_rows(
        [[-1.2, 14, ","], [-0.8, 965, "—"], [-1.3, 16, "."]], 3
    )
    assert tops[0]["id"] == 965
    ok, reasons = glass_verdict(
        bonus_id=NO_ID,
        path_a_id=COMMA_ID,
        path_b_id=COMMA_ID,
        expected_bonus_id=NO_ID,
        expected_next_id=COMMA_ID,
    )
    assert ok and not reasons, reasons
    bad, reasons = glass_verdict(
        bonus_id=NO_ID,
        path_a_id=COMMA_ID,
        path_b_id=EMDASH_ID,
        expected_bonus_id=NO_ID,
        expected_next_id=COMMA_ID,
    )
    assert not bad and reasons
    assert not path_b_ran_verify({})
    assert path_b_ran_verify({"spec_verify_ct": 1})
    assert path_b_ran_verify({"spec_accept_length": 2.0})
    print("self-test ok", flush=True)


def parse_prompt_names(raw: str) -> list[str]:
    names = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [n for n in names if n not in PROMPT_SPECS]
    if unknown:
        known = ", ".join(PROMPT_SPECS)
        raise argparse.ArgumentTypeError(
            f"unknown prompt {unknown}; known: {known}"
        )
    if not names:
        raise argparse.ArgumentTypeError("need at least one prompt name")
    return names


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base", default=DEFAULT_BASE, help="SGLang HTTP base URL")
    p.add_argument(
        "--prompts",
        type=parse_prompt_names,
        default=["glass-windows", "brain-10pct"],
        help="Comma-separated prompt names (glass-windows, brain-10pct).",
    )
    p.add_argument(
        "--path-b-new-tokens",
        type=int,
        default=8,
        help="Tokens to generate on Path B (bonus + TARGET_VERIFY accepted tokens).",
    )
    p.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="JSON report path (outside the repo is fine).",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run parser/verdict checks and exit (no HTTP, no GPU).",
    )
    args = p.parse_args(argv)
    self_test()
    if args.self_test:
        return 0

    try:
        with urllib.request.urlopen(args.base.rstrip("/") + "/health", timeout=5) as r:
            health = r.status
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"health fail at {args.base}: {e}", flush=True)
        return 2
    print(f"health {health}", flush=True)

    info = http_json(args.base, "/server_info", timeout=30)
    if not isinstance(info, dict):
        print(f"server_info failed: {info!r}", flush=True)
        return 2
    snap = server_snapshot(info)
    print(json.dumps({"server": snap}, ensure_ascii=False), flush=True)
    cfg_errs = require_dspark_server(snap)
    if cfg_errs:
        for e in cfg_errs:
            print(f"config fail: {e}", flush=True)
        return 2

    rows: list[dict[str, Any]] = []
    try:
        for name in args.prompts:
            rows.append(
                run_one(
                    args.base,
                    name,
                    PROMPT_SPECS[name],
                    path_b_new_tokens=args.path_b_new_tokens,
                )
            )
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as e:
        print(f"run fail: {e}", flush=True)
        report = {
            "pass": False,
            "error": str(e),
            "server": snap,
            "results": rows,
        }
        _write_report(args.out, report)
        return 2

    gated = [r for r in rows if r.get("gates_exit")]
    overall = all(r.get("pass") for r in gated) if gated else all(
        r.get("pass") for r in rows
    )
    report = {
        "pass": overall,
        "server": snap,
        "leftover_ops_not_isolated": [
            "Engram MODE_VERIFY vs MODE_DECODE vs MODE_EXTEND",
            "MoE expert GEMM / router at M=6 vs M=1",
            "CSA2 T-wide prefill-shaped TARGET_VERIFY vs T=1 DECODE",
            "any other leftover op on the main-model forward",
        ],
        "http_limit": (
            "No per-position unused-draft TARGET_VERIFY logits. "
            "Metric is accepted-token greedy + top-k / watched-id logprobs."
        ),
        "results": rows,
        "out": args.out,
    }
    _write_report(args.out, report)
    print_one_line(rows, overall)
    print(f"wrote {args.out}", flush=True)
    return 0 if overall else 1


def _write_report(path: str, report: dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except OSError as e:
        print(f"warning: could not write {path}: {e}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
