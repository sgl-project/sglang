#!/usr/bin/env python3
"""30-minute token-generation soak against a live DSV4.1-Flash server.

The CPU/CUDA unit benches score a 30k-length tensor once (~10 s). The
cohesion collapse on this box showed up after ~30 min of slow TG
(~8 tok/s with DSpark). This file is that duration: multi-turn
/v1/chat/completions (optional /v1/messages), growing context, hundreds
of decode steps, DSpark accept length per turn.

Not CI. Needs the live 8×V100 unit (default http://127.0.0.1:11435).
Occupies np=1 for ``--seconds`` (default 1800).

Example::

  PYTHONPATH=python python \\
    test/manual/dsv41_v100/test_tg_soak_cohesion.py \\
    --base http://127.0.0.1:11435 --seconds 1800
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:11435"
DEFAULT_SECONDS = 1800
DEFAULT_MAX_TOKENS = 768
DEFAULT_OUT = "tg-soak-cohesion.jsonl"

_WORD_RE = re.compile(r"[A-Za-z0-9_`'.]+|[.!?]")

FACTS = (
    "Keep these distinctions. They are not the same object.\n"
    "FACT A: The paper is Conrey–Ghosh–Gonek. It is a real paper. "
    "thmB in the paper is a 1/2-density result.\n"
    "FACT B: Challenge.lean states two_thirds_simple_on_critical_line as 2/3. "
    "That is a stronger claim than the paper.\n"
    "FACT C: Solution.lean is a proof of the stronger 2/3 claim. "
    "Final.lean is the weaker 1/2 statement. Final.lean matches the paper. "
    "Final.lean is not Challenge.lean.\n"
)

ASK = (
    "In one short paragraph, with no repetition: what does Challenge.lean "
    "claim, what does Final.lean prove, and how does each relate to "
    "Conrey–Ghosh–Gonek? Use the FACT A/B/C labels."
)

CONTINUE = (
    "Continue the working note. Keep FACT A/B/C straight. "
    "Do not invent a loop. Name the next lemma you would prove and why."
)


def filler_block(turn: int, n_lines: int = 72) -> str:
    """Unique numbered lines so context grows like a grep dump, not a copy."""
    rows = [
        f"Turn {turn} workspace dump (each line is a distinct lemma stub):"
    ]
    for i in range(n_lines):
        ident = turn * 1000 + i
        rows.append(
            f"src/mod_{turn:02d}/file_{i:03d}.lean:{i+1}: "
            f"lemma L{turn}_{i} (n : Nat) : n + {ident} = {ident} + n := by "
            f"sorry -- uid={ident} name=PrimeGap.{turn}.{i}"
        )
    return "\n".join(rows)


def tokenize_text(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def end_cycle(
    tokens: list[str],
    *,
    min_period: int = 2,
    max_period: int = 16,
    min_repeats: int = 6,
    tail: int = 256,
) -> dict[str, Any] | None:
    """Exact token cycle stuck at the end (``The file. The file. …``)."""
    if len(tokens) < min_period * min_repeats:
        return None
    window = tokens[-tail:]
    for period in range(min_period, max_period + 1):
        cycle = window[-period:]
        repeats = 0
        i = len(window)
        while i >= period and window[i - period : i] == cycle:
            repeats += 1
            i -= period
        if repeats >= min_repeats:
            return {
                "period": period,
                "repeats": repeats,
                "cycle": cycle,
            }
    return None


def ngram_coverage(
    tokens: list[str], n: int, window: int = 128
) -> tuple[float, tuple[str, ...] | None]:
    w = tokens[-window:]
    if len(w) < n + 4:
        return 0.0, None
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(w) - n + 1):
        g = tuple(w[i : i + n])
        counts[g] = counts.get(g, 0) + 1
    best, c = max(counts.items(), key=lambda kv: kv[1])
    return c / max(len(w) - n + 1, 1), best


def unique_ratio(tokens: list[str], window: int = 128) -> float:
    w = tokens[-window:]
    if not w:
        return 1.0
    return len(set(t.lower() for t in w)) / len(w)


def score_generation(text: str) -> dict[str, Any]:
    tokens = tokenize_text(text)
    cycle = end_cycle(tokens)
    cov2, g2 = ngram_coverage(tokens, 2)
    cov3, g3 = ngram_coverage(tokens, 3)
    uniq = unique_ratio(tokens)
    locked = bool(cycle) or cov3 >= 0.35 or (cov2 >= 0.45 and uniq < 0.20)
    return {
        "n_tokens": len(tokens),
        "unique_ratio_last128": round(uniq, 4),
        "bigram_coverage": round(cov2, 4),
        "bigram": None if g2 is None else list(g2),
        "trigram_coverage": round(cov3, 4),
        "trigram": None if g3 is None else list(g3),
        "end_cycle": cycle,
        "locked": locked,
    }


def skeleton(text: str) -> str:
    """Drop turn/uid/index numbers so a rubber-stamp reply still matches."""
    t = text.lower()
    t = re.sub(r"\d+", "N", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def skeleton_ratio(a: str, b: str) -> float:
    sa, sb = skeleton(a), skeleton(b)
    if not sa or not sb:
        return 0.0
    return SequenceMatcher(None, sa, sb).ratio()


def is_ask_turn(turn: int) -> bool:
    return turn > 0 and turn % 4 == 0


def stamp_report(texts: list[str]) -> dict[str, Any]:
    """Consecutive continue-turns that are the same note with new digits."""
    pairs: list[dict[str, Any]] = []
    streak = 0
    max_streak = 0
    first = None
    for i in range(1, len(texts)):
        if is_ask_turn(i) or is_ask_turn(i - 1):
            streak = 0
            continue
        ratio = skeleton_ratio(texts[i - 1], texts[i])
        stamped = ratio >= 0.88
        pairs.append({"turn": i, "vs": i - 1, "ratio": round(ratio, 4), "stamp": stamped})
        if stamped:
            streak += 1
            if first is None:
                first = i
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {
        "pairs": pairs,
        "max_continue_stamp_streak": max_streak,
        "first_stamp_turn": first,
        "stamped": max_streak >= 3,
    }


JUDGE_INSTRUCTION = """\
You are judging ONE assistant reply from a long coding session.
Do not continue the working note. Score the reply against the ground facts.

Ground facts (these are distinct objects):
""" + FACTS + """
User move this turn: {move}

Previous assistant reply (may be empty):
---
{prev}
---

Current assistant reply:
---
{cur}
---

Return a single JSON object, no markdown fences, keys exactly:
  facts_correct: bool  (if A/B/C are mentioned, they are not mixed up)
  mode: "ok" | "template_stamp" | "contradiction" | "ngram_loop" | "incoherent"
  new_content: bool  (said something the previous reply did not, besides substituted numbers)
  severity: "none" | "minor" | "serious"
  rationale: one sentence
Rules:
- template_stamp = same scaffold as the previous reply, only indices/uids changed.
  That is a cohesion failure even if facts_correct is true.
- contradiction = states A/B/C in a way that cannot all be true.
- ngram_loop = a short phrase repeating as if stuck.
- serious if template_stamp, contradiction, ngram_loop, or incoherent.
- none only if the reply is a real next step (or a correct ASK answer) with intact facts.
"""


def extract_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.S)
    if fence:
        raw = fence.group(1)
    else:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    return json.loads(raw)


def judge_reply(
    base: str,
    *,
    turn: int,
    prev: str,
    cur: str,
) -> dict[str, Any]:
    move = "ASK (restate Challenge vs Final vs the paper)" if is_ask_turn(turn) else "CONTINUE the working note"
    prompt = JUDGE_INSTRUCTION.format(move=move, prev=prev[-1500:] or "(none)", cur=cur[-2500:])
    out = chat_completions(
        base,
        [{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    text = content_chat(out)
    try:
        parsed = extract_json(text)
    except (json.JSONDecodeError, ValueError) as e:
        return {"parse_error": str(e), "raw": text[:800], "severity": "unknown"}
    return parsed


def self_test() -> None:
    loop = "The file. The file. " * 24
    s = score_generation(loop)
    assert s["locked"], s
    assert s["end_cycle"] is not None, s
    clean = (
        "Challenge.lean claims 2/3. Final.lean proves 1/2 and matches the "
        "Conrey–Ghosh–Gonek paper. Solution.lean is the stronger proof."
    )
    c = score_generation(clean)
    assert not c["locked"], c
    varying = (
        "And the paper is Conrey–Ghosh–Gonek. And the paper is a real paper. "
        "And the paper is a real result. And Challenge.lean is 2/3."
    )
    v = score_generation(varying)
    assert not v["locked"], v
    a = (
        "Working note, Turn 9. A real, thmB = 1/2. B Challenge.lean 2/3. "
        "L9_72 uid=9072 n + 9072 = 9072 + n. omega. No 2/3."
    )
    b = (
        "Working note, Turn 10. A real, thmB = 1/2. B Challenge.lean 2/3. "
        "L10_72 uid=10072 n + 10072 = 10072 + n. omega. No 2/3."
    )
    assert skeleton_ratio(a, b) >= 0.88, skeleton_ratio(a, b)
    assert skeleton_ratio(clean, loop) < 0.5
    print("self-test ok", flush=True)


def http_json(
    base: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    method: str | None = None,
    timeout: int = 1200,
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


def server_up(base: str) -> bool:
    try:
        http_json(base, "/health", timeout=10)
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def spec_of(out: dict[str, Any]) -> dict[str, Any]:
    choice0 = (out.get("choices") or [{}])[0]
    sglext = out.get("sglext") or choice0.get("sglext") or {}
    spec = sglext.get("spec_tokens_details") or {}
    if isinstance(spec, list):
        spec = spec[0] if spec else {}
    meta = (out.get("meta_info") or {}) if isinstance(out.get("meta_info"), dict) else {}
    if not spec and meta:
        spec = {
            k: meta[k]
            for k in (
                "spec_accept_length",
                "spec_accept_rate",
                "spec_verify_ct",
                "spec_num_correct_drafts",
            )
            if k in meta
        }
    return spec if isinstance(spec, dict) else {}


def content_chat(out: dict[str, Any]) -> str:
    msg = ((out.get("choices") or [{}])[0]).get("message") or {}
    return msg.get("content") or ""


def content_messages(out: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in out.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
        elif isinstance(block, dict) and block.get("type") == "thinking":
            parts.append(block.get("thinking") or "")
    return "".join(parts)


def chat_completions(
    base: str, messages: list[dict[str, str]], *, max_tokens: int, temperature: float
) -> dict[str, Any]:
    payload = {
        "model": "default",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "return_spec_tokens_details": True,
    }
    out = http_json(base, "/v1/chat/completions", payload)
    if not isinstance(out, dict):
        raise RuntimeError(f"chat.completions failed: {out!r}")
    return out


def anthropic_messages(
    base: str, messages: list[dict[str, str]], *, max_tokens: int, temperature: float
) -> dict[str, Any]:
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    out = http_json(base, "/v1/messages", payload)
    if not isinstance(out, dict):
        raise RuntimeError(f"messages failed: {out!r}")
    return out


def prompt_tokens_via_tokenize(base: str, messages: list[dict[str, str]]) -> int | None:
    try:
        out = http_json(
            base,
            "/v1/tokenize",
            {"model": "default", "messages": messages},
            timeout=60,
        )
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError):
        return None
    if not isinstance(out, dict):
        return None
    toks = out.get("tokens") or out.get("count")
    if isinstance(toks, int):
        return toks
    if toks and isinstance(toks[0], list):
        toks = toks[0]
    if isinstance(toks, list):
        return len(toks)
    return None


def next_user_turn(turn: int) -> str:
    if turn == 0:
        return FACTS + "\nRead this and keep the distinctions straight. Start a working note.\n"
    if turn % 4 == 0:
        return filler_block(turn) + "\n\n" + ASK
    return filler_block(turn) + "\n\n" + CONTINUE


def run_soak(
    *,
    base: str,
    seconds: int,
    max_tokens: int,
    api: str,
    temperature: float,
    jsonl_path: str,
) -> dict[str, Any]:
    if not server_up(base):
        raise SystemExit(f"server not reachable at {base} (GET /health)")

    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    messages: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    t_end = time.time() + seconds
    turn = 0
    first_lock: dict[str, Any] | None = None

    print(
        f"tg-soak start base={base} api={api} seconds={seconds} "
        f"max_tokens={max_tokens} temp={temperature}",
        flush=True,
    )
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        while time.time() < t_end:
            remaining = t_end - time.time()
            messages.append({"role": "user", "content": next_user_turn(turn)})
            n_prompt = prompt_tokens_via_tokenize(base, messages)
            t0 = time.time()
            if api == "messages":
                out = anthropic_messages(
                    base, messages, max_tokens=max_tokens, temperature=temperature
                )
                text = content_messages(out)
                usage = out.get("usage") or {}
                n_out = int(usage.get("output_tokens") or 0)
                spec = spec_of(out)
                finish = out.get("stop_reason")
            else:
                out = chat_completions(
                    base, messages, max_tokens=max_tokens, temperature=temperature
                )
                text = content_chat(out)
                usage = out.get("usage") or {}
                n_out = int(usage.get("completion_tokens") or 0)
                spec = spec_of(out)
                finish = ((out.get("choices") or [{}])[0]).get("finish_reason")
            dt = time.time() - t0
            score = score_generation(text)
            accept = spec.get("spec_accept_length")
            row = {
                "turn": turn,
                "kind": "ask" if is_ask_turn(turn) else "continue",
                "elapsed_s": round(dt, 2),
                "remaining_s": round(t_end - time.time(), 1),
                "prompt_tokens": n_prompt,
                "completion_tokens": n_out,
                "wall_tok_s": None if not n_out or dt <= 0 else round(n_out / dt, 3),
                "finish_reason": finish,
                "spec_accept_length": accept,
                "spec": spec,
                "score": score,
                "dspark_amplified": bool(
                    score["locked"] and accept is not None and float(accept) >= 5.0
                ),
                "text": text,
                "text_tail": text[-1200:],
            }
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(
                json.dumps(
                    {
                        "turn": turn,
                        "prompt_tokens": n_prompt,
                        "completion_tokens": n_out,
                        "elapsed_s": row["elapsed_s"],
                        "tok_s": row["wall_tok_s"],
                        "locked": score["locked"],
                        "kind": row["kind"],
                        "unique": score["unique_ratio_last128"],
                        "trigram": score["trigram"],
                        "trigram_coverage": score["trigram_coverage"],
                        "end_cycle": score["end_cycle"],
                        "spec_accept_length": accept,
                        "dspark_amplified": row["dspark_amplified"],
                        "remaining_s": row["remaining_s"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if score["locked"] and first_lock is None:
                first_lock = {"turn": turn, **row}
                print(f"LOCK at turn={turn} (continuing until deadline)", flush=True)
            messages.append({"role": "assistant", "content": text or " "})
            turn += 1
            if remaining < 5:
                break

    texts = [r.get("text") or r.get("text_tail") or "" for r in rows]
    stamps = stamp_report(texts)
    n_lock = sum(1 for r in rows if r["score"]["locked"])
    n_amp = sum(1 for r in rows if r["dspark_amplified"])
    summary = {
        "turns": len(rows),
        "seconds_requested": seconds,
        "locked_turns": n_lock,
        "dspark_amplified_turns": n_amp,
        "first_lock_turn": None if first_lock is None else first_lock["turn"],
        "stamp": {
            "stamped": stamps["stamped"],
            "max_continue_stamp_streak": stamps["max_continue_stamp_streak"],
            "first_stamp_turn": stamps["first_stamp_turn"],
        },
        "prompt_tokens_last": None if not rows else rows[-1]["prompt_tokens"],
        "jsonl": jsonl_path,
        "pass": n_lock == 0 and not stamps["stamped"],
    }
    print(json.dumps({"stamp_pairs": stamps["pairs"]}, ensure_ascii=False), flush=True)
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    return summary


def load_turn_rows(jsonl_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            if "turn" in o:
                rows.append(o)
    return rows


def rejudge(base: str, jsonl_path: str, *, with_llm: bool) -> dict[str, Any]:
    rows = load_turn_rows(jsonl_path)
    texts = [r.get("text") or r.get("text_tail") or "" for r in rows]
    stamps = stamp_report(texts)
    print(json.dumps({"stamp_pairs": stamps["pairs"]}, ensure_ascii=False), flush=True)
    judges: list[dict[str, Any]] = []
    if with_llm:
        if not server_up(base):
            raise SystemExit(f"server not reachable at {base}")
        # Short-context judge: ASK turns plus first/mid/last continue. Not the 120k session.
        want = {0}
        want.update(i for i, _ in enumerate(texts) if is_ask_turn(i))
        cont = [i for i, _ in enumerate(texts) if i and not is_ask_turn(i)]
        if cont:
            want.update({cont[0], cont[len(cont) // 2], cont[-1]})
        for i in sorted(want):
            prev = texts[i - 1] if i else ""
            t0 = time.time()
            j = judge_reply(base, turn=i, prev=prev, cur=texts[i])
            j["turn"] = i
            j["elapsed_s"] = round(time.time() - t0, 2)
            judges.append(j)
            print(json.dumps({"judge": j}, ensure_ascii=False), flush=True)
    n_lock = sum(1 for r in rows if (r.get("score") or {}).get("locked"))
    serious = [j for j in judges if j.get("severity") == "serious"]
    summary = {
        "turns": len(rows),
        "locked_turns": n_lock,
        "stamp": {
            "stamped": stamps["stamped"],
            "max_continue_stamp_streak": stamps["max_continue_stamp_streak"],
            "first_stamp_turn": stamps["first_stamp_turn"],
        },
        "llm_serious_turns": [j["turn"] for j in serious],
        "pass": n_lock == 0 and not stamps["stamped"] and not serious,
        "jsonl": jsonl_path,
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=os.environ.get("SGLANG_DSV41_BASE", DEFAULT_BASE))
    p.add_argument("--seconds", type=int, default=int(os.environ.get("SGLANG_DSV41_TG_SOAK_SECONDS", DEFAULT_SECONDS)))
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--api", choices=("chat", "messages"), default="chat")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--jsonl", default=DEFAULT_OUT)
    p.add_argument("--self-test", action="store_true")
    p.add_argument(
        "--rejudge",
        action="store_true",
        help="Score an existing jsonl (template-stamp + optional short-context LLM judge). No 30 min TG.",
    )
    p.add_argument("--no-llm-judge", action="store_true")
    args = p.parse_args()
    self_test()
    if args.self_test:
        return 0
    if args.rejudge:
        summary = rejudge(
            args.base,
            args.jsonl,
            with_llm=not args.no_llm_judge,
        )
        return 0 if summary["pass"] else 1
    summary = run_soak(
        base=args.base,
        seconds=args.seconds,
        max_tokens=args.max_tokens,
        api=args.api,
        temperature=args.temperature,
        jsonl_path=args.jsonl,
    )
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
