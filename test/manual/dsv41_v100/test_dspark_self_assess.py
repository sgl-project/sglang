#!/usr/bin/env python3
"""Two-turn greedy quality bench against a live DSV4.1 server.

Turn 1: a common question (temperature=0).
Turn 2: structured JSON self-assessment of that reply.

Heuristics also score turn 1 independently of the model's self-report, so a
confabulating assessor cannot hide a broken sentence.

Not CI. Needs the live 8×V100 unit (default http://127.0.0.1:11435).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:11435"

PROMPTS = [
    "Was Santa Claus invented by the Coca-Cola company? Answer in a few short paragraphs.",
    "Is it true that humans only use 10% of their brain? Answer in a few short paragraphs.",
    "Do old church windows flow downward because glass is a slow liquid? Answer in a few short paragraphs.",
    "Was Napoleon Bonaparte unusually short? Answer in a few short paragraphs.",
    "Do goldfish only remember things for three seconds? Answer in a few short paragraphs.",
    "Did Albert Einstein fail mathematics in school? Answer in a few short paragraphs.",
    "Are bats actually blind? Answer in a few short paragraphs.",
    "Does cracking your knuckles cause arthritis? Answer in a few short paragraphs.",
    "Can you see the Great Wall of China from the Moon with the naked eye? Answer in a few short paragraphs.",
    "Do different regions of the tongue detect only one taste each? Answer in a few short paragraphs.",
    "Do we swallow eight spiders a year in our sleep? Answer in a few short paragraphs.",
    "Does lightning never strike the same place twice? Answer in a few short paragraphs.",
    "Do bulls get angry because of the color red? Answer in a few short paragraphs.",
    "Does shaving make hair grow back thicker or darker? Answer in a few short paragraphs.",
    "Does sugar make children hyperactive? Answer in a few short paragraphs.",
    "Can you catch a cold from being cold? Answer in a few short paragraphs.",
    "Do lemmings jump off cliffs in mass suicide? Answer in a few short paragraphs.",
    "Does the Coriolis effect make toilets flush the other way in the Southern Hemisphere? Answer in a few short paragraphs.",
    "Do carrots significantly improve your night vision? Answer in a few short paragraphs.",
    "Is one side of the Moon permanently dark? Answer in a few short paragraphs.",
    "Are diamonds formed from compressed coal? Answer in a few short paragraphs.",
    "Do ostriches bury their heads in the sand? Answer in a few short paragraphs.",
    "Do hair and nails keep growing after death? Answer in a few short paragraphs.",
    "Do ducks' quacks not echo? Answer in a few short paragraphs.",
    "Would a penny dropped from a skyscraper kill someone on the ground? Answer in a few short paragraphs.",
    "Did Vikings wear horned helmets in battle? Answer in a few short paragraphs.",
    "Do dogs see the world only in black and white? Answer in a few short paragraphs.",
    "Is deoxygenated blood blue inside the body? Answer in a few short paragraphs.",
    "Are the seasons caused by the Earth being closer to the Sun in summer? Answer in a few short paragraphs.",
    "Do camel humps store water? Answer in a few short paragraphs.",
]

ASSESS_INSTRUCTION = (
    "Evaluate ONLY your previous reply for generation defects (not missing trivia). "
    "Return a single JSON object, no markdown fences, with exactly these keys:\n"
    '  broken_sentence: bool  (a sentence that cannot be parsed as English)\n'
    "  duplicate_clause: bool (same idea glued twice, as if two drafts collided)\n"
    "  markdown_glitch: bool (stray **, unbalanced parentheses/quotes)\n"
    "  spelling_inconsistency: bool (same proper name spelled two ways)\n"
    "  invented_self_correction: bool (claimed a typo that was not in the body)\n"
    '  severity: "none" | "minor" | "serious"\n'
    "  issues: array of {kind, quote, fix} quoting the broken span\n"
    "  summary: one sentence\n"
    "Be strict. If the reply is clean, severity is none and issues is []."
)

# Santa-1 sampled: fake footnote. Santa-2 greedy-ish: clause collision.
SANTA_SAMPLED = (
    "Starting in the 1920s ... artist **Haddon Sundblom**. "
    "*(Note: I had a typo — the artist is **Haddon Sunblom**.)*"
)
SANTA_COLLISION = (
    'His feast day is December 6, Dutch children\'s tradition brought '
    '"Sinterklass" to New Amsterdam (New York** — Dutch settlers brought '
    '"Sinterklaas" to New Amsterdam (New York), which Anglicized into '
    '"Santa Claus."'
)
CLEAN = (
    "No. Coca-Cola did not invent Santa Claus. Haddon Sundblom's 1931 ads "
    "popularized one image of an already-existing figure drawn by Thomas Nast."
)

_LEFTOVER_SPECIAL_RE = re.compile(
    r"(?:<\|[A-Za-z0-9_]+\|>|</?think>|<｜[^｜\n]{1,40}｜>)"
)


def score_turn1(text: str) -> dict[str, Any]:
    """Regex/heuristic judge. Independent of the model's self-assessment."""
    flags: dict[str, Any] = {}
    n_bold = text.count("**")
    flags["unbalanced_bold"] = (n_bold % 2) == 1
    flags["collision_glue"] = bool(
        re.search(r"\)\s*\*\*\s*[—–-]", text)
        or re.search(r"\*\*\s*[—–]\s+[A-Z]", text)
        or re.search(r"\([^)]+\*\*", text)
    )
    flags["self_correction_note"] = bool(
        re.search(
            r"(?is)(i had a typo|note:\s*.*typo|the artist is \*\*haddon sunblom)",
            text,
        )
    )
    names = re.findall(r"Sinterkla+s+", text, flags=re.I)
    n_open = text.count("(") - text.count(")")
    flags["unclosed_paren"] = n_open != 0
    leftover = _LEFTOVER_SPECIAL_RE.findall(text)
    flags["leftover_special"] = bool(leftover)
    flags["leftover_special_spans"] = leftover[:8]
    flags["sinterklaas_inconsistency"] = len({n.lower() for n in names}) > 1
    flags["duplicate_span"] = _duplicate_span(text)
    serious = (
        flags["collision_glue"]
        or flags["self_correction_note"]
        or flags["duplicate_span"]
        or flags["sinterklaas_inconsistency"]
        or flags["leftover_special"]
    )
    minor = (flags["unbalanced_bold"] or flags["unclosed_paren"]) and not serious
    flags["severity"] = "serious" if serious else ("minor" if minor else "none")
    return flags


def _duplicate_span(text: str, min_words: int = 5) -> bool:
    words = re.findall(r"[A-Za-z0-9']+", text)
    if len(words) < min_words * 2:
        return False
    seen: dict[tuple[str, ...], int] = {}
    for i in range(0, len(words) - min_words + 1):
        key = tuple(w.lower() for w in words[i : i + min_words])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 2:
            return True
    return False


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


def req(base: str, payload: dict[str, Any], timeout: int = 3600) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    r = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read())


def spec_of(out: dict[str, Any]) -> dict[str, Any]:
    choice0 = (out.get("choices") or [{}])[0]
    sglext = out.get("sglext") or choice0.get("sglext") or {}
    spec = sglext.get("spec_tokens_details") or {}
    if isinstance(spec, list):
        spec = spec[0] if spec else {}
    return spec if isinstance(spec, dict) else {}


def choice0(out: dict[str, Any]) -> dict[str, Any]:
    return (out.get("choices") or [{}])[0]


def content_of(out: dict[str, Any]) -> str:
    msg = choice0(out).get("message") or {}
    return msg.get("content") or ""


def reasoning_of(out: dict[str, Any]) -> str:
    msg = choice0(out).get("message") or {}
    return msg.get("reasoning_content") or ""


def chat(
    base: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    thinking: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": "default",
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "return_spec_tokens_details": True,
        "separate_reasoning": True,
    }
    if thinking:
        payload["chat_template_kwargs"] = {"thinking": True, "enable_thinking": True}
        payload["reasoning_effort"] = "high"
    return req(base, payload)


def self_test() -> None:
    sampled = score_turn1(SANTA_SAMPLED)
    assert sampled["self_correction_note"], sampled
    assert sampled["severity"] == "serious", sampled
    coll = score_turn1(SANTA_COLLISION)
    assert coll["collision_glue"], coll
    assert coll["sinterklaas_inconsistency"], coll
    assert coll["duplicate_span"], coll
    assert coll["severity"] == "serious", coll
    clean = score_turn1(CLEAN)
    assert clean["severity"] == "none", clean
    special = score_turn1(CLEAN + "\n<|OPEN_|>\n")
    assert special["leftover_special"] and special["severity"] == "serious", special
    think = score_turn1(CLEAN + "\n</think>\n")
    assert think["leftover_special"] and think["severity"] == "serious", think
    dsml = score_turn1(CLEAN + "\n<｜Assistant｜>\n")
    assert dsml["leftover_special"] and dsml["severity"] == "serious", dsml
    paren = score_turn1("James, who suggested people don't (not that regions sit idle.")
    assert paren["unclosed_paren"] and paren["severity"] == "minor", paren
    parsed = extract_json(
        'Thanks.\n```json\n{"broken_sentence": true, "severity": "serious", '
        '"issues": [{"kind": "glue", "quote": "x", "fix": "y"}], '
        '"summary": "ok", "duplicate_clause": true, "markdown_glitch": true, '
        '"spelling_inconsistency": true, "invented_self_correction": false}\n```'
    )
    assert parsed["broken_sentence"] is True
    assert len(PROMPTS) >= 30, len(PROMPTS)
    print("self-test ok", flush=True)


def run_one(
    base: str,
    prompt: str,
    *,
    thinking: bool,
    max_tokens_turn1: int,
    max_tokens_turn2: int,
) -> dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]
    t0 = time.time()
    out1 = chat(base, messages, max_tokens=max_tokens_turn1, thinking=thinking)
    dt1 = time.time() - t0
    text1 = content_of(out1)
    reason1 = reasoning_of(out1)
    heur = score_turn1(text1)
    usage1 = out1.get("usage") or {}
    spec1 = spec_of(out1)
    n1 = int(usage1.get("completion_tokens") or 0)
    c1 = choice0(out1)

    messages.append({"role": "assistant", "content": text1})
    messages.append({"role": "user", "content": ASSESS_INSTRUCTION})
    t0 = time.time()
    out2 = chat(base, messages, max_tokens=max_tokens_turn2, thinking=thinking)
    dt2 = time.time() - t0
    text2 = content_of(out2)
    parse_err = None
    assess: dict[str, Any] | None
    try:
        assess = extract_json(text2)
    except (json.JSONDecodeError, ValueError) as e:
        assess = None
        parse_err = str(e)

    model_serious = bool(
        assess
        and (
            assess.get("severity") == "serious"
            or assess.get("broken_sentence")
            or assess.get("duplicate_clause")
            or assess.get("invented_self_correction")
        )
    )
    return {
        "prompt": prompt,
        "thinking": thinking,
        "turn1": {
            "elapsed_s": round(dt1, 2),
            "completion_tokens": n1,
            "wall_tok_s": None if not n1 or dt1 <= 0 else round(n1 / dt1, 3),
            "finish_reason": c1.get("finish_reason"),
            "spec": spec1,
            "heuristics": heur,
            "reasoning_chars": len(reason1),
            "reasoning_empty": thinking and not reason1.strip(),
            "text": text1,
            "reasoning": reason1[:4000] if reason1 else "",
        },
        "turn2": {
            "elapsed_s": round(dt2, 2),
            "completion_tokens": int(
                (out2.get("usage") or {}).get("completion_tokens") or 0
            ),
            "finish_reason": choice0(out2).get("finish_reason"),
            "spec": spec_of(out2),
            "parse_error": parse_err,
            "assess": assess,
            "reasoning_chars": len(reasoning_of(out2)),
            "text": text2[:2000],
        },
        "flags": {
            "heuristic_serious": heur["severity"] == "serious",
            "model_serious": model_serious,
            "agree_serious": (heur["severity"] == "serious") == model_serious,
            "thinking_empty": thinking and not reason1.strip(),
            "leftover_special": bool(heur["leftover_special"]),
            "unclosed_paren": bool(heur["unclosed_paren"]),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--self-test", action="store_true")
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Opt in DSV4.1 thinking (chat_template_kwargs.thinking=true).",
    )
    p.add_argument("--max-tokens-turn1", type=int, default=0)
    p.add_argument("--max-tokens-turn2", type=int, default=0)
    p.add_argument(
        "--jsonl",
        default="dspark-self-assess.jsonl",
    )
    args = p.parse_args()
    self_test()
    if args.self_test:
        return 0

    max1 = args.max_tokens_turn1 or (4096 if args.thinking else 512)
    max2 = args.max_tokens_turn2 or (2048 if args.thinking else 384)

    n = min(args.n, len(PROMPTS))
    rows: list[dict[str, Any]] = []
    with open(args.jsonl, "w", encoding="utf-8") as fh:
        for i, prompt in enumerate(PROMPTS[:n], 1):
            print(f"=== {i}/{n} think={args.thinking} {prompt[:72]} ===", flush=True)
            row = run_one(
                args.base,
                prompt,
                thinking=args.thinking,
                max_tokens_turn1=max1,
                max_tokens_turn2=max2,
            )
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            h = row["turn1"]["heuristics"]
            a = (row["turn2"].get("assess") or {}).get("severity")
            print(
                json.dumps(
                    {
                        "i": i,
                        "thinking": args.thinking,
                        "heur_severity": h["severity"],
                        "heur": {
                            k: h[k]
                            for k in (
                                "collision_glue",
                                "self_correction_note",
                                "duplicate_span",
                                "unbalanced_bold",
                                "unclosed_paren",
                                "leftover_special",
                                "leftover_special_spans",
                                "sinterklaas_inconsistency",
                            )
                        },
                        "model_severity": a,
                        "parse_error": row["turn2"]["parse_error"],
                        "toks": row["turn1"]["completion_tokens"],
                        "reasoning_chars": row["turn1"]["reasoning_chars"],
                        "thinking_empty": row["flags"]["thinking_empty"],
                        "finish_reason": row["turn1"]["finish_reason"],
                        "wall_tok_s": row["turn1"]["wall_tok_s"],
                        "alpha": (row["turn1"]["spec"] or {}).get(
                            "spec_accept_length"
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    n_ser = sum(1 for r in rows if r["flags"]["heuristic_serious"])
    n_model = sum(1 for r in rows if r["flags"]["model_serious"])
    n_agree = sum(1 for r in rows if r["flags"]["agree_serious"])
    n_open = sum(1 for r in rows if r["flags"]["leftover_special"])
    n_paren = sum(1 for r in rows if r["flags"]["unclosed_paren"])
    n_think_empty = sum(1 for r in rows if r["flags"]["thinking_empty"])
    summary = {
        "n": n,
        "thinking": args.thinking,
        "heuristic_serious": n_ser,
        "model_serious": n_model,
        "agree_serious": n_agree,
        "leftover_special": n_open,
        "unclosed_paren": n_paren,
        "thinking_empty": n_think_empty,
        "jsonl": args.jsonl,
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    return 1 if n_ser else 0


if __name__ == "__main__":
    raise SystemExit(main())
