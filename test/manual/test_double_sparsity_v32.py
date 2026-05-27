"""AC-12 hard quality gate for Double Sparsity on DeepSeek-V3.2 (FP8).

This is a **manual** test: it requires two running sglang servers — the
Double Sparsity (DS) server and the DSA baseline reference server.
Per plan §10 AC-12 is HARD: the Loop 4 MVP does not close without
NIAH @ 4K / 16K / 64K within 5 pp of DSA AND MMLU 5-shot within
1.0 pp of DSA. The harness fails the suite when DS - DSA exceeds
either threshold (sign convention: DSA - DS > 5 pp means DS lost
quality, which is what the gate detects).

Skips cleanly when env vars are unset so CI imports do not error.

Usage::

    DS_BASE_URL=http://localhost:30000 \
    DSA_BASE_URL=http://localhost:30001 \
    AC12_NIAH_NUM_PROMPTS=20 \
    python -m unittest test.manual.test_double_sparsity_v32

Optional servers (sensitivity tests):

    DS_CORRUPT_MASK_URL=http://localhost:30002   # DS with random-permuted channel mask
    DS_ZERO_SIG_URL=http://localhost:30003       # DS with zeroed token_label_table.signatures

Quick env knobs:

    AC12_NIAH_NUM_PROMPTS  per-length NIAH trials (default 20)
    AC12_NIAH_MAX_NEW_TOKENS  generation cap (default 64)
    AC12_MMLU_NUM_EXAMPLES MMLU subset size (default 200 — operator
                            override to 14k for the full set on H200)

Negative sensitivity assertions (plan §10 + design doc §9.5 B6):

    corrupt-mask DS_CORRUPT_MASK_URL: NIAH @ 64K drops > 20 pp vs DSA.
    zero-signature DS_ZERO_SIG_URL:  NIAH @ 16K drops > 30 pp vs DSA.

Result artifacts are written under ``development/results/``.
"""

from __future__ import annotations

import json
import os
import random
import time
import unittest
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ----- env-var contract --------------------------------------------------


def _env(name: str) -> Optional[str]:
    return os.environ.get(name)


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


# ----- HTTP helpers (mirror test_dsv32_quality_smoke.py) -----------------


def _post_json(
    url: str, body: Dict[str, Any], *, timeout: float = 600.0,
) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _generate(
    base_url: str, prompt: str, *, max_new_tokens: int = 64,
) -> str:
    """Issue a /generate POST. Mirrors the smoke harness."""
    body = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
    }
    out = _post_json(f"{base_url.rstrip('/')}/generate", body, timeout=600.0)
    text = out.get("text")
    if text is None and "choices" in out:
        text = out["choices"][0].get("text", "")
    return text or ""


# ----- NIAH prompt generator --------------------------------------------


# Deterministic word pool — Lorem-ipsum-style filler that is whitespace-
# stable across Python versions and operating systems. Kept inline so
# the harness has no data-file dependency.
_FILLER_WORDS: List[str] = (
    "the quick brown fox jumps over the lazy dog ".split()
    + "ancient cities grew around rivers for trade and water ".split()
    + "merchants brought silks gold spices and stories from afar ".split()
    + "the library held tomes on philosophy mathematics and natural history ".split()
    + "sailors crossed oceans guided by stars and the trade winds ".split()
    + "mountain ranges separated the eastern and western kingdoms ".split()
    + "scribes copied scrolls by candlelight long into the evening ".split()
    + "harvest festivals celebrated the bounty of the autumn fields ".split()
)


def _make_niah_prompt(
    length_tokens: int, *, seed: int, needle: str,
) -> str:
    """Build a deterministic NIAH prompt.

    Whitespace-tokenizes to approximately ``length_tokens`` words and
    plants ``needle`` exactly once at a deterministic depth in
    [0.35, 0.65] of the prompt. The needle is the exact string the
    model must echo to count as a recall hit. The prompt ends with an
    explicit question.

    Determinism: same (length_tokens, seed, needle) → same string.
    """
    if length_tokens < 64:
        length_tokens = 64
    rng = random.Random(seed)
    pool_size = len(_FILLER_WORDS)
    # We need length_tokens whitespace-separated words. Subtract a
    # constant for the question suffix and the needle phrase.
    suffix_words = (
        "Question: What is the hidden value? Output only the value."
    ).split()
    needle_phrase = f"The hidden value is {needle}."
    needle_word_count = len(needle_phrase.split())
    filler_count = max(0, length_tokens - len(suffix_words) - needle_word_count)

    # Pick a depth in [0.35, 0.65].
    depth_frac = 0.35 + (rng.random() * 0.30)
    insert_at = int(filler_count * depth_frac)
    # Pre-needle and post-needle filler.
    pre_words = [_FILLER_WORDS[rng.randrange(pool_size)] for _ in range(insert_at)]
    post_words = [
        _FILLER_WORDS[rng.randrange(pool_size)]
        for _ in range(filler_count - insert_at)
    ]
    parts = pre_words + needle_phrase.split() + post_words + suffix_words
    return " ".join(parts)


def _niah_needle(length_tokens: int, prompt_idx: int) -> str:
    """Stable per-(length, idx) needle in the form NEEDLE-####."""
    # Use a 4-digit zero-padded suffix that's unique per (L, idx).
    return f"NEEDLE-{length_tokens:05d}-{prompt_idx:03d}"


def _niah_recall_hits(
    needles: List[str], responses: List[str],
) -> int:
    """Count how many responses contain their planted needle as substring."""
    return sum(1 for n, r in zip(needles, responses) if n in r)


# ----- result recording --------------------------------------------------


def _record_artifact(payload: Dict[str, Any], *, suffix: str) -> None:
    out_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "development", "results",
        )
    )
    try:
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        path = os.path.join(out_dir, f"ac12_{suffix}_{ts}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except OSError:
        # Best-effort artifact; gate result still asserts.
        pass


# ----- test scaffolding --------------------------------------------------


@dataclass
class _NIAHRunResult:
    length_tokens: int
    num_prompts: int
    dsa_hits: int
    ds_hits: int
    dsa_recall_pct: float
    ds_recall_pct: float
    delta_pct: float  # dsa_recall_pct - ds_recall_pct (positive = DS lost quality)


def _run_niah(
    ds_url: str, dsa_url: str, *,
    length_tokens: int, num_prompts: int, max_new_tokens: int,
) -> _NIAHRunResult:
    """Generate ``num_prompts`` NIAH prompts at ``length_tokens``, query
    both servers, compute paired recall and delta."""
    needles = [_niah_needle(length_tokens, i) for i in range(num_prompts)]
    prompts = [
        _make_niah_prompt(
            length_tokens, seed=(length_tokens * 10_000 + i), needle=needles[i],
        )
        for i in range(num_prompts)
    ]
    # DSA first per the plan §9.4 "same session, DSA immediately before DS"
    # convention used by the lightweight smoke harness.
    dsa_responses = [
        _generate(dsa_url, p, max_new_tokens=max_new_tokens) for p in prompts
    ]
    ds_responses = [
        _generate(ds_url, p, max_new_tokens=max_new_tokens) for p in prompts
    ]
    dsa_hits = _niah_recall_hits(needles, dsa_responses)
    ds_hits = _niah_recall_hits(needles, ds_responses)
    dsa_recall = (dsa_hits / num_prompts) * 100.0
    ds_recall = (ds_hits / num_prompts) * 100.0
    return _NIAHRunResult(
        length_tokens=length_tokens,
        num_prompts=num_prompts,
        dsa_hits=dsa_hits,
        ds_hits=ds_hits,
        dsa_recall_pct=dsa_recall,
        ds_recall_pct=ds_recall,
        delta_pct=dsa_recall - ds_recall,
    )


# ----- the test class ---------------------------------------------------


@unittest.skipUnless(
    _env("DS_BASE_URL") and _env("DSA_BASE_URL"),
    "DS_BASE_URL and DSA_BASE_URL env vars must both point at running servers.",
)
class TestDoubleSparsityV32Quality(unittest.TestCase):
    """AC-12 hard quality gate. Manual hardware test."""

    NIAH_TOLERANCE_PP = 5.0
    MMLU_TOLERANCE_PP = 1.0
    CORRUPT_MASK_MIN_DROP_PP = 20.0
    ZERO_SIG_MIN_DROP_PP = 30.0

    @classmethod
    def setUpClass(cls):
        cls.ds_url = _env("DS_BASE_URL")
        cls.dsa_url = _env("DSA_BASE_URL")
        cls.num_prompts = _env_int("AC12_NIAH_NUM_PROMPTS", 20)
        cls.max_new_tokens = _env_int("AC12_NIAH_MAX_NEW_TOKENS", 64)

    def _niah_assert(self, length_tokens: int) -> _NIAHRunResult:
        r = _run_niah(
            self.ds_url, self.dsa_url,
            length_tokens=length_tokens,
            num_prompts=self.num_prompts,
            max_new_tokens=self.max_new_tokens,
        )
        _record_artifact(
            {
                "length_tokens": r.length_tokens,
                "num_prompts": r.num_prompts,
                "dsa_recall_pct": r.dsa_recall_pct,
                "ds_recall_pct": r.ds_recall_pct,
                "delta_pct": r.delta_pct,
                "threshold_pp": self.NIAH_TOLERANCE_PP,
            },
            suffix=f"niah_{length_tokens}",
        )
        self.assertLessEqual(
            r.delta_pct, self.NIAH_TOLERANCE_PP,
            f"NIAH @ {length_tokens}: DS recall {r.ds_recall_pct:.1f}% < "
            f"DSA recall {r.dsa_recall_pct:.1f}% by {r.delta_pct:.1f} pp "
            f"(> {self.NIAH_TOLERANCE_PP} pp threshold).",
        )
        return r

    def test_niah_at_4k(self):
        self._niah_assert(4096)

    def test_niah_at_16k(self):
        self._niah_assert(16384)

    def test_niah_at_64k(self):
        self._niah_assert(65536)

    def test_mmlu_5shot(self):
        """MMLU 5-shot via sglang.test.run_eval."""
        try:
            from sglang.test.run_eval import run_eval_once
            from sglang.test.simple_eval_mmlu import MMLUEval
        except ImportError as exc:
            self.skipTest(
                f"sglang.test.run_eval / simple_eval_mmlu unavailable: {exc}"
            )
            return

        from argparse import Namespace
        num_examples = _env_int("AC12_MMLU_NUM_EXAMPLES", 200)
        num_threads = _env_int("AC12_MMLU_NUM_THREADS", 8)

        filename = (
            "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
        )

        def _run_against(base_url: str) -> float:
            eval_obj = MMLUEval(filename, num_examples, num_threads)
            args = Namespace(
                base_url=base_url,
                model=None,
                max_tokens=512,
                top_p=1.0,
                temperature=0.0,
                api="chat",
                reasoning_effort=None,
                stop=None,
            )
            result, _latency, _sampler = run_eval_once(args, base_url, eval_obj)
            return float(result.score) * 100.0

        # DSA first per plan §9.4 convention.
        dsa_score = _run_against(self.dsa_url)
        ds_score = _run_against(self.ds_url)
        delta_pp = dsa_score - ds_score
        _record_artifact(
            {
                "num_examples": num_examples,
                "dsa_score_pct": dsa_score,
                "ds_score_pct": ds_score,
                "delta_pp": delta_pp,
                "threshold_pp": self.MMLU_TOLERANCE_PP,
            },
            suffix="mmlu_5shot",
        )
        self.assertLessEqual(
            delta_pp, self.MMLU_TOLERANCE_PP,
            f"MMLU 5-shot: DS {ds_score:.2f}% < DSA {dsa_score:.2f}% by "
            f"{delta_pp:.2f} pp (> {self.MMLU_TOLERANCE_PP} pp threshold).",
        )

    @unittest.skipUnless(
        _env("DS_CORRUPT_MASK_URL"),
        "DS_CORRUPT_MASK_URL required (boot DS server with "
        "SGLANG_DS_FAULT_INJECT_CORRUPT_MASK=1).",
    )
    def test_niah_64k_sensitivity_corrupt_mask(self):
        """Negative: corrupt mask must drop NIAH @ 64K by > 20 pp."""
        corrupt_url = _env("DS_CORRUPT_MASK_URL")
        r = _run_niah(
            corrupt_url, self.dsa_url,
            length_tokens=65536,
            num_prompts=self.num_prompts,
            max_new_tokens=self.max_new_tokens,
        )
        _record_artifact(
            {
                "length_tokens": r.length_tokens,
                "num_prompts": r.num_prompts,
                "dsa_recall_pct": r.dsa_recall_pct,
                "ds_corrupt_recall_pct": r.ds_recall_pct,
                "delta_pct": r.delta_pct,
                "threshold_pp": self.CORRUPT_MASK_MIN_DROP_PP,
                "expectation": "delta must EXCEED threshold",
            },
            suffix="niah_64k_corrupt_mask",
        )
        self.assertGreater(
            r.delta_pct, self.CORRUPT_MASK_MIN_DROP_PP,
            f"sensitivity: corrupt-mask DS recall {r.ds_recall_pct:.1f}% must "
            f"drop > {self.CORRUPT_MASK_MIN_DROP_PP} pp below DSA "
            f"{r.dsa_recall_pct:.1f}% (observed delta {r.delta_pct:.1f} pp).",
        )

    @unittest.skipUnless(
        _env("DS_ZERO_SIG_URL"),
        "DS_ZERO_SIG_URL required (boot DS server with "
        "SGLANG_DS_FAULT_INJECT_ZERO_SIG=1).",
    )
    def test_niah_16k_sensitivity_zero_signatures(self):
        """Negative: zeroed signatures must drop NIAH @ 16K by > 30 pp."""
        zero_url = _env("DS_ZERO_SIG_URL")
        r = _run_niah(
            zero_url, self.dsa_url,
            length_tokens=16384,
            num_prompts=self.num_prompts,
            max_new_tokens=self.max_new_tokens,
        )
        _record_artifact(
            {
                "length_tokens": r.length_tokens,
                "num_prompts": r.num_prompts,
                "dsa_recall_pct": r.dsa_recall_pct,
                "ds_zero_recall_pct": r.ds_recall_pct,
                "delta_pct": r.delta_pct,
                "threshold_pp": self.ZERO_SIG_MIN_DROP_PP,
                "expectation": "delta must EXCEED threshold",
            },
            suffix="niah_16k_zero_sig",
        )
        self.assertGreater(
            r.delta_pct, self.ZERO_SIG_MIN_DROP_PP,
            f"sensitivity: zero-signature DS recall {r.ds_recall_pct:.1f}% "
            f"must drop > {self.ZERO_SIG_MIN_DROP_PP} pp below DSA "
            f"{r.dsa_recall_pct:.1f}% (observed delta {r.delta_pct:.1f} pp).",
        )


if __name__ == "__main__":
    unittest.main()
