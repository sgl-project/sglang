"""AC-8 lightweight quality smoke for Double Sparsity on DeepSeek-V3.2.

This is a **manual** test: it requires two running sglang servers — the
Double Sparsity (DS) server and the DSA baseline server — generated on
the **same server binary** in the **same restart session** per plan §9.4
to keep the binary-vs-DS-config delta isolated. Skips cleanly when env
vars are unset.

Per AC-8 (plan §9.4):

* 20 deterministic prompts, ``temperature=0``, ``max_new_tokens=256``.
* 4 quality gates against the DSA reference:
    1. ``prefix_match_rate >= 0.80`` — fraction of prompts whose DS
       output matches DSA's first 32 characters.
    2. ``mean_rouge_l >= 0.85`` — mean ROUGE-L F-measure across 20 prompts.
    3. ``niah_mini_recall >= 4/5`` — needle-in-haystack mini test.
    4. ``first_8_tokens_divergence == 0`` — no prompt with first 8 tokens
       entirely different between DS and DSA.

Records the DS and DSA server commit SHAs into
``development/results/dsv32_quality_smoke_<timestamp>.json`` so the
operator can reproduce.

Usage::

    DS_BASE_URL=http://localhost:30000 \
    DSA_BASE_URL=http://localhost:30001 \
    python -m unittest test.manual.test_dsv32_quality_smoke

If either env var is unset, every test skips cleanly. The harness is
deliberately self-contained (no external rouge-score dependency); a
pure-Python ROUGE-L F1 is used so the test file imports in environments
where ``rouge_score`` is not installed.
"""

from __future__ import annotations

import json
import os
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple

import urllib.error
import urllib.request


# ----- Prompt fixture --------------------------------------------------

# 20 deterministic prompts: mix of conversational, factual recall, and
# explanation styles. NIAH-mini test uses a separate set of 5 needle prompts.
SMOKE_PROMPTS: List[str] = [
    "Explain in one sentence what the Pythagorean theorem says.",
    "Translate to French: \"The cat is on the mat.\" Output only the translation.",
    "List three prime numbers between 50 and 80.",
    "Who wrote the play 'Hamlet'? Give just the author's name.",
    "What is the chemical symbol for gold? Output only the symbol.",
    "Complete the sequence: 2, 4, 8, 16, ... What is the next number?",
    "In what year did humans first land on the Moon?",
    "What is the boiling point of water in Celsius at sea level?",
    "Name the largest planet in our solar system. Output only the name.",
    "What does CPU stand for in computing? Output only the expansion.",
    "Compute 17 * 23 and output the result.",
    "Give the SI unit of electric current.",
    "Which ocean is the deepest? Output only the name.",
    "How many sides does a regular hexagon have?",
    "What is the speed of light in vacuum (approximate, in m/s)? Output just the number.",
    "Who painted the Mona Lisa? Output only the painter's name.",
    "What is the capital of Japan? Output only the city name.",
    "How many bytes are in a kilobyte (binary)? Output only the integer.",
    "What is the smallest positive even number? Output only the number.",
    "In Morse code, what is the letter 'E'? Output only the dot/dash.",
]

# NIAH-mini: 5 needle-in-haystack prompts. Each haystack contains exactly
# one unique sentinel needle phrase the model must recall.
NIAH_MINI_PROMPTS: List[Tuple[str, str]] = [
    (
        "Below is a long passage. After it, there is a question.\n\n"
        + ("The library contained many books on every subject. " * 30)
        + "The hidden key is ZEBRA-7. "
        + ("Each shelf was labeled in alphabetical order. " * 30)
        + "\n\nQuestion: What is the hidden key? Output only the key.",
        "ZEBRA-7",
    ),
    (
        "Read this passage and answer the question afterwards.\n\n"
        + ("Trade winds blow consistently in tropical regions. " * 30)
        + "The secret code is MARLIN-42. "
        + ("Sailors used them for centuries to cross oceans. " * 30)
        + "\n\nQuestion: What is the secret code? Output only the code.",
        "MARLIN-42",
    ),
    (
        "Find the answer hidden in the following text.\n\n"
        + ("The forest at dawn smelled of pine and damp earth. " * 30)
        + "The treasure password is ORCHID-99. "
        + ("Birds began their morning chorus as light filtered through the leaves. " * 30)
        + "\n\nQuestion: What is the treasure password? Output only the password.",
        "ORCHID-99",
    ),
    (
        "Long passage follows. Find the answer.\n\n"
        + ("Mountain climbing requires careful preparation and equipment. " * 30)
        + "The summit signal is GLACIER-13. "
        + ("Weather can change rapidly at high altitudes. " * 30)
        + "\n\nQuestion: What is the summit signal? Output only the signal.",
        "GLACIER-13",
    ),
    (
        "Read carefully and answer.\n\n"
        + ("Ancient cities often grew around rivers for water and trade. " * 30)
        + "The city motto is PHARAOH-88. "
        + ("Markets bustled with merchants from distant lands. " * 30)
        + "\n\nQuestion: What is the city motto? Output only the motto.",
        "PHARAOH-88",
    ),
]


# ----- HTTP helpers ----------------------------------------------------


def _post_json(url: str, body: Dict[str, Any], *, timeout: float = 60.0) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str, *, timeout: float = 10.0) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def _generate(base_url: str, prompt: str, *, max_new_tokens: int = 256) -> str:
    """Issue a /generate POST against an sglang HTTP server."""
    body = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
    }
    out = _post_json(f"{base_url.rstrip('/')}/generate", body, timeout=120.0)
    # sglang /generate returns either {"text": "..."} or
    # {"meta_info": {...}, "text": "..."} depending on options.
    text = out.get("text")
    if text is None and "choices" in out:
        text = out["choices"][0].get("text", "")
    return text or ""


def _server_commit_sha(base_url: str) -> Optional[str]:
    """Best-effort: fetch the server's commit SHA from /get_server_info.

    sglang exposes this via the server-info endpoint; the field name has
    drifted across versions, so we look at a few candidates.
    """
    info_raw = _get_text(f"{base_url.rstrip('/')}/get_server_info")
    if info_raw is None:
        return None
    try:
        info = json.loads(info_raw)
    except (json.JSONDecodeError, TypeError):
        return None
    for key in ("commit", "commit_sha", "git_commit", "server_commit_sha"):
        if key in info:
            return str(info[key])
    return None


# ----- Pure-Python ROUGE-L F-measure -----------------------------------


def _rouge_l_f(reference: str, candidate: str) -> float:
    """Compute ROUGE-L F-measure on whitespace tokens.

    Pure-Python LCS so the harness has no extra deps. Returns 0.0 for
    empty inputs.
    """
    ref_toks = reference.split()
    cand_toks = candidate.split()
    if not ref_toks or not cand_toks:
        return 0.0
    # Iterative LCS.
    m, n = len(ref_toks), len(cand_toks)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_toks[i] == cand_toks[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / n
    recall = lcs / m
    return 2 * precision * recall / (precision + recall)


def _first_n_tokens_match(a: str, b: str, n: int = 8) -> bool:
    """Whitespace-token first-n comparison."""
    a_toks = a.split()[:n]
    b_toks = b.split()[:n]
    if not a_toks or not b_toks:
        return False
    # Any overlap at all means they're not "entirely different".
    return any(at == bt for at, bt in zip(a_toks, b_toks))


# ----- Test class ------------------------------------------------------


def _env(name: str) -> Optional[str]:
    return os.environ.get(name)


@unittest.skipUnless(
    _env("DS_BASE_URL") and _env("DSA_BASE_URL"),
    "DS_BASE_URL and DSA_BASE_URL env vars must both point at running servers.",
)
class TestDSv32QualitySmoke(unittest.TestCase):
    """AC-8 lightweight quality smoke. Manual hardware test."""

    PREFIX_MATCH_THRESHOLD = 0.80
    MEAN_ROUGE_L_THRESHOLD = 0.85
    NIAH_MINI_THRESHOLD = 4  # of 5
    PREFIX_MATCH_CHARS = 32

    @classmethod
    def setUpClass(cls):
        cls.ds_url = _env("DS_BASE_URL")
        cls.dsa_url = _env("DSA_BASE_URL")
        cls.ds_commit = _server_commit_sha(cls.ds_url)
        cls.dsa_commit = _server_commit_sha(cls.dsa_url)

    def _run_paired(
        self, prompts: List[str], *, max_new_tokens: int = 256,
    ) -> List[Tuple[str, str, str]]:
        """For each prompt: ``(prompt, dsa_text, ds_text)``."""
        results: List[Tuple[str, str, str]] = []
        for p in prompts:
            # DSA first, then DS — per plan §9.4 the DSA reference is
            # generated in the same session immediately before DS.
            dsa_text = _generate(self.dsa_url, p, max_new_tokens=max_new_tokens)
            ds_text = _generate(self.ds_url, p, max_new_tokens=max_new_tokens)
            results.append((p, dsa_text, ds_text))
        return results

    def _record(self, payload: Dict[str, Any]) -> None:
        out_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "development", "results",
        )
        out_dir = os.path.abspath(out_dir)
        try:
            os.makedirs(out_dir, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            path = os.path.join(out_dir, f"dsv32_quality_smoke_{ts}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except OSError:
            # Best-effort artifact recording; test result still asserts gates.
            pass

    def test_quality_smoke(self):
        """Run all four AC-8 quality gates in one pass."""
        paired = self._run_paired(SMOKE_PROMPTS)
        niah_paired = self._run_paired(
            [p for p, _ in NIAH_MINI_PROMPTS],
            max_new_tokens=16,
        )

        # --- gate 1: prefix-match rate ---
        prefix_match_hits = sum(
            1 for _, dsa, ds in paired
            if ds[: self.PREFIX_MATCH_CHARS] == dsa[: self.PREFIX_MATCH_CHARS]
            and len(dsa) >= self.PREFIX_MATCH_CHARS
        )
        prefix_match_rate = prefix_match_hits / max(1, len(paired))

        # --- gate 2: mean ROUGE-L F ---
        rouges = [_rouge_l_f(dsa, ds) for _, dsa, ds in paired]
        mean_rouge_l = sum(rouges) / max(1, len(rouges))

        # --- gate 3: NIAH-mini recall ---
        niah_hits = sum(
            1 for (_, needle), (_, _dsa, ds) in zip(NIAH_MINI_PROMPTS, niah_paired)
            if needle in ds
        )

        # --- gate 4: first-8-token divergence ---
        first_8_divergences = sum(
            1 for _, dsa, ds in paired
            if not _first_n_tokens_match(dsa, ds, n=8)
        )

        self._record(
            {
                "ds_commit_sha": self.ds_commit,
                "dsa_commit_sha": self.dsa_commit,
                "num_prompts": len(paired),
                "prefix_match_rate": prefix_match_rate,
                "mean_rouge_l": mean_rouge_l,
                "niah_mini_hits": niah_hits,
                "niah_mini_total": len(NIAH_MINI_PROMPTS),
                "first_8_tokens_divergences": first_8_divergences,
                "thresholds": {
                    "prefix_match_rate": self.PREFIX_MATCH_THRESHOLD,
                    "mean_rouge_l": self.MEAN_ROUGE_L_THRESHOLD,
                    "niah_mini_min_hits": self.NIAH_MINI_THRESHOLD,
                    "first_8_tokens_divergences_max": 0,
                },
            }
        )

        self.assertGreaterEqual(
            prefix_match_rate, self.PREFIX_MATCH_THRESHOLD,
            f"prefix-match rate {prefix_match_rate:.3f} < {self.PREFIX_MATCH_THRESHOLD}",
        )
        self.assertGreaterEqual(
            mean_rouge_l, self.MEAN_ROUGE_L_THRESHOLD,
            f"mean ROUGE-L {mean_rouge_l:.3f} < {self.MEAN_ROUGE_L_THRESHOLD}",
        )
        self.assertGreaterEqual(
            niah_hits, self.NIAH_MINI_THRESHOLD,
            f"NIAH-mini recall {niah_hits}/{len(NIAH_MINI_PROMPTS)} < "
            f"{self.NIAH_MINI_THRESHOLD}/{len(NIAH_MINI_PROMPTS)}",
        )
        self.assertEqual(
            first_8_divergences, 0,
            f"{first_8_divergences} prompt(s) with first-8 tokens entirely "
            "different between DS and DSA.",
        )


if __name__ == "__main__":
    unittest.main()
