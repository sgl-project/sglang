"""Reusable library for the DeepSeek-V3.2 Double Sparsity quality smoke.

Lives next to the manual hardware fixture (``test_dsv32_quality_smoke.py``)
so the SAME prompt fixtures, generation path, and gate math are exercised
by three callers:

* the manual simultaneous unittest (``test_dsv32_quality_smoke.py``),
* the single-node sequential CLI (``capture`` then ``compare`` subcommands
  in the same module), and
* the CPU regression
  (``test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py``)
  which round-trips a reference artifact through ``compute_gates`` with no
  live servers.

Why a sequential split: two TP=8 servers cannot co-reside on one 8-GPU node
(plan DEC-2). So the smoke runs DSA first, writes its 20+5 reference outputs
to a JSON artifact, then — after DSA is shut down and DS is booted — loads
that artifact and runs DS against it. ``capture_reference_outputs`` produces
the artifact; ``evaluate_against_references`` consumes it and computes the
four gates.

No external dependencies (pure-Python ROUGE-L); JSON is always parsed with
``json.loads`` and never spliced into source.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import urllib.error
import urllib.request


# ----- Prompt fixtures --------------------------------------------------

# 20 deterministic prompts: mix of conversational, factual recall, and
# explanation styles. NIAH-mini uses a separate set of 5 needle prompts.
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


# ----- Gate thresholds (AC-Q, plan) ------------------------------------

PREFIX_MATCH_THRESHOLD = 0.80
MEAN_ROUGE_L_THRESHOLD = 0.85
NIAH_MINI_THRESHOLD = 4  # of 5
PREFIX_MATCH_CHARS = 32

REFERENCE_SCHEMA = "dsv32_quality_refs_v1"
SMOKE_MAX_NEW_TOKENS = 256
NIAH_MAX_NEW_TOKENS = 16


# ----- HTTP helpers ----------------------------------------------------


def _post_json(url: str, body: Dict[str, Any], *, timeout: float = 120.0) -> Dict[str, Any]:
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


def generate(base_url: str, prompt: str, *, max_new_tokens: int = SMOKE_MAX_NEW_TOKENS) -> str:
    """Issue a deterministic (temperature=0) /generate POST against sglang."""
    body = {
        "text": prompt,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
    }
    out = _post_json(f"{base_url.rstrip('/')}/generate", body, timeout=120.0)
    # sglang /generate returns {"text": "..."} (optionally with meta_info)
    # or an OpenAI-style {"choices": [{"text": ...}]}.
    text = out.get("text")
    if text is None and "choices" in out:
        text = out["choices"][0].get("text", "")
    return text or ""


def server_commit_sha(base_url: str) -> Optional[str]:
    """Best-effort: fetch the server's commit SHA from /get_server_info."""
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


# ----- Pure-Python metrics ---------------------------------------------


def rouge_l_f(reference: str, candidate: str) -> float:
    """ROUGE-L F-measure on whitespace tokens (pure-Python LCS, no deps)."""
    ref_toks = reference.split()
    cand_toks = candidate.split()
    if not ref_toks or not cand_toks:
        return 0.0
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


def first_n_tokens_match(a: str, b: str, n: int = 8) -> bool:
    """Whitespace-token first-n comparison — "any overlap" semantics.

    The gate is that NO prompt has its first-8 tokens *entirely* different
    between DS and DSA. Even a single shared token in the first-n window
    (regardless of position) counts as overlap.
    """
    a_toks = a.split()[:n]
    b_toks = b.split()[:n]
    if not a_toks or not b_toks:
        return False
    return bool(set(a_toks) & set(b_toks))


# ----- Gate computation (the load-bearing, server-free core) -----------


def compute_gates(
    smoke_pairs: List[Tuple[str, str, str]],
    niah_pairs: List[Tuple[str, str]],
    *,
    prefix_chars: int = PREFIX_MATCH_CHARS,
    prefix_threshold: float = PREFIX_MATCH_THRESHOLD,
    rouge_threshold: float = MEAN_ROUGE_L_THRESHOLD,
    niah_min_hits: int = NIAH_MINI_THRESHOLD,
) -> Dict[str, Any]:
    """Compute the four AC-Q gates from already-generated outputs.

    ``smoke_pairs``: list of ``(prompt, dsa_text, ds_text)``.
    ``niah_pairs``: list of ``(needle, ds_text)`` — the needle must appear
    in the DS output (recall is judged on DS, against the known needle).

    Pure function — no I/O, no servers — so the CPU regression can prove the
    gate logic and the capture→compare round-trip independent of hardware.
    """
    n_smoke = max(1, len(smoke_pairs))

    prefix_hits = sum(
        1 for _, dsa, ds in smoke_pairs
        if ds[:prefix_chars] == dsa[:prefix_chars]
    )
    prefix_rate = prefix_hits / n_smoke

    rouges = [rouge_l_f(dsa, ds) for _, dsa, ds in smoke_pairs]
    mean_rouge = sum(rouges) / max(1, len(rouges))

    niah_hits = sum(1 for needle, ds in niah_pairs if needle in ds)

    first8_div = sum(
        1 for _, dsa, ds in smoke_pairs
        if not first_n_tokens_match(dsa, ds, n=8)
    )

    gates = {
        "prefix_match_rate": {
            "value": prefix_rate, "threshold": prefix_threshold,
            "pass": prefix_rate >= prefix_threshold,
        },
        "mean_rouge_l": {
            "value": mean_rouge, "threshold": rouge_threshold,
            "pass": mean_rouge >= rouge_threshold,
        },
        "niah_mini_recall": {
            "hits": niah_hits, "total": len(niah_pairs),
            "threshold": niah_min_hits, "pass": niah_hits >= niah_min_hits,
        },
        "first_8_tokens_divergence": {
            "value": first8_div, "threshold": 0, "pass": first8_div == 0,
        },
    }
    return {
        "num_smoke_prompts": len(smoke_pairs),
        "num_niah_prompts": len(niah_pairs),
        "gates": gates,
        "all_pass": all(g["pass"] for g in gates.values()),
    }


# ----- Sequential capture / compare ------------------------------------


def capture_reference_outputs(
    dsa_url: str,
    *,
    smoke_max_new_tokens: int = SMOKE_MAX_NEW_TOKENS,
    niah_max_new_tokens: int = NIAH_MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Generate the DSA reference outputs and return a serializable artifact.

    Run this with ONLY the DSA server up. The returned dict is written to a
    JSON file and later consumed by ``evaluate_against_references`` while the
    DS server (not DSA) is running.
    """
    smoke = [
        {"prompt": p, "dsa_text": generate(dsa_url, p, max_new_tokens=smoke_max_new_tokens)}
        for p in SMOKE_PROMPTS
    ]
    niah = [
        {
            "prompt": p, "needle": needle,
            "dsa_text": generate(dsa_url, p, max_new_tokens=niah_max_new_tokens),
        }
        for p, needle in NIAH_MINI_PROMPTS
    ]
    return {
        "schema": REFERENCE_SCHEMA,
        "dsa_commit_sha": server_commit_sha(dsa_url),
        "captured_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "smoke_max_new_tokens": smoke_max_new_tokens,
        "niah_max_new_tokens": niah_max_new_tokens,
        "smoke": smoke,
        "niah": niah,
    }


def _validate_reference_artifact(refs: Dict[str, Any]) -> None:
    if not isinstance(refs, dict) or refs.get("schema") != REFERENCE_SCHEMA:
        raise ValueError(
            f"reference artifact schema must be {REFERENCE_SCHEMA!r}, got "
            f"{refs.get('schema')!r}"
        )
    for key in ("smoke", "niah"):
        if not isinstance(refs.get(key), list) or not refs[key]:
            raise ValueError(f"reference artifact {key!r} must be a non-empty list")


def evaluate_against_references(
    ds_url: str,
    refs: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate DS outputs and score the four gates against saved DSA refs.

    Run this with ONLY the DS server up, passing the artifact produced by
    ``capture_reference_outputs`` (against DSA). Returns a recordable payload
    including the gate verdicts and the DS commit SHA.
    """
    _validate_reference_artifact(refs)
    smoke_max = int(refs.get("smoke_max_new_tokens", SMOKE_MAX_NEW_TOKENS))
    niah_max = int(refs.get("niah_max_new_tokens", NIAH_MAX_NEW_TOKENS))

    smoke_pairs: List[Tuple[str, str, str]] = []
    smoke_records: List[Dict[str, str]] = []
    for entry in refs["smoke"]:
        prompt = entry["prompt"]
        dsa_text = entry["dsa_text"]
        ds_text = generate(ds_url, prompt, max_new_tokens=smoke_max)
        smoke_pairs.append((prompt, dsa_text, ds_text))
        smoke_records.append({"prompt": prompt, "dsa_text": dsa_text, "ds_text": ds_text})

    niah_pairs: List[Tuple[str, str]] = []
    niah_records: List[Dict[str, str]] = []
    for entry in refs["niah"]:
        prompt = entry["prompt"]
        needle = entry["needle"]
        ds_text = generate(ds_url, prompt, max_new_tokens=niah_max)
        niah_pairs.append((needle, ds_text))
        niah_records.append(
            {"prompt": prompt, "needle": needle,
             "dsa_text": entry.get("dsa_text", ""), "ds_text": ds_text}
        )

    result = compute_gates(smoke_pairs, niah_pairs)
    result.update(
        {
            "schema": "dsv32_quality_smoke_v1",
            "evaluated_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            "ds_commit_sha": server_commit_sha(ds_url),
            "dsa_commit_sha": refs.get("dsa_commit_sha"),
            "dsa_captured_at": refs.get("captured_at"),
            "smoke_records": smoke_records,
            "niah_records": niah_records,
        }
    )
    return result
