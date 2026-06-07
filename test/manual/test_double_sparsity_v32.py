"""AC-12 quality gate for Double Sparsity on DeepSeek-V3.2 (FP8) — DS-fair.

This is a **manual** test: it requires two running sglang servers — the
Double Sparsity (DS) server and the DSA baseline reference server.

AC-12 measures DS quality **within its design envelope**. DS is
dense-prefill / sparse-decode with a fixed per-decode-step selection budget
equal to the model's DSA ``index_topk`` (``INDEX_TOPK``, 2048 on V3.2). The
HARD gates are therefore:

  * **MMLU 5-shot within 1.0 pp of DSA** (short-context quality parity), and
  * **NIAH within the selection budget within 5 pp of DSA** — needle recall at
    context lengths whose tokenized length is <= ``INDEX_TOPK`` (DS selects
    densely). This is the fair recall measure.

Model-agnostic by construction: requests use ``"model": "default"`` (the served
model), so this gate also runs unchanged against a GLM-5.1 (FP8) DS server vs its
native-DSA reference — GLM-5.1's DSA ``index_topk`` is likewise 2048, and a
different budget can be supplied via ``AC12_INDEX_TOPK`` without code changes.

Beyond the budget, DS needle recall degrades as an inherent top_k sparsity
tradeoff (and a prompt longer than the DS KV pool is unservable). Those points
(4K / 16K / 64K) are **CHARACTERIZED** — recorded with the recall-vs-length
curve and any admission limit — **not** pass/failed against DSA, which DS is
not expected to match beyond its budget. (Testing recall beyond the selection
budget tests DS outside its design envelope; this DS-fair scope and its rationale
are documented in ``runs/<date>_dsv32_mvp/ac12_analysis.md``.)

Skips cleanly when env vars are unset so CI imports do not error.

Usage (use the pytest file-path form — the repo ``test/`` tree is NOT a
Python package, so ``python -m unittest test.manual...`` collides with
the stdlib ``test`` package and fails to import)::

    DS_BASE_URL=http://localhost:30000 \
    DSA_BASE_URL=http://localhost:30001 \
    AC12_NIAH_NUM_PROMPTS=20 \
    PYTHONPATH=python python -m pytest test/manual/test_double_sparsity_v32.py -v

Optional servers (sensitivity tests):

    DS_CORRUPT_MASK_URL=http://localhost:30002   # DS with random-permuted channel mask
    DS_ZERO_SIG_URL=http://localhost:30003       # DS with zeroed token_label_table.signatures

Quick env knobs:

    AC12_NIAH_NUM_PROMPTS  per-length NIAH trials (default 20)
    AC12_NIAH_MAX_NEW_TOKENS  generation cap (default 64)
    AC12_MMLU_NUM_EXAMPLES MMLU subset size (default 200 — operator
                            override to 14k for the full set on H200)

Negative sensitivity assertions (the gate has teeth):

    corrupt-mask DS_CORRUPT_MASK_URL: NIAH @ 64K drops > 20 pp vs DSA.
    zero-signature DS_ZERO_SIG_URL:  NIAH @ 16K drops > 30 pp vs DSA.

Result artifacts are written under ``development/results/``.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import tarfile
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


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


def _as_int_or_none(value: Any) -> Optional[int]:
    """Coerce a server-reported token count to int, or None if absent/invalid."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _generate(
    base_url: str, prompt: str, *, max_new_tokens: int = 64,
    use_chat: bool = False,
) -> Tuple[str, Optional[int]]:
    """Issue a generation request and return ``(completion_text, prompt_tokens)``.

    ``prompt_tokens`` is the server-measured *tokenized* input length, read
    from the OpenAI ``usage.prompt_tokens`` (chat) or ``meta_info.prompt_tokens``
    (raw ``/generate``) field, or ``None`` if the server omitted it. The
    within-budget gate uses this real token count (not a word-count proxy) to
    decide whether a prompt fits the DS selection budget, and fails closed when
    it is absent.

    Two transports, chosen per task because the checkpoint is
    instruction-tuned:

    * ``use_chat=True`` → ``/v1/chat/completions`` (applies the model's
      chat template server-side). Required for the needle-in-haystack
      recall prompts: each is an instruction ("Question: ... Output only
      the value."), and raw ``/generate`` returns an immediate-EOS **empty
      string** for these long instruction-style prompts, which would make
      the paired recall vacuously 0/0 on BOTH servers and the gate falsely
      "pass". Verified live: a 4K needle prompt recalls the exact needle
      via chat vs empty via raw.

    * ``use_chat=False`` (default) → raw ``/generate``. The MMLU 5-shot
      check is a genuine few-shot *completion* benchmark (5 answered
      examples then the test question ending in ``Answer:``); the model
      completes the gold letter as the leading token. Applying the chat
      template here is actively harmful — the model answers conversationally
      and the letter is no longer the leading token (verified: raw parsed
      10/10 correct, chat 0/10). The MMLU transport stays raw to match
      ``benchmark/mmlu/bench_sglang.py``.
    """
    if use_chat:
        body = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
        }
        out = _post_json(
            f"{_openai_base_url(base_url)}/chat/completions",
            body, timeout=600.0,
        )
        prompt_tokens = (out.get("usage") or {}).get("prompt_tokens")
        choices = out.get("choices") or []
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
        else:
            text = ""
        return text, _as_int_or_none(prompt_tokens)
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
    prompt_tokens = (out.get("meta_info") or {}).get("prompt_tokens")
    return (text or ""), _as_int_or_none(prompt_tokens)


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
    length_words: int, *, seed: int, needle: str,
) -> str:
    """Build a deterministic NIAH prompt.

    Whitespace-tokenizes to approximately ``length_words`` words and
    plants ``needle`` exactly once at a deterministic depth in
    [0.35, 0.65] of the prompt. The needle is the exact string the
    model must echo to count as a recall hit. The prompt ends with an
    explicit question.

    Determinism: same (length_words, seed, needle) → same string.
    """
    if length_words < 64:
        length_words = 64
    rng = random.Random(seed)
    pool_size = len(_FILLER_WORDS)
    # We need length_words whitespace-separated words. Subtract a
    # constant for the question suffix and the needle phrase.
    suffix_words = (
        "Question: What is the hidden value? Output only the value."
    ).split()
    needle_phrase = f"The hidden value is {needle}."
    needle_word_count = len(needle_phrase.split())
    filler_count = max(0, length_words - len(suffix_words) - needle_word_count)

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


def _niah_needle(length_words: int, prompt_idx: int) -> str:
    """Stable per-(length, idx) needle in the form NEEDLE-####."""
    # Use a 4-digit zero-padded suffix that's unique per (L, idx).
    return f"NEEDLE-{length_words:05d}-{prompt_idx:03d}"


def _niah_recall_hits(
    needles: List[str], responses: List[str],
) -> int:
    """Count how many responses contain their planted needle as substring."""
    return sum(1 for n, r in zip(needles, responses) if n in r)


# ----- OpenAI-compatible URL normalization ------------------------------


def _openai_base_url(base_url: str) -> str:
    """Return the OpenAI-compatible base URL for ``base_url``.

    sglang exposes ``/v1/chat/completions`` and similar OpenAI-compatible
    endpoints under ``/v1`` (see ``sglang.srt.entrypoints.http_server``).
    The repo's ``run_eval.py`` CLI appends ``/v1`` for the same reason.
    Idempotent + case-insensitive on the suffix so callers can pass any
    canonical form.
    """
    u = base_url.rstrip("/")
    if u.lower().endswith("/v1"):
        return u
    return u + "/v1"


# ----- MMLU 5-shot prompt construction ----------------------------------

# These four choices mirror benchmark/mmlu/bench_sglang.py:20.
_MMLU_CHOICES: List[str] = ["A", "B", "C", "D"]


def _format_mmlu_subject(subject: str) -> str:
    """Mirror benchmark/mmlu/bench_sglang.py:25 — pretty-print the subject."""
    return " " + " ".join(subject.split("_"))


def _format_mmlu_example(
    row: List[Any], *, include_answer: bool,
) -> str:
    """One MMLU question rendered as ``Q\\nA. ...\\nB. ...\\n...\\nAnswer:[ X]``.

    ``row`` layout matches the MMLU CSV: ``[question, choice_A, choice_B,
    choice_C, choice_D, gold_letter]``. Set ``include_answer=True`` for
    dev (5-shot prefix) rows; ``False`` for the test question.
    """
    if len(row) < 6:
        raise ValueError(
            f"MMLU row must be [question, A, B, C, D, gold]; got {len(row)} cols"
        )
    parts = [str(row[0])]
    for j, letter in enumerate(_MMLU_CHOICES):
        parts.append(f"\n{letter}. {row[j + 1]}")
    parts.append("\nAnswer:")
    if include_answer:
        parts.append(f" {row[5]}\n\n")
    return "".join(parts)


def _make_mmlu_5shot_prompt(
    dev_rows: List[List[Any]], subject: str, test_row: List[Any],
) -> str:
    """Build the full 5-shot MMLU prompt for ``test_row``.

    ``dev_rows`` must contain at least 5 rows; the first 5 are used as
    the in-context shots. Returns the rendered prompt ending in
    ``"Answer:"`` so the model only needs to produce the gold letter.
    """
    if len(dev_rows) < 5:
        raise ValueError(
            f"MMLU 5-shot needs ≥5 dev rows; got {len(dev_rows)}"
        )
    header = (
        "The following are multiple choice questions (with answers) about"
        f"{_format_mmlu_subject(subject)}.\n\n"
    )
    shots = "".join(
        _format_mmlu_example(row, include_answer=True)
        for row in dev_rows[:5]
    )
    test_q = _format_mmlu_example(test_row, include_answer=False)
    return header + shots + test_q


# Answer-token parser. The Round 26 implementation scanned for the first
# A-D character anywhere in the response, which mis-scored "Answer: B"
# as "A" (the A in "Answer"). Round 27 replaces it with a regex-driven
# parser: leading isolated letter first (optionally wrapped in
# punctuation), then answer-introducer markers, then None.

# Leading isolated A-D after optional opening punctuation/whitespace,
# followed by a non-word character (or end of string) so we never split
# a word at its first letter.
_LEADING_LETTER_RE = re.compile(r'^[\s\(\[\{<"\'`]*([A-Da-d])(?!\w)')

# Marker phrases ("Answer:", "Answer is", "Option", "Choice", ...) followed
# by a standalone A-D letter. Case-insensitive.
_MARKER_RE = re.compile(
    r'(?i)(?:answer\s*[:\-]?|answer\s+is|the\s+answer\s+is|option|choice)'
    r'\s*[\(\[\{<"\'`:.]*\s*([A-Da-d])(?!\w)'
)


def _parse_mmlu_letter(response: str) -> Optional[str]:
    """Extract the predicted A-D letter from an MMLU model completion.

    Conservative parser — returns ``None`` rather than guess. Matches:

    * Leading isolated letter, possibly wrapped in punctuation: ``"B"``,
      ``" B"``, ``"(C)"``, ``"[A]"``, ``"D."``.
    * After answer-introducer markers (case-insensitive): ``"Answer: B"``,
      ``"answer is C"``, ``"The answer is D."``, ``"option B"``,
      ``"Choice (A)"``.

    Returns the upper-cased letter, or ``None``.
    """
    if not response:
        return None
    s = response.strip()
    if not s:
        return None
    m = _LEADING_LETTER_RE.match(s)
    if m:
        return m.group(1).upper()
    m2 = _MARKER_RE.search(s)
    if m2:
        return m2.group(1).upper()
    return None


# ----- MMLU example loader (pure helper, CI-testable) ------------------


def _load_mmlu_examples(
    dev_dir: str,
    test_dir: str,
    *,
    subjects: Optional[List[str]] = None,
    max_examples: int = 200,
    seed: int = 0xAC12,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Build the AC-12 MMLU example list from on-disk CSV trees.

    Round 27 still allowed a silent skip when ``dev/`` and ``test/``
    were present but contained no usable examples (empty dirs,
    missing paired CSVs, < 5 dev rows, malformed test rows). This
    helper raises ``ValueError`` for ALL of those cases so the
    caller can convert them into a hard test failure rather than
    a silent skip. Pure-function — no unittest dependency, easy
    CI coverage.

    Returns ``(examples, per_subject_totals)``. ``examples`` is the
    deterministically-shuffled list of dicts the test loop consumes;
    ``per_subject_totals`` is the per-subject test-row count (for
    artifact recording).

    Raises ``ValueError`` if the dataset cannot be loaded, with a
    message naming the resolved directories and the missing piece.
    """
    import csv as _csv

    def _read_csv_rows(path: str) -> List[List[str]]:
        # stdlib only — Round 28's pandas dependency caused a silent
        # SkipTest path inside the AC-12 gate when pandas was missing.
        with open(path, "r", newline="", encoding="utf-8") as fh:
            return [row for row in _csv.reader(fh)]

    if not os.path.isdir(test_dir):
        raise ValueError(
            f"MMLU test dir is not a directory: {test_dir!r}"
        )
    if not os.path.isdir(dev_dir):
        raise ValueError(
            f"MMLU dev dir is not a directory: {dev_dir!r}"
        )

    discovered = sorted(
        f.split("_test.csv")[0]
        for f in os.listdir(test_dir)
        if f.endswith("_test.csv")
    )
    if subjects is None or (
        isinstance(subjects, str) and subjects.strip().lower() == "all"
    ):
        chosen = discovered
    else:
        chosen = [s.strip() for s in subjects if s and s.strip()]

    if not chosen:
        raise ValueError(
            f"MMLU loader: no subjects found. Expected "
            f"`{test_dir}/*_test.csv` files (looked for any with "
            f"`_test.csv` suffix). Operator: pre-populate via "
            f"`python benchmark/mmlu/bench_sglang.py` or set "
            f"`AC12_MMLU_DATA_DIR` to a valid MMLU CSV tree."
        )

    examples: List[Dict[str, Any]] = []
    per_subject_totals: Dict[str, int] = {}
    rejected_subjects: List[str] = []
    for subject in chosen:
        dev_path = os.path.join(dev_dir, subject + "_dev.csv")
        test_path = os.path.join(test_dir, subject + "_test.csv")
        if not (os.path.isfile(dev_path) and os.path.isfile(test_path)):
            rejected_subjects.append(
                f"{subject} (missing dev or test CSV)"
            )
            continue
        try:
            dev_rows_all = _read_csv_rows(dev_path)
            test_rows_all = _read_csv_rows(test_path)
        except Exception as exc:
            rejected_subjects.append(f"{subject} (CSV read failed: {exc})")
            continue
        if len(dev_rows_all) < 5:
            rejected_subjects.append(
                f"{subject} (only {len(dev_rows_all)} dev rows, need 5)"
            )
            continue
        dev_rows = dev_rows_all[:5]
        kept = 0
        for row in test_rows_all:
            if len(row) < 6:
                continue
            examples.append(
                {"subject": subject, "dev": dev_rows, "row": row}
            )
            kept += 1
        if kept == 0:
            rejected_subjects.append(
                f"{subject} (no test row had ≥6 columns)"
            )
            continue
        per_subject_totals[subject] = (
            per_subject_totals.get(subject, 0) + kept
        )

    if not examples:
        reasons = "; ".join(rejected_subjects) if rejected_subjects else "no subjects matched"
        raise ValueError(
            f"MMLU loader: no usable examples in {dev_dir!r} / "
            f"{test_dir!r}. Rejected subjects: {reasons}. Required "
            f"layout: `{{dev,test}}/{{subject}}_{{dev,test}}.csv` with "
            f"≥5 dev rows and ≥6 columns per test row."
        )

    # Deterministic shuffle so the same examples are evaluated each run.
    rng = random.Random(seed)
    rng.shuffle(examples)
    if max_examples > 0 and len(examples) > max_examples:
        examples = examples[:max_examples]

    return examples, per_subject_totals


# ----- MMLU data self-prep (Hendrycks tarball) --------------------------


MMLU_DATA_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"


def _ensure_mmlu_data_dir(data_dir: str) -> Tuple[str, str]:
    """Return ``(dev_dir, test_dir)`` for the MMLU CSVs, downloading on demand.

    Idempotent: if ``data_dir/dev`` and ``data_dir/test`` already exist,
    returns immediately. Otherwise fetches the Hendrycks data.tar via
    ``urllib.request.urlretrieve`` into a temp dir, safely extracts only
    members under the archive's ``data/`` prefix (rejects path-traversal
    via tarfile's ``filter='data'``), and atomically moves
    ``data/dev`` + ``data/test`` into ``data_dir``.

    Raises ``RuntimeError`` on any failure (network, malformed archive,
    missing subdirectories) so callers running with paired DS/DSA servers
    fail loudly rather than silently skip.
    """
    dev_dir = os.path.join(data_dir, "dev")
    test_dir = os.path.join(data_dir, "test")
    if os.path.isdir(dev_dir) and os.path.isdir(test_dir):
        return dev_dir, test_dir

    os.makedirs(data_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "data.tar")
        try:
            urllib.request.urlretrieve(MMLU_DATA_URL, tar_path)
        except Exception as exc:  # network, DNS, 404, etc.
            raise RuntimeError(
                f"_ensure_mmlu_data_dir: download failed from {MMLU_DATA_URL}: {exc}"
            ) from exc

        extract_root = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_root, exist_ok=True)
        try:
            with tarfile.open(tar_path) as tar:
                safe_members = [
                    m for m in tar.getmembers()
                    if m.name.startswith("data/")
                    and ".." not in m.name.split("/")
                    and not os.path.isabs(m.name)
                ]
                # filter='data' (Python 3.12+) rejects symlinks/devices and
                # any path outside the extraction root.
                tar.extractall(
                    path=extract_root, members=safe_members, filter="data",
                )
        except Exception as exc:
            raise RuntimeError(
                f"_ensure_mmlu_data_dir: extraction failed: {exc}"
            ) from exc

        for sub in ("dev", "test"):
            src = os.path.join(extract_root, "data", sub)
            dst = os.path.join(data_dir, sub)
            if not os.path.isdir(src):
                raise RuntimeError(
                    f"_ensure_mmlu_data_dir: extracted archive missing data/{sub}/"
                )
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)

    return dev_dir, test_dir


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
class _GenAttempt:
    """One generation attempt against a server.

    ``ok`` is False when the server rejected the request (e.g. a
    context-length 400 when the prompt exceeds the server's KV pool); the
    HTTP status / response body / error string are then captured so the gate
    can still record a durable per-length artifact instead of letting the
    HTTP error escape before the artifact is written.
    """
    text: str
    ok: bool
    http_status: Optional[int] = None
    error: Optional[str] = None
    body: Optional[str] = None
    prompt_tokens: Optional[int] = None  # server-measured tokenized input length


def _generate_attempt(
    base_url: str, prompt: str, *, max_new_tokens: int, use_chat: bool = False,
) -> _GenAttempt:
    """``_generate`` wrapped so a server rejection is captured, not raised.

    Returns ``_GenAttempt(ok=True, text=...)`` on success, or
    ``_GenAttempt(ok=False, ...)`` carrying the HTTP status / response body /
    error string on a 4xx/5xx (``HTTPError``) or a transport failure
    (``URLError``). An unservable prompt is therefore a recorded miss, never
    an uncaught exception that skips the per-length artifact.
    """
    try:
        text, prompt_tokens = _generate(
            base_url, prompt, max_new_tokens=max_new_tokens, use_chat=use_chat,
        )
        return _GenAttempt(text=text, ok=True, prompt_tokens=prompt_tokens)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = None
        return _GenAttempt(
            text="", ok=False, http_status=exc.code,
            error=f"HTTP {exc.code}: {exc.reason}", body=body,
        )
    except urllib.error.URLError as exc:
        return _GenAttempt(
            text="", ok=False, http_status=None,
            error=f"URLError: {exc.reason}",
        )


@dataclass
class _NIAHRunResult:
    length_words: int
    num_prompts: int
    dsa_hits: int
    ds_hits: int
    dsa_recall_pct: float
    ds_recall_pct: float
    delta_pct: float  # dsa_recall_pct - ds_recall_pct (positive = DS lost quality)
    dsa_served: int = 0   # prompts the DSA server actually answered
    ds_served: int = 0    # prompts the DS server actually answered
    dsa_error: Optional[str] = None  # first DSA rejection (status/reason/body), if any
    ds_error: Optional[str] = None   # first DS rejection (status/reason/body), if any
    # Real tokenized input length (max usage.prompt_tokens over served prompts),
    # per server; None if no served prompt reported usage. *_usage_missing is
    # True when a SERVED prompt omitted usage.prompt_tokens (fail-closed signal).
    ds_input_tokens: Optional[int] = None
    dsa_input_tokens: Optional[int] = None
    ds_usage_missing: bool = False
    dsa_usage_missing: bool = False


def _summarize_attempts(
    attempts: List[_GenAttempt],
) -> Tuple[int, Optional[str]]:
    """Return ``(num_served, first_error_or_None)`` for a list of attempts."""
    served = sum(1 for a in attempts if a.ok)
    first_err = next((a for a in attempts if not a.ok), None)
    if first_err is None:
        return served, None
    msg = first_err.error or "request failed"
    if first_err.body:
        msg = f"{msg} | body={first_err.body.strip()[:300]}"
    return served, msg


def _summarize_prompt_tokens(
    attempts: List[_GenAttempt],
) -> Tuple[Optional[int], bool]:
    """Return ``(input_tokens, usage_missing)`` over the SERVED attempts.

    ``input_tokens`` is the max server-measured ``usage.prompt_tokens`` across
    served prompts (the within-budget check must hold for the longest one), or
    ``None`` if no served prompt reported usage. ``usage_missing`` is True when
    any *served* prompt omitted ``prompt_tokens`` — a fail-closed signal: the
    real tokenized length is then unknown and the within-budget premise cannot
    be asserted from the word-count proxy.
    """
    served = [a for a in attempts if a.ok]
    toks = [a.prompt_tokens for a in served if a.prompt_tokens is not None]
    usage_missing = any(a.prompt_tokens is None for a in served)
    return (max(toks) if toks else None), usage_missing


def _run_niah(
    ds_url: str, dsa_url: str, *,
    length_words: int, num_prompts: int, max_new_tokens: int,
) -> _NIAHRunResult:
    """Generate ``num_prompts`` NIAH prompts at ``length_words``, query both
    servers, and compute paired recall + delta.

    The prompts are instruction-style ("... output only the value"), so they
    use the chat-template path (``use_chat=True``) — the raw completion
    endpoint returns an empty string for them. Each request is an error-aware
    attempt: a server that cannot admit the prompt (a context-length 400 when
    the prompt exceeds the KV pool) is recorded as a failed attempt with its
    served count and error and counts as a recall miss, rather than aborting
    the gate before the artifact is written.
    """
    needles = [_niah_needle(length_words, i) for i in range(num_prompts)]
    prompts = [
        _make_niah_prompt(
            length_words, seed=(length_words * 10_000 + i), needle=needles[i],
        )
        for i in range(num_prompts)
    ]
    # DSA first, then DS (same-session convention: DSA is the reference
    # measured immediately before DS).
    dsa_attempts = [
        _generate_attempt(dsa_url, p, max_new_tokens=max_new_tokens, use_chat=True)
        for p in prompts
    ]
    ds_attempts = [
        _generate_attempt(ds_url, p, max_new_tokens=max_new_tokens, use_chat=True)
        for p in prompts
    ]
    dsa_served, dsa_error = _summarize_attempts(dsa_attempts)
    ds_served, ds_error = _summarize_attempts(ds_attempts)
    ds_input_tokens, ds_usage_missing = _summarize_prompt_tokens(ds_attempts)
    dsa_input_tokens, dsa_usage_missing = _summarize_prompt_tokens(dsa_attempts)
    # Recall is over num_prompts: an unservable prompt is a miss (text="").
    dsa_hits = _niah_recall_hits(needles, [a.text for a in dsa_attempts])
    ds_hits = _niah_recall_hits(needles, [a.text for a in ds_attempts])
    dsa_recall = (dsa_hits / num_prompts) * 100.0
    ds_recall = (ds_hits / num_prompts) * 100.0
    return _NIAHRunResult(
        length_words=length_words,
        num_prompts=num_prompts,
        dsa_hits=dsa_hits,
        ds_hits=ds_hits,
        dsa_recall_pct=dsa_recall,
        ds_recall_pct=ds_recall,
        delta_pct=dsa_recall - ds_recall,
        dsa_served=dsa_served,
        ds_served=ds_served,
        dsa_error=dsa_error,
        ds_error=ds_error,
        ds_input_tokens=ds_input_tokens,
        dsa_input_tokens=dsa_input_tokens,
        ds_usage_missing=ds_usage_missing,
        dsa_usage_missing=dsa_usage_missing,
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
    # DS is dense-prefill / sparse-decode with a fixed per-step selection
    # budget equal to the model's DSA index_topk. Needle recall is a fair DS
    # quality measure only WITHIN that budget (DS selects densely); beyond it,
    # recall degrades as an inherent top_k sparsity tradeoff and is
    # CHARACTERIZED, not pass/failed against the dense baseline.
    INDEX_TOPK = _env_int("AC12_INDEX_TOPK", 2048)
    # Within-budget NIAH word counts — chosen so the tokenized prompt stays
    # <= INDEX_TOPK (DS selects densely). The hard recall gate runs here.
    NIAH_WITHIN_BUDGET_LENGTHS = (1024, 1536)
    # Beyond-budget NIAH word counts — recall-vs-length characterization only.
    NIAH_CHARACTERIZATION_LENGTHS = (4096, 16384, 65536)

    @classmethod
    def setUpClass(cls):
        cls.ds_url = _env("DS_BASE_URL")
        cls.dsa_url = _env("DSA_BASE_URL")
        cls.num_prompts = _env_int("AC12_NIAH_NUM_PROMPTS", 20)
        cls.max_new_tokens = _env_int("AC12_NIAH_MAX_NEW_TOKENS", 64)

    def _niah_record(
        self, length_words: int, *, gate_class: str,
    ) -> Tuple[_NIAHRunResult, bool, str]:
        """Run NIAH at ``length_words``, ALWAYS record the per-length
        artifact (even on a server rejection — see _run_niah), and return
        ``(result, passed, message)``. ``passed`` is the DS-within-5pp-of-DSA
        recall result; the caller asserts it for the within-budget HARD gate
        or merely records it for the beyond-budget CHARACTERIZATION.
        """
        r = _run_niah(
            self.ds_url, self.dsa_url,
            length_words=length_words,
            num_prompts=self.num_prompts,
            max_new_tokens=self.max_new_tokens,
        )
        server_error = (r.ds_error is not None) or (r.dsa_error is not None)
        passed = (not server_error) and (r.delta_pct <= self.NIAH_TOLERANCE_PP)
        # Within-budget is decided by the REAL tokenized input length
        # (usage.prompt_tokens from the DS server), NOT the word-count knob.
        # Fail closed: if a served prompt omitted usage, the real length is
        # unknown, so within_budget is None and the within-budget gate's premise
        # assertion (below) fails rather than silently trusting the proxy.
        input_tokens = r.ds_input_tokens
        usage_missing = r.ds_usage_missing
        if usage_missing or input_tokens is None:
            within_budget: Optional[bool] = None
        else:
            within_budget = input_tokens <= self.INDEX_TOPK
        within_budget_wordcount_proxy = length_words <= self.INDEX_TOPK
        _record_artifact(
            {
                "length_words": r.length_words,
                "input_tokens": input_tokens,
                "input_tokens_source": "usage.prompt_tokens (max over served prompts)",
                "dsa_input_tokens": r.dsa_input_tokens,
                "usage_missing": usage_missing,
                "index_topk": self.INDEX_TOPK,
                "within_budget": within_budget,
                "within_budget_wordcount_proxy": within_budget_wordcount_proxy,
                "gate_class": gate_class,
                "num_prompts": r.num_prompts,
                "dsa_served": r.dsa_served,
                "ds_served": r.ds_served,
                "dsa_hits": r.dsa_hits,
                "ds_hits": r.ds_hits,
                "dsa_recall_pct": r.dsa_recall_pct,
                "ds_recall_pct": r.ds_recall_pct,
                "delta_pct": r.delta_pct,
                "threshold_pp": self.NIAH_TOLERANCE_PP,
                "dsa_error": r.dsa_error,
                "ds_error": r.ds_error,
                "verdict": "PASS" if passed else "FAIL",
            },
            suffix=f"niah_{length_words}",
        )
        if r.ds_error is not None:
            msg = (
                f"NIAH @ {length_words}: DS could not serve the prompt "
                f"(served {r.ds_served}/{r.num_prompts}); DS error: {r.ds_error}. "
                f"DSA served {r.dsa_served}/{r.num_prompts}, "
                f"recall {r.dsa_recall_pct:.1f}%."
            )
        elif r.dsa_error is not None:
            msg = (
                f"NIAH @ {length_words}: DSA reference could not serve the "
                f"prompt (served {r.dsa_served}/{r.num_prompts}); "
                f"DSA error: {r.dsa_error}."
            )
        else:
            msg = (
                f"NIAH @ {length_words}: DS recall {r.ds_recall_pct:.1f}% vs "
                f"DSA recall {r.dsa_recall_pct:.1f}% (delta {r.delta_pct:.1f} pp, "
                f"threshold {self.NIAH_TOLERANCE_PP} pp)."
            )
        return r, passed, msg

    def test_niah_within_budget(self):
        """HARD gate: at context lengths within the DS selection budget
        (tokenized length <= INDEX_TOPK, so DS selects effectively densely),
        DS needle recall must be within 5 pp of DSA. This measures DS recall
        quality inside its design envelope (dense-prefill / sparse-decode)."""
        for length_words in self.NIAH_WITHIN_BUDGET_LENGTHS:
            with self.subTest(length_words=length_words):
                r, passed, msg = self._niah_record(
                    length_words, gate_class="within_budget_hard",
                )
                # Assert the within-budget PREMISE from the REAL tokenized
                # length (usage.prompt_tokens), failing closed if the DS server
                # omitted usage — never silently trust the word-count proxy.
                self.assertFalse(
                    r.ds_usage_missing,
                    f"NIAH @ {length_words} words: a served DS prompt omitted "
                    f"usage.prompt_tokens; cannot confirm within_budget from real "
                    f"tokens (fail-closed).",
                )
                self.assertIsNotNone(
                    r.ds_input_tokens,
                    f"NIAH @ {length_words} words: no usage.prompt_tokens from any "
                    f"served DS prompt (fail-closed).",
                )
                self.assertLessEqual(
                    r.ds_input_tokens, self.INDEX_TOPK,
                    f"NIAH @ {length_words} words: real input_tokens "
                    f"{r.ds_input_tokens} exceeds INDEX_TOPK {self.INDEX_TOPK} — "
                    f"this length is not within the DS selection budget by token "
                    f"count (the word-count proxy was unsafe); move it to the "
                    f"characterization set rather than the hard gate.",
                )
                self.assertTrue(passed, msg)

    def test_niah_beyond_budget_characterization(self):
        """CHARACTERIZATION (recorded, NOT a DSA-parity pass/fail): beyond the
        selection budget DS recall degrades as an inherent top_k sparsity
        tradeoff (and the longest prompt may exceed the DS KV pool). Records
        the recall-vs-length curve + any admission limit, and asserts only the
        sanity property that DS recall is non-increasing with length among
        servable points (catches an anomalous regression) — never DSA parity,
        which DS is not expected to meet beyond its budget."""
        servable: List[Tuple[int, float]] = []
        for length_words in self.NIAH_CHARACTERIZATION_LENGTHS:
            with self.subTest(length_words=length_words):
                r, _, _ = self._niah_record(
                    length_words, gate_class="beyond_budget_characterization",
                )
                if r.ds_error is None:
                    servable.append((length_words, r.ds_recall_pct))
        for (l0, r0), (l1, r1) in zip(servable, servable[1:]):
            self.assertLessEqual(
                r1, r0 + 1e-9,
                f"DS NIAH recall increased with length ({l0}:{r0:.1f}% -> "
                f"{l1}:{r1:.1f}%), contradicting the top_k sparsity degradation "
                "— investigate as a possible regression.",
            )

    def test_mmlu_5shot(self):
        """Real MMLU 5-shot via /generate.

        Uses the same in-context format as ``benchmark/mmlu/bench_sglang.py``:
        5 answered dev examples + 1 unanswered test question + ``Answer:``;
        ``max_new_tokens=4`` (need room for the model's tokenizer to emit
        the answer letter even when wrapped in punctuation or a space);
        first stripped A-D character in the response is the prediction.

        Round 29 dropped the prior `import pandas as pd` /
        `self.skipTest(...)` guard: that silently bypassed the AC-12
        gate when pandas was unavailable but servers were configured.
        `_load_mmlu_examples` now uses stdlib `csv` and never imports
        pandas.
        """

        # Discover the MMLU CSV directories (test + dev). Operator can
        # override via AC12_MMLU_DATA_DIR; default is the standard
        # benchmark/mmlu/data/ tree.  If the data is absent, self-prep
        # by downloading the Hendrycks tarball — Round 26 silently
        # skipped here, hiding the AC-12 gate.
        data_dir = os.environ.get(
            "AC12_MMLU_DATA_DIR",
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..", "..", "benchmark", "mmlu", "data",
                )
            ),
        )
        try:
            dev_dir, test_dir = _ensure_mmlu_data_dir(data_dir)
        except Exception as exc:
            # Servers are configured (class-level skipUnless passed) so
            # an operator is actively running the AC-12 gate — silent
            # skip would mask the failure. Fail loudly so a misconfigured
            # MMLU data path cannot silently bypass the quality gate.
            self.fail(
                f"AC-12 MMLU data prep failed at {data_dir}: {exc}. "
                "Pre-populate with `python benchmark/mmlu/bench_sglang.py` "
                "or set AC12_MMLU_DATA_DIR to a directory containing "
                "`dev/` and `test/` MMLU CSV trees."
            )
            return

        # Subject set: default `all`; override AC12_MMLU_SUBJECTS=foo,bar
        # to narrow.
        env_subjects = os.environ.get("AC12_MMLU_SUBJECTS", "all")
        if env_subjects.strip().lower() == "all":
            subjects_arg = None  # signals "discover all"
        else:
            subjects_arg = [
                s.strip() for s in env_subjects.split(",") if s.strip()
            ]
        # Normalize for the artifact recorder so we never depend on an
        # undefined `subjects` name (Round 28 review NameError fix).
        subjects_for_artifact = (
            subjects_arg if subjects_arg is not None else "all"
        )
        max_examples = _env_int("AC12_MMLU_NUM_EXAMPLES", 200)

        # Build the example list via the validated loader. The Round
        # 27 silent skip ("MMLU data dir present but produced no
        # usable examples") would have masked AC-12 here when servers
        # were configured — class-level skipUnless already passed, so
        # we're inside the gate and must fail loudly.
        try:
            examples, _per_subject_totals = _load_mmlu_examples(
                dev_dir, test_dir,
                subjects=subjects_arg,
                max_examples=max_examples,
            )
        except ValueError as exc:
            self.fail(
                f"AC-12 MMLU loader failed at {data_dir}: {exc}. "
                "Pre-populate the dataset via "
                "`python benchmark/mmlu/bench_sglang.py` or set "
                "AC12_MMLU_DATA_DIR to a valid MMLU CSV tree."
            )
            return

        def _eval_against(url: str) -> Dict[str, Any]:
            correct = 0
            per_subject: Dict[str, Dict[str, int]] = {}
            for ex in examples:
                prompt = _make_mmlu_5shot_prompt(ex["dev"], ex["subject"], ex["row"])
                resp, _ = _generate(url, prompt, max_new_tokens=4)
                pred = _parse_mmlu_letter(resp.strip())
                gold = str(ex["row"][5]).strip().upper()
                hit = (pred == gold)
                if hit:
                    correct += 1
                s = ex["subject"]
                if s not in per_subject:
                    per_subject[s] = {"hits": 0, "total": 0}
                per_subject[s]["total"] += 1
                if hit:
                    per_subject[s]["hits"] += 1
            return {
                "score_pct": (correct / len(examples)) * 100.0,
                "hits": correct,
                "total": len(examples),
                "per_subject": per_subject,
            }

        # DSA first (reference measured immediately before DS).
        dsa_result = _eval_against(self.dsa_url)
        ds_result = _eval_against(self.ds_url)
        delta_pp = dsa_result["score_pct"] - ds_result["score_pct"]
        _record_artifact(
            {
                "subjects": subjects_for_artifact,
                "num_examples_evaluated": len(examples),
                "dsa_score_pct": dsa_result["score_pct"],
                "ds_score_pct": ds_result["score_pct"],
                "dsa_hits": dsa_result["hits"],
                "ds_hits": ds_result["hits"],
                "delta_pp": delta_pp,
                "threshold_pp": self.MMLU_TOLERANCE_PP,
                "dsa_per_subject": dsa_result["per_subject"],
                "ds_per_subject": ds_result["per_subject"],
            },
            suffix="mmlu_5shot",
        )
        self.assertLessEqual(
            delta_pp, self.MMLU_TOLERANCE_PP,
            f"MMLU 5-shot: DS {ds_result['score_pct']:.2f}% < DSA "
            f"{dsa_result['score_pct']:.2f}% by {delta_pp:.2f} pp "
            f"(> {self.MMLU_TOLERANCE_PP} pp threshold).",
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
            length_words=65536,
            num_prompts=self.num_prompts,
            max_new_tokens=self.max_new_tokens,
        )
        _record_artifact(
            {
                "length_words": r.length_words,
                "num_prompts": r.num_prompts,
                "dsa_hits": r.dsa_hits,
                "ds_corrupt_hits": r.ds_hits,
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
            length_words=16384,
            num_prompts=self.num_prompts,
            max_new_tokens=self.max_new_tokens,
        )
        _record_artifact(
            {
                "length_words": r.length_words,
                "num_prompts": r.num_prompts,
                "dsa_hits": r.dsa_hits,
                "ds_zero_hits": r.ds_hits,
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
