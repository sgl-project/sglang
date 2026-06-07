#!/usr/bin/env python3
"""Two-column comparator for native_nsa vs double_sparsity bench_serving runs.

Consumes the JSONL output of ``sglang.bench_serving --output-file`` for a
``native_nsa`` baseline run (produced by ``benchmark_baseline.sh``) and a
``double_sparsity`` run (produced by ``benchmark.sh``), and emits a side-by-
side report.

Refuses to publish if the two runs' hardware / TP / page / radix-cache /
concurrency context disagree. The match-enforcement contract from AC-7:

    {GPU id, TP size, page size, radix-cache setting, concurrency}

must be identical between the two columns. All five are enforced by
default; pass ``--allow-gpu-mismatch`` only for deliberate cross-hardware
comparison reports. Bench_serving currently records most of these as
run-level metadata; missing fields are best-effort matched against the
filename tags (e.g. ``native_nsa_gsp_isl4096_osl512_c64.jsonl`` implies
concurrency=64).

SLO gate: each row is annotated with `pass` / `fail` against per-request
decode throughput ``median_decode_throughput_tps >= 30`` (output_tokens /
(e2e - ttft); from bench_serving, with a legacy-fixture fallback) and the
strict tail bar ``ttft_p99_s < 22``.

No-op detector per AC-7: a row is flagged if any of
``selected_tokens == total_tokens`` or ``dense_fallback_total != 0``.

The tool exposes two CLI modes:

* Single-trial AC-7/AC-8 report (legacy): pass ``--baseline`` and
  ``--ds`` to compare one DSA JSONL against one DS JSONL.
* Three-trial AC-11 directional report: pass ``--ac11`` along with
  ``--ac11-baseline-results`` and ``--ac11-ds-results`` lists. Each
  side requires >= 3 trial JSONLs per concurrency; the comparator
  groups by concurrency, takes per-field medians, and enforces the
  AC-11 gates (DS TPS >= 95% of DSA TPS; DS P99 TTFT <= 1.10 * DSA
  P99 TTFT). Exit codes: 0 = all gates pass, 3 = at least one gate
  failed (Markdown output names the failing concurrencies + emits a
  profiling obligation), 2 = input refusal (too few trials, missing
  concurrency, or mismatched per-trial operating point).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


SLO_PER_REQUEST_TPS_P50 = 30.0
SLO_TTFT_P99_S = 22.0


@dataclass
class RunMetrics:
    """Metrics distilled from a single bench_serving JSONL row."""

    concurrency: int
    num_prompts: int
    isl: int
    osl: int
    output_tps_p50: Optional[float]
    output_tps_p99: Optional[float]
    ttft_p50_s: Optional[float]
    ttft_p99_s: Optional[float]
    tpot_p50_ms: Optional[float]
    tpot_p99_ms: Optional[float]
    goodput_under_slo: Optional[float]
    selected_tokens_mean: Optional[float]
    dense_fallback_total: Optional[int]
    total_tokens_mean: Optional[float]
    # bench_serving's wall-clock duration of the measured phase. Used by
    # AC-11 to refuse trials whose real measurement window is below the
    # 600s floor — the sidecar `measurement_window_seconds` is a knob,
    # `duration` is the observation.
    duration_s: Optional[float] = None
    # Achieved (effective) concurrency that bench_serving actually sustained
    # (the JSONL ``concurrency`` float), as opposed to the nominal
    # ``--max-concurrency``. Surfaced in the AC-11 report so a DS column that
    # is admission/queue-bound (e.g. DS at mem 0.6 with a small KV pool) is
    # visible against DSA's near-nominal concurrency, not hidden (#F).
    achieved_concurrency: Optional[float] = None


@dataclass
class RunContext:
    """Hardware / config metadata that must agree across columns."""

    gpu_id: Optional[str]
    tp_size: Optional[int]
    page_size: Optional[int]
    disable_radix_cache: Optional[bool]
    concurrency: Optional[int]


def _filename_concurrency(path: str) -> Optional[int]:
    # The 3-trial sweep appends ``_t<N>`` between the concurrency tag
    # and the ``.jsonl`` extension (e.g. ``_c64_t2.jsonl``); accept
    # both that form and the legacy ``_c64.jsonl`` form.
    m = re.search(r"_c(\d+)(?:_t\d+)?\.jsonl$", path)
    if m:
        return int(m.group(1))
    return None


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Return the percentile of a list without depending on numpy."""

    if not values:
        return None
    sorted_v = sorted(values)
    if len(sorted_v) == 1:
        return float(sorted_v[0])
    k = (len(sorted_v) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = k - lo
    return float(sorted_v[lo] * (1.0 - frac) + sorted_v[hi] * frac)


def _per_request_output_tps(summary: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Derive per-request *generation-rate* tok/s P50/P99 from bench_serving arrays.

    bench_serving with --output-details emits ``output_lens`` and ``itls``.
    Per-request TPS = ``output_lens[i] / sum(itls[i])`` — generation rate
    only. TTFT is intentionally NOT in the denominator: it has its own
    threshold via ``_slo_verdict``, and conflating it here would falsely
    fail runs with high-but-passing TTFT (e.g. 512 tokens, 21 s TTFT, 10 ms
    ITL reports ~100 tok/s as expected, not ~20).

    Returns (None, None) when the required arrays are missing or produce
    no usable rows.
    """

    output_lens = summary.get("output_lens")
    itls = summary.get("itls")
    if not (isinstance(output_lens, list) and isinstance(itls, list)):
        return None, None
    n = min(len(output_lens), len(itls))
    if n == 0:
        return None, None
    per_req: List[float] = []
    for i in range(n):
        olen = output_lens[i]
        if not isinstance(olen, (int, float)) or olen <= 0:
            continue
        itl_row = itls[i] if isinstance(itls[i], list) else []
        itl_sum = sum(v for v in itl_row if isinstance(v, (int, float)))
        if itl_sum > 0:
            per_req.append(float(olen) / float(itl_sum))
    if not per_req:
        return None, None
    return _percentile(per_req, 50.0), _percentile(per_req, 99.0)


def _read_bench_jsonl(path: str) -> Tuple[RunContext, RunMetrics]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"bench file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"bench file is empty: {path}")
    summary = rows[-1] if isinstance(rows[-1], dict) else rows[0]

    # bench_serving nests hardware / TP / radix metadata under server_info.
    # Hand-crafted fixtures may put them at the top level; honor both.
    server_info = summary.get("server_info") or {}
    if not isinstance(server_info, dict):
        server_info = {}

    def _from_either(key: str):
        return server_info.get(key) if key in server_info else summary.get(key)

    concurrency = (
        summary.get("max_concurrency")
        or summary.get("concurrency")
        or server_info.get("max_concurrency")
        or _filename_concurrency(path)
    )
    gpu_raw = _from_either("gpu_id")
    if gpu_raw is None:
        # bench_serving's /server_info emits `device: "cuda"` (a generic
        # device type, not a GPU identifier) and the rank under
        # `base_gpu_id`. Falling back to `device` would collapse GPU 0 and
        # GPU 1 onto the same "cuda" string and defeat the Round-12 default
        # GPU-match gate. Only try `base_gpu_id`; if neither is present,
        # leave gpu_id=None so _match_or_refuse refuses the comparison.
        gpu_raw = _from_either("base_gpu_id")
    context = RunContext(
        gpu_id=str(gpu_raw) if gpu_raw is not None else None,
        tp_size=_from_either("tp_size"),
        page_size=_from_either("page_size"),
        disable_radix_cache=_from_either("disable_radix_cache"),
        concurrency=int(concurrency) if concurrency is not None else None,
    )

    def _float(key: str) -> Optional[float]:
        v = summary.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    def _int(key: str) -> Optional[int]:
        v = summary.get(key)
        return int(v) if isinstance(v, (int, float)) else None

    def _ms_to_s(key: str) -> Optional[float]:
        v = _float(key)
        return None if v is None else v / 1000.0

    # Per-request decode throughput = output_tokens / (e2e_latency - ttft), the
    # canonical client-SLO metric. bench_serving now emits it directly as
    # `median_decode_throughput_tps`; consume that FIRST so a current artifact
    # gates on the resolved metric. The `output_lens / sum(itls)` derivation and
    # the legacy `output_throughput_*` / `per_req_output_tps_*` scalars are kept
    # ONLY as fallback for older fixtures that predate the decode-throughput
    # emission — a current row carrying the new field must not silently fall
    # through to a differently-derived number.
    decode_p50 = _float("median_decode_throughput_tps")
    if decode_p50 is not None:
        per_req_p50 = decode_p50
        # The high-tail p99 is reporting-only (the SLO/AC-11 gate uses p50) and
        # is not emitted as a decode-throughput field; populate it from the
        # --output-details array derivation when present, else leave it unset.
        _, per_req_p99 = _per_request_output_tps(summary)
    else:
        per_req_p50, per_req_p99 = _per_request_output_tps(summary)
        if per_req_p50 is None:
            per_req_p50 = _float("output_throughput_p50") or _float(
                "per_req_output_tps_p50"
            )
        if per_req_p99 is None:
            per_req_p99 = _float("output_throughput_p99") or _float(
                "per_req_output_tps_p99"
            )

    metrics = RunMetrics(
        concurrency=int(concurrency or 0),
        num_prompts=_int("num_prompts") or 0,
        isl=_int("input_len") or _int("median_input_len") or _int("isl") or 0,
        osl=_int("output_len") or _int("median_output_len") or _int("osl") or 0,
        output_tps_p50=per_req_p50,
        output_tps_p99=per_req_p99,
        ttft_p50_s=_ms_to_s("median_ttft_ms") or _float("ttft_p50_s"),
        ttft_p99_s=_ms_to_s("p99_ttft_ms") or _float("ttft_p99_s"),
        tpot_p50_ms=_float("median_tpot_ms"),
        tpot_p99_ms=_float("p99_tpot_ms"),
        goodput_under_slo=_float("goodput_under_slo"),
        selected_tokens_mean=_float("selected_tokens_mean"),
        dense_fallback_total=_int("dense_fallback_total"),
        total_tokens_mean=_float("total_tokens_mean"),
        duration_s=_float("duration"),
        # JSONL ``concurrency`` is the achieved/effective concurrency
        # bench_serving sustained; the nominal value above comes from
        # ``max_concurrency`` / the filename tag.
        achieved_concurrency=_float("concurrency"),
    )
    return context, metrics


def _match_or_refuse(
    baseline: RunContext, ds: RunContext, *, allow_gpu_mismatch: bool = False
) -> List[str]:
    """Return a list of human-readable mismatch reasons (empty = match).

    The AC-7 required-match set is ``{gpu_id, tp_size, page_size,
    disable_radix_cache, concurrency}``. All five are checked by default
    using the same rule: a field that is ``None`` on either side counts as
    a mismatch (``None == None`` is not a verified match) and unequal
    non-None values are a mismatch.

    ``allow_gpu_mismatch=True`` opts out of the ``gpu_id`` check for the
    rare deliberate cross-hardware comparison report.
    """

    reasons: List[str] = []
    required = [
        ("tp_size", baseline.tp_size, ds.tp_size),
        ("page_size", baseline.page_size, ds.page_size),
        ("disable_radix_cache", baseline.disable_radix_cache, ds.disable_radix_cache),
        ("concurrency", baseline.concurrency, ds.concurrency),
    ]
    if not allow_gpu_mismatch:
        required.insert(0, ("gpu_id", baseline.gpu_id, ds.gpu_id))
    for name, b, d in required:
        if b is None or d is None:
            reasons.append(
                f"{name} missing from one or both runs (None is not a match): "
                f"native_nsa={b!r} ds={d!r}"
            )
            continue
        if b != d:
            reasons.append(f"{name} mismatch: native_nsa={b!r} ds={d!r}")
    return reasons


# ----- AC-11 directional comparator (3-trial median, gates) -------------

# Plan §AC-11: "DS-on TPS within 5% of DSA-on TPS at conc=64 (directional
# gate). P99 TTFT ≤ DSA-on P99 TTFT × 1.10. Fixed seed, 600s window, 120s
# warmup, 3 trials, median."
AC11_TPS_FLOOR_RATIO = 0.95   # DS TPS must be ≥ 95% of DSA TPS.
AC11_TTFT_CEIL_RATIO = 1.10   # DS P99 TTFT must be ≤ 1.10× DSA P99 TTFT.
AC11_MIN_TRIALS = 3


def _median(values: List[Optional[float]]) -> Optional[float]:
    """Median ignoring ``None`` values. Returns ``None`` if empty after filtering."""
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    nums.sort()
    n = len(nums)
    mid = n // 2
    if n % 2 == 1:
        return float(nums[mid])
    return float((nums[mid - 1] + nums[mid]) / 2.0)


def _median_metrics(trials: List[RunMetrics]) -> RunMetrics:
    """Aggregate a trial set into a single ``RunMetrics`` via per-field medians.

    ``dense_fallback_total`` is summed (it's a counter, not a sample).
    ``concurrency``, ``num_prompts``, ``isl``, ``osl`` must agree across
    trials — refuse otherwise so a misgrouped sweep cannot silently pass.
    """
    if not trials:
        raise ValueError("AC-11 median: empty trial set")
    first = trials[0]
    for t in trials[1:]:
        for f in ("concurrency", "num_prompts", "isl", "osl"):
            if getattr(t, f) != getattr(first, f):
                raise ValueError(
                    f"AC-11 median: trial {f}={getattr(t, f)} disagrees with "
                    f"first trial {f}={getattr(first, f)} — refusing to median "
                    "across mismatched operating points."
                )
    df_total = None
    df_values = [t.dense_fallback_total for t in trials if t.dense_fallback_total is not None]
    if df_values:
        df_total = int(sum(df_values))
    return RunMetrics(
        concurrency=first.concurrency,
        num_prompts=first.num_prompts,
        isl=first.isl,
        osl=first.osl,
        output_tps_p50=_median([t.output_tps_p50 for t in trials]),
        output_tps_p99=_median([t.output_tps_p99 for t in trials]),
        ttft_p50_s=_median([t.ttft_p50_s for t in trials]),
        ttft_p99_s=_median([t.ttft_p99_s for t in trials]),
        tpot_p50_ms=_median([t.tpot_p50_ms for t in trials]),
        tpot_p99_ms=_median([t.tpot_p99_ms for t in trials]),
        goodput_under_slo=_median([t.goodput_under_slo for t in trials]),
        selected_tokens_mean=_median([t.selected_tokens_mean for t in trials]),
        dense_fallback_total=df_total,
        total_tokens_mean=_median([t.total_tokens_mean for t in trials]),
        duration_s=_median([t.duration_s for t in trials]),
        achieved_concurrency=_median([t.achieved_concurrency for t in trials]),
    )


def _group_by_concurrency(paths: List[str]) -> Dict[int, List[str]]:
    """Group ``paths`` by their resolved concurrency.

    Resolution order matches `_read_bench_jsonl`: per-row `max_concurrency`
    or `concurrency` (preferred — survives renames), then filename suffix
    `_c<N>.jsonl` as a last resort.

    Raises ``ValueError`` when a file is missing, contains malformed JSON,
    or has no resolvable concurrency. An earlier version swallowed
    parse errors and let `_run_ac11_mode` re-trip them as uncaught
    tracebacks.
    """
    by_conc: Dict[int, List[str]] = {}
    for p in paths:
        if not os.path.isfile(p):
            raise ValueError(f"AC-11 group: missing input file {p!r}.")
        # Try the file's row data first. Only fall back to filename when
        # the JSONL parsed cleanly but the parsed context lacks concurrency.
        try:
            ctx, _m = _read_bench_jsonl(p)
            conc = ctx.concurrency
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"AC-11 group: {p!r} is malformed JSONL ({exc}); refusing "
                "to publish."
            ) from exc
        except ValueError:
            # Empty file or shape-violation from _read_bench_jsonl — refuse.
            raise
        if conc is None:
            # Last-resort filename fallback.
            conc = _filename_concurrency(p)
        if conc is None:
            raise ValueError(
                f"AC-11 group: cannot resolve concurrency for {p!r}; "
                "ensure the JSONL row carries `max_concurrency`/`concurrency` "
                "or that the filename ends in `_c<N>.jsonl`."
            )
        by_conc.setdefault(int(conc), []).append(p)
    return by_conc


def _evaluate_ac11_gates(
    dsa_median: RunMetrics, ds_median: RunMetrics,
) -> Dict[str, object]:
    """Evaluate the two AC-11 directional gates for one concurrency.

    Returns a dict with the ratios + pass flags + a human-readable reason
    on failure. Missing inputs → ``"missing-data"`` reason and both gates
    treated as failed.
    """
    result: Dict[str, object] = {
        "tps_ratio": None,
        "ttft_ratio": None,
        "tps_pass": False,
        "ttft_pass": False,
        "reason": "",
    }
    if (
        dsa_median.output_tps_p50 is None
        or ds_median.output_tps_p50 is None
        or dsa_median.ttft_p99_s is None
        or ds_median.ttft_p99_s is None
    ):
        result["reason"] = (
            "missing-data: AC-11 gates need DSA + DS output_tps_p50 + "
            "ttft_p99_s on both sides."
        )
        return result
    if dsa_median.output_tps_p50 <= 0:
        result["reason"] = (
            f"degenerate DSA TPS={dsa_median.output_tps_p50!r} — refusing "
            "to compute ratio."
        )
        return result
    if dsa_median.ttft_p99_s <= 0:
        result["reason"] = (
            f"degenerate DSA P99 TTFT={dsa_median.ttft_p99_s!r} — refusing "
            "to compute ratio."
        )
        return result
    tps_ratio = float(ds_median.output_tps_p50) / float(dsa_median.output_tps_p50)
    ttft_ratio = float(ds_median.ttft_p99_s) / float(dsa_median.ttft_p99_s)
    tps_pass = tps_ratio >= AC11_TPS_FLOOR_RATIO
    ttft_pass = ttft_ratio <= AC11_TTFT_CEIL_RATIO
    reasons: List[str] = []
    if not tps_pass:
        reasons.append(
            f"AC-11 TPS gate failed: DS/DSA = {tps_ratio:.4f} < "
            f"{AC11_TPS_FLOOR_RATIO} (DS={ds_median.output_tps_p50:.2f} tok/s, "
            f"DSA={dsa_median.output_tps_p50:.2f} tok/s)"
        )
    if not ttft_pass:
        reasons.append(
            f"AC-11 TTFT gate failed: DS/DSA P99 = {ttft_ratio:.4f} > "
            f"{AC11_TTFT_CEIL_RATIO} (DS={ds_median.ttft_p99_s:.3f} s, "
            f"DSA={dsa_median.ttft_p99_s:.3f} s)"
        )
    result["tps_ratio"] = tps_ratio
    result["ttft_ratio"] = ttft_ratio
    result["tps_pass"] = tps_pass
    result["ttft_pass"] = ttft_pass
    result["reason"] = "; ".join(reasons) if reasons else ""
    return result


# AC-11 reproducibility floors per plan §AC-11 (fixed seed, 120s warmup,
# 600s measurement).
AC11_MIN_WARMUP_SECONDS = 120.0
AC11_MIN_MEASUREMENT_WINDOW_SECONDS = 600.0

# Server-args keys that legitimately differ between DSA and DS — the
# DS-enablement fields plus the DS-only radix-flip artifact path (the DSA
# baseline has no such field). The locked Option B set (radix cache, TP,
# page, dtype, backends, overlap/piecewise graph) is still enforced, so a
# radix-cache mismatch is refused, not absorbed.
_DS_ONLY_SERVER_ARG_KEYS = frozenset({
    "enable_double_sparsity",
    "double_sparsity_config",
    "double_sparsity_radix_fixture_artifact",
})

# Server-args fields that are NOT operating-point knobs for the DS-vs-DSA
# comparison and so must not refuse the run:
#   - ``random_seed``: a per-boot RNG seed (pure telemetry; the workload seed
#     is matched separately via the sidecar ``seed``).
#   - ``mem_fraction_static``: DS and DSA legitimately differ here — DS
#     reserves a per-rank TokenLabelTable on top of the V3.2 FP8 weights and
#     serves at 0.6, while the baseline serves at ~0.85. This asymmetry is the
#     root of the effective-vs-nominal concurrency gap, which the AC-11 report
#     surfaces explicitly (achieved concurrency per side) rather than hiding —
#     it is NOT a locked Option B field, so it is recorded, not used to refuse.
_AC11_IGNORED_SERVER_ARG_KEYS = frozenset({
    "random_seed",
    "mem_fraction_static",
})


# The AC-11 comparator must compare every stable ``ServerArgs`` launch
# field across DSA and DS — a hand-curated whitelist silently drops
# launch-flag mismatches such as ``disable_cuda_graph`` /
# ``trust_remote_code`` / ``dtype`` / ``max_total_tokens``. Derive
# the full set from ``dataclasses.fields(ServerArgs)`` so any new
# launch flag added to sglang is automatically protected.
#
# Everything else in ``/get_server_info`` (``internal_states``,
# ``kv_events``, ``step_time``, ``last_gen_throughput``,
# ``gpu_memory_used_bytes``, scheduler capacity, …) is dynamic
# telemetry that drifts between sequential trials — none of it is a
# ``ServerArgs`` field, so it is excluded by construction.

def _build_stable_launch_arg_keys() -> "frozenset[str]":
    try:
        # Import lazily so the module still loads in environments where
        # sglang is not on sys.path (the failure is then surfaced
        # loudly at AC-11 invocation rather than at import).
        from dataclasses import fields as _dc_fields
        from sglang.srt.server_args import ServerArgs as _ServerArgs
    except Exception as exc:
        raise RuntimeError(
            "AC-11 comparator requires sglang.srt.server_args.ServerArgs to "
            "derive the stable launch-args projection. Install sglang "
            "(`pip install -e python/` or `PYTHONPATH=python ...`). "
            f"Underlying import error: {exc}"
        ) from exc
    return frozenset(f.name for f in _dc_fields(_ServerArgs))


_AC11_STABLE_LAUNCH_ARG_KEYS = _build_stable_launch_arg_keys()

# Plan §13 (DEC-1): the locked Option B operating point. Both columns
# must publish all of these launch fields — comparator refuses if any
# are absent from the normalized projection. Missing fields would let a
# misconfigured server (e.g. DS launched without the FP8 KV flag) pass
# the comparator silently.
_AC11_OPTION_B_LOCKED_FIELDS = frozenset({
    "model_path",
    "tp_size",
    "page_size",
    "kv_cache_dtype",
    "dsa_prefill_backend",
    "dsa_decode_backend",
    "disable_overlap_schedule",
    "disable_piecewise_cuda_graph",
    "disable_radix_cache",
    "disable_cuda_graph",
})


def _sidecar_path(result_path: str) -> str:
    """Return the path of the ``.meta.json`` sidecar for a bench JSONL."""
    return result_path + ".meta.json"


def _require_sidecar_fields(meta: Dict, *, side: str, path: str) -> None:
    """Refuse sidecars missing reproducibility / workload fields.

    Treating ``None == None`` as "agreement" when both sidecars omit
    the same field is unsafe; enforcing presence + well-typedness
    inside the meta reader makes every later cross-side / per-side
    comparison work on real values.
    """
    seed = meta.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(
            f"AC-11 sidecar {side}={path}: seed must be an int, got "
            f"{seed!r}."
        )

    commit = meta.get("commit_sha")
    if not isinstance(commit, str) or not commit or commit == "unknown":
        raise ValueError(
            f"AC-11 sidecar {side}={path}: commit_sha must be a non-empty "
            f"string and not 'unknown', got {commit!r}."
        )

    chunked = meta.get("chunked_prefill_size")
    chunked_int_ok = (
        isinstance(chunked, int) and not isinstance(chunked, bool) and chunked > 0
    )
    chunked_unknown_ok = isinstance(chunked, str) and chunked == "unknown"
    if not (chunked_int_ok or chunked_unknown_ok):
        raise ValueError(
            f"AC-11 sidecar {side}={path}: chunked_prefill_size must be a "
            f"positive int or the string 'unknown' (the "
            f"_bench_meta_writer fallback), got {chunked!r}."
        )

    for field in ("num_prompts", "isl_total_tokens", "osl_tokens"):
        v = meta.get(field)
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise ValueError(
                f"AC-11 sidecar {side}={path}: {field} must be a positive "
                f"int, got {v!r}."
            )

    sa = meta.get("server_args")
    if not isinstance(sa, dict) or not sa:
        raise ValueError(
            f"AC-11 sidecar {side}={path}: server_args must be a non-empty "
            f"dict, got {sa!r}."
        )


def _read_ac11_meta(result_path: str, *, side: str = "?") -> Dict:
    """Read + validate the ``.meta.json`` sidecar for one bench JSONL.

    Raises ``ValueError`` (string-stable subclass of the standard exception
    hierarchy) when the sidecar is missing, unparseable, non-object,
    flagged with ``server_args_error``, or missing required
    reproducibility fields — so ``_run_ac11_mode`` can convert that into
    clean exit 2 + log.
    """
    sp = _sidecar_path(result_path)
    if not os.path.isfile(sp):
        raise ValueError(
            f"AC-11 sidecar missing: expected {sp}; the trial's `.meta.json` "
            "is required for reproducibility + apples-to-apples checks."
        )
    try:
        with open(sp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"AC-11 sidecar {sp}: malformed JSON ({exc})."
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"AC-11 sidecar {sp}: top-level value must be an object, got "
            f"{type(data).__name__}."
        )
    if data.get("server_args_error"):
        raise ValueError(
            f"AC-11 sidecar {sp}: server_args_error={data['server_args_error']!r} "
            "— refuse to publish until the source server_info was captured cleanly."
        )
    _require_sidecar_fields(data, side=side, path=sp)
    return data


def _normalize_ac11_server_args(meta: Dict) -> Dict:
    """Project ``meta["server_args"]`` onto the full ``ServerArgs`` field
    set minus the DS-only keys and the non-operating-point ignored keys.

    DS may legitimately differ from DSA on the DS-enablement fields and the
    DS-only radix-fixture artifact (``_DS_ONLY_SERVER_ARG_KEYS``), and on the
    recorded-not-matched fields (``_AC11_IGNORED_SERVER_ARG_KEYS``:
    ``random_seed`` per-boot telemetry, ``mem_fraction_static`` DS-vs-DSA
    memory asymmetry). All other launch args — TP size, page size, dtype,
    backends, radix cache, CUDA graph flag, max_total_tokens,
    attention_backend, etc. — must agree. Dynamic ``/get_server_info``
    telemetry is dropped because it is not a ``ServerArgs`` field.
    """
    sa = meta.get("server_args") or {}
    if not isinstance(sa, dict):
        return {}
    excluded = _DS_ONLY_SERVER_ARG_KEYS | _AC11_IGNORED_SERVER_ARG_KEYS
    return {
        k: sa[k] for k in sa.keys() & _AC11_STABLE_LAUNCH_ARG_KEYS
        if k not in excluded
    }


def _require_option_b_locked_fields(
    normalized: Dict, *, side: str, path: str,
) -> None:
    """Refuse normalized server_args projections that omit any of the
    locked Option B launch fields (plan §13 / DEC-1).

    Missing fields would let a misconfigured server (e.g. DS launched
    without ``--kv-cache-dtype fp8_e4m3``) pass the comparator silently.
    """
    missing = sorted(_AC11_OPTION_B_LOCKED_FIELDS - normalized.keys())
    if missing:
        raise ValueError(
            f"AC-11 sidecar {side}={path}: normalized server_args is "
            f"missing locked Option B field(s) {missing!r}; the sidecar "
            "must record every Option B launch flag (plan §13 / DEC-1)."
        )


def _validate_ac11_side_identity(
    meta: Dict, *, expected_side: str, path: str,
) -> None:
    """Refuse sidecars whose declared side identity does not match the
    expected column.

    Because ``_normalize_ac11_server_args`` strips the DS-enablement
    pair from cross-side comparison (those are the ONLY sanctioned
    differences), the comparator must separately prove that the DSA
    column is actually DSA-on and the DS column is actually DS-on.
    Otherwise two columns of native-NSA artifacts pass the gate with
    no DS measurement performed.

    For ``expected_side="DSA"`` the sidecar must declare
    ``mode="native_nsa"`` and must NOT carry the DS-enablement flags.
    For ``expected_side="DS"`` the sidecar must declare
    ``mode="double_sparsity"``, set
    ``server_args.enable_double_sparsity is True``, and supply a
    non-empty ``server_args.double_sparsity_config``.
    """
    if expected_side not in ("DSA", "DS"):
        raise ValueError(
            f"_validate_ac11_side_identity: unknown expected_side="
            f"{expected_side!r}."
        )
    mode = meta.get("mode")
    sa = meta.get("server_args") or {}
    enable = sa.get("enable_double_sparsity") if isinstance(sa, dict) else None
    config = sa.get("double_sparsity_config") if isinstance(sa, dict) else None

    if expected_side == "DSA":
        if mode != "native_nsa":
            raise ValueError(
                f"AC-11 DSA trial {path}: sidecar mode={mode!r} but "
                "expected 'native_nsa' for the DSA baseline column."
            )
        if enable:
            raise ValueError(
                f"AC-11 DSA trial {path}: sidecar declares "
                "enable_double_sparsity=True; the DSA baseline column "
                "must run with Double Sparsity OFF."
            )
        if isinstance(config, str) and config:
            raise ValueError(
                f"AC-11 DSA trial {path}: sidecar carries "
                f"double_sparsity_config={config!r}; the DSA baseline "
                "must not load a DS mask."
            )
    else:  # expected_side == "DS"
        if mode != "double_sparsity":
            raise ValueError(
                f"AC-11 DS trial {path}: sidecar mode={mode!r} but "
                "expected 'double_sparsity' for the DS-on column."
            )
        if enable is not True:
            raise ValueError(
                f"AC-11 DS trial {path}: sidecar declares "
                f"enable_double_sparsity={enable!r}; the DS-on column "
                "must launch with --enable-double-sparsity."
            )
        if not isinstance(config, str) or not config:
            raise ValueError(
                f"AC-11 DS trial {path}: sidecar "
                f"double_sparsity_config={config!r} is empty/missing; "
                "the DS-on column must point at the DS mask artifact."
            )


def _validate_trial_metrics(metrics: RunMetrics, *, side: str, path: str) -> None:
    """Refuse trials that lack the metrics the AC-11 gates need."""
    if metrics.output_tps_p50 is None:
        raise ValueError(
            f"AC-11 trial {side}={path}: missing output_tps_p50 metric "
            "(needed for TPS gate)."
        )
    if metrics.ttft_p99_s is None:
        raise ValueError(
            f"AC-11 trial {side}={path}: missing ttft_p99_s metric "
            "(needed for TTFT gate)."
        )


def _validate_meta_floors(meta: Dict, *, side: str, path: str) -> None:
    """Refuse trials whose sidecar doesn't meet the AC-11 timing floors."""
    warmup = meta.get("warmup_seconds")
    if not isinstance(warmup, (int, float)) or warmup < AC11_MIN_WARMUP_SECONDS:
        raise ValueError(
            f"AC-11 trial {side}={path}: warmup_seconds={warmup!r} does not "
            f"meet the AC-11 minimum of {AC11_MIN_WARMUP_SECONDS}s."
        )
    win = meta.get("measurement_window_seconds")
    if not isinstance(win, (int, float)) or win < AC11_MIN_MEASUREMENT_WINDOW_SECONDS:
        raise ValueError(
            f"AC-11 trial {side}={path}: measurement_window_seconds={win!r} "
            f"does not meet the AC-11 minimum of "
            f"{AC11_MIN_MEASUREMENT_WINDOW_SECONDS}s."
        )


def _validate_jsonl_workload_matches_sidecar(
    metrics: RunMetrics, meta: Dict, *, side: str, path: str,
) -> None:
    """Refuse trials where the JSONL workload disagrees with the sidecar.

    The sidecar is written from env vars before/after the bench run; the
    JSONL is the canonical record of what actually executed. If they
    disagree, the sidecar is lying about the workload that ran and
    every later per-side / cross-side workload-agreement check is
    wrong.
    """
    pairs = (
        ("num_prompts", metrics.num_prompts, meta.get("num_prompts")),
        ("input_len", metrics.isl, meta.get("isl_total_tokens")),
        ("output_len", metrics.osl, meta.get("osl_tokens")),
    )
    for field, jsonl_v, side_v in pairs:
        # Treat 0 as "not emitted by the JSONL" — `_read_bench_jsonl`
        # uses `int(...) or 0` as the default for missing summary keys.
        if not jsonl_v:
            continue
        if side_v != jsonl_v:
            raise ValueError(
                f"AC-11 trial {side}={path}: JSONL summary {field}="
                f"{jsonl_v!r} disagrees with sidecar="
                f"{side_v!r}; the sidecar must reflect the workload "
                "that actually ran."
            )


def _validate_jsonl_duration(
    metrics: RunMetrics, *, side: str, path: str,
) -> None:
    """Refuse trials whose bench_serving wall-clock is below the AC-11 floor.

    Trusting only the sidecar's
    ``measurement_window_seconds`` field even when the JSONL ``duration``
    reported only a few seconds (operator forgot to size ``num_prompts``
    for the requested window). bench_serving's ``duration`` is the
    wall-clock duration of the measured phase, so it is the right
    end-to-end floor.
    """
    d = metrics.duration_s
    if d is None:
        raise ValueError(
            f"AC-11 trial {side}={path}: bench_serving JSONL is missing "
            "the `duration` field; cannot verify the AC-11 measurement "
            "window."
        )
    if d < AC11_MIN_MEASUREMENT_WINDOW_SECONDS:
        raise ValueError(
            f"AC-11 trial {side}={path}: bench_serving wall-clock "
            f"duration={d!r}s is below the AC-11 measurement-window floor "
            f"of {AC11_MIN_MEASUREMENT_WINDOW_SECONDS}s."
        )


def _validate_per_side_agreement(metas: List[Dict], paths: List[str], *, side: str) -> None:
    """All trials within one side must share seed/commit/chunked/server-args
    and ``mem_fraction_static`` (the latter is ignored cross-side but must be
    constant within a side)."""
    if not metas:
        return
    first = metas[0]
    first_norm = _normalize_ac11_server_args(first)
    first_memfrac = (first.get("server_args") or {}).get("mem_fraction_static")
    for m, p in zip(metas[1:], paths[1:]):
        for field in ("seed", "commit_sha", "chunked_prefill_size",
                      "num_prompts", "isl_total_tokens", "osl_tokens"):
            if m.get(field) != first.get(field):
                raise ValueError(
                    f"AC-11 {side}: trial {p} {field}={m.get(field)!r} "
                    f"disagrees with first trial {paths[0]} "
                    f"{field}={first.get(field)!r}."
                )
        if _normalize_ac11_server_args(m) != first_norm:
            raise ValueError(
                f"AC-11 {side}: trial {p} normalized server_args disagrees "
                f"with first trial {paths[0]}."
            )
        # mem_fraction_static is recorded-not-matched ACROSS sides (DSA 0.85
        # vs DS 0.6 is the sanctioned asymmetry, so it is excluded from the
        # normalized projection above), but it must be CONSTANT WITHIN a side
        # — otherwise the comparator would median across per-side mismatched
        # launch knobs (e.g. DSA 0.85/0.80/0.75) without refusing.
        m_memfrac = (m.get("server_args") or {}).get("mem_fraction_static")
        if m_memfrac != first_memfrac:
            raise ValueError(
                f"AC-11 {side}: trial {p} mem_fraction_static={m_memfrac!r} "
                f"disagrees with first trial {paths[0]} "
                f"mem_fraction_static={first_memfrac!r} — within-side launch "
                "knobs must be constant (the DSA-vs-DS cross-side mem-fraction "
                "asymmetry is allowed; per-side drift is not)."
            )


def _validate_cross_side_agreement(
    dsa_meta: Dict, ds_meta: Dict, *, conc: int,
) -> None:
    """One DSA trial vs one DS trial must agree on seed / commit / workload /
    normalized server args. Hardware (gpu/tp/page/radix) is checked
    separately via _match_or_refuse on the JSONL contexts."""
    for field in ("seed", "commit_sha", "chunked_prefill_size",
                  "num_prompts", "isl_total_tokens", "osl_tokens"):
        if dsa_meta.get(field) != ds_meta.get(field):
            raise ValueError(
                f"AC-11 conc={conc}: DSA {field}={dsa_meta.get(field)!r} "
                f"disagrees with DS {field}={ds_meta.get(field)!r} — refuse "
                "to publish apples-vs-oranges comparator."
            )
    dsa_norm = _normalize_ac11_server_args(dsa_meta)
    ds_norm = _normalize_ac11_server_args(ds_meta)
    if dsa_norm != ds_norm:
        diff = sorted(
            (k, dsa_norm.get(k, "<missing>"), ds_norm.get(k, "<missing>"))
            for k in dsa_norm.keys() | ds_norm.keys()
            if dsa_norm.get(k) != ds_norm.get(k)
        )
        raise ValueError(
            f"AC-11 conc={conc}: normalized launch-args server_args differ "
            f"between DSA and DS on {[k for k, _, _ in diff]!r}: {diff!r}. "
            f"Only {sorted(_DS_ONLY_SERVER_ARG_KEYS)} may differ between "
            "sides (plan §AC-11)."
        )


def _render_ac11_markdown(
    by_conc: Dict[int, Dict[str, object]],
) -> str:
    """Render the AC-11 per-concurrency report."""
    rows: List[str] = []
    rows.append("# AC-11 Directional Comparator — DS vs DSA")
    rows.append("")
    rows.append(
        f"Gates: DS TPS ≥ {AC11_TPS_FLOOR_RATIO * 100:.0f}% of DSA TPS; "
        f"DS P99 TTFT ≤ DSA P99 TTFT × {AC11_TTFT_CEIL_RATIO:.2f}. "
        f"At least {AC11_MIN_TRIALS} trials per concurrency, median."
    )
    rows.append("")
    rows.append(
        "| Conc | DSA TPS p50 | DS TPS p50 | TPS ratio | TPS gate | "
        "DSA TTFT p99 | DS TTFT p99 | TTFT ratio | TTFT gate |"
    )
    rows.append(
        "|------|-------------|------------|-----------|----------|"
        "--------------|-------------|------------|-----------|"
    )

    def _fmt(v):
        return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    overall_fail_reasons: List[str] = []
    for conc in sorted(by_conc.keys()):
        row = by_conc[conc]
        dsa_m: RunMetrics = row["dsa_median"]  # type: ignore[assignment]
        ds_m: RunMetrics = row["ds_median"]    # type: ignore[assignment]
        gate: Dict[str, object] = row["gate"]  # type: ignore[assignment]
        tps_gate = "pass" if gate["tps_pass"] else "FAIL"
        ttft_gate = "pass" if gate["ttft_pass"] else "FAIL"
        rows.append(
            f"| {conc} "
            f"| {_fmt(dsa_m.output_tps_p50)} "
            f"| {_fmt(ds_m.output_tps_p50)} "
            f"| {_fmt(gate.get('tps_ratio'))} "
            f"| {tps_gate} "
            f"| {_fmt(dsa_m.ttft_p99_s)} "
            f"| {_fmt(ds_m.ttft_p99_s)} "
            f"| {_fmt(gate.get('ttft_ratio'))} "
            f"| {ttft_gate} |"
        )
        if not gate["tps_pass"] or not gate["ttft_pass"]:
            overall_fail_reasons.append(f"conc={conc}: {gate['reason']}")
    rows.append("")

    # Effective-vs-nominal concurrency (#F): DS at mem 0.6 reserves a per-rank
    # TokenLabelTable on top of the V3.2 FP8 weights, so its KV pool is smaller
    # and it can be admission/queue-bound at high nominal concurrency. Surface
    # the ACHIEVED concurrency each side actually sustained so a TTFT gap that
    # is partly an admission artifact (not pure per-request latency) is visible.
    rows.append("## Effective vs nominal concurrency (#F)")
    rows.append("")
    rows.append("| Conc (nominal) | DSA achieved | DS achieved | DS/nominal |")
    rows.append("|----------------|--------------|-------------|------------|")
    for conc in sorted(by_conc.keys()):
        dsa_m: RunMetrics = by_conc[conc]["dsa_median"]  # type: ignore[assignment]
        ds_m: RunMetrics = by_conc[conc]["ds_median"]    # type: ignore[assignment]
        ds_ach = ds_m.achieved_concurrency
        frac = f"{ds_ach / conc:.0%}" if (ds_ach is not None and conc) else "—"
        rows.append(
            f"| {conc} | {_fmt(dsa_m.achieved_concurrency)} "
            f"| {_fmt(ds_ach)} | {frac} |"
        )
    rows.append("")
    rows.append(
        "When DS achieved concurrency is below nominal while DSA tracks nominal, "
        "the DS P99 TTFT gap is partly queue/admission-bound (a mem-0.6 KV-pool "
        "effect), not solely per-request latency. Per DEC-7 a TTFT/TPS miss is a "
        "recorded directional follow-up, not a build-break."
    )
    rows.append("")
    if overall_fail_reasons:
        rows.append("## AC-11 verdict: FAIL")
        rows.append("")
        rows.append(
            "**Profiling obligation:** the failing concurrencies below "
            "require a captured profile (`development/profile_ds.sh` or "
            "equivalent) before the comparator row can be published."
        )
        rows.append("")
        for reason in overall_fail_reasons:
            rows.append(f"- {reason}")
    else:
        rows.append("## AC-11 verdict: PASS")
    return "\n".join(rows) + "\n"


def _slo_verdict(m: RunMetrics) -> str:
    # output_tps_p50 is the per-request decode throughput (output_tokens /
    # (e2e - ttft)); the TTFT bar is the plan's strict "P99 TTFT < 22 s".
    if m.output_tps_p50 is None or m.ttft_p99_s is None:
        return "missing-data"
    if m.output_tps_p50 >= SLO_PER_REQUEST_TPS_P50 and m.ttft_p99_s < SLO_TTFT_P99_S:
        return "pass"
    return "fail"


def _no_op_status(m: RunMetrics) -> Tuple[str, str]:
    """Return ``(status, reason)`` where status is ``clean`` / ``triggered`` / ``unknown``.

    Returning ``unknown`` when the DS observability fields are absent means
    the report surfaces "we cannot evaluate" instead of falsely printing
    "clean". ``bench_serving`` does NOT emit these fields by default — they
    come from a separate observability path the deploying team must wire.
    """

    missing: List[str] = []
    if m.dense_fallback_total is None:
        missing.append("dense_fallback_total")
    if m.selected_tokens_mean is None:
        missing.append("selected_tokens_mean")
    if m.total_tokens_mean is None:
        missing.append("total_tokens_mean")
    if missing:
        return ("unknown", "no-op inputs missing: " + ", ".join(missing))
    if m.dense_fallback_total != 0:
        return ("triggered", f"dense_fallback={m.dense_fallback_total}")
    if m.selected_tokens_mean == m.total_tokens_mean:
        return ("triggered", "selected_tokens == total_tokens")
    return ("clean", "")


def render_markdown_report(
    baseline_metrics: RunMetrics,
    ds_metrics: RunMetrics,
    *,
    baseline_path: str,
    ds_path: str,
) -> str:
    rows = []
    rows.append("# Double Sparsity vs Native NSA — Comparison Report")
    rows.append("")
    rows.append(f"- native_nsa source: `{baseline_path}`")
    rows.append(f"- double_sparsity source: `{ds_path}`")
    rows.append(f"- concurrency: {ds_metrics.concurrency}")
    rows.append("")
    rows.append("| Metric | native_nsa | double_sparsity |")
    rows.append("|--------|------------|-----------------|")

    def _fmt(v):
        return "—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))

    pairs = [
        ("Per-request decode tok/s P50 (out_tok/(e2e-ttft))", baseline_metrics.output_tps_p50, ds_metrics.output_tps_p50),
        ("Per-request decode tok/s P99", baseline_metrics.output_tps_p99, ds_metrics.output_tps_p99),
        ("TTFT P50 (s)", baseline_metrics.ttft_p50_s, ds_metrics.ttft_p50_s),
        ("TTFT P99 (s)", baseline_metrics.ttft_p99_s, ds_metrics.ttft_p99_s),
        ("TPOT P50 (ms)", baseline_metrics.tpot_p50_ms, ds_metrics.tpot_p50_ms),
        ("TPOT P99 (ms)", baseline_metrics.tpot_p99_ms, ds_metrics.tpot_p99_ms),
        ("Goodput-under-SLO", baseline_metrics.goodput_under_slo, ds_metrics.goodput_under_slo),
        ("Selected tokens (mean)", baseline_metrics.selected_tokens_mean, ds_metrics.selected_tokens_mean),
        ("Total tokens (mean)", baseline_metrics.total_tokens_mean, ds_metrics.total_tokens_mean),
        ("dense_fallback_total", baseline_metrics.dense_fallback_total, ds_metrics.dense_fallback_total),
    ]
    for label, a, b in pairs:
        rows.append(f"| {label} | {_fmt(a)} | {_fmt(b)} |")

    rows.append("")
    rows.append(f"**DS SLO verdict (per-request decode P50 ≥ {SLO_PER_REQUEST_TPS_P50} tok/s, P99 TTFT < {SLO_TTFT_P99_S} s):** {_slo_verdict(ds_metrics)}")
    ds_status, ds_reason = _no_op_status(ds_metrics)
    if ds_status == "clean":
        rows.append("**No-op detector:** clean")
    elif ds_status == "triggered":
        rows.append(f"**No-op detector:** triggered ({ds_reason})")
    else:
        rows.append(f"**No-op detector:** unknown ({ds_reason})")
    return "\n".join(rows) + "\n"


def _run_ac11_mode(args) -> int:
    """AC-11 mode entry point: validate trial sets and enforce gates."""
    if not args.ac11_baseline_results or not args.ac11_ds_results:
        logger.error(
            "AC-11 mode requires --ac11-baseline-results and --ac11-ds-results."
        )
        return 2

    try:
        dsa_by_conc = _group_by_concurrency(args.ac11_baseline_results)
        ds_by_conc = _group_by_concurrency(args.ac11_ds_results)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("AC-11 input refusal: %s", exc)
        return 2

    if set(dsa_by_conc.keys()) != set(ds_by_conc.keys()):
        logger.error(
            "AC-11 input refusal: concurrency sets disagree. "
            "DSA: %s, DS: %s",
            sorted(dsa_by_conc.keys()), sorted(ds_by_conc.keys()),
        )
        return 2

    for conc, paths in dsa_by_conc.items():
        if len(paths) < AC11_MIN_TRIALS:
            logger.error(
                "AC-11 input refusal: DSA conc=%d has only %d trial(s); "
                "need >=%d.",
                conc, len(paths), AC11_MIN_TRIALS,
            )
            return 2
    for conc, paths in ds_by_conc.items():
        if len(paths) < AC11_MIN_TRIALS:
            logger.error(
                "AC-11 input refusal: DS conc=%d has only %d trial(s); "
                "need >=%d.",
                conc, len(paths), AC11_MIN_TRIALS,
            )
            return 2

    by_conc: Dict[int, Dict[str, object]] = {}
    any_fail = False

    def _read_pair(p: str, *, side: str) -> Tuple[RunContext, RunMetrics, Dict]:
        ctx, m = _read_bench_jsonl(p)
        meta = _read_ac11_meta(p, side=side)
        return ctx, m, meta

    for conc in sorted(dsa_by_conc.keys()):
        # Wrap the heavy read pass so any parse/refusal raises become
        # exit-2 with a clean log, not tracebacks.
        try:
            dsa_rows = [_read_pair(p, side="DSA") for p in dsa_by_conc[conc]]
            ds_rows = [_read_pair(p, side="DS") for p in ds_by_conc[conc]]
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error("AC-11 input refusal at conc=%d: %s", conc, exc)
            return 2

        dsa_ctxs = [row[0] for row in dsa_rows]
        dsa_trials = [row[1] for row in dsa_rows]
        dsa_metas = [row[2] for row in dsa_rows]
        ds_ctxs = [row[0] for row in ds_rows]
        ds_trials = [row[1] for row in ds_rows]
        ds_metas = [row[2] for row in ds_rows]

        try:
            # Per-trial: required metrics, JSONL duration floor, sidecar
            # timing floors.
            for m, p in zip(dsa_trials, dsa_by_conc[conc]):
                _validate_trial_metrics(m, side="DSA", path=p)
                _validate_jsonl_duration(m, side="DSA", path=p)
            for m, p in zip(ds_trials, ds_by_conc[conc]):
                _validate_trial_metrics(m, side="DS", path=p)
                _validate_jsonl_duration(m, side="DS", path=p)
            for meta, p, m in zip(dsa_metas, dsa_by_conc[conc], dsa_trials):
                _validate_meta_floors(meta, side="DSA", path=p)
                _validate_ac11_side_identity(meta, expected_side="DSA", path=p)
                if meta.get("concurrency") != conc:
                    raise ValueError(
                        f"AC-11 DSA trial {p}: sidecar "
                        f"concurrency={meta.get('concurrency')!r} does not "
                        f"match the grouping concurrency={conc}."
                    )
                _validate_jsonl_workload_matches_sidecar(
                    m, meta, side="DSA", path=p,
                )
                _require_option_b_locked_fields(
                    _normalize_ac11_server_args(meta),
                    side="DSA", path=p,
                )
            for meta, p, m in zip(ds_metas, ds_by_conc[conc], ds_trials):
                _validate_meta_floors(meta, side="DS", path=p)
                _validate_ac11_side_identity(meta, expected_side="DS", path=p)
                if meta.get("concurrency") != conc:
                    raise ValueError(
                        f"AC-11 DS trial {p}: sidecar "
                        f"concurrency={meta.get('concurrency')!r} does not "
                        f"match the grouping concurrency={conc}."
                    )
                _validate_jsonl_workload_matches_sidecar(
                    m, meta, side="DS", path=p,
                )
                _require_option_b_locked_fields(
                    _normalize_ac11_server_args(meta),
                    side="DS", path=p,
                )
            # Per-side trial-set agreement.
            _validate_per_side_agreement(dsa_metas, dsa_by_conc[conc], side="DSA")
            _validate_per_side_agreement(ds_metas, ds_by_conc[conc], side="DS")
            # Cross-side agreement (workload, seed, commit, server_args mod
            # DS-only keys). Compare first trial of each side — within-side
            # agreement was already enforced above.
            _validate_cross_side_agreement(dsa_metas[0], ds_metas[0], conc=conc)
            # Hardware match (gpu/tp/page/radix/concurrency) via existing
            # AC-7 helper. Allow gpu mismatch only if user requested it.
            # `disable_radix_cache` mismatch is NOT filtered out anymore —
            # AC-11 depends on AC-10 having removed --disable-radix-cache
            # from the DS launcher.
            hw_reasons = _match_or_refuse(
                dsa_ctxs[0], ds_ctxs[0],
                allow_gpu_mismatch=bool(getattr(args, "allow_gpu_mismatch", False)),
            )
            if hw_reasons:
                raise ValueError(
                    f"AC-11 conc={conc}: hardware/operating-point mismatch — "
                    + "; ".join(hw_reasons)
                )
        except ValueError as exc:
            logger.error("AC-11 input refusal at conc=%d: %s", conc, exc)
            return 2

        try:
            dsa_median = _median_metrics(dsa_trials)
            ds_median = _median_metrics(ds_trials)
        except ValueError as exc:
            logger.error("AC-11 median refusal at conc=%d: %s", conc, exc)
            return 2
        gate = _evaluate_ac11_gates(dsa_median, ds_median)
        by_conc[conc] = {
            "dsa_median": dsa_median, "ds_median": ds_median, "gate": gate,
        }
        if not gate["tps_pass"] or not gate["ttft_pass"]:
            any_fail = True

    md = _render_ac11_markdown(by_conc)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info("wrote AC-11 Markdown report to %s", args.output)
    else:
        sys.stdout.write(md)
    if args.json_output:
        payload = {
            "ac11_gates": {
                "tps_floor_ratio": AC11_TPS_FLOOR_RATIO,
                "ttft_ceil_ratio": AC11_TTFT_CEIL_RATIO,
                "min_trials": AC11_MIN_TRIALS,
            },
            "per_concurrency": {
                str(conc): {
                    "dsa_median": asdict(row["dsa_median"]),
                    "ds_median": asdict(row["ds_median"]),
                    "gate": row["gate"],
                }
                for conc, row in by_conc.items()
            },
            "verdict": "FAIL" if any_fail else "PASS",
        }
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("wrote AC-11 JSON report to %s", args.json_output)

    return 3 if any_fail else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark_compare.py",
        description="Side-by-side comparator for native_nsa vs double_sparsity bench_serving runs.",
    )
    parser.add_argument(
        "--ac11",
        action="store_true",
        help=(
            "AC-11 directional comparator mode. Accepts >=3 trial JSONLs per "
            "mode per concurrency via --ac11-baseline-results / "
            "--ac11-ds-results; computes per-concurrency medians and "
            "enforces DS TPS >= 0.95 * DSA TPS and DS P99 TTFT <= 1.10 * "
            "DSA P99 TTFT. Exit 0 on pass, 3 on gate failure, 2 on input "
            "refusal (too few trials, mismatched concurrency set)."
        ),
    )
    parser.add_argument(
        "--ac11-baseline-results", nargs="+", default=None,
        help="AC-11 mode only: paths to >=3 DSA baseline trial JSONLs per concurrency.",
    )
    parser.add_argument(
        "--ac11-ds-results", nargs="+", default=None,
        help="AC-11 mode only: paths to >=3 DS trial JSONLs per concurrency.",
    )
    parser.add_argument(
        "--baseline", default=None,
        help="Single-trial mode: path to native_nsa *.jsonl (AC-7/AC-8 report).",
    )
    parser.add_argument(
        "--ds", default=None,
        help="Single-trial mode: path to double_sparsity *.jsonl (AC-7/AC-8 report).",
    )
    parser.add_argument(
        "--output", default=None, help="Write Markdown report to this path (default stdout)."
    )
    parser.add_argument(
        "--json-output", default=None, help="Write JSON report to this path."
    )
    parser.add_argument(
        "--allow-gpu-mismatch",
        action="store_true",
        help=(
            "Skip the gpu_id match check. By default gpu_id is required to "
            "agree between the two runs (per AC-7's match-enforcement "
            "contract); use this only for deliberate cross-hardware "
            "comparison reports."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.ac11:
        return _run_ac11_mode(args)

    if not (args.baseline and args.ds):
        parser.error(
            "Single-trial mode requires both --baseline and --ds. "
            "For the AC-11 directional gate pass --ac11 + "
            "--ac11-baseline-results + --ac11-ds-results."
        )

    baseline_ctx, baseline_m = _read_bench_jsonl(args.baseline)
    ds_ctx, ds_m = _read_bench_jsonl(args.ds)

    reasons = _match_or_refuse(
        baseline_ctx, ds_ctx, allow_gpu_mismatch=args.allow_gpu_mismatch,
    )
    if reasons:
        logger.error(
            "Refusing to publish two-column report — context disagrees:\n  %s",
            "\n  ".join(reasons),
        )
        return 2

    md = render_markdown_report(
        baseline_m, ds_m, baseline_path=args.baseline, ds_path=args.ds
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info("wrote Markdown report to %s", args.output)
    else:
        sys.stdout.write(md)
    if args.json_output:
        ds_status, ds_reason = _no_op_status(ds_m)
        payload = {
            "baseline_path": args.baseline,
            "ds_path": args.ds,
            "baseline_context": asdict(baseline_ctx),
            "ds_context": asdict(ds_ctx),
            "baseline_metrics": asdict(baseline_m),
            "ds_metrics": asdict(ds_m),
            "ds_slo_verdict": _slo_verdict(ds_m),
            "ds_no_op_flag": {"status": ds_status, "reason": ds_reason},
        }
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("wrote JSON report to %s", args.json_output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
