"""Timing records for the MoE-LoRA laboratory.

The measurement engine is the repo's own kernel-benchmark harness,
``sglang.kernels.jit.benchmark.marker`` — CUDA-event timing, warmup, quantile
metrics, graph-replay mode, capacity-derived L2 rotation, and bandwidth from a
declared memory footprint.  This module adds only what the campaign's evidence
discipline needs on top of it (execution plan section 31.2):

1. a JSON sink of content-addressed records, so a number can be re-adjudicated
   without a rerun;
2. a declared measurement BOUNDARY on every record, because plan section 10
   forbids comparing across boundaries;
3. matched-base pairing, so a LoRA measurement carries the base-only
   denominator measured in the SAME session rather than a remembered one;
4. an explicit cache-state selector, because plan section 7.4 requires both a
   cold and a hot reading and the default rotation only produces the former.

Nothing here decides anything.  Selection lives in the gate packets.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from typing import Any, Callable

import msgspec
import torch

from benchmark.kernels.lora_moe.crossover_ledger import CrossoverLedgerEntry
from sglang.kernels.jit.benchmark.marker import BenchResult, do_bench

# Plan section 10. A comparison may only be made between records that carry the
# same boundary string.
BOUNDARY_ISOLATED = "isolated"
BOUNDARY_PREPARED_INPUT = "prepared_input"
BOUNDARY_ROUTE_INCLUSIVE = "route_inclusive"
BOUNDARY_COMPLETE_LOCAL_MOE = "complete_local_moe"
BOUNDARIES = (
    BOUNDARY_ISOLATED,
    BOUNDARY_PREPARED_INPUT,
    BOUNDARY_ROUTE_INCLUSIVE,
    BOUNDARY_COMPLETE_LOCAL_MOE,
)

# Cache state is DERIVED from the timing mode, not chosen by the caller.
#
# Methodology-audit finding (2026-07-25): the previous "cold"/"hot" selector
# was a NO-OP for zero-argument thunks — marker's rotation machinery sizes the
# rotation from the nbytes of input_args, and a closure passes none, so
# rotate_count was always 1 and every record labelled "cold" was in fact a
# hot-L2 steady state.  The labels below say what marker actually produces:
#
# "l2_hot_graph": use_cuda_graph=True with a closure.  100 in-graph iterations
# on the same addresses; median is a hot-L2 steady state.  CPU launch cost is
# EXCLUDED — replay does not re-run Python or launches.
#
# "l2_flushed_eager": use_cuda_graph=False.  marker zeroes an L2-sized buffer
# before every timed iteration, so inputs are genuinely cold, and the timing
# INCLUDES per-call CPU launch work between the events on the stream.
#
# A producer-realistic state still requires timing the producer+consumer pair
# at a common boundary (plan section 7.4); no flag here can produce it.
CACHE_L2_HOT_GRAPH = "l2_hot_graph"
CACHE_L2_FLUSHED_EAGER = "l2_flushed_eager"
CACHE_STATES = (CACHE_L2_HOT_GRAPH, CACHE_L2_FLUSHED_EAGER)


class TimingRecord(msgspec.Struct, frozen=True, kw_only=True):
    """One measured candidate at one declared boundary."""

    record_id: str
    candidate: str
    boundary: str
    cache_state: str
    params: dict[str, Any]
    median_s: float
    mean_s: float
    # Quartiles of marker's timing samples (each sample is itself a mean over
    # the in-graph loop in graph mode). The spread is decision input — a
    # winner claim needs disjoint spreads, not just ordered medians.
    p25_s: float
    p75_s: float
    replicate_s: tuple[float, ...]
    memory_footprint_bytes: int | None
    bandwidth_gib_s: float | None
    graph_replay: bool
    device_name: str
    source_revision: str
    # Set by `pair_with_base`; the matched base-only denominator for this
    # measurement, measured in the same session at the same boundary.
    base_record_id: str | None = None
    ratio_to_base: float | None = None


class TimingSuite(msgspec.Struct, kw_only=True):
    """A session's records plus the provenance needed to re-read them."""

    suite: str
    device_name: str
    source_revision: str
    torch_version: str
    host: str
    # What the running tree resolved to at suite creation (may be
    # "unknown" on file-synced pod trees) — recorded alongside the
    # caller's claimed source_revision, never instead of it.
    observed_revision: str = "unknown"
    # Content digest of the measuring source files (works on file-synced
    # trees where git state is absent) — see content_fingerprint().
    source_digest: str = "unknown"
    records: list[TimingRecord] = []
    # §31.7 crossover rows found in THIS session; serialized with the suite so
    # the gate packet's ledger is copied from an archive, not reconstructed.
    ledger: list[CrossoverLedgerEntry] = []

    def add(self, record: TimingRecord) -> TimingRecord:
        self.records.append(record)
        return record

    def site_crossover(
        self,
        *,
        site: str,
        boundary: str,
        candidates: tuple[str, ...],
        axis: str,
        crossover_location: str,
        bracketing_low_record_ids: tuple[str, ...],
        bracketing_high_record_ids: tuple[str, ...],
        cache_state: str,
        axis_param: str | None = None,
        workload_params: tuple[str, ...] | None = None,
        notes: str = "",
    ) -> CrossoverLedgerEntry:
        """Append one evidence-bound §31.7 ledger row.

        Provenance comes from THIS suite; every bracketing record ID must
        identify a record the suite measured at the declared boundary and
        cache state; and EVERY declared candidate must have records in
        BOTH bracketing cells — a crossover between two cells cannot be
        claimed off a cell that measured only one arm.
        """
        if set(bracketing_low_record_ids) == set(bracketing_high_record_ids):
            raise ValueError(
                "the low and high bracketing cells cite identical records — "
                "a crossover needs two DISTINCT cells (fourth S3 review)"
            )
        if axis_param is not None:
            # Fifth S3 review + sixth-review fix: distinct record IDs do not
            # prove distinct AXIS cells, and comparing EVERY parameter breaks
            # the real producer (candidates legitimately record their own
            # tuning configs). The WORKLOAD signature is therefore explicit
            # and validated separately from candidate-specific configuration:
            # each cell sits at exactly ONE axis value present on every
            # record; the two cells differ on the axis; every declared
            # workload parameter is single-valued across BOTH cells; and one
            # candidate's records within one cell agree on ALL their
            # parameters (config drift inside an arm's cell is a bug).
            if workload_params is None:
                raise ValueError(
                    "axis_param validation needs the explicit workload "
                    "signature — pass workload_params (sixth S3 review: an "
                    "implicit all-params signature was both fail-open across "
                    "cells and wrong for per-candidate configs)"
                )
            by_id_pre = {record.record_id: record for record in self.records}
            excluded = {"seed", "repeat", "case_id"}

            def cell_signature(ids, cell_name):
                axis_values = set()
                per_candidate: dict[str, bytes] = {}
                workload: dict[str, set[bytes]] = {}
                for record_id in ids:
                    record = by_id_pre.get(record_id)
                    if record is None:
                        continue  # the main loop below raises properly
                    params = dict(record.params)
                    if axis_param not in params:
                        raise ValueError(
                            f"{cell_name} record {record_id!r} lacks the "
                            f"axis parameter {axis_param!r}"
                        )
                    axis_values.add(params.pop(axis_param))
                    for key in excluded:
                        params.pop(key, None)
                    frozen = msgspec.json.encode(
                        {key: params[key] for key in sorted(params)}
                    )
                    prior = per_candidate.setdefault(record.candidate, frozen)
                    if prior != frozen:
                        raise ValueError(
                            f"{cell_name} cell records for candidate "
                            f"{record.candidate!r} disagree on non-axis "
                            "parameters"
                        )
                    for key in workload_params:
                        if key == axis_param:
                            continue
                        if key not in params:
                            raise ValueError(
                                f"{cell_name} record {record_id!r} lacks the "
                                f"declared workload parameter {key!r}"
                            )
                        workload.setdefault(key, set()).add(
                            msgspec.json.encode(params[key])
                        )
                if len(axis_values) != 1:
                    raise ValueError(
                        f"the {cell_name} cell must sit at exactly one "
                        f"{axis_param!r} value; it cites {sorted(map(str, axis_values))}"
                    )
                return axis_values, workload

            low_values, low_workload = cell_signature(bracketing_low_record_ids, "low")
            high_values, high_workload = cell_signature(
                bracketing_high_record_ids, "high"
            )
            if low_values == high_values:
                raise ValueError(
                    f"low and high cells sit at the same {axis_param!r} value "
                    f"{sorted(map(str, low_values))} — a crossover needs two "
                    "distinct axis cells (fifth S3 review)"
                )
            for key in workload_params:
                if key == axis_param:
                    continue
                values = low_workload.get(key, set()) | high_workload.get(key, set())
                if len(values) != 1:
                    raise ValueError(
                        f"workload parameter {key!r} is not single-valued "
                        "across the two bracketing cells — the cells compare "
                        "different workloads, not two points on one axis"
                    )
        by_id = {record.record_id: record for record in self.records}
        case_ids: list[str] = []
        for cell_name, cell_ids in (
            ("low", bracketing_low_record_ids),
            ("high", bracketing_high_record_ids),
        ):
            cited: set[str] = set()
            for record_id in cell_ids:
                record = by_id.get(record_id)
                if record is None:
                    raise ValueError(
                        f"bracketing record {record_id!r} is not in this suite"
                    )
                if record.boundary != boundary or record.cache_state != cache_state:
                    raise ValueError(
                        f"bracketing record {record_id!r} was measured at "
                        f"({record.boundary}, {record.cache_state}), not the "
                        f"declared ({boundary}, {cache_state})"
                    )
                if record.candidate not in candidates:
                    raise ValueError(
                        f"bracketing record {record_id!r} measured candidate "
                        f"{record.candidate!r}, which is not one of the "
                        f"declared {candidates}"
                    )
                cited.add(record.candidate)
                case_id = record.params.get("case_id")
                if case_id is not None and case_id not in case_ids:
                    case_ids.append(case_id)
            missing = set(candidates) - cited
            if missing:
                raise ValueError(
                    f"declared candidates {sorted(missing)} have no records "
                    f"in the {cell_name} bracketing cell — a crossover "
                    "claim needs BOTH arms measured in BOTH cells"
                )
        entry = CrossoverLedgerEntry(
            site=site,
            boundary=boundary,
            candidates=candidates,
            axis=axis,
            crossover_location=crossover_location,
            bracketing_low_record_ids=tuple(bracketing_low_record_ids),
            bracketing_high_record_ids=tuple(bracketing_high_record_ids),
            bracketing_case_ids=tuple(case_ids),
            device=self.device_name,
            source_revision=self.source_revision,
            cache_state=cache_state,
            notes=notes,
        )
        self.ledger.append(entry)
        return entry


def resolve_source_revision() -> str:
    """Best-effort git description of the tree that produced a measurement."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if head.returncode != 0:
        return "unknown"
    suffix = "-dirty" if dirty.stdout.strip() else ""
    return head.stdout.strip() + suffix


def content_fingerprint() -> str:
    """Digest of the lab + sgl_lora source files actually on disk.

    Ninth S3 review: file-synced pod trees resolve to no git state, so a
    revision claim was unverifiable there. This hashes the files that
    produce measurements (sorted relative path + content), giving an
    identity that is comparable across machines regardless of git.
    """
    import pathlib

    # Tenth S3 review: the roots must cover every MEASURED dependency
    # (the SGMV/sgemm kernels live under kernels/ops), and the digest is
    # the full sha256 — truncation weakened the identity for no benefit.
    base = pathlib.Path(__file__).resolve().parents[3]
    roots = (
        pathlib.Path(__file__).resolve().parent,
        base / "python/sglang/srt/lora/sgl_lora",
        base / "python/sglang/kernels/ops/gemm",
        base / "python/sglang/kernels/ops/moe",
    )
    digest = hashlib.sha256()
    for root in roots:
        if not root.is_dir():
            return "unknown"
        for file in sorted(root.rglob("*.py")):
            # macOS AppleDouble stubs ("._foo.py") ride along in tar
            # archives and extract as real files on Linux — they are
            # metadata, not source (tenth-review digest-mismatch root
            # cause, alongside stale overlay files).
            if file.name.startswith("._"):
                continue
            digest.update(str(file.relative_to(root.parent)).encode())
            digest.update(file.read_bytes())
    return "files:" + digest.hexdigest()


def new_suite(suite: str, *, source_revision: str | None = None) -> TimingSuite:
    # Eighth S3 review: a caller-supplied revision is a CLAIM; the suite
    # additionally records what the running tree actually resolves to
    # (full-format short SHA with -dirty, or "unknown" on a file-synced
    # tree), so a dirty checkout can never claim a clean revision silently.
    return TimingSuite(
        suite=suite,
        device_name=torch.cuda.get_device_name(),
        source_revision=source_revision or resolve_source_revision(),
        observed_revision=resolve_source_revision(),
        source_digest=content_fingerprint(),
        torch_version=str(torch.__version__),
        host=platform.node(),
    )


def _record_id(candidate: str, boundary: str, cache_state: str, params: dict) -> str:
    digest_source = msgspec.json.encode(
        {
            "candidate": candidate,
            "boundary": boundary,
            "cache_state": cache_state,
            "params": {key: params[key] for key in sorted(params)},
        }
    )
    return hashlib.sha256(digest_source).hexdigest()[:16]


def measure(
    fn: Callable[[], Any],
    *,
    suite: TimingSuite,
    candidate: str,
    boundary: str,
    params: dict[str, Any],
    graph_replay: bool = True,
    memory_footprint_bytes: int | None = None,
    warmup_iters: int = 50,
    replay_iters: int = 1000,
) -> TimingRecord:
    """Time one zero-argument thunk and append a record to ``suite``.

    ``fn`` takes no arguments and writes into buffers the caller owns, which is
    how every kernel in this stack is shaped.  Because marker computes a memory
    footprint by RE-INVOKING the function and measuring its return value, that
    path is disabled here and the caller declares
    ``memory_footprint_bytes`` explicitly instead — the kernels return ``None``
    and a second invocation would be both wrong and wasteful.
    """
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown boundary {boundary!r}; expected one of {BOUNDARIES}")
    cache_state = CACHE_L2_HOT_GRAPH if graph_replay else CACHE_L2_FLUSHED_EAGER

    result: BenchResult = do_bench(
        fn,
        use_cuda_graph=graph_replay,
        warmup_iters=warmup_iters,
        replay_iters=replay_iters,
        metrics=(0.5, "avg", 0.25, 0.75),
        # Zero-arg closures give marker nothing to rotate, so these flags are
        # inert either way; None documents that no rotation happens.
        graph_clone_args=None,
        graph_clone_kwargs=None,
        disable_log_bandwidth=True,
        memory_output=None,
        memory_args=None,
    )
    median_s, mean_s, p25_s, p75_s = result.times[:4]
    bandwidth = None
    if memory_footprint_bytes is not None and median_s > 0:
        bandwidth = memory_footprint_bytes / (1024**3) / median_s

    return suite.add(
        TimingRecord(
            record_id=_record_id(candidate, boundary, cache_state, params),
            candidate=candidate,
            boundary=boundary,
            cache_state=cache_state,
            params=params,
            median_s=median_s,
            mean_s=mean_s,
            p25_s=p25_s,
            p75_s=p75_s,
            replicate_s=tuple(result.times),
            memory_footprint_bytes=memory_footprint_bytes,
            bandwidth_gib_s=bandwidth,
            graph_replay=graph_replay,
            device_name=suite.device_name,
            source_revision=suite.source_revision,
        )
    )


def pair_with_base(record: TimingRecord, base: TimingRecord) -> TimingRecord:
    """Attach a matched base-only denominator measured in the same session.

    Plan section 14 requires base-only controls measured at the SAME boundary,
    in the same run — a denominator carried over from an earlier session has a
    different clock state and a different build.  Returns a new record; the
    caller replaces the entry it holds.
    """
    if record.boundary != base.boundary:
        raise ValueError(
            "cannot pair measurements taken at different boundaries: "
            f"{record.boundary!r} vs {base.boundary!r}"
        )
    if record.cache_state != base.cache_state:
        raise ValueError(
            "cannot pair measurements taken in different cache states: "
            f"{record.cache_state!r} vs {base.cache_state!r}"
        )
    return msgspec.structs.replace(
        record,
        base_record_id=base.record_id,
        ratio_to_base=record.median_s / base.median_s if base.median_s > 0 else None,
    )


def write_suite(suite: TimingSuite, path: str) -> str:
    """Serialize a suite and return the SHA256 of the bytes written."""
    payload = msgspec.json.format(msgspec.json.encode(suite), indent=2)
    with open(path, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()
