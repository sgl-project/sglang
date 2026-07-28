"""Shared bench hardening helpers (third gate-4 review).

Three review findings live here so every bench applies them identically:

* F3 (fail-open rejection): a swept config may be skipped ONLY for an
  exactly-recognized resource/compiler failure; anything else — above
  all a numeric admission failure — must abort the run. Every skip is
  persisted with its reason so an audit can distinguish "infeasible"
  from "never tried".
* F2 (same-geometry tuning): grouped baselines imported from the main
  B table were tuned on per-expert dense routing; benches with a
  different routing geometry re-tune them locally over a one-step
  NEIGHBORHOOD of the table winner (the table is a strong same-device
  prior; the neighborhood absorbs geometry-induced shifts).
* F6 (auditable provenance): table consumers verify the table's
  KERNEL digest against the running tree (4th review: the digest must
  cover every module that controls a measurement — fixtures, invocation
  paths, execution helpers, this file, and the timing machinery).
"""

from __future__ import annotations

import json
import os

import torch
import triton

from benchmark.kernels.lora_moe.timing import (
    atomic_write_bytes,
    kernel_fingerprint,
)

# Exact signatures ONLY (third review: generic words like "compilation"
# could hide a real kernel bug). Extend deliberately, never broadly.
_SKIP_SIGNATURES = (
    "outofresources",  # triton.runtime.errors.OutOfResources class name
    "out of resource",  # its message text
    "passmanager::run failed",  # Triton MLIR pipeline crash (GB300 sweep)
)


SKIP_STAGE_COMPILER = "compiler"
SKIP_STAGE_RESOURCES = "resources"


def skip_reason(error: Exception) -> str | None:
    """Reason string when a sweep config is skippable, else None.

    AssertionError (the admission gates raise it) is NEVER skippable —
    a config that computes wrong numbers is a bug, not an infeasible
    tile.
    """
    if isinstance(error, AssertionError):
        return None
    text = f"{type(error).__name__} {error}".lower()
    for signature in _SKIP_SIGNATURES:
        if signature in text:
            stage = (
                SKIP_STAGE_COMPILER
                if "passmanager" in signature
                else SKIP_STAGE_RESOURCES
            )
            return f"{stage}: {type(error).__name__}: {signature}"
    return None


def write_skip_sidecar(
    output_path: str,
    skips: list[dict],
    *,
    content_addressed: bool = False,
) -> tuple[str, str]:
    """Atomically persist every skipped config and return path plus SHA256."""
    import hashlib
    from pathlib import Path

    payload = json.dumps(
        {"skipped_configs": skips, "count": len(skips)},
        indent=1,
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    if content_addressed:
        anchor = Path(output_path)
        suffix = anchor.suffix or ".json"
        stem = anchor.stem if anchor.suffix else anchor.name
        sidecar = anchor.with_name(f"{stem}.sweep-skips.sha256-{digest}{suffix}")
        if sidecar.is_symlink():
            raise ValueError(f"refusing content-addressed skip symlink {sidecar}")
        if sidecar.exists():
            if sidecar.read_bytes() != payload:
                raise RuntimeError(
                    f"content-addressed skip ledger {sidecar} does not match "
                    "its sha256 filename"
                )
        else:
            atomic_write_bytes(sidecar, payload)
    else:
        sidecar = Path(output_path + ".skips.json")
        atomic_write_bytes(sidecar, payload)
    print(f"{len(skips)} skipped configs -> {sidecar}")
    return str(sidecar), digest


# 4th review: the feasibility PREDICTOR and the coarse/refine search that
# lived here are DELETED. Replaying them against the exhaustive GB300
# shared-down artifact showed the predictor rejects 9,312 configs that
# actually compiled (including the winner in 32/192 cells) and the
# combined method left cells up to 2.17x off the measured optimum. A
# search that cannot reproduce a known frontier is not a search.
#
# Sweeps are EXHAUSTIVE. The real sweep cost was per-config admission
# calling ``.float()`` on [P, H] tensors (~1.6GB of temporaries per
# config at T=8192), fixed by the memory-BOUNDED EXACT gate below: it
# walks the full output in row chunks, accumulating the same quantities
# require_delta_close derives (signal max-abs/L2, error max-abs/L2),
# then applies the identical criteria — full domain, no huge
# temporaries. (An intermediate row-SUBSAMPLE variant was rejected in
# the 5th review: a strided sample aliased the routed structure and was
# correctness evidence in name only.)
GATE_CHUNK_ROWS = 4096
# Hard ceiling on the accumulate-form rel-L2 allowance (13th review): the
# derived bf16 base-rounding floor is legitimate but must never scale the
# gate past the point where real corruption (dropped delta ~ 1.0 rel-L2)
# could pass. 0.125 leaves an 8x rejection margin.
ACCUMULATE_REL_L2_CEILING = 0.125


def require_delta_close_chunked(
    observed,
    reference,
    *,
    gate_dtype,
    label: str = "",
    observed_base=None,
):
    """Exact full-domain signal gate with bounded peak memory.

    Equivalent verdict to :func:`require_delta_close`, O(GATE_CHUNK_ROWS *
    cols) temporaries instead of O(rows * cols).

    ``observed_base`` is the MATCHED BASE of an accumulate-form arm whose
    output is ``base + delta``. It is SUBTRACTED from the observation per
    chunk so every gate is derived from the LoRA DELTA domain (6th
    review): adding the base to the reference instead would scale the
    thresholds by the base magnitude, and a large base could then hide a
    dropped or corrupted delta entirely. This is the bounded-memory
    counterpart of ``require_signal_close``, including its base
    noise-floor validity check.

    Non-finite values are rejected explicitly: ``max(finite, nan)`` keeps
    the finite value and ``nan > gate`` is False, so a NaN chunk would
    otherwise sail through the final comparison.
    """
    import torch

    from benchmark.kernels.lora_moe.signal_gates import (
        MIN_SIGNAL_TO_NOISE,
        DegenerateSignalError,
        bf16_noise_floor,
        resolve_signal_gates,
    )

    if observed.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: observed {tuple(observed.shape)} vs "
            f"reference {tuple(reference.shape)}"
        )
    if observed_base is not None and observed_base.shape != observed.shape:
        raise ValueError(
            f"observed_base shape {tuple(observed_base.shape)} != observed "
            f"{tuple(observed.shape)}"
        )
    rows = observed.shape[0]
    signal_max = 0.0
    signal_sq = 0.0
    error_max = 0.0
    error_sq = 0.0
    base_max = 0.0
    base_sq = 0.0
    for begin in range(0, max(rows, 1), GATE_CHUNK_ROWS):
        ref = reference[begin : begin + GATE_CHUNK_ROWS].detach().to(torch.float64)
        obs = observed[begin : begin + GATE_CHUNK_ROWS].detach().to(torch.float64)
        if ref.numel() == 0:
            continue
        if observed_base is not None:
            base = (
                observed_base[begin : begin + GATE_CHUNK_ROWS]
                .detach()
                .to(torch.float64)
            )
            if not bool(torch.isfinite(base).all()):
                raise AssertionError(f"non-finite base chunk [{label}]")
            base_max = max(base_max, float(base.abs().max()))
            base_sq += float(torch.linalg.vector_norm(base)) ** 2
            obs = obs - base  # gate in the DELTA domain
        if not bool(torch.isfinite(ref).all()):
            raise AssertionError(f"non-finite reference chunk [{label}]")
        if not bool(torch.isfinite(obs).all()):
            raise AssertionError(f"non-finite observed chunk [{label}]")
        signal_max = max(signal_max, float(ref.abs().max()))
        signal_sq += float(torch.linalg.vector_norm(ref)) ** 2
        error = obs - ref
        error_max = max(error_max, float(error.abs().max()))
        error_sq += float(torch.linalg.vector_norm(error)) ** 2
    if signal_max == 0.0:
        raise DegenerateSignalError(
            f"reference signal is exactly zero [{label}]; use a bitwise check"
        )
    # max_abs_gate = S * signal_fraction and rel_l2_gate depends only on
    # gate_dtype (see resolve_signal_gates), so a one-element probe whose
    # max-abs IS the true global S yields the identical thresholds.
    gates = resolve_signal_gates(
        torch.tensor([signal_max], dtype=torch.float64), gate_dtype=gate_dtype
    )
    if observed_base is not None:
        # Same validity rule require_signal_close applies: a delta that
        # sits under the base's storage noise makes the CASE invalid.
        floor = bf16_noise_floor(torch.tensor([base_max], dtype=torch.float64))
        if signal_max < MIN_SIGNAL_TO_NOISE * floor:
            raise DegenerateSignalError(
                f"signal S={signal_max:.3e} is below {MIN_SIGNAL_TO_NOISE:g} "
                f"BF16 quanta of the base ({floor:.3e}) [{label}]; re-scale"
            )
    signal_l2 = signal_sq**0.5
    observed_rel_l2 = (error_sq**0.5) / signal_l2 if signal_l2 > 0.0 else 0.0
    rel_l2_gate = gates.rel_l2_gate
    if observed_base is not None and signal_l2 > 0.0:
        # Accumulate-form arms WRITE bf16(base + delta); subtracting the
        # base afterwards leaves each element carrying up to one bf16 ulp
        # of the BASE magnitude, so the achievable rel-L2 floor relative
        # to the DELTA is quantum * L2(base)/L2(delta) (13th finding:
        # the fixed 1e-2 gate was UNSATISFIABLE for a perfect kernel at
        # the 12.5% max-amplitude validity boundary — GB300 sm103
        # reduction ordering landed 1.3% over while sm90 passed).
        #
        # 13th review: that floor must be BOUNDED. The max-amplitude
        # validity rule constrains peaks, not norms, so a dense base
        # with a SPARSE delta drives L2(base)/L2(delta) — and with it an
        # unbounded allowance — arbitrarily high (demonstrated: 10,000
        # wrong elements at one bf16 quantum each, rel_l2 = 6.25,
        # accepted). Two closures, both fail-CLOSED:
        #   1. HARD CEILING: the gate never exceeds
        #      ACCUMULATE_REL_L2_CEILING (0.125). A dropped or corrupted
        #      delta errs at rel_l2 ~ 1.0, an 8x margin above it.
        #   2. DEGENERACY: if the derived floor exceeds that ceiling,
        #      the CASE cannot distinguish a correct kernel from a
        #      subtly wrong one in accumulate form — that is a fixture
        #      problem, and it raises instead of gating.
        from benchmark.kernels.lora_moe.signal_gates import (
            BF16_RELATIVE_QUANTUM,
        )

        rounding_floor = BF16_RELATIVE_QUANTUM * (base_sq**0.5) / signal_l2
        derived = 1.5 * rounding_floor
        if derived > ACCUMULATE_REL_L2_CEILING:
            raise DegenerateSignalError(
                f"accumulate-form gate is undecidable [{label}]: the bf16 "
                f"base-rounding floor ({derived:.3e} rel-L2 vs the delta) "
                f"exceeds the hard ceiling {ACCUMULATE_REL_L2_CEILING}; "
                "the delta's L2 is drowned by base storage noise — "
                "re-scale the fixture or gate this arm in memset form"
            )
        rel_l2_gate = min(ACCUMULATE_REL_L2_CEILING, max(rel_l2_gate, derived))
    finite = (
        error_max == error_max
        and observed_rel_l2 == observed_rel_l2
        and error_max != float("inf")
        and observed_rel_l2 != float("inf")
    )
    passed = (
        finite and error_max <= gates.max_abs_gate and observed_rel_l2 <= rel_l2_gate
    )
    if not passed:
        raise AssertionError(
            f"signal gate failed [{label}]: max|err|={error_max:.3e} "
            f"(gate {gates.max_abs_gate:.3e}), rel_l2={observed_rel_l2:.3e} "
            f"(gate {rel_l2_gate:.1e}), S={signal_max:.3e} "
            f"[chunked exact, {rows} rows, finite={finite}]"
        )
    return {
        "signal_max_abs": signal_max,
        "signal_l2": signal_l2,
        "observed_max_abs": error_max,
        "observed_rel_l2": observed_rel_l2,
        "rows_checked": rows,
    }


def padded_block_k_cap(rank: int) -> int:
    """Largest useful power-of-two K tile, capped by the declared grid."""
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    next_power_of_two = 1 << (rank - 1).bit_length()
    return min(128, max(32, next_power_of_two))


# ONE chunk list (10th review): the per-expert bench took its csgmv GRID
# from shared-down's copy but keyed its metadata dict off its own, so any
# divergence was a KeyError at the first csgmv config.
# 13th review: chunk 128 added — a plausible prefill winner the (16, 64)
# grid could not observe, leaving "csgmv wins only at EP8" unclosed.
CSGMV_PADDED_CHUNKS = (16, 64, 128)


def exhaustive_grouped_lora_b_grid(*, rank: int, stock: bool):
    """The shared grouped-B tuning grid used by main-B and shared-down.

    BLOCK_SIZE_K is always a power of two >= 16 (10th review). Triton's
    ``tl.dot`` requires every operand dimension >= 16 and ``tl.arange``
    requires a power-of-two length, so a ``BLOCK_SIZE_K = rank`` insertion
    could ONLY ever hurt: for every power-of-two rank >= 16 it is already
    in the base set (a no-op), and for any other rank (8, 24, 48, 96, ...)
    it emits a value Triton refuses to compile. That CompilationError is
    not a recognized skip signature, so the fail-closed sweep would abort
    the whole run on its first config. Short and non-power-of-two ranks
    are still measured correctly — every B kernel masks K against the true
    rank (``k_mask = k_offsets < RANK``), so BLOCK_SIZE_K=16 covers rank 8
    exactly, merely computing padded lanes.
    """
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    block_ks = {k for k in (16, 32, 64, 128) if k <= padded_block_k_cap(rank)}
    extra = {"BLOCK_SIZE_M": 16} if stock else {}
    for bn in (32, 64, 128, 256, 512, 1024):
        for bk in sorted(block_ks):
            for group_m in (1, 4, 8, 16):
                for warps in ((4,) if bn < 128 else (4, 8)):
                    for stages in ((2, 3) if bn < 256 else (2, 3, 4)):
                        yield {
                            "BLOCK_SIZE_N": bn,
                            "BLOCK_SIZE_K": bk,
                            "GROUP_SIZE_M": group_m,
                            "num_warps": warps,
                            "num_stages": stages,
                            **extra,
                        }


def exhaustive_sgmv_grid(
    *,
    rank: int,
    n_columns: int,
    rows_axis: str = "BLOCK_S",
    rows_values=(16, 32, 64, 128, 256, 512),
):
    """EXHAUSTIVE tuning grid (4th review: no predictor, no coarse stage).

    Width-filtering (BLOCK_N <= max(the site's slice width, 64)) and
    BLOCK_K <= the next power-of-two padded rank (with a minimum cap of
    32) are DIMENSIONAL constraints, not performance predictions. The
    minimum padded N tile is 64; the K grid includes every legal
    power-of-two tile through ``padded_block_k_cap(rank)``. Everything
    else is measured.
    """
    for rows in rows_values:
        for block_n in (64, 128, 256, 512, 1024):
            if block_n > max(n_columns, 64):
                continue
            for block_k in (16, 32, 64, 128):
                if block_k > padded_block_k_cap(rank):
                    continue
                for num_warps in (4, 8):
                    for num_stages in (2, 3):
                        config = {
                            rows_axis: rows,
                            "BLOCK_N": block_n,
                            "BLOCK_K": block_k,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                        }
                        yield config


# Bumped whenever the table's required _meta contract changes, so an
# older artifact fails loudly instead of silently skipping a check.
TABLE_SCHEMA_VERSION = "b-table/4"

# 7th review: the table-transfer contract, stated ONCE instead of being
# implied by whatever keys each caller happened to pass.
#
# REQUIRED — a promoted config's meaning depends on these, so a consumer
# whose value differs may not use the table at all.
TABLE_REQUIRED_WORKLOAD = (
    "model_preset",  # geometry: hidden/intermediate widths, experts, top_k
    "adapter_cell",  # slot capacity + base rows change the plan's row domain
    "route_generator",  # validity distribution the winner was tuned against
    # Ownership is NOT retunable: a per-expert table's winners are
    # meaningless for shared-outer weights (different group domain and row
    # mapping), so a mismatch forbids transfer outright.
    "weight_ownership",
)
# MAY DIFFER — only when the consumer declares it and LOCALLY RE-TUNES,
# as the per-expert study does for its requested ranks and route domains.
TABLE_LOCALLY_RETUNED_WORKLOAD = (
    "topology",
    "ranks",
    "sweep_regimes",
    # 8th review: material differences between these benches that a
    # consumer may legitimately re-tune for.
    "expert_id_domain",
)


# 8th review: the digest binds the selected configs to the metadata that
# gives them MEANING. Only volatile audit fields stay outside, so a
# reviewer can still annotate an artifact without breaking its identity.
# 12th review: source_digest is bound too — it is IDENTITY (a tree-content
# digest propagated into every consumer record as table_source_digest),
# not an annotation; unbound, a post-emission edit passed provenance and
# the forged value was stamped into published evidence.
TABLE_DIGEST_BOUND_META = (
    "schema_version",
    "workload",
    "device_name",
    "torch_version",
    "triton_version",
    "kernel_digest",
    "source_digest",
    "sweep_checkpoint_digest",
    "sweep_skips_digest",
)


def require_writable_destination(*paths: str | None) -> None:
    """Fail before CUDA init if any output destination cannot be written.

    12th review: the first filesystem write of a run used to be the sweep
    checkpoint — HOURS in. A mistyped --output directory must be a startup
    error, not an end-of-run evidence loss.
    """
    from pathlib import Path

    for path in paths:
        if path is None:
            continue
        destination = Path(path)
        parent = destination.parent
        if not parent.is_dir():
            raise ValueError(
                f"output destination {path!r}: parent directory {parent} "
                "does not exist"
            )
        if not os.access(parent, os.W_OK):
            raise ValueError(
                f"output destination {path!r}: parent directory {parent} "
                "is not writable"
            )
        if destination.is_symlink():
            raise ValueError(
                f"output destination {path!r} is a symlink; publication "
                "refuses symlinks, so this run could never publish"
            )
        if destination.is_dir():
            raise ValueError(
                f"output destination {path!r} is an existing DIRECTORY; "
                "publication would only fail at the final atomic replace "
                "(12th review) — name a file"
            )
    require_distinct_paths(*paths)


def require_distinct_paths(*paths: str | None) -> None:
    """No two run paths may alias (12th review: --output equal to the
    emitted or consumed table would overwrite it mid-run). Inputs join
    this check but NOT the writability check — a read-only table is
    legitimate."""
    from pathlib import Path

    named = [str(Path(p).resolve()) for p in paths if p is not None]
    if len(set(named)) != len(named):
        raise ValueError(f"run paths alias each other: {sorted(named)}")


def table_content_digest(table: dict) -> str:
    """Canonical SHA256 of selected configs + their binding metadata."""
    import hashlib
    import json

    meta = table.get("_meta", {})
    payload = {
        "configs": {k: v for k, v in table.items() if k != "_meta"},
        "bound_meta": {k: meta.get(k) for k in TABLE_DIGEST_BOUND_META},
    }
    return (
        "sha256:"
        + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    )


def require_sha256_reference(value, *, label: str) -> str:
    """Return a canonical SHA256 reference or fail closed."""
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != len("sha256:") + 64
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{label} must be a canonical sha256:<64 lowercase hex>")
    return value


# ONE regime classifier (squash review: it existed twice — main-B with
# named constants, shared-down with literals — and per-expert imported
# the shared-down copy; a threshold edit in one file silently
# unsynchronized decided-phase regime classification between benches).
DECODE_TINY_T_MAX = 8
DECODE_T_MAX = 256
PREFILL_XL_T_MIN = 4096


def regime_of(num_tokens: int) -> str:
    if num_tokens <= DECODE_TINY_T_MAX:
        return "decode_tiny"
    if num_tokens <= DECODE_T_MAX:
        return "decode"
    if num_tokens < PREFILL_XL_T_MIN:
        return "prefill"
    return "prefill_xl"


# ONE config-key serializer (squash review: main-B and shared-down each
# had one with different token orders, so the same config appeared under
# two spellings across the evidence set; parse_table_config reads both,
# but the canonical spelling is this one).
def config_key(config: dict) -> str:
    parts = [
        f"bn{config['BLOCK_SIZE_N']}",
        f"bk{config['BLOCK_SIZE_K']}",
        f"w{config['num_warps']}",
        f"s{config['num_stages']}",
    ]
    if "SPLIT_K" in config:
        parts.append(f"k{config['SPLIT_K']}")
    if "BLOCK_SIZE_M" in config:
        parts.append(f"m{config['BLOCK_SIZE_M']}")
    if "GROUP_SIZE_M" in config:
        parts.append(f"g{config['GROUP_SIZE_M']}")
    return "-".join(parts)


def skip_entry(reason: str, **context) -> dict:
    """ONE skip-ledger entry shape for every producer (squash review:
    only shared-down persisted the 'stage' field)."""
    return {"stage": reason.split(":")[0], "reason": reason, **context}


def parse_table_config(text: str) -> dict:
    """Strictly parse a current-schema LoRA-B config key."""
    config = {}
    for piece in text.split("-"):
        if piece.startswith("bn"):
            key, value = "BLOCK_SIZE_N", piece[2:]
        elif piece.startswith("bk"):
            key, value = "BLOCK_SIZE_K", piece[2:]
        elif piece.startswith("g"):
            key, value = "GROUP_SIZE_M", piece[1:]
        elif piece.startswith("w"):
            key, value = "num_warps", piece[1:]
        elif piece.startswith("s"):
            key, value = "num_stages", piece[1:]
        elif piece.startswith("m"):
            key, value = "BLOCK_SIZE_M", piece[1:]
        elif piece.startswith("k"):
            key, value = "SPLIT_K", piece[1:]
        else:
            raise ValueError(f"unknown LoRA-B config token {piece!r}")
        if key in config:
            raise ValueError(f"duplicate LoRA-B config field {key}")
        if not value:
            raise ValueError(f"empty LoRA-B config value for {key}")
        config[key] = int(value)
    return config


def require_table_provenance(
    table: dict,
    device,
    *,
    workload: dict,
    locally_retuned: tuple[str, ...] = (),
) -> None:
    """Refuse a table from another device, toolchain, or MEASURED tree.

    The identity that must match is the measured code (kernels, fixtures,
    timing harness) — ``kernel_digest``, which is mandatory under the
    current table schema. ``source_digest`` is still recorded for audit
    but is not the comparison, because a sibling bench's
    sweep-enumeration code cannot change what a promoted config does.

    ``workload`` must carry every key in ``TABLE_REQUIRED_WORKLOAD``;
    those are validated whether or not the caller thought to pass them.
    Keys in ``TABLE_LOCALLY_RETUNED_WORKLOAD`` may differ from the table
    ONLY when named in ``locally_retuned`` by a consumer that actually
    re-tunes those axes.
    """
    meta = table.get("_meta", {})
    checks = [
        ("device_name", torch.cuda.get_device_name(device)),
        ("torch_version", str(torch.__version__)),
        ("triton_version", triton.__version__),
    ]
    if meta.get("schema_version") != TABLE_SCHEMA_VERSION:
        raise ValueError(
            f"table schema_version is {meta.get('schema_version')!r}; this "
            f"run requires {TABLE_SCHEMA_VERSION!r} — regenerate the table"
        )
    # 7th review: under this schema kernel_digest is emitted by
    # construction, so absence is corruption — never a legacy case to
    # tolerate by falling back to the weaker whole-tree digest.
    if not meta.get("kernel_digest") or meta["kernel_digest"] == "unknown":
        raise ValueError(
            f"table has no usable kernel_digest under {TABLE_SCHEMA_VERSION}; "
            "regenerate it"
        )
    if not meta.get("source_digest") or meta["source_digest"] == "unknown":
        raise ValueError(
            f"table has no usable source_digest under {TABLE_SCHEMA_VERSION}; "
            "regenerate it"
        )
    for key in ("sweep_checkpoint_digest", "sweep_skips_digest"):
        try:
            require_sha256_reference(meta.get(key), label=f"table {key}")
        except ValueError as error:
            raise ValueError(
                f"table has no usable {key} under {TABLE_SCHEMA_VERSION}; "
                "regenerate it"
            ) from error
    checks.append(("kernel_digest", kernel_fingerprint()))
    # 6th review: the content digest is mandatory. Optional fields could
    # be deleted to bypass the check entirely.
    recorded = meta.get("table_content_digest")
    if recorded is None:
        raise ValueError("table has no table_content_digest; regenerate it")
    actual = table_content_digest(table)
    if recorded != actual:
        raise ValueError(
            f"table contents were modified after emission: recorded "
            f"{recorded}, actual {actual}"
        )
    want_workload = meta.get("workload")
    if want_workload is None:
        raise ValueError("table has no workload identity; regenerate it")
    unknown = set(locally_retuned) - set(TABLE_LOCALLY_RETUNED_WORKLOAD)
    if unknown:
        raise ValueError(
            f"{sorted(unknown)} are REQUIRED transfer invariants and cannot "
            "be declared locally-retuned"
        )
    # Every required invariant is checked, whether or not the caller
    # thought to pass it (the subset-only check was the 7th review's F2).
    for key in TABLE_REQUIRED_WORKLOAD:
        if key not in workload:
            raise ValueError(
                f"caller must declare workload[{key!r}] — it is a required "
                "table-transfer invariant"
            )
        if key not in want_workload:
            raise ValueError(
                f"table workload lacks required invariant {key!r}; regenerate it"
            )
        recorded_value = want_workload.get(key)
        # 10th review: a table that literally records null is not a match
        # for a null caller value — it is an unusable table.
        if recorded_value is None:
            raise ValueError(f"table workload[{key!r}] is null; regenerate the table")
        if workload[key] is None:
            raise ValueError(f"caller workload[{key!r}] must not be null")
        if recorded_value != workload[key]:
            raise ValueError(
                f"table workload[{key!r}] is {recorded_value!r}; "
                f"this run requires {workload[key]!r}"
            )
    # 8th review: fail CLOSED. Every retunable field must be SUPPLIED,
    # then either match the table or be explicitly declared — omitting it
    # used to skip the check entirely.
    for key in TABLE_LOCALLY_RETUNED_WORKLOAD:
        if key not in workload:
            raise ValueError(
                f"caller must declare workload[{key!r}] — supply it and, if "
                "this bench re-tunes for its own value, name it in "
                "locally_retuned"
            )
        if key not in want_workload:
            raise ValueError(
                f"table workload lacks {key!r}; a declared exemption cannot "
                "excuse a field the source table never recorded"
            )
        recorded_value = want_workload.get(key)
        # Validate both sides before applying the local-retuning exemption:
        # a null is missing identity, not a legitimate alternate value.
        if recorded_value is None:
            raise ValueError(f"table workload[{key!r}] is null; regenerate the table")
        if workload[key] is None:
            raise ValueError(f"caller workload[{key!r}] must not be null")
        if key in locally_retuned:
            continue
        if recorded_value != workload[key]:
            raise ValueError(
                f"table workload[{key!r}] is {recorded_value!r} vs "
                f"{workload[key]!r}; declare it in locally_retuned if this "
                "bench re-tunes for its own value"
            )
    for field, want in checks:
        if meta.get(field) != want:
            raise ValueError(
                f"B config table _meta[{field!r}] is {meta.get(field)!r}; "
                f"this run requires {want!r}"
            )
