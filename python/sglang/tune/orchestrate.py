"""End-to-end tuning flow: gate -> isolated bench -> pick winners -> write both artifacts.

sglang tune --model-path ... --tp-size ...           (real, inside a live tree)
sglang tune --mock-device NVIDIA_H20 --mock-sm 90     (GPU-free demo of the whole flow)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from .candidates import candidate_backends
from .device import DeviceInfo
from .isolation import CandidateResult, run_candidate_isolated
from .pagesize import default_page_size
from .shapes import AttnProfile, decode_grid, prefill_grid
from .writer import build_config, save_committed, save_local_cache

logger = logging.getLogger("sglang.attune")


def _bench_phase(
    phase: str,
    grid,
    dev: DeviceInfo,
    profile: AttnProfile,
    mock: bool,
    isolate: bool,
    bandwidth_divergent: bool,
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Benchmark every eligible candidate over the phase grid, pick the winner per bucket."""
    bucket_keys = [s.bucket_key() for s in grid]
    eligible, pruned = candidate_backends(dev, profile, phase)
    logger.info("[attune] %s eligible: %s | pruned: %s", phase, eligible, pruned)

    results: List[CandidateResult] = []
    skipped: Dict[str, str] = dict(pruned)  # static-pruned reasons carry through
    for backend in eligible:
        r = run_candidate_isolated(
            backend,
            phase,
            bucket_keys,
            profile,
            bandwidth_divergent,
            mock=mock,
            isolate=isolate,
        )
        if r.failure:
            # Partial grids survive: buckets measured before the failure still compete.
            note = (
                r.failure
                if not r.latencies
                else f"{r.failure} (kept {len(r.latencies)}/{len(bucket_keys)} buckets)"
            )
            skipped[backend] = note  # empirically-discovered failure
            logger.warning(
                "[attune] %s/%s failed empirically: %s", phase, backend, note
            )
        if r.latencies:
            results.append(r)
        for key, why in r.skipped_shapes.items():  # per-shape skips (e.g. OOM trim)
            skipped[f"{backend}@{key}"] = why

    body: Dict[str, dict] = {}
    for key in bucket_keys:
        best_b, best_us = None, None
        for r in results:
            us = r.latencies.get(key)
            if us is None:
                continue
            if best_us is None or us < best_us:
                best_b, best_us = r.backend, us
        if best_b is None:
            continue
        cell = {"backend": best_b, "latency_us": round(best_us, 2)}
        if phase == "decode":
            cell["page_size"] = default_page_size(best_b)
        body[key] = cell
    return body, skipped


def run_tune(
    dev: DeviceInfo,
    profile: AttnProfile,
    packaged_dir: str,
    local_cache_dir: Optional[str] = None,
    mock: bool = True,
    isolate: bool = True,
    provenance: Optional[dict] = None,
    phases: Tuple[str, ...] = ("decode", "prefill"),
) -> dict:
    """Run the full tune and write both artifacts. Returns the config object."""
    # A crude bandwidth-divergent heuristic for the mock model: H20-class parts share an SM
    # predicate with their flagships but have far lower FLOPS/bandwidth ratios.
    bandwidth_divergent = "H20" in dev.name

    decode_body, skip_d = ({}, {})
    prefill_body, skip_p = ({}, {})
    if "decode" in phases:
        decode_body, skip_d = _bench_phase(
            "decode", decode_grid(), dev, profile, mock, isolate, bandwidth_divergent
        )
    if "prefill" in phases:
        prefill_body, skip_p = _bench_phase(
            "prefill", prefill_grid(), dev, profile, mock, isolate, bandwidth_divergent
        )

    skipped = {f"decode/{k}": v for k, v in skip_d.items()}
    skipped.update({f"prefill/{k}": v for k, v in skip_p.items()})

    config = build_config(
        dev, profile, decode_body, prefill_body, skipped, provenance or {}
    )
    committed = save_committed(packaged_dir, dev, profile, config)
    logger.info("[attune] wrote committed config: %s", committed)
    if local_cache_dir:
        cache = save_local_cache(local_cache_dir, config)
        logger.info("[attune] wrote local cache: %s", cache)
    return config


def summarize(config: dict) -> str:
    """Human-readable summary: the crossover points Attune found."""
    lines = []
    for phase in ("decode", "prefill"):
        body = config.get(phase, {})
        winners = {}
        for k, v in sorted(body.items(), key=lambda kv: int(kv[0].split(":")[0])):
            winners.setdefault(v["backend"], []).append(k)
        lines.append(
            f"{phase}: " + ", ".join(f"{b}×{len(ks)}" for b, ks in winners.items())
        )
    if config.get("skipped"):
        lines.append(
            "skipped: "
            + ", ".join(f"{k}={v}" for k, v in list(config["skipped"].items())[:6])
        )
    return "\n".join(lines)
