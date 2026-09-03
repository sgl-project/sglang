"""Runtime ingestion — the ~10-line hook that makes Attune a refinement.

Resolution order (mirrors vLLM VLLM_TUNED_CONFIG_FOLDER + SGLang flashinfer_autotune):
  1. SGLANG_ATTUNE_CONFIG_FOLDER env override (user drop-in)
  2. local first-boot fingerprint cache (per-box, extended-hw-keyed)
  3. packaged committed corpus (coarse filename key)
  4. MISS -> return None -> loud warning -> engine falls back to the existing heuristic

The engine-side pass (see attune_select) is inserted AFTER _attention_backend_default and
BEFORE _handle_attention_backend_compatibility. It only proposes a pick among gate-survivors.
Two guardrails are load-bearing:

  * DOUBLE-DUTY GATING: Attune picks the fastest measured backend, but the downstream
    _handle_attention_backend_compatibility remains the final authority — it may still
    override Attune for an edge-case request that violates a hardware limit.
  * FAIL-SAFE: if Attune has no config, or its proposed pick is not among the current
    eligible candidates, attune_select returns None and the engine keeps the heuristic
    default. It never raises. A bad/stale/absent config only ever costs performance,
    never correctness.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

from .device import DeviceInfo
from .pagesize import default_page_size
from .shapes import (
    AttnProfile,
    DecodeShape,
    PrefillShape,
    parse_decode_key,
    parse_prefill_key,
)
from .writer import SCHEMA_VERSION, config_filename, fingerprint

logger = logging.getLogger("sglang.attune")

ENV_FOLDER = "SGLANG_ATTUNE_CONFIG_FOLDER"


@functools.lru_cache(maxsize=64)
def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if cfg.get("schema_version") != SCHEMA_VERSION:
        logger.warning(
            "[attune] ignoring config with schema %s (expected %s): %s",
            cfg.get("schema_version"),
            SCHEMA_VERSION,
            path,
        )
        return None
    return cfg


def get_attune_config(
    dev: DeviceInfo,
    profile: AttnProfile,
    packaged_dir: str,
    local_cache_dir: Optional[str] = None,
) -> Optional[dict]:
    """Return the tuned config for (device, profile), or None on miss (loud warning)."""
    fname = config_filename(dev, profile)

    override = os.environ.get(ENV_FOLDER)
    if override:
        cfg = _load_json(os.path.join(override, fname))
        if cfg:
            return cfg

    if local_cache_dir:
        fp = fingerprint(dev, profile)
        cfg = _load_json(
            os.path.join(local_cache_dir, SCHEMA_VERSION, dev.sm_tag, fp + ".json")
        )
        if cfg:
            return cfg

    cfg = _load_json(os.path.join(packaged_dir, fname))
    if cfg:
        return cfg

    logger.warning(
        "[attune] no tuned attention config for %s / %s. Using the built-in "
        "heuristic (performance may be sub-optimal). Run `sglang tune` to generate one.",
        dev.name,
        profile.family(),
    )
    return None


def _length_of(shape) -> int:
    """The second grid axis, phase-appropriate: decode KV length or prefill seq length."""
    return shape.ctx_len if hasattr(shape, "ctx_len") else shape.seq_len


def _nearest(body: Dict[str, dict], want, keyparse) -> Optional[dict]:
    """Nearest-bucket lookup (MoE-style min-distance), phase-appropriate.
    Batch and length axes are weighted comparably in log2 space, matching the
    ~power-of-two bucket layout of the grid."""
    if not body:
        return None
    best, best_d = None, None
    for k, v in body.items():
        sh = keyparse(k)
        d = abs(math.log2(max(1, sh.batch)) - math.log2(max(1, want.batch))) + abs(
            math.log2(max(1, _length_of(sh))) - math.log2(max(1, _length_of(want)))
        )
        if best_d is None or d < best_d:
            best, best_d = v, d
    return best


def pick_backends(
    cfg: dict,
    eligible_prefill: List[str],
    eligible_decode: List[str],
    workload_hint: Optional[dict] = None,
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Collapse the fine grid to one (prefill_backend, decode_backend, page_size) for init.

    Default (no hint): per phase, the backend that wins the most buckets AMONG the
    currently-eligible candidates — all buckets weighted equally. With a workload hint
    (``{"decode": {"batch": B, "ctx_len": L}, "prefill": {"batch": B, "seq_len": S}}``,
    either phase optional), the nearest measured bucket's winner takes precedence; if
    that winner is not currently eligible, fall back to the vote. Returns
    (None, None, None) if no eligible winner exists — the fail-safe: engine keeps its
    default.
    """

    def vote(body: Dict[str, dict], eligible: List[str]) -> Optional[str]:
        tally: Dict[str, int] = {}
        for v in body.values():
            b = v.get("backend")
            if b in eligible:  # double-duty: never pick a non-survivor
                tally[b] = tally.get(b, 0) + 1
        if not tally:
            return None
        return max(tally, key=tally.get)

    def choose(
        phase: str, body: Dict[str, dict], eligible: List[str], keyparse, shape_cls
    ) -> Optional[str]:
        hint = (workload_hint or {}).get(phase)
        if hint:
            cell = _nearest(body, shape_cls(**hint), keyparse)
            if cell and cell.get("backend") in eligible:
                return cell["backend"]  # hinted nearest bucket, still gated
        return vote(body, eligible)

    prefill = choose(
        "prefill",
        cfg.get("prefill", {}),
        eligible_prefill,
        parse_prefill_key,
        PrefillShape,
    )
    decode = choose(
        "decode", cfg.get("decode", {}), eligible_decode, parse_decode_key, DecodeShape
    )
    page = default_page_size(decode) if decode else None
    return prefill, decode, page


def attune_select(
    dev: DeviceInfo,
    profile: AttnProfile,
    packaged_dir: str,
    eligible_prefill: List[str],
    eligible_decode: List[str],
    local_cache_dir: Optional[str] = None,
    workload_hint: Optional[dict] = None,
) -> Optional[dict]:
    """The engine-init hook. Returns a dict of overrides to splice into ServerArgs, or
    None (keep the heuristic). NEVER raises."""
    try:
        cfg = get_attune_config(dev, profile, packaged_dir, local_cache_dir)
        if not cfg:
            return None
        prefill, decode, page = pick_backends(
            cfg, eligible_prefill, eligible_decode, workload_hint
        )
        if not prefill and not decode:
            logger.warning(
                "[attune] tuned config found but its winners are not currently "
                "eligible; keeping the heuristic default."
            )
            return None
        out = {}
        if prefill:
            out["prefill_attention_backend"] = prefill
        if decode:
            out["decode_attention_backend"] = decode
            if page:
                out["page_size"] = page
        logger.info(
            "[attune] tuned selection: prefill=%s decode=%s page_size=%s",
            prefill,
            decode,
            page,
        )
        return out
    except (
        Exception
    ) as e:  # pragma: no cover - defensive; ingestion must never crash boot
        logger.warning("[attune] selection failed (%s); keeping heuristic default.", e)
        return None
