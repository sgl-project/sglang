"""In-tree copies of DeepSeek-V4 tuning changes that are not yet in the aiter pin.

Three open aiter PRs only change *which* existing kernel aiter picks for a shape:

* ROCm/aiter#6004 -- wqkv_a fp8 blockscale GEMM (N=2048, K=7168): cold-weight rows in
  ``model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv`` and the Triton
  ``GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=2048-K=7168.json``.
* ROCm/aiter#6029 -- wq_b / wo_b fp8 blockscale GEMMs (N=8192 K=1536, N=7168 K=2048):
  cold-weight rows in ``model_configs/a8w8_blockscale_bpreshuffle_tuned_gemm_dsv3.csv``
  and the two matching Triton JSONs.
* ROCm/aiter#6030 -- mHC fused post+pre on gfx950: tile_m=16 for the FP32 fn decode
  GEMM (``aiter.ops.mhc.get_mhc_fused_post_pre_config``, Python only, bit-identical).

The CSV rows of #6004 and #6029 are in ``utils/aiter_tuned_configs/`` with every other
in-tree row.

``install_aiter_intree_tune()`` registers the rest at DeepSeek-V4 model construction,
before any GEMM / mHC config lookup:

* Triton JSON: ``gemm_config_utils.load_config_json`` answers the three target paths
  with the in-tree JSON (aiter has no search-path override for Triton configs).
* mHC: ``aiter.ops.mhc.get_mhc_fused_post_pre_config`` is wrapped to force tile_m=16.

Upstream-first: every item applies only while aiter still ships the exact pre-PR
content (JSON = the recorded sha256; mHC = the old policy still picks tile_m=32 at
decode M). When aiter has merged the PR, or re-tuned the shape some other way, the item
is skipped and aiter's version is used, so upgrading the aiter pin switches over
automatically and never fails at startup.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from sglang.srt.utils.aiter_tuned_tables import (
    APPLIED,
    MISSING,
    SUPERSEDED,
    UPSTREAM,
)

logger = logging.getLogger(__name__)

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "aiter_intree_configs"
)

TRITON_CONFIG_NAME = "GEMM-A8W8_BLOCKSCALE_PRESHUFFLED"
TRITON_ARCH = "gfx950"
_INTREE_TRITON_DIR = os.path.join(
    _CONFIG_DIR, "triton", TRITON_ARCH, "gemm_a8w8_blockscale_preshuffled"
)
# file -> (aiter PR, sha256 of the canonical JSON aiter shipped before the PR)
INTREE_TRITON_JSON = {
    # TODO(aiter#6004): remove once ROCm/aiter#6004 is in the aiter pin
    "GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=2048-K=7168.json": (
        6004,
        "df0cfb42d0e7fd18635950819cf49f85e8b1624c62a21466ac9cef34a5f96e83",
    ),
    # TODO(aiter#6029): remove once ROCm/aiter#6029 is in the aiter pin
    "GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=7168-K=2048.json": (
        6029,
        "b4c00199f522d29d9d157fd3d3d7a5342637db84ebd23a328f56ec052d446516",
    ),
    "GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=8192-K=1536.json": (
        6029,
        "5f22d2e9da4d88d1b4fcf7b08e2b5c71dfa0372ed69fe7df0c10f0a89a9cdc70",
    ),
}

# TODO(aiter#6030): remove once ROCm/aiter#6030 is in the aiter pin
MHC_TILE_M16_MAX_M = 1024
# Decode M where the pre-#6030 gfx950 policy picks tile_m=32 at DeepSeek-V4-Pro's
# hidden size; if aiter already answers 16 for all of them, it has the change.
_MHC_PROBE_HIDDEN = 7168
_MHC_PROBE_MS = (80, 144, 192, 320, 384, 640)


@dataclass
class IntreeReport:
    """Per-item outcome: applied / upstream (aiter has the PR) / superseded (aiter
    changed it differently) / missing (aiter no longer has the shape or file)."""

    triton_json: Dict[str, str] = field(default_factory=dict)
    mhc: Optional[str] = None
    notes: List[str] = field(default_factory=list)


_report: Optional[IntreeReport] = None


def _env_on(name: str) -> bool:
    from sglang.srt.environ import envs

    return getattr(envs, name).get()


# ---------------------------------------------------------------- Triton JSON


def _json_digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def select_triton_overrides(
    aiter_cfg_dir: str, report: IntreeReport
) -> Dict[str, dict]:
    """Map aiter JSON path -> in-tree JSON, for files aiter still ships pre-PR."""
    overrides = {}
    for name, (pr, base_digest) in INTREE_TRITON_JSON.items():
        with open(os.path.join(_INTREE_TRITON_DIR, name)) as f:
            ours = json.load(f)
        aiter_path = os.path.normpath(os.path.join(aiter_cfg_dir, name))
        try:
            with open(aiter_path) as f:
                theirs = json.load(f)
        except FileNotFoundError:
            report.triton_json[name] = MISSING
            continue
        if theirs == ours:
            report.triton_json[name] = UPSTREAM
        elif _json_digest(theirs) == base_digest:
            overrides[aiter_path] = ours
            report.triton_json[name] = APPLIED
        else:
            report.triton_json[name] = SUPERSEDED
    return overrides


def install_triton_json(report: IntreeReport):
    import aiter.ops.triton.utils.gemm_config_utils as gcu
    from aiter.ops.triton.utils.config_utils import resolve_config_dir

    cfg_dir = resolve_config_dir(
        "gemm", TRITON_CONFIG_NAME, backend="triton", arch=TRITON_ARCH
    )
    overrides = select_triton_overrides(cfg_dir, report)
    if not overrides:
        return
    orig = gcu.load_config_json

    def load_config_json(fpath, required=True):
        hit = overrides.get(os.path.normpath(fpath))
        return hit if hit is not None else orig(fpath, required)

    load_config_json.__wrapped__ = orig
    gcu.load_config_json = load_config_json
    gcu._get_gemm_config_cached.cache_clear()


# ---------------------------------------------------------------- mHC tile_m


def make_mhc_tile_m16_config(orig: Callable) -> Callable:
    """aiter#6030: FP32 fn (not packed BF16) decode M <= 1024 uses tile_m=16."""
    # an aiter older than the preshuffled weight / residual layouts takes (m, hidden_size)
    preshuffle_args = "w_preshuffle_bf16" in inspect.signature(orig).parameters

    @functools.lru_cache(maxsize=1024)
    def get_mhc_fused_post_pre_config(
        m: int,
        hidden_size: int,
        w_preshuffle_bf16: bool = False,
        res_preshuffle: bool = False,
    ):
        if preshuffle_args:
            config = orig(
                m,
                hidden_size,
                w_preshuffle_bf16=w_preshuffle_bf16,
                res_preshuffle=res_preshuffle,
            )
        elif w_preshuffle_bf16 or res_preshuffle:
            raise TypeError("this aiter's mHC config has no preshuffled layouts")
        else:
            config = orig(m, hidden_size)
        split_k, tile_m, tile_n, tile_k = config
        if not w_preshuffle_bf16 and m <= MHC_TILE_M16_MAX_M:
            tile_m = 16
        return split_k, tile_m, tile_n, tile_k

    get_mhc_fused_post_pre_config.__wrapped__ = orig
    return get_mhc_fused_post_pre_config


def mhc_needs_tile_m16(orig: Callable, arch: str) -> bool:
    if arch != "gfx950":
        return False
    return any(orig(m, _MHC_PROBE_HIDDEN)[1] != 16 for m in _MHC_PROBE_MS)


def install_mhc_tile_m16(report: IntreeReport):
    import aiter.ops.mhc as aiter_mhc

    orig = aiter_mhc.get_mhc_fused_post_pre_config
    try:
        arch = aiter_mhc.get_gfx_runtime()
    except Exception:  # noqa: BLE001
        arch = "unknown"
    if arch != "gfx950":
        report.mhc = f"not gfx950 ({arch})"
        return
    if not mhc_needs_tile_m16(orig, arch):
        report.mhc = UPSTREAM
        return
    aiter_mhc.get_mhc_fused_post_pre_config = make_mhc_tile_m16_config(orig)
    report.mhc = APPLIED


# ---------------------------------------------------------------- entry


def install_aiter_intree_tune() -> Optional[IntreeReport]:
    """Idempotent; call before the first aiter GEMM / mHC config lookup."""
    global _report
    if _report is not None:
        return _report
    try:
        import aiter  # noqa: F401
    except ImportError:
        return None
    report = IntreeReport()
    steps = []
    if _env_on("SGLANG_AITER_INTREE_GEMM_TUNE"):
        steps.append(install_triton_json)
    if _env_on("SGLANG_AITER_INTREE_MHC_TILE_M16"):
        steps.append(install_mhc_tile_m16)
    for step in steps:
        try:
            step(report)
        except Exception as err:  # noqa: BLE001 - keep aiter's stock behavior
            report.notes.append(f"{step.__name__}: {err!r}")
            logger.warning("aiter in-tree tune: %s skipped: %r", step.__name__, err)
    _report = report
    logger.info(
        "aiter in-tree tune (aiter#6004/#6029/#6030): triton_json=%s mhc_tile_m16=%s",
        report.triton_json,
        report.mhc,
    )
    return report
