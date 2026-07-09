"""Per-layer tensor-parallel layout planning for DiT models.

Under TP, every linear in a DiT can be laid out one of three ways:

- ``shard``: megatron-style Column->Row pair (or a plain RowParallel), paying
  one all-reduce per pair. The right default for large fused pairs.
- ``replicate``: full weight on every rank. Comm-free when the layer's input
  is already full on every rank, at the cost of duplicating the GEMM. Wins for
  small-token branches (e.g. the ~1K-token text FFN of an MMDiT) where the
  per-block all-reduce costs more than the duplicated compute.
- ``col_gather``: ColumnParallel with ``gather_output=True`` — sharded GEMM
  plus one output all-gather. The right layout for large *standalone*
  projections with full input and no Row partner (``img_in``/``proj_out``).

Which layout wins depends on the model (block count, branch token counts),
the TP degree, the machine, and the workload — and, critically, it can NOT be
predicted from isolated communication/GEMM microbenchmarks: measured
end-to-end, eliminating an all-reduce can be worth far more than its isolated
cost (it also removes a sync point and, under torch.compile, a graph break),
while substituting one collective for another (all-reduce -> all-gather) is
worth nothing even when the traffic math says it should win. The measured
example set (Qwen-Image vs FLUX, tp=2 vs tp=4, 512^2 vs 1536^2) is recorded in
``MEASURED_RULES`` below.

The planner therefore resolves each decision through four levels, first match
wins:

1. an explicit plan file (``--dit-tp-plan /path/plan.json``) — exact per-layer
   control; this is also the output format of ``tools/tune_dit_tp_plan.py``,
   which measures candidate plans end-to-end (ABAB-interleaved pairs) on the
   actual machine;
2. structural rules that measurement refuted globally (attention out-projs
   stay sharded: replicating them substitutes collectives instead of
   eliminating one — measured e2e-null; the ColumnParallel+gather variant is
   strictly worse);
3. ``MEASURED_RULES``: per-(model, branch) defaults validated end-to-end,
   gated on TP degree and (for aggressive rules) the expected workload;
4. conservative fallback: shard everything. An un-measured model never gets a
   speculative layout (the qwen text-FFN rule measurably does *not* transfer
   to FLUX, so there is no blanket MMDiT rule).
"""

from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

DIT_TP_PLAN_ENV = "SGLANG_DIT_TP_PLAN"
DIT_TP_PLAN_WORKLOAD_ENV = "SGLANG_DIT_TP_PLAN_WORKLOAD"

PLAN_AUTO = "auto"
PLAN_FULL = "full"
PLAN_AGGRESSIVE = "aggressive"
_PLAN_MODES = (PLAN_AUTO, PLAN_FULL, PLAN_AGGRESSIVE)


class ShardScheme(str, Enum):
    SHARD = "shard"
    REPLICATE = "replicate"
    COL_GATHER = "col_gather"


# Schemes each layer role currently has a forward path for. A plan file
# requesting anything else is an error, not a silent fallback.
SUPPORTED_SCHEMES: dict[str, frozenset[ShardScheme]] = {
    "ffn": frozenset({ShardScheme.SHARD, ShardScheme.REPLICATE}),
    # Replicating an attention out-proj needs an input all-gather (its
    # contraction dim is head-sharded): that substitutes a collective instead
    # of eliminating one. Measured e2e: replicate = null (+0.8%, mixed-sign
    # ABAB paired diffs), col_gather = strictly worse (two gathers per call).
    "attn_out": frozenset({ShardScheme.SHARD}),
}


@dataclass(frozen=True)
class MeasuredRule:
    """An e2e-validated layout default for one (model family, branch).

    Only rules confirmed with ABAB-interleaved paired e2e runs belong here —
    isolated per-step timings both over- and under-state real server effects.
    ``evidence`` records the measurement so future readers can re-check it.
    """

    scheme: ShardScheme
    max_tp: Optional[int] = None
    max_image_area: Optional[int] = None
    aggressive_only: bool = False
    evidence: str = ""

    def applies(self, tp_size: int, mode: str, workload_area: Optional[int]) -> bool:
        if self.max_tp is not None and tp_size > self.max_tp:
            return False
        if self.aggressive_only and mode != PLAN_AGGRESSIVE:
            return False
        if self.max_image_area is not None:
            if workload_area is None or workload_area > self.max_image_area:
                return False
        return True


# Keyed by (model_family, branch). The absence of an entry means "shard" —
# deliberately so: FLUX.1-dev's ff_context has the same structure as the
# qwen_image text FFN, yet replicating it measured +0.8% e2e (19 dual-stream
# blocks vs 60, so a third of the all-reduces saved for the same duplicated
# GEMM). Per-model measurement, not per-architecture analogy.
MEASURED_RULES: dict[tuple[str, str], tuple[MeasuredRule, ...]] = {
    ("qwen_image", "text"): (
        MeasuredRule(
            scheme=ShardScheme.REPLICATE,
            max_tp=2,
            evidence=(
                "H100 e2e ABAB x3, 1024x1024/50 steps: full-shard 5.58s -> "
                "5.12s (-8%); 512^2 -9%; 1536^2 +0.4% (kept); tp=4 +1.1% "
                "regression, hence max_tp=2"
            ),
        ),
    ),
    ("qwen_image", "image"): (
        MeasuredRule(
            scheme=ShardScheme.REPLICATE,
            max_tp=2,
            max_image_area=640 * 640,
            aggressive_only=True,
            evidence=(
                "H100 e2e ABAB x3, 512x512: 4.11s vs 5.04s full-shard "
                "(-18%); regresses at >=1024^2, so aggressive-only and "
                "gated on the expected workload"
            ),
        ),
    ),
}


def parse_workload(workload: Optional[str]) -> Optional[int]:
    """Parse ``WxH`` or ``WxHxF`` into an image area in pixels."""
    if not workload:
        return None
    parts = workload.lower().replace("*", "x").split("x")
    if len(parts) not in (2, 3):
        raise ValueError(
            f"Invalid --dit-tp-plan-workload {workload!r}: expected WxH or WxHxF"
        )
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as e:
        raise ValueError(
            f"Invalid --dit-tp-plan-workload {workload!r}: {e}"
        ) from None
    return width * height


@dataclass
class TPShardPlan:
    """Resolved plan configuration (mode or explicit rules)."""

    mode: str = PLAN_AUTO
    # Plan-file rules: layer-prefix glob -> scheme, checked in file order.
    rules: dict[str, ShardScheme] = field(default_factory=dict)
    workload_area: Optional[int] = None
    source: str = "default"

    @classmethod
    def load(cls, spec: Optional[str], workload: Optional[str]) -> "TPShardPlan":
        area = parse_workload(workload)
        if not spec or spec in _PLAN_MODES:
            return cls(mode=spec or PLAN_AUTO, workload_area=area, source="mode")
        with open(spec) as f:
            data = json.load(f)
        rules = {
            pattern: ShardScheme(scheme)
            for pattern, scheme in data.get("rules", {}).items()
        }
        mode = data.get("mode", PLAN_AUTO)
        if mode not in _PLAN_MODES:
            raise ValueError(f"Invalid mode {mode!r} in plan file {spec}")
        if workload is None and data.get("workload"):
            area = parse_workload(data["workload"])
        return cls(mode=mode, rules=rules, workload_area=area, source=spec)


class TPShardPlanner:
    def __init__(self, plan: TPShardPlan, tp_size: int):
        self.plan = plan
        self.tp_size = tp_size
        self._logged: set[tuple[str, str, str]] = set()

    def decide(
        self,
        *,
        role: str,
        model_family: str,
        branch: str,
        prefix: str = "",
    ) -> ShardScheme:
        """Resolve the layout for one layer.

        Args:
            role: "ffn" or "attn_out" (see SUPPORTED_SCHEMES).
            model_family: registry key, e.g. "qwen_image".
            branch: which stream feeds the layer ("text" | "image" | "joint").
            prefix: full layer path, matched against plan-file globs.
        """
        supported = SUPPORTED_SCHEMES[role]
        scheme, why = self._resolve(role, model_family, branch, prefix, supported)
        key = (model_family, branch, role)
        if key not in self._logged:
            self._logged.add(key)
            logger.info(
                "TP shard plan [%s.%s %s] (tp=%d, plan=%s): %s (%s)",
                model_family,
                branch,
                role,
                self.tp_size,
                self.plan.source if self.plan.rules else self.plan.mode,
                scheme.value,
                why,
            )
        return scheme

    def decide_ffn(self, *, model_family: str, branch: str, prefix: str = "") -> ShardScheme:
        return self.decide(
            role="ffn", model_family=model_family, branch=branch, prefix=prefix
        )

    def _resolve(
        self,
        role: str,
        model_family: str,
        branch: str,
        prefix: str,
        supported: frozenset[ShardScheme],
    ) -> tuple[ShardScheme, str]:
        # TP off: every scheme degenerates to a plain linear; keep the default.
        if self.tp_size <= 1:
            return ShardScheme.SHARD, "tp disabled"
        # 1. Explicit plan-file rule.
        for pattern, scheme in self.plan.rules.items():
            if fnmatch.fnmatch(prefix, pattern):
                if scheme not in supported:
                    raise ValueError(
                        f"Plan rule {pattern!r} requests {scheme.value!r} for "
                        f"{prefix!r}, but role {role!r} only supports "
                        f"{sorted(s.value for s in supported)}"
                    )
                return scheme, f"plan rule {pattern!r}"
        # 2. Structural rule: roles with a single supported scheme.
        if len(supported) == 1:
            (only,) = supported
            return only, "structural"
        # 3. Full-shard mode.
        if self.plan.mode == PLAN_FULL:
            return ShardScheme.SHARD, "plan=full"
        # 4. Measured registry.
        for rule in MEASURED_RULES.get((model_family, branch), ()):
            if rule.scheme not in supported:
                continue
            if rule.applies(self.tp_size, self.plan.mode, self.plan.workload_area):
                return rule.scheme, f"measured: {rule.evidence}"
        # 5. Conservative default.
        return ShardScheme.SHARD, "default (no measured rule)"


_PLANNER: Optional[TPShardPlanner] = None


def get_tp_shard_planner() -> TPShardPlanner:
    """Process-wide planner, built lazily from server args (env as fallback).

    Model ``__init__`` runs after distributed init in every serving path, so
    the TP world size is available by the time the first decision is made.
    """
    global _PLANNER
    if _PLANNER is None:
        _PLANNER = _build_planner()
    return _PLANNER


def reset_tp_shard_planner() -> None:
    """Testing hook: drop the cached planner so the next call rebuilds it."""
    global _PLANNER
    _PLANNER = None


def _build_planner() -> TPShardPlanner:
    spec = os.environ.get(DIT_TP_PLAN_ENV)
    workload = os.environ.get(DIT_TP_PLAN_WORKLOAD_ENV)
    if spec is None or workload is None:
        try:
            from sglang.multimodal_gen.runtime.server_args import (
                get_global_server_args,
            )

            args = get_global_server_args()
            if spec is None:
                spec = getattr(args, "dit_tp_plan", None)
            if workload is None:
                workload = getattr(args, "dit_tp_plan_workload", None)
        except Exception:  # pragma: no cover - server args unavailable in CI
            pass
    try:
        from sglang.multimodal_gen.runtime.distributed import get_tp_world_size

        tp_size = get_tp_world_size()
    except Exception:  # pragma: no cover - distributed not initialized
        tp_size = 1
    return TPShardPlanner(TPShardPlan.load(spec, workload), tp_size)
