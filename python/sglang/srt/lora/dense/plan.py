"""Dense LoRA kernel, overlap, routing and tile selection.

``configs/<arch>.plans.json`` uses the first compatible matching row; order
narrow rules first. Missing tables or matches use the defaults below.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from functools import cache

import pydantic

from sglang.srt.lora.utils import Phase

logger = logging.getLogger(__name__)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


class AFamily(str, Enum):
    GROUPED = "grouped"  # aligned token route, one tile per (block, N tile)
    PER_ROW = "per_row"  # raw route, one program per route row: a GEMV
    ALL_SLOTS = "all_slots"  # one dense matmul over every slot's A


class BFamily(str, Enum):
    GROUPED = "grouped"
    PER_ROW = "per_row"


class DenseLoraKind(str, Enum):
    """Layer-kind filter for compatible plan rows."""

    LINEAR = "linear"
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"
    SINK_GATE_UP = "sink_gate_up"  # Inkling sink gate/up
    SINK_DOWN = "sink_down"  # Inkling per-expert sink down: the windowed shrink


class Overlap(str, Enum):
    NONE = "none"  # base GEMM, then A, then B into the base output
    A = "a"  # A on the side stream while the base GEMM runs; B after
    AB_DELTA = "ab_delta"  # A and B on the side stream into a delta; add after


def _a_tiles() -> dict[str, int]:
    return {
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }


def _b_tiles() -> dict[str, int]:
    # Defaults for decode; prefill tables override these tiles.
    return {
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
    }


@dataclass(frozen=True, slots=True)
class DensePlan:
    a_family: AFamily = AFamily.GROUPED
    b_family: BFamily = BFamily.GROUPED
    overlap: Overlap = Overlap.NONE
    block_size: int = 16
    a_tiles: dict[str, int | str] = field(
        default_factory=_a_tiles
    )  # SPLIT_MODE is a str
    b_tiles: dict[str, int] = field(default_factory=_b_tiles)

    def __post_init__(self) -> None:
        if self.a_family is AFamily.ALL_SLOTS and self.b_family is not BFamily.PER_ROW:
            raise ValueError(
                "all_slots A writes one bridge plane per slot; only per_row B reads planes"
            )
        if (
            int(self.a_tiles.get("SPLIT_K", 1)) > 1
            and self.a_family is not AFamily.GROUPED
        ):
            raise ValueError("SPLIT_K is a grouped-A tile key")
        if self.a_tiles.get("SPLIT_MODE", "serial") not in ("serial", "planes"):
            raise ValueError("SPLIT_MODE is 'serial' or 'planes'")

    @property
    def needs_aligned_route(self) -> bool:
        return self.a_family is AFamily.GROUPED or self.b_family is BFamily.GROUPED


def _decode_default() -> DensePlan:
    return DensePlan(
        block_size=16,
        a_tiles={**_a_tiles(), "SPLIT_K": 4},
    )


def _prefill_default() -> DensePlan:
    return DensePlan(
        block_size=64,
        b_tiles={**_b_tiles(), "BLOCK_SIZE_N": 64, "num_warps": 8},
    )


class _PlanSpecModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    a_family: AFamily = AFamily.GROUPED
    b_family: BFamily = BFamily.GROUPED
    overlap: Overlap = Overlap.NONE
    block_size: int = 16
    a_tiles: dict[str, int | str] = pydantic.Field(default_factory=_a_tiles)
    b_tiles: dict[str, int] = pydantic.Field(default_factory=_b_tiles)


class _PlanRowModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    phase: Phase | None = None
    kinds: list[DenseLoraKind] | None = None
    max_tokens: int | None = None
    max_rank: int | None = None
    # Geometry bands: shrink input width and total output width across slices.
    min_in_features: int | None = None
    max_in_features: int | None = None
    min_out_features: int | None = None
    max_out_features: int | None = None
    spec: _PlanSpecModel = pydantic.Field(default_factory=_PlanSpecModel)

    @pydantic.model_validator(mode="after")
    def _kinds_can_run_the_plan(self) -> _PlanRowModel:
        # Explicit sink_down rows must obey the same restriction as _resolve.
        if (
            self.kinds
            and DenseLoraKind.SINK_DOWN in self.kinds
            and self.spec.a_family is AFamily.ALL_SLOTS
        ):
            raise ValueError(
                f"row {self.name!r}: an all_slots shrink cannot serve sink_down "
                "(the windowed shrink is not one dense matmul)"
            )
        return self


class _PlansFileModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    rows: list[_PlanRowModel]


def _read_table(name: str) -> dict | None:
    from sglang.srt.environ import envs

    override_dir = envs.SGLANG_LORA_DENSE_CONFIG_DIR.get()
    for directory in filter(None, (override_dir, _CONFIG_DIR)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            if directory == override_dir:
                logger.info(
                    "dense LoRA table %r loaded from override dir %s", name, directory
                )
            with open(path) as handle:
                return json.load(handle)
    return None


@cache
def _load_plans(architecture: str) -> _PlansFileModel | None:
    raw = _read_table(f"{architecture}.plans.json")
    return None if raw is None else _PlansFileModel.model_validate(raw)


class DensePlanTable:
    """Resolves the plan for one (kind, phase, num_tokens) at forward time."""

    def __init__(self, architecture: str, max_rank: int) -> None:
        table = _load_plans(architecture)
        self._rows = table.rows if table is not None else []
        self._max_rank = max_rank
        self._bounds = sorted(
            {row.max_tokens for row in self._rows if row.max_tokens is not None}
        )
        self._cache: dict[tuple[DenseLoraKind, Phase, int, int, int], DensePlan] = {}

    def plan_for(
        self,
        kind: DenseLoraKind,
        phase: Phase,
        num_tokens: int,
        in_features: int = 0,
        out_features: int = 0,
    ) -> DensePlan:
        # Token counts bucket to the row bounds, so the cache stays small.
        key = (kind, phase, self._bucket(num_tokens), in_features, out_features)
        plan = self._cache.get(key)
        if plan is None:
            plan = self._resolve(kind, phase, num_tokens, in_features, out_features)
            self._cache[key] = plan
        return plan

    def _bucket(self, num_tokens: int) -> int:
        for bound in self._bounds:
            if num_tokens <= bound:
                return bound
        return 1 << 30

    def _resolve(
        self,
        kind: DenseLoraKind,
        phase: Phase,
        num_tokens: int,
        in_features: int,
        out_features: int,
    ) -> DensePlan:
        for row in self._rows:
            if row.phase is not None and row.phase is not phase:
                continue
            if row.kinds is not None and kind not in row.kinds:
                continue
            # The windowed shrink (rank block b reads input columns [b*K, (b+1)*K))
            # is not one dense matmul: an all_slots row cannot serve it.
            if (
                kind is DenseLoraKind.SINK_DOWN
                and row.spec.a_family is AFamily.ALL_SLOTS
            ):
                continue
            if row.max_tokens is not None and num_tokens > row.max_tokens:
                continue
            if row.max_rank is not None and self._max_rank > row.max_rank:
                continue
            if row.min_in_features is not None and in_features < row.min_in_features:
                continue
            if row.max_in_features is not None and in_features > row.max_in_features:
                continue
            if row.min_out_features is not None and out_features < row.min_out_features:
                continue
            if row.max_out_features is not None and out_features > row.max_out_features:
                continue
            return DensePlan(**row.spec.model_dump())
        return _decode_default() if phase is Phase.DECODE else _prefill_default()
