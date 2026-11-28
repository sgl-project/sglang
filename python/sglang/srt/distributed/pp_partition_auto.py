"""Boot-time PP layer split from each GPU's measured speed (SGLANG_ENABLE_PP_PARTITION_AUTO).

GPUs of one model run at slightly different speeds (power and clock limits), and a
pipeline runs at the pace of its slowest (stage, GPU) pair. Before the weights load,
every rank measures its GPU with a short concurrent burn (or takes a fixed factor
from SGLANG_PP_PARTITION_AUTO_SPEED) and the ranks all-gather the results. Each rank
then runs the same deterministic search on the same inputs: the contiguous split that
minimizes the slowest stage's cost x speed factor, then the sum of squares of the
stage costs. The split replaces the default one through SGLANG_PP_LAYER_PARTITION,
which the model reads when it builds its layer range. Burn noise can change the split
between boots on the same hardware; all ranks of one boot always agree.

Layer costs are uniform unless a cost model (SGLANG_PP_PARTITION_AUTO_COST_MODEL)
gives a cost per layer kind. Kinds are attention types, so layers of one kind that
differ otherwise (e.g. the leading dense layers of an MoE model) share one cost.
Every stage keeps every layer kind of a hybrid model. The cost model can also bound
each stage's KV token capacity and its activation headroom outside the static
memory pools.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import msgspec

logger = logging.getLogger(__name__)

# Speed factors relative to the median stage. Fitted per-GPU factors of one model
# spanned 0.956 to 1.069 over three node pairs; these bounds allow about three times
# that, symmetric in ratio (0.8 = 1 / 1.25), and a GPU beyond them was disturbed.
_FACTOR_BOUNDS = (0.8, 1.25)


class PPPartitionCostModel(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Per-layer-kind forward cost and memory of a model, read from JSON."""

    # Forward cost per layer of each kind, in any unit (kinds not listed cost 1),
    # plus the first stage's (embedding) and the last stage's (LM head) extra cost.
    layer_cost: Dict[str, float] = {}
    first_stage_cost: float = 0.0
    last_stage_cost: float = 0.0
    # KV capacity bound, on when kv_bytes_per_token is set: a stage's KV token
    # capacity is what its static memory holds after weights and per-layer state.
    layer_weight_bytes: Dict[str, float] = {}
    layer_state_bytes: Dict[str, float] = {}
    kv_bytes_per_token: Dict[str, float] = {}
    first_stage_weight_bytes: float = 0.0
    last_stage_weight_bytes: float = 0.0
    # The split keeps this fraction of the default split's KV token capacity, the
    # minimum over stages.
    kv_floor: float = 1.0
    # Activation headroom bound, on when proxy_bytes_per_token is set: entry e is the
    # bytes per token of the activations a stage ending at layer e sends (e = 0 ..
    # num_layers). Outside the static pools, a stage holds proxy_buffers_in received
    # and proxy_buffers_out sent messages of proxy_tokens tokens (0: the chunked
    # prefill size), plus activation_headroom_bytes and activation_headroom_fraction of
    # the smallest stage's memory outside the static pools.
    proxy_bytes_per_token: List[float] = []
    proxy_tokens: int = 0
    proxy_buffers_in: int = 1
    proxy_buffers_out: int = 1
    activation_headroom_bytes: float = 0.0
    activation_headroom_fraction: float = 0.0

    @classmethod
    def from_json(cls, data: bytes, kinds: Sequence[str]) -> PPPartitionCostModel:
        """Decode and validate a JSON cost model for a model with these layer kinds."""
        m = msgspec.json.decode(data, type=cls)
        m._validate(set(kinds), len(kinds))
        return m

    def _validate(self, kinds: set, num_layers: int) -> None:
        tables = {
            "layer_cost": self.layer_cost,
            "layer_weight_bytes": self.layer_weight_bytes,
            "layer_state_bytes": self.layer_state_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
        }
        for name, table in tables.items():
            unknown = sorted(set(table) - kinds)
            if unknown:
                raise ValueError(
                    f"cost model {name} has keys {unknown} that are not layer kinds "
                    f"of this model {sorted(kinds)}"
                )
            _check_values(name, table.values(), minimum=0.0)
        _check_values("layer_cost", self.layer_cost.values(), minimum=math.ulp(0.0))
        # Stage extras are fitted intercepts and may be negative; a split whose stage
        # cost is not positive is infeasible (_Search.feasible).
        _check_values(
            "first/last stage costs",
            (self.first_stage_cost, self.last_stage_cost),
            minimum=-math.inf,
        )
        # Even a one-layer stage cannot hold negative weight bytes.
        least_weight = min(self.layer_weight_bytes.get(k, 0.0) for k in kinds)
        _check_values(
            "first/last stage weight bytes plus the least layer weight",
            (
                self.first_stage_weight_bytes + least_weight,
                self.last_stage_weight_bytes + least_weight,
            ),
            minimum=0.0,
        )
        _check_values(
            "proxy and headroom fields",
            (
                *self.proxy_bytes_per_token,
                self.proxy_tokens,
                self.proxy_buffers_in,
                self.proxy_buffers_out,
                self.activation_headroom_bytes,
            ),
            minimum=0.0,
        )
        if self.proxy_bytes_per_token and len(self.proxy_bytes_per_token) != (
            num_layers + 1
        ):
            raise ValueError(
                f"cost model proxy_bytes_per_token has {len(self.proxy_bytes_per_token)} "
                f"entries; it needs one per end layer 0..{num_layers}"
            )
        if not 0.0 <= self.activation_headroom_fraction < 1.0:
            raise ValueError(
                "cost model activation_headroom_fraction must be in [0, 1)"
            )
        if not 0.0 < self.kv_floor <= 1.0:
            raise ValueError("cost model kv_floor must be in (0, 1]")


def _check_values(name: str, values, minimum: float) -> None:
    for value in values:
        if math.isnan(value) or math.isinf(value) or value < minimum:
            raise ValueError(f"cost model {name} has an invalid value {value!r}")


class StageMemory(msgspec.Struct, frozen=True):
    """Bytes per stage: static_bytes holds the static pools (weights, state, KV),
    free_bytes the slack left outside them, which activations use."""

    static_bytes: Tuple[float, ...]
    free_bytes: Tuple[float, ...]


def default_partition(num_layers: int, pp_size: int) -> Tuple[int, ...]:
    """The split get_pp_indices makes without SGLANG_PP_LAYER_PARTITION."""
    from sglang.srt.distributed.utils import get_pp_indices

    bounds = [get_pp_indices(num_layers, s, pp_size) for s in range(pp_size)]
    return tuple(end - start for start, end in bounds)


def _prefix(values: Sequence[float]) -> List[float]:
    out = [0.0]
    for value in values:
        out.append(out[-1] + value)
    return out


class _Search:
    def __init__(
        self,
        kinds: Sequence[str],
        m: PPPartitionCostModel,
        pp_size: int,
        memory: Optional[StageMemory],
        proxy_tokens: int,
    ) -> None:
        self.num_layers = len(kinds)
        self.m = m
        self.pp_size = pp_size
        self.memory = memory
        self.proxy_tokens = proxy_tokens
        # Prefix sums over the layers, so any stage's totals cost O(1).
        self._cost = _prefix([m.layer_cost.get(k, 1.0) for k in kinds])
        self._kv = _prefix([m.kv_bytes_per_token.get(k, 0.0) for k in kinds])
        self._used = _prefix(
            [
                m.layer_weight_bytes.get(k, 0.0) + m.layer_state_bytes.get(k, 0.0)
                for k in kinds
            ]
        )
        # A hybrid model's stages each size a pool per layer kind and fail on an
        # empty one, so every stage keeps every kind.
        required = sorted(set(kinds)) if len(set(kinds)) > 1 else []
        self._required = [_prefix([k == kind for k in kinds]) for kind in required]
        self.headroom_on = memory is not None and bool(m.proxy_bytes_per_token)
        if self.headroom_on:
            self.reserved = m.activation_headroom_bytes + (
                m.activation_headroom_fraction * min(memory.free_bytes)
            )

    def cost(self, s: int, start: int, n: int) -> float:
        cost = self._cost[start + n] - self._cost[start]
        if s == 0:
            cost += self.m.first_stage_cost
        if s == self.pp_size - 1:
            cost += self.m.last_stage_cost
        return cost

    def kv_capacity(self, s: int, start: int, n: int) -> float:
        """Tokens of KV the stage holds; inf without a KV model or KV layers."""
        per_token = self._kv[start + n] - self._kv[start]
        if self.memory is None or per_token <= 0:
            return math.inf
        used = self._used[start + n] - self._used[start]
        if s == 0:
            used += self.m.first_stage_weight_bytes
        if s == self.pp_size - 1:
            used += self.m.last_stage_weight_bytes
        return (self.memory.static_bytes[s] - used) / per_token

    def min_kv_capacity(self, part: Sequence[int]) -> float:
        starts = [sum(part[:s]) for s in range(len(part))]
        return min(self.kv_capacity(s, starts[s], n) for s, n in enumerate(part))

    def headroom(self, s: int, start: int, n: int) -> float:
        """Bytes left outside the static pools after the stage's activations."""
        if not self.headroom_on:
            return math.inf
        per_token = self.m.proxy_bytes_per_token
        tokens = self.proxy_tokens
        used = self.reserved
        if s > 0:
            used += self.m.proxy_buffers_in * tokens * per_token[start]
        if s < self.pp_size - 1:
            used += self.m.proxy_buffers_out * tokens * per_token[start + n]
        return self.memory.free_bytes[s] - used

    def feasible(self, s: int, start: int, n: int, kv_floor: float) -> bool:
        if any(count[start + n] == count[start] for count in self._required):
            return False
        # A negative first/last-stage intercept fits stages of many layers only.
        if self.cost(s, start, n) <= 0:
            return False
        if self.headroom(s, start, n) < 0:
            return False
        return kv_floor <= 0 or self.kv_capacity(s, start, n) >= kv_floor

    def best(
        self,
        speed: Sequence[float],
        kv_floor: float,
        cap: float = math.inf,
    ) -> Tuple[float, Tuple[int, ...]]:
        """With cap = inf: the least max stage cost (and a split reaching it). With a
        finite cap: the split whose stage costs stay within it with the least sum of
        squares. Ties go to the smallest split tuple, so the result depends only on
        the inputs."""
        num_layers, pp = self.num_layers, self.pp_size
        minmax = math.isinf(cap)
        none = (math.inf, ())
        # best[j]: (objective, split) placing the first j layers on stages 0..s.
        best = [none] * (num_layers + 1)
        best[0] = (0.0, ())
        for s in range(pp):
            nxt = [none] * (num_layers + 1)
            for j in range(s + 1, num_layers - (pp - 1 - s) + 1):
                for n in range(1, j - s + 1):
                    prev_value, prev_part = best[j - n]
                    if math.isinf(prev_value) or not self.feasible(
                        s, j - n, n, kv_floor
                    ):
                        continue
                    c = round(self.cost(s, j - n, n) * speed[s], 6)
                    if c > cap:
                        continue
                    value = max(prev_value, c) if minmax else prev_value + c * c
                    cand = (round(value, 6), prev_part + (n,))
                    if cand < nxt[j]:
                        nxt[j] = cand
            best = nxt
        return best[num_layers]


def best_partition(
    kinds: Sequence[str],
    speed: Sequence[float],
    m: PPPartitionCostModel,
    memory: Optional[StageMemory] = None,
    kv_floor: float = 0.0,
    proxy_tokens: int = 0,
) -> Tuple[Tuple[int, ...], float]:
    """The split of len(kinds) layers over len(speed) stages minimizing the slowest
    stage's cost x speed, then the sum of squares of all stages' (more slack on the
    others). Every stage keeps every layer kind of a hybrid model, >= kv_floor KV
    tokens and a non-negative activation headroom (when the cost model defines them).
    Returns the split and its slowest stage's cost."""
    search = _Search(kinds, m, len(speed), memory, proxy_tokens)
    value, part = search.best(speed, kv_floor)
    if not part:
        raise RuntimeError(
            f"no PP split of {len(kinds)} layers over {len(speed)} stages keeps "
            + ", ".join(
                ["every layer kind", "a positive cost"]
                + ([f"{kv_floor:.0f} KV tokens"] if kv_floor > 0 else [])
                + (["the activation headroom"] if search.headroom_on else [])
            )
            + " on every stage"
        )
    _, part = search.best(speed, kv_floor, cap=value)
    return part, value


def burn_unit_ms(seconds: float, device, hidden_size: int) -> Tuple[float, int]:
    """Mean time of one burn unit (4 bf16 GEMMs at a prefill shape, then a 1 GiB
    copy) over the second half of a `seconds` burn, and the burn's peak device bytes.
    Callers start it on every GPU at once, so each runs at its loaded clock."""
    import torch

    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    a = torch.randn(16384, hidden_size, device=device, dtype=torch.bfloat16)
    b = torch.randn(hidden_size, 4096, device=device, dtype=torch.bfloat16)
    c = torch.empty(16384, 4096, device=device, dtype=torch.bfloat16)
    x = torch.empty(512 << 20, device=device, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    times, t0 = [], time.time()
    while time.time() < t0 + seconds:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall = time.time()
        start.record()
        for _ in range(4):
            torch.matmul(a, b, out=c)
        y.copy_(x)
        end.record()
        end.synchronize()
        times.append((wall, start.elapsed_time(end)))
    half = [ms for w, ms in times if w >= t0 + seconds / 2] or [ms for _, ms in times]
    peak = torch.cuda.max_memory_allocated(device) - base
    del a, b, c, x, y
    torch.cuda.empty_cache()
    return sum(half) / len(half), peak


class _RankRow(msgspec.Struct, frozen=True):
    """What every rank contributes to the gather."""

    pp_rank: int
    speed: float
    burn_peak_bytes: int
    error: str


def _gather_or_raise(
    all_gather: Callable[[object], List[object]], row: _RankRow
) -> List[_RankRow]:
    """All-gather one row per rank; if any rank failed, raise the same error on all."""
    rows = all_gather(row)
    errors = [f"rank {i}: {r.error}" for i, r in enumerate(rows) if r.error]
    if errors:
        raise RuntimeError("PP partition auto failed: " + "; ".join(errors))
    return rows


def _speed_factors(
    stage_speed: Sequence[float], scale: float
) -> Tuple[List[float], List[int]]:
    """Per-stage factors around the median stage (higher = slower), clamped to
    _FACTOR_BOUNDS; also returns the stages that were clamped. The median keeps an
    outlier from shifting the healthy stages, which would loosen its own bound. With
    two stages the median is their mean, so an outlier cannot be told apart there and
    the clamp bounds the pair's ratio at 1.25 / 0.8."""
    median = statistics.median(stage_speed)
    low, high = _FACTOR_BOUNDS
    raw = [1.0 + scale * (v / median - 1.0) for v in stage_speed]
    factors = [min(max(f, low), high) for f in raw]
    clamped = [s for s, (f, g) in enumerate(zip(raw, factors)) if f != g]
    if clamped:
        logger.warning(
            "PP partition auto: speed factors %s outside %s clamped on stages %s",
            [round(f, 4) for f in raw],
            _FACTOR_BOUNDS,
            clamped,
        )
    return factors, clamped


def resolve(
    *,
    kinds: Sequence[str],
    pp_rank: int,
    pp_size: int,
    all_gather: Callable[[object], List[object]],
    measure: Callable[[], Tuple[float, int]],
    pre_model_load_bytes: float,
    mem_fraction_static: float,
    m: PPPartitionCostModel,
    proxy_tokens: int,
    fixed_speed: Sequence[str] = (),
    burn_scale: float = 1.0,
) -> Tuple[Tuple[int, ...], dict]:
    """Collective over the world group: every rank measures (or takes its fixed
    factor), the ranks gather speed and free memory, and each runs the same search on
    the same rows, so all ranks choose the same split. A rank whose measurement fails
    still joins the gather, and then every rank raises its error."""
    try:
        if fixed_speed:
            factors = [float(v) for v in fixed_speed]
            if len(factors) != pp_size or not all(
                v > 0 and math.isfinite(v) for v in factors
            ):
                raise ValueError(
                    f"SGLANG_PP_PARTITION_AUTO_SPEED needs {pp_size} positive "
                    f"factors, got {list(fixed_speed)}"
                )
            speed, peak = factors[pp_rank], 0
        else:
            speed, peak = measure()
        row = _RankRow(pp_rank, speed, peak, "")
    except Exception as exc:  # The other ranks are already headed for the gather.
        row = _RankRow(pp_rank, math.nan, 0, repr(exc))
    rows = _gather_or_raise(all_gather, row)
    stage_speed = [max(r.speed for r in rows if r.pp_rank == s) for s in range(pp_size)]
    # As kv_cache_configurator sizes the pools: the world's least free memory before
    # the load, split into static (x mem_fraction_static) and slack (the rest).
    free = pre_model_load_bytes
    memory = StageMemory(
        static_bytes=(free * mem_fraction_static,) * pp_size,
        free_bytes=(free * (1.0 - mem_fraction_static),) * pp_size,
    )
    # burn_scale stretches the burn's speed spread around 1 (fixed factors are used as is).
    factors, clamped = _speed_factors(stage_speed, 1.0 if fixed_speed else burn_scale)
    search = _Search(kinds, m, pp_size, memory, proxy_tokens)
    reference = default_partition(len(kinds), pp_size)
    floor = 0.0
    if m.kv_bytes_per_token:
        floor = m.kv_floor * search.min_kv_capacity(reference)
    part, value = best_partition(kinds, factors, m, memory, floor, proxy_tokens)
    starts = [sum(part[:s]) for s in range(pp_size)]
    ref_value = max(
        search.cost(s, sum(reference[:s]), reference[s]) * factors[s]
        for s in range(pp_size)
    )
    report = {
        "rank_speed": [round(r.speed, 3) for r in rows],
        "burn_peak_mib": max(r.burn_peak_bytes for r in rows) >> 20,
        "free_gib_min": round(free / (1 << 30), 1),
        "factors": [round(x, 4) for x in factors],
        "clamped_stages": clamped,
        "source": "fixed" if fixed_speed else f"burn x{burn_scale:g}",
        "reference": list(reference),
        "reference_max_cost": round(ref_value, 3),
        "split": list(part),
        "split_max_cost": round(value, 3),
    }
    if floor > 0:
        report["kv_floor_tokens"] = int(floor)
        report["kv_tokens_min"] = int(search.min_kv_capacity(part))
    if search.headroom_on:
        report["headroom_mib"] = [
            int(search.headroom(s, starts[s], part[s])) >> 20 for s in range(pp_size)
        ]
    return part, report


def layer_kinds(model_config) -> List[str]:
    """Each layer's kind: full or linear attention in a hybrid linear-attention
    model, full or sliding-window attention in a hybrid SWA model, else one kind."""
    from sglang.srt.configs.hybrid_arch import mambaish_config

    num_layers = model_config.num_hidden_layers
    hybrid = mambaish_config(model_config)
    if hybrid is not None:
        full = set(hybrid.full_attention_layer_ids)
        return [
            "full_attention" if i in full else "linear_attention"
            for i in range(num_layers)
        ]
    if model_config.is_hybrid_swa:
        swa = set(model_config.swa_attention_layer_ids)
        return [
            "sliding_attention" if i in swa else "full_attention"
            for i in range(num_layers)
        ]
    return ["layer"] * num_layers


def _load_cost_model(
    *, path: str, kinds: Sequence[str], is_rank0: bool, preflight_error: str, all_gather
) -> PPPartitionCostModel:
    """Rank 0 reads the JSON; every rank gathers it with every rank's preflight
    error, so a failure on one rank fails all of them instead of leaving them
    waiting in a later collective."""
    data, error = None, preflight_error
    if is_rank0 and path and not error:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as exc:
            error = repr(exc)
    rows = all_gather((data, error))
    errors = [f"rank {i}: {e}" for i, (_, e) in enumerate(rows) if e]
    if errors:
        raise RuntimeError("PP partition auto failed: " + "; ".join(errors))
    data = rows[0][0]
    return (
        PPPartitionCostModel.from_json(data, kinds) if data else PPPartitionCostModel()
    )


def maybe_apply(
    *,
    model_config,
    is_draft_worker: bool,
    mem_fraction_static: float,
    pre_model_load_memory: float,
) -> None:
    """SGLANG_ENABLE_PP_PARTITION_AUTO: choose the split before the target model's
    weights load and set SGLANG_PP_LAYER_PARTITION in this process (every rank sets
    the same value). A collective over the world group. pre_model_load_memory is the
    runner's world-minimum free memory before the load, in GiB."""
    import torch

    from sglang.srt.distributed.parallel_state import get_world_group
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel, get_schedule

    par = get_parallel()
    if (
        par.pp_size <= 1
        or is_draft_worker
        or not envs.SGLANG_ENABLE_PP_PARTITION_AUTO.get()
    ):
        return
    world = get_world_group()

    def gather(obj):
        out = [None] * world.world_size
        torch.distributed.all_gather_object(out, obj, group=world.cpu_group)
        return out

    preflight = ""
    if envs.SGLANG_PP_LAYER_PARTITION.get() is not None:
        preflight = (
            "SGLANG_PP_LAYER_PARTITION and SGLANG_ENABLE_PP_PARTITION_AUTO are "
            "mutually exclusive; unset one"
        )
    elif not torch.cuda.is_available():
        preflight = "SGLANG_ENABLE_PP_PARTITION_AUTO needs CUDA GPUs"
    kinds = layer_kinds(model_config)
    m = _load_cost_model(
        path=envs.SGLANG_PP_PARTITION_AUTO_COST_MODEL.get(),
        kinds=kinds,
        is_rank0=world.rank_in_group == 0,
        preflight_error=preflight,
        all_gather=gather,
    )
    burn_s = envs.SGLANG_PP_PARTITION_AUTO_BURN_S.get()

    def measure():
        # Every GPU burns at the same time; the runner has already set this rank's device.
        torch.distributed.barrier(group=world.cpu_group)
        device = torch.device("cuda", torch.cuda.current_device())
        return burn_unit_ms(burn_s, device, model_config.hidden_size)

    proxy_tokens = m.proxy_tokens
    if m.proxy_bytes_per_token and not proxy_tokens:
        proxy_tokens = get_schedule().chunked_prefill_size
    t0 = time.time()
    part, report = resolve(
        kinds=kinds,
        pp_rank=par.pp_rank,
        pp_size=par.pp_size,
        all_gather=gather,
        measure=measure,
        pre_model_load_bytes=pre_model_load_memory * (1 << 30),
        mem_fraction_static=mem_fraction_static,
        m=m,
        proxy_tokens=proxy_tokens,
        fixed_speed=envs.SGLANG_PP_PARTITION_AUTO_SPEED.get(),
        burn_scale=envs.SGLANG_PP_PARTITION_AUTO_BURN_SCALE.get(),
    )
    # Process-wide, like a user-set split; draft workers run in a singleton PP group,
    # which get_pp_indices exempts from it.
    envs.SGLANG_PP_LAYER_PARTITION.set(",".join(map(str, part)))
    report["seconds"] = round(time.time() - t0, 1)
    if world.rank_in_group == 0:
        logger.info("PP partition auto: %s", json.dumps(report))
