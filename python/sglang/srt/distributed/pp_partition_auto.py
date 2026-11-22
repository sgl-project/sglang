"""Boot-time PP layer split from each GPU's measured speed (SGLANG_ENABLE_PP_PARTITION_AUTO).

GPUs of one model run at slightly different speeds (power and clock limits), and a
pipeline runs at the pace of its slowest (stage, GPU) pair. Before the weights load,
every rank measures its GPU with a short concurrent burn (or takes a fixed factor
from SGLANG_PP_PARTITION_AUTO_SPEED), the ranks all-gather the results, and each
runs the same deterministic search: the contiguous split that minimizes the slowest
stage's cost x speed factor, then the sum of squares of the stage costs. Rank 0's
split is broadcast and every rank asserts it computed the same one, then it replaces
SGLANG_PP_LAYER_PARTITION, which the model reads when it builds its layer range.

Layer costs are uniform unless a cost model (SGLANG_PP_PARTITION_AUTO_COST_MODEL)
gives a cost per layer kind. Every stage keeps every layer kind of a hybrid model.
The cost model can also bound each stage's KV token capacity and its activation
headroom outside the static memory pools.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import msgspec

logger = logging.getLogger(__name__)


class PPPartitionCostModel(msgspec.Struct, frozen=True, kw_only=True):
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
    def from_json(cls, path: str) -> PPPartitionCostModel:
        with open(path, "rb") as f:
            return msgspec.json.decode(f.read(), type=cls)


class StageMemory(msgspec.Struct, frozen=True):
    """Bytes per stage: static_bytes holds the static pools (weights, state, KV),
    free_bytes the rest, which activations use."""

    static_bytes: Tuple[float, ...]
    free_bytes: Tuple[float, ...]


def even_partition(num_layers: int, pp_size: int) -> Tuple[int, ...]:
    """get_pp_indices' default split: the last `num_layers % pp_size` stages take one
    layer more."""
    base, extra = divmod(num_layers, pp_size)
    return tuple(base + (s >= pp_size - extra) for s in range(pp_size))


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
            assert len(m.proxy_bytes_per_token) == len(kinds) + 1, (
                "proxy_bytes_per_token needs one entry per end layer 0..num_layers"
            )
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
            f"no PP split of {len(kinds)} layers over {len(speed)} stages keeps every "
            f"layer kind, {kv_floor:.0f} KV tokens and the activation headroom on "
            "every stage"
        )
    _, part = search.best(speed, kv_floor, cap=value)
    return part, value


def burn_unit_ms(seconds: float, device, hidden_size: int) -> float:
    """Mean time of one burn unit (4 bf16 GEMMs at a prefill shape, then a 1 GiB
    copy) over the second half of a `seconds` burn. Callers start it on every GPU at
    once, so each runs at its loaded clock."""
    import torch

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
    del a, b, c, x, y
    torch.cuda.empty_cache()
    return sum(half) / len(half)


def resolve(
    *,
    kinds: Sequence[str],
    pp_rank: int,
    pp_size: int,
    all_gather: Callable[[object], List[object]],
    broadcast_from_0: Callable[[object], object],
    measure: Callable[[], float],
    total_bytes: float,
    mem_fraction_static: float,
    m: PPPartitionCostModel,
    reference: Tuple[int, ...],
    proxy_tokens: int,
    fixed_speed: Sequence[float] = (),
    kv_floor: float = 1.0,
    burn_scale: float = 1.0,
) -> Tuple[Tuple[int, ...], dict]:
    """Collective over the world group: gather (pp_rank, speed, GPU memory) from every
    rank, take each stage's slowest rank and smallest GPU, search, broadcast rank 0's
    split and assert every rank found the same. Returns the split and a report."""
    speed = float(fixed_speed[pp_rank]) if fixed_speed else measure()
    rows = all_gather((pp_rank, speed, total_bytes))
    stage_speed = [max(v for r, v, _ in rows if r == s) for s in range(pp_size)]
    totals = [min(b for r, _, b in rows if r == s) for s in range(pp_size)]
    memory = StageMemory(
        static_bytes=tuple(t * mem_fraction_static for t in totals),
        free_bytes=tuple(t * (1.0 - mem_fraction_static) for t in totals),
    )
    mean = sum(stage_speed) / pp_size
    # burn_scale stretches the burn's speed spread around 1 (fixed factors are used as is).
    scale = 1.0 if fixed_speed else burn_scale
    factors = [1.0 + scale * (v / mean - 1.0) for v in stage_speed]
    search = _Search(kinds, m, pp_size, memory, proxy_tokens)
    floor = 0.0
    if m.kv_bytes_per_token:
        # KV token capacity is the stage minimum; keep kv_floor x the reference's.
        floor = kv_floor * search.min_kv_capacity(reference)
    part, value = best_partition(kinds, factors, m, memory, floor, proxy_tokens)
    chosen = tuple(broadcast_from_0(part))
    assert chosen == part, f"PP split {part} differs from rank 0's {chosen}"
    starts = [sum(part[:s]) for s in range(pp_size)]
    ref_value = max(
        search.cost(s, sum(reference[:s]), reference[s]) * factors[s]
        for s in range(pp_size)
    )
    report = {
        "factors": [round(x, 4) for x in factors],
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


def maybe_apply(
    *, model_config, is_draft_worker: bool, mem_fraction_static: float
) -> None:
    """SGLANG_ENABLE_PP_PARTITION_AUTO: choose the split before the target model's
    weights load and set SGLANG_PP_LAYER_PARTITION in this process (every rank sets
    the same value). A collective over the world group."""
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
    kinds = layer_kinds(model_config)
    path = envs.SGLANG_PP_PARTITION_AUTO_COST_MODEL.get()
    m = PPPartitionCostModel.from_json(path) if path else PPPartitionCostModel()
    env_part = envs.SGLANG_PP_LAYER_PARTITION.get()
    reference = (
        tuple(int(x) for x in env_part.split(","))
        if env_part
        else even_partition(len(kinds), par.pp_size)
    )
    world = get_world_group()
    device = torch.device("cuda", torch.cuda.current_device())
    total = torch.cuda.get_device_properties(device).total_memory
    burn_s = envs.SGLANG_PP_PARTITION_AUTO_BURN_S.get()

    def gather(obj):
        out = [None] * world.world_size
        torch.distributed.all_gather_object(out, obj, group=world.cpu_group)
        return out

    def bcast(obj):
        box = [obj]
        torch.distributed.broadcast_object_list(
            box, src=world.ranks[0], group=world.cpu_group
        )
        return box[0]

    def measure():
        # Every GPU burns at the same time.
        torch.distributed.barrier(group=world.cpu_group)
        return burn_unit_ms(burn_s, device, model_config.hidden_size)

    t0 = time.time()
    part, report = resolve(
        kinds=kinds,
        pp_rank=par.pp_rank,
        pp_size=par.pp_size,
        all_gather=gather,
        broadcast_from_0=bcast,
        measure=measure,
        total_bytes=total,
        mem_fraction_static=mem_fraction_static,
        m=m,
        reference=reference,
        proxy_tokens=m.proxy_tokens or get_schedule().chunked_prefill_size,
        fixed_speed=[float(x) for x in envs.SGLANG_PP_PARTITION_AUTO_SPEED.get()],
        kv_floor=envs.SGLANG_PP_PARTITION_AUTO_KV_FLOOR.get(),
        burn_scale=envs.SGLANG_PP_PARTITION_AUTO_BURN_SCALE.get(),
    )
    envs.SGLANG_PP_LAYER_PARTITION.set(",".join(map(str, part)))
    report["seconds"] = round(time.time() - t0, 1)
    if world.rank_in_group == 0:
        logger.info("PP partition auto: %s", json.dumps(report))
