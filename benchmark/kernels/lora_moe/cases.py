"""Resolved benchmark cases for the BF16 MoE-LoRA campaign (plan §5).

A :class:`MoeLoraBenchCase` is one immutable, fully resolved run description:
scalar values and resolved structures, never lists of candidate values.  Case
records are content-addressed (``case_id``) and serialize to JSON so every
result can be re-adjudicated against the exact case that produced it.

Tensor materialization is deterministic from the case seeds; the route
statistics stored on the case are recomputed from the same seeds and must
match at materialization time.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import msgspec
import torch

from benchmark.kernels.lora_moe.routes import (
    RouteStats,
    generate_token_lora_mapping,
    generate_topk_ids,
    generate_topk_weights,
    resolve_route_stats,
)

# --- Declared vocabulary (typed strings; production enums land with their
# --- first production consumer, per the lifecycle StageValue discipline).

EXPERT_FORMS = ("gated_two_slice", "nongated_one_slice")
ACTIVATIONS = ("silu_glu", "relu2")
EXPERT_ID_DOMAINS = ("ep_local", "global")
SHARED_FACTOR_SIGNATURES = (
    "per_expert",
    "shared_gate_up_a",
    "shared_down_b",
    "shared_both",
)
BASE_PROVIDERS = ("reference_loop", "deep_gemm_masked", "flashinfer_fused")
PROVIDER_GATE_UP_LAYOUTS = ("gate_then_up", "up_then_gate")
EXECUTION_MODES = (
    "eager",
    "decode_graph",
    "breakable_prefill_graph",
    "diagnostic_graph",
)
OVERLAP_STRATEGIES = ("serial_materialized_control",)
DETERMINISTIC_POLICIES = ("fixed_order",)
ROUTE_COEFF_PRECISIONS = ("fp32", "bf16_rounded")
CACHE_STATES = ("hot", "producer_realistic")
SLICE_TARGETS = ("both", "gate_only", "up_only")


class ModelGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """Plan §12.1 model geometry preset (labels geometry, not a selector)."""

    name: str
    hidden_size: int  # H_model
    moe_hidden_size: int  # H_moe
    intermediate_size_global: int  # I_global
    num_experts_global: int  # E_global
    top_k: int
    expert_form: str
    activation: str


MODEL_PRESETS: dict[str, ModelGeometry] = {
    preset.name: preset
    for preset in (
        ModelGeometry(
            name="qwen35_35b",
            hidden_size=2048,
            moe_hidden_size=2048,
            intermediate_size_global=512,
            num_experts_global=256,
            top_k=8,
            expert_form="gated_two_slice",
            activation="silu_glu",
        ),
        ModelGeometry(
            name="qwen35_397b",
            hidden_size=4096,
            moe_hidden_size=4096,
            intermediate_size_global=1024,
            num_experts_global=512,
            top_k=10,
            expert_form="gated_two_slice",
            activation="silu_glu",
        ),
        ModelGeometry(
            name="kimi_k25",
            hidden_size=7168,
            moe_hidden_size=7168,
            intermediate_size_global=2048,
            num_experts_global=384,
            top_k=8,
            expert_form="gated_two_slice",
            activation="silu_glu",
        ),
        ModelGeometry(
            name="glm_52",
            hidden_size=6144,
            moe_hidden_size=6144,
            intermediate_size_global=2048,
            num_experts_global=256,
            top_k=8,
            expert_form="gated_two_slice",
            activation="silu_glu",
        ),
        ModelGeometry(
            name="nemotron3_super",
            hidden_size=4096,
            moe_hidden_size=1024,
            intermediate_size_global=2688,
            num_experts_global=512,
            top_k=22,
            expert_form="nongated_one_slice",
            activation="relu2",
        ),
        ModelGeometry(
            name="nemotron3_nano",
            hidden_size=2688,
            moe_hidden_size=2688,
            intermediate_size_global=1856,
            num_experts_global=128,
            top_k=6,
            expert_form="nongated_one_slice",
            activation="relu2",
        ),
        # Synthetic smoke geometries for cheap CPU/GPU guardrails; never a
        # performance anchor.
        ModelGeometry(
            name="tiny_smoke",
            hidden_size=64,
            moe_hidden_size=64,
            intermediate_size_global=192,
            num_experts_global=8,
            top_k=2,
            expert_form="gated_two_slice",
            activation="silu_glu",
        ),
        ModelGeometry(
            name="tiny_smoke_relu2",
            hidden_size=64,
            moe_hidden_size=64,
            intermediate_size_global=96,
            num_experts_global=8,
            top_k=2,
            expert_form="nongated_one_slice",
            activation="relu2",
        ),
    )
}


class Topology(msgspec.Struct, frozen=True, kw_only=True):
    """Resolved parallel topology; one GPU may simulate the local shape.

    A simulated shape must always be labeled ``local-shape proxy`` in reports;
    it is never distributed evidence.
    """

    tp_size: int = 1
    ep_size: int = 1
    moe_dp_size: int = 1
    ep_rank: int = 0

    def moe_tp_size(self) -> int:
        if self.tp_size % self.ep_size:
            raise ValueError("tp_size must be divisible by ep_size")
        return self.tp_size // self.ep_size


class AdapterCell(msgspec.Struct, frozen=True, kw_only=True):
    """Plan §12.2 compact adapter occupancy cell ``(L_active, B_base, L_capacity)``."""

    active_adapters: int  # L_active
    include_base_rows: bool  # B_base
    slot_capacity: int  # L_capacity
    # Active slot IDs; defaults to the first L_active slots. Sparse/permuted
    # occupancy is expressed by listing non-contiguous IDs.
    active_slot_ids: tuple[int, ...] = ()

    def resolved_slot_ids(self) -> tuple[int, ...]:
        slots = (
            self.active_slot_ids
            if self.active_slot_ids
            else tuple(range(self.active_adapters))
        )
        if len(slots) != self.active_adapters:
            raise ValueError("active_slot_ids length must equal active_adapters")
        if any(slot < 0 or slot >= self.slot_capacity for slot in slots):
            raise ValueError("active slot IDs must lie inside slot capacity")
        return slots


# The §12.2 recommended compact cells, in (L_active, B_base, L_capacity) form.
COMPACT_ADAPTER_CELLS: tuple[AdapterCell, ...] = (
    AdapterCell(active_adapters=0, include_base_rows=True, slot_capacity=8),
    AdapterCell(active_adapters=1, include_base_rows=False, slot_capacity=1),
    AdapterCell(active_adapters=1, include_base_rows=False, slot_capacity=8),
    AdapterCell(active_adapters=1, include_base_rows=True, slot_capacity=8),
    AdapterCell(active_adapters=3, include_base_rows=False, slot_capacity=8),
    AdapterCell(active_adapters=4, include_base_rows=True, slot_capacity=5),
    AdapterCell(active_adapters=7, include_base_rows=True, slot_capacity=8),
    AdapterCell(active_adapters=8, include_base_rows=False, slot_capacity=8),
)


class MoeLoraBenchCase(msgspec.Struct, frozen=True, kw_only=True):
    """One immutable resolved case (plan §5)."""

    case_id: str
    device: str
    source_revision: str

    # Geometry.
    model_preset: str
    hidden_size: int
    moe_hidden_size: int
    intermediate_size_global: int
    num_experts_global: int
    top_k: int
    expert_form: str
    activation: str

    # Resolved topology (local-shape proxy unless truly distributed).
    tp_size: int
    ep_size: int
    moe_dp_size: int
    ep_rank: int
    num_experts_local: int
    intermediate_size_local: int
    intermediate_size_physical: int
    global_expert_offset: int

    # Route domain.
    expert_id_domain: str
    num_tokens: int
    route_generator: str
    route_seed: int
    weight_distribution: str
    weight_seed: int
    route_coeff_precision: str
    routed_scaling_factor: float
    p_valid: int
    p_aligned: int
    e_hit: int
    group_count: int
    group_size_histogram: dict[int, int]

    # Adapter state.
    active_adapters: int
    include_base_rows: bool
    slot_capacity: int
    active_slot_ids: tuple[int, ...]
    mapping_seed: int
    active_rank: int
    max_rank: int
    physical_rank: int
    shared_factor_signature: str

    # Targeting.
    slice_target: str

    # Provider and execution.
    base_provider: str
    provider_gate_up_layout: str
    execution_mode: str
    overlap_strategy: str
    output_dtype: str
    deterministic_policy: str
    cache_state: str

    # Data materialization.
    data_seed: int
    routing_block_size: int


class CaseTensors(msgspec.Struct, kw_only=True):
    """Deterministically materialized inputs for one case (CPU tensors)."""

    hidden_states: torch.Tensor  # [T, H_moe] BF16
    topk_ids: torch.Tensor  # [T, K] int32, declared domain
    topk_weights: torch.Tensor  # [T, K] FP32 normalized
    token_lora_mapping: torch.Tensor  # [T] int64, -1 = base
    routed_expert_to_factor_id: torch.Tensor | None  # provider-domain map
    w13: torch.Tensor  # [E_local, slices*I_local, H_moe] BF16
    w2: torch.Tensor  # [E_local, H_moe, I_local] BF16
    lora_a_gate_up: torch.Tensor  # [L_cap, E_f, slices*R_phys, H_moe] BF16
    lora_b_gate_up: torch.Tensor  # [L_cap, E_local, slices*I_local, R_phys] BF16
    lora_a_down: torch.Tensor  # [L_cap, E_local, R_phys, I_local] BF16
    lora_b_down: torch.Tensor  # [L_cap, E_f_down, H_moe, R_phys] BF16


def capture_source_revision(repo_root: Path | None = None) -> str:
    """Record the exact source identity for the case record."""
    root = repo_root or Path(__file__).resolve().parents[3]
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{head}{'-dirty' if dirty else ''}"


def _slice_count(expert_form: str) -> int:
    if expert_form == "gated_two_slice":
        return 2
    if expert_form == "nongated_one_slice":
        return 1
    raise ValueError(f"unknown expert form {expert_form!r}")


def _factor_expert_count(shared: str, site: str, num_experts_local: int) -> int:
    if site == "gate_up_a":
        return 1 if shared in ("shared_gate_up_a", "shared_both") else num_experts_local
    if site == "down_b":
        return 1 if shared in ("shared_down_b", "shared_both") else num_experts_local
    return num_experts_local


def build_case(
    *,
    device: str,
    model_preset: str,
    topology: Topology = Topology(),
    adapter_cell: AdapterCell,
    route_generator: str,
    num_tokens: int,
    active_rank: int,
    max_rank: int | None = None,
    physical_rank: int | None = None,
    shared_factor_signature: str = "per_expert",
    slice_target: str = "both",
    expert_id_domain: str = "ep_local",
    weight_distribution: str = "seeded_random",
    route_coeff_precision: str = "fp32",
    routed_scaling_factor: float = 1.0,
    base_provider: str = "reference_loop",
    provider_gate_up_layout: str = "gate_then_up",
    execution_mode: str = "eager",
    overlap_strategy: str = "serial_materialized_control",
    output_dtype: str = "bfloat16",
    cache_state: str = "hot",
    routing_block_size: int = 16,
    seed: int = 0,
    source_revision: str | None = None,
) -> MoeLoraBenchCase:
    """Resolve one immutable case, including host-side route statistics."""
    geometry = MODEL_PRESETS[model_preset]
    if geometry.top_k > 0 and num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    moe_tp = topology.moe_tp_size()
    if geometry.num_experts_global % topology.ep_size:
        raise ValueError("E_global must divide by ep_size")
    if geometry.intermediate_size_global % moe_tp:
        raise ValueError("I_global must divide by moe_tp")
    num_experts_local = geometry.num_experts_global // topology.ep_size
    intermediate_local = geometry.intermediate_size_global // moe_tp
    resolved_max_rank = max_rank if max_rank is not None else active_rank
    resolved_physical_rank = (
        physical_rank if physical_rank is not None else resolved_max_rank
    )
    if not active_rank <= resolved_max_rank <= resolved_physical_rank:
        raise ValueError("require active_rank <= max_rank <= physical_rank")

    route_seed = seed * 1000 + 1
    weight_seed = seed * 1000 + 2
    mapping_seed = seed * 1000 + 3
    data_seed = seed * 1000 + 4

    routable = (
        geometry.num_experts_global
        if expert_id_domain == "global" or route_generator == "no_local"
        else num_experts_local
    )
    topk_ids = generate_topk_ids(
        route_generator=route_generator,
        num_tokens=num_tokens,
        top_k=geometry.top_k,
        num_routable_experts=routable,
        num_local_experts=num_experts_local,
        seed=route_seed,
    )
    token_lora_mapping = generate_token_lora_mapping(
        num_tokens=num_tokens,
        active_slot_ids=adapter_cell.resolved_slot_ids(),
        include_base_rows=adapter_cell.include_base_rows,
        seed=mapping_seed,
    )
    factor_map = _build_factor_map(
        expert_id_domain=expert_id_domain,
        route_generator=route_generator,
        num_experts_global=geometry.num_experts_global,
        num_experts_local=num_experts_local,
        ep_rank=topology.ep_rank,
    )
    stats = resolve_route_stats(
        topk_ids=topk_ids,
        token_lora_mapping=token_lora_mapping,
        factor_expert_count=num_experts_local,
        max_loras=adapter_cell.slot_capacity,
        block_size=routing_block_size,
        routed_expert_to_factor_id=factor_map,
    )

    body = dict(
        device=device,
        source_revision=source_revision or capture_source_revision(),
        model_preset=model_preset,
        hidden_size=geometry.hidden_size,
        moe_hidden_size=geometry.moe_hidden_size,
        intermediate_size_global=geometry.intermediate_size_global,
        num_experts_global=geometry.num_experts_global,
        top_k=geometry.top_k,
        expert_form=geometry.expert_form,
        activation=geometry.activation,
        tp_size=topology.tp_size,
        ep_size=topology.ep_size,
        moe_dp_size=topology.moe_dp_size,
        ep_rank=topology.ep_rank,
        num_experts_local=num_experts_local,
        intermediate_size_local=intermediate_local,
        intermediate_size_physical=intermediate_local,
        global_expert_offset=topology.ep_rank * num_experts_local,
        expert_id_domain=expert_id_domain,
        num_tokens=num_tokens,
        route_generator=route_generator,
        route_seed=route_seed,
        weight_distribution=weight_distribution,
        weight_seed=weight_seed,
        route_coeff_precision=route_coeff_precision,
        routed_scaling_factor=routed_scaling_factor,
        p_valid=stats.p_valid,
        p_aligned=stats.p_aligned,
        e_hit=stats.e_hit,
        group_count=stats.group_count,
        group_size_histogram=stats.group_size_histogram,
        active_adapters=adapter_cell.active_adapters,
        include_base_rows=adapter_cell.include_base_rows,
        slot_capacity=adapter_cell.slot_capacity,
        active_slot_ids=adapter_cell.resolved_slot_ids(),
        mapping_seed=mapping_seed,
        active_rank=active_rank,
        max_rank=resolved_max_rank,
        physical_rank=resolved_physical_rank,
        shared_factor_signature=shared_factor_signature,
        slice_target=slice_target,
        base_provider=base_provider,
        provider_gate_up_layout=provider_gate_up_layout,
        execution_mode=execution_mode,
        overlap_strategy=overlap_strategy,
        output_dtype=output_dtype,
        deterministic_policy="fixed_order",
        cache_state=cache_state,
        data_seed=data_seed,
        routing_block_size=routing_block_size,
    )
    digest_source = msgspec.json.encode(
        {key: body[key] for key in sorted(body) if key != "source_revision"}
    )
    case_id = hashlib.sha256(digest_source).hexdigest()[:16]
    return MoeLoraBenchCase(case_id=case_id, **body)


def _build_factor_map(
    *,
    expert_id_domain: str,
    route_generator: str,
    num_experts_global: int,
    num_experts_local: int,
    ep_rank: int,
) -> torch.Tensor | None:
    """Provider-domain expert IDs -> local factor slots (``-1`` non-owned)."""
    if expert_id_domain == "ep_local" and route_generator != "no_local":
        return None
    domain = (
        num_experts_global
        if expert_id_domain == "global" or route_generator == "no_local"
        else num_experts_local
    )
    factor_map = torch.full((domain,), -1, dtype=torch.int32)
    offset = ep_rank * num_experts_local if expert_id_domain == "global" else 0
    owned = torch.arange(num_experts_local, dtype=torch.int32)
    factor_map[offset : offset + num_experts_local] = owned
    return factor_map


def materialize_case_tensors(case: MoeLoraBenchCase) -> CaseTensors:
    """Deterministically build the CPU input tensors for one case.

    Scaling keeps the LoRA signal well above the BF16 noise floor of the base
    output; the signal-gate engine still refuses any case that lands below the
    §21 validity floor.  LoRA rank tails beyond ``active_rank`` are ZERO by
    the loader contract (kernels execute the full physical rank).
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.data_seed)
    slices = _slice_count(case.expert_form)
    h = case.moe_hidden_size
    i_local = case.intermediate_size_local
    e_local = case.num_experts_local
    l_cap = case.slot_capacity
    r_phys = case.physical_rank
    r_active = case.active_rank

    def randn(*shape: int, scale: float) -> torch.Tensor:
        return (
            torch.randn(shape, generator=generator, dtype=torch.float32) * scale
        ).to(torch.bfloat16)

    hidden = randn(case.num_tokens, h, scale=1.0)
    w13 = randn(e_local, slices * i_local, h, scale=h**-0.5)
    w2 = randn(e_local, h, i_local, scale=i_local**-0.5)

    e_f_gate = _factor_expert_count(case.shared_factor_signature, "gate_up_a", e_local)
    e_f_down = _factor_expert_count(case.shared_factor_signature, "down_b", e_local)

    def factor(*shape: int, scale: float, rank_axis: int) -> torch.Tensor:
        tensor = randn(*shape, scale=scale)
        if r_active < r_phys:
            index = [slice(None)] * tensor.ndim
            if rank_axis >= 0:
                index[rank_axis] = slice(r_active, r_phys)
                tensor[tuple(index)] = 0
        return tensor

    # Gate/up A packs per-slice factors at physical-rank-spaced offsets.
    lora_a_gate_up = torch.zeros(
        (l_cap, e_f_gate, slices * r_phys, h), dtype=torch.bfloat16
    )
    for slice_id in range(slices):
        block = factor(l_cap, e_f_gate, r_phys, h, scale=h**-0.5, rank_axis=2)
        lora_a_gate_up[:, :, slice_id * r_phys : (slice_id + 1) * r_phys, :] = block

    lora_b_gate_up = torch.zeros(
        (l_cap, e_local, slices * i_local, r_phys), dtype=torch.bfloat16
    )
    for slice_id in range(slices):
        block = factor(
            l_cap, e_local, i_local, r_phys, scale=0.5 * r_phys**-0.5, rank_axis=3
        )
        lora_b_gate_up[:, :, slice_id * i_local : (slice_id + 1) * i_local, :] = block

    lora_a_down = factor(
        l_cap, e_local, r_phys, i_local, scale=i_local**-0.5, rank_axis=2
    )
    lora_b_down = factor(
        l_cap, e_f_down, h, r_phys, scale=0.5 * r_phys**-0.5, rank_axis=3
    )

    _apply_slice_target(case, lora_a_gate_up, lora_b_gate_up, r_phys, i_local)

    topk_ids = generate_topk_ids(
        route_generator=case.route_generator,
        num_tokens=case.num_tokens,
        top_k=case.top_k,
        num_routable_experts=(
            case.num_experts_global
            if case.expert_id_domain == "global"
            or case.route_generator == "no_local"
            else e_local
        ),
        num_local_experts=e_local,
        seed=case.route_seed,
    )
    return CaseTensors(
        hidden_states=hidden,
        topk_ids=topk_ids,
        topk_weights=generate_topk_weights(
            weight_distribution=case.weight_distribution,
            num_tokens=case.num_tokens,
            top_k=case.top_k,
            seed=case.weight_seed,
        ),
        token_lora_mapping=generate_token_lora_mapping(
            num_tokens=case.num_tokens,
            active_slot_ids=case.active_slot_ids,
            include_base_rows=case.include_base_rows,
            seed=case.mapping_seed,
        ),
        routed_expert_to_factor_id=_build_factor_map(
            expert_id_domain=case.expert_id_domain,
            route_generator=case.route_generator,
            num_experts_global=case.num_experts_global,
            num_experts_local=e_local,
            ep_rank=case.ep_rank,
        ),
        w13=w13,
        w2=w2,
        lora_a_gate_up=lora_a_gate_up,
        lora_b_gate_up=lora_b_gate_up,
        lora_a_down=lora_a_down,
        lora_b_down=lora_b_down,
    )


def _apply_slice_target(
    case: MoeLoraBenchCase,
    lora_a_gate_up: torch.Tensor,
    lora_b_gate_up: torch.Tensor,
    physical_rank: int,
    intermediate_local: int,
) -> None:
    """Zero the untargeted gate/up slice factors (gate-only / up-only cases)."""
    if case.slice_target == "both":
        return
    if case.expert_form != "gated_two_slice":
        raise ValueError("slice targeting requires the gated two-slice form")
    drop = 1 if case.slice_target == "gate_only" else 0
    lora_a_gate_up[:, :, drop * physical_rank : (drop + 1) * physical_rank, :] = 0
    lora_b_gate_up[
        :, :, drop * intermediate_local : (drop + 1) * intermediate_local, :
    ] = 0


__all__ = [
    "ACTIVATIONS",
    "BASE_PROVIDERS",
    "CACHE_STATES",
    "COMPACT_ADAPTER_CELLS",
    "DETERMINISTIC_POLICIES",
    "EXECUTION_MODES",
    "EXPERT_FORMS",
    "EXPERT_ID_DOMAINS",
    "MODEL_PRESETS",
    "OVERLAP_STRATEGIES",
    "PROVIDER_GATE_UP_LAYOUTS",
    "ROUTE_COEFF_PRECISIONS",
    "SHARED_FACTOR_SIGNATURES",
    "SLICE_TARGETS",
    "AdapterCell",
    "CaseTensors",
    "ModelGeometry",
    "MoeLoraBenchCase",
    "Topology",
    "build_case",
    "capture_source_revision",
    "materialize_case_tensors",
]
