"""Shared-outer gate/up-A forms at the SERVING-REAL boundary (fifth S3 review).

The archived v1 grid compared the forms at an ideal PREPARED boundary (all
metadata prebuilt, backend chunk policy 16/32/128, dense segment grid).  The
fifth review showed that boundary answers "which kernel is faster given free
metadata" but not "which form wins in serving", because the forms differ in
WHERE their metadata lives:

* the control's pair-domain outer plan and the masked T-plan depend on this
  LAYER's ``topk_ids`` — rebuilt per layer in production;
* the unmasked T-plan and the SGMV segment metadata depend only on
  ``token_slots`` — per-BATCH, built once per forward (the chunked backend's
  own granularity), amortized to ~zero per layer.

So this grid runs ONE boundary, ROUTE_INCLUSIVE, defined as "includes every
PER-LAYER metadata build the form causes" (§63.17): per-layer plans (the
per-expert plan B needs — every arm — plus the arm's own layer-dependent
A-side plan) build inside the thunk; per-batch metadata is prebuilt, which
is not an exclusion — the work does not exist at layer scope.  Params carry
``a_metadata_scope`` so records are self-describing.

Arms (challengers all decided against ``control``):

* ``control``          — outer pair plan (in-thunk) + grouped A + B.
* ``dedup_masked``     — masked T-plan (in-thunk; the mask is topk-dependent)
                         + grouped A over T rows + B(``intermediate_top_k=K``).
* ``dedup_unmasked``   — PREBUILT unmasked T-plan (per-batch, reusable) +
                         grouped A over T rows including no-local tokens
                         (the waste the mask exists to trim, now priced) + B.
* ``sgmv_serving16``   — PREBUILT unmasked segments at the SERVING DEFAULT
                         chunk 16 (``max_lora_chunk_size`` defaults to 16, so
                         ``_determine_chunk_size_for_tokens`` returns 16 at
                         every T) + ``chunked_sgmv_lora_shrink_forward`` + B.
* ``sgmv_maxchunk``    — same, chunk from the 16/32/128 T-policy (the
                         operator opt-in ``--max-lora-chunk-size 128``); this
                         is the v1 grid's chunk geometry.
* ``sgmv_unchunked``   — PREBUILT one-unchunked-segment-per-adapter-run +
                         ``sgemm_lora_a_fwd`` + B.
* ``sgmv_graph16``     — decode cells only (T <= 256): serving CUDA-graph
                         geometry — ``use_cuda_graph=True`` with
                         ``weight_indices``/``seg_indptr`` padded to the
                         decode capacity (``ceil(1/MIN_CHUNK)*bs = T``
                         segments; tail segments empty, early-return), the
                         geometry a captured decode graph replays.

Admission per fixture, before any timing record: masked dedup delta is
BITWISE equal to control; unmasked dedup delta is BITWISE equal to control
(B never reads a no-local token's extra bridge row); each SGMV bridge is
signal-gate-close to the dedup bridge on covered tokens and its delta
gate-close to control's; the graph-geometry bridge is BITWISE equal to the
dense-grid chunk-16 bridge (padding must be output-invisible).

Mixed-rank contract (fifth review): stacked gate/up A factors are
MAX-RANK-SPACED — the up slice starts at row ``R_phys`` regardless of the
active rank — so ``lora_ranks`` handed to SGMV is FORCED to the physical
rank (§63.16).  ``run_mixed_rank_admission`` pins both directions on real
mixed-rank cases (active < physical): forced-physical passes every gate,
and a true-rank ``lora_ranks`` is REQUIRED TO FAIL (it reads the gate
factor's zero tail as the up slice and leaves the up columns unwritten).
Timing needs no mixed cells: every arm executes physical-rank work, so the
mix affects the contract, not the cost.

SGMV records carry the resolved ``get_lora_shrink_config`` tile and the
synthesized segment geometry, so a future retune is visible in provenance.

Usage::

    python3 -m benchmark.kernels.lora_moe.bench_sgmv_real \
        --output sgmv_serving_r16_r64.json --source-revision <sha> \
        [--ranks 16,64] [--adapters 1,4,8] [--domains ep_local,global]
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import torch

from benchmark.kernels.lora_moe.bench_shared_dedup import _SharedFixture
from benchmark.kernels.lora_moe.cases import AdapterCell, Topology, build_case
from benchmark.kernels.lora_moe.crossover_ledger import decide_cell
from benchmark.kernels.lora_moe.lora_a_execution import LoraAExecutionSpec
from benchmark.kernels.lora_moe.lora_a_shared import (
    build_token_adapter_plan,
    masked_token_slots_for_plan,
    run_shared_gate_up,
)
from benchmark.kernels.lora_moe.signal_gates import require_delta_close
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_ROUTE_INCLUSIVE,
    CACHE_L2_HOT_GRAPH,
    measure,
    new_suite,
    write_suite,
)
from sglang.kernels.ops.gemm.chunked_sgmv_shrink import (
    chunked_sgmv_lora_shrink_forward,
)
from sglang.kernels.ops.gemm.lora_tuning_config import get_lora_shrink_config
from sglang.kernels.ops.gemm.sgemm_lora_a import sgemm_lora_a_fwd
from sglang.srt.lora.sgl_lora.bf16 import grouped_lora_a, stock_grouped_lora_b
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.utils import LoRABatchInfo

T_GRID = (16, 64, 256, 2048, 8192)
DECODE_T_MAX = 256  # serving CUDA graphs are a decode geometry
SERVING_CHUNK = 16  # MIN_CHUNK_SIZE == default max_lora_chunk_size
SEEDS = (11, 137, 997)
REPEATS = 2
LORA_A = PROVISIONAL_LAUNCH_CONFIG.lora_a
LORA_B = PROVISIONAL_LAUNCH_CONFIG.lora_b

SPEC_CONTROL = LoraAExecutionSpec(site="gate_up", ownership="grouped")
SPEC_DEDUP = LoraAExecutionSpec(
    site="gate_up", ownership="grouped", shared_handling="token_dedup"
)
SPEC_SGMV_CHUNKED = LoraAExecutionSpec(
    site="gate_up",
    ownership="segmented",
    shared_handling="token_dedup",
    variant="chunked",
)
SPEC_SGMV_UNCHUNKED = LoraAExecutionSpec(
    site="gate_up",
    ownership="segmented",
    shared_handling="token_dedup",
    variant="unchunked",
)
# Chunk size and graph padding are launch geometry, not kernel identity —
# the typed spec names the kernel family; the suffix names the geometry.
ARM_KEYS = {
    "control": SPEC_CONTROL.key() + "_charged",
    "dedup_masked": SPEC_DEDUP.key() + "_masked_charged",
    "dedup_unmasked": SPEC_DEDUP.key() + "_unmasked_prepared",
    "sgmv_serving16": SPEC_SGMV_CHUNKED.key() + "_serving16",
    "sgmv_maxchunk": SPEC_SGMV_CHUNKED.key() + "_maxchunk",
    "sgmv_unchunked": SPEC_SGMV_UNCHUNKED.key() + "_prepared",
    "sgmv_unchunked_stable": SPEC_SGMV_UNCHUNKED.key() + "_stable",
    "sgmv_graph16": SPEC_SGMV_CHUNKED.key() + "_graph16",
}
ARM_METADATA_SCOPE = {
    "control": "per_layer",
    "dedup_masked": "per_layer",
    "dedup_unmasked": "per_batch",
    "sgmv_serving16": "per_batch",
    "sgmv_maxchunk": "per_batch",
    "sgmv_unchunked": "per_batch",
    "sgmv_unchunked_stable": "per_batch",
    "sgmv_graph16": "per_batch",
}


def _policy_chunk_size(num_tokens: int, *, max_chunk_size: int) -> int:
    """Mirror of ``_determine_chunk_size_for_tokens`` (chunked_backend.py)."""
    if max_chunk_size <= SERVING_CHUNK:
        return SERVING_CHUNK
    if num_tokens >= 256:
        chunk = 128
    elif num_tokens >= 64:
        chunk = 32
    else:
        chunk = 16
    return min(max_chunk_size, chunk)


def _sorted_valid_runs(slots: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Stable adapter-sorted permutation of valid slots + per-token weights."""
    num_tokens = slots.shape[0]
    order = torch.argsort(slots.to(torch.int64), stable=True)
    num_valid = int((slots >= 0).sum())
    permutation = order[num_tokens - num_valid :].to(torch.int32)
    weights_per_token = slots[permutation.long()].cpu().tolist()
    return permutation, weights_per_token


def synthesize_chunked_batch_info(
    slots: torch.Tensor,
    *,
    max_loras: int,
    physical_rank: int,
    device: torch.device,
    chunk: int,
    graph_capacity: int | None = None,
) -> LoRABatchInfo:
    """Segment metadata the chunked backend would hand this batch.

    ``lora_ranks`` is FORCED to the physical rank (§63.16 mixed-rank
    contract).  ``graph_capacity`` pads ``weight_indices``/``seg_indptr``
    to the captured-graph segment capacity with empty tail segments and
    sets ``use_cuda_graph=True`` — the kernel then launches ``capacity``
    programs per column tile, production's replay geometry.
    """
    permutation, weights_per_token = _sorted_valid_runs(slots)
    seg_indptr = [0]
    seg_weights: list[int] = []
    index = 0
    while index < len(weights_per_token):
        run_end = index
        while (
            run_end < len(weights_per_token)
            and weights_per_token[run_end] == weights_per_token[index]
        ):
            run_end += 1
        for start in range(index, run_end, chunk):
            seg_indptr.append(min(start + chunk, run_end))
            seg_weights.append(weights_per_token[index])
        index = run_end
    num_segments = len(seg_weights)
    if graph_capacity is not None:
        if num_segments > graph_capacity:
            raise AssertionError(
                f"synthesized {num_segments} segments exceed the graph "
                f"capacity {graph_capacity}"
            )
        total = seg_indptr[-1]
        seg_indptr += [total] * (graph_capacity - num_segments)
        seg_weights += [0] * (graph_capacity - num_segments)
    return LoRABatchInfo(
        use_cuda_graph=graph_capacity is not None,
        bs=num_segments,
        num_segments=num_segments,
        seg_indptr=torch.tensor(seg_indptr, dtype=torch.int32, device=device),
        weight_indices=torch.tensor(seg_weights, dtype=torch.int32, device=device),
        lora_ranks=torch.full(
            (max_loras,), physical_rank, dtype=torch.int32, device=device
        ),
        scalings=torch.ones(max_loras, dtype=torch.float, device=device),
        max_len=chunk,
        seg_lens=None,
        permutation=permutation.to(device),
    )


def synthesize_unchunked_batch_info(
    slots: torch.Tensor,
    *,
    max_loras: int,
    physical_rank: int,
    device: torch.device,
    capacity_segments: int | None = None,
    max_len_ceiling: int | None = None,
) -> LoRABatchInfo:
    """One UNCHUNKED segment per adapter run (the triton dense backend's
    shape, ``sgemm_lora_a_fwd``): grid axis 0 is sized by the LONGEST
    segment, so ragged adapter runs pad — the load imbalance the chunked
    variant exists to fix, measured here instead of assumed."""
    permutation, weights_per_token = _sorted_valid_runs(slots)
    seg_indptr = [0]
    seg_weights: list[int] = []
    index = 0
    while index < len(weights_per_token):
        run_end = index
        while (
            run_end < len(weights_per_token)
            and weights_per_token[run_end] == weights_per_token[index]
        ):
            run_end += 1
        seg_indptr.append(run_end)
        seg_weights.append(weights_per_token[index])
        index = run_end
    num_segments = len(seg_weights)
    if capacity_segments is not None:
        if num_segments > capacity_segments:
            raise AssertionError(
                f"{num_segments} adapter runs exceed capacity_segments="
                f"{capacity_segments}"
            )
        total = seg_indptr[-1]
        seg_indptr += [total] * (capacity_segments - num_segments)
        seg_weights += [0] * (capacity_segments - num_segments)
        num_segments = capacity_segments
    seg_lens = [seg_indptr[i + 1] - seg_indptr[i] for i in range(num_segments)]
    max_len = max(seg_lens) if seg_lens else 0
    if max_len_ceiling is not None:
        # A reusable captured graph must launch the grid its worst batch
        # needs; empty tail rows early-return inside the kernel.
        max_len = max_len_ceiling
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=num_segments,
        num_segments=num_segments,
        seg_indptr=torch.tensor(seg_indptr, dtype=torch.int32, device=device),
        weight_indices=torch.tensor(seg_weights, dtype=torch.int32, device=device),
        lora_ranks=torch.full(
            (max_loras,), physical_rank, dtype=torch.int32, device=device
        ),
        scalings=torch.ones(max_loras, dtype=torch.float, device=device),
        max_len=max_len,
        seg_lens=torch.tensor(seg_lens, dtype=torch.int32, device=device),
        permutation=permutation.to(device),
    )


def _build_case(
    *,
    device,
    domain,
    active,
    num_tokens,
    rank,
    seed,
    source_revision,
    physical_rank=None,
):
    return build_case(
        device=str(device),
        model_preset="qwen35_35b",
        topology=Topology(tp_size=8, ep_size=8),
        adapter_cell=AdapterCell(
            active_adapters=active,
            include_base_rows=True,
            slot_capacity=8,
        ),
        route_generator="iid",
        expert_id_domain=domain,
        num_tokens=num_tokens,
        active_rank=rank,
        physical_rank=physical_rank,
        shared_factor_signature="shared_gate_up_a",
        seed=seed,
        source_revision=source_revision,
    )


class _ServingArms:
    """The seven serving-real thunks over one fixture (per-batch metadata
    prebuilt at construction; per-layer plans build inside each thunk)."""

    def __init__(self, fixture: _SharedFixture, device: torch.device) -> None:
        case = fixture.case
        self.fixture = fixture
        self.slots_unmasked = fixture.base.token_slots
        self.slots_masked = masked_token_slots_for_plan(
            fixture.base.token_slots,
            fixture.topk_ids_local,
            num_local_experts=case.num_experts_local,
        )
        # Per-BATCH metadata (topk-independent), prebuilt once like the
        # serving forward does.
        self.tplan_unmasked = build_token_adapter_plan(
            self.slots_unmasked,
            max_loras=case.slot_capacity,
            block_size=case.routing_block_size,
        )
        common = dict(
            max_loras=case.slot_capacity,
            physical_rank=case.physical_rank,
            device=device,
        )
        self.info_serving16 = synthesize_chunked_batch_info(
            self.slots_unmasked, chunk=SERVING_CHUNK, **common
        )
        self.chunk_policy = _policy_chunk_size(case.num_tokens, max_chunk_size=128)
        self.info_maxchunk = synthesize_chunked_batch_info(
            self.slots_unmasked, chunk=self.chunk_policy, **common
        )
        self.info_unchunked = synthesize_unchunked_batch_info(
            self.slots_unmasked, **common
        )
        # Seventh S3 review: the compact unchunked form has data-dependent
        # bs/max_len — captured per fixture, that is ideal fixed geometry,
        # not a reusable production graph. The capacity-stable form fixes
        # num_segments = slot_capacity (empty segments early-return) and
        # max_len = T (the grid ceiling a captured graph must launch), so
        # one capture serves every batch of the bucket.
        self.info_unchunked_stable = synthesize_unchunked_batch_info(
            self.slots_unmasked,
            capacity_segments=case.slot_capacity,
            max_len_ceiling=case.num_tokens,
            **common,
        )
        self.graph_capacity = None
        self.info_graph16 = None
        if case.num_tokens <= DECODE_T_MAX:
            # Decode graph: 1 token/req -> ceil(1/MIN_CHUNK) segment/req,
            # bs = T -> capacity = T segments (chunked_backend
            # init_cuda_graph_batch_info).
            self.graph_capacity = case.num_tokens
            self.info_graph16 = synthesize_chunked_batch_info(
                self.slots_unmasked,
                chunk=SERVING_CHUNK,
                graph_capacity=self.graph_capacity,
                **common,
            )
        self.a_shared_3d = fixture.base.a_gate_up.view(
            case.slot_capacity, -1, case.moe_hidden_size
        )

    def _b_from_token_bridge(self, bridge: torch.Tensor) -> None:
        fixture = self.fixture
        stock_grouped_lora_b(
            bridge,
            fixture.base.b_gate_up,
            fixture.delta,
            fixture.per_expert_plan(),  # per-layer, charged in every arm
            destination_offsets=(0, fixture.base.intermediate),
            config=LORA_B,
            intermediate_top_k=fixture.case.top_k,
        )

    def control(self) -> None:
        self.fixture.control_leg()

    def dedup_masked(self) -> None:
        self.fixture.dedup_leg()

    def dedup_unmasked(self) -> None:
        fixture = self.fixture
        grouped_lora_a(
            fixture.base.hidden_states,
            fixture.base.a_gate_up,
            fixture.rank_out_tokens,
            self.tplan_unmasked,
            config=LORA_A,
        )
        self._b_from_token_bridge(fixture.rank_out_tokens)

    def _sgmv(self, spec: LoraAExecutionSpec, info: LoRABatchInfo) -> None:
        # Sixth S3 review: the SGMV thunks route through the typed
        # shared-site executor, so the recorded spec IS the kernel run.
        fixture = self.fixture
        run_shared_gate_up(
            spec,
            hidden_states=fixture.base.hidden_states,
            gate_up_a=fixture.base.a_gate_up,
            gate_up_b=fixture.base.b_gate_up,
            rank_out=fixture.rank_out_tokens,
            gate_up_delta=fixture.delta,
            a_route=None,  # segmented forms take their route from segments
            per_expert_route=fixture.per_expert_plan(),  # per-layer, charged
            intermediate_size=fixture.base.intermediate,
            config_a=LORA_A,
            config_b=LORA_B,
            segment_info=info,
        )

    def sgmv_serving16(self) -> None:
        self._sgmv(SPEC_SGMV_CHUNKED, self.info_serving16)

    def sgmv_maxchunk(self) -> None:
        self._sgmv(SPEC_SGMV_CHUNKED, self.info_maxchunk)

    def sgmv_graph16(self) -> None:
        self._sgmv(SPEC_SGMV_CHUNKED, self.info_graph16)

    def sgmv_unchunked(self) -> None:
        self._sgmv(SPEC_SGMV_UNCHUNKED, self.info_unchunked)

    def sgmv_unchunked_stable(self) -> None:
        self._sgmv(SPEC_SGMV_UNCHUNKED, self.info_unchunked_stable)

    def thunks(self) -> dict:
        arms = {
            "control": self.control,
            "dedup_masked": self.dedup_masked,
            "dedup_unmasked": self.dedup_unmasked,
            "sgmv_serving16": self.sgmv_serving16,
            "sgmv_maxchunk": self.sgmv_maxchunk,
            "sgmv_unchunked": self.sgmv_unchunked,
            "sgmv_unchunked_stable": self.sgmv_unchunked_stable,
        }
        if self.info_graph16 is not None:
            arms["sgmv_graph16"] = self.sgmv_graph16
        return arms

    def sgmv_params(self, name: str) -> dict:
        """Resolved launch geometry + tile config for SGMV records."""
        info = {
            "sgmv_serving16": self.info_serving16,
            "sgmv_maxchunk": self.info_maxchunk,
            "sgmv_unchunked": self.info_unchunked,
            "sgmv_unchunked_stable": self.info_unchunked_stable,
            "sgmv_graph16": self.info_graph16,
        }.get(name)
        if info is None:
            return {}
        case = self.fixture.case
        if name.startswith("sgmv_unchunked"):
            # Seventh S3 review: sgemm_lora_a_fwd does not consult the
            # chunked tuning table — its blocks are hardcoded in the
            # kernel wrapper. Record what actually launches.
            resolved = {"BLOCK_S": 16, "BLOCK_K": 256, "BLOCK_R": 16}
        else:
            resolved = dict(
                get_lora_shrink_config(
                    K=case.moe_hidden_size,
                    R=case.physical_rank,
                    num_slices=2,
                    chunk_size=info.max_len,
                )
            )
        return {
            "sgmv_chunk_size": info.max_len,
            "sgmv_segments": info.num_segments,
            "sgmv_graph_capacity": (
                info.weight_indices.shape[0] if info.use_cuda_graph else None
            ),
            "sgmv_launch_config": resolved,
        }


def _admit(arms: _ServingArms, label: str) -> None:
    """Every arm must reproduce the control delta before any timing."""
    fixture = arms.fixture
    covered = arms.slots_masked >= 0

    fixture.delta.fill_(71.0)
    arms.control()
    control_delta = fixture.delta.clone()

    fixture.delta.fill_(-3.0)
    arms.dedup_masked()
    dedup_bridge = fixture.rank_out_tokens.clone()
    if not torch.equal(control_delta, fixture.delta):
        raise AssertionError(f"masked dedup != control bitwise at {label}")

    fixture.delta.fill_(17.0)
    fixture.rank_out_tokens.fill_(0.0)
    arms.dedup_unmasked()
    if not torch.equal(control_delta, fixture.delta):
        raise AssertionError(f"unmasked dedup != control bitwise at {label}")

    for name, bridge_fn in (
        (
            "sgmv_serving16",
            lambda: chunked_sgmv_lora_shrink_forward(
                fixture.base.hidden_states, arms.a_shared_3d, arms.info_serving16, 2
            ),
        ),
        (
            "sgmv_maxchunk",
            lambda: chunked_sgmv_lora_shrink_forward(
                fixture.base.hidden_states, arms.a_shared_3d, arms.info_maxchunk, 2
            ),
        ),
        (
            "sgmv_unchunked",
            lambda: sgemm_lora_a_fwd(
                fixture.base.hidden_states, arms.a_shared_3d, arms.info_unchunked, 2
            ),
        ),
    ):
        bridge = bridge_fn()
        if bool(covered.any()):
            require_delta_close(
                bridge[covered].float(),
                dedup_bridge[covered].float(),
                gate_dtype=torch.bfloat16,
                label=f"{name} bridge vs dedup bridge {label}",
            )
        fixture.delta.fill_(29.0)
        getattr(arms, name)()
        require_delta_close(
            fixture.delta.float(),
            control_delta.float(),
            gate_dtype=torch.bfloat16,
            label=f"{name} delta vs control {label}",
        )

    compact = sgemm_lora_a_fwd(
        fixture.base.hidden_states, arms.a_shared_3d, arms.info_unchunked, 2
    )
    stable = sgemm_lora_a_fwd(
        fixture.base.hidden_states,
        arms.a_shared_3d,
        arms.info_unchunked_stable,
        2,
    )
    if bool(covered.any()) and not torch.equal(compact[covered], stable[covered]):
        raise AssertionError(
            f"capacity-stable unchunked geometry changed output at {label} — "
            "padded segments/grid must be output-invisible"
        )

    if arms.info_graph16 is not None:
        dense = chunked_sgmv_lora_shrink_forward(
            fixture.base.hidden_states, arms.a_shared_3d, arms.info_serving16, 2
        )
        padded = chunked_sgmv_lora_shrink_forward(
            fixture.base.hidden_states, arms.a_shared_3d, arms.info_graph16, 2
        )
        if bool(covered.any()) and not torch.equal(dense[covered], padded[covered]):
            raise AssertionError(
                f"graph-capacity padding changed SGMV output at {label} — "
                "padded segments must be output-invisible"
            )


def run_mixed_rank_admission(device: torch.device, domains, source_revision) -> None:
    """§63.16 both directions: forced-physical passes, true-rank must fail.

    Stacked gate/up A rows are max-rank-spaced (up starts at row R_phys), so
    a true-rank ``lora_ranks`` makes the shrink read the gate factor's zero
    tail as the up slice and leaves the physical up columns unwritten.
    """
    for domain in domains:
        for active_rank, physical_rank in ((8, 16), (32, 64)):
            case = _build_case(
                device=device,
                domain=domain,
                active=4,
                num_tokens=64,
                rank=active_rank,
                physical_rank=physical_rank,
                seed=SEEDS[0],
                source_revision=source_revision,
            )
            fixture = _SharedFixture(case, device)
            arms = _ServingArms(fixture, device)
            label = (
                f"mixed-rank {domain} active r{active_rank} "
                f"physical r{physical_rank}"
            )
            _admit(arms, label)

            true_rank_info = synthesize_chunked_batch_info(
                arms.slots_unmasked,
                max_loras=case.slot_capacity,
                physical_rank=case.physical_rank,
                device=device,
                chunk=SERVING_CHUNK,
            )
            true_rank_info.lora_ranks.fill_(active_rank)
            fixture.delta.fill_(71.0)
            arms.control()
            control_delta = fixture.delta.clone()
            fixture.delta.fill_(13.0)
            bridge = chunked_sgmv_lora_shrink_forward(
                fixture.base.hidden_states, arms.a_shared_3d, true_rank_info, 2
            )
            arms._b_from_token_bridge(bridge)
            if torch.allclose(
                fixture.delta.float(), control_delta.float(), rtol=3e-2, atol=3e-2
            ):
                raise AssertionError(
                    f"true-rank lora_ranks unexpectedly matched control at "
                    f"{label} — the forced-physical contract would be vacuous"
                )
            print(f"admitted {label}: forced-physical passes, true-rank fails")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="16,64")
    parser.add_argument("--adapters", default="1,4,8")
    parser.add_argument("--domains", default="ep_local,global")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)
    ranks = tuple(int(rank) for rank in arguments.ranks.split(","))
    adapter_counts = tuple(int(count) for count in arguments.adapters.split(","))
    domains = tuple(arguments.domains.split(","))
    suite = new_suite("sgmv_serving_real", source_revision=arguments.source_revision)

    run_mixed_rank_admission(device, domains, suite.source_revision)

    samples: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    records: dict[tuple, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    samples_eager: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for domain in domains:
        for rank in ranks:
            for active in adapter_counts:
                for num_tokens in T_GRID:
                    for seed in SEEDS:
                        case = _build_case(
                            device=device,
                            domain=domain,
                            active=active,
                            num_tokens=num_tokens,
                            rank=rank,
                            seed=seed,
                            source_revision=suite.source_revision,
                        )
                        fixture = _SharedFixture(case, device)
                        arms = _ServingArms(fixture, device)
                        _admit(
                            arms,
                            f"{domain} T={num_tokens} r{rank} L{active} seed {seed}",
                        )

                        # Per-batch metadata cost diagnostics (host wall-clock;
                        # NOT records — paid once per forward, not per layer).
                        if seed == SEEDS[0]:
                            torch.cuda.synchronize()
                            begin = time.perf_counter()
                            for _ in range(20):
                                synthesize_chunked_batch_info(
                                    arms.slots_unmasked,
                                    max_loras=case.slot_capacity,
                                    physical_rank=case.physical_rank,
                                    device=device,
                                    chunk=SERVING_CHUNK,
                                )
                            torch.cuda.synchronize()
                            synth_us = (time.perf_counter() - begin) / 20 * 1e6
                            print(
                                f"meta {domain} r{rank} L{active} T={num_tokens}: "
                                f"sgmv per-batch synth {synth_us:7.1f}us  "
                                f"segments {arms.info_serving16.num_segments}"
                            )

                        thunks = arms.thunks()
                        cell = (domain, rank, active, num_tokens)
                        # Production decode replays graphs; production
                        # prefill is EAGER (seventh S3 review) — prefill
                        # cells record both modes, decided separately.
                        modes = (True,) if num_tokens <= DECODE_T_MAX else (True, False)
                        for repeat in range(REPEATS):
                            names = (
                                tuple(thunks)
                                if repeat % 2 == 0
                                else tuple(thunks)[::-1]
                            )
                            for name in names:
                                for graph in modes:
                                    record = measure(
                                        thunks[name],
                                        suite=suite,
                                        candidate=ARM_KEYS[name],
                                        boundary=BOUNDARY_ROUTE_INCLUSIVE,
                                        params={
                                            "case_id": case.case_id,
                                            "T": num_tokens,
                                            "rank": rank,
                                            "active_adapters": active,
                                            "expert_id_domain": domain,
                                            "seed": seed,
                                            "repeat": repeat,
                                            "config_a": dict(LORA_A),
                                            "config_b": dict(LORA_B),
                                            "a_metadata_scope": (
                                                ARM_METADATA_SCOPE[name]
                                            ),
                                            **arms.sgmv_params(name),
                                        },
                                        graph_replay=graph,
                                    )
                                    target = samples if graph else samples_eager
                                    target[cell][name].append(record.median_s)
                                    if graph:
                                        records[cell][name].append(record.record_id)

    challengers = tuple(name for name in ARM_KEYS if name != "control")
    for challenger in challengers:
        for domain in domains:
            for rank in ranks:
                for active in adapter_counts:
                    decisions = {}
                    for num_tokens in T_GRID:
                        cell = (domain, rank, active, num_tokens)
                        if not samples[cell][challenger]:
                            continue  # graph arm runs decode cells only
                        decision = decide_cell(
                            arm_a="control",
                            samples_a=samples[cell]["control"],
                            arm_b=challenger,
                            samples_b=samples[cell][challenger],
                        )
                        decisions[num_tokens] = decision
                        print(
                            f"{domain:8s} r{rank:<4d} L{active} T={num_tokens:<5d} "
                            f"{challenger:14s} geo(c/x)={decision.geo_a_over_b:.3f} "
                            f"-> {decision.winner or 'tied'}"
                        )
                    decided_ts = [t for t in T_GRID if t in decisions]
                    for t_low, t_high in zip(decided_ts, decided_ts[1:]):
                        low, high = decisions[t_low], decisions[t_high]
                        if (
                            low.winner is not None
                            and high.winner is not None
                            and low.winner != high.winner
                        ):
                            suite.site_crossover(
                                site="gate_up_a_shared",
                                boundary=BOUNDARY_ROUTE_INCLUSIVE,
                                candidates=(
                                    ARM_KEYS["control"],
                                    ARM_KEYS[challenger],
                                ),
                                axis=(
                                    f"num_tokens (serving-real boundary, "
                                    f"rank={rank}, L={active}, domain={domain})"
                                ),
                                crossover_location=f"T in ({t_low}, {t_high}]",
                                bracketing_low_record_ids=tuple(
                                    records[(domain, rank, active, t_low)]["control"]
                                    + records[(domain, rank, active, t_low)][challenger]
                                ),
                                bracketing_high_record_ids=tuple(
                                    records[(domain, rank, active, t_high)]["control"]
                                    + records[(domain, rank, active, t_high)][
                                        challenger
                                    ]
                                ),
                                cache_state=CACHE_L2_HOT_GRAPH,
                                axis_param="T",
                                # a_metadata_scope and the sgmv_* geometry
                                # keys are candidate-specific, not workload.
                                workload_params=(
                                    "rank",
                                    "active_adapters",
                                    "expert_id_domain",
                                    "config_a",
                                    "config_b",
                                ),
                                notes=(
                                    f"{low.winner} wins T={t_low} "
                                    f"(margin {low.margin():.3f}), {high.winner} "
                                    f"T={t_high} (margin {high.margin():.3f})"
                                ),
                            )

    for challenger in challengers:
        for cell in sorted(samples_eager):
            if not samples_eager[cell][challenger]:
                continue
            decision = decide_cell(
                arm_a="control",
                samples_a=samples_eager[cell]["control"],
                arm_b=challenger,
                samples_b=samples_eager[cell][challenger],
            )
            print(
                f"EAGER {cell[0]:8s} r{cell[1]:<4d} L{cell[2]} T={cell[3]:<5d} "
                f"{challenger:22s} geo(c/x)={decision.geo_a_over_b:.3f} "
                f"-> {decision.winner or 'tied'}"
            )

    digest = write_suite(suite, arguments.output)
    print(f"{len(suite.records)} records -> {arguments.output} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
