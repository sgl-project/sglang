"""Shared-outer DOWN-B v7: exhaustive own-geometry tuning, sgmv vs csgmv.

History: v1 (untuned challenger) ruled void; v2 tuned every family and
reversed the verdict (unchunked sgmv-accum wins EP-shaped cells 2.3-9.5x
under EXACT metadata; chunked expand spill-cliff root-caused, NCU pair in
ncu_sdb_probe.py); v3 added ranks 16-128, validity presets, the padded
arm; v4 reopened capacity-padded CSGMV (segment-count padding beats the
row-ceiling overlaunch at high sparsity). The third gate-4 review found
v4 still not authoritative:

* grouped baselines were imported from the main B table — tuned on
  PER-EXPERT dense routing, not this bench's shared-outer geometry;
* only 2 tuning regimes (T=1 ran T=64 configs; T=8192 ran T=2048's);
* config rejection was fail-open (generic signature match);
* padded sgmv/csgmv were never adjudicated head-to-head by decide_cell.

v5 fixed all four (what still runs today):

* four tuning regimes (decode_tiny T=4 / decode T=64 / prefill T=2048 /
  prefill_xl T=8192), matching the shared regime map;
* skips only on exact resource/compiler signatures, persisted to a
  sidecar; numeric failures abort (bench_common);
* every headline pair — including sgmv_accum_padded vs
  csgmv_accum_padded — is emitted through decide_cell by this producer;
* grid ceilings extended: sgmv BS<=512 & BN<=1024, csgmv BN<=512;
* [SUPERSEDED in v7] grouped stock/one_launch were locally re-tuned over
  a one-step neighborhood of the MAIN-B table winner.

v6 (4th review): the pruned tuner is DELETED — sweeps are exhaustive
(its predictor rejected 9,312 configs that compiled, including 32/192
cell winners, and left cells 2.17x off the measured optimum); T=1/T=8
added; skips carry a stage label. ([SUPERSEDED in the 5th review] its
row-SUBSAMPLE admission — replaced by the bounded-memory EXACT chunked
gate in bench_common.)

BOUNDARY HONESTY unchanged: grouped arms build their outer plan
IN-THUNK; sgmv/csgmv segments are host-synthesized (PREPARED — a bound
favoring them; losing here is a safe rejection, winning obligates a
charged device-side segment builder).

v7 (9th review), STRUCTURAL: this bench no longer consumes the main-B
config table at all. ``weight_ownership`` is a REQUIRED, non-retunable
transfer invariant; the main table is ``per_expert`` and this bench is
``shared_outer``, so seeding its grouped baselines from that table (even
through a one-step neighborhood) would carry a per-expert-biased winner
into a shared-outer conclusion. Both grouped arms are now swept
EXHAUSTIVELY on this bench's own geometry, exactly like its sgmv/csgmv
arms, and there is no table plumbing left to get wrong.

Usage::

    python3 -m benchmark.kernels.lora_moe.bench_shared_down_b \
        --output sdb7.json --source-revision <sha>
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
import triton

from benchmark.kernels.lora_moe.bench_common import (
    CSGMV_PADDED_CHUNKS,
    DECODE_T_MAX,
    config_key,
    exhaustive_grouped_lora_b_grid,
    exhaustive_sgmv_grid,
    regime_of,
    require_delta_close_chunked,
    require_writable_destination,
    skip_entry,
    skip_reason,
    write_skip_sidecar,
)
from benchmark.kernels.lora_moe.bench_lora_a import _LegFixture
from benchmark.kernels.lora_moe.bench_sgmv_real import (
    synthesize_chunked_batch_info,
    synthesize_unchunked_batch_info,
)
from benchmark.kernels.lora_moe.cases import AdapterCell, Topology, build_case
from benchmark.kernels.lora_moe.crossover_ledger import decide_cell
from benchmark.kernels.lora_moe.lora_a_candidates import run_lora_a
from benchmark.kernels.lora_moe.lora_a_execution import LoraAExecutionSpec
from benchmark.kernels.lora_moe.lora_b_candidates import run_lora_b
from benchmark.kernels.lora_moe.lora_b_execution import LoraBExecutionSpec
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_PREPARED_INPUT,
    BOUNDARY_ROUTE_INCLUSIVE,
    measure,
    new_suite,
    write_suite,
)
from sglang.kernels.ops.gemm.chunked_sgmv_expand import _chunked_lora_expand_kernel
from sglang.kernels.ops.gemm.sgemm_lora_b import _sgemm_lora_b_kernel
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import (
    ROUTE_ALIGNED,
    build_virtual_expert_routing,
)

T_GRID = (1, 4, 8, 16, 64, 256, 2048, 8192)  # 4th review: T=1/T=8 added
SWEEP_T = {"decode_tiny": 4, "decode": 64, "prefill": 2048, "prefill_xl": 8192}
SEEDS = (11, 137, 997)
REPEATS = 2
VALIDITY_PRESETS = {
    "dense": (8, "ep_local"),
    "ep8": (8, "global"),
    "ep4": (4, "global"),
    "ep2": (2, "global"),
}
GROUPED_ARMS = ("stock_charged", "one_launch_charged")
ARMS = GROUPED_ARMS + (
    "sgmv_memset",
    "sgmv_accum",
    "sgmv_accum_padded",
    "csgmv_accum_padded",
)
BOUNDARIES = {
    "stock_charged": BOUNDARY_ROUTE_INCLUSIVE,
    "one_launch_charged": BOUNDARY_ROUTE_INCLUSIVE,
    "sgmv_memset": BOUNDARY_PREPARED_INPUT,
    "sgmv_accum": BOUNDARY_PREPARED_INPUT,
    "sgmv_accum_padded": BOUNDARY_PREPARED_INPUT,
    "csgmv_accum_padded": BOUNDARY_PREPARED_INPUT,
}
SUITE_NAME = "shared_down_b_v7"
WEIGHT_OWNERSHIP = "shared_outer"
DECIDED_PAIRS = [("stock_charged", arm) for arm in ARMS[1:]]
DECIDED_PAIRS += [("one_launch_charged", arm) for arm in ARMS[2:]]
DECIDED_PAIRS += [
    ("sgmv_accum", "sgmv_accum_padded"),
    ("sgmv_accum_padded", "csgmv_accum_padded"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--validity", default="dense,ep8,ep4,ep2")
    return parser


def build_run_setup(arguments) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Resolve and validate the sweep axes before any CUDA initialization."""
    rank_values = tuple(value.strip() for value in arguments.ranks.split(","))
    preset_values = tuple(value.strip() for value in arguments.validity.split(","))
    if any(not value for value in rank_values):
        raise ValueError("ranks must not contain empty entries")
    if any(not value for value in preset_values):
        raise ValueError("validity must not contain empty entries")
    ranks = tuple(int(value) for value in rank_values)
    presets = preset_values
    if not ranks or any(rank <= 0 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("ranks must be a non-empty list of unique positive integers")
    if not presets or len(set(presets)) != len(presets):
        raise ValueError("validity must be a non-empty list of unique presets")
    unknown = set(presets) - set(VALIDITY_PRESETS)
    if unknown:
        raise ValueError(f"unknown validity presets: {sorted(unknown)}")
    return ranks, presets


def record_workload_identity(case) -> dict:
    """Resolved workload identity stamped on every shared-down record."""
    return {
        "model_preset": case.model_preset,
        "adapter_cell": (
            f"active{case.active_adapters}_cap{case.slot_capacity}_"
            f"base{int(case.include_base_rows)}"
        ),
        "route_generator": case.route_generator,
        "topology": (f"tp{case.tp_size}_ep{case.ep_size}_moedp{case.moe_dp_size}"),
        "ep_rank": case.ep_rank,
        "expert_id_domain": case.expert_id_domain,
        "evidence_scope": "single_gpu_local_shape_proxy",
        "weight_ownership": WEIGHT_OWNERSHIP,
    }


def _measurement_params(case, **params) -> dict:
    # Record-level identity is deliberate: one suite spans several simulated
    # EP topologies and expert-ID domains, so a single header value would lie.
    return {"workload": record_workload_identity(case), **params}


def _grouped_grid(rank: int, family: str):
    """EXHAUSTIVE grouped grid — this bench tunes its OWN baselines.

    9th review, structural: the main-B table declares
    ``weight_ownership="per_expert"``; this bench is ``shared_outer``,
    a REQUIRED and non-retunable transfer invariant. Seeding a
    shared-outer conclusion from a per-expert-biased winner (even with a
    one-step neighborhood around it) is exactly the bias the contract
    exists to prevent, so there is no table here at all.
    """
    if family not in ("stock", "one_launch"):
        raise ValueError(f"unknown grouped family {family!r}")
    yield from exhaustive_grouped_lora_b_grid(rank=rank, stock=family == "stock")


def _sgmv_grid(rank: int, n_columns: int):
    """EXHAUSTIVE (4th review: the pruned search lost the frontier)."""
    return exhaustive_sgmv_grid(rank=rank, n_columns=n_columns)


def _csgmv_grid(rank: int, n_columns: int):
    for chunk in CSGMV_PADDED_CHUNKS:
        yield from exhaustive_sgmv_grid(
            rank=rank,
            n_columns=n_columns,
            rows_axis="chunk",
            rows_values=(chunk,),
        )


def _sgmv_key(config: dict) -> str:
    prefix = f"c{config['chunk']}-" if "chunk" in config else ""
    tile = f"bs{config['BLOCK_S']}-" if "BLOCK_S" in config else ""
    return (
        f"{prefix}{tile}bn{config['BLOCK_N']}-bk{config['BLOCK_K']}"
        f"-w{config['num_warps']}-s{config['num_stages']}"
    )


class _DownBFixture:
    def __init__(self, case, device):
        self.leg = _LegFixture(case, device)
        self.case = case
        self.device = device
        per_expert = self._per_expert_plan()
        run_lora_a(
            LoraAExecutionSpec(site="down", ownership="grouped"),
            input=self.leg.act_pair,
            weight=self.leg.a_down,
            output=self.leg.down_rank_out,
            routing=per_expert,
            config=PROVISIONAL_LAUNCH_CONFIG.lora_a,
        )
        topk = self.leg.topk_ids
        valid = (topk >= 0) & (topk < case.num_experts_local)
        pair_slots = torch.where(
            valid,
            self.leg.token_slots[:, None].expand_as(topk),
            torch.full_like(topk, -1),
        ).reshape(-1)
        self.pair_valid = (pair_slots >= 0) & (pair_slots < case.slot_capacity)
        self.unchunked_info = synthesize_unchunked_batch_info(
            pair_slots,
            max_loras=case.slot_capacity,
            physical_rank=case.physical_rank,
            device=device,
        )
        num_pairs = self.leg.num_pairs
        self.padded_info = synthesize_unchunked_batch_info(
            pair_slots,
            max_loras=case.slot_capacity,
            physical_rank=case.physical_rank,
            device=device,
            capacity_segments=case.slot_capacity,
            max_len_ceiling=num_pairs,
        )
        self.chunked_padded_infos = {
            chunk: synthesize_chunked_batch_info(
                pair_slots,
                max_loras=case.slot_capacity,
                physical_rank=case.physical_rank,
                device=device,
                chunk=chunk,
                graph_capacity=((num_pairs + chunk - 1) // chunk + case.slot_capacity),
            )
            for chunk in CSGMV_PADDED_CHUNKS
        }
        self.slice_offsets = torch.tensor(
            [0, case.moe_hidden_size], dtype=torch.int32, device=device
        )
        self.b_down_3d = self.leg.b_down.view(
            case.slot_capacity, case.moe_hidden_size, case.physical_rank
        )
        self.down_base = torch.randn(
            num_pairs, case.moe_hidden_size, dtype=torch.bfloat16, device=device
        )
        self.accum_buffer = torch.empty_like(self.down_base)

    def _per_expert_plan(self):
        return build_virtual_expert_routing(
            self.leg.topk_ids,
            self.leg.token_slots,
            lora_experts_per_adapter=self.case.num_experts_local,
            max_loras=self.case.slot_capacity,
            block_size=self.case.routing_block_size,
            view=ROUTE_ALIGNED,
        )

    def outer_plan(self):
        return build_virtual_expert_routing(
            self.leg.topk_ids,
            self.leg.token_slots,
            lora_experts_per_adapter=1,
            max_loras=self.case.slot_capacity,
            block_size=self.case.routing_block_size,
            shared_outer_local_expert_count=self.case.num_experts_local,
            view=ROUTE_ALIGNED,
        )

    def grouped_arm(self, arm: str, config: dict):
        spec = LoraBExecutionSpec(
            site="down",
            ownership="grouped",
            slicing=("per_slice" if arm == "stock_charged" else "one_launch_sliced"),
        )
        run_lora_b(
            spec,
            bridge=self.leg.down_rank_out,
            weight=self.b_down_3d,
            destination=self.leg.down_delta,
            routing=self.outer_plan(),  # per-layer under EP: charged
            destination_offsets=(0,),
            config=config,
        )

    def sgmv(self, info, config: dict, output: torch.Tensor):
        bridge = self.leg.down_rank_out
        weights = self.b_down_3d
        hidden = self.case.moe_hidden_size
        grid = (
            triton.cdiv(info.max_len, config["BLOCK_S"])
            * triton.cdiv(hidden, config["BLOCK_N"]),
            info.bs,
        )
        _sgemm_lora_b_kernel[grid](
            bridge,
            weights,
            output,
            hidden,
            self.case.physical_rank,
            bridge.stride(0),
            bridge.stride(1),
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            output.stride(0),
            output.stride(1),
            info.seg_lens,
            info.seg_indptr,
            info.weight_indices,
            info.lora_ranks,
            info.permutation,
            True,  # SORTED_BY_ADAPTER
            config["BLOCK_S"],
            config["BLOCK_N"],
            config["BLOCK_K"],
            info.scalings,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )

    def csgmv(self, config: dict, output: torch.Tensor):
        info = self.chunked_padded_infos[config["chunk"]]
        hidden = self.case.moe_hidden_size
        segment_grid = info.weight_indices.shape[0]  # padded capacity
        grid = (triton.cdiv(hidden, config["BLOCK_N"]), 1, segment_grid)
        _chunked_lora_expand_kernel[grid](
            x=self.leg.down_rank_out,
            weights=self.b_down_3d,
            output=output,
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            seg_indptr=info.seg_indptr,
            weight_indices=info.weight_indices,
            lora_ranks=info.lora_ranks,
            permutation=info.permutation,
            num_segs=segment_grid,
            scalings=info.scalings,
            slice_offsets=self.slice_offsets,
            NUM_SLICES=1,
            OUTPUT_DIM=hidden,
            MAX_RANK=self.case.physical_rank,
            BLOCK_M=info.max_len,  # == chunk (synthesis contract)
            BLOCK_N=config["BLOCK_N"],
            BLOCK_K=config["BLOCK_K"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )

    def run(self, arm: str, tuned: dict):
        config = tuned[arm]
        if arm in GROUPED_ARMS:
            self.grouped_arm(arm, config)
        elif arm == "sgmv_memset":
            self.leg.down_delta.zero_()
            self.sgmv(self.unchunked_info, config, self.leg.down_delta)
        elif arm == "sgmv_accum":
            # Replays re-accumulate garbage VALUES but identical WORK;
            # admission validates values with a fresh base copy.
            self.sgmv(self.unchunked_info, config, self.accum_buffer)
        elif arm == "sgmv_accum_padded":
            self.sgmv(self.padded_info, config, self.accum_buffer)
        else:
            self.csgmv(config, self.accum_buffer)

    def reference_delta(self, label: str) -> torch.Tensor:
        """Oracle from the TRUSTED DEFAULT config (5th review).

        Building the reference with a locally SELECTED config made that
        config its own oracle: a row-local error the gate missed would
        have biased every comparison in the cell. PROVISIONAL_LAUNCH_CONFIG
        is the production default, pinned against the FP32 torch oracle by
        the registered test suite, and is independent of both the table and
        this bench's tuning.
        """
        self.leg.down_delta.fill_(71.0)
        self.grouped_arm("stock_charged", dict(PROVISIONAL_LAUNCH_CONFIG.lora_b))
        reference = self.leg.down_delta.clone()
        if not bool((reference[~self.pair_valid] == 0).all()):
            raise AssertionError(f"stock zero-fill broken {label}")
        return reference


def gate_promoted_arms(fixture, tuned: dict, label: str):
    """The PRODUCTION sequence: build the trusted oracle, then full-gate
    EVERY promoted arm, stock included.

    8th review: tests previously inspected ``reference_delta``'s signature
    and drove the arm loop separately, so the two could drift apart while
    both tests stayed green. Production and the registered test call
    THIS, so the composed sequence itself is covered (7th review: the
    stale-signature + ARMS[1:] regression had no durable guard).
    """
    reference = fixture.reference_delta(label)
    admitted = []
    for arm in ARMS:
        _admit(fixture, arm, tuned, reference, label)
        admitted.append(arm)
    return tuple(admitted)


def _admit(fixture: _DownBFixture, arm: str, tuned: dict, reference, label: str):
    if arm.endswith("_padded") or arm == "sgmv_accum":
        fixture.accum_buffer.copy_(fixture.down_base)
        fixture.run(arm, tuned)
        require_delta_close_chunked(
            fixture.accum_buffer,
            reference,
            observed_base=fixture.down_base,
            gate_dtype=torch.bfloat16,
            label=f"{arm} vs base+delta {label}",
        )
    else:
        fixture.leg.down_delta.fill_(17.0)
        fixture.run(arm, tuned)
        require_delta_close_chunked(
            fixture.leg.down_delta,
            reference,
            gate_dtype=torch.bfloat16,
            label=f"{arm} vs trusted-default reference {label}",
        )


def _build_case(device, *, preset, num_tokens, rank, seed, source_revision):
    ep_size, domain = VALIDITY_PRESETS[preset]
    return build_case(
        device=str(device),
        model_preset="qwen35_35b",
        topology=Topology(tp_size=8, ep_size=ep_size),
        adapter_cell=AdapterCell(
            active_adapters=4, include_base_rows=True, slot_capacity=8
        ),
        route_generator="iid",
        expert_id_domain=domain,
        num_tokens=num_tokens,
        active_rank=rank,
        shared_factor_signature="shared_down_b",
        seed=seed,
        source_revision=source_revision,
    )


def main() -> int:
    # There is deliberately no --config-table: shared-outer owns its baselines.
    arguments = _build_parser().parse_args()
    ranks, presets = build_run_setup(arguments)
    device = torch.device(arguments.device)
    require_writable_destination(arguments.output)
    torch.cuda.set_device(device)
    suite = new_suite(
        SUITE_NAME,
        source_revision=arguments.source_revision,
        producer_files=(__file__,),
    )
    skips: list[dict] = []

    # ---- SWEEP per (preset, rank, regime): exhaustive grouped grids on the
    # ACTUAL shared-outer geometry + sgmv exact/padded + csgmv padded.
    best: dict = {}
    for preset in presets:
        for rank in ranks:
            for regime, num_tokens in SWEEP_T.items():
                case = _build_case(
                    device,
                    preset=preset,
                    num_tokens=num_tokens,
                    rank=rank,
                    seed=SEEDS[0],
                    source_revision=suite.source_revision,
                )
                fixture = _DownBFixture(case, device)
                hidden = case.moe_hidden_size
                label = f"{preset} r{rank} {regime}(T={num_tokens})"
                reference = fixture.reference_delta(label)

                def timed(fn, family, config_label):
                    return measure(
                        fn,
                        suite=suite,
                        candidate=f"sweep_{family}",
                        boundary=(
                            BOUNDARY_ROUTE_INCLUSIVE
                            if family in ("stock", "one_launch")
                            else BOUNDARY_PREPARED_INPUT
                        ),
                        params=_measurement_params(
                            case,
                            case_id=case.case_id,
                            phase="sweep",
                            T=num_tokens,
                            rank=rank,
                            validity=preset,
                            regime=regime,
                            family=family,
                            config=config_label,
                        ),
                        graph_replay=True,
                        warmup_iters=10,
                        replay_iters=100,
                    ).median_s

                # grouped arms: EXHAUSTIVE on THIS bench's own geometry
                # (9th review — no per-expert table seed, no neighborhood).
                for family, arm in (
                    ("stock", "stock_charged"),
                    ("one_launch", "one_launch_charged"),
                ):
                    best_cfg, best_med, skipped_here = None, float("inf"), 0
                    for config in _grouped_grid(rank, family):
                        try:
                            fixture.leg.down_delta.fill_(-3.0)
                            fixture.grouped_arm(arm, config)
                            torch.cuda.synchronize()
                            require_delta_close_chunked(
                                fixture.leg.down_delta,
                                reference,
                                gate_dtype=torch.bfloat16,
                                label=f"grouped {family} {label}",
                            )
                        except Exception as error:
                            reason = skip_reason(error)
                            if reason is None:
                                raise
                            skipped_here += 1
                            skips.append(
                                skip_entry(
                                    reason,
                                    family=family,
                                    cell=label,
                                    config=config_key(config),
                                )
                            )
                            continue
                        median = timed(
                            lambda a=arm, c=config: fixture.grouped_arm(a, c),
                            family,
                            config_key(config),
                        )
                        if median < best_med:
                            best_cfg, best_med = dict(config), median
                    if best_cfg is None:
                        raise RuntimeError(f"no admissible {family} config {label}")
                    best[(preset, rank, regime, family)] = best_cfg
                    print(
                        f"SWEEP {label} [{family}]: {config_key(best_cfg)} "
                        f"({best_med * 1e6:.1f}us, {skipped_here} skipped)",
                        flush=True,
                    )

                # sgmv/csgmv families
                for metadata, grid_fn, runner in (
                    (
                        "exact",
                        _sgmv_grid,
                        lambda c: fixture.sgmv(
                            fixture.unchunked_info, c, fixture.accum_buffer
                        ),
                    ),
                    (
                        "padded",
                        _sgmv_grid,
                        lambda c: fixture.sgmv(
                            fixture.padded_info, c, fixture.accum_buffer
                        ),
                    ),
                    (
                        "csgmv_padded",
                        _csgmv_grid,
                        lambda c: fixture.csgmv(c, fixture.accum_buffer),
                    ),
                ):
                    best_cfg, best_med = None, float("inf")
                    skipped_here = 0
                    for config in grid_fn(rank, hidden):
                        try:
                            fixture.accum_buffer.copy_(fixture.down_base)
                            runner(config)
                            torch.cuda.synchronize()
                            # 5th review: EXACT full-domain gate, bounded
                            # memory; base added per chunk.
                            require_delta_close_chunked(
                                fixture.accum_buffer,
                                reference,
                                observed_base=fixture.down_base,
                                gate_dtype=torch.bfloat16,
                                label=f"sweep sgmv/{metadata} {label}",
                            )
                        except Exception as error:
                            reason = skip_reason(error)
                            if reason is None:
                                raise
                            skipped_here += 1
                            skips.append(
                                skip_entry(
                                    reason,
                                    family=metadata,
                                    cell=label,
                                    config=_sgmv_key(config),
                                )
                            )
                            continue
                        median = timed(
                            lambda c=config, r=runner: r(c),
                            metadata,
                            _sgmv_key(config),
                        )
                        if median < best_med:
                            best_cfg, best_med = dict(config), median
                    if best_cfg is None:
                        raise RuntimeError(
                            f"no admissible sgmv/{metadata} config {label}"
                        )
                    best[(preset, rank, regime, metadata)] = best_cfg
                    print(
                        f"SWEEP {label} [{metadata}]: {_sgmv_key(best_cfg)} "
                        f"({best_med * 1e6:.1f}us, {skipped_here} skipped)",
                        flush=True,
                    )

    # ---- DECIDED: seeded interleaved comparison, everything tuned on
    # this geometry at this regime.
    samples: dict = defaultdict(lambda: defaultdict(list))
    for preset in presets:
        for rank in ranks:
            for num_tokens in T_GRID:
                regime = regime_of(num_tokens)
                tuned = {
                    "stock_charged": best[(preset, rank, regime, "stock")],
                    "one_launch_charged": best[(preset, rank, regime, "one_launch")],
                    "sgmv_memset": best[(preset, rank, regime, "exact")],
                    "sgmv_accum": best[(preset, rank, regime, "exact")],
                    "sgmv_accum_padded": best[(preset, rank, regime, "padded")],
                    "csgmv_accum_padded": best[(preset, rank, regime, "csgmv_padded")],
                }
                modes = (True,) if num_tokens <= DECODE_T_MAX else (True, False)
                for seed in SEEDS:
                    case = _build_case(
                        device,
                        preset=preset,
                        num_tokens=num_tokens,
                        rank=rank,
                        seed=seed,
                        source_revision=suite.source_revision,
                    )
                    fixture = _DownBFixture(case, device)
                    label = f"{preset} r{rank} T={num_tokens} s{seed}"
                    # EVERY promoted arm is full-gated, stock included
                    # (5th review): a selected config is never its own oracle.
                    gate_promoted_arms(fixture, tuned, label)
                    cell = (preset, rank, num_tokens)
                    for graph in modes:
                        mode = "graph" if graph else "eager"
                        for repeat in range(REPEATS):
                            names = ARMS if repeat % 2 == 0 else tuple(reversed(ARMS))
                            for arm in names:
                                record = measure(
                                    lambda a=arm: fixture.run(a, tuned),
                                    suite=suite,
                                    candidate=f"shared_down_b_{arm}",
                                    boundary=BOUNDARIES[arm],
                                    params=_measurement_params(
                                        case,
                                        case_id=case.case_id,
                                        phase="decided",
                                        T=num_tokens,
                                        rank=rank,
                                        validity=preset,
                                        seed=seed,
                                        repeat=repeat,
                                        regime=regime,
                                        arm_config=(
                                            _sgmv_key(tuned[arm])
                                            if "sgmv" in arm
                                            else config_key(tuned[arm])
                                        ),
                                    ),
                                    graph_replay=graph,
                                )
                                samples[(*cell, mode)][arm].append(record.median_s)

    for cell in sorted(samples):
        for arm_a, arm_b in DECIDED_PAIRS:
            decision = decide_cell(
                arm_a=arm_a,
                samples_a=samples[cell][arm_a],
                arm_b=arm_b,
                samples_b=samples[cell][arm_b],
                boundary_a=BOUNDARIES[arm_a],
                boundary_b=BOUNDARIES[arm_b],
            )
            print(
                f"{cell[3]:5s} {cell[0]:6s} r{cell[1]:<4d} T={cell[2]:<5d} "
                f"{arm_a}/{arm_b:20s} geo(a/b)={decision.geo_a_over_b:.3f} -> "
                f"{decision.winner or 'tied'} [{decision.scope}]"
            )

    # 13th review: secondary skip ledgers were mutable and unbound; they
    # are now immutable content-addressed files whose digest is printed
    # into the adjudicated log stream.
    write_skip_sidecar(arguments.output, skips, content_addressed=True)
    digest = write_suite(suite, arguments.output)
    print(f"{len(suite.records)} records -> {arguments.output} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
