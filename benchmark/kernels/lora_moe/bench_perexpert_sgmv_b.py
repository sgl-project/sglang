"""Per-expert-B SGMV v3: same-geometry tuning, 4 regimes, +padded csgmv.

v1 measured the candidate the campaign had wrongly called "structurally
impossible" (virtual-expert segments through the unchunked SGMV kernel)
and found it wins EP-sparse routing, loses dense. The third gate-4
review found v2 still not authoritative:

* grouped baselines imported from the main B table were tuned on DENSE
  ep_local routing — the global-domain cells compared against
  wrong-geometry baselines;
* two tuning regimes only (T=1 ran T=64 configs; T=8192 ran T=2048's);
* padded coverage lacked CSGMV even though its segment-count padding
  advantage applies identically here;
* config rejection was fail-open; padded pairs were adjudicated by
  post-hoc scripts, not decide_cell.

v3 fixes all of it: in-bench grouped neighborhood re-tuning per
(domain, rank, regime) on this bench's own routing geometry; four
tuning regimes; csgmv_pe_padded arm (chunked kernel over virtual-expert
segments, NUM_SLICES=2 at gate/up via slice_offsets, segment-capacity
padding); exact-signature skips persisted to a sidecar; every headline
pair emitted through decide_cell; sgmv ceilings extended (BS<=512,
BN<=1024 width-filtered, BK<=128).

All sgmv/csgmv arms run the MEMSET form (per-expert deltas are consumed
by the activation join / combine — there is no base to accumulate
into), at the PREPARED boundary (host-built segments favor them; a loss
here is a safe rejection).

Usage::

    python3 -m benchmark.kernels.lora_moe.bench_perexpert_sgmv_b \
        --output pe_sgmv3.json --source-revision <sha>

12th review: this bench consumes NO config table. Its grouped baselines
are swept exhaustively on its own geometry (the shared grid), exactly
like its sgmv/csgmv challengers — a one-step neighborhood around the
main table's dense/ep_local winner gave the challenger a full grid while
the baseline got 13 points. Grouped-vs-sgmv decisions are emitted with
scope=ceiling (boundaries differ); the sgmv-family internal pairs are
charged same-boundary comparisons.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
import triton

from benchmark.kernels.lora_moe import bench_shared_down_b as _sdb_module
from benchmark.kernels.lora_moe.bench_common import (
    CSGMV_PADDED_CHUNKS,
    DECODE_T_MAX,
    config_key,
    exhaustive_grouped_lora_b_grid,
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
from benchmark.kernels.lora_moe.bench_shared_down_b import (
    _csgmv_grid,
    _sgmv_grid,
    _sgmv_key,
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
from sglang.srt.lora.sgl_lora.bf16 import stock_grouped_lora_b
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import ROUTE_ALIGNED

T_GRID = (1, 4, 8, 16, 64, 256, 2048, 8192)  # 4th review: T=8 was missing
SWEEP_T = {"decode_tiny": 4, "decode": 64, "prefill": 2048, "prefill_xl": 8192}
SEEDS = (11, 137, 997)
REPEATS = 2
GROUPED_ARMS = ("stock", "one_launch")
ARMS = GROUPED_ARMS + ("sgmv_pe", "sgmv_pe_padded", "csgmv_pe_padded")
# Grouped arms build their route plan IN-THUNK; sgmv arms consume
# host-synthesized segments. decide_cell records the mismatch as
# scope=ceiling (12th review).
BOUNDARIES = {
    "stock": BOUNDARY_ROUTE_INCLUSIVE,
    "one_launch": BOUNDARY_ROUTE_INCLUSIVE,
    "sgmv_pe": BOUNDARY_PREPARED_INPUT,
    "sgmv_pe_padded": BOUNDARY_PREPARED_INPUT,
    "csgmv_pe_padded": BOUNDARY_PREPARED_INPUT,
}
DECIDED_PAIRS = [("stock", arm) for arm in ARMS[1:]]
DECIDED_PAIRS += [("one_launch", arm) for arm in ARMS[2:]]
DECIDED_PAIRS += [
    ("sgmv_pe", "sgmv_pe_padded"),
    ("sgmv_pe_padded", "csgmv_pe_padded"),
]


def build_run_setup(arguments) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Parse and validate the run axes before any CUDA state is initialized."""
    rank_fields = tuple(value.strip() for value in arguments.ranks.split(","))
    if not rank_fields or any(not value for value in rank_fields):
        raise ValueError("--ranks must be a non-empty comma-separated list")
    try:
        ranks = tuple(int(value) for value in rank_fields)
    except ValueError as error:
        raise ValueError("--ranks values must be integers") from error
    if any(rank <= 0 for rank in ranks):
        raise ValueError("--ranks values must be positive")
    if len(set(ranks)) != len(ranks):
        raise ValueError("--ranks values must be unique")

    domains = tuple(value.strip() for value in arguments.domains.split(","))
    if not domains or any(not value for value in domains):
        raise ValueError("--domains must be a non-empty comma-separated list")
    unknown_domains = sorted(set(domains) - {"ep_local", "global"})
    if unknown_domains:
        raise ValueError(f"unknown expert-id domains: {unknown_domains}")
    if len(set(domains)) != len(domains):
        raise ValueError("--domains values must be unique")
    return ranks, domains


def record_workload_identity(case) -> dict:
    """Resolved workload identity stamped on every per-expert record.

    10th review: shared-down stamps this on every record while this suite
    recorded none, so a pe artifact could not be audited for the workload
    it actually measured. One suite spans both expert-ID domains, so the
    identity belongs on the RECORD, not the header.
    """
    return {
        "model_preset": case.model_preset,
        "adapter_cell": (
            f"active{case.active_adapters}_cap{case.slot_capacity}_"
            f"base{int(case.include_base_rows)}"
        ),
        "route_generator": case.route_generator,
        "topology": f"tp{case.tp_size}_ep{case.ep_size}_moedp{case.moe_dp_size}",
        "ep_rank": case.ep_rank,
        "expert_id_domain": case.expert_id_domain,
        "evidence_scope": "single_gpu_local_shape_proxy",
        "weight_ownership": "per_expert",
    }


class _PerExpertFixture:
    """B fixture plus virtual-expert pair segments for the sgmv arms."""

    def __init__(self, case, device):
        self.leg = _LegFixture(case, device)
        self.case = case
        plan = self._plan()
        for site in ("gate_up", "down"):
            inp, weight, out = self.leg.site_buffers(site)
            run_lora_a(
                LoraAExecutionSpec(site=site, ownership="grouped"),
                input=inp,
                weight=weight,
                output=out,
                routing=plan,
                config=PROVISIONAL_LAUNCH_CONFIG.lora_a,
            )
        # Virtual-expert pair segments: weight_indices carry
        # adapter*E + expert, indexing the flattened [L*E, N, R] weights.
        # Bypasses lora_expert_map — identity only at ep_rank 0.
        if case.ep_rank != 0:
            raise AssertionError(
                "veid segmentation assumes the identity global->local "
                "expert map (ep_rank 0)"
            )
        experts = case.num_experts_local
        ids = self.leg.topk_ids.to(torch.int64).reshape(-1)
        slots = (
            self.leg.token_slots.to(torch.int64)[:, None]
            .expand_as(self.leg.topk_ids)
            .reshape(-1)
        )
        valid = (
            (ids >= 0) & (ids < experts) & (slots >= 0) & (slots < case.slot_capacity)
        )
        veids = torch.where(valid, slots * experts + ids, torch.full_like(ids, -1))
        self.pair_valid = valid
        num_groups = case.slot_capacity * experts
        num_pairs = self.leg.num_pairs
        self.info = synthesize_unchunked_batch_info(
            veids.to(torch.int32),
            max_loras=num_groups,
            physical_rank=case.physical_rank,
            device=device,
        )
        self.padded_info = synthesize_unchunked_batch_info(
            veids.to(torch.int32),
            max_loras=num_groups,
            physical_rank=case.physical_rank,
            device=device,
            capacity_segments=num_groups,
            max_len_ceiling=num_pairs,
        )
        # Chunked (csgmv) padded metadata: capacity = worst-case chunk
        # count plus one boundary segment per virtual expert.
        self.chunked_padded_infos = {
            chunk: synthesize_chunked_batch_info(
                veids.to(torch.int32),
                max_loras=num_groups,
                physical_rank=case.physical_rank,
                device=device,
                chunk=chunk,
                graph_capacity=(num_pairs + chunk - 1) // chunk + num_groups,
            )
            for chunk in CSGMV_PADDED_CHUNKS
        }
        self.slice_offsets = {}
        for site in ("gate_up", "down"):
            _, weight, _, offsets = self.b_args(site)
            width = weight.shape[1] // len(offsets)
            bounds = [i * width for i in range(len(offsets) + 1)]
            self.slice_offsets[site] = torch.tensor(
                bounds, dtype=torch.int32, device=device
            )

    def _plan(self):
        return self.leg.route(ROUTE_ALIGNED)

    def b_args(self, site: str):
        if site == "gate_up":
            return (
                self.leg.gate_rank_out,
                self.leg.b_gate_up,
                self.leg.gate_up_delta,
                (0, self.leg.intermediate),
            )
        return (self.leg.down_rank_out, self.leg.b_down, self.leg.down_delta, (0,))

    def grouped_arm(self, site: str, family: str, config: dict):
        bridge, weight, destination, offsets = self.b_args(site)
        slicing = "per_slice" if family == "stock" else "one_launch_sliced"
        run_lora_b(
            LoraBExecutionSpec(site=site, ownership="grouped", slicing=slicing),
            bridge=bridge,
            weight=weight,
            destination=destination,
            routing=self._plan(),  # charged: per-layer under EP
            destination_offsets=offsets,
            config=config,
        )

    def _sgmv_launch_slice(self, bridge, weight, destination, config, info):
        rank = self.case.physical_rank
        n_columns = weight.shape[1]
        grid = (
            triton.cdiv(info.max_len, config["BLOCK_S"])
            * triton.cdiv(n_columns, config["BLOCK_N"]),
            info.bs,
        )
        _sgemm_lora_b_kernel[grid](
            bridge,
            weight,
            destination,
            n_columns,
            rank,
            bridge.stride(0),
            bridge.stride(1),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            destination.stride(0),
            destination.stride(1),
            info.seg_lens,
            info.seg_indptr,
            info.weight_indices,
            info.lora_ranks,
            info.permutation,
            True,  # sorted by virtual expert
            config["BLOCK_S"],
            config["BLOCK_N"],
            config["BLOCK_K"],
            info.scalings,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )

    def sgmv_arm(self, site: str, config: dict, info=None):
        """Memset + one unchunked launch per slice over slice VIEWS."""
        info = self.info if info is None else info
        bridge, weight, destination, offsets = self.b_args(site)
        rank = self.case.physical_rank
        width = weight.shape[1] // len(offsets)
        destination.zero_()
        if info.bs == 0:  # no valid pairs: zero-fill IS the answer
            return
        for slice_id, offset in enumerate(offsets):
            self._sgmv_launch_slice(
                bridge[:, slice_id * rank : (slice_id + 1) * rank],
                weight[:, slice_id * width : (slice_id + 1) * width, :],
                destination[:, offset : offset + width],
                config,
                info,
            )

    def csgmv_arm(self, site: str, config: dict):
        """Memset + ONE chunked launch covering every slice natively."""
        info = self.chunked_padded_infos[config["chunk"]]
        bridge, weight, destination, offsets = self.b_args(site)
        num_slices = len(offsets)
        width = weight.shape[1] // num_slices
        destination.zero_()
        segment_grid = info.weight_indices.shape[0]  # padded capacity
        grid = (
            triton.cdiv(width, config["BLOCK_N"]),
            num_slices,
            segment_grid,
        )
        _chunked_lora_expand_kernel[grid](
            x=bridge,
            weights=weight,
            output=destination,
            output_stride_0=destination.stride(0),
            output_stride_1=destination.stride(1),
            seg_indptr=info.seg_indptr,
            weight_indices=info.weight_indices,
            lora_ranks=info.lora_ranks,
            permutation=info.permutation,
            num_segs=segment_grid,
            scalings=info.scalings,
            slice_offsets=self.slice_offsets[site],
            NUM_SLICES=num_slices,
            OUTPUT_DIM=weight.shape[1],
            MAX_RANK=self.case.physical_rank,
            BLOCK_M=info.max_len,  # == chunk (synthesis contract)
            BLOCK_N=config["BLOCK_N"],
            BLOCK_K=config["BLOCK_K"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )

    def run(self, site: str, arm: str, tuned: dict):
        if arm in GROUPED_ARMS:
            self.grouped_arm(site, arm, tuned[arm])
        elif arm == "sgmv_pe":
            self.sgmv_arm(site, tuned[arm])
        elif arm == "sgmv_pe_padded":
            self.sgmv_arm(site, tuned[arm], info=self.padded_info)
        else:
            self.csgmv_arm(site, tuned[arm])

    def reference(self, site: str, label: str) -> torch.Tensor:
        bridge, weight, destination, offsets = self.b_args(site)
        destination.fill_(71.0)
        stock_grouped_lora_b(
            bridge,
            weight,
            destination,
            self._plan(),
            destination_offsets=offsets,
            config=PROVISIONAL_LAUNCH_CONFIG.lora_b,
        )
        reference = destination.clone()
        if not bool((reference[~self.pair_valid] == 0).all()):
            raise AssertionError(f"stock zero-fill broken {label}")
        return reference


def _build_case(device, *, domain, num_tokens, rank, seed, source_revision):
    return build_case(
        device=str(device),
        model_preset="qwen35_35b",
        topology=Topology(tp_size=8, ep_size=8),
        adapter_cell=AdapterCell(
            active_adapters=4, include_base_rows=True, slot_capacity=8
        ),
        route_generator="iid",
        expert_id_domain=domain,
        num_tokens=num_tokens,
        active_rank=rank,
        seed=seed,
        source_revision=source_revision,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--domains", default="ep_local,global")
    arguments = parser.parse_args()
    ranks, domains = build_run_setup(arguments)
    device = torch.device(arguments.device)
    require_writable_destination(arguments.output)
    torch.cuda.set_device(device)
    suite = new_suite(
        "perexpert_sgmv_b_v3",
        source_revision=arguments.source_revision,
        producer_files=(
            __file__,
            # 8th review: grid/config/regime helpers are IMPORTED from
            # this sibling, so a mid-run edit there changes what this
            # producer does and must invalidate publication too.
            _sdb_module.__file__,
        ),
    )
    skips: list[dict] = []

    # ---- SWEEP per (domain, rank, regime, site).
    best: dict = {}
    for domain in domains:
        for rank in ranks:
            for regime, num_tokens in SWEEP_T.items():
                case = _build_case(
                    device,
                    domain=domain,
                    num_tokens=num_tokens,
                    rank=rank,
                    seed=SEEDS[0],
                    source_revision=suite.source_revision,
                )
                fixture = _PerExpertFixture(case, device)
                for site in ("gate_up", "down"):
                    _, weight, destination, offsets = fixture.b_args(site)
                    width = weight.shape[1] // len(offsets)
                    label = f"{domain} {site} r{rank} {regime}"
                    reference = fixture.reference(site, label)

                    def timed(fn, family, config_label):
                        return measure(
                            fn,
                            suite=suite,
                            candidate=f"sweep_{family}",
                            boundary=(
                                BOUNDARY_ROUTE_INCLUSIVE
                                if family in GROUPED_ARMS
                                else BOUNDARY_PREPARED_INPUT
                            ),
                            params={
                                "workload": record_workload_identity(case),
                                "case_id": case.case_id,
                                "phase": "sweep",
                                "T": num_tokens,
                                "rank": rank,
                                "site": site,
                                "expert_id_domain": domain,
                                "regime": regime,
                                "family": family,
                                "config": config_label,
                            },
                            graph_replay=True,
                            warmup_iters=10,
                            replay_iters=100,
                        ).median_s

                    # 12th review: grouped baselines are swept EXHAUSTIVELY
                    # on THIS bench's geometry, exactly like the sgmv arms.
                    # The previous one-step neighborhood around the main
                    # table's dense/ep_local winner gave the challenger a
                    # full grid while the baseline got 13 points — an
                    # asymmetry that inflates sparse/global sgmv wins.
                    for family in GROUPED_ARMS:
                        best_cfg, best_med = None, float("inf")
                        for config in exhaustive_grouped_lora_b_grid(
                            rank=rank, stock=family == "stock"
                        ):
                            try:
                                destination.fill_(-3.0)
                                fixture.grouped_arm(site, family, config)
                                torch.cuda.synchronize()
                                require_delta_close_chunked(
                                    destination,
                                    reference,
                                    gate_dtype=torch.bfloat16,
                                    label=f"nbhd {family} {label}",
                                )
                            except Exception as error:
                                reason = skip_reason(error)
                                if reason is None:
                                    raise
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
                                lambda s=site, f=family, c=config: (
                                    fixture.grouped_arm(s, f, c)
                                ),
                                family,
                                config_key(config),
                            )
                            if median < best_med:
                                best_cfg, best_med = dict(config), median
                        if best_cfg is None:
                            raise RuntimeError(f"no admissible {family} config {label}")
                        best[(domain, site, rank, regime, family)] = best_cfg
                        print(
                            f"SWEEP {label} [{family}]: "
                            f"{config_key(best_cfg)} ({best_med * 1e6:.1f}us)",
                            flush=True,
                        )

                    for metadata, grid_fn, runner in (
                        (
                            "sgmv_pe",
                            _sgmv_grid,
                            lambda s, c: fixture.sgmv_arm(s, c),
                        ),
                        (
                            "sgmv_pe_padded",
                            _sgmv_grid,
                            lambda s, c: fixture.sgmv_arm(
                                s, c, info=fixture.padded_info
                            ),
                        ),
                        (
                            "csgmv_pe_padded",
                            _csgmv_grid,
                            lambda s, c: fixture.csgmv_arm(s, c),
                        ),
                    ):
                        best_cfg, best_med = None, float("inf")
                        skipped_here = 0
                        for config in grid_fn(rank, width):
                            try:
                                destination.fill_(17.0)
                                runner(site, config)
                                torch.cuda.synchronize()
                                require_delta_close_chunked(
                                    destination,
                                    reference,
                                    gate_dtype=torch.bfloat16,
                                    label=f"sweep {metadata} {label}",
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
                                lambda s=site, c=config, r=runner: r(s, c),
                                metadata,
                                _sgmv_key(config),
                            )
                            if median < best_med:
                                best_cfg, best_med = dict(config), median
                        if best_cfg is None:
                            raise RuntimeError(
                                f"no admissible {metadata} config {label}"
                            )
                        best[(domain, site, rank, regime, metadata)] = best_cfg
                        print(
                            f"SWEEP {label} [{metadata}]: "
                            f"{_sgmv_key(best_cfg)} "
                            f"({best_med * 1e6:.1f}us, {skipped_here} skipped)",
                            flush=True,
                        )

    # ---- DECIDED.
    samples: dict = defaultdict(lambda: defaultdict(list))
    for domain in domains:
        for rank in ranks:
            for num_tokens in T_GRID:
                regime = regime_of(num_tokens)
                modes = (True,) if num_tokens <= DECODE_T_MAX else (True, False)
                for seed in SEEDS:
                    case = _build_case(
                        device,
                        domain=domain,
                        num_tokens=num_tokens,
                        rank=rank,
                        seed=seed,
                        source_revision=suite.source_revision,
                    )
                    fixture = _PerExpertFixture(case, device)
                    for site in ("gate_up", "down"):
                        tuned = {
                            arm: best[(domain, site, rank, regime, arm)] for arm in ARMS
                        }
                        label = f"{domain} {site} r{rank} T={num_tokens} s{seed}"
                        reference = fixture.reference(site, label)
                        _, _, destination, _ = fixture.b_args(site)
                        # 5th review: stock included — every promoted arm
                        # is full-gated against the trusted-default oracle.
                        for arm in ARMS:
                            destination.fill_(17.0)
                            fixture.run(site, arm, tuned)
                            require_delta_close_chunked(
                                destination,
                                reference,
                                gate_dtype=torch.bfloat16,
                                label=f"{arm} vs trusted-default {label}",
                            )
                        cell = (domain, site, rank, num_tokens)
                        for graph in modes:
                            mode = "graph" if graph else "eager"
                            for repeat in range(REPEATS):
                                names = (
                                    ARMS if repeat % 2 == 0 else tuple(reversed(ARMS))
                                )
                                for arm in names:
                                    record = measure(
                                        lambda s=site, a=arm, t=tuned: (
                                            fixture.run(s, a, t)
                                        ),
                                        suite=suite,
                                        candidate=f"perexpert_b_{arm}",
                                        boundary=(
                                            BOUNDARY_ROUTE_INCLUSIVE
                                            if arm in GROUPED_ARMS
                                            else BOUNDARY_PREPARED_INPUT
                                        ),
                                        params={
                                            "workload": record_workload_identity(case),
                                            "case_id": case.case_id,
                                            "phase": "decided",
                                            "T": num_tokens,
                                            "rank": rank,
                                            "site": site,
                                            "expert_id_domain": domain,
                                            "seed": seed,
                                            "repeat": repeat,
                                            "regime": regime,
                                            "arm_config": (
                                                _sgmv_key(tuned[arm])
                                                if "sgmv" in arm
                                                else config_key(tuned[arm])
                                            ),
                                        },
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
                f"{cell[4]:5s} {cell[0]:8s} {cell[1]:8s} r{cell[2]:<4d} "
                f"T={cell[3]:<5d} {arm_a}/{arm_b:18s} "
                f"geo(a/b)={decision.geo_a_over_b:.3f} -> "
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
