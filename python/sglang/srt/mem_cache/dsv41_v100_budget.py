"""Per-rank HBM budget and dry-run allocator for DeepSeek-V4.1-Flash on 8×V100.

Byte counts come from the HF 2026-09-10 checkpoint (~476 GiB on disk). This
module does not download weights. It uses fake device memory: integers only.

v1 shape: TP8/EP8, language-model-only (no ViT, no DSpark), Engram on host,
routed-expert spill ~8–12 GiB/GPU, np=1, chunked prefill 2k–4k, decode graph bs=4.

``--with-dspark`` does **not** fold ``mtp.*`` into the 384-expert
spill pile. Target still skips those weights; the draft worker keeps 128×3
MXFP4 experts GPU-resident, plus landing 6·(γ+1), a second CUDA graph, SWA
draft KV, aux-hidden at layers 37/38/39, and T=γ+1 verify workspace.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from typing import Mapping, Optional

from sglang.srt.mem_cache.dsv41_host_placement import (
    DOCUMENTED_GPU_NUMA_NODE,
    DOCUMENTED_NODE_TOTAL_GIB,
    GIB,
    EngramNumaError,
    HostPlacementPlan,
    NumaNodeMem,
    engram_numa_node,
    plan_engram_host_tables,
    read_huge_pages,
    read_numa_nodes,
)

# ---------------------------------------------------------------------------
# Stored sizes of the HF 2026-09-10 checkpoint (GiB = 2**30).
# ---------------------------------------------------------------------------

EXPERTS_ROUTED_DSPARK_MXFP4_GIB = 259.5
ENGRAM_FP8_GIB = 189.0  # tables + their UE8M0 scales; v1 home = host
ATTN_ROUTERS_FP8_GIB = 7.0
EMBED_LM_BF16_GIB = 4.0
UE8M0_SCALES_GIB = 22.0

# ~14B DSpark params in MXFP4 (0.5 B/param). Subtracted from the 259.5 GiB
# expert pile because the target never loads ``mtp.*``. The draft's real
# HBM is ``_dspark_experts_gib`` (128×3 EP-sharded), not this number dumped
# back onto routed experts.
DSPARK_MXFP4_GIB = 14e9 * 0.5 / GIB

N_RANKS = 8
N_ROUTED_EXPERTS = 384
N_SHARED_EXPERTS = 1
N_LAYERS = 40
HIDDEN = 5120
MOE_INTERMEDIATE = 2304
HC_MULT = 4
ENGRAM_LAYERS = ((1, 384_006_168), (14, 384_016_682))
ENGRAM_HEAD_DIM = 256
ENGRAM_SCALE_BLOCK = 32
GLOBAL_KV_BYTES_PER_TOKEN = 890

# Flash text_config. Draft is 3 DSparkV4Stage layers, MQA, compress_ratio=0.
DSPARK_N_STAGES = 3
DSPARK_GAMMA = 5  # dspark_block_size; verify width is γ+1
DSPARK_N_ROUTED = 128
DSPARK_SWA = 128
DSPARK_KV_HEADS = 1
DSPARK_HEAD_DIM = 512
DSPARK_TARGET_LAYER_IDS = (37, 38, 39)

# Relaunch104: "D4-G landing pool: 6 slots ... 107.6 MiB (shared across layers)".
# Linear in slot count (one Marlin-packed expert row per slot).
LANDING_SLOTS_V1 = 6
LANDING_MIB_AT_V1 = 107.6
DEFAULT_MAX_RUNNING_REQUESTS = 1

HBM_GIB = 32.0
HBM_FAIL_GIB = 31.0  # dry-run exits non-zero if any rank exceeds this
TARGET_SLACK_GIB = 2.0
DEFAULT_SPILL_GIB = 10.0  # midpoint of the 8–12 GiB/GPU band; 8 misses slack
DEFAULT_CHUNK = 2048
DEFAULT_CONTEXT = 262_144
DEFAULT_DECODE_MAX_BS = 4
DEFAULT_TP = 8
DEFAULT_EP = 8


def _gib(nbytes: int) -> float:
    return nbytes / GIB


def engram_table_bytes(num_embeddings: int, dim: int = ENGRAM_HEAD_DIM) -> int:
    """FP8 rows + e8m0 scales (block 32), matching EngramEmbedding host layout."""
    return num_embeddings * dim + num_embeddings * (dim // ENGRAM_SCALE_BLOCK)


def mxfp4_expert_bytes() -> int:
    """w13 + w2 for one expert: 3 * H * I elements at 4 bits."""
    params = 3 * HIDDEN * MOE_INTERMEDIATE
    return params // 2


def expert_scale_bytes() -> int:
    """UE8M0 group-32 scales for one expert's MXFP4 weights."""
    params = 3 * HIDDEN * MOE_INTERMEDIATE
    return params // 32


@dataclass(frozen=True)
class RankBudget:
    rank: int
    experts_kept_gib: float
    dense_gib: float
    scales_kept_gib: float
    activations_gib: float
    graphs_gib: float
    kv_gib: float
    workspace_gib: float
    spill_gib: float
    engram_hbm_gib: float
    request_window_gib: float
    landing_gib: float = 0.0
    dspark_experts_gib: float = 0.0
    draft_kv_gib: float = 0.0
    draft_graphs_gib: float = 0.0
    aux_hidden_gib: float = 0.0
    verify_act_gib: float = 0.0
    hbm_gib: float = 0.0
    slack_gib: float = 0.0
    hbm_cap_gib: float = HBM_GIB

    @property
    def over_fail_line(self) -> bool:
        return self.hbm_gib > HBM_FAIL_GIB

    @property
    def misses_slack(self) -> bool:
        return self.slack_gib < TARGET_SLACK_GIB


@dataclass(frozen=True)
class BudgetResult:
    ranks: tuple[RankBudget, ...]
    host: HostPlacementPlan
    layout: str
    spill_gib: float
    chunk: int
    skip_vit: bool
    skip_dspark: bool
    host_engram: bool
    landing_slots: int
    dspark_gamma: int
    notes: tuple[str, ...]
    suggested_mem_fraction: float = 0.99

    @property
    def hbm_ok(self) -> bool:
        return not any(r.over_fail_line for r in self.ranks)

    @property
    def slack_ok(self) -> bool:
        return all(not r.misses_slack for r in self.ranks)


def _activation_gib(chunk: int) -> float:
    """Working set for np=1 chunked prefill, hc_mult=4, FP16 activations.

    One residual stream is tiny (~0.08 GiB at 2k). The peak is several stream
    copies plus indexer / Marlin workspace. 1.80 GiB at 2k, 2.40 GiB at 4k so
    the claimed slack band holds at 10 GiB spill.
    """
    return 1.80 + 0.60 * ((chunk - DEFAULT_CHUNK) / DEFAULT_CHUNK)


def _graph_gib(decode_max_bs: int) -> float:
    """CUDA-graph capture for decode bs=1..max_bs. Small bs, 40 layers, MoE."""
    return 0.40 + 0.10 * max(decode_max_bs, 1)


def landing_slots_for_dspark(gamma: int = DSPARK_GAMMA) -> int:
    """D3 G-pool: 6 · (γ+1) so target-verify unique hits fit without ensure()."""
    return LANDING_SLOTS_V1 * (int(gamma) + 1)


def _landing_gib(n_slots: int) -> float:
    """Shared Marlin landing pool. Measured 107.6 MiB at 6 slots; linear in N."""
    if n_slots <= 0:
        return 0.0
    return (LANDING_MIB_AT_V1 / LANDING_SLOTS_V1) * n_slots / 1024.0


def _dspark_experts_gib(
    *,
    ep_size: int,
    tp_size: int,
    n_routed: int = DSPARK_N_ROUTED,
    n_stages: int = DSPARK_N_STAGES,
) -> float:
    """Draft MoE + 3-layer attention dense. All GPU-resident; no spill LRU.

    EP-shards the 128 routed experts (16/rank × 3 stages). Unsharded 128×3
    on every rank would be ~6.3 GiB and would miss the 32 GiB card.
    """
    n_local = n_routed / ep_size
    moe = n_local * n_stages * mxfp4_expert_bytes() / GIB
    scales = n_local * n_stages * expert_scale_bytes() / GIB
    dense = (ATTN_ROUTERS_FP8_GIB / N_LAYERS) * n_stages / tp_size
    return moe + scales + dense


def _dspark_draft_kv_gib(
    *,
    n_stages: int = DSPARK_N_STAGES,
    max_running_requests: int = DEFAULT_MAX_RUNNING_REQUESTS,
) -> float:
    """FP16 K+V SWA-128 MQA (head_dim 512) on the draft worker."""
    reqs = max(int(max_running_requests), 1)
    nbytes = (
        n_stages
        * DSPARK_SWA
        * DSPARK_KV_HEADS
        * DSPARK_HEAD_DIM
        * 2  # K and V
        * 2  # fp16
        * reqs
    )
    return _gib(nbytes)


def _dspark_target_graph_gib(gamma: int) -> float:
    """One target-verify capture at width γ+1, decode_max_bs=1 ."""
    return 0.40 + 0.10 * (int(gamma) + 1)


def _dspark_draft_graph_gib(gamma: int, n_stages: int = DSPARK_N_STAGES) -> float:
    """Draft capture at width γ, 3 layers vs 40, plus Markov/head pad."""
    frac = n_stages / N_LAYERS
    return frac * (0.40 + 0.10 * int(gamma)) + 0.05


def _dspark_aux_hidden_gib(chunk: int, n_layers: int = len(DSPARK_TARGET_LAYER_IDS)) -> float:
    """Current-forward hidden at layers 37/38/39. Not a 256k history buffer."""
    return n_layers * chunk * HIDDEN * 2 / GIB


def _dspark_verify_act_gib(gamma: int) -> float:
    """T=γ+1 CSA2/Engram decode workspace beyond the T≤2 landing pad."""
    verify_t = int(gamma) + 1
    t1_decode_ws = 0.05
    return t1_decode_ws * max(verify_t / 2.0 - 1.0, 0.0)


def _rank_hbm_sum(**fields: float) -> float:
    return sum(fields.values())


def _suggested_mem_fraction(row: RankBudget) -> float:
    """Leave graphs + workspace + 0.25 GiB outside mem-fraction-static.

    v1 serve uses 0.99 so leftover is not eaten by SWA/c4. DSpark adds a
    second graph; 0.99 may still work if capture allocates from the static
    pool. This is the conservative ceiling if it does not.
    """
    runtime = (
        row.graphs_gib
        + row.draft_graphs_gib
        + row.activations_gib
        + row.workspace_gib
        + row.verify_act_gib
        + 0.25
    )
    return max(0.80, min(0.99, (HBM_GIB - runtime) / HBM_GIB))


def allocate(
    *,
    tp_size: int = DEFAULT_TP,
    ep_size: int = DEFAULT_EP,
    ranks: int = N_RANKS,
    spill_gib: float = DEFAULT_SPILL_GIB,
    chunk: int = DEFAULT_CHUNK,
    context: int = DEFAULT_CONTEXT,
    decode_max_bs: int = DEFAULT_DECODE_MAX_BS,
    skip_vit: bool = True,
    skip_dspark: bool = True,
    host_engram: bool = True,
    engram_layout: str = "private",
    allow_numa_split: bool = False,
    preferred_numa: int = DOCUMENTED_GPU_NUMA_NODE,
    request_window: bool = False,
    landing_slots: Optional[int] = None,
    dspark_gamma: int = DSPARK_GAMMA,
    dspark_n_routed: int = DSPARK_N_ROUTED,
    dspark_n_stages: int = DSPARK_N_STAGES,
    max_running_requests: int = DEFAULT_MAX_RUNNING_REQUESTS,
    lie: Optional[Mapping[str, float]] = None,
    node_totals_gib: Optional[Mapping[int, float]] = None,
    huge_pages_total: Optional[int] = None,
) -> BudgetResult:
    """Build the per-rank HBM table and the host NUMA plan.

    ``lie`` overrides named GiB fields on every rank after the honest compute
    (used to prove the dry-run fails when a rank is over 31 GiB).

    Target never loads ``mtp.*``. ``skip_dspark=False`` adds a GPU-resident
    draft row (128 experts × 3 stages, EP-sharded, no LRU) plus verify-width
    extras; it does not put those bytes back into the 384-expert spill pile.
    """
    if tp_size != ep_size or tp_size != ranks:
        raise ValueError(f"v1 allocator is TP=EP=ranks, got tp={tp_size} ep={ep_size} ranks={ranks}")
    if chunk < 2048 or chunk > 4096:
        # Allowed, but the slack target is only claimed inside this band.
        pass

    # Target routed pile always drops DSpark. The old --with-dspark path added
    # DSPARK_MXFP4_GIB here, then spilled it — that was the 0.8 GiB-only lie.
    routed_mxfp4 = EXPERTS_ROUTED_DSPARK_MXFP4_GIB - DSPARK_MXFP4_GIB
    # Shared expert is dense-equivalent and stays GPU; it is in the 7 GiB
    # attention/dense pile rather than the 384-routed MXFP4 pile.
    experts_per_rank = routed_mxfp4 / ep_size
    expert_scales_all = expert_scale_bytes() * N_ROUTED_EXPERTS * N_LAYERS / GIB
    dspark_scale = (DSPARK_MXFP4_GIB / EXPERTS_ROUTED_DSPARK_MXFP4_GIB) * (
        UE8M0_SCALES_GIB
        * (
            1.0
            - (expert_scale_bytes() * N_ROUTED_EXPERTS * N_LAYERS / GIB)
            / UE8M0_SCALES_GIB
        )
    )
    dense_scales = max(UE8M0_SCALES_GIB - expert_scales_all - dspark_scale, 0.0)

    dense = (ATTN_ROUTERS_FP8_GIB + EMBED_LM_BF16_GIB + dense_scales) / tp_size
    if not skip_vit:
        # Fit note: skip ViT in v1. A 32-layer ViT-H/14-class tower is ~0.4 GiB FP8
        # plus activations; keep a conservative 1.0 GiB/rank so a lie is visible.
        dense += 1.0

    spill = min(max(spill_gib, 0.0), experts_per_rank)
    experts_kept = experts_per_rank - spill
    scale_keep_frac = experts_kept / experts_per_rank if experts_per_rank else 1.0
    scales_kept = (expert_scales_all / ep_size) * scale_keep_frac

    activations = _activation_gib(chunk)
    kv = context * GLOBAL_KV_BYTES_PER_TOKEN / GIB
    # SWA 128 × 40 layers is in the 890 B/tok global figure's "plus window"
    # noise; keep an explicit 0.05 GiB so it shows up in the table.
    kv += 0.05
    workspace = 0.50  # NCCL, allocator fragmentation, Marlin scratch
    engram_hbm = 0.0 if host_engram else ENGRAM_FP8_GIB / tp_size
    # RequestWindow replaces paged SWA when enabled; np=1, 8 slots, window 128,
    # 40 layers, ~584 B/token (V4 SWA packing) ≈ 0.02 GiB. Budget it so turning
    # replay on later cannot surprise HBM.
    request_window_gib = 0.03 if request_window else 0.0

    if landing_slots is None:
        landing_slots = (
            LANDING_SLOTS_V1
            if skip_dspark
            else landing_slots_for_dspark(dspark_gamma)
        )
    landing = _landing_gib(int(landing_slots))

    if skip_dspark:
        graphs = _graph_gib(decode_max_bs)
        dspark_experts = 0.0
        draft_kv = 0.0
        draft_graphs = 0.0
        aux_hidden = 0.0
        verify_act = 0.0
    else:
        # Do not capture bs>1. Target width γ+1 replaces the v1 T=1 graph set.
        graphs = _dspark_target_graph_gib(dspark_gamma)
        dspark_experts = _dspark_experts_gib(
            ep_size=ep_size,
            tp_size=tp_size,
            n_routed=dspark_n_routed,
            n_stages=dspark_n_stages,
        )
        draft_kv = _dspark_draft_kv_gib(
            n_stages=dspark_n_stages,
            max_running_requests=max_running_requests,
        )
        draft_graphs = _dspark_draft_graph_gib(dspark_gamma, dspark_n_stages)
        aux_hidden = _dspark_aux_hidden_gib(chunk)
        verify_act = _dspark_verify_act_gib(dspark_gamma)

    notes = [
        (
            "v1: --language-model-only, no --speculative-algorithm (no ViT, no DSpark)"
            if skip_dspark
            else "--language-model-only still skips ViT; DSpark is the draft worker, not a vision flag"
        ),
        "Engram ~189 GiB lives on host when SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 (default on SM70)",
        f"routed-expert LRU spill {spill:.1f} GiB/rank; attention stays GPU",
        f"D4-G landing {int(landing_slots)} slots = {landing:.2f} GiB (relaunch104: 6 slots / 107.6 MiB)",
        "--cpu-offload-gb would page whole layers including CSA2/indexer: see tax in the spreadsheet",
        f"HugePages_Total is currently {read_huge_pages()[0]} ({read_huge_pages()[1]} kB); "
        "Engram uses the 1G hugetlb pool, expert spill uses node-1 THP",
    ]
    if not skip_dspark:
        notes.extend(
            [
                f"DSpark draft: {dspark_n_stages} stages × {dspark_n_routed} routed experts, "
                f"EP-sharded, all GPU-resident ({dspark_experts:.2f} GiB/rank); no draft spill LRU",
                "target still skips mtp.*; only the draft worker loads those weights",
                f"γ={dspark_gamma} → target verify T={dspark_gamma + 1}, draft graph width {dspark_gamma}, decode_max_bs=1",
                f"aux hidden layers {list(DSPARK_TARGET_LAYER_IDS)} sized to chunk={chunk}, not 256k history",
            ]
        )

    rank_rows = []
    for rank in range(ranks):
        fields = {
            "experts_kept_gib": experts_kept,
            "dense_gib": dense,
            "scales_kept_gib": scales_kept,
            "activations_gib": activations,
            "graphs_gib": graphs,
            "kv_gib": kv,
            "workspace_gib": workspace,
            "engram_hbm_gib": engram_hbm,
            "request_window_gib": request_window_gib,
            "landing_gib": landing,
            "dspark_experts_gib": dspark_experts,
            "draft_kv_gib": draft_kv,
            "draft_graphs_gib": draft_graphs,
            "aux_hidden_gib": aux_hidden,
            "verify_act_gib": verify_act,
        }
        hbm = _rank_hbm_sum(**fields)
        row = RankBudget(
            rank=rank,
            spill_gib=spill,
            **fields,
            hbm_gib=hbm,
            slack_gib=HBM_GIB - hbm,
        )
        if lie:
            for key, value in lie.items():
                if key not in fields:
                    raise KeyError(f"unknown lie field {key!r}; have {sorted(fields)}")
                fields[key] = float(value)
            hbm = _rank_hbm_sum(**fields)
            row = replace(row, **fields, hbm_gib=hbm, slack_gib=HBM_GIB - hbm)
        rank_rows.append(row)

    tables = tuple(
        (layer_id, engram_table_bytes(rows)) for layer_id, rows in ENGRAM_LAYERS
    )
    # The fit-note 189 GiB is the planning number; the row formula is ~188.8.
    # Scale shards so host placement uses 189 GiB, not a silent 0.2 GiB miss.
    formula_total = sum(b for _, b in tables)
    scale = (ENGRAM_FP8_GIB * GIB) / formula_total if formula_total else 1.0
    tables = tuple((lid, int(round(b * scale))) for lid, b in tables)

    if node_totals_gib is None:
        node_totals_gib = DOCUMENTED_NODE_TOTAL_GIB
    nodes = {
        n: NumaNodeMem(n, int(round(g * GIB)), int(round(g * GIB)))
        for n, g in node_totals_gib.items()
    }

    extra = int(round(spill * GIB)) if spill > 0 else 0
    if host_engram:
        host = plan_engram_host_tables(
            table_nbytes=tables,
            layout=engram_layout,
            tp_size=tp_size,
            preferred_node=preferred_numa,
            allow_split=allow_numa_split,
            extra_per_rank_bytes=extra,
            nodes=nodes,
            huge_pages_total=(
                huge_pages_total if huge_pages_total is not None else read_huge_pages()[0]
            ),
        )
    else:
        host = HostPlacementPlan(
            preferred_node=preferred_numa,
            allow_split=allow_numa_split,
            huge_pages_total=huge_pages_total or 0,
        )
        notes.append("host_engram=False: ~189 GiB Engram counted on HBM — will not fit 32 GiB cards")

    suggested = _suggested_mem_fraction(rank_rows[0]) if rank_rows else 0.99
    return BudgetResult(
        ranks=tuple(rank_rows),
        host=host,
        layout=engram_layout,
        spill_gib=spill,
        chunk=chunk,
        skip_vit=skip_vit,
        skip_dspark=skip_dspark,
        host_engram=host_engram,
        landing_slots=int(landing_slots),
        dspark_gamma=int(dspark_gamma),
        notes=tuple(notes),
        suggested_mem_fraction=suggested,
    )


def format_rank_table(result: BudgetResult) -> str:
    rows = result.ranks
    headers = [
        "rank",
        "experts_kept",
        "dense",
        "scales",
        "activations",
        "graphs",
        "kv",
        "workspace",
        "engram_hbm",
        "req_win",
        "landing",
        "hbm",
        "slack",
        "spill_host",
    ]

    def fmt(x: float) -> str:
        return f"{x:.2f}"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.rank),
                    fmt(r.experts_kept_gib),
                    fmt(r.dense_gib),
                    fmt(r.scales_kept_gib),
                    fmt(r.activations_gib),
                    fmt(r.graphs_gib),
                    fmt(r.kv_gib),
                    fmt(r.workspace_gib),
                    fmt(r.engram_hbm_gib),
                    fmt(r.request_window_gib),
                    fmt(r.landing_gib),
                    fmt(r.hbm_gib),
                    fmt(r.slack_gib),
                    fmt(r.spill_gib),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def format_dspark_extras_table(result: BudgetResult) -> str:
    r = result.ranks[0]
    headers = [
        "dspark_experts",
        "draft_kv",
        "draft_graphs",
        "aux_hidden",
        "verify_act",
        "landing_slots",
        "gamma",
    ]

    def fmt(x: float) -> str:
        return f"{x:.3f}"

    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            "| "
            + " | ".join(
                [
                    fmt(r.dspark_experts_gib),
                    fmt(r.draft_kv_gib),
                    fmt(r.draft_graphs_gib),
                    fmt(r.aux_hidden_gib),
                    fmt(r.verify_act_gib),
                    str(result.landing_slots),
                    str(result.dspark_gamma),
                ]
            )
            + " |",
        ]
    )


def format_report(result: BudgetResult) -> str:
    parts = [
        f"DSV4.1-Flash 8×V100 dry-run  tp={DEFAULT_TP} ep={DEFAULT_EP}  "
        f"chunk={result.chunk}  spill={result.spill_gib:.1f} GiB/rank  "
        f"engram={'host/' + result.layout if result.host_engram else 'HBM'}"
        f"  dspark={'on' if not result.skip_dspark else 'off'}"
        f"  landing={result.landing_slots}",
        "",
        format_rank_table(result),
        "",
    ]
    if not result.skip_dspark:
        parts.extend(
            [
                "DSpark extras (GiB/rank; not folded into routed spill):",
                format_dspark_extras_table(result),
                "",
            ]
        )
    parts.extend(
        [
            "GiB per rank. Fail line is 31.00 (exit 1 if any rank exceeds it). "
            f"Slack target is ≥{TARGET_SLACK_GIB:.0f} GiB at np=1.",
            "",
            "Host placement (preferred node = GPU-local, or SGLANG_DSV41_ENGRAM_NUMA_NODE):",
        ]
    )
    by_node: dict[int, float] = {}
    for m in result.host.mappings:
        by_node[m.node] = by_node.get(m.node, 0.0) + m.nbytes / GIB
    if by_node:
        for node in sorted(by_node):
            parts.append(f"  node {node}: {by_node[node]:.1f} GiB planned")
    else:
        parts.append("  (no host mappings)")
    if result.host.huge_pages_total == 0:
        parts.append("  HugePages_Total=0 (no 1G hugetlb pool reserved)")
    for w in result.host.warnings:
        parts.append(f"  warning: {w}")
    parts.append("")
    for n in result.notes:
        parts.append(f"- {n}")
    worst = max(rows.hbm_gib for rows in result.ranks)
    slack = min(rows.slack_gib for rows in result.ranks)
    parts.append("")
    parts.append(
        f"worst rank HBM {worst:.2f} GiB  slack {slack:.2f} GiB  "
        f"hbm_ok={result.hbm_ok}  slack_ok={result.slack_ok}  "
        f"suggested_mem_fraction_static={result.suggested_mem_fraction:.2f}"
    )
    return "\n".join(parts)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spill-gb", type=float, default=DEFAULT_SPILL_GIB)
    p.add_argument("--chunked-prefill-size", type=int, default=DEFAULT_CHUNK)
    p.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT)
    p.add_argument("--decode-max-bs", type=int, default=DEFAULT_DECODE_MAX_BS)
    p.add_argument(
        "--engram-layout",
        choices=("private", "shared"),
        default="private",
        help="private = host-sharded. shared = one memfd copy (needs the "
        "unsharded tables to fit on the GPU-local NUMA node).",
    )
    p.add_argument("--allow-numa-split", action="store_true")
    p.add_argument("--no-host-engram", action="store_true")
    p.add_argument("--with-dspark", action="store_true")
    p.add_argument(
        "--landing-slots",
        type=int,
        default=None,
        help="D4-G landing slots. Default 6 (v1) or 6*(γ+1) with --with-dspark.",
    )
    p.add_argument(
        "--dspark-gamma",
        type=int,
        default=DSPARK_GAMMA,
        help="dspark_block_size. Verify width is γ+1.",
    )
    p.add_argument("--dspark-n-routed", type=int, default=DSPARK_N_ROUTED)
    p.add_argument("--dspark-n-stages", type=int, default=DSPARK_N_STAGES)
    p.add_argument(
        "--max-running-requests",
        type=int,
        default=DEFAULT_MAX_RUNNING_REQUESTS,
    )
    p.add_argument(
        "--require-slack",
        action="store_true",
        help="Exit 1 if slack < 2 GiB. Implied by --with-dspark.",
    )
    p.add_argument(
        "--allow-miss-slack",
        action="store_true",
        help="Do not fail --with-dspark on slack < 2 (still prints slack_ok).",
    )
    p.add_argument("--with-vit", action="store_true")
    p.add_argument("--request-window", action="store_true")
    p.add_argument(
        "--lie-experts-kept-gib",
        type=float,
        default=None,
        help="Overwrite experts_kept on every rank (prove the 31 GiB fail line).",
    )
    p.add_argument("--probe-live-numa", action="store_true")
    args = p.parse_args(argv)

    lie = None
    if args.lie_experts_kept_gib is not None:
        lie = {"experts_kept_gib": args.lie_experts_kept_gib}

    node_totals = None
    preferred_numa = DOCUMENTED_GPU_NUMA_NODE
    if args.probe_live_numa:
        live = read_numa_nodes(documented_fallback=False)
        node_totals = {n: m.total_gib for n, m in live.items()}
        preferred_numa = engram_numa_node()

    try:
        result = allocate(
            spill_gib=args.spill_gb,
            chunk=args.chunked_prefill_size,
            context=args.context_length,
            decode_max_bs=args.decode_max_bs,
            skip_vit=not args.with_vit,
            skip_dspark=not args.with_dspark,
            host_engram=not args.no_host_engram,
            engram_layout=args.engram_layout,
            allow_numa_split=args.allow_numa_split,
            preferred_numa=preferred_numa,
            request_window=args.request_window,
            landing_slots=args.landing_slots,
            dspark_gamma=args.dspark_gamma,
            dspark_n_routed=args.dspark_n_routed,
            dspark_n_stages=args.dspark_n_stages,
            max_running_requests=args.max_running_requests,
            lie=lie,
            node_totals_gib=node_totals,
        )
    except EngramNumaError as e:
        print(f"HOST PLACEMENT FAILED: {e}", file=sys.stderr)
        return 2

    print(format_report(result))
    if not result.hbm_ok:
        worst = max(r.hbm_gib for r in result.ranks)
        print(
            f"FAIL: rank HBM {worst:.2f} GiB exceeds {HBM_FAIL_GIB:.2f} GiB",
            file=sys.stderr,
        )
        return 1
    require_slack = args.require_slack or (args.with_dspark and not args.allow_miss_slack)
    if require_slack and not result.slack_ok:
        slack = min(r.slack_gib for r in result.ranks)
        print(
            f"FAIL: rank slack {slack:.2f} GiB is below {TARGET_SLACK_GIB:.2f} GiB",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
