"""Step-4 LoRA-B schedule cells (plan §64.1).

Five sections, selectable with ``--sections``:

* ``floor`` — the destination WRITE floor: filling the [P, 2I] / [P, H]
  delta buffer is work no B schedule avoids, so its cost bounds the
  headroom any kernel comparison can possibly recover. Recorded per
  (site, T) as a diagnostic candidate before anything else runs.
* ``per_stage`` — BOUNDARY_ISOLATED: one B execution per thunk over a
  prebuilt plan and a REAL bridge (grouped A fills it once per fixture).
  Default-config screening across the four families.
* ``sweep`` — per-family config grids, emitting a per-device table keyed
  ``{site: {rank: {family: {regime: cfg}}}}`` for decode-tiny, decode,
  prefill, and XL-prefill (per the ninth-review lesson — one decode-tuned
  config misrepresents prefill). The BLOCK_K axis relative to the rank IS
  the whole-rank vs looped-rank comparison; the emitted table records
  which won.
* ``decided`` — the ratified seeded methodology (3 seeds x 2 interleaved
  repeats, unanimity + margin) at the tuned configs, stock vs each
  challenger, both sites, graph mode everywhere plus EAGER at prefill
  (production prefill is eager). Crossovers enter the evidence-bound
  ledger with explicit workload signatures.
* ``leg`` — BOUNDARY_ROUTE_INCLUSIVE: one thunk = every route build the
  (A, B) schedule pair needs plus all four LoRA GEMMs of one MoE leg.
  Includes the ALL-INDEXED leg — indexed A + indexed B builds ZERO route
  kernels, the configuration Step 3 could not measure because stock B
  forced the aligned plan.

Usage::

    python3 -m benchmark.kernels.lora_moe.bench_lora_b \
        --output lora_b_v1.json --source-revision <sha> \
        [--sections floor,per_stage,sweep] [--ranks 16,32,64,128] \
        [--config-table <emitted json>]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import msgspec
import torch
import triton

from benchmark.kernels.lora_moe.bench_common import (
    DECODE_T_MAX,
    TABLE_SCHEMA_VERSION,
    config_key,
    exhaustive_grouped_lora_b_grid,
    padded_block_k_cap,
    parse_table_config,
    regime_of,
    require_delta_close_chunked,
    require_distinct_paths,
    require_sha256_reference,
    require_table_provenance,
    require_writable_destination,
    skip_entry,
    skip_reason,
    table_content_digest,
    write_skip_sidecar,
)
from benchmark.kernels.lora_moe.bench_lora_a import _LegFixture
from benchmark.kernels.lora_moe.cases import AdapterCell, Topology, build_case
from benchmark.kernels.lora_moe.crossover_ledger import decide_cell
from benchmark.kernels.lora_moe.lora_a_candidates import (
    INDEXED_DEFAULT_CONFIG,
    run_lora_a,
)
from benchmark.kernels.lora_moe.lora_a_execution import LoraAExecutionSpec
from benchmark.kernels.lora_moe.lora_b_candidates import (
    INDEXED_B_DEFAULT_CONFIG,
    ONE_LAUNCH_DEFAULT_CONFIG,
    RANK_SPLIT_DEFAULT_CONFIG,
    rank_split_b_workspace,
    rank_split_workspace_fits,
    run_lora_b,
)
from benchmark.kernels.lora_moe.lora_b_execution import LoraBExecutionSpec
from benchmark.kernels.lora_moe.timing import (
    BOUNDARY_ISOLATED,
    BOUNDARY_ROUTE_INCLUSIVE,
    CACHE_L2_HOT_GRAPH,
    atomic_write_bytes,
    kernel_fingerprint,
    measure,
    new_suite,
    write_content_addressed_suite,
    write_suite,
)
from sglang.srt.lora.sgl_lora.bf16 import stock_grouped_lora_b
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import ROUTE_ALIGNED, ROUTE_RAW

# T=1/4/8 close the gate-4 small-batch gap (the claimed batch-1 niche).
T_GRID = (1, 4, 8, 16, 64, 256, 2048, 8192)
LEG_RANKS = (16, 64)
LEG_T_GRID = (1, 4, 8, 16, 64, 256, 2048)
SECTION_ORDER = ("floor", "per_stage", "sweep", "decided", "leg")
SEEDS = (11, 137, 997)
REPEATS = 2
ADAPTER_CELL = AdapterCell(active_adapters=4, include_base_rows=True, slot_capacity=8)
FAMILIES = ("stock", "lean_two_launch", "one_launch", "indexed", "rank_split")
DEFAULT_SWEEP_REGIMES = "4,64,2048,8192"
# Decided-only arm (never swept): the lean body run at ONE-LAUNCH's
# promoted config — isolates pure launch fusion from the config
# interaction that independently-tuned lean carries (2nd review, F4).
MATCHED_ARM = "lean_matched"
# Canonical adjudication pairs (8th review: module level + tested, so the
# rank_split-vs-promoted-kernel regression cannot silently return).
DECIDED_PAIRS = [
    ("stock", family)
    for family in ("lean_two_launch", "one_launch", "indexed", "rank_split")
] + [
    ("stock", MATCHED_ARM),
    ("one_launch", MATCHED_ARM),
    ("one_launch", "lean_two_launch"),
    ("one_launch", "indexed"),
    ("one_launch", "rank_split"),
]


# Main-B consumes promoted configs without re-tuning them, so every table
# workload field must match the consuming run.
LOCALLY_RETUNED = ()


def build_transfer_request(arguments) -> dict:
    """The EXACT workload request main-B sends when consuming a table."""
    return {
        "model_preset": arguments.model_preset,
        "adapter_cell": _adapter_cell_key(),
        "route_generator": "iid",
        "weight_ownership": "per_expert",
        "topology": "tp8_ep8",
        "expert_id_domain": "ep_local",
        "ranks": ",".join(str(rank) for rank in parse_rank_axis(arguments.ranks)),
        "sweep_regimes": ",".join(
            str(tokens) for tokens in parse_sweep_axis(arguments.sweep_regimes)
        ),
    }


def build_main_table(
    *,
    best: dict,
    arguments,
    suite,
    kernel_digest: str,
    sweep_checkpoint_digest: str,
    sweep_skips_digest: str,
) -> dict:
    """Assemble the config table EXACTLY as the sweep publishes it.

    9th review: the emitter used to hand-roll both the digest and the
    metadata, so a test could only approximate what production writes.
    Production and the round-trip test now call THIS.
    """
    require_sha256_reference(sweep_checkpoint_digest, label="sweep_checkpoint_digest")
    require_sha256_reference(sweep_skips_digest, label="sweep_skips_digest")
    table: dict = {
        "_meta": {
            "schema_version": TABLE_SCHEMA_VERSION,
            "objective": "min median per (site, rank, family, regime_class)",
            "source_revision": suite.source_revision,
            "observed_revision": suite.observed_revision,
            "source_digest": suite.source_digest,
            "kernel_digest": kernel_digest,
            "sweep_checkpoint_digest": sweep_checkpoint_digest,
            "sweep_skips_digest": sweep_skips_digest,
            "device_name": suite.device_name,
            "torch_version": suite.torch_version,
            "triton_version": triton.__version__,
            "workload": build_transfer_request(arguments),
        }
    }
    for (site, rank, family, regime_class), key in best.items():
        table.setdefault(site, {}).setdefault(str(rank), {}).setdefault(family, {})[
            regime_class
        ] = key
    # A published table is complete for every core family and regime at
    # each advertised rank. Individual consumers perform their own
    # section-specific preflight when loading it.
    ranks = parse_rank_axis(arguments.ranks)
    # 11th review: guarantee what BOTH consumers index. The leg section
    # uses a hard-coded LEG_RANKS, so a table emitted at another rank axis
    # published happily and then died at leg load with a message blaming
    # the table. Validate leg only when its ranks are actually covered,
    # and say plainly when they are not.
    sections = {"decided"}
    if set(LEG_RANKS).issubset(set(ranks)):
        sections.add("leg")
    else:
        print(
            f"NOTE: --ranks {arguments.ranks} does not cover LEG_RANKS "
            f"{LEG_RANKS}; this table cannot serve --sections leg"
        )
    require_main_table_for_sections(table, ranks=ranks, sections=sections)
    table["_meta"]["table_content_digest"] = table_content_digest(table)
    return table


MAIN_TABLE_REQUIRED_FAMILIES = (
    "stock",
    "lean_two_launch",
    "one_launch",
    "indexed",
)
MAIN_TABLE_REQUIRED_REGIMES = tuple(dict.fromkeys(regime_of(t) for t in T_GRID))
RANK_SPLIT_REQUIRED_REGIMES = ("decode_tiny", "decode")


def _rank_split_is_swept(rank: int, regime: str) -> bool:
    """Whether the producer measures this rank-split table cell.

    11th review: this models the regime and empty-grid skips but NOT the
    sweep's workspace-cap skip (SPLIT_K * destination bytes >
    RANK_SPLIT_WORKSPACE_CAP_BYTES), which is why the suite is now
    checkpointed before publication — a cell emptied by that cap would
    otherwise abort the emitter after the sweep. The cap cannot bite at
    any current preset (worst case ~92 MB against a 256 MiB cap), so this
    stays a documented approximation rather than a duplicated calculation.
    """
    return regime in RANK_SPLIT_REQUIRED_REGIMES and bool(
        _sweep_grid("rank_split", rank)
    )


def require_main_table_cells(
    table: dict,
    *,
    ranks: tuple[int, ...],
    families: tuple[str, ...] = MAIN_TABLE_REQUIRED_FAMILIES,
    regimes: tuple[str, ...] = MAIN_TABLE_REQUIRED_REGIMES,
    check_rank_split: bool = False,
) -> None:
    """Fail closed unless ``table`` covers the requested consumer surface.

    ``rank_split`` is required at ranks where its production sweep grid is
    nonempty and omitted only where that grid is structurally empty.
    """
    missing = [
        f"{site}/r{rank}/{family}/{regime}"
        for site in ("gate_up", "down")
        for rank in ranks
        for family in families
        for regime in regimes
        if regime not in table.get(site, {}).get(str(rank), {}).get(family, {})
    ]
    if check_rank_split:
        for site in ("gate_up", "down"):
            for rank in ranks:
                if not any(
                    _rank_split_is_swept(rank, regime)
                    for regime in RANK_SPLIT_REQUIRED_REGIMES
                ):
                    continue
                rank_split = table.get(site, {}).get(str(rank), {}).get("rank_split")
                missing.extend(
                    f"{site}/r{rank}/rank_split/{regime}"
                    for regime in RANK_SPLIT_REQUIRED_REGIMES
                    if regime not in (rank_split or {})
                )
    if missing:
        raise RuntimeError(
            f"table is incomplete for the requested B consumers: "
            f"missing {len(missing)} "
            f"cells, e.g. {sorted(missing)[:5]}"
        )


def require_table_emission_regimes(sweep_regimes: str) -> None:
    """Require one positive sweep anchor for each reusable-table regime."""
    anchors = parse_sweep_axis(sweep_regimes)
    requested = tuple(regime_of(value) for value in anchors)
    missing = set(MAIN_TABLE_REQUIRED_REGIMES) - set(requested)
    duplicates = sorted(
        regime for regime in set(requested) if requested.count(regime) > 1
    )
    if missing or duplicates:
        raise ValueError(
            "--emit-config-table requires exactly one sweep anchor for every "
            f"regime; missing={sorted(missing)}, duplicate={duplicates}"
        )


def require_main_table_for_sections(
    table: dict, *, ranks: tuple[int, ...], sections: set[str]
) -> dict:
    """Validate and parse the exact config cells used by selected sections."""
    required_cells = set()
    if "decided" in sections:
        require_main_table_cells(table, ranks=ranks, check_rank_split=True)
        required_cells.update(
            (site, rank, family, regime)
            for site in ("gate_up", "down")
            for rank in ranks
            for family in MAIN_TABLE_REQUIRED_FAMILIES
            for regime in MAIN_TABLE_REQUIRED_REGIMES
        )
        required_cells.update(
            (site, rank, "rank_split", regime)
            for site in ("gate_up", "down")
            for rank in ranks
            for regime in RANK_SPLIT_REQUIRED_REGIMES
            if _rank_split_is_swept(rank, regime)
        )
    if "leg" in sections:
        leg_families = ("stock", "one_launch", "indexed")
        leg_regimes = tuple(dict.fromkeys(regime_of(t) for t in LEG_T_GRID))
        require_main_table_cells(
            table,
            ranks=LEG_RANKS,
            families=leg_families,
            regimes=leg_regimes,
        )
        required_cells.update(
            (site, rank, family, regime)
            for site in ("gate_up", "down")
            for rank in LEG_RANKS
            for family in leg_families
            for regime in leg_regimes
        )
    parsed = {}
    valid_grids = {}
    for site, rank, family, regime in sorted(required_cells):
        text = table[site][str(rank)][family][regime]
        if not isinstance(text, str):
            raise ValueError(
                f"table config {site}/r{rank}/{family}/{regime} must be a string"
            )
        try:
            config = parse_table_config(text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid table config {site}/r{rank}/{family}/{regime}: {text!r}"
            ) from error
        grid_key = (family, rank)
        if grid_key not in valid_grids:
            valid_grids[grid_key] = {
                tuple(sorted(candidate.items()))
                for candidate in _sweep_grid(family, rank)
            }
        if tuple(sorted(config.items())) not in valid_grids[grid_key]:
            raise ValueError(
                f"table config {site}/r{rank}/{family}/{regime} "
                f"is outside the declared grid: {text!r}"
            )
        parsed[(site, rank, family, regime)] = config
    return parsed


def parse_rank_axis(ranks: str) -> tuple[int, ...]:
    """Parse the benchmark rank axis before CUDA initialization."""
    try:
        values = tuple(int(value.strip()) for value in ranks.split(","))
    except ValueError as error:
        raise ValueError("--ranks must contain comma-separated integers") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError("--ranks must contain positive integers")
    if len(values) != len(set(values)):
        raise ValueError("--ranks must not contain duplicates")
    return values


def parse_sweep_axis(sweep_regimes: str) -> tuple[int, ...]:
    """Parse a canonical, positive, unique sweep-token axis."""
    try:
        values = tuple(int(value.strip()) for value in sweep_regimes.split(","))
    except ValueError as error:
        raise ValueError(
            "--sweep-regimes must contain comma-separated integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise ValueError("--sweep-regimes must contain positive token counts")
    if len(values) != len(set(values)):
        raise ValueError("--sweep-regimes must not contain duplicate token counts")
    return values


def parse_sections(sections: str) -> set[str]:
    """Parse a unique, known section set before CUDA initialization."""
    values = tuple(value.strip() for value in sections.split(","))
    if not values or any(not value for value in values):
        raise ValueError("--sections must contain at least one section")
    if len(values) != len(set(values)):
        raise ValueError("--sections must not contain duplicates")
    unknown = sorted(set(values) - set(SECTION_ORDER))
    if unknown:
        raise ValueError(f"unknown --sections values: {unknown}")
    return set(values)


def write_sweep_checkpoint(suite, output_path: str) -> tuple[str, str]:
    """Publish a marked, immutable sweep snapshot without touching final output."""
    checkpoint = msgspec.structs.replace(
        suite,
        suite=f"{suite.suite}_sweep_checkpoint",
    )
    return write_content_addressed_suite(
        checkpoint,
        output_path,
        label="sweep-checkpoint",
    )


def write_config_table(table: dict, output_path: str) -> None:
    """Atomically publish one validated config table."""
    payload = json.dumps(table, indent=1, sort_keys=True).encode()
    atomic_write_bytes(output_path, payload)


def _spec(site: str, family: str) -> LoraBExecutionSpec:
    if family == "stock":
        return LoraBExecutionSpec(site=site, ownership="grouped")
    if family in ("lean_two_launch", MATCHED_ARM):
        return LoraBExecutionSpec(
            site=site, ownership="grouped", slicing="lean_per_slice"
        )
    if family == "one_launch":
        return LoraBExecutionSpec(
            site=site, ownership="grouped", slicing="one_launch_sliced"
        )
    if family == "indexed":
        return LoraBExecutionSpec(
            site=site, ownership="indexed", slicing="one_launch_sliced"
        )
    if family == "rank_split":
        return LoraBExecutionSpec(
            site=site,
            ownership="grouped",
            slicing="one_launch_sliced",
            reduction="deterministic_rank_split",
        )
    raise ValueError(family)


def _default_config(family: str) -> dict:
    return dict(
        {
            "stock": PROVISIONAL_LAUNCH_CONFIG.lora_b,
            "lean_two_launch": ONE_LAUNCH_DEFAULT_CONFIG,
            "one_launch": ONE_LAUNCH_DEFAULT_CONFIG,
            "indexed": INDEXED_B_DEFAULT_CONFIG,
            "rank_split": RANK_SPLIT_DEFAULT_CONFIG,
        }[family]
    )


def _adapter_cell_key() -> str:
    return (
        f"active{ADAPTER_CELL.active_adapters}_"
        f"cap{ADAPTER_CELL.slot_capacity}_"
        f"base{int(ADAPTER_CELL.include_base_rows)}"
    )


def _sweep_grid(family: str, rank: int) -> list[dict]:
    """Per-family config grid; BLOCK_K spans below AND at/above the rank so
    the looped-rank vs whole-rank question is answered by the same sweep.

    Gate-4 finding 3: the previous grid topped out at BN=128 and every
    winner sat AT that boundary, so nothing proved the frontier. BN now
    extends to 256/512 (infeasible combinations are auto-rejected at
    admission), GROUP_SIZE_M gains 1, and big tiles get a 4-stage row.
    """
    if family in ("stock", "lean_two_launch", "one_launch"):
        return list(exhaustive_grouped_lora_b_grid(rank=rank, stock=family == "stock"))
    # rank_split uses tl.dot and therefore keeps the grouped kernel's
    # power-of-two, >=16 K tiles. Indexed-B is an elementwise reduction:
    # tl.arange accepts smaller power-of-two extents, so rank 8 must retain
    # BK8 as a valid candidate rather than being forced to padded BK16.
    # 11th review: the SAME padded cap the grouped/sgmv grids use. These
    # diverged in the previous commit, so at ranks 48/96 indexed was tuned
    # over a strictly narrower K axis than the families it is adjudicated
    # against — a biased arm verdict, and the whole-rank tile the module
    # docstring promises was never measured for this family.
    block_ks = sorted({k for k in (16, 32, 64, 128) if k <= padded_block_k_cap(rank)})
    if family == "indexed":
        indexed_block_ks = set(block_ks)
        if 0 < rank < 16 and rank & (rank - 1) == 0:
            indexed_block_ks.add(rank)
        # 2nd review F2: every table cell sat AT the old BN=64 ceiling;
        # extend BN/warps/stages so the tiny-decode frontier is real.
        return [
            {
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "num_warps": warps,
                "num_stages": stages,
            }
            for bn in (16, 32, 64, 128, 256)
            for bk in sorted(indexed_block_ks)
            for warps in (2, 4, 8)
            for stages in (2, 3)
        ]
    if family == "rank_split":
        # 12th review: the old grid stopped at BN=64 / GM=8 / 4 warps and
        # every archived winner sat AT the BN ceiling, so "rank_split
        # uniformly rejected" was only established for a hobbled variant.
        # It now explores the same BN/GM/warp/stage space as the grouped
        # families it is adjudicated against; infeasible tiles are
        # rejected at admission or persisted as workspace-cap skips.
        return [
            {
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": group_m,
                "SPLIT_K": split,
                "num_warps": warps,
                "num_stages": stages,
            }
            for bn in (32, 64, 128, 256, 512, 1024)
            for bk in [k for k in block_ks if k * 2 <= rank]
            for group_m in (1, 4, 8, 16)
            for split in (2, 4, 8)
            for warps in ((4,) if bn < 128 else (4, 8))
            for stages in ((2, 3) if bn < 256 else (2, 3, 4))
            if rank // split >= bk
        ]
    raise ValueError(family)


class _BFixture:
    """A-leg fixture plus REAL bridges (grouped A output, not random)."""

    def __init__(self, case, device):
        self.leg = _LegFixture(case, device)
        self.case = case
        self.aligned = self.leg.route(ROUTE_ALIGNED)
        self.raw = self.leg.route(ROUTE_RAW)
        for site in ("gate_up", "down"):
            inp, weight, out = self.leg.site_buffers(site)
            run_lora_a(
                LoraAExecutionSpec(site=site, ownership="grouped"),
                input=inp,
                weight=weight,
                output=out,
                routing=self.aligned,
                config=PROVISIONAL_LAUNCH_CONFIG.lora_a,
            )

    def b_args(self, site: str):
        """(bridge, weight, destination, offsets) for one B site."""
        if site == "gate_up":
            return (
                self.leg.gate_rank_out,
                self.leg.b_gate_up,
                self.leg.gate_up_delta,
                (0, self.leg.intermediate),
            )
        return (
            self.leg.down_rank_out,
            self.leg.b_down,
            self.leg.down_delta,
            (0,),
        )

    def run_family(self, site, family, config, workspace=None):
        bridge, weight, destination, offsets = self.b_args(site)
        run_lora_b(
            _spec(site, family),
            bridge=bridge,
            weight=weight,
            destination=destination,
            routing=self.raw if family == "indexed" else self.aligned,
            destination_offsets=offsets,
            config=config,
            workspace=workspace,
        )


def _trusted_reference(fixture: _BFixture, site: str) -> torch.Tensor:
    """Cache the trusted-default reference per (fixture, site).

    6th review perf note: _admit rebuilt and cloned this for EVERY swept
    config. It depends only on the fixture and site, so build it once.
    """
    cache = fixture.__dict__.setdefault("_reference_cache", {})
    if site not in cache:
        bridge, weight, destination, offsets = fixture.b_args(site)
        destination.fill_(71.0)
        stock_grouped_lora_b(
            bridge,
            weight,
            destination,
            fixture.aligned,
            destination_offsets=offsets,
            config=PROVISIONAL_LAUNCH_CONFIG.lora_b,
        )
        cache[site] = destination.clone()
    return cache[site]


def _admit(fixture: _BFixture, site, family, config, workspace, label):
    """Zero-fill + trusted-reference admission before any timing record."""
    _, _, destination, _ = fixture.b_args(site)
    reference = _trusted_reference(fixture, site)
    destination.fill_(-3.0)
    fixture.run_family(site, family, config, workspace)
    # 5th review: bounded-memory EXACT gate (was a full .float() pair per
    # swept config — the actual cost driver of this sweep).
    require_delta_close_chunked(
        destination,
        reference,
        gate_dtype=torch.bfloat16,
        label=label,
    )


def _rank_split_ok(fixture: _BFixture, site, config) -> bool:
    _, _, destination, _ = fixture.b_args(site)
    return rank_split_workspace_fits(destination, split_k=int(config["SPLIT_K"]))


def _make_workspace(fixture: _BFixture, site, config):
    _, _, destination, _ = fixture.b_args(site)
    return rank_split_b_workspace(destination, split_k=int(config["SPLIT_K"]))


def build_parser() -> argparse.ArgumentParser:
    """Module-level so tests can validate the DEFAULTS production ships."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--model-preset", default="qwen35_35b")
    parser.add_argument("--sections", default="floor,per_stage")
    # Path C consumes all four regime classes. Keep the producer default
    # complete so the default Path-A table passes the production preflight.
    parser.add_argument("--sweep-regimes", default=DEFAULT_SWEEP_REGIMES)
    parser.add_argument("--emit-config-table", default=None)
    parser.add_argument("--config-table", default=None)
    return parser


def validate_run_arguments(arguments) -> set[str]:
    """Every pre-CUDA argument check, shared by production and tests.

    12th review: three end-of-run failures moved to startup — unwritable
    output destinations (the run could measure for hours and then lose
    everything), --emit-config-table without the sweep section (silently
    never written), and sweep+decided in one process (decided would
    adjudicate at PROVISIONAL defaults, not the winners the sweep just
    computed — the table must flow through a file).
    """
    parse_rank_axis(arguments.ranks)
    parse_sweep_axis(arguments.sweep_regimes)
    sections = parse_sections(arguments.sections)
    require_writable_destination(arguments.output, arguments.emit_config_table)
    require_distinct_paths(
        arguments.output, arguments.emit_config_table, arguments.config_table
    )
    if arguments.emit_config_table:
        require_table_emission_regimes(arguments.sweep_regimes)
        if "sweep" not in sections:
            raise ValueError(
                "--emit-config-table requires the sweep section; without it "
                "the flag is silently ignored and no table is ever written"
            )
    if "sweep" in sections and "decided" in sections:
        raise ValueError(
            "run decided in a separate invocation consuming the emitted "
            "table; in one process it would adjudicate at provisional "
            "defaults, not the sweep's winners"
        )
    if ({"decided", "leg"} & sections) and not arguments.config_table:
        raise ValueError(
            "--sections decided/leg require --config-table; without one "
            "they silently adjudicate at built-in default configs instead "
            "of the sweep's winners (squash review: this contract was "
            "documented but unenforced)"
        )
    return sections


def main() -> int:
    arguments = build_parser().parse_args()
    ranks = parse_rank_axis(arguments.ranks)
    sections = validate_run_arguments(arguments)
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)
    start_kernel_digest = kernel_fingerprint()
    suite = new_suite(
        "lora_b_schedules",
        source_revision=arguments.source_revision,
        producer_files=(__file__,),
    )

    config_table = None
    parsed_table_configs = {}
    if arguments.config_table:
        with open(arguments.config_table) as handle:
            config_table = json.load(handle)
        # 6th review: main-B must check its OWN workload too, or
        # --model-preset could disagree with the table it consumes.
        require_table_provenance(
            config_table,
            device,
            workload=build_transfer_request(arguments),
            locally_retuned=LOCALLY_RETUNED,
        )
        parsed_table_configs = require_main_table_for_sections(
            config_table, ranks=ranks, sections=sections
        )
        table_digests = {
            "table_source_digest": config_table["_meta"]["source_digest"],
            "table_kernel_digest": config_table["_meta"].get("kernel_digest"),
            "table_content_digest": config_table["_meta"]["table_content_digest"],
            "table_sweep_checkpoint_digest": config_table["_meta"][
                "sweep_checkpoint_digest"
            ],
            "table_sweep_skips_digest": config_table["_meta"]["sweep_skips_digest"],
        }
    else:
        table_digests = {}

    def tuned_config(site, rank, family, regime_class) -> dict:
        if config_table is None:
            return _default_config(family)
        return parsed_table_configs[(site, rank, family, regime_class)]

    def build_fixture(rank, num_tokens, seed):
        case = build_case(
            device=str(device),
            model_preset=arguments.model_preset,
            topology=Topology(tp_size=8, ep_size=8),
            adapter_cell=ADAPTER_CELL,
            route_generator="iid",
            num_tokens=num_tokens,
            active_rank=rank,
            seed=seed,
            source_revision=suite.source_revision,
        )
        return _BFixture(case, device)

    if "floor" in sections:
        for num_tokens in T_GRID:
            fixture = build_fixture(16, num_tokens, SEEDS[0])
            for site in ("gate_up", "down"):
                _, _, destination, _ = fixture.b_args(site)
                record = measure(
                    lambda d=destination: d.fill_(0),
                    suite=suite,
                    candidate=f"write_floor_{site}",
                    boundary=BOUNDARY_ISOLATED,
                    params={
                        "case_id": fixture.case.case_id,
                        "T": num_tokens,
                        "site": site,
                        "bytes": destination.numel() * 2,
                        "role": "write_floor",
                    },
                    graph_replay=True,
                )
                print(
                    f"floor {site:8s} T={num_tokens:<5d} "
                    f"{record.median_s * 1e6:7.2f} us "
                    f"({destination.numel() * 2 / record.median_s / 2**30:6.1f} "
                    "GiB/s)"
                )

    if "per_stage" in sections:
        for rank in ranks:
            for num_tokens in T_GRID:
                fixture = build_fixture(rank, num_tokens, SEEDS[0])
                for site in ("gate_up", "down"):
                    for family in FAMILIES:
                        config = _default_config(family)
                        workspace = None
                        if family == "rank_split":
                            if int(config["SPLIT_K"]) * 2 > rank:
                                continue
                            if not _rank_split_ok(fixture, site, config):
                                print(
                                    f"per_stage r{rank} T={num_tokens} {site} "
                                    "rank_split SKIPPED (workspace cap)"
                                )
                                continue
                            workspace = _make_workspace(fixture, site, config)
                        _admit(
                            fixture,
                            site,
                            family,
                            config,
                            workspace,
                            f"{family} {site} r{rank} T={num_tokens}",
                        )
                        record = measure(
                            lambda s=site, f=family, c=config, w=workspace: (
                                fixture.run_family(s, f, c, w)
                            ),
                            suite=suite,
                            candidate=_spec(site, family).key(),
                            boundary=BOUNDARY_ISOLATED,
                            params={
                                "case_id": fixture.case.case_id,
                                "T": num_tokens,
                                "rank": rank,
                                "site": site,
                                "family": family,
                                "config": dict(config),
                            },
                            graph_replay=True,
                        )
                        print(
                            f"per_stage r{rank:<4d} T={num_tokens:<5d} "
                            f"{site:8s} {family:10s} "
                            f"{record.median_s * 1e6:8.2f} us"
                        )

    if "sweep" in sections:
        sweep_ts = parse_sweep_axis(arguments.sweep_regimes)
        best: dict = {}
        sweep_skips: list[dict] = []
        for rank in ranks:
            for num_tokens in sweep_ts:
                regime_class = regime_of(num_tokens)
                fixture = build_fixture(rank, num_tokens, SEEDS[0])
                for site in ("gate_up", "down"):
                    for family in FAMILIES:
                        for config in _sweep_grid(family, rank):
                            workspace = None
                            if family == "rank_split":
                                if regime_class not in RANK_SPLIT_REQUIRED_REGIMES:
                                    continue
                                # Squash review (F3): a cap-rejected config
                                # used to vanish without a ledger entry,
                                # contradicting "every skip is persisted".
                                if not _rank_split_ok(fixture, site, config):
                                    sweep_skips.append(
                                        skip_entry(
                                            "resources: rank_split workspace " "cap",
                                            site=site,
                                            rank=rank,
                                            family=family,
                                            regime=regime_class,
                                            config=config_key(config),
                                        )
                                    )
                                    continue
                                workspace = _make_workspace(fixture, site, config)
                            try:
                                _admit(
                                    fixture,
                                    site,
                                    family,
                                    config,
                                    workspace,
                                    f"sweep {family} {config_key(config)} "
                                    f"{site} r{rank} T={num_tokens}",
                                )
                            except Exception as error:
                                # Fail-closed (3rd review): only exact
                                # resource/compiler signatures skip;
                                # numeric admission failures ABORT.
                                reason = skip_reason(error)
                                if reason is None:
                                    raise
                                sweep_skips.append(
                                    skip_entry(
                                        reason,
                                        family=family,
                                        site=site,
                                        rank=rank,
                                        T=num_tokens,
                                        config=config_key(config),
                                    )
                                )
                                continue
                            record = measure(
                                lambda s=site, f=family, c=config, w=workspace: (
                                    fixture.run_family(s, f, c, w)
                                ),
                                suite=suite,
                                candidate=(
                                    f"sweep_{_spec(site, family).key()}_"
                                    f"{config_key(config)}"
                                ),
                                boundary=BOUNDARY_ISOLATED,
                                params={
                                    "case_id": fixture.case.case_id,
                                    "T": num_tokens,
                                    "rank": rank,
                                    "site": site,
                                    "family": family,
                                    "config": dict(config),
                                    "regime_class": regime_class,
                                },
                                graph_replay=True,
                            )
                            key = (site, rank, family, regime_class)
                            if key not in best or record.median_s < best[key][0]:
                                best[key] = (record.median_s, config_key(config))
        for (site, rank, family, regime_class), (median, key) in sorted(best.items()):
            print(
                f"sweep-best {site:8s} r{rank:<4d} {family:10s} "
                f"{regime_class:7s} {key:22s} {median * 1e6:8.2f} us"
            )
        # Preserve the sweep before table publication without making a
        # partial run look like the canonical final artifact. Both the
        # measurement suite and its skip ledger are immutable,
        # content-addressed files; the promoted table binds their digests.
        checkpoint_path, sweep_digest = write_sweep_checkpoint(suite, arguments.output)
        skips_path, skips_digest = write_skip_sidecar(
            arguments.output,
            sweep_skips,
            content_addressed=True,
        )
        print(
            f"{len(suite.records)} records -> {checkpoint_path} "
            f"sha256 {sweep_digest} (pre-publish checkpoint)"
        )
        print(f"sweep skip ledger -> {skips_path} sha256 {skips_digest}")
        if arguments.emit_config_table:
            # 5th review: the digest must be recomputed at PUBLISH time and
            # match the value taken at startup, else a file overlay during
            # the run would label old imported code with a new digest.
            end_kernel_digest = kernel_fingerprint()
            if end_kernel_digest != start_kernel_digest:
                raise RuntimeError(
                    "kernel fingerprint changed mid-run "
                    f"({start_kernel_digest} -> {end_kernel_digest}); the "
                    "source tree was overlaid while measuring — refusing to "
                    "publish a table whose identity is ambiguous"
                )
            table = build_main_table(
                best={k: v[1] for k, v in best.items()},
                arguments=arguments,
                suite=suite,
                kernel_digest=end_kernel_digest,
                sweep_checkpoint_digest=f"sha256:{sweep_digest}",
                sweep_skips_digest=f"sha256:{skips_digest}",
            )
            write_config_table(table, arguments.emit_config_table)
            print(f"B config table -> {arguments.emit_config_table}")

    samples: dict = defaultdict(lambda: defaultdict(list))
    records: dict = defaultdict(lambda: defaultdict(list))
    if "decided" in sections:
        for rank in ranks:
            for num_tokens in T_GRID:
                regime_class = regime_of(num_tokens)
                modes = (True,) if num_tokens <= DECODE_T_MAX else (True, False)
                for seed in SEEDS:
                    fixture = build_fixture(rank, num_tokens, seed)
                    for site in ("gate_up", "down"):
                        arms = {}
                        for family in FAMILIES:
                            # rank_split exists only where its sweep grid is
                            # structurally nonempty, and only at decode
                            # regimes. decode_tiny is included per the
                            # 2nd-verify blocker — tiny T IS the arm's
                            # low-wave niche.
                            if family == "rank_split" and not _rank_split_is_swept(
                                rank, regime_class
                            ):
                                continue
                            if (
                                config_table is not None
                                and family not in config_table[site][str(rank)]
                            ):
                                if family == "rank_split":
                                    continue  # structurally empty sweep grid
                                raise KeyError(
                                    f"table lacks {family} at {site}/r{rank}"
                                )
                            config = tuned_config(site, rank, family, regime_class)
                            workspace = None
                            if family == "rank_split":
                                if int(config.get("SPLIT_K", 2)) * 2 > rank or (
                                    not _rank_split_ok(fixture, site, config)
                                ):
                                    continue
                                workspace = _make_workspace(fixture, site, config)
                            _admit(
                                fixture,
                                site,
                                family,
                                config,
                                workspace,
                                f"decided {family} {site} r{rank} "
                                f"T={num_tokens} s{seed}",
                            )
                            arms[family] = (config, workspace)
                        if "one_launch" in arms:
                            matched_config = arms["one_launch"][0]
                            _admit(
                                fixture,
                                site,
                                MATCHED_ARM,
                                matched_config,
                                None,
                                f"decided {MATCHED_ARM} {site} r{rank} "
                                f"T={num_tokens} s{seed}",
                            )
                            arms[MATCHED_ARM] = (matched_config, None)
                        for graph in modes:
                            mode = "graph" if graph else "eager"
                            for repeat in range(REPEATS):
                                names = (
                                    tuple(arms)
                                    if repeat % 2 == 0
                                    else tuple(arms)[::-1]
                                )
                                for family in names:
                                    config, workspace = arms[family]
                                    record = measure(
                                        lambda s=site, f=family, c=config, w=(
                                            workspace
                                        ): fixture.run_family(s, f, c, w),
                                        suite=suite,
                                        candidate=(
                                            _spec(site, family).key()
                                            + (
                                                "_matched"
                                                if family == MATCHED_ARM
                                                else ""
                                            )
                                        ),
                                        boundary=BOUNDARY_ISOLATED,
                                        params={
                                            "case_id": fixture.case.case_id,
                                            "T": num_tokens,
                                            "rank": rank,
                                            "site": site,
                                            "family": family,
                                            "seed": seed,
                                            "repeat": repeat,
                                            "config": dict(config),
                                            # 13th review: sweeps select
                                            # under hot graph replay, so
                                            # an eager record is a
                                            # CONFIG-TRANSFER observation,
                                            # not an eager-optimal one.
                                            "config_source": "graph_swept",
                                            **table_digests,
                                        },
                                        graph_replay=graph,
                                    )
                                    cell = (site, rank, num_tokens, mode)
                                    samples[cell][family].append(record.median_s)
                                    if graph:
                                        records[cell][family].append(record.record_id)
        # Canonical pairs emitted BY THE PRODUCER (3rd review F1): every
        # headline comparison goes through decide_cell here, never
        # through post-hoc scripts. Defined at MODULE level (8th review)
        # so a registered test can assert the list, not grep for it.
        decided_pairs = list(DECIDED_PAIRS)
        for cell in sorted(samples):
            site, rank, num_tokens, mode = cell
            for arm_a, arm_b in decided_pairs:
                if not samples[cell].get(arm_a) or not samples[cell].get(arm_b):
                    continue
                decision = decide_cell(
                    arm_a=arm_a,
                    samples_a=samples[cell][arm_a],
                    arm_b=arm_b,
                    samples_b=samples[cell][arm_b],
                    boundary_a=BOUNDARY_ISOLATED,
                    boundary_b=BOUNDARY_ISOLATED,
                )
                print(
                    f"decided {mode:5s} {site:8s} r{rank:<4d} "
                    f"T={num_tokens:<5d} {arm_a}/{arm_b:16s} "
                    f"geo(a/b)={decision.geo_a_over_b:.3f} -> "
                    f"{decision.winner or 'tied'}"
                )
        # Ledger: adjacent decided flips along T, graph mode.
        for site in ("gate_up", "down"):
            for rank in ranks:
                for family in FAMILIES[1:]:
                    decisions = {}
                    for num_tokens in T_GRID:
                        cell = (site, rank, num_tokens, "graph")
                        if not samples[cell].get(family):
                            continue
                        decisions[num_tokens] = decide_cell(
                            arm_a="stock",
                            samples_a=samples[cell]["stock"],
                            arm_b=family,
                            samples_b=samples[cell][family],
                            boundary_a=BOUNDARY_ISOLATED,
                            boundary_b=BOUNDARY_ISOLATED,
                        )
                    ts = [t for t in T_GRID if t in decisions]
                    for t_low, t_high in zip(ts, ts[1:]):
                        low, high = decisions[t_low], decisions[t_high]
                        if low.winner and high.winner and low.winner != high.winner:
                            suite.site_crossover(
                                site=f"{site}_b",
                                boundary=BOUNDARY_ISOLATED,
                                candidates=(
                                    _spec(site, "stock").key(),
                                    _spec(site, family).key(),
                                ),
                                axis=f"num_tokens (rank={rank}, site={site})",
                                crossover_location=f"T in ({t_low}, {t_high}]",
                                bracketing_low_record_ids=tuple(
                                    records[(site, rank, t_low, "graph")]["stock"]
                                    + records[(site, rank, t_low, "graph")][family]
                                ),
                                bracketing_high_record_ids=tuple(
                                    records[(site, rank, t_high, "graph")]["stock"]
                                    + records[(site, rank, t_high, "graph")][family]
                                ),
                                cache_state=CACHE_L2_HOT_GRAPH,
                                axis_param="T",
                                workload_params=("rank", "site"),
                            )

    if "leg" in sections:
        LEGS = {
            "grouped_stock": ("grouped", "stock"),
            "grouped_1launch": ("grouped", "one_launch"),
            "grouped_indexedB": ("grouped", "indexed"),
            "all_indexed": ("indexed", "indexed"),
        }
        leg_samples: dict = defaultdict(lambda: defaultdict(list))
        for rank in LEG_RANKS:
            for num_tokens in LEG_T_GRID:
                modes = (True,) if num_tokens <= DECODE_T_MAX else (True, False)
                regime_class = regime_of(num_tokens)
                for seed in SEEDS:
                    fixture = build_fixture(rank, num_tokens, seed)
                    leg = fixture.leg

                    def make_thunk(a_own, b_family):
                        a_specs = {
                            site: LoraAExecutionSpec(site=site, ownership=a_own)
                            for site in ("gate_up", "down")
                        }
                        # 2nd review F2: legs previously ran DEFAULT B
                        # configs, understating every arm; use the
                        # device table when provided.
                        b_configs = {
                            site: tuned_config(site, rank, b_family, regime_class)
                            for site in ("gate_up", "down")
                        }
                        a_config = (
                            INDEXED_DEFAULT_CONFIG
                            if a_own == "indexed"
                            else PROVISIONAL_LAUNCH_CONFIG.lora_a
                        )
                        needs_plan = a_own == "grouped" or b_family != "indexed"
                        # 12th review: grouped/grouped legs used to build a
                        # raw route they never consumed — unused in-thunk
                        # work charged to exactly the control legs.
                        needs_raw = a_own == "indexed" or b_family == "indexed"

                        def thunk():
                            plan = leg.route(ROUTE_ALIGNED) if needs_plan else None
                            raw = leg.route(ROUTE_RAW) if needs_raw else None
                            for site in ("gate_up", "down"):
                                inp, weight, out = leg.site_buffers(site)
                                run_lora_a(
                                    a_specs[site],
                                    input=inp,
                                    weight=weight,
                                    output=out,
                                    routing=raw if a_own == "indexed" else plan,
                                    config=a_config,
                                )
                                bridge, b_weight, destination, offsets = fixture.b_args(
                                    site
                                )
                                run_lora_b(
                                    _spec(site, b_family),
                                    bridge=bridge,
                                    weight=b_weight,
                                    destination=destination,
                                    routing=(raw if b_family == "indexed" else plan),
                                    destination_offsets=offsets,
                                    config=b_configs[site],
                                )

                        return thunk, b_configs

                    # 7th review: a warm invocation proves only that the
                    # composed route/A/B path did not crash. Gate every leg
                    # numerically against the FIRST leg's outputs (all legs
                    # compute the same math by construction) before timing.
                    leg_reference: dict[str, torch.Tensor] = {}
                    for leg_name, (a_own, b_family) in LEGS.items():
                        thunk, b_configs = make_thunk(a_own, b_family)
                        for site in ("gate_up", "down"):
                            _, _, destination, _ = fixture.b_args(site)
                            destination.fill_(53.0)  # poison before the leg
                        thunk()
                        for site in ("gate_up", "down"):
                            _, _, destination, _ = fixture.b_args(site)
                            if site not in leg_reference:
                                leg_reference[site] = destination.clone()
                            else:
                                require_delta_close_chunked(
                                    destination,
                                    leg_reference[site],
                                    gate_dtype=torch.bfloat16,
                                    label=(
                                        f"leg {leg_name} {site} vs "
                                        f"{next(iter(LEGS))} r{rank} "
                                        f"T={num_tokens} s{seed}"
                                    ),
                                )
                        for graph in modes:
                            mode = "graph" if graph else "eager"
                            for repeat in range(REPEATS):
                                record = measure(
                                    thunk,
                                    suite=suite,
                                    candidate=f"leg_{leg_name}",
                                    boundary=BOUNDARY_ROUTE_INCLUSIVE,
                                    params={
                                        "case_id": fixture.case.case_id,
                                        "T": num_tokens,
                                        "rank": rank,
                                        "leg": leg_name,
                                        "seed": seed,
                                        "repeat": repeat,
                                        "b_configs": {
                                            s: config_key(c)
                                            for s, c in b_configs.items()
                                        },
                                        **table_digests,
                                    },
                                    graph_replay=graph,
                                )
                                leg_samples[(rank, num_tokens, mode)][leg_name].append(
                                    record.median_s
                                )
        LEG_PAIRS = [("grouped_stock", name) for name in list(LEGS)[1:]]
        LEG_PAIRS.append(("grouped_1launch", "all_indexed"))
        for cell in sorted(leg_samples):
            for arm_a, leg_name in LEG_PAIRS:
                if not leg_samples[cell].get(arm_a):
                    continue
                decision = decide_cell(
                    arm_a=arm_a,
                    samples_a=leg_samples[cell][arm_a],
                    arm_b=leg_name,
                    samples_b=leg_samples[cell][leg_name],
                    boundary_a=BOUNDARY_ROUTE_INCLUSIVE,
                    boundary_b=BOUNDARY_ROUTE_INCLUSIVE,
                )
                print(
                    f"leg {cell[2]:5s} r{cell[0]:<4d} T={cell[1]:<5d} "
                    f"{arm_a}/{leg_name:16s} "
                    f"geo(a/b)={decision.geo_a_over_b:.3f} -> "
                    f"{decision.winner or 'tied'} [{decision.scope}]"
                )

    digest = write_suite(suite, arguments.output)
    print(f"{len(suite.records)} records -> {arguments.output} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
