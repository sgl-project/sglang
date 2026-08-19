# MoE LoRA config files

Two JSON files per device-architecture key space:

- `{arch}.plans.json` — execution-plan rows, loaded by
  `sglang/srt/lora/moe/execution_plan.py` (`load_plans`/`resolve_plans`)
- `{arch}.tiles.json` — per-row launch-tile rules, loaded by
  `sglang/srt/lora/moe/launch_config.py` (`resolve_tiles`)

Architectures:

- `gb300.*` — every SM100-family device (B200, GB200/GB300)
- `h200.*` — SM90
- `default.*` — served when no file covers the architecture; routes
  everything through the conservative serial fallback
- `base_gemm/` — M-bucketed base-GEMM launch tables (separate key space:
  provider × geometry × device, not plan rows; see its README)

Plans file:

```json
{
  "arch": "gb300",
  "domain": {"max_hidden": 4096, "max_local_experts": 512},
  "scenarios": [ { "name", "layout", "phase", "max_rank", "base_gemm_rows",
                   "plan" }, ... ],
  "fallback":  [ ...same row shape, matched when the geometry is outside
                 "domain" or no scenario row matches... ]
}
```

`base_gemm_rows` is how the routed activation rows reach the base GEMM —
`expert_major` for the padded `[E, m_max, K]` per-expert slabs, `route_major`
for one flat buffer of aligned per-expert segments. It is a table value
because the surrounding stages are built for it. WHICH VENDOR implements that
row order is not: `--moe-lora-base-gemm` picks CuteDSL or DeepGEMM at serving
time, defaulting to CuteDSL, and a geometry CuteDSL cannot admit falls back to
DeepGEMM automatically. Do not confuse `base_gemm_rows` with `layout` above:
that one is the ADAPTER weight layout (per-expert or shared-outer).

Plan selection happens ONCE, at weight bind: every selection input (layout,
phase, pool-padded rank, geometry) is a server-lifetime constant, so
nothing plan-shaped remains on the forward path. Rows are matched FIRST-HIT
IN ORDER, so put more specific rows (rank bands) above catch-alls. `layout`
("per_expert"/"shared") and `phase` ("decode"/"prefill") are exact keys
(absent = wildcard); `max_rank` admits ranks up to its bound. Rows are
activation-agnostic: the layer injects its own activation when the plan is
built. That was a deliberate collapse (2026-08-16) and it retired three
measured rows, so a ReLU2 MoE is now served by the SwiGLU winners:

- shared layout, both phases: the old ReLU2 twins were byte-identical to
  their SwiGLU rows, so nothing changed;
- per-expert decode: same plan shape, but the ReLU2 row carried its own tile
  set and now inherits the SwiGLU ladder (which is banded by token count,
  where the ReLU2 row was not);
- per-expert prefill on SM100: materially different — the retired
  `prefill.relu2` ran the expert-major (masked) rows with an early gate/up-A
  window and a late down-A+B window and no scatter, where `prefill.serial`
  runs the route-major (contiguous) rows strictly serially with the scatter
  epilogue.

Both are numerically correct (every provider registers relu2 for the fused
middle); the ReLU2-tuned schedules are simply no longer served. Re-adding
them means re-adding an `activation` predicate to the row model. `plan` carries kernel families, fusion shape, overlap windows, and
route builder (see `build_plan` in execution_plan.py for the field list).
Every served row is validated through the execution-plan contracts at bind
time — a malformed row fails startup, never serves.

Tiles file:

```json
{
  "arch": "gb300",
  "rules": {
    "<plan row name>": [ { "max_rank", "max_tokens", "sites" }, ... ]
  }
}
```

The tile pick is the ONLY per-forward decision: at bind time the rank rules
are filtered against the bound rank, and each forward takes the first rule
whose `max_tokens` admits the batch (a rule without `max_tokens` terminates
the ladder). `sites` holds `MoeLoraLaunchConfig` fields (per-site Triton
tile dicts). A plan row with no rules serves the built-in heuristics.

Route block size (`routing_block_size`):

One aligned route serves every grouped LoRA kernel on a plan row, and its
block is each kernel's row tile — of nothing else. The base GEMMs never read
it (their flat buffer uses their own `m_alignment`, 128 on CuteDSL, and their
own token-width tiles from `base_gemm/`); the fused middle and finalize are
pair-domain through `src2dst`. A padded slot costs masked `tl.dot` lanes
inside whichever LoRA kernel tiles over it, and nothing anywhere else.

Shipped values: decode rows 16 (their measured optimum at 1-16 pairs per
group; also the tensor-core floor), per-expert prefill 32.

There USED to be a second granularity — `gate_up_a_routing_block_size`, a
private gate/up-A list at 64 over a shared 16 — retired 2026-08-19 along
with its dual-granularity fused builder (~500 lines). The full retirement
matrix, three models with per-expert adapters, chunks 8k-32k, two rounds per
arm, prefill throughput vs that split:

| chunk sweep | one block of 16 | one block of 32 |
|---|---|---|
| Qwen3.5-35B 8k/16k/32k (H200, noise <=1.4%) | -2.4..-3.2% | +0.0..+2.0% |
| Inkling-Small 8k/16k (GB300) | -3.1..-5.4% | -0.9..+2.1% |
| Qwen3.5-397B 8k/16k (GB300, noise 0.4%) | -4.5..-5.7% | -1.0..+0.3% |

One block of 32 is within noise of the split everywhere but two cells that
cancel (-1.0% on 397B at 16k, +2.0% on Qwen at 32k), so the split's whole
value was recoverable by picking the right single number. 16 — the old
shared value — is NOT that number: gate/up-A's weight slab (2R x hidden, the
largest K-deep panel of the four kernels) loses 4x its fetch amortization
there. Do not "simplify" this value downward without rerunning the matrix;
the harness is `run_chunk_*.sh` in the campaign records.

Occupancy is what moves the optimum — routed pairs per virtual expert,
`tokens x top_k / (local_experts x live adapter slots)`. A 4096-token
prefill of a 256-expert model with 4 adapters resident is 1024 groups of
~32 pairs, not the thousands the token count suggests. The tuner sweeps
this knob per phase end-to-end (decode scored on output throughput, prefill
on input throughput).

The SHARED-OUTER prefill rows remain the open lead: they run the opposite
regime (4 virtual experts, ~16k pairs each, padding 0.4% of slots) and a
kernel-level sweep on GB300 puts a block of 128 at +19.6% (2k tokens) to
+26.0% (8k) over the 16 they ship, with no padding tax to give it back.

Route builder launch tiles (`fused_align.py`, `joint_routing.py`):

These are module constants, not table entries and not autotuned, because
graph capture wants one launch shape per call site. The route build is three
kernels: count pairs per bucket, plan the blocks, then label blocks and place
pairs. `HIST_BLOCK`/`HIST_WARPS` size the count, `SCAN_CHUNK`/`SCAN_WARPS`
the plan, `EXPAND_BLOCK`/`EXPAND_WARPS` the place.

Swept on GB300 2026-07-25 (64 points over 4 cells), then re-verified
per-kernel on H200/B200/GB300 2026-08-19 (5 cells x 48 configs, profiler GPU
time, noise <=1%). Decision: KEEP ALL SIX. No alternative wins everywhere.
The plan and place tiles are minimax-optimal — every other value regresses
7-18% in some cell — and the count tile has rivals with a ~4% better median
that give back up to 1.9% elsewhere. A per-cell oracle would save 0.1-2.7us
of an 8-60us build, so a per-geometry table buys nothing.

MEASURE THESE WITH THE PROFILER, per kernel. CUDA events around the Python
call cannot see them: 70-90% of that wall time is host-side launch work, and
the first attempt at this sweep measured the shipped config 9.6% FASTER THAN
ITSELF, and one arch's three stages each +13% while their combination went
-3.9%. Per-kernel `self_device_time_total` has a <=1% floor.

`joint_routing.py` inherits the same three values without its own sweep; its
kernels write two count arrays per pass, so the shapes need not transfer.
The JIT id-pass tile in `routing.py` (`block_size = 1024`, a flat map over
pairs) was never swept either. Both are open, both are ~1-3us of exposure.

Cost of one route build, measured on B200 2026-08-19 (16,384 pairs = 2048
tokens at top-8, block 32, profiler GPU time per kernel):

| geometry | count | plan | place | total |
|---|---|---|---|---|
| per-expert 128 experts x 4 adapters (513 buckets) | 4.4 | 2.3 | 4.0 | 10.7us |
| per-expert 256 x 8 (2049 buckets) | 3.0 | 3.8 | 4.3 | 11.1us |
| per-expert 256 x 32 (8193 buckets) | 3.0 | 8.9 | 5.1 | 16.9us |
| shared-outer, 4 adapters (5 buckets) | 4.3 | 2.2 | 4.4 | 10.9us |
| shared-outer, 1 adapter (2 buckets) | 12.6 | 3.1 | 12.4 | 28.1us |

This is paid ONCE PER MoE LAYER per forward, not once per forward — each
layer routes its tokens differently, so the route cannot be reused. Multiply
by the model's MoE layer count before comparing against anything.

Two open leads, both UNCHANGED pending end-to-end evidence:

- The plan kernel is a single thread block, so it grows with bucket count:
  2.3us at 513 buckets, 8.9us at 8193, where it is half the build and 149 of
  the GPU's SMs are idle. A two-pass multi-block scan would fix it. Only
  configs with many adapters resident reach the bad end.
- Atomic contention is harmless until one bucket takes more than about 4,000
  pairs, then the count and place kernels nearly triple. Shared-outer with a
  single live adapter is exactly that case (28.1us vs 10.9us with four
  adapters) and it is a shipping config. A per-thread-block histogram, or
  warp-aggregated slot claims, would fix it for small bucket counts only.

Not a lead, and worth recording so nobody re-derives it: the padding fill's
2D tile is `EXPAND_BLOCK x routing_block_size`, which looks like it should
blow up registers as the block grows. It does not. Blocks 16 through 512 all
compile with zero spills (32 registers, 56 at block 64), and 256 and 512 run
correctly. Whatever limits the route block, it is not this kernel.

Correctness coverage for the fused builder, B200 2026-08-19: differentially
diffed against a torch reference over 21 geometries — per-expert and
shared-outer, blocks 16-128, 513 to 8193 buckets (four scan chunks), all four
ways a pair can be invalid, a single hot bucket, a count landing exactly on a
block boundary, single token, top-1, and a zero-token idle rank. Each case
checks the padded pair count, every block label including the out-of-plan -1
tail, each bucket's real slots as a set plus its exact padding, that the
counters came back zeroed, and that a second call agrees with the first.
Unverified end-to-end; a shared-outer adapter on 397B or Inkling is the
vehicle. A block of 256 exceeds shared memory for these tiles.

A second retired-but-documented lead: which kernel most wants a big block is
set by the model's shape (gate/up-A slab 2R x hidden vs down-A slab R x
intermediate, equal at I = 2H). Our fleet is all I <= H. At a Mixtral-shaped
4096/14336 under production geometry, down-A at 16 wastes 54-65% of what
becomes the layer's largest LoRA kernel (0.7-1.0 ms per 8k chunk) — for such
a model, sweep this knob first and expect a bigger winner.

One caution for anyone re-deriving any of this with a kernel-level sweep:
it measures LoRA-only percentages at whatever occupancy the harness picked,
and both of this file's past mistakes came from that — quoting kernel wins
without LoRA's share of the layer, and measuring at occupancies production
never runs. Rank tiles within one stage with kernel sweeps; decide route
values end-to-end.

Some rules carry byte-identical `sites` ON PURPOSE, and nothing enforces the
pairing — the format is deliberately raw values with no reference/alias
mechanism. As shipped: `decode.per_expert` rules 0+1 share one config
(rank rule and its M≤4 bucket), rules 2+3 share another (the M≤16 bucket and
the rank-32 rule that extends the same tiles to 32 tokens), and the
`fallback.*` rows repeat one config across all three arch files. If you
retune one rule of such a pair, retune its twin in the same edit, or the
"same tiles, wider window" intent silently splits. This is a fact about the
current tuning, not an invariant — a future sweep may legitimately split a
pair, which is why no test pins it. Editing by hand? Read
`benchmark/kernels/lora_moe/README.md` first for the selection-cascade traps
and the measurement protocol.

Both loaders are pydantic models with `extra="forbid"`: a field this build
does not understand aborts startup instead of silently widening a match.
Two annotation fields are declared and ignored by selection, so tuned
tables stay loadable: a row may carry `provenance`, and a plans file may
carry `seeded_for` (both written by the tuner below).

Every value in the shipped files is a measured sweep winner from the
2026-08 best-config campaign (bs 1–32, 4k/1k, mlpb=4, r16/32; Qwen3.5-35B,
Qwen3.5-397B, Inkling-Small on GB300/B200/H200).

`gb300.*` keys the whole SM100 family, so what that claim rests on, exactly:

- Plans, confirmed on B200 (2026-08-14, 14-arm e2e sweep, three models):
  decode B family, split-K, row domain, overlap windows, route builder and
  route PDL all picked the same winner as on GB300.
- Tiles, confirmed on B200 (2026-08-17, 54-arm 4-candidate forcing sweep: the
  per-expert decode row at r16/32/64, the per-expert prefill row and both
  shared rows, on all three models, plus r32 at tp2 and tp8): B200 picks the
  same winner as GB300 in every cell. Forcing the wrong tile costs up to 4.9%
  on the rank rule and 5–26% for the built-in heuristics (the 26% is 397B
  shared-outer at bs1), so the ladder earns its place on both dies, at every TP
  measured.
- The one place both dies wanted something the table did not have: at rank 32
  the M≤16 tiles keep winning above the small-M buckets, where the ladder used
  to fall to the wildcard row. `{"max_rank": 32}` encodes that. Confirmed on
  two geometries: +4.4% at bs32 on 35B (committed table vs old, GB300) and
  +2.0% on Inkling per-expert, each with the cells the rule must not touch
  reading inside ±0.7%; and measured out to the largest batch the decode graph
  captures — +3.5% at 64 tokens, +3.3% at 128 — which is why the rule carries
  no `max_tokens` bound.
- NOT measured on B200: tile values for the ≤4-token bucket, whose sites are
  byte-identical to the rank rule's, so forcing them re-runs the incumbent
  rather than testing it. Those inherit GB300's numbers.

Provenance and the full axis inventory: the campaign's best-config tables
document, plus `B200_POLICY_ADJUDICATION_20260814.md`.

To onboard a model whose geometry the shipped `domain`/rows do not cover:

```
python benchmark/kernels/lora_moe/tune_lora_config.py --model <path> ...
```

and point `SGLANG_LORA_MOE_CONFIG_DIR` at its output directory; files
there take precedence over the packaged ones, per architecture.
