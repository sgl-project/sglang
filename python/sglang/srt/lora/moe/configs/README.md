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
  everything through the conservative serial fallback. Prefill fallback
  rows stay route-major: the masked slab domain scales as
  `local_experts x chunk tokens x hidden` (tens of GiB at real chunk
  sizes) where route-major scales with routed pairs.
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
row order is not: the `--moe-runner-backend lora_<vendor>` name picks
`cutedsl` (the bf16/fp8 default), `triton` (the A/B arm), or `marlin` (the
nvfp4 default) at serving time; a vendor serving no layers of a weight family
resolves to that family's default, logged. Triton and Marlin gather rows by
the sort metadata and have no masked slab domain, so they run a decode row's
`expert_major` request on their route-major class. A geometry a vendor cannot
admit fails at attach. Do not confuse `base_gemm_rows`
with `layout` above: that one is the ADAPTER weight layout (per-expert or
shared-outer).

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
  window and a late down-A+B window and no into-base epilogue, where the
  per-expert prefill rows run the route-major (contiguous) rows strictly
  serially with down-B adding into the base down output.

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
own token-width tiles from `base_gemm/`); the fused act and finalize are
pair-domain through `pair_to_row`. A padded slot costs masked `tl.dot` lanes
inside whichever LoRA kernel tiles over it, and nothing anywhere else.

Shipped values: decode rows 16 (their measured optimum at 1-16 pairs per
group; also the tensor-core floor), prefill rows 32, except the H200
shared-prefill rank bands, which carry their own winners (16-64 by rank).

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
there. Do not "simplify" this value downward without rerunning the matrix.

Occupancy is what moves the optimum — routed pairs per virtual expert,
`tokens x top_k / (local_experts x live adapter slots)`. A 4096-token
prefill of a 256-expert model with 4 adapters resident is 1024 groups of
~32 pairs, not the thousands the token count suggests. The tuner sweeps
this knob per phase end-to-end (decode scored on output throughput, prefill
on input throughput).

The SHARED-OUTER prefill rows remain the open lead: they run the opposite
regime (4 virtual experts, ~16k pairs each, padding 0.4% of slots) and a
kernel-level sweep on GB300 put a block of 128 at +19.6% (2k tokens) to
+26.0% (8k) over a block of 16, with no padding tax to give it back. The
shipped rows have since moved to 32 (H200's top band to 64); the 128 cell
has not been rerun against them.

Route builder launch tiles (constants in `routing.py`, kernels in
`kernels/routing.py`):

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

The JIT id-pass tile in `routing.py` (`block_size = 1024`, a flat map over
pairs) was never swept; it is ~1-3us of exposure.

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

End-to-end share, measured 2026-08-19 on B200 tp4, Qwen3.5-35B (40 MoE
layers, top-8, 256 experts), one 8192-token prefill chunk = 65,536 pairs,
rank-0 torch profile:

| route family | builder (as then shipped) | per layer | per prefill | share of prefill |
|---|---|---|---|---|
| per-expert (now `prefill.per_expert`) | fused aligned | 30.4us | 1.22ms | 1.5% |
| shared-outer (now `prefill.shared`) | joint (since deleted) | 32.7us | 1.31ms | 1.5% |

Those are after the per-block counting below. Before it they were 43.7us /
1.75ms / 2.4% and 116.4us / 4.66ms / 5.4%.

Wall clock for that change, shared-outer arm, no profiler attached: prefill
73.75 -> 70.53ms median, 4.37% faster. Four interleaved servers off/on/off/on,
8 repeats each, and the two arms' ranges do not overlap (off floor 72.56 vs on
ceiling 71.44), which is the only reason a 4% effect is believable here -- the
same measurement WITH the profiler attached swung 71-100ms on one config.
Removing 3.35ms of kernel time bought 3.22ms of wall clock.

Read that before optimizing the route build. Today a shared-outer plan
builds its two aligned views with the standard builder, the shared one forked
onto the workspace side stream (`parallel_shared_outer`; the joint builder it
replaced is recorded below); a per-expert plan builds one. Per-expert DECODE
builds neither -- 8 pairs is below `FUSED_ALIGN_MIN_PAIRS`, so it runs the JIT
id pass at 1.2us per layer.

BEWARE when A/B-ing adapter counts: at batch size 1 a single request uses a
single adapter, so `max_loras_per_batch` 1 and 4 put every pair on the SAME
counter and measure identically (58.0 vs 58.5us here). That flatness is not
evidence of no contention -- it is evidence the knob did nothing.

Verdicts on the two leads this evidence settles:

- The single-thread-block plan kernel is NOT worth fixing. It is 2.7-3.8us per
  layer in both families, 0.13% of the step. Its 8.9us at 8193 buckets only
  arrives with many adapters resident, which batch-1 serving never reaches.
- Atomic contention WAS worth fixing, and is fixed: `add_counts_inline` and
  `claim_slots_inline` in `kernels/routing.py` count a block's pairs before touching
  global memory, so a block spends one atomic per bucket instead of one per
  pair. Landed per-layer prefill numbers, 8192-token chunk:

| kernel | before | after | |
|---|---|---|---|
| joint count | 58.0us | 7.9us | 7.3x |
| joint plan | 3.8us | 2.9us | 1.3x |
| joint place | 54.6us | 21.9us | 2.5x |
| fused count | 21.4us | 8.0us | 2.7x |
| fused place | 19.6us | 19.6us | no path applies |

  `fused_align`'s place kernel keeps its per-pair claim: handing out individual
  slots across 257 buckets would need 257 running sums per block, which costs
  more than the atomics it would replace. Only the shared route, with a handful
  of buckets, can claim per block.

The per-block paths CANNOT simply replace the per-pair ones -- they pay a fixed
cost per block, so they lose badly on small work, which is why `count_bins` and
`CLAIM_MIN_PAIRS_PER_BUCKET` gate them on the host and both per-pair paths
stay. Measured win factor by pairs and live adapters (above 1.00 means the
per-block path wins):

| pairs | 1 live | 2 | 4 | 8 |
|---|---|---|---|---|
| 65,536 | 4.78x | 2.31x | 1.27x | 1.28x |
| 32,768 | 2.39x | 1.32x | 0.73x | 0.71x |
| 16,384 | 1.28x | 0.72x | 0.44x | 0.45x |
| 4,096 | 0.66x | 0.56x | 0.40x | 0.38x |
| 512 | 0.33x | 0.32x | 0.31x | 0.38x |

The crossover sits near 12,000 pairs per bucket for slot claims. Counting has
its own bound because it wins at far lower occupancy -- 65,536 pairs over 256
buckets is only 256 each and still gains 3.7x -- but stops paying above 512
bins (0.47x at 2048). Both choices are made on the host from the pair and
bucket counts, so each call site keeps one launch shape and graph capture is
unaffected. Do NOT branch inside a kernel on a device value.

What is left: the counting path is bounded at 512 bins, so a per-expert route
with more than 511 buckets -- 256 experts with 2 or more adapter slots -- still
counts one pair at a time. Raising `COUNT_MAX_BINS` needs 1024 measured first;
2048 is known to lose.

MEASURED AND DECLINED, so nobody re-derives it: the plan kernel's `CHUNK` is a
flat 2048 lanes whatever the bucket count, which is 8x wider than the 257
buckets a per-expert route has and 1024x wider than the shared route's 2.
Narrowing it to fit does help that kernel -- at 257 buckets a 512 tile is
1.70us against 2.10us, and at 2 buckets a 16 tile is 1.31us against 1.98us --
but the whole win is 0.4-0.7us per layer, 0.5% of a decode step and 0.02% of
a prefill, well under what an end-to-end A/B can resolve, so the flat 2048
stays. (The numbers were taken on the since-deleted joint kernel, where the
wider route also masked the narrow one's gain.) Above 1024 buckets 2048 is
simply correct.

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

The gate/up A family is NOT a tuner axis, so its prefill/decode split was
argued rather than measured until 2026-08-19. It is measured now, B200,
Qwen3.5-35B shared-outer, one plan row flipped per arm via a config-dir
override, base/variant interleaved over two rounds:

| flip | metric | base | flipped |
|---|---|---|---|
| `decode.shared` -> `token_grouped` | decode tput | 5881 tok/s | 5266, **-10.5%** |
| `prefill.shared` -> `grouped` | prefill ttft | 70.87ms | 73.45ms, **-3.6%** |

Both shipped choices are right, and the control is that each variant moved only
its own metric: the decode flip left prefill at 71.00 vs 70.87ms, the prefill
flip left decode at 5881.8 vs 5881.1 tok/s. Note the decode loss is far larger
than the extra route explains -- that route is only 5.5us per layer, built by
the shared CUDA align because one row per token with one bucket per adapter
falls under both fused thresholds. Flipping the family also forces a token-major
gate/up bridge, and that is where the decode cost actually lives. A family flip
is never just one stage.

Provenance and the full axis inventory live in the 2026-08 campaign
records.

Down-tail sweep, 3 models x 3 architectures x 2 adapter layouts (2026-08-20).
Instrument: one `MoeLoraRunner.run` at each model's real per-rank geometry
(Qwen3.5-35B h2048/i128/E256/k8, Qwen3.5-397B h4096/i256 or i128/E512/k10,
Inkling-Small h4096/i512 or i256/E256/k6), against the row `resolve_plans`
actually selects for that arch and layout -- so an arm differs from production
by exactly one plan field. Wall clock over batches of back-to-back calls, arm
order rotated per repeat, first repeat discarded. Sanity: the measured layer
time x layer count is 43-59% of the runbook's e2e prefill time, so a layer-level
delta is worth roughly half that at e2e prefill.

EAGER results are reported first because they are misleading, and the reason is
the point. Eagerly, the DOWN_A window swings +11% to -1.7% depending on model and
architecture -- Qwen3.5-35B loses 6-11% at <=4096 tokens everywhere, and on GB300
at every token count. None of that survives CUDA-graph capture, which is what
production prefill runs at the `--cuda-graph-bs-prefill` buckets: capture turns
the fork's event record/wait into graph nodes and the overhead disappears.

CAPTURED results, every config captured twice so the graph-to-graph spread is
measured per config rather than assumed (a single capture of an identical plan
drifted ~1%, the size of the effect). `noise` is that spread for the shipped
config; nothing inside it means anything.

| arch | model | layout | tokens | noise | DOWN_A | adopt into-base |
|---|---|---|---:|---:|---:|---:|
| B200 | q35 | shared | 2048 | 0.94% | -2.69% | **-5.31%** |
| B200 | q35 | shared | 8192 | 0.67% | +0.05% | **-6.84%** |
| B200 | q397 | shared | 2048 | 0.01% | -0.67% | **-2.62%** |
| B200 | q397 | shared | 8192 | 0.73% | -0.76% | **-3.73%** |
| B200 | ink | shared | 2048 | 0.11% | -0.06% | **-1.89%** |
| B200 | ink | shared | 8192 | 0.90% | +0.03% | **-3.51%** |
| GB300 | q35 | shared | 2048 | 0.94% | -2.65% | **-6.20%** |
| GB300 | q35 | shared | 8192 | 0.39% | -0.30% | **-6.71%** |
| GB300 | q397 | shared | 2048 | 0.02% | -0.40% | **-1.84%** |
| GB300 | q397 | shared | 8192 | 0.30% | -0.40% | **-3.37%** |
| GB300 | ink | shared | 2048 | 0.00% | -0.62% | **-2.28%** |
| GB300 | ink | shared | 8192 | 0.21% | +0.63% | **-1.79%** |

Two conclusions.

**The DOWN_A window is not worth shipping.** Across 36 captured cells its effect
runs -2.7% to +2.0% with no consistent sign by model, architecture, or token
count, and most cells sit inside their own noise floor. The eligibility rule
admits it so a table *can* ask for it, and the eager numbers explain why an
eager measurement would have shipped it by mistake.

**Shared-outer prefill rows should adopt `down_b_into_base`.** They ship with it
off; turning it on wins 1.8-6.8% of MoE-LoRA layer time in every cell measured,
on both SM100 architectures and all three models, at both prefill graph buckets,
far outside the noise floor. H200's shared rank bands 9-64
(`prefill.shared.rank_le16`/`rank_le64`) finalize through `shared_token_delta`,
so they have no standalone down-B stage and no into-base axis -- those rows
are untouched.

Shared expert-major fallback row (2026-08-20). The row
`fallback.prefill.shared` uses `expert_major` row order
and was not in the sweep above, so it kept the epilogue off. It serves only
out-of-domain geometry, which means hidden above 4096 or more than 512 local
experts. Two real models land there: GLM-5.2 and full Inkling, both hidden
6144. 16 captured cells, each config captured twice:

| arch | model | 1024 | 2048 | 4096 | 8192 |
|---|---|---:|---:|---:|---:|
| GB300 | GLM-5.2 | +1.67% | +1.68% | +1.64% | +1.42% |
| GB300 | Inkling | +1.02% | +1.21% | +1.38% | +0.87% |
| B200 | GLM-5.2 | +1.23% | +1.41% | +0.69% | - |
| B200 | Inkling | +0.34% | +0.66% | +0.87% | - |
| H200 | GLM-5.2 | +2.03% | - | - | - |
| H200 | Inkling | +1.40% | - | - | - |

Here `+` means faster with the epilogue on. All 16 cells are faster. The range
is +0.34% to +2.03%, the median is +1.30%, and the worst per-cell noise floor
is 0.55%. The empty cells are a limit of the harness, not of the row: each arm
holds its own workspace, and one expert-major slab at hidden 6144 is up to
24.8 GB. The row now uses the epilogue in all three tables. The `default`
table cannot reach it in serving, because the runner admits SM90 and SM100
only, but the three tables stay aligned.

The numerics check needed a new bound at this size. The epilogue rounds
`base + delta` to BF16 one time, and the shipped path rounds the delta first.
At hidden 6144 the largest difference was 0.125, which is one BF16 unit at the
largest row value of 18.1. Only 9 elements of 12.6 million passed a fixed
tolerance of 3e-2, and each one is a near-zero output where the base and the
delta cancel. Use a tolerance that scales with the row size, not a fixed one.

H200 shared prefill, rank bands other than 16 (2026-08-20). Every sweep above
used rank 16, which selects a shared-finalize row with no into-base axis.
Two other bands keep a standalone down-B: `prefill.shared.rank_le8` for rank 8
and below, and the unbounded `prefill.shared` above rank 64. 12 captured
cells, each config captured twice:

| rank | row | model | 2048 | 8192 |
|---:|---|---|---:|---:|
| 8 | prefill.shared.rank_le8 | Qwen3.5-35B | +6.63% | +9.76% |
| 8 | prefill.shared.rank_le8 | Qwen3.5-397B | +4.38% | +7.99% |
| 8 | prefill.shared.rank_le8 | Inkling-Small | +4.21% | +6.76% |
| 128 | prefill.shared | Qwen3.5-35B | +2.17% | +2.40% |
| 128 | prefill.shared | Qwen3.5-397B | +1.31% | +2.57% |
| 128 | prefill.shared | Inkling-Small | +1.71% | +2.20% |

All 12 cells are faster, from +1.31% to +9.76%, median +3.39%, worst noise
floor 1.04%. These are the largest gains the epilogue gives anywhere. The gain
falls as the rank rises, which fits: the delta buffer this removes is one row
for each pair whatever the rank, so its cost is a larger share of a small-rank
forward. Both rows now use the epilogue. The shared-finalize bands stay
off, because they own no separate down-B stage.

Also seen, not acted on: on PER-EXPERT rows at the 2048 bucket, removing
into-base measured -2.06% (B200) and -2.03% (GB300) for Qwen3.5-397B, while at
8192 removing it costs +0.85% to +2.99%. That is a token-banded preference the
plan table cannot express -- `max_rank` is its only predicate -- so it is a lead,
not a change.

Route-builder sweep: joint vs parallel-two-stream vs serial (2026-08-20, all
three architectures, shared layout, the five shipped joint rows' shapes).
Same instrument discipline as the down-tail sweep above: shipped row, captured
graphs, every config captured twice, deltas vs the joint arm. 48 cells
(3 models x {decode bs 1/8/16/32, prefill 2048/8192} x 3 archs; H200
contributes decode only -- its rank-16 shared prefill row already uses the
standard builder).

- Serial two builds: clearly worse, +1% to +15% (worst at decode bs1) --
  the joint builder was genuinely earning its keep against serial.
- Parallel two-stream (`parallel_shared_outer`: the two standard aligned
  builds forked on the workspace side stream): matches or beats joint.
  20 wins / 19 washes / 9 losses; every loss <= +1.5%; decode median -0.46%
  (mean -1.01%), prefill median -0.10%. Biggest win: Qwen3.5-397B decode bs1
  on GB300, -7.95% of layer time (noise 0.08%) -- at E=512 the joint fused
  chain drags 2049 buckets of label metadata for a 10-pair forward, while the
  parallel arm's tiny builds fall through the fused-align thresholds to the
  JIT align path. That per-route builder choice is structural to the parallel
  form: each route picks its own builder, which one dual-headed launch cannot.

Content equivalence was proven separately (bucket -> pair-set identical at
1..8192 tokens). Two representation differences are benign by construction:
intra-bucket pair order is atomic-claim nondeterministic even between two runs
of the SAME builder, and the JIT align path lays sentinel blocks first where
the fused path lays them last -- consumers skip `-1` blocks wherever they sit.

e2e sanity at the runbook protocol (B200, shared layout, joint vs parallel,
override proven to flip the resolved builder): decode moved positive in all
eight cells -- Qwen3.5-35B +0.7..+1.3%, Qwen3.5-397B +0.5..+2.6% with the
largest gain at bs1, exactly where the layer-level route-build win is biggest.
Prefill flat within noise. On this evidence every shipped joint row moved to
`parallel_shared_outer`, and the dual-headed joint machinery was deleted from
the route kernels: `_hist`/`_scan`/`_place` are single-route (the constexpr
NEED branches compiled to exactly these instantiations before, verified by a
post-deletion timing spot check matching to the microsecond), and
`_build_aligned` builds one route per call.

Route builder on the H200 prefill rows (2026-08-20, added after the flip). The
48-cell sweep above reached one prefill row only, SM100's `prefill.shared`,
where parallel against joint was a wash: 12 cells, median +0.10%, range -0.87%
to +1.70%. It reached no H200 prefill row at all, because that sweep used rank
16, and rank 16 on H200 selects a shared-finalize row that never used the
joint builder. So two H200 prefill rows changed builder with no measurement. This
closes that gap. Joint is deleted, so the comparison is parallel against
serial:

| rank | row | median | range |
|---:|---|---:|---:|
| 8 | prefill.shared.rank_le8 | +0.80% | +0.26% .. +1.22% |
| 128 | prefill.shared | +0.90% | -0.18% .. +0.99% |

Here `+` means parallel is faster. Parallel wins 11 of 12 cells. The one loss
is -0.18%, inside that cell's 0.54% noise floor.

The honest summary for prefill: parallel matches joint, and it beats serial by
about 1%. Decode is where parallel gains over joint. Prefill keeps parallel
because joint no longer exists and serial is worse, not because parallel is
faster than what it replaced.

A general lesson from this sweep and the into-base sweep: rank 16 selects a
different row from rank 8 or rank 128 on H200. A sweep at one rank does not
cover a table whose rows carry `max_rank`.

To onboard a model whose geometry the shipped `domain`/rows do not cover:

```
python benchmark/kernels/lora_moe/tune_lora_config.py --model-path <path> --emit-seed --out <dir>
```

and point `SGLANG_LORA_MOE_CONFIG_DIR` at that directory; files there take
precedence over the packaged ones, per architecture. The seed only reuses the
existing plans for the wider geometry: validate provider admission and
correctness on that geometry before serving it, then benchmark it with the
campaign protocol below. `--check` reports rows for one `--quant-family`.

The finalize family names in the evidence below are the current ones:
`shared_token_delta` was called `shared_token_gemm` and `shared_one_pass` was
called `shared_mapped_reduce` while these sweeps ran (renamed 2026-09-03).

Shared prefill finalize, rank 16-32 (2026-09-03). Which finalize should the
shared-outer prefill rows use? Four arms on the same servers: prefill of 4096
tokens per request at batch 32/16/8/1, two rounds each with the arm order
reversed. Decode throughput and the greedy probes did not change in any cell.
Columns where one control round hit a one-off outlier are left out.

H200, change in TTFT against the then-shipped `shared_rank_reduce` row
(`prefill.shared.rank_le16`); negative is faster:

| model | shared_token_delta | shared_one_pass | materialized + into_base |
|---|---:|---:|---:|
| Qwen3.5-35B bf16 | -2.2% .. -2.8% | -1.5% .. -1.8% | +3.7% .. +6.4% |
| Qwen3.5-35B FP8 | -2.6% .. -2.9% | -1.6% .. -2.1% | +2.1% .. +6.4% |
| Inkling-Small NVFP4 (marlin) | -2.8% .. -3.5% | -0.6% .. -1.7% | +1.8% .. +4.0% |
| Qwen3.5-397B FP8, tp 8 | -3.1% .. -3.2% | -0.1% .. -0.4% | +4.9% .. +9.3% |

GB300, change in TTFT against the shipped `shared_token_delta` row
(`prefill.shared.rank_le32`):

| model | shared_rank_reduce | shared_one_pass | materialized + into_base |
|---|---:|---:|---:|
| Qwen3.5-35B FP8, rank 16 | +0.6% .. +2.7% | +1.0% .. +3.4% | +1.5% .. +4.4% |
| Inkling-Small NVFP4, rank 32 | +1.4% .. +1.6% | +2.2% .. +2.5% | +1.4% .. +2.7% |
| Qwen3.5-397B NVFP4, rank 16 | +0.1% .. +1.4% | +0.3% .. +2.0% | +1.7% .. +3.6% |

`shared_token_delta` wins every cell on both architectures. The H200 rank 9-16
row moved to it and `shared_rank_reduce` was deleted: its tail re-read the
shared down-B once per (token, tile), and nothing else selected it. The
rank-reduce kernel stays as the first stage of `shared_token_delta`, and
`shared_one_pass` stays for the gb300 NVFP4 decode row, which this sweep
did not measure. Numerics: all three shared finalizes sit within 8e-4 of an
fp32 reference on both tables' prefill rows with the cutedsl and triton row
orders.

Shared decode finalize (2026-09-03). The same four finalizes on the shared
decode rows, decode-heavy requests of 128 input and 256 output tokens at batch
64/32/8/1, two rounds each with the arm order reversed. Rounds reproduce to
within 0.1%, so decode is a far cleaner measurement than prefill. Change in
decode tokens per second; positive is faster.

H200 `decode.shared`, against the then-shipped materialized finalize:

| model | shared_rank_reduce | shared_token_delta | shared_one_pass |
|---|---:|---:|---:|
| Qwen3.5-35B bf16 | -0.5% .. +0.9% | -7.7% .. -15.7% | +1.5% .. +2.4% |
| Qwen3.5-35B FP8 | -0.7% .. +0.4% | -8.0% .. -16.2% | +1.4% .. +2.2% |
| Inkling-Small NVFP4 (marlin) | -0.7% .. +0.2% | -5.8% .. -10.6% | -0.1% .. +0.6% |
| Qwen3.5-397B FP8, tp 8 | -0.2% .. +0.3% | -3.4% .. -13.4% | +0.5% .. +2.4% |

GB300, against the shipped row of each cell:

| model (row, shipped finalize) | shared_rank_reduce | shared_token_delta | shared_one_pass | materialized |
|---|---:|---:|---:|---:|
| Qwen3.5-35B FP8 (`decode.shared`, materialized) | -1.0% .. -1.5% | -6.0% .. -14.0% | +0.7% .. +2.3% | repeat, within 0.1% |
| Inkling-Small NVFP4 (`decode.shared.nvfp4`, mapped) | -0.1% .. -1.3% | -3.5% .. -11.7% | repeat, within 1% | -1.7% .. -5.2% |
| Qwen3.5-397B NVFP4 (`decode.shared.nvfp4`, mapped) | -0.5% .. -1.9% | -4.9% .. -17.8% | repeat, within 0.2% | -4.9% .. -8.0% |
| Qwen3.5-397B FP8, tp 2 (`decode.shared.fp8.e_ge512`, materialized) | not run | not run | +0.0% .. +1.4% | repeat |

Decode inverts the prefill result. `shared_token_delta` pays its extra GEMM
and token-delta pass on every step and loses 3-18%. `shared_one_pass`
is best or tied in all eight cells, so every shared decode row now uses it:
`decode.shared` on both tables and `decode.shared.fp8.e_ge512` moved to it,
and the NVFP4 decode row already had it (materialized would cost that row
2-8%). The one-pass finalize owns down-B, which leaves only down-A to overlap,
so those rows keep `down_a`. The 512-expert FP8 cell ran at tp 2, where the
hybrid state buffers cap running requests at 43, so it has no batch-64 point.

Shared prefill finalize, every rank band (2026-09-03). The rank 16-32 sweep
above left the rank 8 band and the bands above 32 (GB300) and 64 (H200) on
materialized + into_base. This sweep ran the same arms at ranks 8, 64, 128 and
256 with synthetic shared-outer adapters (N(0, 0.01) factors, the same recipe
as the rank 16 ones), prefill 4096 tokens per request at batch 32/16/8/1, two
rounds each. Change in TTFT against the shipped row; negative is faster.

| arch | model | rank | row (shipped finalize) | shared_token_delta | shared_one_pass | materialized + into_base |
|---|---|---:|---|---:|---:|---:|
| H200 | Qwen3.5-35B bf16 | 8 | `prefill.shared.rank_le8` (materialized) | -6.2% .. -6.5% (bs1 -2.7%) | +5.5% .. +9.6% | shipped |
| H200 | Inkling-Small NVFP4 | 8 | `prefill.shared.rank_le8` (materialized) | -4.3% .. -4.6% (bs1 -3.1%) | +7.4% .. +11.0% | shipped |
| H200 | Qwen3.5-397B FP8 | 8 | `prefill.shared.rank_le8` (materialized) | -9.0% .. -9.2% (bs1 -4.3%) | +4.4% .. +9.7% | shipped |
| H200 | Qwen3.5-35B bf16 | 64 | `prefill.shared.rank_le64` (token_delta) | shipped | +6.5% .. +9.3% | +6.7% .. +11.8% |
| H200 | Qwen3.5-397B FP8 | 64 | `prefill.shared.rank_le64` (token_delta) | shipped | +8.4% .. +9.5% | +11.3% .. +17.3% |
| H200 | Qwen3.5-35B bf16 | 128 | `prefill.shared` (materialized) | -5.7% .. -6.1% (bs1 -4.0%) | +10.7% .. +14.8% | shipped |
| H200 | Inkling-Small NVFP4 | 128 | `prefill.shared` (materialized) | -3.8% .. -4.1% (bs1 -2.7%) | +13.1% .. +18.2% | shipped |
| H200 | Qwen3.5-397B FP8 | 128 | `prefill.shared` (materialized) | -8.1% .. -8.2% (bs1 -4.8%) | +12.0% .. +17.6% | shipped |
| H200 | Qwen3.5-35B bf16 | 256 | `prefill.shared` (materialized) | -5.3% .. -5.6% (bs1 -3.3%) | +1321% .. +1591% | shipped |
| GB300 | Qwen3.5-35B FP8 | 8 | `prefill.shared.rank_le32` (token_delta) | shipped | +9.2% .. +12.0% | +5.1% .. +6.2% (bs1 +0.9%) |
| GB300 | Inkling-Small NVFP4 | 8 | `prefill.shared.rank_le32` (token_delta) | shipped | +6.9% .. +7.1% | +2.2% .. +2.4% (bs1 +0.7%) |
| GB300 | Qwen3.5-35B FP8 | 64 | `prefill.shared` (materialized) | -3.3% .. -4.3% (bs1 -0.3%) | +0.4% .. +2.8% | shipped |
| GB300 | Inkling-Small NVFP4 | 64 | `prefill.shared` (materialized) | -2.2% .. -2.7% (bs1 -1.2%) | +1.6% .. +3.8% | shipped |
| GB300 | Qwen3.5-35B FP8 | 128 | `prefill.shared` (materialized) | -3.5% .. -5.7% (bs1 -2.6%) | +6.2% .. +11.0% | shipped |
| GB300 | Qwen3.5-35B FP8 | 256 | `prefill.shared` (materialized) | -5.9% .. -6.1% (bs1 -3.5%) | +618% .. +788% | shipped |

`shared_token_delta` wins every band on both architectures, so every in-domain
shared prefill row now finalizes through it: `prefill.shared.rank_le8` and
`prefill.shared` on H200 and `prefill.shared` on GB300 moved, and they lose
their standalone down-B stage and into-base axis with it. The out-of-domain
fallback rows follow the same evidence rather than the pre-sweep default:
`fallback.prefill.shared` finalizes through `shared_token_delta` in all three
tables, and a `fallback.decode.shared` row ahead of the layout-agnostic
`fallback.decode` gives shared-outer decode `shared_one_pass` there too
(the per-expert fallbacks keep materialized, the only finalize a per-expert
adapter can use). The `fallback.decode.shared` tile rule copies
`decode.shared` on H200 and GB300 (same plan, phase and layout; only the
domain differs) and `fallback.decode` plus the built-in one-pass tile in the
default table, so no shipped row runs on the built-in defaults unnoticed. The one-pass prefill numbers above rank 16 are confounded: the
prefill rows' `shared_token_delta.tail` tile is 1024 wide with 8 warps, tuned
for the token-delta tail, and at the time both families read one shared tile
section, so the one-pass kernel ran under it (each family now has its own
section, `shared_token_delta` and `shared_one_pass`); so at rank 256 it
spills a 1024 x 256 B tile (the kernel alone runs at 1.5 ms for 8192 tokens
with a 128-wide tile). No prefill row ships it, and the decode rows use the
128-wide tile.

Shared decode finalize at the rank extremes (2026-09-03). The decode sweep
above ran at rank 16. Same protocol at ranks 8, 128 and 256 on Qwen3.5-35B;
change in decode tokens/s of materialized against the shipped one-pass finalize:

| arch | rank | materialized vs shipped `shared_one_pass` |
|---|---:|---:|
| H200 bf16 | 8 | -1.6% .. -3.0% |
| H200 bf16 | 128 | -0.7% .. +0.1% |
| H200 bf16 | 256 | -0.2% .. -0.7% (bs8 +0.9%) |
| GB300 FP8 | 8 | -0.5% .. -2.5% |
| GB300 FP8 | 128 | -0.2% .. -0.7% |
| GB300 FP8 | 256 | -0.7% .. -1.4% (bs1 +0.1%) |

The one-pass finalize keeps its lead at rank 8 and ties at 128 and 256, so the
decode rows stay on it across the whole rank range.

Token route from the request segments (2026-09-03). The shared-outer token
route, one row per token grouped by adapter slot, used to be built like the
pair routes: a histogram over tokens keyed by slot, a scan, and a placement
pass, three kernels plus a masking pass, every forward. That is a sort of
something already sorted: a request's tokens are contiguous in the batch and
share one slot, and the batch carries the request boundaries (`seg_indptr`).
One program now walks the requests and writes each one's tokens in place,
padded to whole blocks, keyed by its slot; a request without an adapter adds
no block. Same route contract, same consumers (`token_grouped` A and the
`shared_token_delta` finalize). Tokens whose experts all live on other ranks
are no longer dropped from the route; their rows are computed and never read.

TTFT with the segment route against the histogram route, prefill 4096 tokens
per request, bs 32/16/8/1, two rounds each, H200, rank 16:

| model | bs 32 | bs 16 | bs 8 | bs 1 |
|---|---:|---:|---:|---:|
| Qwen3.5-35B bf16 | -1.4% | -1.4% | -1.1% | -1.2% |
| Inkling-Small NVFP4 | -0.7% | -1.1% | -2.1% | -0.5% |

Per-pair inner product on the NVFP4 decode row (2026-09-03). The gb300 NVFP4
decode row runs `token_dense` A, `per_pair` B over its slot planes, `per_pair`
down-A and `shared_one_pass`. A variant of both per-pair kernels on
`tl.dot`, a 16-row tile holding one live row, was measured against the shipped
`tl.sum` kernels on that row (128 input, 256 output tokens, bs 64/32/8/1, two
rounds):

| model | tl.sum vs tl.dot variant, decode tok/s |
|---|---:|
| Inkling-Small NVFP4, rank 32 | +0.5% .. +1.7% |
| Qwen3.5-397B NVFP4, rank 16 | -0.2% .. +0.3% |

The `tl.sum` kernels win or tie, so no dot variant ships: one kernel per site
serves every raw route.

Base GEMM orientation (2026-09-04). The CuTeDSL grouped GEMMs run one
orientation: the weight on the MMA M axis and the tokens on the N axis, which
lets the token tile go down to 8 wide (`GroupedGemmConfig.mma_tiler_mn =
(128, token_width)`). The other orientation, tokens on M, was carried as a
second bf16 wrapper and a `swap_ab` flag but never shipped and had no
benchmark driver in the tree, so it was removed. The evidence at removal:

- Decode: a tile floor, not a measurement. With tokens on M the MMA M tile is
  64 or 128 rows, so an expert holding three tokens pads to 64. Tokens on N
  take an 8-wide tile.
- Prefill: one proxy micro-bench from the 2026-08 quant campaign, tokens on N
  62.4 us against tokens on M 63.6 us, a tie within noise. Never re-run end
  to end. The contiguous prefill kernel implements only the tokens-on-N
  orientation: its segment fold adds a segment base to the token tile index
  on the scheduler's M axis.
- Hopper: the wrapper was kept as a fallback in case the 8-wide swapped WGMMA
  path failed on SM90. It ships.

To re-test tokens on M in a later tuning pass: branch
`sgl-lora-v2-tokens-on-m-orientation` (commit 184981b256) holds the last tree
with the wrapper (`grouped_gemm` in `kernels/cutedsl/api.py`, the `swap_ab`
config field, and the `prepare_masked_bf16` branch that picked shapes and output
major by it). The kernels still accept `swap_ab=False` for the masked row
domain, so restoring the host side is enough to sweep decode; a prefill
comparison would first need the contiguous segment fold written for the
token-on-M axis. Expect decode to lose by the tile floor; the open question is
only whether a wide prefill tile prefers tokens on M by more than the 2% seen.
