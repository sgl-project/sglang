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

Route block size (`routing_block_size`, `gate_up_a_routing_block_size`):

The route's block size IS `BLOCK_SIZE_M` for every LoRA stage that reads it,
and the shared route serves down-A, gate/up-B and down-B. Gate/up-A may take
a SECOND route at its own block, and on the shipped rows it does. Both values
are measured, and they are NOT the same knob:

- the route block is the ROW TILE of the LoRA kernels that ride the route,
  and of nothing else. The base GEMMs never read it: they lay out their flat
  buffer at their own segment alignment (`m_alignment`, 128 on CuteDSL —
  DeepGEMM's contiguous m-alignment) and tile it with their own token widths
  (8/64/128) from the `base_gemm/` tables. The fused middle and the finalize
  address rows through `src2dst`, pair-domain. No activation buffer's size
  depends on the route block; only the route's own int32 slot list grows
  (316 KB at 16 vs 508 KB at 64 for an 8k-token chunk).
- a padded slot is paid INSIDE whichever LoRA kernel tiles over it: a masked
  `tl.dot` lane still burns its full K-deep tile FLOPs. So each block trades
  weight refetches (fewer when bigger) against masked lanes (more), privately
  per kernel.
- the shared route's riders (down-A, down-B, standalone gate/up-B) carry
  weight panels of 2-32 KB, so a bigger block saves them little, while their
  masked lanes scale with each kernel's K x N. Measured end-to-end, 16 -> 64
  is ~0 on Qwen and negative on Inkling, whose wider rows make every masked
  lane dearer. The floor — 16, the tensor-core minimum — wins.
- gate/up-A's panel is 128 KB (K = hidden), so at prefill occupancy the
  refetch savings dominate its masked lanes: 64 wins. It writes by ORIGINAL
  PAIR ID, so the shared route consumes its bridge with no conversion, and
  past 16k pairs (any real prefill chunk) the fused dual builder emits both
  granularities in one hist/scan/expand pass — the second route is nearly
  free.

That asymmetry is why the shipped `prefill.serial` rows run 16 on the shared
route and 64 for gate/up-A, and why collapsing them onto one block is a
regression in both directions. Measured end-to-end 2026-08-18 on H200 with
Qwen3.5-35B (4k in / 1k out, bs 1-32, two rounds per arm, noise floor ±0.3%),
against that shipped split:

| one block for everything | bs1 | bs8 | bs16 | bs32 |
|---|---|---|---|---|
| 16  | −0.1% | −2.3% | −2.6% | −2.5% |
| 32  | +0.1% | −0.1% | +0.2% | +0.0% |
| 64  | +0.2% | +0.3% | +0.2% | +0.4% |
| 256 | −21.3% | −28.0% | −28.6% | −28.6% |

Uniform 64 looks free there, but it is not: on Inkling-Small, whose wider
rows (hidden 4096 vs Qwen's 2048) make every masked lane in its LoRA kernels
cost more, uniform 64 loses 1.0-4.4% on B200 and 4.8-8.5% on GB300 (bs 1-16, one round
per arm, decode controls flat; bs32 flat). The shipped split is the best
configuration measured on both models.

Occupancy is what sets the shared block -- routed pairs per virtual expert,
`tokens × top_k ÷ (local_experts × live adapter slots)`. Do not estimate it
from the token count alone: a 4096-token prefill of a 256-expert model with 4
adapters resident is 1024 virtual experts and only ~32 pairs each, not the
thousands a token count suggests. At that occupancy padding dominates and the
shared block wants to stay small; the 256 row above is that same effect taken
to its conclusion.

The SHARED-OUTER prefill rows are the opposite regime and are NOT tuned for
it. Shared-outer collapses every routed id onto one LoRA expert per adapter,
so the same forward has 4 virtual experts instead of 1024 and ~16k pairs in
each -- padding costs 0.4% of rows there, against ~98% for the per-expert
rows at a block of 64. A kernel-level sweep on GB300 (down-A, gate/up-B,
down-B at the token_dedup tiles) puts a block of 128 at **+19.6% at 2048
tokens and +26.0% at 8192** over the 16 those rows ship, with no padding
penalty to give it back. That is unverified end-to-end and the rows still
ship 16; it is the open lead here, and Inkling or 397B with a shared-outer
adapter is the vehicle for it. A block of 256 does not compile for these
tiles -- it exceeds shared memory.

One caution for anyone re-deriving this. A kernel-level sweep that times only
the four LoRA stages gets the route STRUCTURE wrong in both directions, while
looking rigorous:

- its geometry lied. An E=32 harness (128 virtual experts) put the measured
  cells at 128-512 pairs per group; production is ~32-128 over 1024 groups,
  where a big block's masked lanes eat most of what its fewer weight fetches
  save. Its +15-21% for one block of 64 was real at its own occupancy and
  converts to ~0 on Qwen / negative on Inkling at production's — and a
  LoRA-only percentage must be scaled by LoRA's share of the layer before it
  means anything end-to-end;
- it cannot price the split's second route. Charged as a standalone build it
  scored one block of 16 as ~10% better than the shipped split at 1k tokens,
  when end-to-end that collapse LOSES 2.5% -- in production the fused dual
  builder makes both routes in one pass, so the sweep was billing the split
  for a cost it does not pay. An E=32 harness geometry (128 virtual experts
  against production's 1024) does not represent gate/up-A's economics either.

Use kernel sweeps to rank tiles within one stage at a fixed route. Decide
route structure -- how many routes, which stage reads which -- end-to-end.

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
