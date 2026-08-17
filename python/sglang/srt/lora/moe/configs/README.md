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
  "scenarios": [ { "name", "layout", "phase", "max_rank", "provider",
                   "plan" }, ... ],
  "fallback":  [ ...same row shape, matched when the geometry is outside
                 "domain" or no scenario row matches... ]
}
```

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
  `prefill.relu2` ran the masked `cutedsl` provider with an early gate/up-A
  window and a late down-A+B window and no scatter, where `prefill.serial`
  runs `cutedsl_contiguous` strictly serially with the scatter epilogue.

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
  on the rank rule and 5–26% for the built-in heuristics — the more local
  experts, the more the table is worth — so the ladder earns its place on both
  dies, at every TP measured.
- The one place both dies wanted something the table did not have: at rank 32
  the M≤16 tiles keep winning up to 32 tokens, where the ladder used to fall
  to the wildcard row. `{"max_rank": 32, "max_tokens": 32}` encodes exactly
  that measured window — above 32 tokens at that rank nothing was measured,
  so it still serves the wildcard row. Confirmed on two geometries: +4.4% at
  bs32 on 35B (committed table vs old, GB300) and +2.0% on Inkling per-expert,
  each with the cells the rule must not touch reading inside ±0.7%.
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
