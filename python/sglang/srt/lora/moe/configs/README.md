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
built. `plan` carries kernel families, fusion shape, overlap windows, and
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
- Tiles, partly confirmed on B200: the per-expert decode ladder was measured
  against the same five candidate tile sets across r8–r128, and B200 chose
  the same winner as GB300 at every rank with matching magnitudes (using the
  wrong tile costs ~25% at r8 on both dies), which is what the rank rule in
  `gb300.tiles.json` encodes.
- NOT measured on B200: tile values for the other sites and M buckets
  (prefill rows, shared decode, the ≤4/≤16 token buckets) and the optional
  tp{2,4,8} tile-forcing confirmation. Those inherit GB300's numbers.

Provenance and the full axis inventory: the campaign's best-config tables
document, plus `B200_POLICY_ADJUDICATION_20260814.md`.

To onboard a model whose geometry the shipped `domain`/rows do not cover:

```
python benchmark/kernels/lora_moe/tune_lora_config.py --model <path> ...
```

and point `SGLANG_LORA_MOE_CONFIG_DIR` at its output directory; files
there take precedence over the packaged ones, per architecture.
