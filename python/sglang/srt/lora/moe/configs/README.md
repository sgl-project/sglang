# MoE LoRA config files

One JSON file per device-architecture key space, consumed by
`sglang/srt/lora/moe/config.py`:

- `gb300.json` — every SM100-family device (B200, GB200/GB300)
- `h200.json` — SM90
- `default.json` — served when no file covers the architecture; routes
  everything through the conservative serial fallback
- `base_gemm/` — M-bucketed base-GEMM launch tables (separate key space:
  provider × geometry × device, not scenario rows; see its README)

Each file:

```json
{
  "arch": "gb300",
  "domain": {"max_hidden": 4096, "max_local_experts": 512},
  "scenarios": [ { "name", "when", "provider", "plan", "config" }, ... ],
  "fallback":  [ ...same row shape, matched when the geometry is outside
                 "domain" or no scenario row matches... ]
}
```

Rows are matched FIRST-HIT IN ORDER, so put more specific predicates
(token/rank tiers) above catch-alls. `when` supports: `layout`
("per_expert"/"shared"), `activation`, `phase` ("decode"/"prefill"),
`max_tokens`, `max_rank`, `min_local_experts`; absent keys are wildcards.
`plan` carries kernel families, fusion shape, overlap windows, and route
builder (see `_build_plan` in config.py for the field list); `config`
carries the launch tiles. Every row is validated through the execution-plan
contracts at load time — a malformed row fails startup, never serves.

Every value in the shipped files is a measured sweep winner from the
2026-08 best-config campaign (bs 1–32, 4k/1k, mlpb=4, r16/32; Qwen3.5-35B,
Qwen3.5-397B, Inkling-Small on GB300/B200/H200). gb300.json's claim over
the whole SM100 family was re-validated on B200 (2026-08-14, 14-arm e2e
sweep): every swept axis — decode B family, split-K, row domain, overlap
windows, route builder, route PDL — confirmed the shipped winner.
Provenance and the full axis inventory: the campaign's best-config tables
document.

To onboard a model whose geometry the shipped `domain`/rows do not cover:

```
python benchmark/kernels/lora_moe/tune_lora_config.py --model <path> ...
```

and point `SGLANG_LORA_MOE_CONFIG_DIR` at its output directory; files
there take precedence over the packaged ones, per architecture.
