# MoE LoRA execution plans

`<arch>.plans.json` selects the base row layout and LoRA execution stages for
`lora_triton`, `lora_cutedsl` and `lora_marlin`. Capability major 9 selects
`sm90`; major 10 or above selects `sm100`. A missing architecture table uses
`default.plans.json`.

## Selection and overrides

`SGLANG_LORA_MOE_CONFIG_DIR` supplies override files before packaged files.
Tables are process-cached; use a fresh server to evaluate an override.

The file contains a tuned `domain`, ordered `scenarios`, and conservative
`fallback` rows. Scenario selection matches adapter layout, phase, pool rank,
resident expert count and quantization. Shapes outside the tuned domain use
fallback rows. Row names are labels, not selectors.

Each row declares:

- `base_gemm_rows`: `expert_major` or `route_major`;
- `plan`: A/B, activation and finalization families plus overlap and routing;
- `tiles`: ordered rank/token rules containing per-site launch configurations.

Rank filters are applied before the first matching token bound is selected.
An unbounded token rule ends that rank's ladder. `{ "sites": {} }` explicitly
uses launch defaults; a supplied site section must be complete. Invalid keys,
tiles or shadowed ladders fail at loading, even in an unselected row.

The plan determines which route views and buffers are consumed. Per-expert
adapters have factors for each resident expert; shared-outer adapters share
the gate/up A and down B factors across experts. Raw views retain token/expert
pairs. Aligned views bucket and pad pairs for grouped kernels. Shared-token
plans instead group one row per token. Route construction must not create
views the selected stages do not need.

Runtime orchestration remains in `srt/lora`: MoE plans, runner and base-GEMM
providers under `srt/lora/moe`, the shared workspace in `srt/lora/workspace.py`.
Device kernels live in `kernels/ops/lora/common` for
shared routing/A/B operations and `kernels/ops/lora/moe` for MoE stages and
CuTeDSL kernels. Existing upstream LoRA registry and dense token kernels keep
their paths.

## Tuning and evidence

Use the [MoE tuner](../../../../../../benchmark/kernels/lora_moe/README.md)
to search bounded launch tiles while retaining plan families and base-GEMM
configurations. It reports whole-runner fixture timings, not model TPS.
The [base-GEMM store](base_gemm/README.md) has a separate schema and admission
policy.

New production rows need numerical checks, independent locked validation and
model-level target/control measurements on their exact device and affected
selector regions. Do not copy isolated kernel winners into broader rules or
treat aggregate gains as proof that every cell is regression-free.

Historical exploration and measurements remain in Git history; private
campaign paths and retired tools are not part of the public tuning contract.
