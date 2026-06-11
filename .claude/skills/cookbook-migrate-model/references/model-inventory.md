# Legacy-page inventory — remaining migrations

Loaded on demand by the `cookbook-migrate-model` skill. Per-model quirks for
the 18 in-scope legacy pages still to migrate (Qwen3.5 done — PR #27848).
Surveyed 2026-06-10 from the legacy generators + MDX §5 blocks; **re-verify
against the live files at migration time** — pages keep receiving updates.

Excluded from the migration (stay legacy until the template grows multimodal
support): GLM-4.5V, GLM-4.6V, GLM-OCR, GLM-Glyph, Qwen3-VL, Qwen2.5-VL.
Do NOT touch `qwen3-coder-480b-a35b-deployment.jsx` — referenced by the Ascend
docs and not part of this migration.

## Batch order

1. **Skill smoke (first, solo):** Kimi-K2.6 — smallest mapping distance,
   widest hw coverage, clean 0.5.9 measured data. Validates that an agent can
   execute this skill unaided; feed findings back into the skill.
2. **Batch A (flagships):** GLM-5.1, GLM-5, Kimi-K2.5, MiniMax-M2.5,
   MiniMax-M2.7, Qwen3.6 ("Qwen3.5 minus AMD, zero data" — do right after the
   pilot conventions).
3. **Batch B (mid-gen):** GLM-4.7 (first real use of the gpuCount→strategies +
   SUPPORT-matrix conventions — human spot-check advised), GLM-4.7-Flash,
   Kimi-K2, Qwen3-Next, Qwen3-Coder, Qwen3-Coder-Next.
4. **Batch C (legacy):** GLM-4.6, GLM-4.5, MiniMax-M2, Kimi-Linear, Qwen3
   (largest variant fan-out — needs the codegen helper).

## Per-model quirks

| Page | Legacy dims (radio unless noted) | HW | Entrypoint | Measured blocks / version | Quirks |
|---|---|---|---|---|---|
| GLM-4.5 | quant(bf16/fp8) × strategy **checkbox**(tp req + dp/ep/mtp) × thinking × toolcall | AMD only (mi300x/325x/355x) | `launch_server --model` | **0** (methodology only) / env stated 0.5.6.post1 | zero data → whole matrix yellow, skip benchmarks file; `SGLANG_ENABLE_SPEC_V2=1` prefix on MTP |
| GLM-4.6 | same checkbox shape as 4.5 | h100/h200/b200 + AMD×3 | `launch_server` | 10 / 0.5.6.post1 | "BF16 > 8×H100" error combo → absent cell + tips; known inbound anchor `GLM-4.6V.mdx` → `GLM-4.6#4-2-3-thinking-budget` (add `<a id>` shim) |
| GLM-4.7 | quant(nvfp4/fp8/bf16) × **gpus(2/4/8)** × strategy checkbox × thinking × toolcall, SUPPORT-matrix gated | b200/gb200/h200 + AMD×3 | `launch_server`, mixes `--tp`/`--tp-size` | 8 / 0.5.12 (NV) + 0.5.6.post1 (AMD) | gpuCount → budget-tier strategies; measured best = **B200 TP=2 NVFP4** (≠ generator default — verified cell follows it); per-cell `sglang_version` differs NV vs AMD |
| GLM-4.7-Flash | quant(bf16 only) × strategy checkbox(tp/dp/mtp), tp=1 | h100/h200/b200 | `launch_server` | 10 / 0.5.7 | trivial matrix; b200 adds `--speculative-draft-attention-backend triton` |
| GLM-5 | quant dyn(bf16/fp8/nvfp4) × reasoning × toolcall × dpattention × speculative | h200/b200/h100 + **merged "MI300X/MI325X" chip** + mi355x | `sglang serve` | 4 / `commit 947927bdb` | split the merged AMD chip into two ids (duplicate cells); NVFP4 hides all toggles → single recipe cell; DSA model — candidate for `hisparse` axis ONLY if the legacy page documents it; AMD tilelang DSA flags |
| GLM-5.1 | quant dyn(bf16/fp8) × reasoning × toolcall × dpattention × speculative | h200/b200/**gb300**/h100 + AMD×3 | `sglang serve` | 4 / `commit 947927bdb` | version is a commit hash — record verbatim |
| Kimi-K2 | modelname(**instruct/thinking**) × strategy checkbox(tp/dp/ep) × reasoning × toolcall | h200/b200 + AMD×3 | `python3 launch_server` | 3 / 0.5.6.post1 | variants = instruct/thinking; instruct+reasoning was an error pseudo-cell → parsers chip `hide:{variant:["instruct"]}`; `SGLANG_ROCM_FUSED_DECODE_MLA=0` |
| Kimi-K2.5 | quant(**int4**/nvfp4) × reasoning × toolcall × dpattention × speculative(h200/b300 only) | h200/**b300** + AMD×4 (incl mi350x) | `sglang serve` | 10 / 0.5.6.post2 + 0.5.9 | spec preset must carry `--speculative-draft-model-path lightseekorg/kimi-k2.5-eagle3-mla`, chip-gated h200/b300; **rocm700** (not 720) docker tags — copy verbatim; vision prose carries over; AMD `SGLANG_USE_AITER=1`+MLA env |
| Kimi-K2.6 | reasoning × toolcall × dpattention (no quant dim — native INT4) | h200/b200/b300/gb200/gb300 + AMD×4 | `sglang serve` | 10 / 0.5.9 | single real-precision quant id `int4`; accuracy suites beyond default keys → `accuracyLabels` (needs #27842); vision prose |
| Kimi-Linear | model(instruct) × strategy(tp) × reasoning(→error) × toolcall | AMD only ×3 | `python3 launch_server` | 4 / 0.5.7 | fully degenerate (single variant/quant/strategy); reasoning error → chip disable |
| MiniMax-M2 | modelname(**M2.1/M2** two repos) × strategy(tp) × thinking × toolcall | AMD only ×3 | `sglang serve` | 4 / 0.5.7 | two checkpoints → `variants` |
| MiniMax-M2.5 | **gpuCount**(2 AMD-only / 4 / 8) × thinking × toolcall | h200/b200/**a100**/h100 + AMD×3 | `launch_server` | 10 / 0.5.8 + 0.5.9 | A100 → `config.hardware` nvidia entry; measured B200 8× TP8 EP8 ran with parsers ON — cells still ship WITHOUT parser flags (Playground-only rule); note it in the benchmarks header |
| MiniMax-M2.7 | **gpuCount dyn**(2/4/8; Xeon→TP=6) × thinking × toolcall | h200/b200/gb300/a100/h100 + AMD×3 + **xeon** | `sglang serve` | 3 / 0.5.10.post1 | A100 + Xeon custom hardware; Xeon cells under one tier only |
| Qwen3 | modelsize(**8**) × quant(bf16/fp8) × **category(base/instruct/thinking)** × reasoningParser × toolcall; `hasThinkingVariants` only 235b/30b/4b | b200/h100/h200 + AMD×3 + xeon | `launch_server` | 10 / 0.5.6 | biggest fan-out: variants = deployable checkpoints (`-Instruct-2507`/`-Thinking-2507` suffixed repos); h100-235b-bf16 error combo; needs codegen helper |
| Qwen3.6 | modelSize(35b-a3b/27b) × quant(fp8/bf16) × reasoning × toolcall × speculative × mambaCache | h100/h200/b200 + xeon | `launch_server` | **0** | all yellow; skip benchmarks file; mamba coupling same as Qwen3.5 |
| Qwen3-Next | size(80b) × quant(bf16/fp8) × **thinking = Instruct/Thinking checkpoints** × toolcall × speculative × mambaCache | b200/h200/h100 + AMD×3 + xeon | `launch_server --model` | 4 / 0.5.6 | thinking is a checkpoint variant (→ `variants`), not a flag |
| Qwen3-Coder | modelSize(480b/30b) × quant(bf16/fp8/nvfp4) × toolcall | AMD×3 + b200/gb200 + xeon | `launch_server` | 12 / 0.5.7 | `SGLANG_USE_AITER=0` (OPPOSITE of Qwen3.5 — copy per page, never normalize across the family); 30b has no NVIDIA entries; BF16-on-NVIDIA error combo |
| Qwen3-Coder-Next | quant(bf16/fp8) × toolcall × mambaCache | h200/h100/b200 + AMD×3 + xeon | `launch_server` | 10 / "0.5.8+" | clean; version string verbatim |

## Measured-data reality check

Most cells will be yellow after migration — that is the chosen policy, not a
defect. Inherited recipes that look memory-infeasible by the TP/VRAM formula
(found so far: Qwen3.5 `122b|mi325x|bf16|tp1` and `122b|mi355x|bf16|tp1`) are
kept verbatim, stay yellow, and get listed in the PR body. A separate
re-verification track (devbox benchmarking + Playground Submit ↗) turns cells
green over time; it does not block migrations.
