# Legacy → config-driven dimension mapping

Loaded on demand by the `cookbook-migrate-model` skill. How to translate a
legacy generator's option space into the 5-dim matrix + Playground axes.
Field schemas live in `../../cookbook-add-model/references/authoring-reference.md`;
this file is about the *mapping decisions*.

## 1. Legacy control → new home

| Legacy control | New home | Rule |
|---|---|---|
| hardware radio | `match.hw` | Catalog ids as-is. Off-catalog hardware → `config.hardware` entry — e.g. A100 `{id:"a100", label:"A100", vram:"80GB", vendor:"nvidia"}` (merges into the NVIDIA row), Xeon `{id:"xeon", label:"Xeon", vram:"host RAM", vendor:"intel"}` (engine renders a new INTEL row; any vendor key works). A merged chip like GLM-5's "MI300X/MI325X" splits into two ids with duplicated cells (cells are denormalized by design). |
| model-size / model-name radio | `variants` | One variant per deployable checkpoint family; single `{id:"default"}` when there's no variant axis (then `modelNames` keys drop the variant half). |
| quantization radio | `quantizations` | Real precision ids (`bf16`/`fp8`/`fp4`/`int4`/…). One `fp4` id even when checkpoints differ per vendor — route via `hw\|variant\|quant` triple keys in `modelNames` (NVFP4 on Blackwell vs AMD MXFP4 is the precedent); per-hw greying falls out of which cells exist. |
| toggle that **couples** with other parts of the command (changes TP/mem/EP) | `strategies` | The Playground applies pure flag diffs — it cannot do coupled changes. Example: Qwen3.5's MTP toggle bumps TP on three H100 combos → strategies `low-latency` (MTP on) / `high-throughput` (MTP off). GPU-count radios (GLM-4.7, MiniMax-M2.5/2.7) → budget-tier strategies with the legacy SUPPORT matrix preserved by which cells exist. Degenerate single-entry `strategies` is legal and renders as one chip. |
| toggle that only adds/removes its own flags | Playground axis + bake | If the legacy default was ON (or the measured command included it), bake the flags into cells AND declare the axis so users can strip (red strikethrough). If default was OFF, keep cells clean and offer the axis preset only. parsers → `parsers` axis; MTP/EAGLE presets → `speculative` axis; dp-attention → `attention.dpAttn` or a strategy (DSv4 semantics) depending on coupling. |
| per-combo hidden option (e.g. spec hidden on Xeon) | absent cells | Don't create cells for combos the legacy widget couldn't produce; the engine greys them automatically. `# Error:` pseudo-commands → no cell + explanation in §2 tips and/or a chip `disable`/`disableReason`. |
| coupled secondary knob with no axis (e.g. mamba cache V1/V2) | cells + prose | No new engine axis for a migration. Bake the correct value per cell following the legacy coupling (Qwen3.5: MTP ⇒ `--mamba-scheduler-strategy extra_buffer` on NVIDIA; AMD/Xeon ⇒ V1/no flag), document the knob in §2 tips. |

## 2. Command rewrite table (the ONLY allowed normalizations)

| Legacy | New |
|---|---|
| `python(3) -m sglang.launch_server` | (engine emits `sglang serve`; cells hold flags only) |
| `--model X` / `--model-path X` | `--model-path {{MODEL_NAME}}` + `modelNames` key |
| `--tp-size N` | `--tp N` |
| `--speculative-algo X` (abbreviated) | `--speculative-algorithm X` — the Playground spec axis strips/derives by the full first token only; an abbreviated alias would survive toggles and double up |
| `--expert-parallel-size N` | `--ep N` — the Playground EP knob recognizes/strips only `--ep`; the long form would survive toggles and double up |
| (absent) | append `--host {{HOST_IP}}`, `--port {{PORT}}` to every cell |
| `--nnodes N --node-rank … --dist-init-addr …` literals | delete; `match.nodes: "multi-N"` + `nodesOptions` entry — the engine injects the trio after the last parallelism anchor plus the multi-node header comment |
| env-var command prefixes | verbatim into `cell.env[]` (never drop/normalize) |
| flag order as emitted | re-sort to canonical: `--trust-remote-code` → `--model-path` → parallelism (`--tp`/`--dp`/`--enable-dp-attention`/EP) → MoE → tuning → `--host`/`--port` (Playground insert anchors assume this). Keep the legacy relative order within the tuning span so commands stay eyeball-diffable. |

Caveats discovered in the pilot:
- The Playground `moe.ep` knob only understands `--ep` — normalize a legacy
  `--expert-parallel-size N` to `--ep N` (alias, see table above) so the knob
  can recognize/strip it.
- `multiNodeHints` only for hw whose fabric needs manual NIC env (gb200-class);
  standard-IB H100 multi-node needs none.
- `dockerImages`: only the tags the legacy page pinned. CPU/Xeon stays unmapped
  (`:dev` fallback) with a "install from source" tip.

## 2b. Playground axes: opt-out, not opt-in

The legacy page's silence about a feature does NOT mean the axis is dropped.
Every cookbook ships the **general axes** by default — `attention`
(TP/CP/DP-Attn), `moe` (backend + EP) for MoE models, `parsers`,
`speculative`, `pdDisagg`, `hicache` — then adds model-specific axes, and
deletes ONLY axes the model genuinely cannot use (`hisparse` is DSA-only;
MegaMoE is DeepSeek-V4 Blackwell-only). Knobs meaningless for a subset of
variants/hw get `disable` + `disableReason` (per-chip constraints), not
removal — e.g. MoE backend/EP greyed out on dense variants.

`speculative` presets must include every algorithm that actually appears on
the page — including the measured command's algorithm when it differs from
the generator default (Qwen3.5 ships both NEXTN and EAGLE) — otherwise the
verified cell's baseline can't be re-applied after a strip.

## 3. Verified policy mechanics

- Green requires measured data + flag equality with the measured command (see
  SKILL.md hard rule 3). Order `cells[]` so the verified flagship cell is
  **first** — `cells[0]` is the page's initial selection.
- When the measured command and the generator default disagree (Qwen3.5: bench
  ran `NEXTN` + `SGLANG_USE_CUDA_IPC_TRANSPORT=1`, generator emitted `EAGLE` +
  fusion flags), the verified cell mirrors the measurement; the generator
  default lives on as the not-verified sibling cells. Offer BOTH as Playground
  `speculative` presets and explain the split in §2 tips.
- Benchmarks accuracy keys outside gpqa/aime25/gsm8k need `config.accuracyLabels`
  (engine PR #27842); until merged the card degrades gracefully to the keys it
  knows. `defaultAccuracy` paints every *entry-bearing* cell of a variant — under
  the strict policy prefer per-entry `accuracy` on the measured cell only.

## 4. Per-family strategy sets (pre-designed; adjust at inventory time)

**Naming rule:** strategy ids reuse the canonical serving-strategy vocabulary —
`low-latency` / `balanced` / `high-throughput` (as established by DeepSeek-V4)
— never model-specific ids like `mtp`/`no-mtp`. A degenerate single strategy is
`balanced`. Deviate only when none of the three honestly describes the
difference (e.g. pure GPU-budget tiers), and confirm the naming with the
maintainer before authoring cells.

| Family | strategies | Notes |
|---|---|---|
| GLM-4.5, GLM-4.6 | `balanced` (TP) + `high-throughput` (TP+DP+EP) | MTP → speculative axis |
| GLM-4.7 | `low-latency`(2 GPUs) / `balanced`(4) / `high-throughput`(8) — gpus 2/4/8 + SUPPORT matrix; confirm naming, tiers are GPU budgets | measured-best B200 TP=2 NVFP4 → the verified cell |
| GLM-4.7-Flash | single `balanced` (tp=1) | dp/mtp → Playground |
| GLM-5, GLM-5.1 | `low-latency` (spec on per legacy condition) / `high-throughput` (dp-attention) | NVFP4 has a single recipe (legacy UI hid all toggles) |
| Kimi-K2 | `balanced` (tp8) + `high-throughput` (dp4+ep4) | variants = instruct/thinking; reasoning chip `hide` on instruct |
| Kimi-K2.5, K2.6 | `low-latency` / `high-throughput` | K2.5 spec preset carries `--speculative-draft-model-path …eagle3-mla`, chip-gated to h200/b300 |
| Kimi-Linear, MiniMax-M2, Qwen3, Qwen3.6, Qwen3-Next, Qwen3-Coder, Qwen3-Coder-Next | single `balanced` | |
| MiniMax-M2.5, M2.7 | `low-latency`(2) / `balanced`(4) / `high-throughput`(8=tp8+ep8) — confirm naming, tiers are GPU budgets | Xeon (M2.7) cells under one tier only |
| Qwen3.5 (DONE — pilot) | `low-latency` (MTP on) / `high-throughput` (MTP off) | see §5 |

Qwen3 variant fan-out: variants = deployable checkpoints size-ordered
(`235b-instruct`, `235b-thinking`, `235b`, `30b-*`, `32b`, …); do NOT abuse
strategies for the instruct/thinking category. Trim original-hybrid chips to
the ones the legacy page actually measured.

## 5. Worked example — the Qwen3.5 pilot (PR #27848)

Decisions log, in the order they came up:

1. **Strategy split over Playground toggle** because MTP couples with TP on
   three H100 combos (35B/27B BF16: tp2↔tp1+mem0.88; 122B FP8: tp4↔tp2).
   Canonical naming: `low-latency` = MTP on (legacy default), `high-throughput`
   = MTP off. Result: 186 cells = 87 low-latency + 99 high-throughput (Xeon has
   no low-latency cells — the legacy widget hid the MTP toggle there).
2. **Verified cell follows the measurement**: H200/397B/BF16/low-latency =
   `SGLANG_USE_CUDA_IPC_TRANSPORT=1` env + `--speculative-algorithm NEXTN`
   (normalized spelling) + measured flag set; all other cells = generator
   output verbatim with `EAGLE`. Both presets exposed on the speculative axis.
3. **FP4 single quant id** with `hw|variant|quant` modelNames keys →
   `nvidia/...NVFP4` (b200/b300) vs `amd/...MXFP4` (mi355x).
4. **Xeon** as `config.hardware` `vendor:"intel"`; cells carry
   `--device cpu --disable-overlap-schedule`; no docker mapping.
5. **Playground axes**: the full general set per §2b — attention
   (TP/CP/DP-Attn), moe (DeepEP backend + EP knob, `disable`+reason on the
   dense variants), parsers, speculative (NEXTN + EAGLE — both algorithms
   appear on the page), pdDisagg, hicache. Excluded as inapplicable:
   hisparse (DSA-only), MegaMoE (DSv4 Blackwell-only). The legacy
   `--expert-parallel-size 8` flag is normalized to `--ep 8` for the EP knob.
6. **Benchmarks**: one entry (the measured cell) only — entry-less cells render
   "pending" without stubs. `tokens_per_sec_per_gpu` = output tok/s ÷ 8.
   `sglang_version: "main branch"` verbatim. MMMU via `accuracyLabels` +
   sample-count detail in `notes`.
7. **Codegen + audit scripts** (adapt per model): a generator-port script that
   emits the cells literal, and an independent audit that `git show`s the
   ORIGINAL generator, stubs `useState`/`useEffect`, calls its
   `generateCommand(values)` per combo via indirect eval, and token-diffs
   against the new cells (expected deltas only). Read the legacy source via
   `git show main:<path>` — NOT `HEAD:` (the migration branch's HEAD has
   already deleted the file, so the audit breaks after the deletion commit).
   Re-run the audit after ANY later cells revision (renames included). Pilot
   result: 185/185 identical + 1 intentional override. Scripts are archived
   in PR #27848's description (collapsed details block).
8. **Inherited-infeasible combos kept verbatim** (e.g. 122B BF16 tp1 on
   mi325x: 244 GB weights vs 256 GB VRAM with mem-fraction 0.8) — they stay
   yellow and are listed in the PR body for the re-verification track.
9. **Browser-smoke probe pitfall**: multiple programmatic `.click()` calls in
   one synchronous eval batch under React 18 — the DOM reads between them are
   stale and look like snap-logic bugs. One click per eval, then settle.
