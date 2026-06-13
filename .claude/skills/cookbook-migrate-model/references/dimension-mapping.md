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
| toggle that **couples** with other parts of the command (changes TP/mem/EP), OR one the legacy page labels with **operating-point words** | `strategies` | The Playground applies pure flag diffs — it cannot do coupled changes. Example: Qwen3.5's MTP toggle bumps TP on three H100 combos → strategies `low-latency` (MTP on) / `high-throughput` (MTP off). **Naming counts like coupling**: GLM-5.1's / Kimi-K2.6's `dpattention` adds only `--dp N --enable-dp-attention` (uncoupled), but its options are subtitled "Low Latency" / "High Throughput" — the page's own named operating-point split → strategies; a flag-only spec toggle riding alongside it stays a Playground axis and bakes per its legacy default. GPU-count radios (GLM-4.7, MiniMax-M2.5/2.7) → budget-tier strategies with the legacy SUPPORT matrix preserved by which cells exist. Strategy count follows the page's operating points: 1 → `balanced`, 2 → `low-latency`+`high-throughput`, 3 → the full trio (§4). |
| toggle that only adds/removes its own flags | Playground axis (+ bake, EXCEPT parsers and accuracy-degrading flags) | **Parsers (`--reasoning-parser` / `--tool-call-parser`) are NEVER baked into cells** — Deployment commands ship without them regardless of the legacy default or the measured command; the `parsers` axis adds them on top (DSv4 convention; cells mirror the legacy generator's parsers-OFF output). Accuracy-degrading toggles are never baked either — §2 caveats (axis-only, accuracy-safe cells). Other flag-only toggles: legacy default ON → bake into cells AND declare the axis so users can strip (red strikethrough); default OFF → keep cells clean, axis preset only. MTP/EAGLE presets → `speculative` axis; dp-attention → a strategy when the legacy page labels it as the operating-point split or when coupled (see the row above), else `attention.dpAttn`. **No fitting built-in axis ≠ drop the feature** — EVERY legacy control survives as an interactive control (a dimension or a Playground axis), never a tips-only mention. When none of the built-in axes fits (Nemotron3's "KV Cache DType" radio is the precedent), add the axis via the engine-axis flow: a separate PRIOR engine PR, then the migration config declares it (hard rule 4). |
| per-combo hidden option (e.g. spec hidden on Xeon) | absent cells | Don't create cells for combos the legacy widget couldn't produce; the engine greys them automatically. `# Error:` pseudo-commands → no cell + explanation in §2 tips and/or a chip `disable`/`disableReason`. |
| coupled secondary knob (e.g. mamba cache V1/V2) | cells + Playground axis | Bake the correct value per cell following the legacy coupling (Qwen3.5: MTP ⇒ `--mamba-scheduler-strategy extra_buffer` on NVIDIA; AMD/Xeon ⇒ V1/no flag) and document the coupling in §2 tips — AND surface the knob as a Playground axis like every other legacy feature (row above; add the axis when none fits). Baking alone is NOT enough — the every-feature rule supersedes the pilot's cells+prose-only treatment of Qwen3.5's mamba knob (retrofit pending); Qwen3.6 / Qwen3-Coder-Next carry the same knob, so their migrations need the axis in place first. |

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
- **Accuracy-degrading flags** (`--kv-cache-dtype fp8_e4m3`, W4A4-style
  runtime quant) — deterministic rule, enforced in migration without
  asking:
  - offered as a legacy **selectable option/toggle** → never select it;
    cells mirror the accuracy-safe side (even if the legacy default was the
    lossy side). The option itself **must survive as a Playground axis** — the user's choice may not degrade to a
    tips mention. Use the existing axis when one fits (DSv4 gates W4A4
    behind `megamoeQuant`); when the playground lacks one — e.g.
    Nemotron3-Ultra's "KV Cache DType" radio (None default / fp8_e4m3 /
    bf16) has no kv-cache axis today — **add the axis** via the engine-axis
    flow: a separate PRIOR engine PR (hard rule 4, like accuracyLabels
    #27842), then the migration config declares it;
  - baked into the recipe's **unconditional/default command** → keep it
    verbatim. The legacy measurements ran with it, and fp8 KV halves KV
    memory — stripping could OOM the recipe. Expect this pattern: legacy
    AMD recipes routinely append `--kv-cache-dtype fp8_e4m3` ("for memory
    efficiency"), and GLM-5's NVFP4 path ships it too — all keep.

  (Only migration gets this auto-keep — faithfulness wins here. On new
  pages the same flags are flag-and-confirm with the maintainer:
  authoring-reference §2.2 / review checklist.)

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

The `parsers` axis is **add-only**: `--reasoning-parser` /
`--tool-call-parser` are never part of any Deployment cell (see §1) — the
axis adds them on top of the base command, so toggling a parser renders a
green addition, never a strikethrough.

## 3. Verified policy mechanics

- Green requires measured data + flag equality with the measured command (see
  SKILL.md hard rule 3). Order `cells[]` so the verified flagship cell is
  **first** — `cells[0]` is the page's initial selection.
- When the measured command and the generator default disagree (Qwen3.5: bench
  ran `NEXTN` + `SGLANG_USE_CUDA_IPC_TRANSPORT=1`, generator emitted `EAGLE` +
  fusion flags), the verified cell mirrors the measurement; the generator
  default lives on as the not-verified sibling cells. Offer BOTH as Playground
  `speculative` presets and explain the split in §2 tips.
- `config.accuracyLabels` is REQUIRED whenever benchmarks carry accuracy data —
  the engine ships no default eval set (#27842); without it the accuracy rows
  silently don't render. `defaultAccuracy` paints every *entry-bearing* cell of
  a variant — under the strict policy prefer per-entry `accuracy` on the
  measured cell only.

## 4. Per-family strategy sets (survey sketches — re-derive from the live page)

The family table below was sketched from the 2026-06-10 survey at PAGE level.
At migration time **re-derive it from the live generator**: pages drift
(precedent: Kimi-K2.6's live page has a speculative toggle the survey notes
lack), and the per-combination rule means gated/hidden toggles — typically on
Xeon, AMD, or a single-recipe quant like NVFP4 — produce `balanced` combos the
page-level sketch doesn't show.

**Strategy-set rule — the count follows the page's operating points** (ids
always from the DeepSeek-V4 vocabulary, never model-specific ids like
`mtp`/`no-mtp`):

- **1 operating point** (a single recipe, no performance toggle) → a single
  **`balanced`** strategy. Never invent a second recipe just to fill chips.
- **2 operating points** → **`low-latency` + `high-throughput`**. When the
  legacy toggle is MTP / speculative decoding, the mapping is a
  **deterministic default — apply it without asking**: MTP on →
  `low-latency`, MTP off → `high-throughput`. (Why it's near-certain:
  speculative decoding cuts per-token latency at low concurrency, but at
  saturation the draft+verify overhead costs more than it saves — DSv4's
  high-throughput recipes disable MTP for the same reason.) Other toggles
  map by the same serving semantics — the two recurring **high-throughput
  markers** are **dp-attention ON** (MLA-attention models) and **EP / DP+EP
  ON** (MoE models): both shard work across ranks for saturated throughput
  at some per-request latency cost. These directions apply to the toggle
  CHOSEN as the strategy dimension (§1); a flag-only spec toggle riding
  alongside a named operating-point toggle stays a Playground axis and bakes
  per its legacy default — GLM-5.1's spec defaults ON, so its flags bake
  into BOTH tiers there. Only if a legacy page documents the OPPOSITE slant
  (e.g. "enable MTP for high throughput") stop and confirm with the
  maintainer.
- **3 operating points** → the **full trio** (the ideal — e.g. GPU-budget
  tiers 2/4/8).

**Signal-driven tiers (hard rule).** A cell goes under `low-latency` /
`high-throughput` ONLY on a signal present in the legacy source: an explicit
performance toggle (MTP/speculative, dp-attention, EP, gpuCount, …), a named
recipe/strategy checkbox, option subtitles ("Low Latency" / "High
Throughput"), or prose stating the operating point. Reading such
a signal is SGLang-level serving semantics (MTP favors latency on any
vendor's silicon), so any migrator can tier any vendor's cells without
hardware-specific judgment. **No signal → `balanced`** — legacy silence is
itself information: the page offered that command as the hardware's
general-purpose operating point, and `balanced` transcribes exactly that.
Never derive a slant from your own hardware intuition ("this flag combo
feels throughput-tuned"); re-tiering on measured evidence is the hardware
owner's follow-up PR, not part of a migration. A toggle that maps to no
dimension, or a suspected undocumented slant → stop and ask the maintainer.

The tiers apply **per (hw × variant × quant) combination**, not just per page:
a combination with fewer operating points than the page parks its cells in
the semantically honest tier. A single recipe with a signal-evidenced slant
goes to that tier (DSv4's RTX PRO 6000 → `low-latency`: workstation card,
low-batch Marlin recipe — the recipe's own SGLang-legible content is the
evidence); a general-purpose recipe with no latency/throughput slant goes to
`balanced` (Qwen3.5's Xeon → `balanced`). Never park a no-slant recipe under
`low-latency`/`high-throughput` just because the page's toggle mapping lands
there — that reads as a semantic lie ("CPU = high-throughput?"). The page's
`strategies` list is the union of tiers actually used (a mixed
[low-latency, balanced, high-throughput] page where GPUs use the two ends and
CPU uses the middle is fine); the engine greys unused chips per selection and
auto-snaps, no extra config needed.

Deviations (e.g. how to name pure GPU-budget tiers) need maintainer sign-off.
The MDX strategy bullets describe serving semantics in the DSv4 style
(single-user chat / typical multi-user / batch jobs), with at most a one-line
model-specific note — never toggle-/migration-centric explanations.

| Family | strategies | Notes |
|---|---|---|
| Gemma4 | `low-latency` (MTP on — the legacy toggle's own "Lower Latency" subtitle) / `high-throughput` (MTP off); mi300x hides the toggle → its single recipe → `balanced` (trio union, Qwen3.5 Xeon pattern) | variants = e2b/e4b/12b/31b/26b-a4b; checkpoint radio Standard(BF16)/QAT(q4_0) → quant ids via `modelNames`; §3.3 prose carries AMD recipes beyond the widget's mi300x — maintainer call on cells-from-prose vs tips; vision/audio invocation prose carries over (deployment matrix is text-standard); "gemma4 branch" version → speed drops, MMLU/GSM8K accuracy keeps (mind the few-shot vs run_eval harness footnote); dedicated multi-arch dev images verbatim |
| Nemotron3-Ultra | dpattention carries "Low latency"/"High throughput" subtitles (naming rule) but THREE perf controls stack — multi-value DP-Attention (2/4/8) × MTP × EP — design the tier mapping via the step-2 table; maintainer sign-off required | NVIDIA-only (h100→gb300) with a per-quant verified-hw SUPPORT matrix → absent cells; "Model" radio = the quant dim (BF16 / NVFP4 Blackwell-only); TP radio 8/16 — TP=16 is 2-node → `nodes` dim; **kvcache radio (None/fp8_e4m3/bf16) → NEW Playground axis, engine PR FIRST** (every-feature rule §1); `launch_server` + spec-V2 env prefix verbatim; dedicated `dev-nemotron3-ultra(+cu13)` images verbatim ("not in any stable release"); "main branch" version → speed drops, GSM8K accuracy keeps |
| GLM-4.5, GLM-4.6 | `low-latency` (TP, + MTP from the legacy checkbox) / `high-throughput` (TP+DP+EP) | |
| GLM-4.7 | `low-latency`(2 GPUs) / `balanced`(4) / `high-throughput`(8) — gpus 2/4/8 + SUPPORT matrix; confirm naming, tiers are GPU budgets | measured-best B200 TP=2 NVFP4 → the verified cell |
| GLM-4.7-Flash | `low-latency` (tp1 + MTP from the legacy checkbox) / `high-throughput` (DP) | derive from the legacy dp/mtp checkboxes |
| GLM-5, GLM-5.1 | `low-latency` (dpattention off) / `high-throughput` (dpattention on) — the dpattention radio carries the page's own "Low Latency"/"High Throughput" subtitles (naming rule, §1) | spec is flag-only, default ON, hidden on AMD → bakes into both tiers on NVIDIA + `speculative` axis; NVFP4 hides all toggles → single no-signal recipe → `balanced` (page ships the trio union) |
| Kimi-K2 | `low-latency` (tp8) / `high-throughput` (dp4+ep4) | variants = instruct/thinking; reasoning chip `hide` on instruct |
| Kimi-K2.5, K2.6 | `low-latency` (dpattention off) / `high-throughput` (dpattention on) — same named-subtitle pattern as GLM-5.1 | K2.5 spec preset carries `--speculative-draft-model-path …eagle3-mla`, chip-gated to h200/b300; K2.6's live page has a NVIDIA-only spec toggle, default OFF → `speculative` axis only, no bake (missed by the survey) |
| Qwen3.6, Qwen3-Next | `low-latency` (MTP on, the legacy speculative toggle) / `high-throughput` (MTP off); Xeon hides the toggle → its single recipe → `balanced` (page ships the trio) | same pattern as the Qwen3.5 pilot |
| Kimi-Linear, MiniMax-M2, Qwen3, Qwen3-Coder, Qwen3-Coder-Next | single `balanced` — one recipe, no performance toggle (rule above: 1 operating point → `balanced`) | renders as one chip; Qwen3-Coder-Next has NO speculative dim on the live page (quant × toolcall × mambaCache only — an earlier sketch wrongly lumped it with Qwen3.6) |
| MiniMax-M2.5, M2.7 | `low-latency`(2) / `balanced`(4) / `high-throughput`(8=tp8+ep8) — confirm naming, tiers are GPU budgets | Xeon (M2.7) is a single no-slant recipe (fixed TP=6) → `balanced` (per-combination rule, Qwen3.5 Xeon precedent) |
| Qwen3.5 (DONE — pilot) | `low-latency` (MTP on) / `high-throughput` (MTP off); Xeon's single no-slant recipe → `balanced` (the page ships the full trio) | see §5 |

Qwen3 variant fan-out: variants = deployable checkpoints size-ordered
(`235b-instruct`, `235b-thinking`, `235b`, `30b-*`, `32b`, …); do NOT abuse
strategies for the instruct/thinking category. Trim original-hybrid chips to
the ones the legacy page actually measured.

## 5. Worked example — the Qwen3.5 pilot (PR #27848)

Decisions log, in the order they came up:

1. **Strategy split over Playground toggle** because MTP couples with TP on
   three H100 combos (35B/27B BF16: tp2↔tp1+mem0.88; 122B FP8: tp4↔tp2).
   Canonical naming: `low-latency` = MTP on (legacy default), `high-throughput`
   = MTP off. Xeon has a single operating point (the legacy widget hid the MTP
   toggle there) and its recipe has no latency/throughput slant → its 12 cells
   park under `balanced` (per-combination placement; parking them under
   high-throughput as a toggle-mapping side effect read as a semantic lie).
   Result: 186 cells = 87 low-latency + 87 high-throughput + 12 balanced; the
   page ships the full trio and the engine greys unused chips per selection.
2. **Verified cell follows the measurement**: H200/397B/BF16/low-latency =
   `SGLANG_USE_CUDA_IPC_TRANSPORT=1` env + `--speculative-algorithm NEXTN`
   (normalized spelling) + measured flag set **minus the parser flags** (the
   measured run had both parsers on; cells never carry them — noted in the
   benchmarks header). All other cells = the generator's parsers-OFF output
   verbatim with `EAGLE`. Both spec presets exposed on the speculative axis.
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
   "pending" without stubs. The legacy speed numbers were DROPPED: they were
   measured on a drifting "main branch" build, which is no version anchor
   (speed migrates only under an exact release tag / commit hash — hard rule
   2), so the entry carries accuracy only (GSM8K + MMMU via `accuracyLabels`,
   sample counts in `notes`) and no `sglang_version`.
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
