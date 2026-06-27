---
name: cookbook-migrate-model
description: Migrate a legacy-template SGLang cookbook page (monolithic per-model generator under docs_new/src/snippets/autoregressive/) onto the config-driven template (shared _deployment.jsx / _playground.jsx engines + per-model config). Use when asked to migrate, convert, or port an existing cookbook page — NOT for brand-new models (use cookbook-add-model for those). Run with /cookbook-migrate-model <Model page name, e.g. GLM-5.1>.
---

# Cookbook Migrate Model

Convert one legacy cookbook page to the config-driven format, faithfully. The
legacy page — its generator widget and its measured benchmark blocks — is the
**single source of truth**. You are transcribing it into the new data model,
not improving it.

Reuses the `cookbook-add-model` skill's assets (read them on demand):
- `../cookbook-add-model/templates/config.jsx.tmpl`, `page.mdx.tmpl`, `benchmarks.jsx.tmpl`
- `../cookbook-add-model/references/authoring-reference.md` (config/cells/playground contract)
- `../cookbook-add-model/references/mintlify-authoring.md` (MDX rules)

Migration-specific references in this skill:
- [references/dimension-mapping.md](references/dimension-mapping.md) — legacy-control → new-dimension mapping rules, command rewrite table, per-family strategy sets, and the Qwen3.5 pilot as a worked example (PR #27848).

The round's per-model inventory (scope, batch order, quirks, measured-data
survey) is tracked by the migration maintainer outside the repo — expect it in
your dispatch prompt, or ask for it.

## Hard rules (non-negotiable)

1. **Never modernize.** Env vars, flags, TP values, docker tags, version strings
   are copied verbatim from the legacy page — even when today's defaults differ
   (e.g. `SGLANG_ENABLE_SPEC_V2=1` is now default; keep it anyway). The recipe
   that was verified is the recipe as written. Allowed normalizations are ONLY
   the five alias rewrites in dimension-mapping.md §2 (`launch_server`→`sglang
   serve`, `--model`→`--model-path`, `--tp-size`→`--tp`, abbreviated
   `--speculative-algo`→`--speculative-algorithm`,
   `--expert-parallel-size`→`--ep`). **Accuracy-degrading flags**
   (`--kv-cache-dtype fp8_e4m3`, W4A4-style runtime quant) follow a
   deterministic rule — enforced in migration, no asking: offered as a
   legacy **selectable option** → never select it (cells mirror the
   accuracy-safe side), and the option **survives as a Playground axis** —
   an existing one where it fits, else add one via a separate prior engine
   PR (rule 4); a legacy choice never degrades to a tips mention. Baked
   into the recipe's **default/unconditional command** → keep it verbatim
   (the recipe was measured with it, and fp8 KV halves KV memory —
   stripping could OOM it). See dimension-mapping.md §2 caveats.
2. **Never invent versions or numbers.** Benchmark numbers only from the
   legacy page's measured blocks, and a result migrates ONLY when its
   `sglang_version` is a **reproducible anchor** — the bar is reproducibility,
   not "must be a release":
   - ✅ release tag/version (`v0.5.9` / `0.5.9`), commit hash, OR — for
     **Day-0 support** (the enabling PR isn't merged and no release is cut
     yet) — a specific **PR (`PR #27944`) or commit** you can `gh pr checkout`
     / `git checkout <sha>`. Commit is most precise; a PR pin is fine for day-0.
   - ❌ a moving ref — `"main branch"`, `"main (2026-06-11)"`, open-ended
     `"0.5.8+"` — is NOT reproducible: **drop the WHOLE result (speed AND
     accuracy)**, not just speed. Keep `benchmarkCommands` so ⚡Reproduce still
     guides re-measurement against a pinned build.
   **Never inherit cross-model numbers** — measurements the legacy page
   attributes to a *different model* (e.g. a K2.6 page carrying K2.5-measured
   speed) are dropped regardless of version. When kept, `sglang_version` is the
   legacy page's string verbatim. Docker tags only the ones the legacy page
   pinned (unmapped hw falls back to `:dev`).
3. **Verified policy (strictest tier).** `verified: true` ONLY when (a) the
   legacy page has concrete measured data for that exact 5-dim combo AND
   (b) the cell's flags equal the deployment command used for that measurement
   — modulo `{{HOST_IP}}`/`{{PORT}}`, the five alias rewrites, and **parser
   flags**: `--reasoning-parser`/`--tool-call-parser` are stripped from every
   cell (Playground-only feature; when the measured run had them on, say so in
   the benchmarks file header). When the measured command diverges from the
   generator default, **the verified cell follows the measured command**; the
   generator default stays as the sibling strategy/cell or a tips note. Everything else is unverified (yellow) —
   including combos that look memory-infeasible; keep them verbatim and list
   them in the PR body for the re-verification track.
4. **Engines are read-only.** `_deployment.jsx` / `_playground.jsx` must not
   change in a migration PR. Model-specific features are config DATA consumed
   by generic axis handlers (MegaMoE precedent), so they need NO engine change.
   A **titled single-select that strips a flag family** — KV Cache DType
   (`--kv-cache-dtype`), mamba (`--mamba-scheduler-strategy`), … — is already
   covered by the merged generic **`flagSelects`** axis: declare it in the
   config (a list of `{ id, title, stripPrefixes, options }`; see the Qwen3.5
   mamba example), **no engine PR**. Only a genuinely new control *shape* that
   `flagSelects` can't express would need a one-time generic primitive (never a
   model-named handler) on a separate prior PR (engine-axis.md).
5. **`github.cookbookModel` must be set** (`<hf-org>/<page-slug>`, e.g.
   `qwen/qwen3.5`) and the block never pruned — without it Submit ↗ mislabels
   as deepseek-v4. The issue template itself needs NO edits (free-form input).
6. **Strategy tiers are signal-driven.** A cell goes under `low-latency` /
   `high-throughput` ONLY on a signal present in the legacy source (an
   explicit performance toggle, a named recipe, or prose stating the
   operating point); **no signal → `balanced`**. Never derive a slant from
   your own hardware intuition — re-tiering on measured evidence is the
   hardware owner's follow-up PR, not part of a migration
   (dimension-mapping.md §4).

## Workflow (one model = one PR)

### 1. Inventory the legacy assets
- Read the legacy generator (`docs_new/src/snippets/autoregressive/<slug>-deployment.jsx`)
  end-to-end: every option dimension (radio vs checkbox vs dynamic), every
  gate/SUPPORT matrix, the full emitted command per reachable combo (env
  prefixes, `# Error` pseudo-commands included).
- Read the legacy MDX: §2 install docker tags → `dockerImages` (pinned, not
  upgraded); §3.2 tips → new §2; §4 invocation examples → new §3 (keep real
  Output Examples verbatim); §5 benchmark blocks → transcribe each measured
  block: deploy command used, bench command (dataset/isl/osl/num-prompts/
  concurrency), Mean TTFT/TPOT, output tok/s, hardware, version string.
- Inbound-anchor sweep: `grep -rn "<PageName>" docs_new/ --include='*.mdx'` —
  find links/`#fragments` into this page (`mint broken-links` does NOT check
  fragments). Fix referrers or add `<a id="old-anchor" />` shims in the same PR.
- Check the maintainer-provided inventory notes for this model's known quirks —
  but treat them (and the §4 family table) as a **survey snapshot**: re-verify
  every dimension against the live legacy files before mapping. Pages keep
  receiving updates (precedent: Kimi-K2.6's live generator has a speculative
  toggle the 2026-06-10 survey notes lack).

### 2. Design the 5-dim mapping
Apply [references/dimension-mapping.md](references/dimension-mapping.md). Key
decision — which legacy toggle becomes the `strategies` dimension: a toggle
that **changes other parts of the command** (TP, mem) must (the Playground
can't do coupled changes), and so does a toggle the legacy page itself labels
with operating-point words — e.g. a `dpattention` radio whose options are
subtitled "Low Latency" / "High Throughput" (GLM-5.1 / Kimi-K2.6 pattern) —
even when its flags are uncoupled (`--dp N --enable-dp-attention` is a pure
flag add). Any other toggle that only adds/removes its own flags becomes a
Playground axis with the flags baked into cells when the legacy default was
ON — EXCEPT parsers:
`--reasoning-parser`/`--tool-call-parser` are NEVER baked into cells, they are
Playground-only (DSv4 convention). **Every legacy control survives as an
interactive control** — a dimension or a Playground axis, never a tips-only
mention — and a model-specific control is **config data, not engine code**
(MegaMoE W4A4 is all DSv4 config on the existing `moe` axis). It's pure
config whenever it fits an existing axis's data schema. A **titled
single-select that strips a flag family** (Nemotron3's "KV Cache DType",
mamba `--mamba-scheduler-strategy`, …) fits the merged generic **`flagSelects`**
axis — so it too is config-only (declare a `flagSelects` list). Only a control
whose *shape* `flagSelects` still can't express would need a ONE-TIME generic
primitive (never a model-named handler) on a separate PRIOR engine PR, keeping
the migration PR data-only (hard rule 4, engine-axis.md). The strategy count follows the page's
operating points: **one recipe → a single `balanced`; two → `low-latency` +
`high-throughput`; three → the full trio (the ideal)**. The tiers apply per
(hw × variant × quant) combination — a single-recipe combination on a
multi-strategy page parks under its semantically honest tier (no
latency/throughput slant → `balanced`; the page's list is the union). When the
legacy toggle is MTP / speculative decoding, the direction is a deterministic
default — apply without asking: **MTP on → `low-latency`, MTP off →
`high-throughput`** (reversed only with maintainer confirmation). Tier
placement is signal-driven (hard rule 6). Never invent a recipe just to fill
strategy chips (see dimension-mapping.md §4). Record the outcome as a
**strategy mapping table** for the PR body — one row per group of
combinations sharing the same legacy signal (e.g. "all GPU combos: MTP
toggle → low-latency / high-throughput"; "xeon: (none) → balanced"), with a
one-line rationale each; don't enumerate 60 identical rows. The table is what
hardware owners sign off on at review.

### 3. Generate the config (codegen, then audit)
- For >~30 cells, port the legacy `generateCommand()` into a throwaway Node
  script that enumerates combos and emits the `cells:[...]` literal (output
  must stay a pure literal — Mintlify forbids runtime spreads/calls). Apply the
  verified-cell override in the script. See the pilot scripts embedded in the
  worked example of dimension-mapping.md §5.
- **Independent equivalence audit (required):** extract the ORIGINAL generator
  from git (`git show main:<path>` — NOT `HEAD:`, the migration branch deletes
  the file, see dimension-mapping.md §5 item 7), stub React hooks, run it for
  every combo, and diff token-by-token against the new cells. Expected deltas
  only: the appended `--host {{HOST_IP}}`/`--port {{PORT}}`, the
  engine-injected multi-node trio, the §2 alias rewrites (the entrypoint
  rewrite doesn't appear in cell tokens — cells hold flags only; the audit
  script normalizes it on the legacy side), and the intentional verified-cell
  override. Paste the PASS count + the audit script in the PR body
  (collapsed `<details>`).
- Hand-author the non-cells fields per authoring-reference.md. Structural
  self-checks: every cell resolves a `modelNames` key; no `--nnodes/--node-rank/
  --dist-init-addr/--host/--port` literals; every `{{KEY}}` declared; every
  `supportedHardware` id has ≥1 cell.

### 4. Benchmarks file
One entry per measured block only (cells without entries already render
"pending" — bare `{match}` stubs are unnecessary). `tokens_per_sec_per_gpu` =
output tok/s ÷ (tp × nnodes); TTFT/TPOT take the Mean rows; put the workload's
`num_prompts` into `workload`. **`config.accuracyLabels` is required whenever
the benchmarks carry accuracy data** — the engine ships no default eval set
(#27842), so missing labels means the accuracy rows silently don't render;
extra context (sample counts, suites that don't fit) goes in the entry's
`notes`. Zero-measured-data pages: skip the file and the `benchmarks` prop
entirely, but keep `benchmarkCommands` so ⚡Reproduce still guides users.

### 5. Rewrite the MDX
From `page.mdx.tmpl`: keep the original `title` (nav identity), write a fresh
SEO `description` (top-level — delete any legacy `metatags.description`), **no
`tag: NEW`** (a migration is not a launch), **no `mode:`**. Install accordion
carries the legacy install content + pinned images. Keep the template's
DSv4-style strategy bullets — serving semantics first (single-user chat /
typical multi-user / batch throughput), trimmed to the strategies the page
ships, plus at most a one-line note on what each strategy changes on this
model; do NOT rewrite them as toggle-/migration-centric explanations.
Legacy §5 benchmark prose is deleted (numbers → benchmark card, commands →
⚡Reproduce); legacy prose deploy commands are deleted (doc↔config parity —
fold their unique flags into §2 tips). Invocation examples + real outputs carry
over verbatim, but **wrapped in Accordions** — §3 commands and outputs are
collapsible (required, DeepSeek-V4 pattern): code in an
`<Accordion title="… (Python)">`, output in a following
`<Accordion title="Example Output">`; legacy pages kept them inline.

### 6. Delete the legacy generator
Remove `docs_new/src/snippets/autoregressive/<slug>-deployment.jsx` and its
import. `grep -rn "<slug>-deployment" docs_new/` must return nothing (config
provenance comments must not name the deleted path). Site wiring needs **no
changes**: docs.json path/title unchanged, vendor card + logo already exist.

### 7. Validate
- `grep -rn '__[A-Z_]*__'` on the new files (no template tokens).
- `cd docs_new && mint validate && mint broken-links` (pre-existing breaks on
  main are not yours — say so in the PR).
- `mint dev` browser smoke: initial selection = the verified cell (first in
  `cells[]`) with green badge; multi-node cells show the injected trio +
  header; AMD cells show env prefixes; Docker mode wraps with the pinned image
  and passes cell env as `--env`; condition-hidden combos grey out; benchmark
  card values; NO parser flags in any Deploy command; Playground parser
  toggles ADD the parser flags (green additions) while spec toggles strike
  the baked spec flags (red); Submit ↗ prefills this model. **Probe pitfall:** drive at most
  ONE programmatic click per evaluation and wait for React to settle —
  multiple clicks in one synchronous script batch and read stale DOM.
- Token-level audit from step 3 passes.

### 8. PR + review
One PR per model. PR body: migration framing, verified policy applied, the
strategy mapping table (step 2), the audit PASS count + script, any
inherited-infeasible combos flagged for re-verification. Then run `/cookbook-review-pr <N>` and fix findings.
FYI: docs previews only build for in-repo (`sgl-project/sglang`) branches —
a fork-headed PR is perfectly fine but renders no preview; a maintainer can
re-push the branch in-repo if a preview is wanted for review.

### 9. Keep this skill current
Any new convention, engine behavior, or pitfall you discover while migrating
(naming decisions, audit-script gotchas, review-rule conflicts, …) MUST be fed
back into this skill — same PR if it's skill-file-only, or an immediate
follow-up commit on the skill's branch/PR. The next agent runs on what's
written here, not on your session's context.
