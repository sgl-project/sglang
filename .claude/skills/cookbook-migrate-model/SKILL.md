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
- [references/model-inventory.md](references/model-inventory.md) — per-model quirks for the remaining legacy pages (dimensions, hardware, env vars, measured-data inventory, batch order).

## Hard rules (non-negotiable)

1. **Never modernize.** Env vars, flags, TP values, docker tags, version strings
   are copied verbatim from the legacy page — even when today's defaults differ
   (e.g. `SGLANG_ENABLE_SPEC_V2=1` is now default; keep it anyway). The recipe
   that was verified is the recipe as written. Allowed normalizations are ONLY
   the five alias rewrites in dimension-mapping.md §2 (`launch_server`→`sglang
   serve`, `--model`→`--model-path`, `--tp-size`→`--tp`, abbreviated
   `--speculative-algo`→`--speculative-algorithm`,
   `--expert-parallel-size`→`--ep`).
2. **Never invent versions or numbers.** `sglang_version` is the legacy page's
   string verbatim — including "main branch", "0.5.8+", or a commit hash.
   Benchmark numbers only from the legacy page's measured blocks. Docker tags
   only the ones the legacy page pinned (unmapped hw falls back to `:dev`).
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
   change in a migration PR. If the model needs an engine capability (a new
   axis, accuracy labels, …), that is a separate prior PR.
5. **`github.cookbookModel` must be set** (`<hf-org>/<page-slug>`, e.g.
   `qwen/qwen3.5`) and the block never pruned — without it Submit ↗ mislabels
   as deepseek-v4. The issue template itself needs NO edits (free-form input).

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
- Check `references/model-inventory.md` for this model's known quirks.

### 2. Design the 5-dim mapping
Apply [references/dimension-mapping.md](references/dimension-mapping.md). Key
decision: a legacy toggle that **changes other parts of the command** (TP, mem)
becomes a `strategies` entry (the Playground can't do coupled changes); a
toggle that only adds/removes its own flags becomes a Playground axis with the
flags baked into cells when the legacy default was ON — EXCEPT parsers:
`--reasoning-parser`/`--tool-call-parser` are NEVER baked into cells, they are
Playground-only (DSv4 convention). The strategy set
defaults to the full trio `low-latency` / `balanced` / `high-throughput`;
**`low-latency` and `high-throughput` are mandatory on every page** — a
single-strategy page is not acceptable; include `balanced` whenever the page
has a third operating point (see dimension-mapping.md §4).

### 3. Generate the config (codegen, then audit)
- For >~30 cells, port the legacy `generateCommand()` into a throwaway Node
  script that enumerates combos and emits the `cells:[...]` literal (output
  must stay a pure literal — Mintlify forbids runtime spreads/calls). Apply the
  verified-cell override in the script. See the pilot scripts embedded in the
  worked example of dimension-mapping.md §5.
- **Independent equivalence audit (required):** extract the ORIGINAL generator
  from git (`git show HEAD:<path>`), stub React hooks, run it for every combo,
  and diff token-by-token against the new cells. Expected deltas only: the
  appended `--host {{HOST_IP}}`/`--port {{PORT}}`, the engine-injected
  multi-node trio, the four alias rewrites, and the intentional verified-cell
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
audit PASS count + script, any inherited-infeasible combos flagged for
re-verification. Then run `/cookbook-review-pr <N>` and fix findings.
FYI: docs previews only build for in-repo (`sgl-project/sglang`) branches —
a fork-headed PR is perfectly fine but renders no preview; a maintainer can
re-push the branch in-repo if a preview is wanted for review.

### 9. Keep this skill current
Any new convention, engine behavior, or pitfall you discover while migrating
(naming decisions, audit-script gotchas, review-rule conflicts, …) MUST be fed
back into this skill — same PR if it's skill-file-only, or an immediate
follow-up commit on the skill's branch/PR. The next agent runs on what's
written here, not on your session's context.
