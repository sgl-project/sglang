---
name: cookbook-review-pr
description: Review a pull request against the SGLang Cookbook (docs_new/, Mintlify) contribution checklist — the config-driven format (per-model config + benchmarks JSX consumed by the shared _deployment.jsx / _playground.jsx engines). Run with /cookbook-review-pr <PR number>.
---

# Cookbook Review PR

Fetch the diff, run the checklist, report what you find. The cookbook is **config-driven**:
shared engines (`_deployment.jsx`, `_playground.jsx`) with NO model-specific code; each
model is a data `config` (+ optional `benchmarks`) under `src/snippets/configs/<vendor>/`
plus an MDX page. This checklist targets that layout. Field-schema detail lives in
`.claude/skills/cookbook-add-model/references/authoring-reference.md` — defer to it rather
than restating.

## Usage

```
/cookbook-review-pr <PR number>
```

## Steps

1. `gh pr view <N> --repo sgl-project/sglang --json title,body,files,author,baseRefName,headRefName,commits,reviews`
2. `gh pr diff <N> --repo sgl-project/sglang`
3. `gh pr list --repo sgl-project/sglang --state open --search "<model name>"` (duplicate check)
4. Run every checklist item against the diff.
5. Output per-file verdicts + overall recommendation.

## Checklist

### 1. File hygiene
- A cookbook PR should only touch: `docs_new/src/snippets/configs/<vendor>/*.jsx`
  (config + benchmarks), `docs_new/cookbook/**/*.mdx`, `docs_new/docs.json`,
  `docs_new/cookbook/<category>/intro.mdx` (vendor card), `docs_new/cards/logos/<vendor>.png`
  (new vendor only). Flag stray files (`settings.local.json`, lockfiles, IDE configs).
- Pages must be `.mdx`, not `.md`. Files end with a trailing newline. Check commit history
  for unrelated commits accidentally included.
- **Engines untouched**: `_deployment.jsx` / `_playground.jsx` should NOT change in a
  model-add PR (adding a model is data-only). Engine edits = a separate axis/feature PR
  (see `cookbook-add-model/references/engine-axis.md`); review them against that checklist.

### 2. Config quality (the per-model `config`)
- Single `export const config = { ... }` literal — **no** function calls, spreads,
  fragment refs, or IIFE (Mintlify re-evals at hydration → `ReferenceError`).
- No `!(x in y)` anywhere (Mintlify AST walker crashes) — use `obj.key === undefined`.
- `supportedHardware` ⊆ `HARDWARE_CATALOG` (in `_deployment.jsx`) ∪ `config.hardware`. A
  model-specific GPU the shared catalog lacks must be declared in `config.hardware`
  (`{id,label,vram,vendor}`), **not** added to the engine catalog.
- `placeholders` declares every `{{KEY}}` used in `curl` or any cell.
- `modelNames` covers every cell (by `hw|variant|quant` triple or `variant|quant` pair).
- `strategies` count matches the page's operating points — 1 recipe → a single `balanced`;
  2 → `low-latency` + `high-throughput`; 3 → the full trio. Tiers apply per
  (hw × variant × quant) combination: a single-recipe combination must park under its
  semantically honest tier (clear slant → that tier, e.g. a workstation card under
  `low-latency`; no slant → `balanced`, e.g. a CPU platform) — **flag a no-slant recipe
  parked under low-latency/high-throughput**. Mixed unions
  like [low-latency, balanced, high-throughput] with per-selection greying are fine. Also
  flag model-specific ids (e.g. `mtp`), and flag an INVERTED speculative mapping — the
  deterministic default is MTP/spec-decoding ON → `low-latency`, OFF → `high-throughput`
  (at saturation the draft+verify overhead outweighs the speedup); the reverse needs an
  explicit maintainer-confirmed justification in the PR. The MDX strategy bullets describe serving semantics
  in the DSv4 style (single-user chat / typical multi-user / batch jobs), not internal
  toggles.
- `dockerImages` covers the hw ids that have cells (else users hit the `:dev` fallback); a
  `hw|quant` key (resolved before the plain `hw`) is valid when one quant on a shared GPU needs
  a different image (e.g. an FP4 dev build) — don't flag those.
- `multiNodeHints` present ONLY for hw whose fabric needs manual NIC env (e.g. `gb200`
  NVL72) — NOT every `multi-N` hw (standard-IB DeepEP / Marlin multi-node don't need it).
- `github.cookbookModel` is set to the model's HF id (`<hf-org>/<model-slug>`). The issue
  template's `model` field is a free-form input prefilled from this value; if the config
  omits the `github` block, the engine falls back to `deepseek-ai/deepseek-v4` and the
  page's submissions get mislabeled.
- `playgroundFeatures` is opt-OUT: the **general axes ship on every cookbook by default**
  (`attention` TP/CP/DP-Attn, `moe` backend+EP for MoE models, `parsers`, `speculative`,
  `pdDisagg`, `hicache`) — flag a missing general axis unless the model genuinely cannot
  use it. Model-specific axes only where applicable (MegaMoE backend + `megamoeQuant`
  only on Blackwell MoE, gated by `requiresHw`; `hisparse` only DSA-style). Knobs that are
  meaningless for a subset of variants/hw are `disable`d with a reason, not silently live
  (e.g. MoE knobs greyed on dense variants). No empty/stub axes.
- **No leftover `__TOKEN__`** — the config was stamped from the template and every
  placeholder is filled (`grep -rn '__[A-Z_]*__'` on the new config/benchmarks/MDX returns
  nothing).
- **All-hardware considered**: every `supportedHardware` id (from the catalog or `config.hardware`) has ≥1 cell OR is a deliberate
  greyed "coming soon"; AMD was pruned or kept on purpose (not a leftover template family).

### 3. Cells / 5-dim matrix
- Every cell `match` has EXACTLY the 5 keys (`hw`, `variant`, `quant`, `strategy`, `nodes`).
- `env` / `flags` are flat literals (only `{{PLACEHOLDER}}` subst) — no shared
  `commonFlags` reference (Mintlify won't inline it).
- NO `--nnodes` / `--node-rank` / `--dist-init-addr` literals in multi-node cells
  (the renderer injects them from `match.nodes`).
- NO literal `--host` / `--port` — use `{{HOST_IP}}` / `{{PORT}}`.
- NO `--reasoning-parser` / `--tool-call-parser` in any cell — parsers are a
  Playground-only feature added on top of the base command (DSv4 convention);
  flag any cell that bakes them in.
- Accuracy-degrading flags in cells — runtime quant below the checkpoint
  (e.g. MegaMoE W4A4 — DSv4 gates it behind the Playground's `megamoeQuant`)
  and lossy `--kv-cache-dtype` (e.g. `fp8_e4m3` over a higher-precision-KV
  checkpoint): **flag for explicit maintainer confirmation**. Output quality
  should be exactly what the quant chip declares, so absent a recorded
  sign-off in the PR (e.g. carried verbatim from a measured legacy recipe's
  default command), request the flag move to Playground/tips.
- Flag order: `--model-path` first (an optional `--trust-remote-code` may precede it —
  the DSv4 cells do), then parallelism, then MoE, then tuning, `--host`/`--port`
  last (the playground's insert anchors assume this).
- TP/memory sanity: `model_weight_GB / (tp × gpu_mem)` fits with ~20–30% headroom
  (BF16 ≈ params×2 GB, FP8 ≈ ×1, FP4 ≈ ×0.5; MoE uses **total** weight, not active params).

### 4. Benchmarks
- Each `benchmarks[]` entry's `match` tuple corresponds to a real cell.
- `accuracyLabels` is present whenever the benchmarks carry accuracy data — the engine
  ships NO default eval set; without it the accuracy rows silently don't render.
  `defaultAccuracy` / per-cell `accuracy` / `benchmarkCommands.accuracy` keys all
  ∈ `config.accuracyLabels`.
- A benchmark's quantization must match a variant actually listed — `(BF16)` on a model
  that only released FP8/FP4 is a factual bug.
- `benchmarkCommands.speed` is `python3 -m sglang.bench_serving` (the workload), separate
  from the `sglang serve` deploy command, and should carry `--flush-cache`: bench_serving's
  `random` prompts are deterministic, so a warm rerun hits the radix cache and inflates
  throughput — speed numbers are measured cache-cold.
- `sglang_version` is a real build the author ran (a release, or `dev`/nightly) — not a
  guessed/placeholder value (no leftover `0.0.0`).
- **Latency percentile**: `config.latencyPercentile` (default `"P50"`, or `"Mean"`) matches the
  percentile the TTFT/TPOT values actually are — the card renders `TTFT (<pct>)`. A benchmarks
  entry may carry its own `latencyPercentile` to override the page value per cell
  (entry → config → `"P50"`): on a P50 page, kept legacy Mean cells must set it — a
  `sglang_version` tag alone doesn't convey the percentile. (`"Mean"` is temporary — legacy
  data is being re-measured to P50.)
- **Throughput convention**: `tokens_per_sec_per_gpu` is stored as **total (in+out)/GPU**
  = `output tok/s/GPU × (isl+osl)/osl`, shown by the card as-is. Flag output-only values.
- **Consistent accuracy harness across entries**: every value under one `accuracyLabels`
  column must be produced by the SAME harness — flag a page that, say, measures one
  platform's GSM8K with `few_shot_gsm8k --num-questions 200` and another's with
  `run_eval --eval-name gsm8k --num-examples 1319` and shows both as one "GSM8K %"
  (the scores aren't comparable). Either standardize on one harness (matching
  `benchmarkCommands.accuracy`) or require an explicit per-entry note. Common when folding
  a second contributor's measurements (e.g. an AMD/ROCm PR) into the page.

### 5. Doc ↔ config parity (the #1 finding)
- Any `sglang serve` command shown in MDX prose (config tips, benchmark section) must
  equal what the engine emits from the corresponding cell — same flags, same order. Drift
  here is the most common review miss.

### 6. Commands / port
- Launch uses `sglang serve` — flag any `python -m sglang.launch_server` /
  `python3 -m sglang.launch_server` (deprecated). The engine already emits `sglang serve`;
  guard against prose/cells reintroducing the old launcher.
- Port `30000` everywhere (launch, curl, client `base_url`, bench) — flag `8000`.
  Launch port must match client/curl port on the same page.

### 7. Frontmatter
- Every new MDX page has `title:` and a **top-level** `description:` (a real one-line value
  prop, not copied from another vendor) — NOT `metatags.description` (non-canonical; the
  top-level field is what renders as the subtitle and SEO meta — see mintlify-authoring).
- **No `mode: wide` on a model page** — it hides the right-hand "On this page" ToC that every
  other model page has. Leave `mode` unset (the Deploy/Playground panels self-cap at 900px, so
  the default column holds them fine). `mode: wide` belongs only on category `intro.mdx` grids.
- `tag: NEW` only for genuine new launches; when one is added, stale `tag: NEW` on older
  pages should be dropped in the same PR (`grep -RlE "^tag: NEW" docs_new/cookbook/`).
- MDX imports BOTH `Deployment` and `Playground` from `/src/snippets/...` (absolute).
- Deploy heading slugs to `deployment` (or `deploy`), Playground to `playground` — so
  "↑ Switch base" and "Open the Playground →" scroll. No numbered headings for these two.

### 8. Navigation & homepage
- New page → `docs_new/docs.json` updated: under the right vendor group inside
  `navigation` → Cookbook → Autoregressive Models, root-relative, **no `.mdx`**:
  `cookbook/<category>/<Vendor>/<Model>`.
- Homepage `<Card href>` in `docs_new/cookbook/<category>/intro.mdx` points to the vendor's
  flagship; new vendors get a new `<Card>` + a logo at `docs_new/cards/logos/<vendor>.png` —
  **940×525 RGBA transparent, icon-only (no wordmark)**, lowercase filename, tracked via
  `git add -f` (`*.png` is gitignored repo-wide). Card order matches the `docs.json` nav order.
- Don't change `docs_new/cookbook/intro.mdx` for individual model adds (top-level only).

### 9. Links & factual
- HuggingFace URLs resolve to a real model. License section matches the actual HF license
  (don't copy from another model). Docker images from `lmsysorg/sglang`; no `sgl-project-dev`.
  The image **tag** is a real build (a release the author ran, or `:dev`/nightly) — not a
  guessed version.
- Internal links root-relative, no extension (`/cookbook/.../<Model>`); flag `.md`/`.mdx`
  or `../`-relative links. `docs.sglang.io` is canonical.
- No Google-Drive image links (don't render). Shell placeholders are `export VAR=<value>`,
  not `${VAR}` (a bash no-op).
- **Parser ids must exist in the code registries** on the PR's target branch: every
  `--reasoning-parser X` / `--tool-call-parser Y` named in prose or in
  `playgroundFeatures.parsers` flags is a registered key in
  `python/sglang/srt/parser/reasoning_parser.py` (DetectorMap) /
  `python/sglang/srt/function_call/function_call_parser.py` (ToolCallParserEnum) —
  prose naming a near-miss id (e.g. the reasoning id where the tool id differs) is a
  factual bug. `--…-parser auto` is acceptable ONLY if the template-detection rules
  (`python/sglang/srt/managers/template_detection.py`) actually resolve THIS model's
  chat template to the right parser — no rule match means auto silently disables the
  parser; when in doubt require explicit ids (the DSv4 page pins explicit ids).

### 9b. MDX authoring (Mintlify) — detail in `cookbook-add-model/references/mintlify-authoring.md`
- **Forbidden syntax**: no Docusaurus admonitions (`:::`), `@site`/`@theme`, GitHub alert
  blocks (`> [!NOTE]`), markdown **pipe tables** (use JSX `<table>`), inline `<details>`,
  or unknown components. `<CardGroup>`/`<Card>` only on category `intro.mdx`, not model pages.
- Code fences are **labeled** (e.g. `python Example` / `bash Command` / `text Output` after
  the opening fence); a fenced block nested inside another uses four backticks outside.
- §3 commands and outputs are **collapsible** (DeepSeek-V4 pattern): every runnable
  example wrapped in an `<Accordion>`, its real output in a following
  `<Accordion title="Example Output">` (`Pending update...` only with user
  acknowledgement). Flag bare/inline example blocks and `**Output Example:**` headings.
- Reasoning-parser example matches the parser's **output shape**: separate-field
  (`reasoning_content` + `content`) vs inline `<think>` tags parsed out of `content`.
- No hardcoded sampling params (`temperature` / `top_p`) in sample code (SGLang uses
  `generation_config.json` defaults); listing them in §1 informationally is fine.

### 10. Quantization rules
- **NVFP4** checkpoints are Blackwell-only (B200/B300/GB300) — never AMD. An AMD FP4 cell
  is legitimate ONLY when the vendor published an **MXFP4** checkpoint for it (e.g.
  `amd/Qwen3.5-397B-A17B-MXFP4` on MI355X) — verify the HF repo resolves; otherwise the
  AMD FP4 chip must be absent/`disabled`.
- BF16 / FP8 work on NVIDIA and AMD. `--kv-cache-dtype fp8_e4m3` in a cell is an
  accuracy-degrading flag — see §3 (needs explicit maintainer sign-off; default
  home is Playground/tips).

### 11. Scope
- Changes match the PR title. Flag global changes hiding behind a platform-specific title
  (e.g. an "H200 FP8" PR that adds a flag to ALL cells). Unmentioned side-fixes belong in
  the PR body.

### 12. Duplicate PRs
- Another open PR for the same model? Flag it; compare completeness; note merge-conflict
  risk on `docs.json` + the vendor card; flag a superseded older PR by the same author.

### 13. Build / validate
```bash
cd docs_new
mint validate
mint broken-links
```
Optional: `mint dev` for a visual smoke test.

### 14. Reviewer feedback
- `gh api repos/sgl-project/sglang/pulls/<N>/comments` — have prior reviewer requests been
  addressed? Unresolved requested-changes should be flagged.

### 15. Grammar & spelling
- Check added/changed prose for typos and grammar (e.g. "recommend" vs "recommended").
  Flag each with the exact wrong text + correction.

## Output

Per file:
- ✅ PASS
- ⚠️ ISSUE: \<what\>
- 🔴 BLOCK: \<what\>

Overall: **APPROVE** / **REQUEST CHANGES** / **BLOCKED**
