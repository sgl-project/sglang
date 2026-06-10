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
- `dockerImages` covers the hw ids that have cells (else users hit the `:dev` fallback).
- `multiNodeHints` present ONLY for hw whose fabric needs manual NIC env (e.g. `gb200`
  NVL72) — NOT every `multi-N` hw (standard-IB DeepEP / Marlin multi-node don't need it).
- `github.cookbookModel` matches the issue-template `model` dropdown value.
- `playgroundFeatures` axes are pruned to what the model supports — no empty/stub axes
  (the `moe` axis's MegaMoE backend option + `megamoeQuant` block only on Blackwell MoE,
  gated by `requiresHw`; `hisparse` only DSA-style; `pdDisagg.router` only with a PD topology).
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
- Flag order: `--model-path` first, then parallelism, then MoE, then tuning, `--host`/`--port`
  last (the playground's insert anchors assume this).
- TP/memory sanity: `model_weight_GB / (tp × gpu_mem)` fits with ~20–30% headroom
  (BF16 ≈ params×2 GB, FP8 ≈ ×1, FP4 ≈ ×0.5; MoE uses **total** weight, not active params).

### 4. Benchmarks
- Each `benchmarks[]` entry's `match` tuple corresponds to a real cell.
- `defaultAccuracy` keys ∈ `ACCURACY_LABELS` (and `benchmarkCommands.accuracy`).
- A benchmark's quantization must match a variant actually listed — `(BF16)` on a model
  that only released FP8/FP4 is a factual bug.
- `benchmarkCommands.speed` is `python3 -m sglang.bench_serving` (the workload), separate
  from the `sglang serve` deploy command.
- `sglang_version` is a real build the author ran (a release, or `dev`/nightly) — not a
  guessed/placeholder value (no leftover `0.0.0`).

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
- Every new MDX page has `title:` and `metatags.description:` (a real one-line value prop,
  not copied from another vendor).
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

### 9b. MDX authoring (Mintlify) — detail in `cookbook-add-model/references/mintlify-authoring.md`
- **Forbidden syntax**: no Docusaurus admonitions (`:::`), `@site`/`@theme`, GitHub alert
  blocks (`> [!NOTE]`), markdown **pipe tables** (use JSX `<table>`), inline `<details>`,
  or unknown components. `<CardGroup>`/`<Card>` only on category `intro.mdx`, not model pages.
- Code fences are **labeled** (e.g. `python Example` / `bash Command` / `text Output` after
  the opening fence); a fenced block nested inside another uses four backticks outside.
- Every runnable invocation block is followed by `**Output Example:**` + a `text Output`
  fenced block (real output, or `Pending update...` only with user acknowledgement).
- Reasoning-parser example matches the parser's **output shape**: separate-field
  (`reasoning_content` + `content`) vs inline `<think>` tags parsed out of `content`.
- No hardcoded sampling params (`temperature` / `top_p`) in sample code (SGLang uses
  `generation_config.json` defaults); listing them in §1 informationally is fine.

### 10. Quantization rules
- FP4 is Blackwell-only (B200/B300/GB300) — never AMD; AMD FP4 chips must be `disabled`.
- BF16 / FP8 work on NVIDIA and AMD. FP8 configs adding `--kv-cache-dtype fp8_e4m3` should
  note the accuracy trade-off.

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
