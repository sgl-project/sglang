---
name: cookbook-add-model
description: Add a new model to the SGLang Cookbook (docs_new/, Mintlify), config-driven format — instantiate the model-agnostic template into a per-model config (+ benchmarks) JSX under src/snippets/configs/, an MDX page, the docs.json nav entry, NEW-tag hygiene, and the homepage vendor card. Interactive, multi-phase. Run with /cookbook-add-model.
disable-model-invocation: true
---

# Add a model to the SGLang Cookbook

The cookbook is **config-driven**: two shared engines contain NO model-specific code —
`docs_new/src/snippets/_deployment.jsx` (the 5-dim deploy matrix) and
`_playground.jsx` (the diff-based override Playground). Adding a model = adding **data**:
a per-model `config` (+ optional `benchmarks`) consumed by both engines, plus an MDX page
that imports them. No engine edits.

**Instantiate the model-agnostic template** (NOT a clone of any live cookbook — the
template is decoupled and covers all hardware + all axes):
- `templates/config.jsx.tmpl` → `docs_new/src/snippets/configs/<hf-org>/<model-slug>.jsx`
- `templates/benchmarks.jsx.tmpl` → `…/<model-slug>-benchmarks.jsx` (skip if no numbers)
- `templates/page.mdx.tmpl` → `docs_new/cookbook/<category>/<Vendor>/<ModelName>.mdx`

The template uses explicit `__TOKEN__` placeholders; you fill them, prune what the model
lacks, and replace the EXAMPLE cells with verified recipes. DeepSeek-V4 is a populated
*instance* you can consult, but is not the template.

**Deep references (read on demand, don't inline):**
- [references/authoring-reference.md](references/authoring-reference.md) — field-by-field config / cells / playground / MDX contract.
- [references/mintlify-authoring.md](references/mintlify-authoring.md) — MDX rules (forbidden syntax, JSX tables, labeled fences) + invocation-example patterns. Read before writing §1–§3 prose.
- [references/engine-axis.md](references/engine-axis.md) — adding a new Playground feature axis (rare engine work).

## Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────┐
│  cookbook/<category>/<Vendor>/<Model>.mdx                       │
│  import { Deployment } from "/src/snippets/_deployment.jsx";    │
│  import { Playground } from "/src/snippets/_playground.jsx";    │
│  import { config }     from "/src/snippets/configs/.../X.jsx";  │
│  <Deployment config={config} />   <Playground config={config} />│
└─────────────────────────────────────────────────────────────────┘
              │ (config passed as React prop)
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  src/snippets/configs/<vendor>/<model>.jsx                      │
│  export const config = {                                        │
│    supportedHardware, variants, quantizations, strategies, ...  │
│    cells: [ { match:{hw,variant,quant,strategy,nodes},          │
│              env:[...], flags:[...] }, ... ],   // 5-dim matrix  │
│    playgroundFeatures: { attention, moe, parsers, ... },        │
│  };                                                             │
└─────────────────────────────────────────────────────────────────┘
              │ (consumed by BOTH engines — no model code in engines)
              ▼
┌──────────────────────────────────┬──────────────────────────────┐
│  _deployment.jsx                 │  _playground.jsx             │
│  Renders the verified matrix;    │  Renders override chips +    │
│  one cell → its env/flags.       │  diff against the cell.      │
└──────────────────────────────────┴──────────────────────────────┘
```

The two widgets stay in sync via: the **URL hash** (deploy mirrors its selection;
playground reads it), the **`sglang-deploy-sel` custom event** (deploy dispatches on
every change; playground listens — `replaceState` doesn't fire `hashchange`), and the
shared **`sglang-deploy-env` localStorage key** (HOST/PORT placeholders).

> The template is **autoregressive**. Diffusion / omni pages follow their own category
> structure — don't force the config-driven template on them; still obey the Mintlify /
> NEW-tag / docs.json / category-card / validation rules below.

---

**Interactive, multi-step workflow. Collect inputs incrementally — don't ask for
everything upfront.** The real work is the verified `cells[]` recipes + measured
benchmarks (Phases 2 + 4); everything else is filling the template.

## Phase 1 — Collect inputs

1. **Model card** — HuggingFace repo/URL. **Fetch the page** and extract description,
   param count, architecture, context length, **license**. (Fetching guards factual bugs
   like an off-by-a-few-B param count.) If the model isn't public, ask the user.
2. **Variants / quantizations** — keep separate: variants are size/mode (e.g. Flash/Pro,
   Instruct/Thinking); quantizations come from the HF card / linked repos (BF16/FP8/FP4/…).
   Default to BF16 when a full-precision repo exists.
3. **Tested hardware + parallelism** — which platforms are actually tested, and TP/EP/DP
   for each. List only tested hw (unlisted greys out).
4. **Verified launch recipes** — the full `sglang serve` flags per
   (hw × variant × quant × strategy × nodes) combo → these become `cells[]`. Rewrite any
   `python -m sglang.launch_server` to `sglang serve` form.
5. **Pre-flight**: `gh pr list --repo sgl-project/sglang --search "<model>"` (dup check).

**Hardware reference** (matches the engine's `HARDWARE_CATALOG`):

| Platform | Vendor | VRAM | Docker image |
|---|---|---|---|
| H100 | NVIDIA | 80GB | `lmsysorg/sglang:<ver>` |
| H200 | NVIDIA | 141GB | `lmsysorg/sglang:<ver>` |
| B200 | NVIDIA | 192GB | `lmsysorg/sglang:<ver>` |
| B300 | NVIDIA | 288GB | `lmsysorg/sglang:<ver>` (or `-cu130` when required) |
| GB200 | NVIDIA | 192GB | `lmsysorg/sglang:<ver>` (or `-cu130`) |
| GB300 | NVIDIA | 288GB | `lmsysorg/sglang:<ver>` (or `-cu130`) |
| MI300X | AMD | 192GB | `lmsysorg/sglang:<ver>-rocm720-mi30x` |
| MI325X | AMD | 256GB | `lmsysorg/sglang:<ver>-rocm720-mi30x` |
| MI350X | AMD | 288GB | `lmsysorg/sglang:<ver>-rocm720-mi35x` |
| MI355X | AMD | 288GB | `lmsysorg/sglang:<ver>-rocm720-mi35x` |

- **TP sizing** (sanity-check recipes): `weight_GB / gpu_mem`, round up to a power of 2,
  ~20–30% headroom. BF16 ≈ params×2 GB, FP8 ≈ ×1, FP4 ≈ ×0.5. MoE → **total** weight, not
  active params. FP4 is Blackwell-only (B200/B300/GB200/GB300). GB200/GB300 single-node
  hosts are typically **4 GPUs** (TP=4 ceiling).
- **Platform flags**: Blackwell may need `--attention-backend trtllm_mha`; AMD typically
  needs `--attention-backend triton` + env `SGLANG_USE_AITER=1` /
  `SGLANG_ROCM_FUSED_DECODE_MLA=0` (check AITER TP constraints, e.g. `heads_per_gpu % 16 == 0`).
- **EP** (MoE): 8-GPU NVIDIA `--tp 8 --ep 8`; AMD `EP = TP`; small NVIDIA (TP≤4) omit
  `--ep` unless benchmarked. (The template's AMD example cell shows these.)

## Phase 2 — Instantiate the template

1. **Copy** the three template files to their target paths (above). Note the two
   vendor-folder conventions: under `configs/` the folder is the **HuggingFace org**
   (`deepseek-ai`); under `cookbook/` it's the **display vendor** (`DeepSeek`).
2. **Replace every `__TOKEN__`**: `__MODEL_DISPLAY__`, `__MODEL_SLUG__`, `__HF_ORG__`,
   `__HF_REPO__`, `__REASONING_PARSER__`, `__TOOLCALL_PARSER__`, `__ONE_LINER__`. Verify
   none remain: `grep -rn '__[A-Z_]*__' <new files>`.
3. **Prune** to what the model supports (delete, don't stub) — using
   [references/authoring-reference.md](references/authoring-reference.md):
   - `supportedHardware` + the EXAMPLE cells: keep your tested families; **delete the
     `mi*` ids + AMD example cell if no AMD recipe**, etc.
   - `playgroundFeatures` axes: delete `megamoe` (non-Blackwell-MoE), `hisparse`
     (non-DSA), `pdDisagg`/`router` (no PD), the `parsers` axis (no parsers), etc.
   - `quantizations` / `variants`: drop what the model doesn't ship; collapse `variants`
     to single `default` if there's no variant axis (then drop the `variant` half of
     `modelNames`/`defaultAccuracy` keys).
4. **Fill `cells[]`** with the verified recipes from Phase 1 (replace every EXAMPLE cell;
   set `verified: true` only on tested combos), and `modelNames` with real HF slugs,
   `dockerImages` for your hw, `multiNodeHints` only for fabric-specific hw (e.g. gb200).

### Site-wiring (do all three)

- **`docs_new/docs.json`** — add the page under Cookbook → `<category>` → `<Vendor>`, at
  the **top** of that vendor's `pages` (root-relative, no `.mdx`:
  `cookbook/<category>/<Vendor>/<Model>`). New vendor group → insert in the section's
  local ordering.
- **NEW-tag hygiene** — the new page keeps `tag: NEW` (from the template). Scan the
  vendor dir for existing NEW and strip it from siblings; verify ≤1:
  `grep -rn 'tag: NEW' docs_new/cookbook/<category>/<Vendor>/` → at most one result. (Scan
  files; don't assume the first `docs.json` entry holds NEW.)
- **Homepage card** — `docs_new/cookbook/<category>/intro.mdx`: if the org already has a
  `<Card>`, update only its `href` (keep `img`). If the org is **new**, add a `<Card>`
  **and tell the user to provide `docs_new/cards/logos/<org-slug>.png`** — never invent or
  copy a logo.

## Phase 3 — Validate

```bash
cd docs_new
mint validate        # frontmatter, missing nav entries, MDX/JSX errors
mint broken-links
mint dev             # visual smoke test at http://localhost:3000/cookbook/<category>/<Vendor>/<Model>
```

Spot-check: cells render sensible commands; URL-hash nav persists across reload; the
Playground inherits the Deploy selection live; each axis toggle produces the expected
diff; Docker mode wraps in `docker run` with the right image; multi-node cells emit the
hints + `--nnodes N`; cURL resolves the model name; the NEW badge shows on the new page
and is gone from same-vendor siblings; the homepage card points to the new model.

## Phase 4 — Interactive testing

The user deploys each cell, runs the benches, and pastes results; you fill the data:
- mark each tested `cells[]` entry `verified: true` (absent = yellow/unverified badge);
- fill the `<model>-benchmarks.jsx` entries (one per cell `match`) with measured
  speed/accuracy; set model-level `defaultAccuracy` per variant. Leave a cell's entry as a
  bare `match` stub if it has no numbers yet (the card shows "pending").

## Phase 5 — Prose & config tips

**Read [references/mintlify-authoring.md](references/mintlify-authoring.md) first** (it
carries the parser-output-shape / thinking-mode / Output-Example / no-hardcoded-sampling
rules + the Mintlify forbidden-syntax list). Then rewrite the MDX prose from the HF card +
user notes: §1 Model Introduction (description, links, params, license, variants table),
§2 Configuration Tips (hw-specific tuning, caveats), §3 Advanced Usage (Reasoning /
Tool-Calling / HiCache — keep only what applies; match the reasoning example to the
parser's output shape; each runnable block gets an `**Output Example:**`).

## Phase 6 — Review

```
/cookbook-review-pr <PR number>
```

## Git workflow

Always branch — never commit to main directly.

```bash
git checkout -b add-<model>-cookbook
git add docs_new/src/snippets/configs/<hf-org>/<slug>.jsx \
        docs_new/src/snippets/configs/<hf-org>/<slug>-benchmarks.jsx \
        docs_new/cookbook/<category>/<Vendor>/<Model>.mdx \
        docs_new/docs.json docs_new/cookbook/<category>/intro.mdx
git commit -m "Add <Display-Name> cookbook"
git push -u origin add-<model>-cookbook
gh pr create --title "Add <Display-Name> cookbook" --body "..."
```
