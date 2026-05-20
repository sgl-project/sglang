---
name: cookbook-add-model
description: Add a new model to the SGLang Mintlify cookbook — MDX page, reference-derived autoregressive config-generator JSX snippet, docs.json registration, NEW tag handling, and category-intro card update.
disable-model-invocation: true
---

# Add New Model to SGLang Cookbook (Mintlify)

Interactive, multi-step workflow. Collect inputs incrementally; do not ask for everything upfront.

The cookbook lives in `docs_new/` and uses Mintlify (`.mdx`, `docs.json`, `mint`). There is no YAML model corpus and no shared Docusaurus `ConfigGenerator` component. New autoregressive deployment snippets are self-contained `.jsx` files stamped from [templates/config-generator.jsx.tmpl](templates/config-generator.jsx.tmpl). Diffusion and omni pages keep their category-specific structures unless the user explicitly asks to unify them.

## Phase 1: Collect Inputs

Ask for and record:

1. **Model card** — HuggingFace model name or URL. Fetch it to confirm description, capabilities, parameter count, architecture, context length, variants, and license. If the model is private/unreleased, ask the user for those facts.
2. **Cookbook category** — `autoregressive`, `diffusion`, or `omni`. Use the unified generator template for new autoregressive pages first; for diffusion/omni, follow the closest existing category page and snippet.
3. **Model variants and quantizations** — keep model variant and quantization separate:
   - model variants are size/architecture/mode choices such as `397B-A17B`, `35B-A3B`, `Instruct`, or `Thinking`;
   - quantization choices come from the HuggingFace model card or linked model repos, such as `BF16`, `FP8`, `NVFP4`, `MXFP4`, or `AWQ`;
   - default to `BF16` for each model variant when a BF16/full-precision repo exists;
   - each model variant may expose a different set of quantizations.
4. **Reference deployment command per variant + quantization** — each `(model variant, quantization)` pair needs at least one user-provided `sglang serve --model-path ...` command. If the source uses `python -m sglang.launch_server`, rewrite it to `sglang serve` form and confirm the rewritten command. Parse and preserve:
   - baseline hardware, quantization, `tp`, `ep`, `dp`, `--mem-fraction-static`, optional env prefixes, required flags, parsers, chat template, MTP/EAGLE, Mamba, KV/cache, and other feature flags;
   - default `strategies` and `features` for that variant + quantization;
   - which flags are required for correctness vs optional optimizations.
5. **Feature/strategy applicability** — infer only from the reference command, model card, SGLang docs, or user confirmation. If it is unclear whether MTP, Mamba cache, KV FP8, parser, chat template, DP attention, EP, or an env prefix applies to a model/hardware/variant, ask before adding it.
6. **Verified matrix** — for each tested `(variant x quantization x hardware)` combo, record `{ env, tp, ep, mem, flags, strategies, features }`. Untested combos can be auto-estimated from that quantization's reference command, but only for single-node commands.
7. **Optimized commands** — if the user supplies hand-tuned launches, store them under the verified combo as verbatim `optimizedCommands`. Optimized is a deployment strategy option, not a feature option. Support at least:
   - `default`: optimized command without MTP;
   - `mtp`: optimized command with MTP/EAGLE.
   Never rewrite an optimized command. If only an optimized command is provided, use it as capability evidence and as `optimizedCommands`, then derive the best non-optimized baseline from its parsed fields.
8. **SGLang version** — used in benchmark metadata and Docker image tags.
9. **Tested hardware platforms** — show the full list below and let the user pick. Only mark a combo verified when it was actually tested.

Hardware reference:

| Platform | Vendor | Memory | Docker Image |
|----------|--------|--------|--------------|
| A100     | NVIDIA | 80GB   | `lmsysorg/sglang:<ver>` |
| H100     | NVIDIA | 80GB   | `lmsysorg/sglang:<ver>` |
| H200     | NVIDIA | 141GB  | `lmsysorg/sglang:<ver>` |
| B200     | NVIDIA | 180GB  | `lmsysorg/sglang:<ver>` |
| B300     | NVIDIA | 275GB  | `lmsysorg/sglang:<ver>` or `-cu130` when required |
| GB200    | NVIDIA | 192GB  | `lmsysorg/sglang:<ver>-cu130` |
| GB300    | NVIDIA | 275GB  | `lmsysorg/sglang:<ver>-cu130` |
| MI300X   | AMD    | 192GB  | `lmsysorg/sglang:<ver>-rocm720-mi30x` |
| MI325X   | AMD    | 256GB  | `lmsysorg/sglang:<ver>-rocm720-mi30x` |
| MI350X   | AMD    | 288GB  | `lmsysorg/sglang:<ver>-rocm720-mi35x` |
| MI355X   | AMD    | 288GB  | `lmsysorg/sglang:<ver>-rocm720-mi35x` |

TP calculation for auto-generation:

- `model_weight_GB / gpu_mem_GB`, round up to nearest power of 2. Leave 20-30% headroom.
- BF16 ≈ params * 2 GB, FP8 ≈ params * 1 GB, FP4 ≈ params * 0.5 GB
- FP4 is Blackwell-only (B200/B300)
- MoE models: use total weight size (all experts), not active params

Platform notes:

- Blackwell/Grace-Blackwell may need `--attention-backend trtllm_mha`; GB200/GB300 need `-cu130` Docker tag (CUDA 13) and typically max out at 4 GPUs per node.
- AMD typically needs `--attention-backend triton`; model-specific AMD env vars include `SGLANG_USE_AITER=1` and `SGLANG_ROCM_FUSED_DECODE_MLA=0`.
- AMD MoE/MLA: check AITER kernel constraints on TP (e.g., `heads_per_gpu % 16 == 0`)
- Multi-node commands are never auto-generated. If a model does not fit within one node, the generator should output not available and ask for a verified multi-node command.

Expert Parallelism (EP) for MoE models — common patterns observed:

- 8-GPU NVIDIA: `--tp 8 --ep 8`
- AMD (all TP sizes): `EP = TP` (e.g., `--tp 4 --ep 4`)
- Smaller NVIDIA configs (TP≤4): omit `--ep` unless explicitly benchmarked — don't blindly scale EP

Before starting files, check duplicate work:

```bash
gh pr list --search "<model name>"
```

## Phase 2: Create Files

Read the relevant current files before writing:

- **Template**: [templates/config-generator.jsx.tmpl](templates/config-generator.jsx.tmpl) — source of truth for new autoregressive snippets.
- **Similar MDX**: pick a recent page under `docs_new/cookbook/<category>/`.
- **Category intro**: `docs_new/cookbook/autoregressive/intro.mdx`, `docs_new/cookbook/diffusion/intro.mdx`, or `docs_new/cookbook/omni/intro.mdx`.
- **Navigation**: `docs_new/docs.json`.

### Step 1: Create the MDX page

Path: `docs_new/cookbook/<category>/<Vendor>/<ModelName>.mdx`.

Required frontmatter:

See [references/frontmatter-example.mdx](references/frontmatter-example.mdx).

For **autoregressive** pages, use five top-level sections:

1. **Model Introduction** — concise intro, Key Features, Available Models when there are multiple entries, benchmark/feature tables as JSX tables, Recommended Generation Parameters, License, HF/blog links. If there is only one model entry, inline the HF link in the intro.
2. **SGLang Installation** — link to `../../../docs/get-started/install`; include a Docker Images by Hardware Platform JSX table for tested platforms. Example: [references/installation-table-example.jsx](references/installation-table-example.jsx).
3. **Model Deployment** — `### 3.1 Basic Configuration` embeds the snippet:
   see [references/deployment-snippet-embed-example.mdx](references/deployment-snippet-embed-example.mdx).
   Put imports in the MDX body, not frontmatter. Use absolute `/src/snippets/...` imports, never `@site/...`. Add `### 3.2 Configuration Tips`.
4. **Model Invocation** — include a canonical deployment command at top, then `### 4.1 Basic Usage` linking to `../../../docs/basic_usage/send_request`, and `### 4.2 Advanced Usage` with reasoning, tool calling, multimodal/tool-call subsections only when applicable. Each followed by an `**Output Example:**` + ```text block. Use `Pending update...` placeholders if the model isn't yet deployed.
5. **Benchmark** — order is strict: `### 5.1 Accuracy Benchmark`, then `### 5.2 Speed Benchmark`. `Pending update...` placeholders are acceptable for unfinished runs. Benchmark test-environment metadata (Hardware, Model quantization, TP, SGLang version, Docker image) must match a quantization actually listed in Section 1 — `(BF16)` on a model that only released INT4 is a factual bug.

Benchmark commands — each benchmark has two pieces. The **deploy** (server launch at the top of the section) uses `sglang serve`. The **bench workload** uses `python3 -m sglang.bench_serving` (never bare `python -m`).

Accuracy benchmark commands:

**SGLang built-in benchmarks** (lightweight, no extra deps):
- GSM8K: `python3 benchmark/gsm8k/bench_sglang.py --port <port>`
- MMLU: `python3 benchmark/mmlu/bench_sglang.py --port <port>`
- MMMU: `python3 benchmark/mmmu/bench_sglang.py --port <port>` — uses a universal answer regex that works across models. Don't use model-specific parsing (e.g., `<|begin_of_box|>`) as it breaks with standard answer formats. Note: this is plain MMMU, not MMMU-Pro or MMMU-Pro-Vision — those are separate benchmarks.
- Latency: `python3 -m sglang.bench_serving --backend sglang --num-prompts 10 --max-concurrency 1 ...`
- Throughput: `python3 -m sglang.bench_serving --backend sglang --num-prompts 1000 --max-concurrency 100 ...`

**Heavier reasoning/MCQ suites** via [NVIDIA NeMo-Skills](https://github.com/NVIDIA-NeMo/Skills) (GPQA Diamond, AIME, MMLU-Pro, etc.):
- `ns prepare_data <dataset>` then `ns eval --server_type=openai --server_address=http://localhost:30000/v1 --model=<hf-path> --benchmarks=<name>:<epochs> ...`
- For reasoning-mode models, pass `++parse_reasoning=True` so the grader sees the answer, not the `<think>` content.
- MMLU-Pro is 10-choice — use `++prompt_config=eval/aai/mcq-10choices` (not the 4-choice config).
- Give extended-thinking models enough budget: `++inference.tokens_to_generate=120000` is typical. A 32K cap often produces a spiky "No Answer" rate; call this out in the results if it happens.
- Report results as a table with `pass@1 (avg-of-N)`, `majority@N`, `pass@N`, plus a `No Answer` column if non-zero. Example: [references/benchmark-results-table-example.jsx](references/benchmark-results-table-example.jsx).

Don't add multiple scenarios or concurrency levels unless asked.

For **diffusion** and **omni** pages, do not force the autoregressive five-section/generator structure. Follow the closest existing category page, but still obey Mintlify formatting, NEW tag, `docs.json`, category card, and validation rules.

**JSX table pattern**: use JSX tables for every table. Markdown pipe tables are forbidden for new pages. Base pattern: [references/jsx-table-pattern.jsx](references/jsx-table-pattern.jsx).

For 3-col or 5-col tables, adjust `<colgroup>` and alternate column background colors.

Mintlify rules:

- Allowed docs components: `<Card>`, `<CardGroup>`, `<Note>`, `<Tip>`, `<Warning>`, `<Info>`, `<Accordion>`, `<AccordionGroup>`, `<Steps>`, `<Step>`, `<Tabs>`, `<Tab>`, `<CodeGroup>`, `<Frame>`, `<Icon>`.
- `<CardGroup>`/`<Card>` are for intro pages only, not individual model pages.
- Forbidden syntax: Docusaurus admonitions (`:::note` etc.), `@site/...`, `@theme/...`, GitHub alert blocks (`> [!NOTE]`), markdown pipe tables, inline `<details>/<summary>`, and unknown components.
- Use labeled fences: ` ```python Example`, ` ```shell Command`, ` ```bash Command`, ` ```text Output`.
- When nesting code fences inside another code fence, use four backticks for the outer fence.

Content rules carried over from the old cookbook:

- Deploy commands use `sglang serve --model-path ...`. Benchmark workload commands use `python3 -m sglang.bench_serving ...`; built-in accuracy scripts use `python3 benchmark/...`.
- Keep one canonical deployment command when possible; avoid duplicating launch commands across invocation and benchmark sections. Refer back to the canonical command or state the exact generator selection.
- Launch port must match client, curl, and benchmark port on the same page; default to `30000`.
- Every runnable invocation code block needs `**Output Example:**` followed by a ` ```text Output` block. Real output is preferred; `Pending update...` is acceptable only with user acknowledgement.
- Do not hardcode sampling params (`temperature`, `top_p`) in sample code; SGLang uses `generation_config.json` defaults. Listing recommended params in Section 1 is fine.
- Benchmark metadata quantization must match a variant listed in Section 1.
- License must match the actual HF license; never copy from another model.
- No Google Drive images and no shell no-ops like `export VAR=${VAR}`.
- Reasoning parser examples must match the parser output shape:
  - separate field parsers (`kimi_k2`, most qwen/glm): print `reasoning_content` and `content`;
  - inline tag parsers (`minimax-append-think`): parse `<think>...</think>` from `content`.
- Thinking-mode tool-call follow-up may put final text in `reasoning_content`; print both `reasoning_content` and `content`.
- Format raw API objects into readable Reasoning / Content / Tool Calls blocks.

Notes:
- Nested code blocks: use four backticks ```````` for the outer block
- Don't hardcode sampling params (`temperature`, `top_p`) in sample code — SGLang uses `generation_config.json` defaults. (It's fine to list "Recommended Generation Parameters" informationally in Section 1.)
- Hybrid reasoning models: show both thinking-on (default) and thinking-off (`enable_thinking: False`) examples
- Separate Instruct/Thinking variants (e.g., Qwen3-Next): model name changes, handled by ConfigGenerator
- Format raw API response objects (e.g., `ChatCompletionMessage(...)`) into readable structured output
- Tool-call follow-up on thinking-mode models: the final assistant response may put text in `reasoning_content` instead of (or in addition to) `content`. When writing the example, print both so the output isn't misleadingly `None`.
- Invocation section output format: immediately after each code block, add `**Output Example:**` followed by a ```text fenced block with the real run output. Keep the text verbatim from the server — don't paraphrase.

### Step 2: Update `docs.json`, NEW tags, and category intro card

`docs_new/docs.json`:

- Add the new page under the Cookbook tab, in the correct category and vendor group.
- Put the new page at the top of that vendor's `pages` list.
- If the vendor group is new, insert it in the category list in the local ordering style used by that section.

NEW tags:

- The new page must have `tag: NEW`.
- Scan the same `docs_new/cookbook/<category>/<Vendor>/` directory for existing `tag: NEW` and remove it from siblings.
- Do not assume the current first `docs.json` page is the one holding NEW; scan the files.
- Verify: `grep -rn 'tag: NEW' docs_new/cookbook/<category>/<Vendor>/` returns at most one result.

Category intro card:

- Autoregressive: `docs_new/cookbook/autoregressive/intro.mdx`
- Diffusion: `docs_new/cookbook/diffusion/intro.mdx`
- Omni: `docs_new/cookbook/omni/intro.mdx`

If the org already has a `<Card>`, update only `href` to the new page and keep the existing `img`. If the org is new, add a card:
see [references/category-intro-card-example.jsx](references/category-intro-card-example.jsx).

Tell the user to provide `docs_new/cards/logos/<org-slug>.png`; do not invent or copy a logo.

### Step 3: Create the autoregressive config generator

Path: `docs_new/src/snippets/autoregressive/<name>-deployment.jsx`.

Stamp from [templates/config-generator.jsx.tmpl](templates/config-generator.jsx.tmpl). Fill in `CONFIG` and the exported component name. Keep helpers/render logic unchanged unless the model truly needs a template-level behavior change.

Template behavior:

- Row 1: all 11 GPUs grouped by NVIDIA / AMD. Each hardware option should show its VRAM subtitle from the hardware reference table (for example `80GB`, `141GB`, `275GB`). Green dot means verified. Unverified combos have no dot and no warning banner.
- Row 2: model variant buttons only. Do not mix quantization into the variant label unless the model itself is quantization-specific.
- Row 3: quantization buttons for the selected model variant. Quantization choices come from the HuggingFace model card / linked repos. BF16 is the default when available.
- Row 4: strategy checkboxes. TP is required. EP shows only for applicable MoE configs. MTP shows only when confirmed by the selected variant/quantization. Optimized appears here, not in features, and is enabled only when the selected strategy profile has a matching verbatim optimized command. PP is intentionally hidden for now.
- Row 5: feature buttons. Parsers are parameterized from `CONFIG.model.toolCallParser` and `CONFIG.model.reasoningParser`; no `<parser>` placeholders. Parser buttons default enabled when configured. Mamba/KV FP8/chat template appear only when confirmed and default from the selected quantization reference or explicit CONFIG.
- Layout: keep the generator inside the normal Mintlify content column. Do not use negative margins or viewport-width breakout styles that can overlap the left nav or right TOC/sidebar. Hardware rows may wrap within the content column.
- `modelVariants[].quantizations[].reference` is mandatory. It stores parsed fields from the user-provided reference command and is the basis for unverified auto-generation.
- `reference.env`, `estimates[hardwareId].env`, and `verified[variantId][quantizationId][hardwareId].env` are optional arrays of `KEY=VALUE` strings. Most combos omit them; include them only when the author-provided command actually needs them.
- `verified[variantId][quantizationId][hardwareId]` overrides reference-derived estimates for tested combos.
- `optimizedCommands.default` and `optimizedCommands.mtp` are optional verbatim commands on verified combos.

Command generation rules:

- Verified combo: use verified data.
- Unverified combo: derive from the selected variant + quantization reference, adjusting only single-node TP, mem, and platform-required flags.
- Single-node maxTP defaults to 8; GB200/GB300 maxTP is 4.
- Non-optimized env prefixes come from resolved data in this order: `verified.env`, else `estimate.env`, else `reference.env`.
- For non-optimized MTP commands, prepend `SGLANG_ENABLE_SPEC_V2=1` unless it is already present.
- Do not invent non-MTP env prefixes. Preserve them only when explicitly provided or confirmed.
- If estimated TP exceeds maxTP, show:
  ```text
  # Not available: estimated deployment requires multiple nodes.
  # Please provide a verified multi-node deployment command.
  ```
- Do not auto-generate multi-node commands.
- Single-node DP emits `--dp <tp> --enable-dp-attention`.
- Optimized strategy:
  - with MTP selected, emit `optimizedCommands.mtp` if present;
  - without MTP, emit `optimizedCommands.default` if present;
  - if the selected profile has no optimized command, keep the toggle disabled or show an explicit not-available message;
  - never append, remove, or rewrite flags inside optimized commands.

Doc-generator parity:

- Every launch command shown in the MDX must match the generator output for the same variant/hardware/strategy/features.
- If a flag appears in generated AMD/B200/GB commands, the corresponding documented command must include it too.

## Phase 3: Validate

Run from `docs_new/`:

```bash
mint validate
mint broken-links
mint dev
```

Visual checks:

- NEW badge renders on the new page and is gone from same-org siblings.
- Category intro card points to the new model.
- Generator shows all 11 GPUs grouped by vendor.
- Verified combos show green dots; unverified combos show no warning banner.
- Too-large single-node estimates show the not-available command.
- Parser/features hide, show, enable, and default according to CONFIG/reference.
- Optimized default and optimized MTP emit the correct verbatim commands.

## Phase 4: Interactive Testing

User deploys the model, runs scripts, and pastes results. Replace `Pending update...` placeholders with actual outputs:

1. Invocation results: basic generation, streaming/reasoning, tool calling, multimodal/tool-call if applicable.
2. Accuracy benchmarks: GSM8K, MMLU, MMMU, GPQA/AIME/MMLU-Pro if applicable.
3. Speed benchmarks: latency and throughput.

## Phase 5: Configuration Tips

Ask for recommended settings, known issues, DP attention tradeoffs, hardware-specific `--mem-fraction-static`, parser/template nuances, and any unsupported hardware/feature combinations. Add them to `### 3.2 Configuration Tips`.

## Phase 6: Review Checklist

Can be triggered with `/cookbook-add-model review`.

**MDX**

- [ ] Frontmatter has `title`, `metatags.description`, `tag: NEW`.
- [ ] Same category/vendor directory has at most one `tag: NEW`.
- [ ] Category intro `<Card href>` points to the new model.
- [ ] New org card has a real planned logo path and the user was told to provide the asset.
- [ ] Imports use absolute `/src/snippets/...` paths in the MDX body.
- [ ] No markdown pipe tables, Docusaurus syntax, GitHub alert blocks, `<details>`, `@site`, or unknown components.
- [ ] Code fences are labeled; nested fences use four backticks outside.
- [ ] Runnable invocation blocks have `**Output Example:**` plus ` ```text Output`.
- [ ] Deploy commands use `sglang serve --model-path`; bench workload commands use `python3 -m sglang.bench_serving`.
- [ ] One canonical deployment command is reused or referenced; no unnecessary duplicate launch blocks.
- [ ] Ports match across launch, client/curl, and benchmarks.
- [ ] Benchmark order is accuracy first, speed second.
- [ ] Benchmark metadata quantization matches a listed variant.
- [ ] Reasoning/tool-call examples match actual parser output shape.
- [ ] License matches the HF card.
- [ ] No hardcoded sampling params, Google Drive images, or `export VAR=${VAR}` no-ops.

**docs.json**

- [ ] New model is at the top of its vendor `pages` array.
- [ ] New vendor group is inserted in the section's local ordering style.

**Config generator**

- [ ] Stamped from `templates/config-generator.jsx.tmpl`; only CONFIG/export name changed unless justified.
- [ ] Exactly one top-level `export const <Name>Deployment = () => { ... }`.
- [ ] No bare module imports.
- [ ] All CONFIG, helpers, styles, and render code are closure-scoped inside the component.
- [ ] All 11 GPUs are present and grouped NVIDIA / AMD.
- [ ] Hardware options display the correct VRAM subtitle from the hardware reference.
- [ ] Every variant has a parsed `reference` command.
- [ ] `verified` covers only tested combos.
- [ ] Optional env prefixes are preserved only where confirmed.
- [ ] Unverified combos have no green dot and no warning banner.
- [ ] Over-maxTP single-node estimates show the not-available command.
- [ ] Parser and feature flags are parameterized; no `<parser>` placeholders.
- [ ] MTP/Mamba/KV FP8/chat template are shown only when confirmed.
- [ ] Optimized default and optimized MTP commands are verbatim and selected by strategy profile.
- [ ] No dead code or unreachable feature/strategy branches.
- [ ] Doc-generator parity is checked.

**Validation**

- [ ] `mint validate` passes.
- [ ] `mint broken-links` passes.
- [ ] `mint dev` visual inspection passes.

## Git Workflow

Always create a branch; never commit directly to main.

Example sequence: [references/git-workflow-example.sh](references/git-workflow-example.sh).

Do not commit generated YAML or compiled files; the new cookbook has none.
