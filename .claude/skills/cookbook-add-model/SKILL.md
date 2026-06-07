---
name: cookbook-add-model
description: Add a new model to the SGLang Cookbook (Mintlify) under docs_new/, including the MDX page, deployment snippet, navigation entry, and homepage card. Run with /cookbook-add-model.
disable-model-invocation: true
---

# Add New Model to SGLang Cookbook

The cookbook lives in `docs_new/` (Mintlify). Pages are `.mdx`, deployment widgets are JSX snippets under `docs_new/src/snippets/autoregressive/`, and navigation is configured in `docs_new/docs.json`.

Interactive, multi-step workflow. Collect inputs incrementally — don't ask for everything upfront.

## Phase 1: Collect Initial Inputs

Ask the user for:

1. **Model Card** — HuggingFace model name or URL (e.g., `Qwen/Qwen3-Coder-Next`). Fetch the page to extract description, capabilities, etc. If the model isn't public yet, ask the user to paste what they know (name, param count, architecture, capabilities, context length).
2. **Model Variants** — Multiple sizes (e.g., 480B/30B) or quantizations (BF16/FP8)? Which to include? This affects the deployment snippet's options and doc examples. See `docs_new/src/snippets/autoregressive/qwen3-next-deployment.jsx` and `docs_new/src/snippets/autoregressive/devstral-2-deployment.jsx` for multi-variant patterns.
3. **Deployment Command** — Full `sglang serve --model-path` command with all flags (tp, dp, ep, etc.). Not `python -m sglang.launch_server` (deprecated, issue #33). If the model card provides one, use it as a starting point but verify format.
4. **SGLang Version** — Version being tested (e.g., `v0.5.10`). Used in the Docker image tags listed in Section 2 of the doc.
5. **Hardware Platforms** — Which platforms are tested? Show the full list (A100, H100, H200, B200, B300, GB300, MI300X, MI325X, MI350X, MI355X) and let the user pick. Only include tested platforms — don't assume anything. For each, confirm TP degree and any platform-specific flags. GB300 on a typical single-node host ships with 4 GPUs, so TP=4 is the practical ceiling there — confirm the actual node topology with the user rather than assuming.

## Phase 2: Create Scaffolding

Read ALL reference templates first, then create files.

### Reference Templates

- **Doc**: Find a similar model under `docs_new/cookbook/autoregressive/` (e.g., `Qwen/Qwen3-Next.mdx`, `DeepSeek/DeepSeek-V4.mdx`, `Mistral/Devstral-2.mdx`).
- **Snippet**: Similar deployment widget under `docs_new/src/snippets/autoregressive/` (e.g., `qwen3-next-deployment.jsx`, `devstral-2-deployment.jsx`).
- **Navigation**: `docs_new/docs.json` — find the existing vendor group under `navigation` → Cookbook → Autoregressive Models.
- **Vendor card**: `docs_new/cookbook/autoregressive/intro.mdx` — `<CardGroup>` with one `<Card>` per vendor.

### Key Rules

- The cookbook is **Mintlify**, not Docusaurus. Pages are `.mdx`, navigation is `docs.json`, build uses `mint`. There is **no `data/models/` YAML pipeline** in this repo — don't create or edit any.
- Snippet files are FLAT under `docs_new/src/snippets/autoregressive/` — kebab-case filename ending in `-deployment.jsx`, single named export `<ModelName>Deployment` (PascalCase + `Deployment`). No vendor subdirs, no shared base class.
- All commands use `sglang serve` — never `python -m sglang.launch_server` or `python3 -m sglang.launch_server`.
- All files end with a trailing newline.
- Standard server port is **30000** (used in launch commands, curl, client `base_url`, and bench commands on the same page).
- Check open PRs first (`gh pr list --search "<model name>"`) to avoid duplicate work.

### Hardware Reference

Only include platforms the user has actually tested.

| Platform | Vendor | Memory | Docker Image |
|----------|--------|--------|--------------|
| A100     | NVIDIA | 80GB   | `lmsysorg/sglang:<ver>` |
| H100     | NVIDIA | 80GB   | `lmsysorg/sglang:<ver>` |
| H200     | NVIDIA | 141GB  | `lmsysorg/sglang:<ver>` |
| B200     | NVIDIA | 180GB  | `lmsysorg/sglang:<ver>` |
| B300     | NVIDIA | 275GB  | `lmsysorg/sglang:<ver>` (or `-cu130` for CUDA 13) |
| GB300    | NVIDIA | 275GB  | `lmsysorg/sglang:<ver>-cu130` (Grace-Blackwell, CUDA 13 required; typical single-node host = 4 GPUs → TP=4) |
| MI300X   | AMD    | 192GB  | `lmsysorg/sglang:<ver>-rocm720-mi30x` |
| MI325X   | AMD    | 256GB  | `lmsysorg/sglang:<ver>-rocm720-mi30x` |
| MI350X   | AMD    | 288GB  | `lmsysorg/sglang:<ver>-rocm720-mi35x` |
| MI355X   | AMD    | 288GB  | `lmsysorg/sglang:<ver>-rocm720-mi35x` |

**TP calculation**: `model_weight_GB / gpu_mem_GB`, round up to nearest power of 2. Leave 20-30% headroom.
- BF16 ≈ params * 2 GB, FP8 ≈ params * 1 GB, FP4 ≈ params * 0.5 GB
- FP4 is Blackwell-only (B200/B300)
- MoE models: use total weight size (all experts), not active params

**Platform-specific flags** (only add if tested):
- Blackwell (B200/B300/GB300): may need `--attention-backend trtllm_mha`; GB300 needs the `-cu130` Docker tag (CUDA 13)
- AMD: typically needs `--attention-backend triton`
- AMD env vars: `SGLANG_USE_AITER=1`, `SGLANG_ROCM_FUSED_DECODE_MLA=0`
- AMD MoE/MLA: check AITER kernel constraints on TP (e.g., `heads_per_gpu % 16 == 0`)

**Expert Parallelism (EP)** for MoE models — common patterns observed:
- 8-GPU NVIDIA: `--tp 8 --ep 8`
- AMD (all TP sizes): `EP = TP` (e.g., `--tp 4 --ep 4`)
- Smaller NVIDIA configs (TP≤4): omit `--ep` unless explicitly benchmarked — don't blindly scale EP

### Step 1: Create the MDX page

Create `docs_new/cookbook/autoregressive/<Vendor>/<ModelName>.mdx` with this frontmatter:

```yaml
---
title: <Model Name>
metatags:
    description: "Deploy <Model Name> with SGLang - <one-line value prop>."
---
```

Optional frontmatter:
- `tag: NEW` — only for genuinely new launches; remove the tag from older pages when adding a new one (don't accumulate). Search current `tag: NEW` usages with `grep -RlE "^tag: NEW" docs_new/cookbook/`.
- `sidebarTitle`, `icon`, `mode: wide` — match the surrounding pages in the same vendor folder.

Section structure (mirror an existing page like `Qwen/Qwen3-Next.mdx` or `Mistral/Devstral-2.mdx`):

- **Section 1: Model Introduction** — lean. Key Features (bullets), Benchmarks as a **table** (not bullets), Recommended Generation Parameters, License, HF/blog links. Don't duplicate an "Architecture" table from the HF card unless it adds info. If "Available Models" has only one entry, skip the list — inline the single HF link in the intro paragraph.
- **Section 2: SGLang Installation** — link to the [official install guide](../../../docs/get-started/install) and add a **Docker Images by Hardware Platform** table for the tested platforms. Example:
  ```
  | Hardware Platform                      | Docker Image                                  |
  | ---                                    | ---                                           |
  | NVIDIA A100 / H100 / H200 / B200       | `lmsysorg/sglang:<ver>`                       |
  | NVIDIA B300 / GB300                    | `lmsysorg/sglang:<ver>-cu130`                 |
  | AMD MI300X / MI325X                    | `lmsysorg/sglang:<ver>-rocm720-mi30x`         |
  | AMD MI350X / MI355X                    | `lmsysorg/sglang:<ver>-rocm720-mi35x`         |
  ```
- **Section 3: Model Deployment** — embed the deployment snippet + Configuration Tips. Example:
  ```mdx
  import { <ModelName>Deployment } from '/src/snippets/autoregressive/<modelname>-deployment.jsx';

  ### 3.1 Basic configuration

  <<ModelName>Deployment />

  ### 3.2 Configuration tips
  ...
  ```
- **Section 4: Model Invocation** — one documented deployment command at top, then test scripts (multimodal, reasoning, tool calling, mm+tool) each followed by an `**Output Example:**` + ```text block. Use `Pending update...` placeholders if the model isn't yet deployed.
- **Section 5: Benchmarks** — `Pending update...` placeholders are acceptable for unfinished runs. Benchmark test-environment metadata (Hardware, Model quantization, TP, SGLang version, Docker image) must match a quantization actually listed in Section 1 — `(BF16)` on a model that only released INT4 is a factual bug.

Benchmark commands — each benchmark has two pieces. The **deploy** (server launch at the top of the section) uses `sglang serve`. The **bench workload** uses `python3 -m sglang.bench_serving` (never bare `python -m`).

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
- Report results as a table with `pass@1 (avg-of-N)`, `majority@N`, `pass@N`, plus a `No Answer` column if non-zero:
  ```
  | Evaluation Mode    | Accuracy | No Answer |
  |--------------------|----------|-----------|
  | pass@1 (avg-of-8)  | 84.91%   | 3.54%     |
  | **majority@8**     | **88.89%** | 0.00%  |
  | pass@8             | 96.46%   | 0.00%     |
  ```

Keep benchmarks concise. Order: accuracy first, then speed. Don't add multiple scenarios or concurrency levels unless asked.

Notes:
- Use `.mdx` (not `.md`). Mintlify supports JSX components — use `<Note>`, `<Warning>`, `<Tabs>`, `<CardGroup>` etc. when they help. Default to plain markdown unless a component clearly improves the page.
- Nested code blocks: use four backticks ```````` for the outer block.
- Don't hardcode sampling params (`temperature`, `top_p`) in sample code — SGLang uses `generation_config.json` defaults. (It's fine to list "Recommended Generation Parameters" informationally in Section 1.)
- Hybrid reasoning models: show both thinking-on (default) and thinking-off (`enable_thinking: False`) examples.
- Separate Instruct/Thinking variants (e.g., Qwen3-Next): model name changes, handled by the deployment snippet.
- Format raw API response objects (e.g., `ChatCompletionMessage(...)`) into readable structured output.
- Tool-call follow-up on thinking-mode models: the final assistant response may put text in `reasoning_content` instead of (or in addition to) `content`. When writing the example, print both so the output isn't misleadingly `None`.
- Invocation section output format: immediately after each code block, add `**Output Example:**` followed by a ```text fenced block with the real run output. Keep the text verbatim from the server — don't paraphrase.

### Step 2: Update navigation and homepage

**`docs_new/docs.json`** — find the Cookbook → Autoregressive Models → `<Vendor>` group under the `navigation` block and append the new page path:

```json
{
  "group": "<Vendor>",
  "pages": [
    "cookbook/autoregressive/<Vendor>/<NewModel>",
    "cookbook/autoregressive/<Vendor>/<ExistingModel>"
  ]
}
```

Use root-relative paths **without `.mdx` extension**. Order pages newest-first inside the group (matching how vendors typically list their lineup).

**`docs_new/cookbook/autoregressive/intro.mdx`** — vendor `<CardGroup>`. The card per vendor points to the **flagship/canonical** model. If the new model becomes the new flagship for an existing vendor, update that vendor's `<Card href="...">` to the new page. If you're adding a brand-new vendor, append a new `<Card>`:

```mdx
<Card
  title="<Vendor>"
  mode="card"
  href="/cookbook/autoregressive/<Vendor>/<FlagshipModel>"
  img="/cards/logos/<vendor>.png"
/>
```

For new vendors, drop the logo asset at `docs_new/cards/logos/<vendor>.png` (256×256 PNG, transparent background where possible).

**Don't touch** `docs_new/cookbook/intro.mdx` — that's the cookbook root and only lists the two top-level categories (Autoregressive / Diffusion).

### Step 3: Create the deployment snippet

Create `docs_new/src/snippets/autoregressive/<modelname>-deployment.jsx` (kebab-case filename, single file — no folder, no `index.js`).

Skeleton (copy from `devstral-2-deployment.jsx` for a simple multi-variant case, or `qwen3-next-deployment.jsx` for hardware × quantization × thinking × tool-call combinations):

```jsx
export const <ModelName>Deployment = () => {
  const options = {
    hardware: { ... },        // tested platforms only
    modelsize: { ... },       // multi-variant only
    quantization: { ... },    // BF16 / FP8 / FP4
    // any other toggles: tool-call, reasoning parser, DP attention, etc.
  };

  const modelConfigs = {
    // per-variant: modelId, tpByHardware, allowedWeights
  };

  // generateCommand() — emits the full `sglang serve` command for the current selection
  // ...
};
```

- Snippets are **self-contained** (no shared base class). Copy the rendering boilerplate (styles, radio/checkbox handling, dark-mode detection) from a sibling snippet — don't try to abstract it.
- `export const <ModelName>Deployment` — named export, PascalCase ending in `Deployment`. The MDX page imports it by name.
- Only list tested platforms in the hardware option.
- `modelConfigs` must have an entry for **every** variant the user can pick — missing combos crash at render time.
- Each radio group has exactly one `default: true`; option `id`s are unique within a group.
- Platform detection in `generateCommand`:
  ```js
  const isAMD = ['mi300x','mi325x','mi350x','mi355x'].includes(hardware);
  const isBlackwell = ['b200','b300'].includes(hardware);
  if (isAMD) { /* AMD-specific flags */ }
  if (isBlackwell) { /* Blackwell-specific flags */ }
  ```
- `commandRule: (value) => ...` on an option is fine for simple "if X then add `--flag`" cases (see Qwen3-Next's `thinking` option). For anything that depends on multiple values, do it in `generateCommand` instead.
- Default any reasoning/tool-call parser toggle to **Enabled**.

**Reasoning parser**: For hybrid models, use Enabled/Disabled toggle (the model always thinks; parser just separates output). For separate Instruct/Thinking variants, toggle changes the model name suffix.

Reasoning parsers fall into **two client-side patterns** — the sample code in Section 4 needs to match:
- **Separate field** (e.g., `--reasoning-parser kimi_k2`, most qwen/glm parsers): thinking text lands in `message.reasoning_content`, answer in `message.content`. Print both.
- **Inline tags** (e.g., `--reasoning-parser minimax-append-think`): thinking is wrapped in `<think>...</think>` inside `message.content`. The client has to parse the tags itself. For streaming demos, walk a buffer looking for `<think>` / `</think>` markers and split as you print.

Pick the pattern from the model card / SGLang docs for that specific parser before writing the example.

**DP Attention**: `Disabled (Low Latency)` / `Enabled (High Throughput)`. The `--dp` value commonly matches `--tp` but this isn't mandatory. Handle in `generateCommand`:
```js
if (values.dpattention === 'enabled') {
  cmd += ` \\\n  --dp ${tpValue} \\\n  --enable-dp-attention`;
}
```
In config tips, describe `--dp` matching `--tp` as a common pattern, not a requirement.

**Large models (>400B)**: BF16 needs ~2x GPUs vs FP8. Reflect this in `modelConfigs`. Omit combos that don't fit.

**Platform-required flags**: If a platform requires certain flags to function at all (e.g., AMD MI355X needs `--attention-backend triton`), add them unconditionally for that platform — NOT gated behind optional checkboxes like "Performance Optimizations". Optional optimizations go inside checkbox guards; required-to-work flags go outside.

**Doc ↔ snippet parity**: The launch command shown anywhere in the MDX page (e.g., the `sglang serve` block in the AMD benchmark section) must be byte-for-byte identical to what the snippet emits when that hardware is selected. If you add `--kv-cache-dtype fp8_e4m3` or `--mem-fraction-static 0.8` for AMD in the snippet, the documented AMD command needs it too — and vice versa. Drift here is the single most common review finding.

**No dead code**: Don't define `commandRule` on options if `generateCommand` handles them directly (the rules will never be called). Don't use `getDynamicItems` if the items don't depend on other option values — use static `items` instead. Don't leave unused helper functions.

**No silent ignores**: If a feature (e.g., DP attention) is unsupported on a platform, either disable the UI option or show an explicit message (like a "Work In Progress" note). Never silently drop user selections.

**Scope discipline**: If adding support for one platform, don't accidentally add global flags. Always check conditionals: `if (quantization === 'fp8')` without a hardware guard affects ALL platforms. Be explicit: `if (hardware === 'h200' && quantization === 'fp8')`.

**License accuracy**: Always verify the actual HuggingFace model license before writing the license section. Don't copy from other model docs — licenses vary (Apache 2.0, MIT, community licenses, etc.).

## Phase 3: Validate and Preview

Validate the page (catches frontmatter issues, missing nav entries, broken internal links, MDX/JSX errors):

```bash
cd docs_new
mint validate
mint broken-links
```

Preview locally:

```bash
cd docs_new
mint dev
```

Check the page renders at `http://localhost:3000/cookbook/autoregressive/<Vendor>/<ModelName>`.

Install `mint` first if needed: `npm i -g mint`.

## Phase 4: Interactive Testing

User deploys the model, runs test scripts, pastes results. Replace `Pending update...` / `TODO` placeholders with actual outputs:
1. Invocation results (code gen, streaming, tool calls)
2. Accuracy benchmarks (GSM8K, MMLU)
3. Speed benchmarks (latency, throughput)

## Phase 5: Configuration Tips

Ask for:
- Recommended settings, known issues, optimization tips
- DP attention trade-offs
- Hardware-specific `mem-fraction-static` values

Add to the doc's Section 3.2.

## Phase 6: Final Review

Can be triggered with `/cookbook-add-model review`. Also consider running `/cookbook-review-pr` on the PR for an automated checklist pass.

Review the complete documentation for:
- File extension is `.mdx` (not `.md`) and ends with a trailing newline.
- Frontmatter has at minimum `title:` and `metatags.description:`.
- Nested code block formatting (use ```````` for outer blocks containing ` ``` `).
- Consistent port numbers across all commands, curl examples, and client code (use **30000**, not 8000).
- Launch port matches client/curl `base_url` port on the same page.
- No duplicate deployment commands (reference the one at the top of Section 4).
- All `Pending update...` / `TODO` placeholders replaced with actual results — OR explicitly left pending with the user's acknowledgement.
- **Benchmark metadata quantization matches a variant listed in Section 1** — e.g., if only INT4 is released, a benchmark "Test Environment" saying `Model: X (BF16)` is a factual bug.
- **Doc ↔ snippet parity**: for each hardware, the launch command shown in the doc (benchmark section, tips, etc.) must equal the snippet's output for that hardware — same flags, same order. Drift here is the #1 review finding.
- Snippet defaults match the documented deployment command at the top of Section 4.
- Snippet's `export const <Name>Deployment` matches the import in the MDX page (common copy-paste bug).
- Benchmark sections contain **two** commands, each with its own rule:
  - **Deploy** (the server launch): always `sglang serve ...` — never `python -m sglang.launch_server` or `python3 -m sglang.launch_server` (deprecated).
  - **Bench** (the workload): always `python3 -m sglang.bench_serving ...` — never bare `python -m sglang.bench_serving`.
  - Ports must match between the two commands on the same page.
- Reasoning mode examples show both thinking-on and thinking-off patterns (for hybrid reasoning models).
- Tool-call follow-up on thinking-mode models prints both `reasoning_content` and `content` (the latter can be `None` when the response is reasoning-only).
- Each invocation code block is followed by an `**Output Example:**` + ```text block with real server output.
- `modelConfigs` covers every (variant × quantization × hardware) combination the UI allows.
- DP attention `--dp` value dynamically matches `--tp` in the snippet.
- Navigation entry added in `docs_new/docs.json` under the right vendor group.
- Vendor card in `docs_new/cookbook/autoregressive/intro.mdx` points to the right flagship; new vendors have a card and a logo asset under `docs_new/cards/logos/`.
- `tag: NEW` is used sparingly: when adding a new NEW page, remove the tag from the older entries so the homepage isn't flooded.
- Section 1 is lean: no duplicated "Architecture" table when the HF card already has it, Benchmarks rendered as a table (not bullets), single-entry "Available Models" lists inlined.
- Raw API response objects (e.g., `ChatCompletionMessage(...)`) are formatted into readable structured output (Reasoning/Content/Tool Calls sections).
- Reasoning parser sample code matches the parser's actual output shape: `reasoning_content` field for separate-field parsers (`kimi_k2` etc.), `<think>...</think>` tag parsing in `content` for inline-tag parsers (`minimax-append-think` etc.).
- Section 2 includes a Docker Images by Hardware Platform table covering every platform listed in the deployment snippet.
- License section matches the actual HuggingFace model license (verify — don't copy from other models).
- No dead code in the snippet (unused `commandRule`, unused helper functions, `getDynamicItems` returning static arrays).
- Platform-required flags are unconditional (not behind optional checkboxes).
- Unsupported features show explicit messages, not silent no-ops.
- No images hosted on Google Drive (sharing links don't render in markdown).
- Shell environment blocks use proper placeholders (`export VAR=<your-value>`), not `export VAR=${VAR}` (which is a bash no-op).
- Grammar and spelling checked in all added documentation text.

## Git Workflow

Always create a new branch — never commit to main directly.

```bash
git checkout -b add-<model-name>
# ... make changes ...
git add docs_new/cookbook/autoregressive/<Vendor>/<ModelName>.mdx \
        docs_new/src/snippets/autoregressive/<modelname>-deployment.jsx \
        docs_new/docs.json \
        docs_new/cookbook/autoregressive/intro.mdx
git commit -m "Add <Model Name> cookbook"
git push -u origin add-<model-name>
gh pr create --title "Add <Model Name> cookbook" --body "..."
```

When updating the homepage card, verify the doc has real content — not just a placeholder stub.
