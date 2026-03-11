---
name: add-model
description: Add a new model to the SGLang documentation cookbook, including MDX docs, JSX deployment snippet, and navigation entry.
disable-model-invocation: true
---

# Add New Model to SGLang Docs Cookbook

This is an interactive, multi-step workflow. Collect inputs incrementally from the user as needed — do NOT ask for everything upfront.

## Phase 1: Collect Initial Inputs

Ask the user for:

1. **Category** — Is this an autoregressive model, a diffusion model, or an omni model? This determines:
   - Doc path: `docs/cookbook/<category>/<Vendor>/<Model>.mdx`
   - Snippet path: `docs/src/snippets/<category>/<name>-deployment.jsx`
   - Navigation group in `docs/docs.json`
   - Template style (autoregressive models get chat/completion examples; diffusion models get image generation examples; omni models get multimodal examples)

2. **Model Card** — HuggingFace model name or URL (e.g., `Qwen/Qwen3-Coder-Next` or `https://huggingface.co/deepseek-ai/DeepSeek-V3`). Fetch the page to extract model description, supported capabilities, and other details. If the model is not yet public or the page is inaccessible, ask the user to paste as much information as they know (model name, parameter count, architecture, capabilities, context length, etc.).

3. **Model Variants** — Check if the model family has multiple size variants (e.g., 480B/30B) and quantization options (e.g., BF16/FP8). Ask the user which variants to include. This affects:
   - JSX snippet: selector options for model size and quantization
   - Documentation: model name references and deployment examples

4. **Full Deployment Command** — Complete SGLang launch command using `sglang serve` (NOT the deprecated `python -m sglang.launch_server`), including all strategy and optimization flags (tp, dp, ep, enable_dp_attention, etc.). If the model card already provides an SGLang deployment command, offer it as a default option — but verify it uses the current `sglang serve` format.

5. **Hardware Platforms** — Ask which hardware platforms have been tested. Only include tested platforms in the deployment snippet. Do NOT assume AMD GPU support unless explicitly confirmed.

## Phase 2: Create Scaffolding

With model card and deployment command in hand, read ALL reference templates first, then create files.

### Reference Templates

Read these files to understand existing patterns before creating anything:
- **Doc template**: Find a similar model doc under `docs/cookbook/<category>/` (e.g., an existing `.mdx` file for the same vendor or model type)
- **Snippet template**: Find a similar JSX snippet under `docs/src/snippets/<category>/` (e.g., an existing `*-deployment.jsx` file)
- **Navigation**: `docs/docs.json`

### Important Patterns

- JSX deployment snippets go in `docs/src/snippets/<category>/<name>-deployment.jsx` as **functional components** with **named exports** (e.g., `export const QwenDeployment = () => { ... }`)
- JSX snippets use **inline styles** and a **dark mode detection** pattern with `useState` + `useEffect` (check `window.matchMedia('(prefers-color-scheme: dark)')`)
- AMD GPUs (MI300X/MI325X/MI355X) typically need `--attention-backend triton` — only include if tested
- AMD Docker image naming: `rocm720-mi30x` for MI300X/MI325X, `rocm720-mi35x` for MI355X
- **CRITICAL**: All commands must use `sglang serve`, never `python -m sglang.launch_server` (deprecated)
- Before creating a new model, check open PRs (`gh pr list --search "<model name>"`) to avoid duplicate work

### Step 1: Create documentation file

Create `docs/cookbook/<category>/<Vendor>/<ModelName>.mdx` with ALL sections pre-populated.

**For autoregressive models:**
- Section 1: Model introduction (from model card)
- Section 2: SGLang installation instructions
- Section 3: Model deployment section (embed the JSX deployment snippet)
- Section 4: Model invocation — include deployment command at the TOP of this section, then pre-fill test scripts (code generation, streaming, tool calling if supported) with `TODO` placeholders for outputs
- Section 5: Benchmarks — pre-fill with benchmark commands and `TODO` placeholders for results

**For diffusion models:**
- Section 1: Model introduction (from model card)
- Section 2: SGLang installation instructions
- Section 3: Model deployment section (embed the JSX deployment snippet)
- Section 4: Image generation examples — text-to-image, image-to-image if supported, with `TODO` placeholders
- Section 5: Performance notes

**For omni models:**
- Section 1: Model introduction (from model card)
- Section 2: SGLang installation instructions
- Section 3: Model deployment section (embed the JSX deployment snippet)
- Section 4: Multimodal examples — text, audio, vision as applicable, with `TODO` placeholders
- Section 5: Performance notes

**Benchmark commands reference (autoregressive):**
- GSM8K: `python3 benchmark/gsm8k/bench_sglang.py --port <port>`
- MMLU: `python3 benchmark/mmlu/bench_sglang.py --port <port>`
- Latency benchmark: `python3 -m sglang.bench_serving --backend sglang --num-prompts 10 --max-concurrency 1 ...`
- Throughput benchmark: `python3 -m sglang.bench_serving --backend sglang --num-prompts 1000 --max-concurrency 100 ...`

Keep benchmarks concise — only include latency and throughput for speed benchmarks. Do NOT add multiple scenarios or multiple concurrency levels unless the user requests it.

**Important**:
- When the output contains nested markdown code blocks (e.g., model outputs python code), use four backticks ```````` for the outer block to avoid rendering issues.
- Do NOT hardcode sampling parameters (`temperature`, `top_p`, `top_k`) in code examples. SGLang automatically applies the recommended parameters from the model's `generation_config.json`.

**Reasoning / Thinking mode in code examples:**
- For **hybrid reasoning models** (thinking is always on by default), show TWO examples:
  1. **Thinking mode (default)**: No extra parameters needed, show `reasoning_content` streaming
  2. **Instruct mode (thinking off)**: Show `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
- For **models with separate Instruct/Thinking variants** (e.g., Qwen3-Next): The model name changes, handled by the deployment snippet
- Ask the user which pattern applies if unclear from the model card

### Step 2: Update navigation

Edit `docs/docs.json` to add the new entry under the appropriate category group.

### Step 3: Create JSX deployment snippet

Create `docs/src/snippets/<category>/<name>-deployment.jsx` based on the deployment command provided.

The JSX snippet must be a **functional component** with a **named export**:

```jsx
import React, { useState, useEffect } from 'react';

export const ModelNameDeployment = () => {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    setIsDark(mq.matches);
    const handler = (e) => setIsDark(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  // ... component logic with inline styles
};
```

Key considerations:
- Use **inline styles** (no CSS imports)
- Detect dark mode with the `useState` + `useEffect` pattern shown above
- Define hardware/quantization options as data within the component
- Generate the `sglang serve` command dynamically based on user selections
- For AMD hardware, include `--attention-backend triton` flag
- Default all parsers to **Enabled** (e.g., tool call parser, reasoning parser)

**DP Attention option:**
- If the model supports DP attention, add it as a toggle
- The `--dp` value should **dynamically match** `--tp` value in the generated command

**For models with multiple variants**, add selector UI:
- Model size selector (e.g., 480B / 30B)
- Quantization selector (e.g., BF16 / FP8)

## Phase 3: Build Check

Run the Mintlify validation to verify everything compiles:

```bash
cd docs && npx mintlify validate
```

Start the dev server to verify rendering:

```bash
cd docs && npx mintlify dev
```

Verify the new page renders correctly.

## Phase 4: Interactive Testing

The user will deploy the model and run the test scripts from the documentation page. They will paste results back, and we update the `TODO` placeholders with actual outputs.

This covers:
1. **Model Invocation** — code generation, streaming, tool calling results
2. **Speed Benchmarks** — latency and throughput results (autoregressive)
3. **Accuracy Benchmarks** — GSM8K, MMLU results (autoregressive)

## Phase 5: Configuration Tips

Ask the user for any extra information:
- Recommended settings
- Known issues or limitations
- Optimization tips
- DP attention trade-offs (high throughput vs low latency)
- Hardware-specific notes

Add these to the documentation.

## Phase 6: Final Review

Can be triggered with `/add-model review`. Also consider running `/review-pr` on the PR for an automated checklist pass.

Review the complete documentation for:
- Nested code block formatting (use ```````` for outer blocks containing ` ``` `)
- Consistent port numbers across all commands (use 30000, not 8000)
- No duplicate deployment commands (reference the one at the top of Section 4)
- All `TODO` placeholders replaced with actual results
- JSX snippet uses named export (`export const XxxDeployment`), NOT `export default`
- All commands use `sglang serve` — no deprecated `python -m sglang.launch_server`
- Reasoning mode examples show both thinking-on and thinking-off patterns (for hybrid reasoning models)
- Dark mode detection pattern is present in JSX snippet
- Navigation entry in `docs/docs.json` is correct
