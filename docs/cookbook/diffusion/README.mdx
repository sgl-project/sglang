---
title: "Diffusion Cookbook"
description: "Author and review deployment-focused diffusion model recipes for SGLang."
metatags:
    description: "SGLang diffusion cookbook structure, model-page standards, and contribution guidance."
---

# SGLang Diffusion Cookbook

The diffusion cookbook is a set of model-specific, validated deployment guides. Each page
should let a reader answer three questions quickly: what the model is good at, which
checkpoint and base topology to choose, and which optional features change performance,
memory, or quality.

## What every model page includes

1. A compact capability tag row using the shared `DiffusionModelTags` component.
2. A short introduction that states the model's capability range, strongest use case, and
   an important limitation or deployment tradeoff.
3. Verified base launch recipes for the hardware actually tested.
4. Matching request examples for every public task or checkpoint partition.
5. Optional attention, quantization, caching, compile, and offload recipes with explicit
   quality contracts and validation scope.

Tags describe the model, not every runtime flag. Prefer `video + audio`, `multi-reference`,
or `4–15 seconds` over generic labels such as `native`, `fast`, or `high quality`.

## Keep the command picker small

The command picker selects a runnable base recipe: hardware topology, checkpoint partition,
request mode, and a validated placement profile. Independent feature knobs do not need to
become picker dimensions.

Document orthogonal controls as feature overlays instead:

| Feature class | Where it belongs | Required context |
| --- | --- | --- |
| Attention backend | Optional feature overlays | Default backend, precision/approximation contract, installation, measured hardware |
| Online or pre-quantized weights | Optional feature overlays | Weight source, protected precision-sensitive layers, quality and memory tradeoff |
| Cache-DiT and sampling controls | Request/sampling recipes | Quality level, incompatible placement modes, task-specific validation |
| TP, Ulysses, Ring, FSDP, residency | Base picker only for verified cells | Exact GPU count, topology, memory, and latency evidence |

For memory placement, prefer the unified
[`--component-residency`](/docs/sglang-diffusion/api/cli#component-residency)
selector. Each component resolves to exactly one of `resident`,
`component-offload`, or `layerwise-offload`. Existing options such as
`--dit-cpu-offload`, `--text-encoder-cpu-offload`,
`--image-encoder-cpu-offload`, and `--vae-cpu-offload` remain supported by all
recipes that already use them.

## Contributing a model

Use the repo-local `cookbook-add-model` skill and its
`templates/diffusion-page.mdx.tmpl` template. The full category contract lives in
`.claude/skills/cookbook-add-model/references/diffusion-authoring.md`.

Before opening a PR, run:

```bash Command
node docs/scripts/check_cookbook_configs.mjs
cd docs
mint validate
mint broken-links --check-anchors --check-redirects
```

The static cookbook check requires every diffusion model page to use the shared tag widget,
the standard introduction heading, 4–6 tags, and a substantive two-paragraph lead. Mintlify
validation then catches MDX, navigation, and internal-link problems.

For runtime concepts and platform support, see the
[SGLang Diffusion documentation](/docs/sglang-diffusion/index) and
[supported model catalog](/docs/sglang-diffusion/compatibility_matrix).
