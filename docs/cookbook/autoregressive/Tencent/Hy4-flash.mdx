---
title: Hy4-flash
description: "Deploy Tencent Hy4-flash with SGLang — launch recipes for the ~220B Mixture-of-Experts model with sparse + sliding-window attention and Engram memory, in BF16 and FP8 on H200, B300 and GB300."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

For all methods and hardware platforms, see the [official SGLang installation guide](/docs/get-started/install). The two paths below match the **Python / Docker** toggle in the command panel.

<Tabs>

<Tab title="Docker">

```bash Command
docker pull lmsysorg/sglang:hy4-flash
```

For how to launch the image, see [Install → Method 3: Using Docker](/docs/get-started/install#method-3-using-docker), substituting the inner `sglang serve ...` with what the command generator below produces.

</Tab>

<Tab title="Python (pip / uv)">

Hy4-flash support is not in a released SGLang wheel yet. Install SGLang from source on the model-support branch ([sgl-project/sglang#51159](https://github.com/sgl-project/sglang/pull/51159)):

```bash Command
git clone -b hy4_flash_squash https://github.com/sgl-project/sglang.git
cd sglang
pip install --upgrade pip
pip install -e "python"
```

Then run the **Python** output of the command panel below in that environment.

</Tab>

</Tabs>

</Accordion>

Pick your hardware + recipe to generate the launch command.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/tencent/hy4-flash.jsx";
import { benchmarks } from "/src/snippets/configs/tencent/hy4-flash-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

## Playground

The Playground is where you experiment with **SGLang features beyond the verified matrix**. The Deploy panel above only emits combinations the SGLang team has signed off on; the Playground lets you turn on prefill/decode disaggregation and hierarchical KV caching (HiCache) on top of whichever cell the Deploy panel is currently showing.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model Introduction

**Hy4-flash** is Tencent's ~220B-parameter (~12B active) Mixture-of-Experts model with 512 routed experts, interleaving DeepSeek-style sparse MLA attention with sliding-window attention and adding Engram n-gram memory tables.

**Available models:**

- [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash) — BF16
- [tencent/Hy4-flash-FP8](https://huggingface.co/tencent/Hy4-flash-FP8) — FP8 (e4m3, 128×128 block scales)

## 2. Advanced Usage

### 2.1 HiCache (Hierarchical KV Caching)

Hy4-flash supports SGLang's hierarchical KV cache, including the host-memory (L2) tier and storage (L3) backends. Toggle it in the **HiCache** card of the [Playground above](#playground) to add the flags to your launch command.

### 2.2 PD Disaggregation

Prefill and decode can run on separate SGLang instances. Pick a role and transfer backend in the **PD Disagg** card of the [Playground above](#playground); it also generates the `sgl-router` command that fronts both roles.
