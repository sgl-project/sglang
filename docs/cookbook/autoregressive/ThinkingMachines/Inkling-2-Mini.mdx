---
title: Inkling-2-Mini
description: "Deploy Inkling-2-Mini with SGLang — verified launch commands, with and without DSpark speculative decoding, for Thinking Machines' looped NVFP4 / MXFP8 Mixture-of-Experts model on GB300, H200, RTX PRO 6000, RTX 5090, DGX Spark, AMD MI350X / MI355X, and TPU7x."
tag: NEW
---

<Callout icon="rocket" color="#d55816">
SGLang supports Inkling-2-Mini from day 0. See the launch blog post: [SGLang and Miles Add Day-0 Support for Inkling 2](https://www.sglang.io/blog/inkling-2-small-day0-support).
</Callout>

| Hardware | Supported | Docker image |
| --- | --- | --- |
| B300 / GB300 | Yes | `lmsysorg/sglang:inkling-2-mini` |
| B200 / GB200 | Yes | `lmsysorg/sglang:inkling-2-mini` |
| H200 | Yes | `lmsysorg/sglang:inkling-2-mini` |
| RTX PRO 6000 | Yes | `lmsysorg/sglang:inkling-2-mini` |
| RTX 5090 | Yes | `lmsysorg/sglang:inkling-2-mini` |
| DGX Spark | Yes | `lmsysorg/sglang:inkling-2-mini` |
| MI350X / MI355X | Yes | `lmsysorg/sglang:inkling-2-mini-mi35x` |
| TPU7x | Yes | `lmsysorg/sglang:inkling-2-mini-tpu` |

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

For all install methods and hardware platforms, see the [official SGLang installation guide](../../../docs/get-started/install).

<Note>Inkling-2-Mini support is not in an SGLang release yet. Install from its pull request, [sgl-project/sglang#51926](https://github.com/sgl-project/sglang/pull/51926), or use the `inkling-2-mini` images below.</Note>

<Tabs>

<Tab title="Python (pip / uv)">

```bash Command
pip install --upgrade pip
pip install 'git+https://github.com/sgl-project/sglang.git@refs/pull/51926/head#subdirectory=python'
```

Then run the **Python** output of the command panel below.

</Tab>

<Tab title="Docker">

```bash Command
docker pull lmsysorg/sglang:inkling-2-mini       # NVIDIA Hopper / Blackwell
docker pull lmsysorg/sglang:inkling-2-mini-mi35x # AMD MI350X / MI355X
docker pull lmsysorg/sglang:inkling-2-mini-tpu   # Google TPU7x
```

For how to launch the image, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). Substitute the inner `sglang serve ...` with what the command generator below produces.

</Tab>

</Tabs>

</Accordion>

Pick your hardware to generate the launch command. Each platform has a **Balanced** recipe (plain decode) and a **DSpark** recipe (speculative decoding on the bundled MTP draft head).

Every recipe passes the `inkling` reasoning parser and the Inkling tool-call parser, which is named `inkling2` in SGLang and `inkling` in sglang-torchtpu.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/thinkingmachines/inkling-2-mini.jsx";
import { benchmarks } from "/src/snippets/configs/thinkingmachines/inkling-2-mini-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

## Playground

The Playground is where you experiment with **SGLang features beyond the verified matrix**. The Deploy panel above only emits combinations that have been signed off; the Playground lets you turn on additional knobs on top of whichever cell the Deploy panel is currently showing. Any change flips the badge to **Not Verified** until the new configuration is run end-to-end.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model Introduction

**Inkling-2-Mini** is a looped Mixture-of-Experts model from Thinking Machines with **26B total and 5.4B active parameters** per token. It takes text and image input. Its 24 layers each have 128 routed experts (top-8, no shared experts), and layers 6–17 run three passes per token. A 3-layer DSpark draft head is bundled for speculative decoding. The weights take about 18 GB, so SGLang serves it on a single GPU, including desktop and workstation GPUs, or across several GPUs or TPU cores with tensor and expert parallelism.

**Resources:** HuggingFace — [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini).

## 2. Configuration Tips

**Architecture.** Inkling-2-Mini has 24 physical layers. Layers 6–17 form a loop core that runs three times per token, so each token makes 48 layer visits, the same number as Inkling-2-Small. The routed experts of layers 0–19 are NVFP4 and those of layers 20–23 are MXFP8. Hidden size is 2304 and the expert intermediate size is 1152.

**Loop options.** On a variable-loop checkpoint, the Playground's **Loop Passes** card sets `--inkling2-loop-num-passes` and the **Loop KV Cache** card sets `--inkling2-loop-share-cache`. Fewer passes trade accuracy for speed: on gsm8k, 1 pass scored 84.0% against 88.5% at the default 3. Sharing one K/V cache across passes about doubles the KV pool and scored 90.0%. DSpark accepts shared KV but refuses a non-default pass count at startup.

**Parallelism.** The data-center GPU recipes run TP4 / EP4: one GB300 tray, or half of an 8-GPU H200 or MI350X / MI355X node. RTX PRO 6000, RTX 5090 and DGX Spark run TP1 / EP1 on a single GPU. TPU7x runs TP8 / EP8 on one host. Keep EP equal to TP: the NVFP4 experts cannot be TP-sharded, because their activation scales must cover each expert's full input width.

**Batch-size-1 decode.** Decode at batch size 1 takes about as long per token as on Inkling-2-Small, because both models make 48 layer visits per token.

**Parsers.** Pass `--reasoning-parser inkling` and `--tool-call-parser inkling2` (`inkling` on sglang-torchtpu) explicitly; the recipes do.

**Speculative decoding.** DSpark proposes from the bundled MTP draft head.

**Multimodal.** The recipes pass `--enable-multimodal` so the server accepts image inputs; drop it for text-only serving.

**PD disaggregation.** The Playground's **PD Disaggregation** card turns the selected recipe into a prefill or decode role and prints the router command; on GB300 it adds the Mooncake MNNVL environment. It was validated as 1P1D on one 4-GPU GB300 tray: prefill at TP2 / EP2 on GPUs 0–1 and decode at TP2 / EP2 on GPUs 2–3, each with `--mem-fraction-static 0.85`. Sequential requests gave the same tokens and log-probabilities as a colocated TP2 server. See [PD disaggregation](../../../docs/advanced_features/pd_disaggregation) for the general setup.

**Hierarchical KV cache.** The host tier (`--enable-hierarchical-cache`) was validated on GB300 at TP2 / EP2: a prefix reloaded from host memory gave the same output log-probabilities as a server without it. Storage backends have not been run.

**Prefill CUDA graphs.** Keep the default prefill CUDA graph backend. `--cuda-graph-backend-prefill breakable` crashes on the first replay.

**RTX PRO 6000, RTX 5090 and DGX Spark.** Each runs on one GPU at TP1 / EP1 with the FA4 attention backend and the default BF16 GEMM backend, which on these GPUs adds small-batch decode GEMVs. The DSpark recipes verify 8 draft tokens per step and turn on four `SGLANG_OPT_*` speculative-decoding optimizations; on RTX PRO 6000 and RTX 5090 they cap `--max-running-requests` at 4. RTX 5090 (32 GB) caps the context at 65536 tokens for Balanced and 32768 for DSpark to fit its KV pool. DGX Spark sets `MAX_JOBS=4` and `--mem-fraction-static 0.45`. Each of these six recipes scored 93.6–94.2% on all 1319 gsm8k questions as printed.

**AMD.** MI350X and MI355X (both gfx950) run the same recipes, on the Triton attention backend with `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`, because the image's default INT8 quick reduce is lossy. Other ROCm GPUs refuse at launch. For PD, use the NiXL transfer backend: it matched a colocated server token for token, while Mooncake over TCP did not.

**TPU7x.** One TPU7x host (4 chips, 8 TensorCores) runs TP8 / EP8 on sglang-torchtpu. The first start compiles the serving shapes (about 10 minutes); later starts reuse the compilation cache.

<a id="tpu" />

## 3. TPU7x deployment

Inkling-2-Mini runs on a 4-chip TPU7x slice (one host, 8 TensorCores) as one TP8 · EP8 engine through [SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu).

<Note>Inkling-2-Mini support is not in an SGLang-TorchTPU release yet. Use the `inkling-2-mini-tpu` image.</Note>

Select **TPU7x** and **DSpark** above for the command. It sets only what Inkling-2-Mini needs; everything else is left at its default. The DSpark draft is bundled in the checkpoint.

The DSpark command was verified with GSM8K and an image-input check; AIME26 and GPQA Diamond were also run with the tuned flags of the Inkling-2-Small recipe `recipe/configs/inkling2/server.py`. The **Balanced** recipe, the same command without speculative decoding, has not been run end to end. Not yet run on TPU7x: contexts above 131072 tokens, the **Loop Passes** and **Loop KV Cache** options, PD disaggregation, and HiCache.
