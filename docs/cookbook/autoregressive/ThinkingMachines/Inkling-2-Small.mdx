---
title: Inkling-2-Small
description: "Deploy Inkling-2-Small with SGLang — verified launch commands, with and without DSpark speculative decoding, for Thinking Machines' NVFP4 / MXFP8 Mixture-of-Experts model on GB300, GB200, B300, B200, Vera Rubin, H200, AMD MI350X / MI355X, and TPU7x."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

For all install methods and hardware platforms, see the [official SGLang installation guide](../../../docs/get-started/install).

<Note>Inkling-2-Small support is not in an SGLang release yet. Use the `dev-inkling-2-small` images below.</Note>

<Tabs>

<Tab title="Python (pip / uv)">

```bash Command
pip install --upgrade pip
pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=python'
```

Then run the **Python** output of the command panel below.

</Tab>

<Tab title="Docker">

```bash Command
docker pull lmsysorg/sglang:dev-inkling-2-small         # NVIDIA Hopper / Blackwell
docker pull lmsysorg/sglang:dev-inkling-2-small-cu134   # NVIDIA Vera Rubin
docker pull lmsysorg/sglang:dev-inkling-2-small-mi35x   # AMD MI350X / MI355X
docker pull lmsysorg/sglang:dev-inkling-2-small-tpu     # Google TPU7x
```

For how to launch the image, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). Substitute the inner `sglang serve ...` with what the command generator below produces.

</Tab>

</Tabs>

</Accordion>

Pick your hardware to generate the launch command. Each platform ships a **Balanced** recipe (plain decode) and a **DSpark** recipe (speculative decoding on the bundled MTP draft head). Both serve the full 1M-token context.

Every recipe enables the `inkling` reasoning parser and the Inkling-2-Small tool-call parser, which is named `inkling2` in SGLang and `inkling` in sglang-torchtpu.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config }     from "/src/snippets/configs/thinkingmachines/inkling-2-small.jsx";
import { benchmarks } from "/src/snippets/configs/thinkingmachines/inkling-2-small-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />

## Playground

The Playground is where you experiment with **SGLang features beyond the verified matrix**. The Deploy panel above only emits combinations that have been signed off; the Playground lets you turn on additional knobs on top of whichever cell the Deploy panel is currently showing. Any change flips the badge to **Not Verified** until the new configuration is run end-to-end.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

{/* §1 Model Introduction: to be added with the model card. */}

## 2. Configuration Tips

**Checkpoints.** The **Quantization** row selects the checkpoint. `thinkingmachines/Inkling-2-Small` stores the routed experts of layers 0–39 in NVFP4 and of layers 40–47 in MXFP8 (about 394 GB). `thinkingmachines/Inkling-2-Small-MXFP8` stores every routed expert in MXFP8 (about 623 GB) and has recipes for Blackwell and Vera Rubin. At TP4 its weights take about 151 GB per GPU, against 94 GB for the NVFP4 checkpoint, so the 288 GB GPUs keep a smaller KV pool, and B200 and GB200 (192 GB) run it at TP8 / EP8: the full B200 node, or two GB200 trays.

**Parallelism.** Blackwell recipes run TP4 / EP4 (one GB200 / GB300 tray, or half of an 8-GPU B200 / B300 node), except the MXFP8 checkpoint on B200 and GB200, which runs TP8 / EP8 (the full B200 node, or two GB200 trays). H200 and AMD run TP8 / EP8. Keep EP equal to TP: the NVFP4 experts cannot be TP-sharded, because their activation scales must cover each expert's full input width.

**MegaMoE (Blackwell).** `--moe-a2a-backend megamoe` runs the routed experts of every forward with at least 64 tokens through one fused DeepGEMM kernel (dispatch, both expert GEMMs, the capped SwiGLU, and the combine). The Blackwell and Vera Rubin recipes use it for Balanced and DSpark. Balanced decode below 64 concurrent requests stays on the routed kernels, so for Balanced the gain is mostly prefill: on GB300 it cut median time to first token by 19–29% with no decode slowdown. It needs a DeepGEMM build with Inkling-2-Small support; without one the server fails at load. AMD refuses MegaMoE at launch. Keep `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096`: it covers a 16384-token prefill chunk at EP4 and costs about 1.2% of the KV pool, against 6.2% at the global default of 8192. If the build is not available, pick **Routed experts** in the Playground's MoE card.

**Context length and batch size.** Both strategies serve the full 1,048,576-token window; the MTP draft head covers it as well. DSpark recipes raise `--max-running-requests` from the speculative-decoding default of 48. At TP4, B200 and GB200 (192 GB) hold about a third of GB300's KV pool, which limits concurrency at long context; for more KV capacity on B200, use TP8 on the full node.

**Parsers.** On Inkling-2-Small the server picks `--reasoning-parser inkling` and `--tool-call-parser inkling2` by itself when they are unset; the recipes pass them explicitly so the command shows them.

**Speculative decoding.** DSpark proposes from the bundled MTP draft head and verifies 8 tokens per request per step.

**Multimodal.** The recipes pass `--enable-multimodal` so the server accepts image inputs; drop it for text-only serving.

**PD disaggregation.** The Playground's **PD Disaggregation** card turns the selected recipe into a prefill or decode role and prints the router command; on GB200 / GB300 it adds the Mooncake MNNVL environment. It was validated with the NVFP4 checkpoint as 1P1D on one 4-GPU GB300 tray, without MegaMoE: prefill at TP2 / EP2 on GPUs 0–1 and decode at TP2 / EP2 on GPUs 2–3. Set `--mem-fraction-static 0.86` on the decode server there; the automatic value (0.963) ran out of memory at startup. This split does not fit the MXFP8 checkpoint, whose weights take about 311 GB per GPU at TP2; keep each MXFP8 role at the TP of its recipe. See [PD disaggregation](../../../docs/advanced_features/pd_disaggregation) for the general setup.

**AMD.** The ROCm image exports `ROCM_QUICK_REDUCE_QUANTIZATION=INT8`, a lossy all-reduce; the recipe sets it to `NONE`. Attention runs on the Triton backend. Inkling-2-Small on ROCm is validated on gfx950 (MI350X / MI355X) only; other ROCm GPUs refuse at launch.

**Vera Rubin.** The recipes run on one 4-GPU VR200 tray with the GB300 flags (TP4 / EP4, FA4, MegaMoE). They need the `dev-inkling-2-small-cu134` image, which carries the DeepGEMM MegaMoE build for SM107. Both set `SGLANG_OPT_INKLING2_MEGA_MOE_MIN_TOKENS=192`.

**TPU7x.** One TPU7x host (4 chips, 8 TensorCores) runs TP8 / EP8 on sglang-torchtpu. The first start compiles the shapes listed in `--tpu-compile-bs` and `--tpu-compile-tokens`; later starts reuse the compilation cache.

<a id="tpu" />

## 3. TPU7x deployment

Inkling-2-Small runs on a 4-chip TPU7x slice (one host, 8 TensorCores) as one TP8 · EP8 engine through [SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu). Run its recipe from the repository root:

```bash Command
python3 -m benchmark.tpu.launch_server --config recipe/configs/inkling2/server.py
```

Prefix caching, KV offloading and tool calls are supported for Inkling-2-Small on TPU7x. TPU7x supports KV offloading with `--enable-hierarchical-cache --hicache-io-backend raiden --hicache-mem-layout page_first_fused --hicache-ratio 3`; storage tiers are not supported.

## 4. Advanced Usage

<a id="epd-disaggregation" />

### 4.1 Encoder disaggregation (EPD)

Encoder disaggregation runs the vision tower in its own server. The encoder server (`--encoder-only`) loads only the vision tower; the language server (`--language-only`) skips it and receives the image embeddings from the encoder. The validated topology uses the NVFP4 checkpoint on one 4-GPU GB300 tray: an encoder at TP1 on GPU 1 and a language server at TP2 / EP2 on GPUs 2–3, without MegaMoE. The MXFP8 checkpoint does not fit a TP2 language server. Start the encoder first.

<Note>Encoder disaggregation for Inkling-2-Small is not in the `dev-inkling-2-small` images yet.</Note>

<Accordion title="Encoder server (GB300)">

```bash Command
CUDA_VISIBLE_DEVICES=1 \
sglang serve \
  --model-path thinkingmachines/Inkling-2-Small \
  --tp 1 \
  --ep 1 \
  --attention-backend fa4 \
  --bf16-gemm-backend torch \
  --dtype bfloat16 \
  --page-size 128 \
  --enable-multimodal \
  --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler \
  --host 0.0.0.0 \
  --port 30400
```

</Accordion>

<Accordion title="Language server (GB300)">

```bash Command
CUDA_VISIBLE_DEVICES=2,3 \
sglang serve \
  --model-path thinkingmachines/Inkling-2-Small \
  --tp 2 \
  --ep 2 \
  --attention-backend fa4 \
  --bf16-gemm-backend torch \
  --dtype bfloat16 \
  --page-size 128 \
  --grammar-backend xgrammar \
  --enable-multimodal \
  --reasoning-parser inkling \
  --tool-call-parser inkling2 \
  --mem-fraction-static 0.86 \
  --language-only \
  --encoder-urls http://localhost:30400 \
  --encoder-transfer-backend zmq_to_scheduler \
  --host 0.0.0.0 \
  --port 30000
```

</Accordion>

The vision tower's output depends slightly on which images it encodes together. When the same images are encoded together, greedy outputs and log-probabilities matched a colocated TP2 server exactly, on single- and multi-image requests. The same held with a prefill/decode split behind the PD router: the language server above with `--disaggregation-mode prefill` on GPUs 0–1 (sharing GPU 1 with the encoder, `--mem-fraction-static 0.80`), and a decode server on GPUs 2–3. The encoder process used under 12 GB.

- **Alpha channel.** Images with transparency can give different outputs than colocated serving, because the encoder loads images through a different path.
- **Keep the encoder reachable.** When a language server has no healthy encoder, it falls back to local encoding, which a language-only server cannot run, and the server stops.

See the [encoder disaggregation guide](../../../docs/advanced_features/epd_disaggregation) for the general architecture.
