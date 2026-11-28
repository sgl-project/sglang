---
title: PPLX-Decider-v1.1-27B
description: "Deploy PPLX-Decider-v1.1-27B with SGLang. Perplexity's updated decision model reads the whole prompt in its full-attention layers and returns calibrated probabilities for choice, yes or no, and score questions on /v1/systemone, on one NVIDIA GB300 or H200 in BF16."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

For all methods and hardware platforms, see the [official SGLang installation guide](../../../docs/get-started/install). The two paths below match the **Python / Docker** toggle in the command panel. PPLX-Decider-v1.1 support lands in [#42645](https://github.com/sgl-project/sglang/pull/42645), after v0.5.21, so until a release includes it, install a [nightly build](/docs/get-started/install#nightly-builds) built after that merge.

<Tabs>

<Tab title="Python (pip / uv)">

```bash Command
pip install --upgrade pip
pip install uv
uv pip install --prerelease=allow --index-strategy unsafe-best-match --extra-index-url https://docs.sglang.ai/whl/cu130/ sglang
```

Then run the **Python** output of the command panel below in that environment.

</Tab>

<Tab title="Docker">

```bash Command
docker pull lmsysorg/sglang:dev
```

The image is multi-arch, so the same tag runs on x86 H200 hosts and Arm GB300 hosts. For how to launch it, see [Install → Method 3: Using Docker](../../../docs/get-started/install#method-3-using-docker). Substitute the inner `sglang serve ...` with what the command generator below produces.

</Tab>

</Tabs>

</Accordion>

Pick your GPU to generate the launch command. The model runs on one GPU in BF16 and needs no flag: SGLang reads the checkpoint's `decision_config.json`, runs the full-attention layers over the whole prompt as the model was trained, and turns off the radix cache and chunked prefill, which would show those layers only part of the prompt.

import { Deployment } from "/src/snippets/_deployment.jsx"
import { config }     from "/src/snippets/configs/perplexity-ai/pplx-decider-v1.1-27b.jsx"
import { benchmarks } from "/src/snippets/configs/perplexity-ai/pplx-decider-v1.1-27b-benchmarks.jsx"

<Deployment config={config} benchmarks={benchmarks} />

<Note>
  The GB300 recipe was measured end to end on one GB300 at [#42645](https://github.com/sgl-project/sglang/pull/42645): four request shapes from 1 to 256 concurrent requests, Belebele and WinoGrande accuracy, and per-question parity with the checkpoint's own reference implementation. The H200 command is the same and has not been run yet. [§3](#3-benchmarks) has the method and the full tables.
</Note>

## Playground

The Playground layers SGLang features on top of the recipe above. Any override changes the badge to **Not Verified** until that exact configuration is tested end to end. The [Configuration Tips](#2-configuration-tips) say which overrides were measured on GB300. Tensor parallelism times data-parallel replicas must fit the GPUs on one node.

import { Playground } from "/src/snippets/_playground.jsx"

<Playground config={config} />

## 1. Model Introduction

**PPLX-Decider-v1.1-27B** is the successor to [PPLX-Decider-v1-27B](/cookbook/autoregressive/Perplexity/PPLX-Decider-v1-27B), Perplexity's decision model fine-tuned from [Qwen3.8-27B](/cookbook/autoregressive/Qwen/Qwen3.8-27B) under the Apache-2.0 license. Like v1, it replaces the language model head with a readout over 255 answer codes and answers choice, yes or no (`noul`), and score questions about a text state and optional images with a probability for every option, without generating text.

Two things changed. The 16 full-attention layers no longer use a causal mask, so every token attends to the whole prompt, while the 48 Gated DeltaNet layers stay causal. And training grew from 73,000 to 626,033 rows, most of them from [tasksource](https://github.com/sileod/tasksource). The prompt format, answer codes, and readout layout are the same as v1's, but the calibrated temperature is 1.0087 instead of 2.2076, so probabilities from the two versions are not interchangeable.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "30%"}} />
    <col style={{width: "70%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Property</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Checkpoint</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}><a href="https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b">perplexity-ai/pplx-decider-v1.1-27b</a> (BF16)</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Base model</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Qwen3.8-27B, one epoch of supervised fine-tuning on 626,033 decision rows (530,103 from tasksource)</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Attention</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Noncausal in the 16 full-attention layers, causal in the 48 Gated DeltaNet layers</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Question types</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}><code>choice</code> (up to 255 options), <code>noul</code> (yes or no), <code>score</code> (up to 10 levels)</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Inputs</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Text or JSON state, optional images</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Prompt length</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Trained on question prompts of up to 8,192 tokens, and the backbone accepts 262,144</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>License</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Apache-2.0</td>
    </tr>
  </tbody>
</table>

Perplexity reports a Decision Index of 61.56 for v1.1, up from 56.4 for v1:

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "34%"}} />
    <col style={{width: "18%"}} />
    <col style={{width: "24%"}} />
    <col style={{width: "24%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Decision Index category</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>Jev</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>PPLX-Decider-v1-27B</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>PPLX-Decider-v1.1-27B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Knowledge</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}><strong>51.4</strong></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>40.9</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>48.18</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Language</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>62.0</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>63.5</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}><strong>69.45</strong></td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Retrieval</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>55.4</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>54.9</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}><strong>61.26</strong></td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Tools</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>75.1</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}><strong>79.3</strong></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>78.88</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Arts</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>37.7</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>39.4</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}><strong>44.66</strong></td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}><strong>Overall</strong></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>57.9</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>56.4</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}><strong>61.56</strong></td>
    </tr>
  </tbody>
</table>

**Resources:** [HuggingFace](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b), [reference model code](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/main/source/src/autojev/model.py), [Decision models in SGLang](/docs/supported-models/decision_models).

## 2. Configuration Tips

- **Route.** Send requests to `/v1/systemone`, the System One API the checkpoint was trained for. `/v1/decisions` refuses this checkpoint because its labels and prompt are not the ones the readout learned, and chat or generation requests are not meaningful without a language model head.
- **Clients.** Clients of the System One API, including the TypeSafe SDKs, work by pointing their base URL at the server. See [System One compatible API](/docs/supported-models/decision_models#system-one-compatible-api) for the request and response reference.
- **Images.** Pass each image in `images` as base64 bytes, a base64 data URL, or an `http(s)` URL. They precede the text in every question. The processor resizes every image to between 65,536 and 262,144 pixels, so one image costs 64 to 256 prompt tokens.
- **Noncausal attention is automatic.** SGLang reads `attention_mode: noncausal_full_attention` from `decision_config.json` and runs the full-attention layers as encoder layers, which the TRT-LLM, FlashInfer, and Triton attention backends support. It also turns off the radix cache and chunked prefill, logging `Radix cache and chunked prefill are disabled for a decision checkpoint with noncausal full attention`: a cached prefix or a prefill chunk would see only part of a prompt whose every position depends on the rest. These two settings override any flag you pass. A prompt longer than the 16,384-token prefill budget is still read in one pass: an 18,585-token prompt matched the reference implementation to four decimal places on GB300.
- **No prefix reuse.** Because every layer output depends on the whole prompt, nothing carries over between requests, not even an identical one. The cost of a request is roughly *questions × prompt tokens*, as on v1.
- **Prompt length.** Like v1, the model was trained on question prompts of up to 8,192 tokens, and the reference code refuses longer ones. SGLang serves longer prompts in one pass, but their accuracy is untested.
- **Attention backend.** On GB300, Auto resolves to the TRT-LLM kernel with 64-token pages. `--attention-backend flashinfer` and `--attention-backend triton` matched the reference implementation as closely, with the same top option on 99.8% of the accuracy items. The H200 path has not been run yet.
- **Speed.** One GB300 reads about 27,000 prompt tokens per second once batches are full, the same as v1. Measured back to back on the same GPU against v1's recipe, v1.1 was within 2% on short questions and four-question requests, and an 8K-token state saturated 6% lower (3.13 against 3.32 requests per second), because noncausal attention computes the full attention matrix in its 16 full-attention layers instead of half of it.
- **More GPUs.** The backbone and prompt sizes are v1's, so the [v1 page's scaling measurements](/cookbook/autoregressive/Perplexity/PPLX-Decider-v1-27B#2-configuration-tips) are the reference: data-parallel replicas (`--dp-size N`) for throughput, tensor parallelism only for latency on long prompts. They were not repeated for v1.1.
- **Upgrading from v1.** Re-check any thresholds you tuned on v1 probabilities: v1.1 has a different temperature and changed its top answer on 8% of the Belebele and WinoGrande items.

## 3. Benchmarks

All numbers in this section come from one NVIDIA GB300 (288 GB) at [#42645](https://github.com/sgl-project/sglang/pull/42645), with checkpoint revision `3b45dea`, served with the GB300 recipe above unless a row says otherwise. The method and request shapes are those of the [v1 page](/cookbook/autoregressive/Perplexity/PPLX-Decider-v1-27B#3-benchmarks), whose speed script works here unchanged.

### 3.1 Speed

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "36%"}} />
    <col style={{width: "13%"}} />
    <col style={{width: "17%"}} />
    <col style={{width: "17%"}} />
    <col style={{width: "17%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Request</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>Concurrent</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>Median latency</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>Requests/s</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>Prompt tokens/s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Short question, 382 tokens</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>1</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>53 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>18.9</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>7,203</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>4</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>106 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>37.6</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>14,362</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>16</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>275 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>57.4</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>21,909</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>64</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>931 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>67.5</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>25,770</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>256</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>3,489 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>71.1</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>27,139</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Four questions on a 1K-token state, 4,540 tokens</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>1</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>183 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>5.5 (22 decisions/s)</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>24,782</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>8</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>1,315 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>6.0 (24 decisions/s)</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>27,398</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>32</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>4,882 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>6.1 (24 decisions/s)</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>27,649</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>8K-token state, 8,224 tokens</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>1</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>334 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>2.99</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>24,620</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>4</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>1,280 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>3.12</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>25,682</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>16</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>5,101 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>3.13</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>25,726</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>One 1024×768 image, 429 tokens</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>1</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>118 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>8.3</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>3,541</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>16</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>466 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>33.6</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>14,398</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>64</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>1,197 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>49.9</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>21,416</td>
    </tr>
  </tbody>
</table>

### 3.2 Accuracy and parity

Both sets go through `/v1/systemone` as one `choice` question per item, with the same conversion as the v1 page. The reference rows run the same items through the checkpoint's own `DecisionModel` one at a time.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "50%"}} />
    <col style={{width: "25%"}} />
    <col style={{width: "25%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Implementation</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>Belebele (eng_Latn, 900)</th>
      <th style={{textAlign: "right", padding: "10px 12px", fontWeight: 700}}>WinoGrande (xl dev, 1,267)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>SGLang, GB300 recipe (TRT-LLM attention)</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}><strong>97.67%</strong></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}><strong>92.42%</strong></td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>SGLang, <code>--attention-backend flashinfer</code></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>97.67%</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>92.58%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>SGLang, <code>--attention-backend triton</code></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>97.67%</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>92.42%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Checkpoint's reference <code>DecisionModel</code> (Transformers, BF16, GB300)</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>97.67%</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>92.50%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>v1.1 served with causal attention, as SGLang did before #42645</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>97.00%</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>88.87%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>PPLX-Decider-v1-27B on SGLang, for comparison</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>96.67%</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>84.21%</td>
    </tr>
  </tbody>
</table>

Against the reference implementation, the recipe picks the same option on 2,164 of 2,167 items, and each of the three differences is a near-tie within 0.031 of an even split. Per item, the largest probability difference averages 0.002 (99th percentile 0.030, maximum 0.089). Prompts of 11,099 and 18,585 tokens, an image question, and a score question also matched the reference, the largest difference being 0.015 on the score question. Served with causal attention instead, v1.1 changed its top answer on 101 of the 2,167 items and lost 3.6 points on WinoGrande.

<Accordion title="Check accuracy on your deployment (Python)">

```python Example
import asyncio

import aiohttp
from datasets import load_dataset

URL = "http://localhost:30000/v1/systemone"


async def main():
    rows = load_dataset("facebook/belebele", "eng_Latn", split="test")
    limit = asyncio.Semaphore(32)

    async def ask(session, row):
        body = {
            "model": "perplexity-ai/pplx-decider-v1.1-27b",
            "state": row["flores_passage"],
            "questions": {
                "answer": {
                    "type": "choice",
                    "instructions": row["question"],
                    "criteria": {str(k): row[f"mc_answer{k}"] for k in range(1, 5)},
                }
            },
        }
        async with limit, session.post(URL, json=body) as response:
            response.raise_for_status()
            answer = (await response.json())["answers"]["answer"]
        return answer["choice"] == row["correct_answer_num"]

    async with aiohttp.ClientSession() as session:
        correct = await asyncio.gather(*(ask(session, row) for row in rows))
    print(f"Belebele eng_Latn: {sum(correct)}/{len(correct)} = {100 * sum(correct) / len(correct):.2f}%")


asyncio.run(main())
```

</Accordion>

<Accordion title="Example Output">

```text Output
Belebele eng_Latn: 879/900 = 97.67%
```

</Accordion>

## 4. Advanced Usage

### 4.1 Text Decisions

<Accordion title="Text Decision Example (Python)">

```python Example
import requests

response = requests.post(
    "http://localhost:30000/v1/systemone",
    json={
        "model": "perplexity-ai/pplx-decider-v1.1-27b",
        "state": "My Stripe integration keeps failing. I'm losing sales. Please help ASAP.",
        "questions": {
            "routing": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {
                    "billing": "Charges and refunds",
                    "technical_support": "Integration errors",
                    "sales": "Questions about buying a product",
                },
            },
            "urgency": {"type": "noul", "instructions": "Does this message express urgency?"},
        },
    },
    timeout=60,
)
response.raise_for_status()
for name, answer in response.json()["answers"].items():
    print(name, answer)
```

</Accordion>

<Accordion title="Example Output">

```text Output
routing {'type': 'choice', 'choice': 'technical_support', 'confidence': 0.9977658531751561, 'probabilities': {'billing': 0.0014030901008239556, 'technical_support': 0.9985105687834377, 'sales': 8.634111573847184e-05}}
urgency {'type': 'noul', 'noul': 0.9995671977710632}
```

</Accordion>

### 4.2 Image Decisions

The example output below came from a 512×512 solid red test image.

<Accordion title="Image Decision Example (Python)">

```python Example
import base64
from pathlib import Path

import requests

image = base64.b64encode(Path("screenshot.png").read_bytes()).decode("ascii")
response = requests.post(
    "http://localhost:30000/v1/systemone",
    json={
        "model": "perplexity-ai/pplx-decider-v1.1-27b",
        "state": "Look at the supplied image.",
        "images": [image],
        "questions": {
            "color": {
                "type": "choice",
                "instructions": "What is the dominant color?",
                "criteria": {"red": "Red", "green": "Green", "blue": "Blue", "other": "Another color"},
            }
        },
    },
    timeout=60,
)
response.raise_for_status()
print(response.json()["answers"]["color"])
```

</Accordion>

<Accordion title="Example Output">

```text Output
{'type': 'choice', 'choice': 'red', 'confidence': 0.9989117512916561, 'probabilities': {'red': 0.9991838134687421, 'green': 6.743375714875418e-05, 'blue': 8.12085765657198e-05, 'other': 0.0006675441975434462}}
```

</Accordion>
