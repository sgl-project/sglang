---
title: PPLX-Decider-v1-27B
description: "Deploy PPLX-Decider-v1-27B with SGLang. Perplexity's decision model, fine-tuned from Qwen3.8-27B, returns calibrated probabilities for choice, yes or no, and score questions on /v1/systemone, on one NVIDIA H200 or GB300 in BF16."
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

For all methods and hardware platforms, see the [official SGLang installation guide](../../../docs/get-started/install). The two paths below match the **Python / Docker** toggle in the command panel. PPLX-Decider support landed after v0.5.21, so until a release includes it, install a [nightly build](/docs/get-started/install#nightly-builds).

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

Pick your GPU to generate the launch command. The model runs on one GPU in BF16, and the decision support needs no flag: SGLang reads the checkpoint's `decision_config.json`, loads its decision readout, and answers on `/v1/systemone` with the prompt, answer codes, and calibrated temperature the model was trained with. The GB300 recipe adds `--disable-radix-cache` for speed, as explained in [Configuration Tips](#2-configuration-tips).

import { Deployment } from "/src/snippets/_deployment.jsx"
import { config }     from "/src/snippets/configs/perplexity-ai/pplx-decider-v1-27b.jsx"
import { benchmarks } from "/src/snippets/configs/perplexity-ai/pplx-decider-v1-27b-benchmarks.jsx"

<Deployment config={config} benchmarks={benchmarks} />

<Note>
  The H200 recipe is the original one, validated against the checkpoint's own reference implementation on one H200 when support landed ([#42183](https://github.com/sgl-project/sglang/pull/42183)), and it has no speed numbers yet. The GB300 recipe adds `--disable-radix-cache` (see [Configuration Tips](#2-configuration-tips)) and was measured end to end on one GB300 at SGLang commit `70f0b7351e`: four request shapes from 1 to 256 concurrent requests, Belebele and WinoGrande accuracy, and per-question parity with the reference implementation. [§3](#3-benchmarks) has the method and the full tables.
</Note>

## Playground

The Playground layers SGLang features on top of the recipe above. Any override changes the badge to **Not Verified** until that exact configuration is tested end to end. The [Configuration Tips](#2-configuration-tips) say which overrides were measured on GB300 and what they did. Tensor parallelism times data-parallel replicas must fit the GPUs on one node.

import { Playground } from "/src/snippets/_playground.jsx"

<Playground config={config} />

## 1. Model Introduction

**PPLX-Decider-v1-27B** is a decision model from Perplexity, fine-tuned from [Qwen3.8-27B](/cookbook/autoregressive/Qwen/Qwen3.8-27B) under the Apache-2.0 license. It replaces the language model head with a readout over 255 answer codes and answers choice, yes or no (`noul`), and score questions about a text state and optional images with a probability for every option, without generating text. Each question is one prefill pass, so the cost of a decision is the cost of reading its prompt. Its successor, [PPLX-Decider-v1.1-27B](/cookbook/autoregressive/Perplexity/PPLX-Decider-v1.1-27B), adds noncausal full attention and more training data.

The backbone is the Qwen3.8-27B hybrid: 64 layers in which 48 Gated DeltaNet (linear attention) layers alternate with 16 gated full-attention layers, a 5120-wide hidden state, and the Qwen3.8 vision encoder, for 26.1B parameters (52 GB in BF16). The checkpoint ships a bare `Qwen3_5Model` backbone, the 255-row readout in `readout.safetensors`, and `decision_config.json` with the answer-code token ids and a calibrated temperature of 2.2076. SGLang serves it as `Qwen3_5ForConditionalGeneration` and writes the readout into the language model head rows of the code tokens, so the probabilities come out of the regular scoring path.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "34%"}} />
    <col style={{width: "66%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Property</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Checkpoint</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><a href="https://huggingface.co/perplexity-ai/pplx-decider-v1-27b">perplexity-ai/pplx-decider-v1-27b</a> (BF16)</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.05)"}}>Base model</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Qwen3.8-27B, one epoch of supervised fine-tuning on decision data</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Question types</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}><code>choice</code> (up to 255 options), <code>noul</code> (yes or no), <code>score</code> (up to 10 levels)</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.05)"}}>Inputs</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Text or JSON state, optional images</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Prompt length</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Trained on question prompts of up to 8,192 tokens, and the backbone accepts 262,144</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.05)"}}>License</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Apache-2.0</td>
    </tr>
  </tbody>
</table>

Perplexity reports 85.71% overall accuracy across 11 public benchmarks for this checkpoint, against 74.76% for Qwen3.8-27B, measured through the Perplexity API. The per-benchmark table is on the model card. [§3](#3-benchmarks) measures accuracy and speed on SGLang.

**Resources:** [HuggingFace](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b), [reference inference code](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b/blob/main/inference.py), [Decision models in SGLang](/docs/supported-models/decision_models).

## 2. Configuration Tips

- **Route.** Send requests to `/v1/systemone`, the System One API the checkpoint was trained for. `/v1/decisions` refuses this checkpoint because its labels and prompt are not the ones the readout learned, and chat or generation requests are not meaningful without a language model head.
- **Clients.** Clients of the System One API, including the TypeSafe SDKs, work by pointing their base URL at the server. See [System One compatible API](/docs/supported-models/decision_models#system-one-compatible-api) for the request and response reference.
- **Images.** Pass each image in `images` as base64 bytes, a base64 data URL, or an `http(s)` URL. They precede the text in every question, as in the checkpoint's reference code. The checkpoint's processor resizes every image to between 65,536 and 262,144 pixels, so one image costs 64 to 256 prompt tokens: a 1024×768 screenshot is 234 tokens, and a 4K one is no more expensive than 1080p (252). Crop to the region that matters rather than sending a larger image.
- **What a decision costs.** Each question is its own prompt: an ~84-token wrapper (system message, chat template, answer instructions), the state, the question, and its options, read in one prefill pass. Several questions in one request are scored together, but each re-reads the whole state, so the cost of a request is roughly *questions × state tokens*.
- **Prefix cache.** On this hybrid model the prefix cache resumes from where an earlier prompt ended, because that is where the Gated DeltaNet state is saved, not from any shared prefix. An identical request measured 215 ms cold and 56 ms cached on GB300, but a different question about the same state, and the other questions in the same request, reused none of the state. The GB300 recipe therefore turns the cache off with `--disable-radix-cache`: unique-state traffic loses no reuse, each request holds one state slot instead of five, and the per-request state bookkeeping goes away, which took a short question from 59 to 51 ms and an image request from 126 to 87 ms with the attention kernel held fixed. Turn it back on only if your traffic repeats identical requests, the same state and the same question, often enough to matter.
- **Prompt length.** The model was fine-tuned on question prompts of up to 8,192 tokens, and the reference code refuses longer ones. SGLang serves the backbone's full 262,144-token context, so keep states within the trained length. A longer prompt is answered, but its accuracy is untested.
- **Throughput is compute-bound.** A decision is pure prefill, and one GB300 reads about 27,000 prompt tokens per second in BF16 once batches are full: about 71 requests per second for a 382-token question, and 3.3 for an 8K-token state. Past that point, more concurrency only adds queueing. The tables in [§3](#3-benchmarks) map concurrency to latency, so pick the highest concurrency that fits your latency budget.
- **Memory.** The weights take 52 GB, and the default `--mem-fraction-static` hands the rest of the GPU to the KV and Gated DeltaNet state pools. The startup log line `max_running_requests is capped to ... by the mamba state cache` is expected: on GB300, each request reserves five state slots with the prefix cache on (a cap of 118) and one with it off (593). Even the lower cap only binds for questions shorter than about 140 tokens, because a prefill batch fills its 16,384-token chunk first. `--mamba-ssm-dtype bfloat16` halves the state pool but measured no faster, so keep the checkpoint's FP32 state.
- **GB300 kernels.** With the prefix cache off, SGLang picks the TRT-LLM attention kernel with 64-token pages on GB300 (Triton with 1-token pages while the cache is on), and that kernel is where the long-state gain comes from: an 8K-token state drops from 377 to 325 ms and saturates at 3.30 instead of 2.87 requests per second, the same as forcing `--attention-backend trtllm_mha --page-size 64` with the cache on. FlashInfer's Gated DeltaNet prefill (`--linear-attn-prefill-backend flashinfer`) adds another 4-8% on text, taking the 8K-token state to 300 ms and 3.56 requests per second, but it made image requests slower (101 vs 87 ms alone, 9% less throughput), so the recipe leaves it to the Playground for text-only traffic. A larger prefill chunk (`--chunked-prefill-size 32768`) did not help: one 16,384-token chunk already keeps the GPU busy.
- **More GPUs.** The model fits one GPU, so add throughput with data-parallel replicas: `--dp-size 2` on two GB300s served 139 short questions per second (1.96× one GPU) and 6.5 8K-token states per second, and because the questions of one request are spread over the replicas, it also cut four questions on a 1K-token state from 179 to 104 ms. The extra dispatch hop added about 15 ms to a lone short question. Tensor parallelism only pays off for latency on long states: `--tp 2` took an 8K-token state from 325 to 208 ms, but delivered less throughput than two replicas and made a short question slower (68 ms), because the all-reduce dominates small batches. SGLang Model Gateway does not route `/v1/systemone` yet, so to spread load over several servers instead of `--dp-size`, use a plain HTTP load balancer.
- **FP8.** `--quantization fp8` quantizes the BF16 weights at load time and measured about 20% more throughput on GB300 (84 vs 70 requests per second for short questions), with the same Belebele and WinoGrande scores. But its probabilities moved by 0.010 on average and up to 0.20 against the reference implementation, four times the drift of any BF16 configuration, and 0.9% of answers changed their top option. The calibrated probabilities are this model's output, so the recipes stay in BF16, and the Playground offers FP8 for workloads that only use the top choice.
- **Reproducibility.** Two launches of the same recipe returned bit-identical probabilities for the same requests at the same concurrency. Against the checkpoint's reference implementation, the largest probability difference per question averages 0.003 (see [§3](#3-benchmarks)), because batch composition and kernels differ, and near-ties at 0.500 can flip.

## 3. Benchmarks

All numbers in this section come from one NVIDIA GB300 (288 GB) at SGLang commit `70f0b7351e`, with checkpoint revision `5117a6c`, served with the GB300 recipe above unless a row says otherwise.

**Method.** A closed-loop client keeps a fixed number of `/v1/systemone` requests in flight. Every request carries a unique random state, so requests share nothing but the system message, and each concurrency level starts after 8 warmup requests and a cache flush. Latency is the median end-to-end time of a request, and prompt tokens come from each response's `usage.input_tokens`. Four request shapes cover the common cases: one four-option choice question about a 256-token state, four questions (two choice, one yes or no, one score) about a 1,024-token state, one question about an 8,192-token state, and one question about a 1024×768 JPEG with a 64-token state.

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
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>51 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>19.4</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>7,413</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>4</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>103 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>32.8</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>12,525</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>16</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>267 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>59.0</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>22,531</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>64</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>882 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>70.8</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>27,039</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>256</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>3,521 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>71.1</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>27,164</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Four questions on a 1K-token state, 4,540 tokens</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>1</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>179 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>5.6 (22 decisions/s)</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>25,291</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>8</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>1,321 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>6.0 (24 decisions/s)</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>27,332</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>32</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>5,206 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>6.0 (24 decisions/s)</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>27,208</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>8K-token state, 8,224 tokens</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>1</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>325 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>3.08</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>25,297</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>4</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>1,220 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>3.26</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>26,785</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>16</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>4,796 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>3.30</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>27,175</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>One 1024×768 image, 429 tokens</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>1</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>87 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>11.5</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>4,945</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>16</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>414 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>37.5</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>16,078</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>64</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>1,170 ms</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>52.6</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>22,572</td>
    </tr>
  </tbody>
</table>

One GB300 tops out at about 27,000 prompt tokens per second for every text shape, so requests per second follow from prompt length. Latency rises with concurrency once that ceiling is reached, so choose the concurrency from the latency you can accept. A decision has no output tokens, so in the benchmark card above TTFT is the whole request latency, and TPOT and interactivity do not apply.

### 3.2 Accuracy and parity

Both sets go through `/v1/systemone` as one `choice` question per item. Belebele sends the passage as the state, the question as the instructions, and the four answers as options `1` to `4`. WinoGrande sends the sentence as the state, asks which option fills the blank, and offers the two candidates. The reference row runs the same items through the checkpoint's own `DecisionModel` one at a time.

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
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>SGLang, GB300 recipe</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}><strong>96.67%</strong></td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}><strong>84.21%</strong></td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>SGLang, no extra flag</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>96.67%</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>84.37%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Checkpoint's reference <code>DecisionModel</code> (Transformers, BF16, GB300)</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>96.67%</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.02)"}}>84.37%</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Perplexity, model card (Perplexity API, its own prompt conversion)</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>94.00%</td>
      <td style={{padding: "9px 12px", textAlign: "right", backgroundColor: "rgba(255,255,255,0.05)"}}>83.30%</td>
    </tr>
  </tbody>
</table>

Against the reference implementation, the recipe picks the same option on 2,161 of 2,167 items, and each of the six differences is a near-tie that one side scores 0.500 against 0.500 and the other 0.486 against 0.514. Per item, the largest probability difference averages 0.003 (99th percentile 0.024, maximum 0.054). Perplexity's figures come from a different prompt conversion, so they are context rather than a target.

### 3.3 What else was measured

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "33%"}} />
    <col style={{width: "45%"}} />
    <col style={{width: "22%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Change from the GB300 recipe</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Measured effect</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Verdict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Prefix cache on (no extra flag)</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Short question 58 ms alone, 8K-token state 377 ms alone and 2.87 requests/s saturated (recipe: 51 ms, 325 ms, 3.30)</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Only for traffic that repeats identical requests</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}><code>--linear-attn-prefill-backend flashinfer</code></td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Text 4-8% faster (8K-token state 300 ms, 3.56 requests/s), images slower (101 ms alone, 9% less throughput)</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Text-only traffic</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}><code>--dp-size 2</code> (2 GPUs)</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>139 short questions/s and 6.5 8K-token states/s, four questions on a 1K-token state in 104 ms, a lone short question 15 ms slower</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Scale-out</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}><code>--tp 2</code> (2 GPUs)</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>8K-token state 208 ms alone, 124 short questions/s saturated, a lone short question in 68 ms</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Long states with a latency target</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}><code>--quantization fp8</code></td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>About 20% more throughput, same accuracy, probabilities shifted by 0.010 on average and up to 0.20</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Only when you use the top choice alone</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}><code>--chunked-prefill-size 32768 --max-prefill-tokens 32768</code></td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>No gain (within 4%)</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.05)"}}>Keep the default</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}><code>--mamba-ssm-dtype bfloat16</code></td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>No gain, though it halves the state pool</td>
      <td style={{padding: "9px 12px", textAlign: "left", backgroundColor: "rgba(255,255,255,0.02)"}}>Keep the default</td>
    </tr>
  </tbody>
</table>

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
            "model": "perplexity-ai/pplx-decider-v1-27b",
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
Belebele eng_Latn: 870/900 = 96.67%
```

</Accordion>

<Accordion title="Measure speed on your deployment (Python)">

Run it as `python bench.py <state words> <concurrency> <requests>`. With 270 words, a request is 378 tokens, close to the short shape above.

```python Example
import asyncio
import random
import statistics
import sys
import time

import aiohttp

BASE = "http://localhost:30000"
STATE_WORDS, CONCURRENCY, REQUESTS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
WORDS = ("customer payment refund invoice order delivery account login error update server outage "
         "contract price plan report urgent today please help the a and with for from to of in on").split()
QUESTION = {
    "type": "choice",
    "instructions": "Which team should handle this request?",
    "criteria": {"billing": "Charges and refunds", "technical_support": "Integration errors",
                 "sales": "Questions about buying a product", "legal": "Contracts and liability"},
}


def body(rng):
    # A unique random state per request, so no request reuses another's cache.
    state = " ".join(rng.choices(WORDS, k=STATE_WORDS))
    return {"model": "perplexity-ai/pplx-decider-v1-27b", "state": state, "questions": {"q": QUESTION}}


async def main():
    rng = random.Random(0)
    bodies = [body(rng) for _ in range(REQUESTS)]
    latencies, tokens = [], 0
    async with aiohttp.ClientSession() as session:
        for item in [body(rng) for _ in range(8)]:  # warmup
            async with session.post(f"{BASE}/v1/systemone", json=item) as response:
                response.raise_for_status()
        async with session.post(f"{BASE}/flush_cache") as response:
            await response.read()
        queue = iter(bodies)

        async def worker():
            nonlocal tokens
            for item in queue:
                start = time.perf_counter()
                async with session.post(f"{BASE}/v1/systemone", json=item) as response:
                    response.raise_for_status()
                    tokens += (await response.json())["usage"]["input_tokens"]
                latencies.append(time.perf_counter() - start)

        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(CONCURRENCY)))
        elapsed = time.perf_counter() - start
    print(f"{REQUESTS / elapsed:.1f} req/s, {tokens / elapsed:.0f} prompt tok/s, "
          f"P50 {1000 * statistics.median(latencies):.1f} ms, {tokens / REQUESTS:.0f} tokens/request")


asyncio.run(main())
```

</Accordion>

<Accordion title="Example Output">

```text Output
$ python bench.py 270 1 64
19.6 req/s, 7393 prompt tok/s, P50 51.0 ms, 378 tokens/request
$ python bench.py 270 64 512
68.4 req/s, 25865 prompt tok/s, P50 899.0 ms, 378 tokens/request
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
        "model": "perplexity-ai/pplx-decider-v1-27b",
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
routing {'type': 'choice', 'choice': 'technical_support', 'confidence': 0.987854396908662, 'probabilities': {'billing': 0.004840538662952211, 'technical_support': 0.9919029312724412, 'sales': 0.0032565300646064375}}
urgency {'type': 'noul', 'noul': 0.9929980993021188}
```

</Accordion>

### 4.2 Image Decisions

The example output below came from a mostly red test image.

<Accordion title="Image Decision Example (Python)">

```python Example
import base64
from pathlib import Path

import requests

image = base64.b64encode(Path("screenshot.png").read_bytes()).decode("ascii")
response = requests.post(
    "http://localhost:30000/v1/systemone",
    json={
        "model": "perplexity-ai/pplx-decider-v1-27b",
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
{'type': 'choice', 'choice': 'red', 'confidence': 0.9498445161931509, 'probabilities': {'red': 0.9623833871448632, 'green': 0.009006826727044857, 'blue': 0.00926546931819573, 'other': 0.01934431680989631}}
```

</Accordion>
