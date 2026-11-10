# GLM-5.3-Flash on 8× V100

GLM-5.3-Flash on 8× V100-SXM2-32GB, set up for one agentic coding session: a main conversation that grows towards 200k tokens, subagent conversations next to it, and many short tool-call turns. A turn that extends a cached conversation prefills only the new tokens. The cache holds the linear-attention states of two conversations side by side, so a subagent call does not evict the main session. MTP only proposes tokens; the target model decides each one. On the check prompts, greedy output with MTP matched the server without MTP token for token.

## Get the model

The validated checkpoint is RadixArk's NVFP4 quantisation (ModelOpt). 190 GB. The MTP draft layer is in the same repo. The export carries a vision tower; this port serves text only (`--language-only`). The model is MIT-licensed.

```bash
pip install -U "huggingface_hub[cli]"
hf download RadixArk/GLM-5.3-Flash-NVFP4 \
  --local-dir ~/models/GLM-5.3-Flash-NVFP4

export GLM53_MODEL=~/models/GLM-5.3-Flash-NVFP4
```

| | |
|---|---|
| checkpoint | [`RadixArk/GLM-5.3-Flash-NVFP4`](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4) |
| base model | [`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) |
| quantisation | NVFP4 weights (W4A4 export) run as W4A16 with FP16 activations, FP16 KV cache. The draft layer's BF16 routed experts are quantised to NVFP4 at load |

## Talking to it

```bash
# OpenAI-compatible
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-flash-nvfp4","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'

# Anthropic Messages API — Claude Code connects to this directly
curl http://127.0.0.1:11435/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"glm53-flash-nvfp4","max_tokens":256,"messages":[{"role":"user","content":"Hello"}]}'
```

The model reasons before it answers (`glm45` parser), and tool calls come back as structured `tool_calls` (`glm47` parser). An empty `content` with a large `completion_tokens` means the reply hit `max_tokens` while still thinking.

The model has three reasoning efforts: Low, High and Max. The server reads `reasoning_effort` (OpenAI) or `output_config.effort` (Anthropic, which Claude Code's effort setting sends) and maps it as follows: `minimal`/`low` to Low, `medium`/`high` to High, and `xhigh`/`max` (or no effort) to Max. Claude Code's default is `xhigh`, so it runs at Max.

## Measured performance

The reference recipe below: TP=8, MTP 3 steps / 4 draft tokens, `--max-running-requests 1`, temperature 0. KV pool 242,880 tokens, context 240,640.

**Decode** (2026-10-06). Six chat prompts, 512 tokens each, streamed from `/generate`; tok/s excludes TTFT. Accept length is tokens per target forward.

| prompt | tok/s | accept length |
|---|---:|---:|
| code | 174.3 | 2.89 |
| refactor | 180.0 | 2.99 |
| explain | 152.3 | 2.52 |
| math | 196.6 | 3.26 |
| agent step | 179.0 | 2.98 |
| German prose | 175.3 | 2.89 |
| **all six** | **175.2** | |

**Agent session** (2026-10-02). Through `/v1/chat/completions` with a `bash` tool: source files as the first user message, then three turns that each append an assistant tool call and its result (~500–700 new tokens). TTFT includes rendering and tokenizing the whole chat on the server. The first request with a new set of tools also compiles its tool-call grammar, about 7 s once per tool set; the rows below were measured after that.

| context | cold first turn | tool turns 1 / 2 / 3 |
|---|---:|---:|
| 32,187 tokens | 16.1 s | 554 / 490 / 443 ms |
| 128,188 tokens | 68.1 s | 701 / 619 / 550 ms |
| 190,188 tokens | 104.7 s | 768 / 702 / 602 ms |

**Main session plus subagents** (2026-10-02). A 190k main conversation, then two fresh ~20k subagent conversations of three turns each, with a main turn after each subagent. The main turns stay cached:

| turn | TTFT |
|---|---:|
| main, cold / next turn | 103.9 s / 757 ms |
| subagent 1, cold / turns 2–3 | 9.68 s / 538, 515 ms |
| main | 694 ms |
| subagent 2, cold / turns 2–3 | 9.62 s / 531, 503 ms |
| main | 633 ms |

## Reference recipe

The wrapper is the supported entry and the recipe the numbers above were measured on. It defaults to MTP with 3 steps; `GLM53_MTP_STEPS=0` turns MTP off and switches the memory and context defaults to the no-MTP values below.

```bash
export GLM53_MODEL=~/models/GLM-5.3-Flash-NVFP4
bash scripts/serve_glm53_flash_nvfp4_v100.sh
```

Expanded (what that script runs with MTP). The env block is not implied by the CLI flags: it turns on the Volta decode kernels and the two-level all-reduce.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_CUDA_ARCH_LIST=7.0
export FLASHINFER_DISABLE_VERSION_CHECK=1
export NCCL_ALGO=allreduce:tree
export SGLANG_MAMBA_CONV_DTYPE=float16
export SGLANG_SM70_FORCE_FP16=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_NUMA_BIND_V2=0
export SGLANG_SM70_GLM_NVFP4_GEMV=1
export SGLANG_SM70_GLM_NVFP4_MOE_DECODE=1
export SGLANG_SM70_DENSE_GEMV=1
export SGLANG_DSV41_HIER_AR=1
export SGLANG_DSV41_HIER_AR_CA=1
export SGLANG_DSV41_HIER_AR_PUSH=1
export SGLANG_NVFP4_CKPT_NVFP4_NEXTN_MOE=1

python -m sglang.launch_server \
  --model-path "${GLM53_MODEL}" \
  --served-model-name glm53-flash-nvfp4 \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --dtype float16 \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend marlin \
  --language-only \
  --tensor-parallel-size 8 \
  --ep-size 1 \
  --attention-backend dsa \
  --linear-attn-backend triton \
  --kv-cache-dtype auto \
  --disable-custom-all-reduce \
  --disable-prefill-cuda-graph \
  --cuda-graph-bs-decode 1 \
  --max-running-requests 1 \
  --max-mamba-cache-size 12 \
  --mamba-max-states-per-path 2 \
  --mamba-full-memory-ratio 0.15 \
  --mamba-radix-cache-strategy extra_buffer \
  --chunked-prefill-size 2048 \
  --max-prefill-tokens 2048 \
  --warmups prefix_reuse,sampling \
  --sleep-on-idle \
  --context-length 240640 \
  --mem-fraction-static 0.935 \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path "${GLM53_MODEL}" \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --host 0.0.0.0 \
  --port 11435
```

| knob | ship value | why |
|---|---|---|
| MTP | on, 3 steps / 4 draft tokens | Fastest measured. On an earlier build 2 steps gave 117 tok/s against about 123 for 3; 4 steps (5-row verify) falls off the 4-row kernels and dropped to 34 |
| draft experts | NVFP4 at load (`SGLANG_NVFP4_CKPT_NVFP4_NEXTN_MOE=1`) | Draft weights 2.4 → 1.06 GB per rank, which buys KV pool, and acceptance is the same as the BF16 draft |
| `--ep-size` | 1 | Experts TP-sliced: every rank does the same MoE work per token. With EP=8 the other ranks waited for the one holding most routes |
| `--max-running-requests` | 1 | The decode and verify kernels give each row the same result as batch 1 only while the whole launch has at most 4 rows (1 request × 4 draft tokens) |
| `--mem-fraction-static` | 0.935 | Leaves a 242,880-token KV pool. Without MTP use 0.88 (225,600-token pool) |
| `--context-length` | 240,640 | Kept below the pool so prompt plus completion always fits. Without MTP use 223,232 |
| `--max-mamba-cache-size` / `--mamba-max-states-per-path` | 12 / 2 | A running request pins 4 slots and admission wants 3 free. With 8 slots a subagent call evicted the main session's states, which cost a 100k+ re-prefill on the next main turn |
| prefill chunk | 2048 | The measured value. Larger chunks have not been tried with the memory a 0.935 pool leaves |
| `--warmups prefix_reuse,sampling` | on | Loads the kernels of a cache hit and a grammar-constrained decode at startup, and builds FlashInfer's sampling module. Otherwise they load on the first real tool turn, when little memory is free, and the first sampled request waits for the build |
| `--sleep-on-idle` | on | Without it the eight idle scheduler loops keep about 5 CPU cores busy. Decode speed is the same either way |
| `--disable-custom-all-reduce` + `SGLANG_DSV41_HIER_AR*` | on | The 8-GPU mesh is two NVLink quads plus bridges. Custom all-reduce inside each quad, then across the bridge pair |

Without MTP (`GLM53_MTP_STEPS=0` in the wrapper): drop the five `--speculative-*` lines and `SGLANG_NVFP4_CKPT_NVFP4_NEXTN_MOE`, and use `--mem-fraction-static 0.88 --context-length 223232`.

## Limitations

- One request at a time. A second request queues behind the first.
- Text only (`--language-only`).
- A fresh subagent conversation pays its own cold prefill (about 10 s at 20k tokens).
- The first request with a new set of tools compiles its tool-call grammar, about 7 s once per tool set.
- Checked with the benchmarks above and GSM8K (0.923 over 1,319 questions with MTP, on an earlier build of this recipe). No multi-hour soak yet.

## PCIe V100s

The recipe is measured on SXM2. The two-level all-reduce (`SGLANG_DSV41_HIER_AR*`) assumes two NVLink quads joined by bridges; on PCIe cards set `SGLANG_DSV41_HIER_AR=0` before the script. Untested on an 8-card PCIe box; expect slower decode.
