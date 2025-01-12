<div align="center"  id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![](https://img.shields.io/badge/Gurubase-(experimental)-006BFF)](https://gurubase.io/g/sglang)

</div>

--------------------------------------------------------------------------------

For KV cache offloading preview users, install `gmlwns2000/hip-ainl` first.

```bash
# Following setting is tested on 1x RTX 4090 24GB

# Maximum batch size to capture. Larger size requires more temporary buffers.
export SRT_MAX_BATCH=1;
# You can disable chunked prefill by change this into -1.
export CHUNK_PREFILL=16384;
# Any RoPE based attention models are supported in theoritically.
# However currently we are supports `llama.py` models. (Llama Family)
export MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4";
# You can set upper limit of maximum extended context window. 
# Training-free and unlimited.
export EXTENDED_CONTEXT_LEN=196608;
# You can change this flag into 1, if you want test online cache update. (exprimental)
export DEBUG_ONLINE=0;

python -m sglang.launch_server \
    --model-path $MODEL \
    # Supports auto and fp8_e5m2. Online cache updates are not supports fp8.
    --kv-cache-dtype auto \
    # You can increase this to use multi GPU.
    --tp-size 1 \
    --port 30000 \
    --chunked-prefill-size $CHUNK_PREFILL \
    --max-prefill-tokens $CHUNK_PREFILL \
    --context-length $EXTENDED_CONTEXT_LEN \
    --max-total-tokens $EXTENDED_CONTEXT_LEN \
    --enable-hip-attention \
    # You can turn off this flag to disable offloading. 
    # Offloading may have difference in decoding result.
    --enable-hip-offload \
    # For on-gpu offloading cache in masking kernel, 
    # allocate size of cache in num of tokens. This is shared by whole batch.
    --hip-max-mask-cache-token-size 32000 \
    # For on-gpu offloading cache in block sparse attention kernel, 
    # allocate size of cache in num of tokens. This is shared by whole batch.
    --hip-max-sa-cache-token-size 10000;
```

--------------------------------------------------------------------------------

| [**Blog**](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
| [**Documentation**](https://sgl-project.github.io/)
| [**Join Slack**](https://join.slack.com/t/sgl-fru7574/shared_invite/zt-2tmmp6flg-89dOlJW2TjnBrTRk1I_~GA)
| [**Join Bi-Weekly Development Meeting**](https://docs.google.com/document/d/1xEow4eIM152xNcRxqZz9VEcOiTQo8-CEuuQ5qTmkt-E/edit?usp=sharing)
| [**Slides**](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides) |

## News
- [2024/12] 🔥 SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- [2024/10] 🔥 The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/09] SGLang v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
- [2024/07] Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).

<details>
<summary>More</summary>

- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/04] SGLang is used by the official **LLaVA-NeXT (video)** release ([blog](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## About
SGLang is a fast serving framework for large language models and vision language models.
It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.
The core features include:

- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, jump-forward constrained decoding, overhead-free CPU scheduler, continuous batching, token attention (paged attention), tensor parallelism, FlashInfer kernels, chunked prefill, and quantization (FP8/INT4/AWQ/GPTQ).
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Extensive Model Support**: Supports a wide range of generative models (Llama, Gemma, Mistral, QWen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork), with easy extensibility for integrating new models.
- **Active Community**: SGLang is open-source and backed by an active community with industry adoption.

## Getting Started
- [Install SGLang](https://sgl-project.github.io/start/install.html)
- [Send requests](https://sgl-project.github.io/start/send_request.html)
- [Backend: SGLang Runtime (SRT)](https://sgl-project.github.io/backend/backend.html)
- [Frontend: Structured Generation Language (SGLang)](https://sgl-project.github.io/frontend/frontend.html)

## Benchmark And Performance
Learn more in our release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

## Roadmap
[Development Roadmap (2024 Q4)](https://github.com/sgl-project/sglang/issues/1487)

## Adoption and Sponsorship
The project is supported by (alphabetically): AMD, Baseten, Etched, Hyperbolic, Jam & Tea Studios, LinkedIn, Meituan, NVIDIA, RunPod, Stanford, UC Berkeley, xAI and 01.AI.

## Acknowledgment and Citation
We learned from the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
Please cite our paper, [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), if you find the project useful.
