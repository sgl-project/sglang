<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://lmsys.org/blog/"><b>Blog</b></a> |
<a href="https://docs.sglang.io/"><b>Documentation</b></a> |
<a href="https://roadmap.sglang.io/"><b>Roadmap</b></a> |
<a href="https://slack.sglang.io/"><b>Join Slack</b></a> |
<a href="https://meet.sglang.io/"><b>Weekly Dev Meeting</b></a> |
<a href="https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides"><b>Slides</b></a>
</p>

## News
- [2025/01] 🔥 SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2026-01-16-sglang-diffusion/)).
- [2025/12] SGLang provides day-0 support for latest open models ([MiMo-V2-Flash](https://lmsys.org/blog/2025-12-16-mimo-v2-flash/), [Nemotron 3 Nano](https://lmsys.org/blog/2025-12-15-run-nvidia-nemotron-3-nano/), [Mistral Large 3](https://github.com/sgl-project/sglang/pull/14213), [LLaDA 2.0 Diffusion LLM](https://lmsys.org/blog/2025-12-19-diffusion-llm/), [MiniMax M2](https://lmsys.org/blog/2025-11-04-miminmax-m2/)).
- [2025/10] 🔥 SGLang now runs natively on TPU with the SGLang-Jax backend ([blog](https://lmsys.org/blog/2025-10-29-sglang-jax/)).
- [2025/09] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part II): 3.8x Prefill, 4.8x Decode Throughput ([blog](https://lmsys.org/blog/2025-09-25-gb200-part-2/)).
- [2025/09] SGLang Day 0 Support for DeepSeek-V3.2 with Sparse Attention ([blog](https://lmsys.org/blog/2025-09-29-deepseek-V32/)).
- [2025/08] SGLang x AMD SF Meetup on 8/22: Hands-on GPU workshop, tech talks by AMD/xAI/SGLang, and networking ([Roadmap](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_roadmap.pdf), [Large-scale EP](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_ep.pdf), [Highlights](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_highlights.pdf), [AITER/MoRI](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_aiter_mori.pdf), [Wave](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_wave.pdf)).

<details>
<summary>More</summary>

- [2025/11] SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)).
- [2025/10] PyTorch Conference 2025 SGLang Talk ([slide](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/sglang_pytorch_2025.pdf)).
- [2025/10] SGLang x Nvidia SF Meetup on 10/2 ([recap](https://x.com/lmsysorg/status/1975339501934510231)).
- [2025/08] SGLang provides day-0 support for OpenAI gpt-oss model ([instructions](https://github.com/sgl-project/sglang/issues/8833))
- [2025/06] SGLang, the high-performance serving infrastructure powering trillions of tokens daily, has been awarded the third batch of the Open Source AI Grant by a16z ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
- [2025/05] Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
- [2025/06] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part I): 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
- [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
- [2025/03] SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
- [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
- [2025/01] SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
- [2024/12] v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
- [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).
- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## About
SGLang is a high-performance serving framework for large language models and multimodal models.
It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters.
Its core features include:

- **Fast Runtime**: Provides efficient serving with RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.
- **Broad Model Support**: Supports a wide range of language models (Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse), reward models (Skywork), and diffusion models (WAN, Qwen-Image), with easy extensibility for adding new models. Compatible with most Hugging Face models and OpenAI APIs.
- **Extensive Hardware Support**: Runs on NVIDIA GPUs (GB200/B300/H100/A100/Spark), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, and more.
- **Active Community**: SGLang is open-source and supported by a vibrant community with widespread industry adoption, powering over 400,000 GPUs worldwide.
- **RL & Post-Training Backbone**: SGLang is a proven rollout backend across the world, with native RL integrations and adoption by well-known post-training frameworks such as [**AReaL**](https://github.com/inclusionAI/AReaL), [**Miles**](https://github.com/radixark/miles), [**slime**](https://github.com/THUDM/slime), [**Tunix**](https://github.com/google/tunix), [**verl**](https://github.com/volcengine/verl) and more.

## Getting Started
- [Install SGLang](https://docs.sglang.io/get_started/install.html)
- [Quick Start](https://docs.sglang.io/basic_usage/send_request.html)
- [Backend Tutorial](https://docs.sglang.io/basic_usage/openai_api_completions.html)
- [Frontend Tutorial](https://docs.sglang.io/references/frontend/frontend_tutorial.html)
- [Contribution Guide](https://docs.sglang.io/developer_guide/contribution_guide.html)

## Benchmark and Performance
Learn more in the release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/), [GB200 rack-scale parallelism](https://lmsys.org/blog/2025-09-25-gb200-part-2/).

## Adoption and Sponsorship
SGLang has been deployed at large scale, generating trillions of tokens in production each day. It is trusted and adopted by a wide range of leading enterprises and institutions, including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, InnoMatrix, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Jam & Tea Studios, Baseten, and other major technology organizations across North America and Asia.
As an open-source LLM inference engine, SGLang has become the de facto industry standard, with deployments running on over 400,000 GPUs worldwide.
SGLang is currently hosted under the non-profit open-source organization [LMSYS](https://lmsys.org/about/).

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact Us
For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at sglang@lmsys.org

## Acknowledgment
We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
