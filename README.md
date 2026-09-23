# SGLang: Fast inference for LLMs and multimodal models

<p align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang" width="400">
</p>

SGLang is an open-source framework for fast inference with large language and multimodal models. Whether you're serving your first model, processing a dataset, or building a training pipeline, SGLang helps you get more out of your hardware, from a single GPU to a distributed cluster.

👋 New here? Start with the example below, explore [sglang.io](https://www.sglang.io/), or find a recipe for your model in the [Cookbook](https://cookbook.sglang.io/). Join our [Slack](https://slack.sglang.io/) for development and discussion. Want to meet the people building SGLang? Come join a meetup, workshop, or office hour. You can find us at [SGLang Events](https://www.sglang.io/events).

## Install and Quick Start

### Install

Choose one of the following options. See the [installation guide](https://docs.sglang.io/docs/get-started/install) for platform-specific requirements.

#### Option 1: Docker

Requires Linux, a supported NVIDIA GPU with a CUDA 13-compatible driver, Docker, and NVIDIA Container Toolkit.

Pull the image, which includes SGLang and its dependencies:

```bash
docker pull lmsysorg/sglang:latest
```

Start a container shell:

```bash
docker run -it --gpus all \
  --shm-size 32g \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  lmsysorg/sglang:latest bash
```

#### Option 2: Install with uv

Alternatively, install SGLang in a Python environment with [uv](https://docs.astral.sh/uv/getting-started/installation/). This requires a compatible Linux/NVIDIA environment.

Create and activate a virtual environment, then install SGLang:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --prerelease=allow sglang
```

### Start the server

The following example serves Qwen3.8-27B-FP8. For other models, find launch commands in the [SGLang Cookbook](https://cookbook.sglang.io/).

Run the following command inside the Docker container or in your activated Python environment:

```bash
sglang serve \
  --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --attention-backend flashinfer \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-full-memory-ratio 4.59 \
  --host 0.0.0.0 \
  --port 30000 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-ssm-dtype float32
```

### Send a request

Send a test request to verify that the server is reachable and the model can generate a response. Once the server is ready, run in another terminal:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.8-27B-FP8",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ]
  }'
```

The answer is in `choices[0].message.content`. See the [Quickstart guide](https://docs.sglang.io/docs/get-started/quickstart) for more API examples.

Clients can use an existing OpenAI-compatible endpoint without installing SGLang locally.

## SGLang Diffusion

SGLang Diffusion accelerates image and video generation with diffusion models. It supports optimized attention kernels, caching, quantization, and multi-GPU inference. See the [installation guide](https://docs.sglang.io/docs/sglang-diffusion/installation) to get started and the [deployment guide](https://docs.sglang.io/docs/sglang-diffusion/deployment_cookbook) for performance and memory configurations.

## SGLang Omni

[SGLang Omni](https://github.com/sgl-project/sglang-omni) is a separate serving framework for audio and unified multimodal models, including text-to-speech (TTS) and automatic speech recognition (ASR). It supports streaming and OpenAI-compatible APIs for speech generation and transcription. See the [documentation](https://sgl-project.github.io/sglang-omni/) for supported models, installation, and usage.

## RL and Post-Training

Use SGLang as the rollout engine in your training workflow. Integrations include [Miles](https://github.com/radixark/miles), [AReaL](https://github.com/inclusionAI/AReaL), [slime](https://github.com/THUDM/slime), [Tunix](https://github.com/google/tunix), and [verl](https://github.com/volcengine/verl). Follow your framework's guide for installation, weight updates, and resource management.

## Mini-SGLang

**Curious how it all works? Start with ~5,000 lines of Python.**

We built [Mini-SGLang](https://github.com/sgl-project/mini-sglang) to make inference engines easier to understand. Read the code, run a model, and try changing how it works. Along the way, you'll explore:

- Radix cache reuses KV cache across requests with shared prefixes.
- Overlap scheduling overlaps CPU scheduling with GPU computation.
- Chunked prefill processes long prompts in chunks.
- Tensor parallelism distributes inference across GPUs.
- Optimized kernels integrate FlashAttention and FlashInfer.

Read the [source code](https://github.com/sgl-project/mini-sglang), follow the [architecture walkthrough](https://github.com/sgl-project/mini-sglang/blob/main/docs/structures.md), or [run your first model](https://github.com/sgl-project/mini-sglang#readme).

## Development and Contributing

Contributions are welcome, from bug fixes and documentation to model support and performance improvements.

### Development setup

Start from the `lmsysorg/sglang:dev` Docker image, which provides development tools and most dependencies. Clone or mount your SGLang checkout inside the container, then install it in editable mode from the repository root so tests use your local Python changes:

```bash
pip install -e "python"
```

In an activated virtual environment, you can use `uv pip install --prerelease=allow -e "python"` instead. See the [development guide](https://docs.sglang.io/docs/developer_guide/development_guide_using_docker) for container setup and testing.

### Contribute

1. Fork the repository and create a branch for your changes. For larger changes, discuss your proposal in a [GitHub issue](https://github.com/sgl-project/sglang/issues) or on [Slack](https://slack.sglang.io/).
2. Make your changes, run the relevant tests, and add regression coverage for fixes or new behavior. Run `pre-commit run --all-files` before submitting.
3. Open a pull request describing the change and how you tested it. Include benchmarks or accuracy evaluations when relevant.

See the [contributor guide](https://docs.sglang.io/docs/developer_guide/contribution_guide) for formatting, testing, and pull request instructions. Documentation contributors can start with the [docs guide](docs/README.md).

## Community and Sponsorship

SGLang is hosted by [LMSYS](https://lmsys.org/about/), a non-profit open-source organization.

- **Community discussions:** Join [Slack](https://slack.sglang.io/) for technical questions and development discussions.
- **Events:** Find meetups, workshops, and office hours on [SGLang Events](https://www.sglang.io/events).
- **Updates:** Follow [X](https://x.com/lmsysorg) and [LinkedIn](https://www.linkedin.com/company/sgl-project/) for project updates, and the [LMSYS Blog](https://lmsys.org/blog/) for release announcements and technical articles.
- **Project resources:** Explore the [documentation](https://docs.sglang.io/), [Cookbook](https://cookbook.sglang.io/), [roadmap](https://roadmap.sglang.io/), [release notes](https://github.com/sgl-project/sglang/releases), [issue tracker](https://github.com/sgl-project/sglang/issues), and [contributor guide](https://docs.sglang.io/docs/developer_guide/contribution_guide).
- **Contact Us:** For enterprise adoption and deployment, technical consulting, sponsorship, or partnership inquiries, please contact [sglang@lmsys.org](mailto:sglang@lmsys.org).
- **Contributor sponsorship:** Long-term active SGLang contributors are eligible for coding agent sponsorship, including Cursor, Claude Code, or OpenAI Codex. To apply, email [sglang@lmsys.org](mailto:sglang@lmsys.org) with links to your key commits or pull requests.

## Trusted by Industry and Research

SGLang serves production workloads across AI labs, cloud platforms, enterprises, and universities.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Organizations adopting SGLang" width="800">

## Acknowledgment
We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
