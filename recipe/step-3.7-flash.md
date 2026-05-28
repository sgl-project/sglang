## 1. Introduction

Step 3.7 Flash is a 198B-parameter Mixture-of-Experts (MoE) vision-language model that combines a 196B-parameter language backbone with a 1.8B-parameter vision encoder for native image understanding. Engineered for high-frequency production workloads, it activates approximately 11B parameters per token and delivers a throughput of up to 400 tokens per second. Step 3.7 Flash supports a 256k context window and offers three selectable reasoning levels (low, medium, and high) so developers can easily balance speed, cost, and cognitive depth.
We built Step 3.7 Flash for developers who need to scale agentic workflows that combine perception, search, and reasoning. It is designed to handle intensive tasks such as parsing massive financial reports in one pass, running multi-step search loops with cross-source verification, or operating concurrent coding agents in high-throughput pipelines.

## 2. Capabilities & Performance

[!NOTE]
The specific benchmark values are based on the [page](https://huggingface.co/stepfun-ai/Step-3.7-Flash).

### Multimodal Perception and Verification

The model delivers top-tier visual intelligence, securing first place on SimpleVQA (Search) with a 79.2 and achieving frontier parity on V\* (Python) at 95.3. These metrics reflect strong visual grounding and retrieval-augmented reasoning beyond basic image description. The model accurately processes dense visual interfaces, such as UI wireframes, application GUIs, and data charts, to map them into structured code. When it encounters an incomplete visual asset, it can independently identify missing data and execute lookups to verify context before returning a factually verified conclusion.

### Workflow Integrity and Tool Orchestration

Execution reliability is critical for autonomous agents. Step 3.7 Flash leads the ClawEval-1.1 benchmark with a score of 67.1, which significantly outperforms the next closest competitor at 59.8. This performance demonstrates high resistance to adversarial traps and strict adherence to system policies during multi-turn orchestration. Backed by scores of 49.5 on Toolathlon and 48.1 on HLE w. Tool, this profile ensures high trajectory integrity. Step 3.7 Flash reliably interacts with external APIs and executes long-horizon workflows without drifting from instructions or violating system constraints.

### Code Engineering and Professional Baselines

Step 3.7 Flash is built for live engineering tasks and secured a definitive second-place finish on SWE-Bench PRO with a score of 56.3. It can independently trace multi-file repositories, isolate bugs from raw issue reports, and generate functional patches that pass automated unit tests. While evaluations like Terminal-Bench 2.1 (59.5) and GPDVal (45.8) show clear areas for future optimization compared to the absolute peak of the cohort, they establish a dependable baseline for system interactions and structured professional deliverables.

## 3. Availability, Deployment, and Ecosystem

- Availability: Step 3.7 Flash is available through StepFun Open Platform at platform.stepfun.ai and platform.stepfun.com, as well as partner platforms including OpenRouter and NVIDIA NIM.
- Deployment: Step 3.7 Flash supports flexible deployment across cloud, data center, and local environments. For large-scale production and enterprise use cases, Step 3.7 Flash can be deployed on modern data center infrastructure. For local and workstation scenarios, it can also run on high-memory devices such as NVIDIA DGX Station, AMD Ryzen AI Max+ 395-based systems, and Mac Studio / Macbook Pro devices with at least 128GB unified memory.
- Ecosystem: Step 3.7 Flash is supported across popular open-source infrastructure for both inference and model development. For inference and serving, developers can use vLLM, SGLang, Hugging Face Transformers, and llama.cpp. For model development workflows, StepFun model support has landed in the NVIDIA Megatron ecosystem, including Megatron Core and Megatron Bridge.

## 4. Examples

You can get started with Step 3.7 Flash in minutes using StepFun's API or via other inference providers.

### 4.1 Chat Example

The base URL for the global StepFun platform is https://api.stepfun.ai/.

```python
from openai import OpenAI

client = OpenAI(api_key="STEP_API_KEY", base_url="https://api.stepfun.com/v1")

completion = client.chat.completions.create(
  model="step-3.7-flash",
  messages=[
    {
      "role": "system",
      "content":"You are an AI assistant provided by StepFun. You are good at Chinese, English, and many other languages, and you can see, think, and act to help users get things done.",
    },
    {
      "role": "user",
      "content": "Introduce StepFun's artificial intelligence capabilities."
    },
  ]
)
```

## 5. Local Deployment

We recommend using the latest nightly build of vLLM.

### 5.1. Install vLLM.

#### via Docker

```
docker pull vllm/vllm-openai:nightly
```

#### or via pip (nightly wheels)

```
pip install -U vllm --pre \
 --index-url https://pypi.org/simple \
 --extra-index-url https://wheels.vllm.ai/nightly 2. Launch the server.
```

## 5.2 Deployment

- For FP8 model

```
  vllm serve <MODEL_PATH_OR_HF_ID> \
   --served-model-name step3p7-flash \
   --tensor-parallel-size 8 \
   --enable-expert-parallel \
   --disable-cascade-attn \
   --reasoning-parser step3p5 \
   --enable-auto-tool-choice \
   --tool-call-parser step3p5 \
   --speculative_config '{"method": "mtp", "num_speculative_tokens": 3}' \
   --trust-remote-code
```

- For BF16 model

```
  vllm serve <MODEL_PATH_OR_HF_ID> \
   --served-model-name step3p7-flash-fp8 \
   --tensor-parallel-size 8 \
   --enable-expert-parallel \
   --disable-cascade-attn \
   --reasoning-parser step3p5 \
   --enab - -auto-tool-choice \
   --tool-call-parser step3p5 \
   --speculative_config '{"method": "mtp", "num_speculative_tokens": 3}' \
   --trust-remote-code
```

- For NVFP4 model
  Compared to standard precisions, running the FP4 quantized version requires modelopt activation and FP8 KV Cache alignment.

```
  python3 -m vllm.entrypoints.openai.api_server \
   --host 0.0.0.0 \
   --port ${PORT} \
   --model stepfun-ai/Step-3.7-Flash-NVFP4 \
   --served-model-name step3p7 \
   --tensor-parallel-size 4 \
   --gpu-memory-utilization 0.9 \
   --enable-expert-parallel \
   --trust-remote-code \
   --quantization modelopt \
   --kv-cache-dtype fp8 \
   --max-model-len 8192 \
   --reasoning-parser step3p5 \
   --enable-auto-tool-choice \
   --tool-call-parser step3p5 \
   --async-scheduling
```

## 5. Local Deployment

### 5.1. Install SGLang.

#### via Docker

docker pull lmsysorg/sglang:latest

#### or from source (pip)

```
pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git" 2. Launch the server.
```

### 5.2 Deployment

For Blackwell GPUs, --mm-attention-backend fa4 may be used.

- For bf16 model

```
   sglang serve --model-path stepfun-ai/Step-3.7-Flash \
    --tp 8 \
    --reasoning-parser step3p5 \
    --tool-call-parser step3p5 \
    --enable-multimodal \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --enable-multi-layer-eagle \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
```

- For fp8 model

```
   sglang serve --model-path stepfun-ai/Step-3.7-Flash-fp8 \
    --tp 8 \
    --ep 4 \
    --reasoning-parser step3p5 \
    --tool-call-parser step3p5 \
    --enable-multimodal \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --enable-multi-layer-eagle \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
```

## 6. Using Step 3.7 Flash on Agent Platforms

You can use Step 3.7 Flash on Agent platforms such as Hermes Agent, Lemonade, OpenClaw, Kilo Code, and more.

## 7. Getting in Touch

As we work to shape the future of AGI by expanding broad model capabilities, we want to ensure we are solving the right problems. We invite you to be part of this continuous feedback loop—your insights directly influence our priorities.

- Join the Conversation: Our Discord community is the primary hub for brainstorming future architectures, proposing capabilities, and getting early access updates 🚀
- Report Friction: Encountering limitations? You can open an issue or start a discussion on GitHub / HuggingFace or flag it directly in our Discord support channels.
  📄 License
  This project is open-sourced under the Apache 2.0 License.
