# Large Language Models

These models accept text input and produce text output (e.g., chat completions). They are primarily large language models (LLMs), some with mixture-of-experts (MoE) architectures for scaling.

## Example launch Command

```shell
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \  # example HF/local path
  --host 0.0.0.0 \
  --port 30000 \
```

## Supporting Matrixs


| Model Family (Variants)             | Example HuggingFace Identifier                     | Description                                                                            |
|-------------------------------------|--------------------------------------------------|----------------------------------------------------------------------------------------|
| **DeepSeek** (v1, v2, v3/R1)        | `deepseek-ai/DeepSeek-R1`                        | Series of advanced reasoning-optimized models (including a 671B MoE) trained with reinforcement learning; top performance on complex reasoning, math, and code tasks. [SGLang provides Deepseek v3/R1 model-specific optimizations](https://docs.sglang.ai/references/deepseek) and [Reasoning Parser](https://docs.sglang.ai/backend/separate_reasoning)|
| **Qwen** (3, 3MoE, 2.5, 2 series)       | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-30B-A3B`       | Alibaba’s latest Qwen3 series for complex reasoning, language understanding, and generation tasks; Support for MoE variants along with previous generation 2.5, 2, etc. [SGLang provides Qwen3 specific reasoning parser](https://docs.sglang.ai/backend/separate_reasoning)|
| **Llama** (2, 3.x, 4 series)        | `meta-llama/Llama-4-Scout-17B-16E-Instruct`       | Meta’s open LLM series, spanning 7B to 400B parameters (Llama 2, 3, and new Llama 4) with well-recognized performance. [SGLang provides Llama-4 model-specific optimizations](https://docs.sglang.ai/references/llama4)  |
| **Mistral** (Mixtral, NeMo, Small3) | `mistralai/Mistral-7B-Instruct-v0.2`             | Open 7B LLM by Mistral AI with strong performance; extended into MoE (“Mixtral”) and NeMo Megatron variants for larger scale. |
| **Gemma** (v1, v2, v3)              | `google/gemma-3-1b-it`                            | Google’s family of efficient multilingual models (1B–27B); Gemma 3 offers a 128K context window, and its larger (4B+) variants support vision input. |
| **Phi** (Phi-3, Phi-4 series)      | `microsoft/Phi-4-multimodal-instruct`            | Microsoft’s Phi family of small models (1.3B–5.6B); Phi-4-mini is a high-accuracy text model and Phi-4-multimodal (5.6B) processes text, images, and speech in one compact model. |
| **MiniCPM** (v3, 4B)               | `openbmb/MiniCPM3-4B`                            | OpenBMB’s series of compact LLMs for edge devices; MiniCPM 3 (4B) achieves GPT-3.5-level results in text tasks. |
| **OLMoE** (Open MoE)               | `allenai/OLMoE-1B-7B-0924`                       | Allen AI’s open Mixture-of-Experts model (7B total, 1B active parameters) delivering state-of-the-art results with sparse expert activation. |
| **StableLM** (3B, 7B)               | `stabilityai/stablelm-tuned-alpha-7b`            | StabilityAI’s early open-source LLM (3B & 7B) for general text generation; a demonstration model with basic instruction-following ability. |
| **Command-R** (Cohere)              | `CohereForAI/c4ai-command-r-v01`                 | Cohere’s open conversational LLM (Command series) optimized for long context, retrieval-augmented generation, and tool use. |
| **DBRX** (Databricks)              | `databricks/dbrx-instruct`                       | Databricks’ 132B-parameter MoE model (36B active) trained on 12T tokens; competes with GPT-3.5 quality as a fully open foundation model. |
| **Grok** (xAI)                     | `xai-org/grok-1`                                | xAI’s grok-1 model known for vast size(314B parameters) and high quality; integrated in SGLang for high-performance inference. |
| **ChatGLM** (GLM-130B family)       | `THUDM/chatglm2-6b`                              | Zhipu AI’s bilingual chat model (6B) excelling at Chinese-English dialogue; fine-tuned for conversational quality and alignment. |
| **InternLM 2** (7B, 20B)           | `internlm/internlm2-7b`                          | Next-gen InternLM (7B and 20B) from SenseTime, offering strong reasoning and ultra-long context support (up to 200K tokens). |
| **ExaONE 3** (Korean-English)      | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`           | LG AI Research’s Korean-English model (7.8B) trained on 8T tokens; provides high-quality bilingual understanding and generation. |
| **Baichuan 2** (7B, 13B)           | `baichuan-inc/Baichuan2-13B-Chat`                | BaichuanAI’s second-generation Chinese-English LLM (7B/13B) with improved performance and an open commercial license. |
| **XVERSE** (MoE)                   | `xverse/XVERSE-MoE-A36B`                         | Yuanxiang’s open MoE LLM (XVERSE-MoE-A36B: 255B total, 36B active) supporting ~40 languages; delivers 100B+ dense-level performance via expert routing. |
| **SmolLM** (135M–1.7B)            | `HuggingFaceTB/SmolLM-1.7B`                      | Hugging Face’s ultra-small LLM series (135M–1.7B params) offering surprisingly strong results, enabling advanced AI on mobile/edge devices. |
| **GLM-4** (Multilingual 9B)        | `ZhipuAI/glm-4-9b-chat`                          | Zhipu’s GLM-4 series (up to 9B parameters) – open multilingual models with support for 1M-token context and even a 5.6B multimodal variant (Phi-4V). |
