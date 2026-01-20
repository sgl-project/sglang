# Support Models on Ascend NPU

This section describes the models supported on the Ascend NPU, including Large Language Models, Multimodal Language
Models, Embedding Models, Reward Models and Rerank Models. Mainstream DeepSeek/Qwen/GLM series are included.
You are welcome to enable various models based on your business requirements.

## Large Language Models

| Models                                     | Model Family                   |               A2 Supported               |               A3 Supported               |
|--------------------------------------------|--------------------------------|:----------------------------------------:|:----------------------------------------:|
| DeepSeek V3/V3.1                           | DeepSeek                       | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| vllm-ascend/DeepSeek-V3.2-Exp-W8A8         | DeepSeek                       | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| vllm-ascend/DeepSeek-R1-0528-W8A8          | DeepSeek                       | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| vllm-ascend/DeepSeek-V2-Lite-W8A8          | DeepSeek                       | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Qwen/Qwen3-30B-A3B-Instruct-2507           | Qwen                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Qwen/Qwen3-32B                             | Qwen                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Qwen/Qwen3-0.6B                            | Qwen                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| vllm-ascend/Qwen3-235B-A22B-W8A8           | Qwen                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Qwen/Qwen3-Next-80B-A3B-Instruct           | Qwen                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot | Qwen                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Qwen/Qwen2.5-7B-Instruct                   | Qwen                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| vllm-ascend/QWQ-32B-W8A8                   | Qwen                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| meta-llama/Llama-4-Scout-17B-16E-Instruct  | Llama                          | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| AI-ModelScope/Llama-3.1-8B-Instruct        | Llama                          | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| LLM-Research/Llama-3.2-1B-Instruct         | Llama                          | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| mistralai/Mistral-7B-Instruct-v0.2         | Mistral                        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| google/gemma-3-4b-it                       | Gemma                          | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| microsoft/Phi-4-multimodal-instruct        | Phi                            | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| allenai/OLMoE-1B-7B-0924                   | OLMoE                          | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| stabilityai/stablelm-2-1_6b                | StableLM                       | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| CohereForAI/c4ai-command-r-v01             | Command-R                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| huihui-ai/grok-2                           | Grok                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| ZhipuAI/chatglm2-6b                        | ChatGLM                        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Shanghai_AI_Laboratory/internlm2-7b        | InternLM 2                     | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct       | ExaONE 3                       | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| xverse/XVERSE-MoE-A36B                     | XVERSE                         | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| HuggingFaceTB/SmolLM-1.7B                  | SmolLM                         | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| ZhipuAI/glm-4-9b-chat                      | GLM-4                          | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| XiaomiMiMo/MiMo-7B-RL                      | MiMo                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| arcee-ai/AFM-4.5B-Base                     | Arcee AFM-4.5B                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Howeee/persimmon-8b-chat                   | Persimmon                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| inclusionAI/Ling-lite                      | Ling                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| ibm-granite/granite-3.1-8b-instruct        | Granite                        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| ibm-granite/granite-3.0-3b-a800m-instruct  | Granite MoE                    | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| databricks/dbrx-instruct                   | DBRX (Databricks)              | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| baichuan-inc/Baichuan2-13B-Chat            | Baichuan 2 (7B, 13B)           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| baidu/ERNIE-4.5-21B-A3B-PT                 | ERNIE-4.5 (4.5, 4.5MoE series) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| openbmb/MiniCPM3-4B                        | MiniCPM (v3, 4B)               | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| openai/gpt-oss-120b                        | GPTOSS                         | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## Multimodal Language Models

| Models                                        | Model Family (Variants)   |               A2 Supported               |               A3 Supported               |
|-----------------------------------------------|---------------------------|:----------------------------------------:|:----------------------------------------:|
| Qwen/Qwen2.5-VL-3B-Instruct                   | Qwen-VL                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Qwen/Qwen2.5-VL-72B-Instruct                  | Qwen-VL                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Qwen/Qwen3-VL-30B-A3B-Instruct                | Qwen-VL                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Qwen/Qwen3-VL-8B-Instruct                     | Qwen-VL                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Qwen/Qwen3-VL-4B-Instruct                     | Qwen-VL                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Qwen/Qwen3-VL-235B-A22B-Instruct              | Qwen-VL                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| deepseek-ai/deepseek-vl2                      | DeepSeek-VL2              | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| deepseek-ai/Janus-Pro-7B                      | Janus-Pro (1B, 7B)        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| openbmb/MiniCPM-V-2_6                         | MiniCPM-V / MiniCPM-o     | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| google/gemma-3-4b-it                          | Gemma 3 (Multimodal)      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| mistralai/Mistral-Small-3.1-24B-Instruct-2503 | Mistral-Small-3.1-24B     | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| microsoft/Phi-4-multimodal-instruct           | Phi-4-multimodal-instruct | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| XiaomiMiMo/MiMo-VL-7B-RL                      | MiMo-VL (7B)              | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| AI-ModelScope/llava-v1.6-34b                  | LLaVA (v1.5 & v1.6)       | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| lmms-lab/llava-next-72b                       | LLaVA-NeXT (8B, 72B)      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| lmms-lab/llava-onevision-qwen2-7b-ov          | LLaVA-OneVision           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Kimi/Kimi-VL-A3B-Instruct                     | Kimi-VL (A3B)             | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| ZhipuAI/GLM-4.5V                              | GLM-4.5V (106B)           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| meta-llama/Llama-3.2-11B-Vision-Instruct      | Llama 3.2 Vision (11B)    | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## Embedding Models

| Models                                    | Model Family             |               A2 Supported               |               A3 Supported               |
|-------------------------------------------|--------------------------|:----------------------------------------:|:----------------------------------------:|
| 	intfloat/e5-mistral-7b-instruct          | E5 (Llama/Mistral based) | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| 	iic/gte_Qwen2-1.5B-instruct              | GTE-Qwen2                | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| 	Qwen/Qwen3-Embedding-8B                  | Qwen3-Embedding          | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| 	Alibaba-NLP/gme-Qwen2-VL-2B-Instruct     | GME (Multimodal)         | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| 	AI-ModelScope/clip-vit-large-patch14-336 | CLIP                     | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| 	BAAI/bge-large-en-v1.5                   | BGE                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## Reward Models

| Models                                      | Model Family              | A2 Supported                             |               A3 Supported               |
|---------------------------------------------|---------------------------|------------------------------------------|:----------------------------------------:|
| 	Skywork/Skywork-Reward-Llama-3.1-8B-v0.2   | Llama3.1 Reward           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| 	Shanghai_AI_Laboratory/internlm2-7b-reward | InternLM 2 Reward         | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| 	Qwen/Qwen2.5-Math-RM-72B                   | Qwen2.5 Reward - Math     | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| 	jason9693/Qwen2.5-1.5B-apeach              | Qwen2.5 Reward - Sequence | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| 	Skywork/Skywork-Reward-Gemma-2-27B-v0.2    | Gemma 2-27B Reward        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |

## Rerank Models

| Models                  | Model Family |               A2 Supported               |               A3 Supported               |
|-------------------------|--------------|:----------------------------------------:|:----------------------------------------:|
| BAAI/bge-reranker-v2-m3 | BGE-Reranker | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
