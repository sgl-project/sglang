# Support Models on Ascend NPU

This section describes the models supported on the Ascend NPU, including Large Language Models, Multimodal Language
Models, Embedding Models, and Rerank Models. Mainstream DeepSeek/Qwen/GLM series are included. You are welcome to enable
various models based on your business requirements.

## Large Language Models

| Model Family                   | Recommend Models                                                                                                         |               A2 Supported               |               A3 Supported               |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------|:----------------------------------------:|:----------------------------------------:|
| DeepSeek                       | DeepSeek V1, V2, V3(V3.1,V3.2), R1                                                                                       | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Qwen                           | Qwen 3, Qwen 3Moe                                                                                                        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Llama                          | meta-llama/Llama-4-Scout-17B-16E-Instruct,<br>AI-ModelScope/Llama-3.1-8B-Instruct,<br>LLM-Research/Llama-3.2-1B-Instruct |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| Mistral                        | mistralai/Mistral-7B-Instruct-v0.2                                                                                       | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Gemma                          | google/gemma-3-4b-it                                                                                                     | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Phi                            | microsoft/Phi-4-multimodal-instruct                                                                                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| OLMoE                          | allenai/OLMoE-1B-7B-0924                                                                                                 |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| StableLM                       | stabilityai/stablelm-2-1_6b                                                                                              |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| Command-R                      | CohereForAI/c4ai-command-r-v01                                                                                           |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| Grok                           | huihui-ai/grok-2                                                                                                         |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| ChatGLM                        | ZhipuAI/chatglm2-6b                                                                                                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| InternLM 2                     | Shanghai_AI_Laboratory/internlm2-7b                                                                                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| ExaONE 3                       | LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct                                                                                     | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| XVERSE                         | xverse/XVERSE-MoE-A36B                                                                                                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| SmolLM                         | HuggingFaceTB/SmolLM-1.7B                                                                                                | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| GLM-4                          | ZhipuAI/glm-4-9b-chat                                                                                                    |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| MiMo                           | XiaomiMiMo/MiMo-7B-RL                                                                                                    | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Arcee AFM-4.5B                 | arcee-ai/AFM-4.5B-Base                                                                                                   | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Persimmon                      | Howeee/persimmon-8b-chat                                                                                                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Ling                           | inclusionAI/Ling-lite                                                                                                    | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Granite                        | ibm-granite/granite-3.1-8b-instruct                                                                                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Granite Moe                    | ibm-granite/granite-3.0-3b-a800m-instruct                                                                                | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| DBRX (Databricks)              | databricks/dbrx-instruct                                                                                                 |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| Baichuan 2 (7B, 13B)           | baichuan-inc/Baichuan2-13B-Chat                                                                                          |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| ERNIE-4.5 (4.5, 4.5MoE series) | baidu/ERNIE-4.5-21B-A3B-PT                                                                                               |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| MiniCPM (v3, 4B)               | openbmb/MiniCPM3-4B                                                                                                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| GPTOSS                         | openai/gpt-oss-120b                                                                                                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |

## Multimodal Language Models

| Model Family                   | Recommend Models                              |               A2 Supported               |               A3 Supported               |
|--------------------------------|-----------------------------------------------|:----------------------------------------:|:----------------------------------------:|
| Qwen-VL (Qwen2 series)         | Qwen/Qwen3-VL-235B-A22B-Instruct              |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| DeepSeek-VL2                   | deepseek-ai/deepseek-vl2                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| Janus-Pro (1B, 7B)             | deepseek-ai/Janus-Pro-7B                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| MiniCPM-V / MiniCPM-o          | openbmb/MiniCPM-V-2_6                         |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| Gemma 3 (Multimodal)           | google/gemma-3-4b-it                          | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Mistral-Small-3.1-24B          | mistralai/Mistral-Small-3.1-24B-Instruct-2503 |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| Phi-4-multimodal-instruct      | microsoft/Phi-4-multimodal-instruct           |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| MiMo-VL (7B)                   | XiaomiMiMo/MiMo-VL-7B-RL                      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| LLaVA (v1.5 & v1.6)            | AI-ModelScope/llava-v1.6-34b                  | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| LLaVA-NeXT (8B, 72B)           | lmms-lab/llava-next-72b                       | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| LLaVA-OneVision                | lmms-lab/llava-onevision-qwen2-7b-ov          |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| Kimi-VL (A3B)                  | Kimi/Kimi-VL-A3B-Instruct                     |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| GLM-4.5V (106B) / GLM-4.1V(9B) | ZhipuAI/GLM-4.5V                              |  **<span style="color: red;">×</span>**  | **<span style="color: green;">√</span>** |
| Llama 3.2 Vision (11B)         | meta-llama/Llama-3.2-11B-Vision-Instruct      |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |

## Embedding Models

| Model Family             | Recommend Models                         |              A2 Supported              |               A3 Supported               |
|--------------------------|------------------------------------------|:--------------------------------------:|:----------------------------------------:|
| E5 (Llama/Mistral based) | intfloat/e5-mistral-7b-instruct          | **<span style="color: red;">×</span>** |  **<span style="color: red;">×</span>**  |
| GTE-Qwen2                | iic/gte_Qwen2-1.5B-instruct              | **<span style="color: red;">×</span>** |  **<span style="color: red;">×</span>**  |
| Qwen3-Embedding          | Qwen/Qwen3-Embedding-8B                  | **<span style="color: red;">×</span>** |  **<span style="color: red;">×</span>**  |
| GME (Multimodal)         | Alibaba-NLP/gme-Qwen2-VL-2B-Instruct     | **<span style="color: red;">×</span>** |  **<span style="color: red;">×</span>**  |
| CLIP                     | AI-ModelScope/clip-vit-large-patch14-336 | **<span style="color: red;">×</span>** | **<span style="color: green;">√</span>** |
| BGE                      | BAAI/bge-large-en-v1.5                   | **<span style="color: red;">×</span>** |  **<span style="color: red;">×</span>**  |

## Reward Models

| Model Family              | Recommend Models                           |              A2 Supported              |               A3 Supported               |
|---------------------------|--------------------------------------------|:--------------------------------------:|:----------------------------------------:|
| Llama3.1 Reward           | Skywork/Skywork-Reward-Llama-3.1-8B-v0.2   | **<span style="color: red;">×</span>** | **<span style="color: green;">√</span>** |
| InternLM 2 Reward         | Shanghai_AI_Laboratory/internlm2-7b-reward | **<span style="color: red;">×</span>** | **<span style="color: green;">√</span>** |
| Qwen2.5 Reward - Math     | Qwen/Qwen2.5-Math-RM-72B                   | **<span style="color: red;">×</span>** | **<span style="color: green;">√</span>** |
| Qwen2.5 Reward - Sequence | jason9693/Qwen2.5-1.5B-apeach              | **<span style="color: red;">×</span>** | **<span style="color: green;">√</span>** |
| Gemma 2-27B Reward        | Skywork/Skywork-Reward-Gemma-2-27B-v0.2    | **<span style="color: red;">×</span>** |  **<span style="color: red;">×</span>**  |

## Rerank Models

| Model Family | Recommend Models        |              A2 Supported              |              A3 Supported              |
|--------------|-------------------------|:--------------------------------------:|:--------------------------------------:|
| BGE-Reranker | BAAI/bge-reranker-v2-m3 | **<span style="color: red;">×</span>** | **<span style="color: red;">×</span>** |
