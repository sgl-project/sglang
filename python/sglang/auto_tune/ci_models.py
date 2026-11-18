"""
Popular models for auto-tune CI:

20 models across different architectures:
- MoE, dense, vision-language
- Different sizes: 7B, 13B, 30B, 70B, 236B, 671B
"""

POPULAR_MODELS = [
    # MoE models
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "DeepSeek-V3",
    "DeepSeek-R1",
    "Mixtral-8x7B-Instruct-v0.1",
    "Mixtral-8x22B-Instruct-v0.1",
    "Dbrx-Instruct",

    # Dense models
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Mistral-7B-Instruct-v0.2",
    "Gemma-2-27B-it",

    # Vision-language models
    "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2-VL-72B-Instruct",
    "LLaVA-1.6-vicuna-7B",
    "InternVL2-8B",

    # Large models
    "DeepSeek-V2-Lite",
]

CI_RUNNERS = [
    "h100",
    "h200",
    "b200",
    "gb200",
    "h20",
]

# TP size per runner (based on GPU count)
RUNNER_TP = {
    "h100": 8,
    "h200": 8,
    "b200": 8,
    "gb200": 8,
    "h20": 8,
}