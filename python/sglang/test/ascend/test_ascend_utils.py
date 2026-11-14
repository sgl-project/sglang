"""Common utilities for testing and benchmarking on NPU.

Keep the weight-path constants sorted by variable name within each section.
"""

import asyncio
import copy
import json
import logging
import os
import random
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Awaitable, Callable, List, NamedTuple, Optional

import requests

from sglang.benchmark.serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    auto_config_device,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

logger = logging.getLogger(__name__)

STDERR_FILENAME = "/tmp/stderr.txt"
STDOUT_FILENAME = "/tmp/stdout.txt"

# npu_memcache L3 backend MetaService ports.
META_SERVICE_PORT = 5007
CONFIG_STORE_PORT = 6007
META_METRICS_PORT = 8007

# npu_memcache L3 backend MetaService readiness.
META_SERVICE_READY_TIMEOUT = 300
META_SERVICE_READY_POLL_INTERVAL = 5

# Timeout for resolving the cluster-wide MetaService host from the harness
# ConfigMap ({pod_name: pod_ip}, written by run_npu_e2e_test.py).
META_SERVICE_MASTER_TIMEOUT = 300

# Model weights storage directory
MODEL_WEIGHTS_DIR = "/root/.cache/modelscope/hub/models/"
HF_MODEL_WEIGHTS_DIR = "/root/.cache/huggingface/hub/"
IMAGES_DIR = "/root/.cache/modelscope/hub/datasets/images/"
VIDEO_DIR = "/root/.cache/modelscope/hub/datasets/video/"

# LLM model weights path
AFM_4_5B_BASE_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "arcee-ai/AFM-4.5B-Base")
BAICHUAN2_13B_CHAT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "baichuan-inc/Baichuan2-13B-Chat"
)
C4AI_COMMAND_R_V01_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "CohereForAI/c4ai-command-r-v01"
)
C4AI_COMMAND_R_V01_CHAT_TEMPLATE_PATH = "/__w/sglang/sglang/test/registered/npu/llm_models/tool_chat_template_c4ai_command_r_v01.jinja"
CHATGLM2_6B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "ZhipuAI/chatglm2-6b")
DBRX_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "AI-ModelScope/dbrx-instruct"
)
DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "vllm-ascend/DeepSeek-R1-0528-W8A8"
)
DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "DeepSeek-V3.2-Exp-W8A8"
)
DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "vllm-ascend/DeepSeek-V3.2-W8A8"
)
DEEPSEEK_V4_FLASH_BF16_MODEL_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "deepseek-ai/DeepSeek-V4-Flash-BF16"
)
DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
)
DEEPSEEK_CODER_1_3_B_BASE_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "deepseek-ai/deepseek-coder-1.3b-base"
)
DOTS_OCR_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "rednote-hilab/dots.ocr")
ECO_TECH_QWEN3_32B_W4A4_LAOS_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Eco-Tech/Qwen3-32B-w4a4-LAOS"
)
ECO_TECH_QWEN3_30B_A3B_W4A4_LAOS_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Eco-Tech/Qwen3-30B-A3B-w4a4-LAOS"
)
ERNIE_4_5_21B_A3B_PT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "baidu/ERNIE-4.5-21B-A3B-PT"
)
EXAONE_3_5_7_8B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
)
GEMMA_3_4B_IT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "google/gemma-3-4b-it")
GEMMA_4_E2B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "google/gemma-4-E2B-it")
GEMMA_4_E4B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "google/gemma-4-E4B-it")
GEMMA_4_26B_A4B_IT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "google/gemma-4-26B-A4B-it"
)
GEMMA_4_31B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "google/gemma-4-31B-it")
GLM_4_9B_CHAT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "ZhipuAI/glm-4-9b-chat")
GLM_5_1_W4A8_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Eco-Tech/GLM-5.1-w4a8")
GLM_5_2_BF16_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "ZhipuAI/GLM-5.2")
GPT_OSS_120B_BF16_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "eigen-ai-labs/gpt-oss-120b-bf16"
)
GRANITE_3_0_3B_A800M_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "ibm-granite/granite-3.0-3b-a800m-instruct"
)
GRANITE_3_1_8B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "ibm-granite/granite-3.1-8b-instruct"
)
GROK_2_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "huihui-ai/grok-2")
GROK_2_WEIGHTS_TOKENIZER_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "huihui-ai/grok-2/tokenizer.tok.json"
)
INTERNLM2_7B_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Shanghai_AI_Laboratory/internlm2-7b"
)
KIMI_K2_THINKING_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Kimi/Kimi-K2-Thinking")
KIMI_K2_5_W4A8_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Eco-Tech/Kimi-K2.5-w4a8")
KIMI_K2_5_EAGLE3_MODEL_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "lightseekorg/kimi-k2.5-eagle3"
)
KIMI_K3_W4A8_INT_MOE_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Kimi/Kimi-K3-w4a8-int-moe"
)
LING_LITE_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "inclusionAI/Ling-lite")
LLAMA_2_7B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "LLM-Research/Llama-2-7B")
LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "AI-ModelScope/Llama-3.1-8B-Instruct"
)
LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "codelion/Llama-3.2-1B-Instruct-tool-calling-lora"
)
LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "suayptalha/FastLlama-3.2-LoRA"
)
LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "LLM-Research/Llama-3.2-1B-Instruct"
)
LLAMA_3_2_1B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "LLM-Research/Llama-3.2-1B")
LLAMA_3_8B_EAGLE_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "lmsys/sglang-EAGLE-LLaMA3-Instruct-8B"
)
LLAMA_3_8B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "LLM-Research/Meta-Llama-3-8B-Instruct"
)

LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "meta-llama/Llama-4-Scout-17B-16E-Instruct"
)
LLaDA2_0_MINI_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "inclusionAI/LLaDA2.0-mini"
)
META_LLAMA_3_1_8B_INSTRUCT = os.path.join(
    MODEL_WEIGHTS_DIR, "LLM-Research/Meta-Llama-3.1-8B-Instruct"
)
MIMO_7B_RL_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "XiaomiMiMo/MiMo-7B-RL")
MIMO_V2_FLASH_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "XiaomiMiMo/MiMo-V2-Flash")
MIMO_V2_5_PRO_FP4_DFLASH_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash"
)
MIMO_V2_5_PRO_FP4_DFLASH_DRAFT_WEIGHTS_PATH = os.path.join(
    MIMO_V2_5_PRO_FP4_DFLASH_WEIGHTS_PATH, "dflash"
)
MIMO_V2_5_W8A8_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "solinliu/MiMo-V2.5-W8A8")
MINICPM3_4B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "OpenBMB/MiniCPM3-4B")
MISTRAL_7B_INSTRUCT_V0_2_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "mistralai/Mistral-7B-Instruct-v0.2"
)
OLMO_2_1124_7B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "allenai/OLMo-2-1124-7B-Instruct"
)
OLMOE_1B_7B_0924_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "allenai/OLMoE-1B-7B-0924"
)
PERSIMMON_8B_CHAT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Howeee/persimmon-8b-chat"
)
PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "microsoft/Phi-4-multimodal-instruct"
)
QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen2.5-7B-Instruct"
)
QWEN3_0_6B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3-0.6B")
QWEN3_5_27B_MODEL_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3.5-27B")
QWEN3_1_7B_GPTQ_INT8_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-1.7B-GPTQ-Int8"
)
QWEN3_235B_A22B_W8A8_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "vllm-ascend/Qwen3-235B-A22B-W8A8"
)
QWEN3_235B_A22B_EAGLE_MODEL_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-235B-A22B-Eagle3"
)
QWEN3_30B_A3B_GPTQ_2507_INT4_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
)
QWEN3_30B_A3B_GGUF_Q4_K_M_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
)
QWEN3_30B_A3B_INSTRUCT_2507_INT4_AUTOROUND_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Intel/Qwen3-30B-A3B-Instruct-2507-int4-AutoRound"
)
QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-30B-A3B-Instruct-2507"
)
QWEN3_4B_GGUF_Q4_K_M_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
)
QWEN3_8B_INT4_AUTOROUND_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Intel/Qwen3-8B-int4-AutoRound"
)
QWEN3_8B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3-8B")
QWEN3_8B_EAGLE3_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3-8B_eagle3")
QWEN3_8B_DECRYPTED_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "YZY/Qwen3-8B")
QWEN3_8B_EAGLE3_DECRYPTED_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "YZY/Qwen3-8B_eagle3"
)
QWEN3_32B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3-32B")
QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot"
)
QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-Next-80B-A3B-Instruct"
)
QWEN3_32B_EAGLE3_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Zjcxy-SmartAI/Eagle3-Qwen3-32B-zh"
)
QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "aleoyang/Qwen3-32B-w8a8-MindIE"
)
QWQ_32B_W8A8_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "vllm-ascend/QWQ-32B-W8A8")
SMOLLM_1_7B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "HuggingFaceTB/SmolLM-1.7B")
SOLAR_10_7B_INSTRUCT_V1_0_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "upstage/SOLAR-10.7B-Instruct-v1.0"
)
STABLELM_2_1_6B_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "stabilityai/stablelm-2-1_6b"
)
STARCODER2_7B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "bigcode/starcoder2-7b")
TRINITY_MINI_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "arcee-ai/Trinity-Mini")
XVERSE_MOE_A36B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "xverse/XVERSE-MoE-A36B")
MINIMAX_M2_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "cyankiwi/MiniMax-M2-BF16")
MINIMAX_M2_5_W8A8_MODEL_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Eco-Tech/MiniMax-M2.5-w8a8-QuaRot"
)
MINIMAX_M2_5_EAGLE3_MODEL_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "sgl-npu/MiniMax-M2.5-eagel-model-0318"
)
EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "sglang-EAGLE3-LLaMA3.1-Instruct-8B"
)

# VLM model weights path
DEEPSEEK_VL2_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "deepseek-ai/deepseek-vl2")
GLM_4_5V_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "ZhipuAI/GLM-4.5V")
JANUS_PRO_1B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "deepseek-ai/Janus-Pro-1B")
JANUS_PRO_7B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "deepseek-ai/Janus-Pro-7B")
KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "moonshotai/Kimi-VL-A3B-Instruct"
)
LLAMA_3_2_11B_VISION_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "LLM-Research/Llama-3.2-11B-Vision-Instruct"
)
LLAVA_NEXT_72B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "lmms-lab/llava-next-72b")
LLAVA_ONEVISION_QWEN2_7B_OV_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "lmms-lab/llava-onevision-qwen2-7b-ov"
)
LLAVA_V1_6_34B_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "AI-ModelScope/llava-v1.6-34b"
)
LLAVA_V1_6_34B_TOKENIZER_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "AI-ModelScope/llava-v1.6-34b/llava-1.6v-34b-tokenizer"
)
MIMO_VL_7B_RL_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "XiaomiMiMo/MiMo-VL-7B-RL")
MINICPM_O_2_6_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "openbmb/MiniCPM-o-2_6")
MINICPM_V_2_6_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "openbmb/MiniCPM-V-2_6")
MISTRAL_SMALL_3_1_24B_INSTRUCT_2503_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
)
QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen2.5-VL-3B-Instruct"
)
QWEN2_5_VL_72B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen2.5-VL-72B-Instruct"
)
QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-VL-4B-Instruct"
)
QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-VL-8B-Instruct"
)
QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-VL-30B-A3B-Instruct"
)
QWEN3_VL_235B_A22B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-VL-235B-A22B-Instruct"
)
QWEN2_0_5B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen2-0.5B-Instruct"
)

QWEN3_30B_A3B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3-30B-A3B")
QWEN3_30B_A3B_W8A8_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-30B-A3B-w8a8"
)
DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "vllm-ascend/DeepSeek-V2-Lite-W8A8"
)

DEEPSEEK_R1_DISTILL_QWEN_7B_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)
DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "DeepSeek-R1-0528-w4a8-per-channel"
)
DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "vllm-ascend/DeepSeek-R1-0528-W8A8"
)
QWEN3_5_397B_W4A8_MODEL_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Eco-Tech/Qwen3.5-397B-A17B-w4a8-mtp"
)
QWEN3_5_397B_W8A8_MODEL_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp"
)

# Embedding model weights path
BGE_LARGE_EN_V1_5_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "bge-large-en-v1.5")
CLIP_VIT_LARGE_PATCH14_336_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "AI-ModelScope/clip-vit-large-patch14-336"
)
E5_MISTRAL_7B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "intfloat/e5-mistral-7b-instruct"
)
GME_QWEN2_VL_2B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
)
GTE_QWEN2_1_5B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "iic/gte_Qwen2-1.5B-instruct"
)
QWEN3_EMBEDDING_8B_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-Embedding-8B"
)

# Rerank model weights path
BGE_RERANKER_V2_M3_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "BAAI/bge-reranker-v2-m3"
)

# Reward model weights path
INTERNLM2_7B_REWARD_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Shanghai_AI_Laboratory/internlm2-7b-reward"
)
QWEN2_5_1_5B_APEACH_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Howeee/Qwen2.5-1.5B-apeach"
)
QWEN2_5_MATH_RM_72B_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen2.5-Math-RM-72B"
)
SKYWORK_REWARD_GEMMA_2_27B_V0_2_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "AI-ModelScope/Skywork-Reward-Gemma-2-27B-v0.2"
)
SKYWORK_REWARD_LLAMA_3_1_8B_V0_2_WEIGHTS_PATH = os.path.join(
    HF_MODEL_WEIGHTS_DIR,
    "models--Skywork--Skywork-Reward-Llama-3.1-8B-v0.2/snapshots/d4117fbfd81b72f41b96341238baa1e3e90a4ce1",
)
KIMI_K2_6_W4A8_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Eco-Tech/Kimi-K2.6-w4a8")
KIMI_K2_6_EAGLE3_MODEL_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "lightseekorg/kimi-k2.6-eagle3"
)
GLM_4_6V_FLASH_MODEL_PATH = os.path.join(MODEL_WEIGHTS_DIR, "ZhipuAI/GLM-4.6V-Flash")
QWEN3_VL_8B_THINKING_MODEL_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-VL-8B-Thinking"
)
QWEN3_VL_30B_A3B_THINKING_MODEL_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-VL-30B-A3B-Thinking"
)
QWEN3_OMNI_30B_A3B_THINKING_MODEL_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-Omni-30B-A3B-Thinking"
)
# Images path
IMAGES_EXAMPLE_PATH = os.path.join(IMAGES_DIR, "example_image.png")
IMAGES_023_PATH = os.path.join(IMAGES_DIR, "023.jpg")
IMAGES_MAN_PATH = os.path.join(IMAGES_DIR, "man.png")
IMAGES_LOGO_PATH = os.path.join(IMAGES_DIR, "logo.png")
VIDEO_JOBS_PATH = os.path.join(VIDEO_DIR, "jobs.mp4")
INVOICE_WITH_BARCODE_LOGO_IMAGES_PATH = os.path.join(
    IMAGES_DIR, "invoice_with_barcode_logo.jpeg"
)
# fmt: on

# Other
DEEPSEEK_CODER_JSON_PATH = "/__w/sglang/sglang/test/registered/npu/basic_function/parameter/deepseek_coder.json"
FR_SPEC_TOKEN_MAP_PATH = "/root/.cache/sglang/FR-Spec/freq_32768.pt"
CONFIG_YAML_PATH = (
    "/__w/sglang/sglang/test/registered/npu/basic_function/config/config.yaml"
)


class ModelTestConfig(NamedTuple):
    """Model weights path and the expected MMLU / GSM8K / MMMU scores."""

    model_path: str
    mmlu_score: Optional[float] = None
    gsm8k_accuracy: Optional[float] = None
    mmmu_accuracy: Optional[float] = None


LLAMA_3_2_1B_INSTRUCT_WEIGHTS_FOR_TEST = ModelTestConfig(
    model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH, mmlu_score=0.2
)

QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_FOR_TEST = ModelTestConfig(
    model_path=QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH, gsm8k_accuracy=0.9
)

QWEN3_32B_WEIGHTS_FOR_TEST = ModelTestConfig(
    model_path=QWEN3_32B_WEIGHTS_PATH, gsm8k_accuracy=0.82
)

QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST = ModelTestConfig(
    model_path=QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH, gsm8k_accuracy=0.92
)

QWQ_32B_W8A8_WEIGHTS_FOR_TEST = ModelTestConfig(
    model_path=QWQ_32B_W8A8_WEIGHTS_PATH, gsm8k_accuracy=0.59
)

# Default configuration for testing
DEFAULT_WEIGHTS_FOR_TEST = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_FOR_TEST


def run_command(cmd, shell=True):
    """Run ``cmd`` and return its stdout."""
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"execute command error: {e}")
        return None


def get_benchmark_args(
    base_url="",
    backend="sglang",
    dataset_name="",
    dataset_path="",
    tokenizer="",
    num_prompts=500,
    sharegpt_output_len=None,
    random_input_len=4096,
    random_output_len=2048,
    sharegpt_context_len=None,
    request_rate=float("inf"),
    disable_stream=False,
    disable_ignore_eos=False,
    seed: int = 0,
    device="auto",
    pd_separated: bool = False,
    lora_name=None,
    lora_request_distribution="uniform",
    lora_zipf_alpha=1.5,
    gsp_num_groups=4,
    gsp_prompts_per_group=4,
    gsp_system_prompt_len=128,
    gsp_question_len=32,
    gsp_output_len=32,
    gsp_num_turns=1,
    header=None,
    max_concurrency=None,
):
    return SimpleNamespace(
        backend=backend,
        base_url=base_url,
        host=None,
        port=None,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        model=None,
        tokenizer=tokenizer,
        num_prompts=num_prompts,
        sharegpt_output_len=sharegpt_output_len,
        sharegpt_context_len=sharegpt_context_len,
        random_input_len=random_input_len,
        random_output_len=random_output_len,
        random_range_ratio=0.0,
        request_rate=request_rate,
        multi=None,
        output_file=None,
        disable_tqdm=False,
        disable_stream=disable_stream,
        return_logprob=False,
        return_routed_experts=False,
        seed=seed,
        disable_ignore_eos=disable_ignore_eos,
        extra_request_body=None,
        apply_chat_template=False,
        profile=None,
        lora_name=lora_name,
        lora_request_distribution=lora_request_distribution,
        lora_zipf_alpha=lora_zipf_alpha,
        prompt_suffix="",
        device=device,
        pd_separated=pd_separated,
        gsp_num_groups=gsp_num_groups,
        gsp_prompts_per_group=gsp_prompts_per_group,
        gsp_system_prompt_len=gsp_system_prompt_len,
        gsp_question_len=gsp_question_len,
        gsp_output_len=gsp_output_len,
        gsp_num_turns=gsp_num_turns,
        header=header,
        max_concurrency=max_concurrency,
        ready_check_timeout_sec=0,
    )


def run_bench_serving(
    model,
    num_prompts,
    request_rate,
    other_server_args,
    dataset_name="random",
    dataset_path="",
    tokenizer=None,
    random_input_len=4096,
    random_output_len=2048,
    sharegpt_context_len=None,
    disable_stream=False,
    disable_ignore_eos=False,
    need_warmup=False,
    seed: int = 0,
    device="auto",
    gsp_num_groups=None,
    gsp_prompts_per_group=None,
    gsp_system_prompt_len=None,
    gsp_question_len=None,
    gsp_output_len=None,
    max_concurrency=None,
    background_task: Optional[Callable[[str, asyncio.Event], Awaitable[None]]] = None,
    lora_name: Optional[str] = None,
    timeout_for_server_launch=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
):
    """Launch the server, run bench_serving against it, and return the result."""

    if device == "auto":
        device = auto_config_device()
    # Launch the server
    base_url = DEFAULT_URL_FOR_TEST
    process = popen_launch_server(
        model,
        base_url,
        timeout=timeout_for_server_launch,
        other_args=other_server_args,
    )

    # Run benchmark
    args = get_benchmark_args(
        base_url=base_url,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        num_prompts=num_prompts,
        random_input_len=random_input_len,
        random_output_len=random_output_len,
        sharegpt_context_len=sharegpt_context_len,
        request_rate=request_rate,
        disable_stream=disable_stream,
        disable_ignore_eos=disable_ignore_eos,
        seed=seed,
        device=device,
        lora_name=lora_name,
        gsp_num_groups=gsp_num_groups,
        gsp_prompts_per_group=gsp_prompts_per_group,
        gsp_system_prompt_len=gsp_system_prompt_len,
        gsp_question_len=gsp_question_len,
        gsp_output_len=gsp_output_len,
        max_concurrency=max_concurrency,
    )

    async def _run():
        if need_warmup:
            warmup_args = copy.deepcopy(args)
            warmup_args.num_prompts = 16
            await asyncio.to_thread(run_benchmark, warmup_args)

        start_event = asyncio.Event()
        stop_event = asyncio.Event()
        task_handle = (
            asyncio.create_task(background_task(base_url, start_event, stop_event))
            if background_task
            else None
        )

        try:
            start_event.set()
            result = await asyncio.to_thread(run_benchmark, args)
        finally:
            if task_handle:
                stop_event.set()
                await task_handle

        return result

    try:
        res = asyncio.run(_run())
    finally:
        kill_process_tree(process.pid)

    assert res["completed"] == num_prompts
    return res


# hook factory
def create_attention_monitor_hook_factory(config):
    """Return a forward hook that logs self-attention states of ``config["layer_index"]``."""
    # Get target layer index from config, default to 0 if not specified
    layer_index = config.get("layer_index", 0)

    # Initialize logging configuration if no handlers are set
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def attention_monitor_hook(module, inputs, output):
        """Log the attention layer's input/output statistics; returns ``output`` unchanged."""
        # Record current timestamp for time-series tracking
        timestamp = time.time()

        # Extract hidden states from inputs (second input tensor of attention layer)
        hidden_states = inputs[1] if inputs and len(inputs) > 1 else None

        # Construct monitoring record with key statistics
        monitor_record = {
            "timestamp": timestamp,
            "layer_index": layer_index,
            "module_type": type(module).__name__,
            # Compute sum of hidden states across last dim, take first 5 elements for logging
            "inputs": hidden_states.sum(-1)[:5] if hidden_states is not None else None,
            # Compute sum of output across last dim, take first 5 elements for logging
            "outputs": output.sum(-1)[:5],
        }

        # Log the monitoring record
        logging.info(f"hook effect: {monitor_record}")

        # Return the original output to maintain normal model forward propagation
        return output

    return attention_monitor_hook


def read_output(output_lines: List[str], filename: str = STDERR_FILENAME):
    """Print the output in real time with another thread."""
    while not os.path.exists(filename):
        time.sleep(0.01)

    pt = 0
    while pt >= 0:
        if pt > 0 and not os.path.exists(filename):
            break
        try:
            lines = open(filename).readlines()
        except FileNotFoundError:
            print(f"{pt=}, {os.path.exists(filename)=}")
            raise
        for line in lines[pt:]:
            print(line, end="", flush=True)
            output_lines.append(line)
            pt += 1
        time.sleep(0.1)


def run_and_check_memory_leak(
    workload_func,
    disable_radix_cache,
    enable_mixed_chunk,
    disable_overlap,
    chunked_prefill_size,
    assert_has_abort,
    api_key: Optional[str] = None,
):
    other_args = [
        "--chunked-prefill-size",
        str(chunked_prefill_size),
        "--log-level",
        "debug",
    ]
    if disable_radix_cache:
        other_args += ["--disable-radix-cache"]
    if enable_mixed_chunk:
        other_args += ["--enable-mixed-chunk"]
    if disable_overlap:
        other_args += ["--disable-overlap-schedule"]

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    port = random.randint(4000, 5000)
    base_url = f"http://127.0.0.1:{port}"

    # Create files and launch the server
    stdout = open(STDOUT_FILENAME, "w")
    stderr = open(STDERR_FILENAME, "w")
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        return_stdout_stderr=(stdout, stderr),
        api_key=api_key,
    )

    # Launch a thread to stream the output
    output_lines = []
    t = threading.Thread(target=read_output, args=(output_lines,))
    t.start()

    # Run the workload
    workload_func(base_url, model)

    # Clean up everything
    kill_process_tree(process.pid)
    stdout.close()
    stderr.close()
    if os.path.exists(STDOUT_FILENAME):
        os.remove(STDOUT_FILENAME)
    if os.path.exists(STDERR_FILENAME):
        os.remove(STDERR_FILENAME)
    kill_process_tree(process.pid)
    t.join()

    # Assert success
    has_new_server = False
    has_leak = False
    has_abort = False
    for line in output_lines:
        if "Uvicorn running" in line:
            has_new_server = True
        if "leak" in line:
            has_leak = True
        if "Abort" in line:
            has_abort = True

    assert has_new_server
    assert not has_leak
    if assert_has_abort:
        assert has_abort


def run_mmlu_test(
    disable_radix_cache=False,
    enable_mixed_chunk=False,
    disable_overlap=False,
    chunked_prefill_size=32,
):
    def workload_func(base_url, model):
        # Imported lazily: in image mode the CI harness overlays only
        # sglang/test/ascend onto the image, and images built before
        # sglang.test.sgl_eval_utils existed must still be able to import the
        # rest of this module (mmlu is the only user of the eval helper).
        from sglang.test.sgl_eval_utils import run_sgl_eval

        # Run the eval
        args = SimpleNamespace(
            base_url=base_url,
            model=model,
            eval_name="mmlu",
            num_examples=128,
            num_threads=128,
        )

        try:
            metrics = run_sgl_eval(args)
            assert metrics["score"] >= 0.65, f"{metrics=}"
        finally:
            pass

    run_and_check_memory_leak(
        workload_func,
        disable_radix_cache,
        enable_mixed_chunk,
        disable_overlap,
        chunked_prefill_size,
        assert_has_abort=False,
    )


def send_concurrent_requests(
    base_url: str,
    num_requests: int,
    num_concurrent: int = 8,
    input_text: str = "The capital of France is",
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    request_timeout: int = 60,
) -> list:
    """Send ``num_requests`` concurrent POSTs to /generate; returns one result dict per request.

    Uses threads: asyncio.gather() over synchronous requests.post() gives no real
    parallelism for blocking I/O.
    """

    results: list = []
    lock = threading.Lock()
    semaphore = threading.Semaphore(num_concurrent)

    def _send_one(task_id: int) -> None:
        semaphore.acquire()
        try:
            response = requests.post(
                f"{base_url}/generate",
                json={
                    "text": input_text,
                    "sampling_params": {
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                    },
                },
                timeout=request_timeout,
            )
            with lock:
                results.append(
                    {
                        "task_id": task_id,
                        "status_code": response.status_code,
                        "text": response.text,
                    }
                )
        except Exception as exc:
            with lock:
                results.append(
                    {
                        "task_id": task_id,
                        "status_code": -1,
                        "text": str(exc),
                    }
                )
        finally:
            semaphore.release()

    threads = [
        threading.Thread(target=_send_one, args=(i,)) for i in range(num_requests)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results


HEADER = """
### Models
| Model | Server | Client | Output Throughput | Expected Output Throughput | Latency | Expected Latency | Accuracy | Expected Accuracy | Status |
| ----- | ------ | ------ | -------- | ------------------ | ------- | ---------------- | -------- | --------- | ------ |
"""


def write_results_to_github_step_summary(results: dict):
    if not is_in_ci():
        return

    write_github_step_summary_once(HEADER)

    get_float = lambda metrics, item, precision: (
        f"{metrics[item]:.{precision}f}"
        if isinstance(metrics.get(item, "-"), (int, float))
        else metrics.get(item, "-")
    )

    summary = ""
    for model, metrics in results.items():
        model = model.replace(MODEL_WEIGHTS_DIR, "").replace(HF_MODEL_WEIGHTS_DIR, "")
        output_throughput = get_float(metrics, "output_throughput", 2)
        output_throughput_threshold = metrics.get("output_throughput_threshold", "N/A")
        accuracy = get_float(metrics, "accuracy", 4)
        accuracy_threshold = metrics.get("accuracy_threshold", "N/A")
        latency = get_float(metrics, "latency", 4)
        latency_threshold = metrics.get("latency_threshold", "N/A")
        server = metrics.get("server", "N/A")
        client = metrics.get("client", "N/A")
        error = metrics.get("error", "")
        status = "✅" if error == "" else "❌ " + str(error)
        summary += f"| {model} | {server} | {client} | {output_throughput} | {output_throughput_threshold} | {latency} | {latency_threshold} | {accuracy} | {accuracy_threshold} | {status} |\n"
    write_github_step_summary(summary)


def write_github_step_summary_once(summary: str):
    if getattr(write_github_step_summary_once, "has_written", False):
        return
    write_github_step_summary_once.has_written = True
    write_github_step_summary(summary)


def check_port_availability(host, port, timeout=3):
    """Check if the port is available.

    Args:
        host (str): Host IP address.
        port (int): Port number.
        timeout (int): Connection timeout in seconds.

    Returns:
        bool: True if port is available, False otherwise.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, int(port)))
            if result == 0:
                return True
            else:
                return False

    except socket.timeout:
        logger.error(f"Port check timeout for {host}:{port} after {timeout}s")
        return False
    except socket.gaierror as e:
        logger.error(f"Port check address resolution error for {host}:{port}: {e}")
        return False
    except socket.error as e:
        logger.error(f"Port check socket error for {host}:{port}: {e}")
        return False
    except ValueError as e:
        logger.error(f"Port check invalid value for {host}:{port}: {e}")
        return False
    except Exception as e:
        logger.error(f"Port check unexpected error for {host}:{port}: {e}")
        return False


_meta_service_process = None


def get_meta_service_process():
    """Return the MetaService process started for L3 on this pod, if any."""
    return _meta_service_process


def get_k8s_api():
    from kubernetes import client, config

    config.load_kube_config(os.environ.get("KUBECONFIG"))
    return client.CoreV1Api()


def query_configmap(name, namespace):
    """Query ConfigMap from Kubernetes.

    Returns the V1ConfigMap object, or None if failed.
    """
    from kubernetes.client.rest import ApiException

    try:
        configmap = get_k8s_api().read_namespaced_config_map(name, namespace)
        logger.info(f"Successfully queried ConfigMap {name} in namespace {namespace}")
        return configmap
    except ApiException as e:
        logger.error(f"Failed to query ConfigMap {name} in namespace {namespace}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error querying ConfigMap: {e}")
        return None


def get_meta_service_host():
    """Return the node IP that hosts the single cluster-wide MetaService.

    The harness writes every pod's IP into the ConfigMap; the first pod of the
    job (``prefill-0`` for PD separation, ``sglang-node-0`` for PD mix) hosts
    the service. Falls back to the local POD_IP when the ConfigMap is not
    available (e.g. a run outside the multi-node harness).
    """
    local_ip = os.getenv("POD_IP", "127.0.0.1")
    namespace = os.getenv("NAMESPACE")
    configmap_name = os.getenv("KUBE_CONFIG_MAP")
    if not (namespace and configmap_name):
        return local_ip

    deadline = time.time() + META_SERVICE_MASTER_TIMEOUT
    while time.time() < deadline:
        configmap = query_configmap(configmap_name, namespace)
        if configmap and configmap.data:
            for pod_name, pod_ip in configmap.data.items():
                if pod_name.endswith("prefill-0") or pod_name.endswith(
                    "sglang-node-0"
                ):
                    return pod_ip
        logger.info("Waiting for the MetaService master pod in the ConfigMap ...")
        time.sleep(META_SERVICE_READY_POLL_INTERVAL)

    logger.warning(
        f"No master pod found in ConfigMap {configmap_name} after "
        f"{META_SERVICE_MASTER_TIMEOUT}s, falling back to local POD_IP {local_ip}"
    )
    return local_ip


def build_hicache_storage_extra_config(
    world_size: int, protocol: str, dram_size: str
):
    """Start the cluster-wide npu_memcache MetaService and return the L3
    backend config for ``--hicache-storage-backend-extra-config``.

    The service is hosted on the first pod of the job (``get_meta_service_host``)
    and every pod's endpoints point at that node IP. It is started only once
    per process, and only on the pod that owns the host IP.
    """
    global _meta_service_process
    host = get_meta_service_host()
    if _meta_service_process is None and host == os.getenv("POD_IP", "127.0.0.1"):
        _meta_service_process = start_meta_service(host)
    return json.dumps(
        {
            "meta_service_url": f"tcp://{host}:{META_SERVICE_PORT}",
            "config_store_url": f"tcp://{host}:{CONFIG_STORE_PORT}",
            "log_level": "info",
            "world_size": world_size,
            "protocol": protocol,
            "dram_size": dram_size,
        }
    )


def start_meta_service(host_ip):
    """Start the npu_memcache MetaService before the model server.

    The L3 npu_memcache backend connects to the Ascend MemCache MetaService
    at startup, so it must already be listening. ``host_ip`` is the node IP
    shared by all pods, which is also what the L3 backend extra config
    targets.

    Returns the MetaService process.
    """
    meta_service_config = {
        "meta_service_url": f"tcp://{host_ip}:{META_SERVICE_PORT}",
        "config_store_url": f"tcp://{host_ip}:{CONFIG_STORE_PORT}",
        "metrics_url": f"http://{host_ip}:{META_METRICS_PORT}",
    }

    # Materialize the config on the pod-local temp dir so runtime ports
    # (chosen by the test case) are not baked into a checked-in file.
    config_path = os.path.join(
        tempfile.gettempdir(), "npu_memcache_metaservice_config.json"
    )
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(meta_service_config, f, indent=4)

    cmd = [
        sys.executable,
        "-m",
        "sglang.srt.mem_cache.storage.npu_memcache.start_meta_service",
        "--config_path",
        config_path,
    ]
    logger.info(f"Starting npu_memcache MetaService: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, start_new_session=True)

    start_time = time.time()
    while time.time() - start_time < META_SERVICE_READY_TIMEOUT:
        if process.poll() is not None:
            raise RuntimeError(
                f"npu_memcache MetaService exited with code "
                f"{process.returncode} before it was ready"
            )
        if check_port_availability(host_ip, META_SERVICE_PORT):
            logger.info(
                f"npu_memcache MetaService is ready at {host_ip}:{META_SERVICE_PORT}"
            )
            return process
        time.sleep(META_SERVICE_READY_POLL_INTERVAL)

    raise RuntimeError(
        f"Timed out after {META_SERVICE_READY_TIMEOUT}s waiting for "
        f"npu_memcache MetaService at {host_ip}:{META_SERVICE_PORT}"
    )
