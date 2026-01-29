"""Common utilities for testing and benchmarking on NPU"""

import os

# Model weights storage directory
MODEL_WEIGHTS_DIR = "/root/.cache/modelscope/hub/models/"

# LLM model weights path
MiniCPM_O_2_6_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "openbmb/MiniCPM-o-2_6")
Llama_3_1_8B_Instruct_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "AI-ModelScope/Llama-3.1-8B-Instruct")
Llama_3_2_1B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "LLM-Research/Llama-3.2-1B")
Llama_3_2_1B_Instruct_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "LLM-Research/Llama-3.2-1B-Instruct")
Llama_3_2_11B_Vision_Instruct_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "LLM-Research/Llama-3.2-11B-Vision-Instruct")
Meta_Llama_3_1_8B_Instruct = os.path.join(MODEL_WEIGHTS_DIR, "LLM-Research/Meta-Llama-3.1-8B-Instruct")
DeepSeek_R1_0528_W8A8_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "vllm-ascend/DeepSeek-R1-0528-W8A8")
DeepSeek_V2_Lite_W8A8_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "vllm-ascend/DeepSeek-V2-Lite-W8A8")
Qwen2_5_7B_Instruct_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen2.5-7B-Instruct")
Qwen3_30B_A3B_Instruct_2507_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3-30B-A3B-Instruct-2507")
