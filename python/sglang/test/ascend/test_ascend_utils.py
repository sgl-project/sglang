"""Common utilities for testing and benchmarking on NPU"""

import asyncio
import copy
import os
import subprocess
from types import SimpleNamespace
from typing import Awaitable, Callable, NamedTuple, Optional

from sglang.bench_serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    auto_config_device,
    popen_launch_server,
)

# Model weights storage directory
MODEL_WEIGHTS_DIR = "/root/.cache/modelscope/hub/models/"
HF_MODEL_WEIGHTS_DIR = "/root/.cache/huggingface/hub/"

# LLM model weights path
LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "AI-ModelScope/Llama-3.1-8B-Instruct"
)
LLAMA_3_2_1B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "LLM-Research/Llama-3.2-1B")
LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "LLM-Research/Llama-3.2-1B-Instruct"
)
LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "codelion/Llama-3.2-1B-Instruct-tool-calling-lora"
)
META_LLAMA_3_1_8B_INSTRUCT = os.path.join(
    MODEL_WEIGHTS_DIR, "LLM-Research/Meta-Llama-3.1-8B-Instruct"
)

DEEPSEEK_R1_0528_W8A8_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "vllm-ascend/DeepSeek-R1-0528-W8A8"
)
DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "vllm-ascend/DeepSeek-V2-Lite-W8A8"
)

QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen2.5-7B-Instruct"
)

AFM_4_5B_BASE_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "arcee-ai/AFM-4.5B-Base")
BAICHUAN2_13B_CHAT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "baichuan-inc/Baichuan2-13B-Chat"
)
C4AI_COMMAND_R_V01_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "CohereForAI/c4ai-command-r-v01"
)
C4AI_COMMAND_R_V01_CHAT_TEMPLATE_PATH = "/__w/sglang/sglang/test/registered/ascend/llm_models/tool_chat_template_c4ai_command_r_v01.jinja"
CHATGLM2_6B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "ZhipuAI/chatglm2-6b")
DBRX_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "AI-ModelScope/dbrx-instruct"
)
DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "DeepSeek-V3.2-Exp-W8A8"
)
DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "vllm-ascend/DeepSeek-V3.2-W8A8"
)
DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
)
ERNIE_4_5_21B_A3B_PT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "baidu/ERNIE-4.5-21B-A3B-PT"
)
EXAONE_3_5_7_8B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
)
GEMMA_3_4B_IT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "google/gemma-3-4b-it")
GLM_4_9B_CHAT_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "ZhipuAI/glm-4-9b-chat")
GRANITE_3_0_3B_A800M_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "ibm-granite/granite-3.0-3b-a800m-instruct"
)
GRANITE_3_1_8B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "ibm-granite/granite-3.1-8b-instruct"
)
GROK_2_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "huihui-ai/grok-2")
INTERNLM2_7B_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Shanghai_AI_Laboratory/internlm2-7b"
)
KIMI_K2_THINKING_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Kimi/Kimi-K2-Thinking")
LING_LITE_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "inclusionAI/Ling-lite")
LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "meta-llama/Llama-4-Scout-17B-16E-Instruct"
)
LLAMA_2_7B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "LLM-Research/Llama-2-7B")
MIMO_7B_RL_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "XiaomiMiMo/MiMo-7B-RL")
MINICPM3_4B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "OpenBMB/MiniCPM3-4B")
MISTRAL_7B_INSTRUCT_V0_2_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "mistralai/Mistral-7B-Instruct-v0.2"
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
QWEN3_0_6B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3-0.6B")
QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-30B-A3B-Instruct-2507"
)
QWEN3_32B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3-32B")
QWEN3_32B_EAGLE3_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3-32B-Eagle3")
QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "aleoyang/Qwen3-32B-w8a8-MindIE"
)
QWEN3_235B_A22B_W8A8_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "vllm-ascend/Qwen3-235B-A22B-W8A8"
)
QWEN3_CODER_480B_A35B_INSTRUCT_W8A8_QUAROT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen3-Coder-480B-A35B-Instruct-w8a8-QuaRot"
)
QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen3-Next-80B-A3B-Instruct"
)
QWQ_32B_W8A8_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "vllm-ascend/QWQ-32B-W8A8")
SMOLLM_1_7B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "HuggingFaceTB/SmolLM-1.7B")
STABLELM_2_1_6B_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "stabilityai/stablelm-2-1_6b"
)
XVERSE_MOE_A36B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "xverse/XVERSE-MoE-A36B")
EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "sglang-EAGLE3-LLaMA3.1-Instruct-8B"
)
DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "DeepSeek-R1-0528-w4a8-per-channel"
)

# VLM model weights path
DEEPSEEK_VL2_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "deepseek-ai/deepseek-vl2")
GLM_4_5V_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "ZhipuAI/GLM-4.5V")
JANUS_PRO_1B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "deepseek-ai/Janus-Pro-1B")
JANUS_PRO_7B_WEIGHTS_PATH = os.path.join(MODEL_WEIGHTS_DIR, "deepseek-ai/Janus-Pro-7B")
KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Kimi/Kimi-VL-A3B-Instruct"
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

DEEPSEEK_R1_DISTILL_QWEN_7B_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
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
SKYWORK_REWARD_GEMMA_2_27B_V0_2_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "AI-ModelScope/Skywork-Reward-Gemma-2-27B-v0.2"
)
INTERNLM2_7B_REWARD_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Shanghai_AI_Laboratory/internlm2-7b-reward"
)
SKYWORK_REWARD_LLAMA_3_1_8B_V0_2_WEIGHTS_PATH = os.path.join(
    HF_MODEL_WEIGHTS_DIR,
    "models--Skywork--Skywork-Reward-Llama-3.1-8B-v0.2/snapshots/d4117fbfd81b72f41b96341238baa1e3e90a4ce1",
)
QWEN2_5_1_5B_APEACH_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Howeee/Qwen2.5-1.5B-apeach"
)
QWEN2_5_MATH_RM_72B_WEIGHTS_PATH = os.path.join(
    MODEL_WEIGHTS_DIR, "Qwen/Qwen2.5-Math-RM-72B"
)


class ModelTestConfig(NamedTuple):
    """
    Configuration for model testing.

    Attributes:
        model_path: Path to the model weights directory
        mmlu_score: Weight for MMLU benchmark score
        gsm8k_accuracy: Weight for GSM8K benchmark score
        mmmu_accuracy: Weight for MMMU benchmark score
    """

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
    """Execute system command and return stdout

    parameter:
        cmd: command to execute
        shell:
        True, Execute command in shell
        False, Commands are invoked directly without shell parsing
    return:
        The result of executing the command
    """
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
    """Constructing the parameter objects needed for inference tests

    Parameters:
        base_url: url
        backend: Inference backend
        dataset_name: Data set name
        dataset_path: Dataset path
        tokenizer: tokenizer
        num_prompts: Total number of test requests
        sharegpt_output_len: Output the number of tokens
        random_input_len: The length of the randomly generated input prompt
        random_output_len: The length of the randomly generated output prompt
        sharegpt_context_len: Sharegpt dataset context length
        request_rate: Request rate
        disable_stream: Disable streaming output
        disable_ignore_eos: Should eos_token be ignored?
        seed: random seed
        device: Device type
        pd_separated: Enable PD separation
        lora_name: LoRA fine-tuning model path
        lora_request_distribution: LoRA request distribution strategy
        lora_zipf_alpha: Control request distribution skewness
        gsp_num_groups: Grouped Sequence Parallelism
        gsp_prompts_per_group: Number of parallel prompts within each group
        gsp_system_prompt_len: GSP system prompts length
        gsp_question_len: GSP question length
        gsp_output_len: GSP output length
        gsp_num_turns: GSP Dialogue Rounds
        header: HTTP request header
        max_concurrency: Maximum number of concurrent requests
    Returns:
        The return parameter is the same as the input.
    """

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
):
    """Start the service and obtain the inference results.

    Parameters:
        model: Model name
        num_prompts: Total number of test requests
        request_rate: Request rate
        other_server_args: Additional configuration when starting the service
        dataset_name: Data set name
        dataset_path: Dataset path
        tokenizer: tokenizer
        random_input_len: The length of the randomly generated input prompt
        random_output_len: The length of the randomly generated output prompt
        sharegpt_context_len: Sharegpt dataset context length
        disable_stream: Disable streaming output
        disable_ignore_eos: Should eos_token be ignored?
        need_warmup: Preheating required
        seed: random seed
        device: Device type
        gsp_num_groups: Grouped Sequence Parallelism
        gsp_prompts_per_group: Number of parallel prompts within each group
        gsp_system_prompt_len: GSP system prompts length
        gsp_question_len: GSP question length
        gsp_output_len: GSP output length
        max_concurrency: Maximum number of concurrent requests
        background_task: Background tasks
        lora_name: LoRA fine-tuning model path
    Returns:
        res: Number of requests successfully completed

    """

    if device == "auto":
        device = auto_config_device()
    # Launch the server
    base_url = DEFAULT_URL_FOR_TEST
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
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
