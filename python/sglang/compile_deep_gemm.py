"""
Compile DeepGEMM Kernels for a model with specify server arguments

This script launches a server for capturing DeepGEMM calls and then compiles the kernels.
It accepts server arguments (the same as launch_server.py).

Usage:
python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code

add `--compile-mode offline` to use offline compile.

"""

import argparse
import dataclasses
import multiprocessing
import os
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

import requests
from tqdm.contrib.concurrent import thread_map
from transformers import AutoConfig

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.layers.quantization.deep_gemm import (
    _KERNEL_HELPER_DICT,
    DeepGemmKernelType,
)
from sglang.srt.layers.vocab_parallel_embedding import pad_vocab_size
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_device_core_count, kill_process_tree
from sglang.srt.warmup import warmup

# Reduce warning
os.environ["SGL_IN_DEEPGEMM_PRECOMPILE_STAGE"] = "1"
# Force enable deep gemm
os.environ["SGL_ENABLE_JIT_DEEPGEMM"] = "1"
# Force enable mha chunked kv for DeepSeek V3 to avoid missing kv_b_proj DeepGEMM case
os.environ["SGL_CHUNKED_PREFIX_CACHE_THRESHOLD"] = "0"


class WeightConfig(NamedTuple):
    N: int  # out_features
    K: int  # in_features
    G: int = 1  # num_groups
    B: int = 1  # batch_size


class CompileMapping(NamedTuple):
    weight_name: str
    kernel_type: DeepGemmKernelType
    parallel_axis: Optional[str] = None  # N, K, G, B


@dataclasses.dataclass
class CompileArgs:
    timeout: int = 3600
    compile_mode: str = "online"

    # Only use for offline mode
    compile_num_sms: Optional[int] = None
    compile_m_range: int = 16384
    compile_workers: int = 4

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--timeout", type=int, default=CompileArgs.timeout)
        parser.add_argument(
            "--compile-mode",
            type=str,
            choices=["online", "offline"],
            default=CompileArgs.compile_mode,
            help="Online mode accept server args while offline mode accept config file",
        )
        parser.add_argument(
            "--compile-num-sms",
            type=int,
            default=CompileArgs.compile_num_sms,
        )
        parser.add_argument(
            "--compile-m-range",
            type=int,
            default=CompileArgs.compile_m_range,
        )
        parser.add_argument(
            "--compile-workers",
            type=int,
            default=CompileArgs.compile_workers,
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


@warmup("compile-deep-gemm")
async def warm_up_compile(tokenizer_manager: TokenizerManager):
    print("\nGenerate warm up request for compiling DeepGEMM...\n")
    generate_req_input = GenerateReqInput(
        input_ids=[0, 1, 2, 3],
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": 8,
            "ignore_eos": True,
        },
    )
    await tokenizer_manager.generate_request(generate_req_input, None).__anext__()


def build_deepseek_v3_weights(model_config) -> Dict[str, WeightConfig]:
    num_attention_heads = model_config.num_attention_heads
    qk_nope_head_dim = model_config.qk_nope_head_dim
    qk_rope_head_dim = model_config.qk_rope_head_dim
    v_head_dim = model_config.v_head_dim
    q_lora_rank = model_config.q_lora_rank
    kv_lora_rank = model_config.kv_lora_rank
    hidden_size = model_config.hidden_size
    intermediate_size = model_config.intermediate_size
    n_routed_experts = model_config.n_routed_experts
    n_shared_experts = model_config.n_routed_experts
    moe_intermediate_size = model_config.moe_intermediate_size
    vocab_size = model_config.vocab_size

    weights: Dict[str, WeightConfig] = {}
    # MLA normal part
    weights["q_a_proj"] = WeightConfig(q_lora_rank, hidden_size)
    weights["q_b_proj"] = WeightConfig(
        num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank
    )
    weights["kv_a_proj_with_mqa"] = WeightConfig(
        kv_lora_rank + qk_rope_head_dim, hidden_size
    )
    weights["kv_b_proj"] = WeightConfig(
        num_attention_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank
    )
    weights["o_proj"] = WeightConfig(hidden_size, num_attention_heads * v_head_dim)
    # MLA absorbed addition part
    weights["w_kc"] = WeightConfig(
        kv_lora_rank, qk_nope_head_dim, B=num_attention_heads
    )
    weights["w_vc"] = WeightConfig(v_head_dim, kv_lora_rank, B=num_attention_heads)
    # Moe Dense part
    weights["gate_up_proj"] = WeightConfig(intermediate_size * 2, hidden_size)
    weights["down_proj"] = WeightConfig(hidden_size, intermediate_size)
    # Shared Expert part
    weights["shared.gate_up_proj"] = WeightConfig(
        moe_intermediate_size * 2 * n_shared_experts, hidden_size
    )
    weights["shared.down_proj"] = WeightConfig(
        hidden_size, moe_intermediate_size * n_shared_experts
    )
    # MoE part
    weights["experts.gate_up_proj"] = WeightConfig(
        moe_intermediate_size * 2, hidden_size, G=n_routed_experts
    )
    weights["experts.down_proj"] = WeightConfig(
        hidden_size, moe_intermediate_size, G=n_routed_experts
    )
    # Other
    weights["lm_head"] = WeightConfig(pad_vocab_size(vocab_size), hidden_size)


def build_deepseek_v3_mapping(
    server_args: ServerArgs, weights: Dict[str, WeightConfig]
):
    KernelType = DeepGemmKernelType

    tp_mapping: List[CompileMapping] = [
        CompileMapping("lm_head", KernelType.GEMM_NT_F8F8BF16, "N"),
    ]
    ep_mapping: List[CompileMapping] = []
    dp_mapping: List[CompileMapping] = [
        CompileMapping("q_a_proj", KernelType.GEMM_NT_F8F8BF16),
        CompileMapping("kv_a_proj", KernelType.GEMM_NT_F8F8BF16),
    ]

    # TP/DP attention
    mapping = [
        CompileMapping("q_b_proj", KernelType.GEMM_NT_F8F8BF16, "N"),
        CompileMapping("kv_b_proj", KernelType.GEMM_NT_F8F8BF16, "N"),
        CompileMapping("w_kc", KernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED, "B"),
        CompileMapping("w_vc", KernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED, "B"),
        CompileMapping("o_proj", KernelType.GEMM_NT_F8F8BF16, "K"),
    ]
    if not server_args.enable_dp_attention:
        tp_mapping.extend(mapping)
    else:
        dp_mapping.extend(mapping)

    # TP/DP moe dense
    mapping = [
        CompileMapping("gate_up_proj", KernelType.GEMM_NT_F8F8BF16, "N"),
        CompileMapping("down_proj", KernelType.GEMM_NT_F8F8BF16, "K"),
    ]
    if server_args.moe_dense_tp_size is None:
        tp_mapping.extend(mapping)
    else:
        dp_mapping.extend(mapping)

    # EP moe
    if server_args.enable_deepep_moe:
        dp_mapping.extend(
            [
                CompileMapping("shared.gate_up_proj", KernelType.GEMM_NT_F8F8BF16),
                CompileMapping("shared.down_proj", KernelType.GEMM_NT_F8F8BF16),
            ]
        )
        ep_mapping.extend(
            [
                # deepep low latency
                CompileMapping(
                    "experts.gate_up_proj",
                    KernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED,
                    "G",
                ),
                CompileMapping(
                    "experts.down_proj", KernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED, "G"
                ),
                # deepep normal
                CompileMapping(
                    "experts.gate_up_proj",
                    KernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG,
                    "G",
                ),
                CompileMapping(
                    "experts.down_proj", KernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG, "G"
                ),
            ]
        )
    elif server_args.enable_ep_moe:
        # shared experts fusion
        ep_mapping.extend(
            [
                CompileMapping(
                    "experts.gate_up_proj",
                    KernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED,
                    "G",
                ),
                CompileMapping(
                    "experts.down_proj", KernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED, "G"
                ),
            ]
        )
        weights["experts.gate_up_proj"].G += server_args.ep_size
        weights["experts.down_proj"].G += server_args.ep_size
    else:
        # TP moe does not use DeepGEMM now
        pass

    return tp_mapping, ep_mapping, dp_mapping


def get_parallel_weight_config(
    weights: Dict[str, WeightConfig], mapping: CompileMapping, parallel_size: int
):
    target = weights[mapping.weight_name]
    if parallel_size == 1 or mapping.parallel_axis is None:
        return target
    else:
        target = WeightConfig(target)  # clone it
        parallel_index = target._fields.index(mapping.parallel_axis)
        target[parallel_index] /= parallel_size
        return target


def compile_mappings(
    compile_args: CompileArgs,
    weights: Dict[str, WeightConfig],
    mappings: List[CompileMapping],
    parallel_size: int,
):
    if compile_args.compile_num_sms is None:
        num_sms = get_device_core_count()
    else:
        num_sms = compile_args.compile_num_sms

    for mapping in mappings:
        kernel_helper = _KERNEL_HELPER_DICT[mapping.kernel_type]
        weight_config = get_parallel_weight_config(weights, mapping, parallel_size)

        n, k, g, b = weight_config
        num_groups = g if b == 1 else b

        print(
            f"DeepGEMM JIT Compiling for "
            f"<{kernel_helper.name}> N={n}, K={k}, num_groups={num_groups}, "
            f"with M=1 to {compile_args.compile_m_range})."
        )

        collected_configs = set()
        for m in range(1, compile_args.compile_m_range + 1):
            # Put config into set to get unique configs and reduce cases to be compiled
            collected_configs.add(
                kernel_helper.configure_func(m, n, k, num_groups, num_sms)
            )
        compile_func = lambda config: kernel_helper.compile_func(
            n, k, num_groups, config
        )
        thread_map(
            compile_func, collected_configs, max_workers=compile_args.compile_workers
        )


def compile_deepseek_v3(
    server_args: ServerArgs, compile_args: CompileArgs, model_config
):
    weights = build_deepseek_v3_weights(model_config)
    tp_mapping, ep_mapping, dp_mapping = build_deepseek_v3_mapping(
        server_args, model_config
    )

    print("Compiling DP part...")
    compile_mappings(compile_args, weights, dp_mapping, 1)
    print("Compiling TP part...")
    compile_mappings(compile_args, weights, tp_mapping, server_args.tp_size)
    print("Compiling EP part...")
    compile_mappings(compile_args, weights, ep_mapping, server_args.ep_size)


def launch_server_internal(server_args):
    try:
        launch_server(server_args)
    except Exception as e:
        raise e
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def launch_server_process_and_send_one_request(
    server_args: ServerArgs, compile_args: CompileArgs
):
    proc = multiprocessing.Process(target=launch_server_internal, args=(server_args,))
    proc.start()
    base_url = f"http://{server_args.host}:{server_args.port}"
    timeout = compile_args.timeout

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
            }
            if server_args.node_rank == 0:
                response = requests.get(f"{base_url}/v1/models", headers=headers)
            else:
                # This http api is created by launch_dummy_health_check_server for none-rank0 node.
                response = requests.get(f"{base_url}/health", headers=headers)
            if response.status_code == 200:
                # Rank-0 node send a request to sync with other node and then return.
                if server_args.node_rank == 0:
                    response = requests.post(
                        f"{base_url}/generate",
                        json={
                            "input_ids": [0, 1, 2, 3],
                            "sampling_params": {
                                "max_new_tokens": 8,
                                "temperature": 0,
                            },
                        },
                        timeout=600,
                    )
                    if response.status_code != 200:
                        error = response.json()
                        raise RuntimeError(f"Sync request failed: {error}")
                # Other nodes should wait for the exit signal from Rank-0 node.
                else:
                    start_time_waiting = time.time()
                    while proc.is_alive():
                        if time.time() - start_time_waiting < timeout:
                            time.sleep(10)
                        else:
                            raise TimeoutError("Waiting for main node timeout!")
                return proc
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError(
        "DeepGEMM Kernels compilation timeout."
        "\n\nFeel free and please restart the command."
    )


def refine_server_args(server_args: ServerArgs, compile_args: CompileArgs):
    # Disbale cuda graph and torch compile to save time
    server_args.disable_cuda_graph = True
    server_args.enable_torch_compile = False
    print(f"Disable CUDA Graph and Torch Compile to save time...")

    # Set watchdog timeout to compile_args.timeout because compilation will take a long time
    server_args.watchdog_timeout = compile_args.timeout
    server_args.warmups = "compile-deep-gemm"


def run_online_compile(server_args: ServerArgs, compile_args: CompileArgs):
    print(
        "Begin DeepGEMM Kernels compilation...\n"
        "It may take a long time and timeout maybe raised "
        "while the compilation is still in progress.\n"
        "Just feel free to restart the command "
        "until the compilation is fully finished.\n"
    )

    proc = launch_server_process_and_send_one_request(server_args, compile_args)

    print("\nDeepGEMM Kernels compilation finished successfully.")

    # Sleep for safety
    time.sleep(10)
    if proc.is_alive():
        # This is the rank0 node.
        kill_process_tree(proc.pid)
    else:
        try:
            kill_process_tree(proc.pid)
        except Exception:
            pass


def run_offline_compile(server_args: ServerArgs, compile_args: CompileArgs):
    # refine tp_size
    if server_args.dp_size > 1 and server_args.dp_size != server_args.tp_size:
        assert server_args.tp_size % server_args.dp_size == 0
        server_args.tp_size /= server_args.dp_size

    config = AutoConfig.from_pretrained(server_args.model, trust_remote_code=True)
    if config.architectures[0] in ["DeepseekV3ForCausalLM"]:
        compile_deepseek_v3(server_args, compile_args, config)
    else:
        raise NotImplementedError(f"Only supports DeepseekV3ForCausalLM now")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    CompileArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    compile_args = CompileArgs.from_cli_args(args)

    if compile_args.compile_mode == "offline":
        print(f"{compile_args=}")
        print(f"{server_args=}")
        run_offline_compile(server_args, compile_args)

    elif compile_args.compile_mode == "online":
        multiprocessing.set_start_method("spawn", force=True)

        refine_server_args(server_args, compile_args)
        run_online_compile(server_args, compile_args)

    else:
        raise NotImplementedError(f"Unknown compile mode: {compile_args.compile_mode}")
