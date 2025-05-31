"""
Compile DeepGEMM Kernels for a model with specify server arguments

This script launches a server for capturing DeepGEMM calls and then compiles the kernels.
It accepts server arguments (the same as launch_server.py).

Usage:
python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code

add `--compile-mode online` to use online compile, which is capturing server's call.

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
    generate_m_range,
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
    parallel_dim: Optional[str] = None  # N, K, G, B


@dataclasses.dataclass
class CompileArgs:
    timeout: int = 3600
    compile_mode: str = "offline"
    output_dir: Optional[str] = None

    # Only use for offline mode
    compile_num_sms: Optional[int] = None
    compile_m_range: Optional[int] = None
    compile_workers: int = 8

    # DeepEP redundant experts, will be removed when DeepEP tbo is ready
    compile_redundant_experts: Optional[int] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--timeout", type=int, default=CompileArgs.timeout)
        parser.add_argument(
            "--compile-mode",
            type=str,
            choices=["online", "offline"],
            default=CompileArgs.compile_mode,
            help=(
                "Offline mode use pre-defined parallel config to generate deep_gemm cases. "
                "While online mode generates cases by capturing server's call."
            ),
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=CompileArgs.output_dir,
            help="directory to save deep_gemm cache",
        )
        parser.add_argument(
            "--compile-num-sms",
            type=int,
            default=CompileArgs.compile_num_sms,
            help="SMs expect to be used in deep_gemm, must set if using two batch overlap",
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
        parser.add_argument(
            "--compile-redundant-experts",
            type=int,
            default=CompileArgs.compile_redundant_experts,
            help="num redundant experts for DeepEP+EPLB+DeepGEMM",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


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
    n_shared_experts = model_config.n_shared_experts
    moe_intermediate_size = model_config.moe_intermediate_size
    vocab_size = model_config.vocab_size

    weights: Dict[str, WeightConfig] = {}
    # MLA normal part
    weights["fused_qkv_a_proj_with_mqa"] = WeightConfig(
        q_lora_rank + kv_lora_rank + qk_rope_head_dim, hidden_size
    )
    weights["q_b_proj"] = WeightConfig(
        num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank
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

    return weights


def build_deepseek_v3_mapping(
    server_args: ServerArgs, weights: Dict[str, WeightConfig]
):
    KernelType = DeepGemmKernelType

    tp_mapping: List[CompileMapping] = [
        # lm_head should never use deep_gemm
        # CompileMapping("lm_head", KernelType.GEMM_NT_F8F8BF16, "N"),
    ]
    dp_mapping: List[CompileMapping] = [
        CompileMapping("fused_qkv_a_proj_with_mqa", KernelType.GEMM_NT_F8F8BF16),
    ]

    # TP/DP attention
    attn_tp_mapping: List[CompileMapping] = [
        CompileMapping("q_b_proj", KernelType.GEMM_NT_F8F8BF16, "N"),
        CompileMapping("kv_b_proj", KernelType.GEMM_NT_F8F8BF16, "N"),
        CompileMapping("w_kc", KernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED, "B"),
        CompileMapping("w_vc", KernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED, "B"),
        CompileMapping("o_proj", KernelType.GEMM_NT_F8F8BF16, "K"),
    ]

    # TP/DP moe dense
    moe_dense_tp_mapping: List[CompileMapping] = [
        CompileMapping("gate_up_proj", KernelType.GEMM_NT_F8F8BF16, "N"),
        CompileMapping("down_proj", KernelType.GEMM_NT_F8F8BF16, "K"),
    ]

    # TP/EP moe
    ep_mapping: List[CompileMapping] = []
    if server_args.enable_deepep_moe:
        dp_mapping.extend(
            [
                CompileMapping("shared.gate_up_proj", KernelType.GEMM_NT_F8F8BF16),
                CompileMapping("shared.down_proj", KernelType.GEMM_NT_F8F8BF16),
            ]
        )

        # For DeepEP+EPLB redundant expert system
        if compile_args.compile_redundant_experts is not None:
            r_experts = compile_args.compile_redundant_experts
            n_experts = weights["experts.gate_up_proj"].G
            weights["experts.gate_up_proj"] = weights["experts.gate_up_proj"]._replace(
                G=n_experts + r_experts
            )
            weights["experts.down_proj"] = weights["experts.down_proj"]._replace(
                G=n_experts + r_experts
            )
            print(
                f'Extend num experts to {weights["experts.gate_up_proj"].G} '
                f"by num_redundant_experts={r_experts}"
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
    else:
        # shared experts fusion just for DeepSeekV3
        if weights["experts.gate_up_proj"].G == 256:
            r_experts = server_args.tp_size
            n_experts = weights["experts.gate_up_proj"].G
            weights["experts.gate_up_proj"] = weights["experts.gate_up_proj"]._replace(
                G=n_experts + r_experts
            )
            weights["experts.down_proj"] = weights["experts.down_proj"]._replace(
                G=n_experts + r_experts
            )
            print(
                f'Extend num experts to {weights["experts.gate_up_proj"].G} '
                f"by tp_size={r_experts}"
            )
        else:
            tp_mapping.extend(
                [
                    CompileMapping(
                        "shared.gate_up_proj", KernelType.GEMM_NT_F8F8BF16, "N"
                    ),
                    CompileMapping(
                        "shared.down_proj", KernelType.GEMM_NT_F8F8BF16, "K"
                    ),
                ]
            )
        if server_args.enable_ep_moe:
            # NOTE: Waiting for EP moe DeepGEMM PR
            # ep_mapping.extend(
            #     [
            #         CompileMapping(
            #             "experts.gate_up_proj",
            #             KernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG,
            #             "G",
            #         ),
            #         CompileMapping(
            #             "experts.down_proj",
            #             KernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG,
            #             "G",
            #         ),
            #     ]
            # )
            pass
        else:
            # NOTE: Waiting for TP moe DeepGEMM PR
            # tp_mapping.extend(
            #     [
            #         CompileMapping(
            #             "experts.gate_up_proj",
            #             KernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG,
            #             "N",
            #         ),
            #         CompileMapping(
            #             "experts.down_proj",
            #             KernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG,
            #             "K",
            #         ),
            #     ]
            # )
            pass

    return tp_mapping, attn_tp_mapping, moe_dense_tp_mapping, ep_mapping, dp_mapping


def get_parallel_weight_config(
    weights: Dict[str, WeightConfig], mapping: CompileMapping, parallel_size: int
):
    target = weights[mapping.weight_name]
    assert target is not None
    if parallel_size == 1 or mapping.parallel_dim is None:
        return target
    else:
        target = target._asdict()
        assert (
            target[mapping.parallel_dim] % parallel_size == 0
        ), f"{target[mapping.parallel_dim]} vs {parallel_size}"
        target[mapping.parallel_dim] //= parallel_size
        return WeightConfig(target["N"], target["K"], target["G"], target["B"])


def compile_mappings(
    mapping_name: str,
    compile_args: CompileArgs,
    weights: Dict[str, WeightConfig],
    mappings: List[CompileMapping],
    parallel_size: int,
):
    if len(mappings) == 0:
        return

    if compile_args.compile_num_sms is None:
        num_sms = get_device_core_count()
    else:
        num_sms = compile_args.compile_num_sms

    print(
        f"Compiling [{mapping_name}] part..., parallel_size={parallel_size}, num_sms={num_sms}"
    )

    for mapping in mappings:
        kernel_helper = _KERNEL_HELPER_DICT[mapping.kernel_type]
        weight_config = get_parallel_weight_config(weights, mapping, parallel_size)

        n, k, g, b = weight_config
        num_groups = g if b == 1 else b

        print(
            f"DeepGEMM JIT Compiling for [{mapping.weight_name}], "
            f"kernel_type=<{kernel_helper.name}> N={n} K={k} num_groups={num_groups}, "
            f"with num_sms={num_sms} M=1to{compile_args.compile_m_range}."
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
    tp_mapping, attn_tp_mapping, moe_dense_tp_mapping, ep_mapping, dp_mapping = (
        build_deepseek_v3_mapping(server_args, weights)
    )

    attn_tp_size = (
        (server_args.tp_size // server_args.dp_size)
        if server_args.enable_dp_attention
        else server_args.tp_size
    )
    moe_dense_tp_size = (
        server_args.tp_size if server_args.moe_dense_tp_size is None else 1
    )

    compile_mappings("DP", compile_args, weights, dp_mapping, 1)
    compile_mappings("TP", compile_args, weights, tp_mapping, server_args.tp_size)
    compile_mappings("Attention", compile_args, weights, attn_tp_mapping, attn_tp_size)
    compile_mappings(
        "MLP", compile_args, weights, moe_dense_tp_mapping, moe_dense_tp_size
    )
    compile_mappings("EP", compile_args, weights, ep_mapping, server_args.ep_size)


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

    start_time = time.perf_counter()
    while time.perf_counter() - start_time < timeout:
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
                    start_time_waiting = time.perf_counter()
                    while proc.is_alive():
                        if time.perf_counter() - start_time_waiting < timeout:
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
    # Disable cuda graph and torch compile to save time
    server_args.disable_cuda_graph = True
    server_args.enable_torch_compile = False
    print(f"Disable CUDA Graph and Torch Compile to save time...")

    # Set watchdog timeout to compile_args.timeout because compilation will take a long time
    server_args.watchdog_timeout = compile_args.timeout
    server_args.warmups = "compile-deep-gemm"


def run_online_compile(server_args: ServerArgs, compile_args: CompileArgs):
    proc = launch_server_process_and_send_one_request(server_args, compile_args)

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
    if compile_args.compile_m_range is None:
        compile_args.compile_m_range = generate_m_range(server_args)

    config = AutoConfig.from_pretrained(server_args.model_path, trust_remote_code=True)
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

    if compile_args.output_dir is not None:
        os.environ["DG_CACHE_DIR"] = compile_args.output_dir

    print(
        "Begin DeepGEMM Kernels compilation...\n"
        "It may take a long time and timeout maybe raised "
        "while the compilation is still in progress.\n"
        "Just feel free to restart the command "
        "until the compilation is fully finished.\n"
    )

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

    print("\nDeepGEMM Kernels compilation finished successfully.")
