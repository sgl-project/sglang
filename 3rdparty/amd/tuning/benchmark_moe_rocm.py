import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tqdm import tqdm
from transformers import AutoConfig

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    fused_moe,
    get_config_file_name,
)

padding_size = 128 if bool(int(os.getenv("SGLANG_MOE_PADDING", "0"))) else 0


def main(model, tp_size, dtype: str, batches):
    method = fused_moe

    for bs in batches:
        run_grid(int(bs), model=model, method=method, tp_size=tp_size, dtype=dtype)


def prune_configs(M, N, K, configs):
    pruned_configs = []
    elemBytes_a = 1  # [DV Note] Hard-coded for float16 (2 bytes)
    elemBytes_b = 1  # [DV Note] Hard-coded for float16 (2 bytes)

    mfma = 16 if M < 32 or N < 32 else 32

    # TODO (zhanglx): figure out the boundary between large and small gemms
    large_gemm = False
    if M >= 2048 and N >= 2048:
        large_gemm = True

    for config in configs:
        BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
        num_warps = config.get("num_warps")
        matrix_instr_nonkdim = config.get("matrix_instr_nonkdim")
        # kpack = config.get("kpack")
        if matrix_instr_nonkdim > mfma:
            continue
        if mfma == 4 and BLOCK_SIZE_K < 64:
            continue
        # some layouts could not work properly in case
        # number elements per thread is less 1
        if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
            continue
        SPLIT_K = 1  # config.get("SPLIT_K")
        GROUP_M = config.get("GROUP_SIZE_M")
        if matrix_instr_nonkdim > BLOCK_SIZE_M or matrix_instr_nonkdim > BLOCK_SIZE_N:
            continue
        if matrix_instr_nonkdim >= M and matrix_instr_nonkdim != BLOCK_SIZE_M:
            continue
        if matrix_instr_nonkdim >= N and matrix_instr_nonkdim != BLOCK_SIZE_N:
            continue
        # Skip BLOCK_SIZE that is too large compare to M/N
        # unless BLOCK_SIZE is already small enough
        if M * 2 < BLOCK_SIZE_M and BLOCK_SIZE_M != 16:
            continue
        if N * 2 < BLOCK_SIZE_N and BLOCK_SIZE_N != 16:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        # skip split_k that leads to EVEN_K = false
        leap = SPLIT_K * BLOCK_SIZE_K
        modv = K % leap
        if modv != 0:
            continue
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
            continue
        # out of shared memory resource
        # TODO (zhanglx): This does not consider the LDS usage in the epilogue
        LDS = (
            BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a
            + BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b
        )
        if LDS > 65536:
            continue
        # Skip small block sizes and num_warps for large gemm
        # For fp16 and f8, we want to only use BLOCK_SIZE >= 64
        if large_gemm:
            if BLOCK_SIZE_M < 64 or BLOCK_SIZE_N < 64:
                continue
            if BLOCK_SIZE_K < 64:
                continue
            if num_warps < 4:
                continue

        pruned_configs.append(config)

    return pruned_configs


def union_of_list_of_dicts(l1, l2):
    result = []
    temp_list = l1.copy()
    temp_list.extend(l2)
    for myDict in temp_list:
        if myDict not in result:
            result.append(myDict)

    return result


def run_grid(bs, model, method, tp_size, dtype: str):

    config = AutoConfig.from_pretrained(model)

    top_k = config.num_experts_per_tok
    d_model = config.hidden_size
    model_intermediate_size = config.intermediate_size
    num_layers = config.num_hidden_layers
    hidden_states_dtype = config.torch_dtype

    if config.num_experts_per_tok:
        if config.architectures[0] == "Grok1ModelForCausalLM":
            num_total_experts = config.num_experts
        else:
            num_total_experts = config.num_local_experts
    else:
        raise ValueError(f"Unsupported Mixtral model {model}")

    # tp_size = 2
    num_warmup_calls = 10
    num_calls = 30

    num_warmup_trials = 1
    num_trials = 1

    full_configs = []

    block_m_range = [16, 32, 64, 128, 256]
    block_n_range = [16, 32, 64, 128, 256]
    block_k_range = [32, 64, 128, 256]  # MUST >= 32
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 4, 8, 16, 32]
    # For now we see better perf with num_stages=0 for all gemm configs we care
    # But keep this explicit so that we do not forget we may need to set it to
    # other values in the future
    num_stage_range = [2]
    waves_per_eu_range = [0, 1, 2, 4, 8]
    # Remove 32 because of triton compiling error
    matrix_instr_nonkdim_range = [16]
    kpack_range = [1, 2]

    for block_size_m in block_m_range:
        for block_size_n in block_n_range:
            for block_size_k in block_k_range:
                for group_size_m in group_m_range:
                    for num_warps in num_warps_range:
                        for num_stages in num_stage_range:
                            for waves_per_eu in waves_per_eu_range:
                                for matrix_instr_nonkdim in matrix_instr_nonkdim_range:
                                    for kpack in kpack_range:
                                        full_configs.append(
                                            {
                                                "BLOCK_SIZE_M": block_size_m,
                                                "BLOCK_SIZE_N": block_size_n,
                                                "BLOCK_SIZE_K": block_size_k,
                                                "GROUP_SIZE_M": group_size_m,
                                                "num_warps": num_warps,
                                                "num_stages": num_stages,
                                                "waves_per_eu": waves_per_eu,
                                                "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                                "kpack": kpack,
                                            }
                                        )

    M1 = bs * 2
    N1 = model_intermediate_size * 2 // tp_size
    K1 = d_model
    prune_configs_1 = prune_configs(M1, N1, K1, full_configs)

    M2 = bs * 2
    N2 = d_model
    K2 = model_intermediate_size // tp_size
    prune_configs_2 = prune_configs(M2, N2, K2, full_configs)

    configs = union_of_list_of_dicts(prune_configs_1, prune_configs_2)

    print(
        f"{bs=} || {len(full_configs)=} | {len(prune_configs_1)=} | \
            {len(prune_configs_2)=} | {len(configs)=}"
    )

    best_config = None
    best_time_us = 1e20

    print(f"{tp_size=} {bs=}")

    for config in tqdm(configs):
        # warmup
        try:
            print(config)
            for _ in range(num_warmup_trials):
                run_timing(
                    num_calls=num_warmup_calls,
                    bs=bs,
                    d_model=d_model,
                    num_total_experts=num_total_experts,
                    top_k=top_k,
                    tp_size=tp_size,
                    model_intermediate_size=model_intermediate_size,
                    method=method,
                    config=config,
                    dtype=dtype,
                    hidden_states_dtype=hidden_states_dtype,
                )
        except triton.runtime.autotuner.OutOfResources:
            continue

        # trial
        for _ in range(num_trials):
            kernel_dur_ms = run_timing(
                num_calls=num_calls,
                bs=bs,
                d_model=d_model,
                num_total_experts=num_total_experts,
                top_k=top_k,
                tp_size=tp_size,
                model_intermediate_size=model_intermediate_size,
                method=method,
                config=config,
                dtype=dtype,
                hidden_states_dtype=hidden_states_dtype,
            )

            kernel_dur_us = 1000 * kernel_dur_ms
            model_dur_ms = kernel_dur_ms * num_layers

            if kernel_dur_us < best_time_us:
                best_config = config
                best_time_us = kernel_dur_us

                tqdm.write(
                    f"{kernel_dur_us=:.1f} {model_dur_ms=:.1f}"
                    f" {bs=} {tp_size=} {top_k=} {num_total_experts=} "
                    f"{d_model=} {model_intermediate_size=} {num_layers=}"
                )

    print("best_time_us", best_time_us)
    print("best_config", best_config)

    # holds Dict[str, Dict[str, int]]
    filename = get_config_file_name(
        num_total_experts,
        model_intermediate_size // tp_size,
        "float8" if dtype == "float8" else None,
    )
    print(f"writing config to file {filename}")
    existing_content = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_content = json.load(f)
    existing_content[str(bs)] = best_config
    with open(filename, "w") as f:
        json.dump(existing_content, f, indent=4)
        f.write("\n")


def run_timing(
    num_calls: int,
    bs: int,
    d_model: int,
    num_total_experts: int,
    top_k: int,
    tp_size: int,
    model_intermediate_size: int,
    method,
    config,
    dtype: str,
    hidden_states_dtype,
) -> float:
    shard_intermediate_size = model_intermediate_size // tp_size

    hidden_states = torch.rand(
        (bs, d_model),
        device="cuda:0",
        dtype=hidden_states_dtype,
    )

    w1 = torch.rand(
        (num_total_experts, 2 * shard_intermediate_size, d_model + padding_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w2 = torch.rand(
        (num_total_experts, d_model, shard_intermediate_size + padding_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None

    if dtype == "float8":
        w1 = w1.to(torch.float8_e4m3fnuz)
        w2 = w2.to(torch.float8_e4m3fnuz)
        w1_scale = torch.ones(
            num_total_experts, device=hidden_states.device, dtype=torch.float32
        )
        w2_scale = torch.ones(
            num_total_experts, device=hidden_states.device, dtype=torch.float32
        )
        a1_scale = torch.ones(1, device=hidden_states.device, dtype=torch.float32)
        a2_scale = torch.ones(1, device=hidden_states.device, dtype=torch.float32)

    gating_output = F.softmax(
        torch.rand(
            (num_calls, bs, num_total_experts),
            device=hidden_states.device,
            dtype=torch.float32,
        ),
        dim=-1,
    )

    ##################################

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(num_calls):
        hidden_states = method(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            gating_output=gating_output[0],
            topk=top_k,
            renormalize=True,
            inplace=True,
            override_config=config,
            use_fp8=dtype == "float8",
        )

    end_event.record()
    end_event.synchronize()

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="benchmark_mixtral_moe",
        description="Benchmark and tune the fused_moe kernel",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["float8", "float16", "bfloat16"],
        help="Data type used for fused_moe kernel computations",
    )
    parser.add_argument("--model", type=str, default="hpcai-tech/grok-1")

    parser.add_argument("--tp-size", type=int, default=2, help="Tensor paralleli size")
    parser.add_argument("-b", "--batches", type=str)

    args = parser.parse_args()

    batches = args.batches.split(",")

    sys.exit(main(args.model, args.tp_size, args.dtype, batches))
