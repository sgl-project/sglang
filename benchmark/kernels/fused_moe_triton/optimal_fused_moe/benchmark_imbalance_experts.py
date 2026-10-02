"""
Fusedmoe imbalance Test
"""

import argparse
import os
import random
from typing import Any, Callable, Tuple
from unittest.mock import patch as context_patch

import matplotlib.pyplot as plt
import pandas as pd
import torch
import triton.language as tl
from acext import fusedmoe_wrapper as acext_fusedmoe_warpper
from acext import get_fusedmoe_status_wrapper as get_acext_fusedmoe_status_wrapper
from sgl_kernel import silu_and_mul

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    get_config_dtype_str,
    invoke_fused_moe_kernel,
    moe_align_block_size,
    outplace_fused_experts,
    try_get_optimal_moe_config,
)
from sglang.srt.layers.moe.fused_moe_triton.fused_valu_moe import (
    invoke_special_optimal_fused_moe_impl,
    prepare_for_dispatch,
)
from sglang.srt.utils import get_enum_from_booleans


def _func_argtype(name):
    functions = [
        optimal_moe,
        reference_fused_moe,
        reference_acext_impl,
        actual_fused_moe,
    ]
    func_map = dict(zip([func.__name__.lower() for func in functions], functions))
    if name.lower() not in func_map.keys():
        raise argparse.ArgumentTypeError(
            f"Invalid Test Function Name {name}! Please choose from: {func_map.keys()}"
        )
    return func_map[name.lower()]


def _data_argtype(name):
    return eval(f"torch.{name.lower()}")


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="Sglang Fusedmoe imbalance Test")
    parser.add_argument("-E", "--num-experts", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("-N", "--immediate-size", type=int, default=6144)
    parser.add_argument("-K", "--hidden-size", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--dtype", type=_data_argtype, default=torch.float16)
    parser.add_argument("--func", type=_func_argtype, default=optimal_moe)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--test-times", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--absent-ratio", type=float, default=0.0)
    parser.add_argument("--perf-iters", type=int, default=16)
    parser.add_argument("--save-imbalance-plot", action="store_true")
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    return parser.parse_args()


def plot_imbalance_dist(topk_ids, opts):
    plt.figure(figsize=(10, 6))
    sorted_value, sorted_ids = torch.sort(
        torch.bincount(topk_ids.view(-1), minlength=opts.num_experts)
    )
    cumsum = torch.cumsum(sorted_value, dim=0)
    valu_tokens = int(os.environ.get("FUSEDMOE_VALU_TOKENS", 115))
    critical_pos = (cumsum > valu_tokens).nonzero()[0][0]

    result = pd.DataFrame(data={"Count": sorted_value.tolist()})
    bars = plt.bar(result.index, result["Count"])
    for i, bar in enumerate(bars):
        if i < critical_pos:
            bar.set_color("lightcoral")
            # bar.set_color('skyblue')
        else:
            bar.set_color("skyblue")
        bar.set_edgecolor("black")

    plt.title(
        f"num_tokens per Expert (E={opts.num_experts}, M={opts.num_tokens}, TopK={opts.topk})\n temperature={opts.temperature} absent_ratio={opts.absent_ratio}",
        fontsize=14,
    )
    plt.xlabel("Expert sorted ID", fontsize=12)
    plt.ylabel("tokens", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("count_plot.jpg")
    plt.show()
    plt.close()


def print_inconsistent(
    a: torch.Tensor, b: torch.Tensor, max_print=10, rtol=1e-05, atol=1e-08
):
    a = a.flatten()
    b = b.flatten()
    inconsistent_mask = torch.abs(a - b) > (atol + rtol * torch.abs(b))
    inconsistent_indices = torch.nonzero(inconsistent_mask, as_tuple=True)
    for i, idx in enumerate(zip(*inconsistent_indices)):
        diff = torch.abs(a[idx] - b[idx])
        acc = diff / b[idx]
        print(
            f"\tInconsistent element at index {idx[0].item()}: a = {a[idx]}, b = {b[idx]}, diff={diff}, accuracy={acc}"
        )
        if i >= max_print:
            break


def quant_data(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x * 128) / 128


def gen_imbalance_dist(opts) -> Tuple[torch.Tensor]:
    logits = torch.randn(opts.num_tokens, opts.num_experts, dtype=torch.float32)
    # logits = torch.rand(opts.num_tokens, opts.num_experts, dtype=torch.float32)
    zero_experts = int(opts.absent_ratio * opts.num_experts)
    indices_to_zero = torch.randperm(opts.num_experts)[:zero_experts]
    logits[:, indices_to_zero] = float("-inf")
    scores = torch.softmax(logits / opts.temperature, dim=-1)
    topk_ids = torch.multinomial(scores, opts.topk, replacement=False)

    # 把得到topk_ids洗乱，保证最大token数专家在随机位置
    lookup_table = torch.randperm(opts.num_experts)
    topk_ids = lookup_table[topk_ids]

    # 根据概率决定权重大小
    topk_weight = torch.gather(logits, dim=1, index=topk_ids)
    topk_weight = torch.softmax(topk_weight, dim=-1)
    topk_weight = quant_data(topk_weight)
    return topk_weight.cuda(), topk_ids.cuda()


def benchmark_single_moe(opts, moe_func: Callable) -> None:
    tokens = (
        torch.randn(
            (opts.num_tokens, opts.hidden_size), device="cuda", dtype=opts.dtype
        )
        / 10
    )
    weight1 = (
        torch.randn(
            (opts.num_experts, opts.immediate_size * 2, opts.hidden_size),
            device="cuda",
            dtype=opts.dtype,
        )
        / 10
    )
    weight2 = (
        torch.randn(
            (opts.num_experts, opts.hidden_size, opts.immediate_size),
            device="cuda",
            dtype=opts.dtype,
        )
        / 10
    )

    tokens = quant_data(tokens)
    weight1 = quant_data(weight1)
    weight2 = quant_data(weight2)

    test_moe_list = [gen_imbalance_dist(opts) for i in range(opts.perf_iters)]

    curr_topk_weight = torch.empty_like(test_moe_list[0][0])
    curr_topk_ids = torch.empty_like(test_moe_list[0][1])
    curr_topk_weight.copy_(test_moe_list[0][0])
    curr_topk_ids.copy_(test_moe_list[0][1])

    if opts.save_imbalance_plot:
        plot_imbalance_dist(curr_topk_ids, opts)

    for i in range(opts.warmup_iters):
        with torch.cuda.nvtx.range(f"Warmup Iter {i}"):
            output = moe_func(tokens, weight1, weight2, curr_topk_weight, curr_topk_ids)

    cudagraph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(cudagraph):
        for i in range(4):
            output = moe_func(tokens, weight1, weight2, curr_topk_weight, curr_topk_ids)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    kernel_times = list()
    with torch.cuda.nvtx.range(f"main-test"):
        # non-overlap test for debug
        with (
            context_patch(
                "sglang.srt.layers.moe.fused_moe_triton.fused_valu_moe.get_work_streams"
            ) as single_strm_context,
            torch.cuda.nvtx.range(f"non overlap"),
        ):
            single_strm_context.side_effect = lambda: [torch.cuda.current_stream()] * 2
            output = moe_func(tokens, weight1, weight2, curr_topk_weight, curr_topk_ids)

        for i in range(opts.perf_iters):
            curr_topk_weight.copy_(test_moe_list[i][0])
            curr_topk_ids.copy_(test_moe_list[i][1])

            start_event.record()
            with torch.cuda.nvtx.range(f"Perf-Iter {i}"):
                cudagraph.replay()
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            kernel_times.append(elapsed_time / 4)

    kernel_times = pd.Series(data=kernel_times[1:])
    average = kernel_times.mean()
    print(
        f"E={opts.num_experts}, M={opts.num_tokens}, N={opts.immediate_size}, K={opts.hidden_size}, topK={opts.topk}, dtype={opts.dtype}, {moe_func.__name__} kernel_time = {average * 1000:.2f} us"
    )

    if opts.accuracy and (moe_func.__name__ != "reference_torch_impl"):
        debug = False
        r_curr_topk_weight = curr_topk_weight.clone()
        r_curr_topk_ids = curr_topk_ids.clone()

        outputs = moe_func(
            tokens, weight1, weight2, r_curr_topk_weight, r_curr_topk_ids, debug=debug
        )
        references = reference_torch_impl2(
            tokens, weight1, weight2, r_curr_topk_weight, r_curr_topk_ids, debug=debug
        )
        # references = reference_fused_moe(tokens, weight1, weight2, curr_topk_weight, curr_topk_ids, debug=debug)

        if not isinstance(outputs, tuple):
            outputs = [outputs]

        if not isinstance(references, tuple):
            references = [references]
        # breakpoint()
        for i, (output, ref) in enumerate(zip(outputs, references)):
            assert torch.isfinite(output).all()
            validate_result = torch.allclose(output, ref, opts.rtol, opts.atol)
            if not validate_result:
                print(f"error at {i} tensor")
                print_inconsistent(output, ref, 32, opts.rtol, opts.atol)
                raise f"Validate Fail!!!!!"
        print(f"Validate Pass at atol={opts.atol}, rtol={opts.rtol}")


def optimal_moe(
    tokens: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    debug=False,
) -> torch.Tensor:

    num_tokens, hidden_size = tokens.shape
    num_experts, N, _ = weight1.shape
    _, topk = topk_ids.shape

    assert weight1.shape[2] == hidden_size
    assert weight2.shape == (num_experts, hidden_size, N // 2)

    intermediate_cache1 = torch.empty(
        num_tokens, topk, N, dtype=tokens.dtype, device=tokens.device
    )
    intermediate_cache2 = torch.empty(
        num_tokens * topk, N // 2, dtype=tokens.dtype, device=tokens.device
    )
    intermediate_cache3 = torch.empty(
        num_tokens, topk, hidden_size, dtype=tokens.dtype, device=tokens.device
    )
    output = torch.empty(
        num_tokens, hidden_size, dtype=tokens.dtype, device=tokens.device
    )

    compute_type = tl.bfloat16 if tokens.dtype == torch.bfloat16 else tl.float16
    config_dtype = get_config_dtype_str(tokens.dtype)
    config = try_get_optimal_moe_config(
        weight1.shape, weight2.shape, topk, config_dtype, num_tokens, version=2
    )

    dispatch_tuple = prepare_for_dispatch(topk_ids, config, num_experts)

    invoke_special_optimal_fused_moe_impl(
        tokens,
        weight1,
        intermediate_cache1,
        topk_weight,
        dispatch_tuple,
        False,
        topk,
        config,
        compute_type,
        True,
    )

    silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)

    invoke_special_optimal_fused_moe_impl(
        intermediate_cache2,
        weight2,
        intermediate_cache3,
        topk_weight,
        dispatch_tuple,
        True,
        1,
        config,
        compute_type,
        False,
    )

    torch.sum(
        intermediate_cache3,
        dim=1,
        out=output,
    )

    if debug:
        return intermediate_cache1, intermediate_cache2, intermediate_cache3, output
    else:
        return output


def actual_fused_moe(
    tokens: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    debug=False,
):
    return outplace_fused_experts(tokens, weight1, weight2, topk_weight, topk_ids)


def reference_fused_moe(
    tokens: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    debug=False,
):

    num_tokens, hidden_size = tokens.shape
    num_experts, N, _ = weight1.shape
    _, topk = topk_ids.shape

    assert weight1.shape[2] == hidden_size
    assert weight2.shape == (num_experts, hidden_size, N // 2)

    intermediate_cache1 = torch.empty(
        num_tokens, topk, N, dtype=tokens.dtype, device=tokens.device
    )
    intermediate_cache2 = torch.empty(
        num_tokens * topk, N // 2, dtype=tokens.dtype, device=tokens.device
    )
    intermediate_cache3 = torch.empty(
        num_tokens, topk, hidden_size, dtype=tokens.dtype, device=tokens.device
    )
    output = torch.empty(
        num_tokens, hidden_size, dtype=tokens.dtype, device=tokens.device
    )

    compute_type = tl.bfloat16 if tokens.dtype == torch.bfloat16 else tl.float16

    config_dtype = get_config_dtype_str(tokens.dtype)
    config = try_get_optimal_moe_config(
        weight1.shape, weight2.shape, topk, config_dtype, num_tokens
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], num_experts
    )
    apply_router_weight_on_input = False
    invoke_fused_moe_kernel(
        tokens,
        weight1,
        intermediate_cache1,
        None,
        None,
        None,
        topk_weight,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk_ids.shape[1],
        config["UP"] if "UP" in config.keys() else config,
        use_valu=config.get("USE_VALU", False),
        compute_type=compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
    )

    silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)

    invoke_fused_moe_kernel(
        intermediate_cache2,
        weight2,
        intermediate_cache3,
        None,
        None,
        None,
        topk_weight,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        config["DOWN"] if "DOWN" in config.keys() else config,
        use_valu=config.get("USE_VALU", False),
        compute_type=compute_type,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
    )

    torch.sum(
        intermediate_cache3,
        dim=1,
        out=output,
    )

    if debug:
        return intermediate_cache1, intermediate_cache2, intermediate_cache3, output
    else:
        return output


def reference_acext_impl(
    tokens: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    debug=False,
):

    def pad_to_multiple_of_16(value):
        padding = (16 - value % 16) % 16
        return value + padding

    Q_type = get_enum_from_booleans(
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        use_fp8_w8a16=False,
    )
    use_acext_impl = get_acext_fusedmoe_status_wrapper(
        tokens,
        weight1,
        weight2,
        topk_weight,
        topk_ids,
        None,
        pad_to_multiple_of_16(int(tokens.shape[0] * topk_ids.shape[1])),
        None,
        None,
        None,
        None,
        None,
        None,
        0,
        1,
        Q_type,
    )
    if use_acext_impl != 0:
        print("ACEXT Refuse, please Check!!!!")
        exit()
    output = torch.empty_like(tokens)
    acext_fusedmoe_warpper(
        tokens,
        weight1,
        weight2,
        topk_weight,
        topk_ids,
        output,
        pad_to_multiple_of_16(int(tokens.shape[0] * topk_ids.shape[1])),
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    )
    return output


def reference_torch_impl(
    tokens: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
):
    """Reference to sglang/test/srt/test_fused_moe.py"""
    num_tokens, hidden_size = tokens.shape
    num_experts, N, _ = weight1.shape
    _, topk = topk_ids.shape
    assert weight1.shape[2] == hidden_size
    assert weight2.shape == (num_experts, hidden_size, N // 2)

    expand_tokens = (
        tokens.view(num_tokens, -1, hidden_size)
        .repeat(1, topk, 1)
        .reshape(-1, hidden_size)
    )
    out = torch.zeros(
        num_tokens * topk, hidden_size, dtype=tokens.dtype, device=tokens.device
    )
    temp = torch.zeros(num_tokens * topk, N, dtype=tokens.dtype, device=tokens.device)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    for expert_id in range(num_experts):
        mask = topk_ids == expert_id
        if mask.sum():
            temp[mask] = expand_tokens[mask] @ weight1[expert_id].transpose(0, 1)
            out[mask] = SiluAndMul()(temp[mask]) @ weight2[expert_id].transpose(0, 1)
    output = out.view(num_tokens, -1, hidden_size) * topk_weight.view(
        num_tokens, -1, 1
    ).to(tokens.dtype)
    output = output.sum(dim=1)
    return output


def reference_torch_impl2(
    tokens: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    debug=False,
):
    """Reference to sglang/test/srt/test_fused_moe.py"""
    """ USE CPU for absolute Accuracy! """
    num_tokens, hidden_size = tokens.shape
    num_experts, N, _ = weight1.shape
    _, topk = topk_ids.shape
    assert weight1.shape[2] == hidden_size
    assert weight2.shape == (num_experts, hidden_size, N // 2)

    out_device = tokens.device
    weight1 = weight1.cpu()
    weight2 = weight2.cpu()
    tokens = tokens.cpu()
    topk_weight = topk_weight.cpu()
    topk_ids = topk_ids.cpu()

    intermediate_cache1 = torch.empty(
        num_tokens * topk, N, dtype=tokens.dtype, device="cpu"
    )
    intermediate_cache2 = torch.empty(
        num_tokens * topk, N // 2, dtype=tokens.dtype, device="cpu"
    )
    intermediate_cache3 = torch.empty(
        num_tokens * topk, hidden_size, dtype=tokens.dtype, device="cpu"
    )
    output = torch.empty(num_tokens, hidden_size, dtype=tokens.dtype, device="cpu")

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    expand_tokens = (
        tokens.view(num_tokens, -1, hidden_size)
        .repeat(1, topk, 1)
        .reshape(-1, hidden_size)
    )

    for expert_id in range(num_experts):
        mask = topk_ids == expert_id
        if mask.sum():
            intermediate_cache1[mask] = expand_tokens[mask] @ weight1[
                expert_id
            ].transpose(0, 1)

    intermediate_cache1_dev = intermediate_cache1.to(out_device)
    intermediate_cache2_dev = torch.empty(
        *intermediate_cache2.shape, dtype=intermediate_cache2.dtype, device=out_device
    )
    silu_and_mul(
        intermediate_cache1_dev.view(-1, N), intermediate_cache2_dev.view(-1, N // 2)
    )
    intermediate_cache2 = intermediate_cache2_dev.cpu()

    for expert_id in range(num_experts):
        mask = topk_ids == expert_id
        if mask.sum():
            intermediate_cache3[mask] = intermediate_cache2[mask] @ weight2[
                expert_id
            ].transpose(0, 1)

    if True:
        intermediate_cache3 = (
            intermediate_cache3.view(num_tokens, -1, hidden_size).to(torch.float32)
            * topk_weight.view(num_tokens, -1, 1).to(torch.float32)
        ).to(tokens.dtype)
    output = intermediate_cache3.sum(dim=1)
    if debug:
        return (
            intermediate_cache1.view(num_tokens, topk, N).to(out_device),
            intermediate_cache2.to(out_device),
            intermediate_cache3.view(num_tokens, topk, hidden_size).to(out_device),
            output.to(out_device),
        )
    else:
        return output.to(out_device)


def main(opts) -> None:
    torch.manual_seed(opts.seed)
    random.seed(opts.seed)
    for i in range(opts.test_times):
        benchmark_single_moe(opts, opts.func)


if __name__ == "__main__":
    # os.environ["USE_ACEXT_CUDA"] = "0"
    main(parse_args())
