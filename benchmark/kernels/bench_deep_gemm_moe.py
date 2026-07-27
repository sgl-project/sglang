# SPDX-License-Identifier: Apache-2.0
"""Compare production masked and contiguous DeepGEMM MoE paths for MiniMax-M3."""

from __future__ import annotations

import argparse
import gc
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch
import triton

import sglang.srt.layers.moe.moe_runner.deep_gemm as deep_gemm_runner
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.deep_gemm_wrapper import compile_utils
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerOutput,
    _post_permute_deep_gemm_to_standard_contig,
    _post_permute_deep_gemm_to_standard_masked,
    _pre_permute_standard_to_deep_gemm_contig,
    _pre_permute_standard_to_deep_gemm_masked,
    _select_contiguous_gemm_options,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput

NUM_ROUTED_EXPERTS = 128
NUM_SHARED_EXPERTS = 1
NUM_EXPERTS = NUM_ROUTED_EXPERTS + NUM_SHARED_EXPERTS
ROUTED_TOPK = 4
TOPK = ROUTED_TOPK + NUM_SHARED_EXPERTS
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 3072
MXFP8_BLOCK_SIZE = 32
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
ROUTED_SCALING_FACTOR = 2.0


@dataclass
class Case:
    dispatch_output: StandardDispatchOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
        ],
    )
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--rep-ms", type=int, default=20)
    return parser.parse_args()


def check_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("MiniMax-M3 MXFP8 DeepGEMM requires SM100+")
    if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        raise RuntimeError("MXFP8 requires DEEPGEMM_SCALE_UE8M0=True")


def configure_standalone_benchmark() -> None:
    # This is a single-GPU kernel benchmark, so avoid requiring a distributed
    # process group or invalidating the reusable staged inputs.
    deep_gemm_runner.dispose_tensor = lambda _: None
    deep_gemm_runner.get_tp_group = lambda: None
    deep_gemm_runner.is_allocation_symmetric = lambda: False
    deep_gemm_runner.use_symmetric_memory = lambda *args, **kwargs: nullcontext()
    compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE = False


def make_packed_scale(num_experts: int, n: int, k: int) -> torch.Tensor:
    from deep_gemm.utils.layout import (
        get_mn_major_tma_aligned_packed_ue8m0_tensor,
    )

    canonical = torch.ones(
        (num_experts, n, k // MXFP8_BLOCK_SIZE),
        device="cuda",
        dtype=torch.float32,
    )
    packed = get_mn_major_tma_aligned_packed_ue8m0_tensor(canonical)
    del canonical
    return packed


def make_quant_info(intermediate_size_per_rank: int) -> DeepGemmMoeQuantInfo:
    gateup_size = 2 * intermediate_size_per_rank
    print("Preparing synthetic MiniMax-M3 weights and scales ...", flush=True)

    w13_weight = torch.empty(
        (NUM_EXPERTS, gateup_size, HIDDEN_SIZE),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    w2_weight = torch.empty(
        (NUM_EXPERTS, HIDDEN_SIZE, intermediate_size_per_rank),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    for expert in range(NUM_EXPERTS):
        w13_weight[expert].fill_(0.03125 * (1 + expert % 4))
        w2_weight[expert].fill_(0.03125 * (1 + expert % 3))

    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        use_fp8=True,
        w13_scale=make_packed_scale(
            NUM_EXPERTS,
            gateup_size,
            HIDDEN_SIZE,
        ),
        w2_scale=make_packed_scale(
            NUM_EXPERTS,
            HIDDEN_SIZE,
            intermediate_size_per_rank,
        ),
        block_shape=[1, MXFP8_BLOCK_SIZE],
        use_mxfp8=True,
    )
    torch.cuda.synchronize()
    return quant_info


def make_config(intermediate_size_per_rank: int) -> MoeRunnerConfig:
    return MoeRunnerConfig(
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size_per_partition=intermediate_size_per_rank,
        layer_id=0,
        top_k=TOPK,
        num_fused_shared_experts=NUM_SHARED_EXPERTS,
        params_dtype=torch.bfloat16,
        moe_ep_size=1,
        activation="silu",
        is_gated=True,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        gemm1_alpha=SWIGLU_ALPHA,
        gemm1_clamp_limit=SWIGLU_LIMIT,
        gate_up_interleaved=False,
    )


def make_case(batch_size: int) -> Case:
    torch.manual_seed(1000 + batch_size)
    hidden_states = torch.randn(
        (batch_size, HIDDEN_SIZE),
        device="cuda",
        dtype=torch.bfloat16,
    )
    routed_ids = (
        torch.rand((batch_size, NUM_ROUTED_EXPERTS), device="cuda")
        .topk(ROUTED_TOPK, dim=-1)
        .indices.to(torch.int32)
    )
    shared_ids = torch.full(
        (batch_size, NUM_SHARED_EXPERTS),
        NUM_ROUTED_EXPERTS,
        device="cuda",
        dtype=torch.int32,
    )
    topk_ids = torch.cat((routed_ids, shared_ids), dim=-1)

    routed_weights = torch.rand(
        (batch_size, ROUTED_TOPK),
        device="cuda",
        dtype=torch.float32,
    )
    routed_weights /= routed_weights.sum(dim=-1, keepdim=True)
    shared_weights = torch.full(
        (batch_size, NUM_SHARED_EXPERTS),
        1.0 / ROUTED_SCALING_FACTOR,
        device="cuda",
        dtype=torch.float32,
    )
    topk_weights = torch.cat((routed_weights, shared_weights), dim=-1)
    return Case(
        StandardDispatchOutput(
            hidden_states,
            None,
            StandardTopKOutput(topk_weights, topk_ids, None),
        )
    )


def pre_masked(case, quant_info, config):
    state = {}
    runner_input = _pre_permute_standard_to_deep_gemm_masked(
        case.dispatch_output,
        quant_info,
        config,
        state,
    )
    return runner_input, state


def pre_contig(case, quant_info, config):
    state = {}
    runner_input = _pre_permute_standard_to_deep_gemm_contig(
        case.dispatch_output,
        quant_info,
        config,
        state,
    )
    return runner_input, state


def post_masked(runner_output, quant_info, config, state):
    return _post_permute_deep_gemm_to_standard_masked(
        runner_output,
        quant_info,
        config,
        state,
    ).hidden_states


def post_contig(runner_output, quant_info, config, state):
    return _post_permute_deep_gemm_to_standard_contig(
        runner_output,
        quant_info,
        config,
        state,
    ).hidden_states


def run_e2e(case, quant_info, config, core, *, contiguous: bool):
    pre = pre_contig if contiguous else pre_masked
    post = post_contig if contiguous else post_masked
    runner_input, state = pre(case, quant_info, config)
    runner_output = core.run(runner_input, quant_info, state)
    return post(runner_output, quant_info, config, state)


def make_masked_core_stages(core, runner_input, quant_info, state):
    hidden_states = runner_input.hidden_states
    hidden_states_scale = runner_input.hidden_states_scale
    masked_m = runner_input.masked_m
    expected_m = runner_input.expected_m
    assert masked_m is not None
    assert expected_m is not None
    assert quant_info.block_shape is not None

    scale_block_size = quant_info.block_shape[1]
    recipe_a = (quant_info.block_shape[0], state["mxfp8_act_gran_k"])
    recipe_a_down = (quant_info.block_shape[0], scale_block_size)
    recipe_b = tuple(quant_info.block_shape)
    num_groups, max_m, _ = hidden_states.shape
    gateup_size = quant_info.w13_weight.size(1)

    gateup_output = torch.empty(
        (num_groups, max_m, gateup_size),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )

    def gemm1():
        return deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (hidden_states, hidden_states_scale),
            (quant_info.w13_weight, quant_info.w13_scale),
            gateup_output,
            masked_m,
            expected_m,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
        )

    def act_quant():
        return deep_gemm_runner._varlen_deep_gemm_silu_mul_quant(
            gateup_output,
            masked_m,
            group_size=scale_block_size,
            topk=core.config.top_k,
            swiglu_limit=None,
            swizzle=core.use_swizzle,
            gemm1_alpha=core.config.gemm1_alpha,
            gemm1_clamp_limit=core.config.gemm1_clamp_limit,
            num_real_tokens=state["topk_ids"].shape[0],
        )

    gemm1()
    down_input, down_input_scale = act_quant()
    down_output = torch.empty(
        (num_groups, max_m, HIDDEN_SIZE),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )

    def gemm2():
        return deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (down_input, down_input_scale),
            (quant_info.w2_weight, quant_info.w2_scale),
            down_output,
            masked_m,
            expected_m,
            recipe_a=recipe_a_down,
            recipe_b=recipe_b,
        )

    return {"gemm1": gemm1, "act": act_quant, "gemm2": gemm2}, down_output


def make_contig_core_stages(core, runner_input, quant_info, state):
    from sglang.kernels.ops.moe.ep_moe_kernels import (
        silu_and_mul_contig_post_quant_packed_fwd,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import (
        create_per_token_group_quant_fp8_output_scale,
    )

    assert quant_info.block_shape is not None
    all_tokens = state["all_tokens"]
    gateup_size = quant_info.w13_weight.size(1)
    scale_block_size = quant_info.block_shape[1]
    recipe = tuple(quant_info.block_shape)
    (
        grouped_layout,
        use_psum_layout,
        gemm1_zero_padding,
        gemm2_zero_padding,
    ) = _select_contiguous_gemm_options(
        runner_input,
        state,
        core.config,
        gateup_size,
    )

    gateup_output = torch.empty(
        (all_tokens, gateup_size),
        device=runner_input.hidden_states.device,
        dtype=torch.bfloat16,
    )

    def gemm1():
        return deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (runner_input.hidden_states, runner_input.hidden_states_scale),
            (quant_info.w13_weight, quant_info.w13_scale),
            gateup_output,
            grouped_layout,
            recipe_a=recipe,
            recipe_b=recipe,
            use_psum_layout=use_psum_layout,
            ensure_zero_padding=gemm1_zero_padding,
            expected_m_for_psum_layout=(
                runner_input.expected_m if use_psum_layout else None
            ),
        )

    down_input = torch.empty(
        (all_tokens, gateup_size // 2),
        device=gateup_output.device,
        dtype=torch.float8_e4m3fn,
    )
    down_input_scale = create_per_token_group_quant_fp8_output_scale(
        x_shape=down_input.shape,
        device=down_input.device,
        group_size=scale_block_size,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )

    def act_quant():
        return silu_and_mul_contig_post_quant_packed_fwd(
            gateup_output,
            down_input,
            down_input_scale,
            scale_block_size,
            core.config.gemm1_alpha,
            core.config.gemm1_clamp_limit,
        )

    gemm1()
    act_quant()
    down_output = torch.empty(
        (all_tokens, HIDDEN_SIZE),
        device=gateup_output.device,
        dtype=torch.bfloat16,
    )

    def gemm2():
        return deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (down_input, down_input_scale),
            (quant_info.w2_weight, quant_info.w2_scale),
            down_output,
            grouped_layout,
            recipe_a=recipe,
            recipe_b=recipe,
            use_psum_layout=use_psum_layout,
            ensure_zero_padding=gemm2_zero_padding,
            expected_m_for_psum_layout=(
                runner_input.expected_m if use_psum_layout else None
            ),
        )

    return {"gemm1": gemm1, "act": act_quant, "gemm2": gemm2}, down_output


def bench(fn: Callable, rep_ms: int) -> float:
    fn()
    torch.cuda.synchronize()
    return (
        triton.testing.do_bench_cudagraph(
            fn,
            rep=rep_ms,
            return_mode="median",
        )
        * 1000
    )


def benchmark_case(case, quant_info, config, core, rep_ms: int):
    result = {}

    masked_input, masked_state = pre_masked(case, quant_info, config)
    masked_stages, masked_down = make_masked_core_stages(
        core,
        masked_input,
        quant_info,
        masked_state,
    )
    masked_output = DeepGemmRunnerOutput(masked_down)

    contig_input, contig_state = pre_contig(case, quant_info, config)
    contig_stages, contig_down = make_contig_core_stages(
        core,
        contig_input,
        quant_info,
        contig_state,
    )
    contig_output = DeepGemmRunnerOutput(contig_down)

    result["masked_prepare"] = bench(
        lambda: pre_masked(case, quant_info, config),
        rep_ms,
    )
    result["contig_prepare"] = bench(
        lambda: pre_contig(case, quant_info, config),
        rep_ms,
    )
    for stage in ("gemm1", "act", "gemm2"):
        result[f"masked_{stage}"] = bench(masked_stages[stage], rep_ms)
        result[f"contig_{stage}"] = bench(contig_stages[stage], rep_ms)
    result["masked_post"] = bench(
        lambda: post_masked(masked_output, quant_info, config, masked_state),
        rep_ms,
    )
    result["contig_post"] = bench(
        lambda: post_contig(contig_output, quant_info, config, contig_state),
        rep_ms,
    )
    result["masked_e2e"] = bench(
        lambda: run_e2e(
            case,
            quant_info,
            config,
            core,
            contiguous=False,
        ),
        rep_ms,
    )
    result["contig_e2e"] = bench(
        lambda: run_e2e(
            case,
            quant_info,
            config,
            core,
            contiguous=True,
        ),
        rep_ms,
    )
    return result


def print_stage(rows, stage: str, title: str) -> None:
    print(f"\n{title} (us)")
    print("batch       masked       contig    contig_speedup_%")
    for batch_size, values in rows:
        masked = values[f"masked_{stage}"]
        contig = values[f"contig_{stage}"]
        speedup = (masked - contig) / masked * 100
        print(f"{batch_size:>5} {masked:12.3f} {contig:12.3f} {speedup:19.3f}")


def print_results(rows) -> None:
    print("\nPositive speedup means contiguous is faster.")
    for stage, title in (
        ("prepare", "Prepare"),
        ("gemm1", "GEMM1"),
        ("act", "Activation + quantization"),
        ("gemm2", "GEMM2"),
        ("post", "Post"),
        ("e2e", "End-to-end"),
    ):
        print_stage(rows, stage, title)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    check_environment()
    configure_standalone_benchmark()
    if INTERMEDIATE_SIZE % args.tp_size != 0:
        raise ValueError("MiniMax-M3 intermediate size must divide --tp-size")

    intermediate_size_per_rank = INTERMEDIATE_SIZE // args.tp_size
    config = make_config(intermediate_size_per_rank)
    quant_info = make_quant_info(intermediate_size_per_rank)
    core = DeepGemmRunnerCore(config)
    print(
        "MiniMax-M3 DeepGEMM MoE benchmark: "
        f"E={NUM_EXPERTS}, topk={TOPK}, TP={args.tp_size}, "
        "CUDA graph timing"
    )

    rows = []
    for batch_size in args.batch_sizes:
        case = make_case(batch_size)
        masked = run_e2e(
            case,
            quant_info,
            config,
            core,
            contiguous=False,
        )
        contig = run_e2e(
            case,
            quant_info,
            config,
            core,
            contiguous=True,
        )
        assert torch.isfinite(masked).all()
        assert torch.count_nonzero(masked).item() > 0
        torch.testing.assert_close(contig, masked, rtol=2e-2, atol=2e-2)
        del masked, contig

        print(f"Benchmarking batch={batch_size} ...", flush=True)
        rows.append(
            (
                batch_size,
                benchmark_case(case, quant_info, config, core, args.rep_ms),
            )
        )
        del case
        gc.collect()
        torch.cuda.empty_cache()

    print_results(rows)


if __name__ == "__main__":
    main()
