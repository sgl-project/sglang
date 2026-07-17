"""
Benchmark and correctness test for sgl_act_mul_blockwise_quant.

Compares the fused kernel against the two-step baseline:
  1. silu_and_mul (bf16 → bf16)
  2. per_token_group_quant_fp8 (bf16 → fp8 + scale)

Usage:
  python test_act_mul_blockwise_quant.py           # correctness test
  python test_act_mul_blockwise_quant.py --bench   # performance benchmark
"""

import argparse

import torch
import torch.nn.functional as F


def ref_act_mul_blockwise_quant(
    input: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_step: int,
    swiglu_limit: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation in PyTorch for correctness comparison."""
    N, two_C = input.shape
    C = two_C // 2

    gate, up = input.float().chunk(2, dim=-1)

    # swiglu_limit clamp
    if swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)

    # SiLU(gate) * up
    result = F.silu(gate) * up

    # Expert mask: zero out rows where expert_id == -1
    for i in range(N):
        block_idx = i // expert_step
        if block_idx < len(expert_ids) and expert_ids[block_idx] == -1:
            result[i] = 0.0

    # Blockwise FP8 quantization (group_size=128)
    num_groups = C // 128
    groups = result.reshape(N, num_groups, 128)
    scale = groups.abs().amax(dim=-1) / 448.0  # [N, num_groups]
    inv_scale = 1.0 / (scale + 1e-8)
    quant = groups * inv_scale.unsqueeze(-1)
    output_fp8 = quant.reshape(N, C).to(torch.float8_e4m3fn)

    return output_fp8, scale


def test_correctness(
    num_tokens: int = 1024,
    hidden_dim: int = 2048,
    expert_step: int = 128,
    swiglu_limit: float = 10.0,
    filter_ratio: float = 0.2,
):
    """Test correctness against reference and act_and_mul_triton baseline."""
    import sys

    sys.path.insert(0, "/sgl-workspace/sglang/python")

    from sgl_kernel import sgl_act_mul_blockwise_quant, sgl_per_token_group_quant_8bit
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        act_and_mul_triton,
    )

    print(
        f"Testing: tokens={num_tokens}, hidden={hidden_dim}, "
        f"expert_step={expert_step}, swiglu_limit={swiglu_limit}"
    )

    # Setup
    torch.manual_seed(42)
    input_tensor = torch.randn(
        num_tokens, 2 * hidden_dim, dtype=torch.bfloat16, device="cuda"
    )

    num_blocks = (num_tokens + expert_step - 1) // expert_step
    expert_ids = torch.randint(0, 32, (num_blocks,), dtype=torch.int32, device="cuda")
    # Mark some blocks as filtered (expert_id = -1)
    num_filtered = int(num_blocks * filter_ratio)
    if num_filtered > 0:
        filter_indices = torch.randperm(num_blocks)[:num_filtered]
        expert_ids[filter_indices] = -1

    num_groups = hidden_dim // 128

    # --- Method 1: PyTorch reference ---
    ref_output, ref_scale = ref_act_mul_blockwise_quant(
        input_tensor, expert_ids, expert_step, swiglu_limit
    )

    # --- Method 2: Baseline (act_and_mul_triton + quant) ---
    config = {
        "BLOCK_SIZE_M": expert_step if expert_step > 1 else 128,
        "BLOCK_SIZE_N": 128,
    }
    _down_moe_use_tma = expert_step > 1
    baseline_intermediate = torch.empty(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda"
    )
    act_and_mul_triton(
        input_tensor,
        baseline_intermediate,
        config,
        topk_ids=None if _down_moe_use_tma else expert_ids.unsqueeze(1),
        expert_ids=expert_ids if _down_moe_use_tma else None,
        down_moe_use_tma=_down_moe_use_tma,
        activation="silu",
        swiglu_limit=swiglu_limit if swiglu_limit > 0 else None,
    )
    baseline_output = torch.empty(
        num_tokens, hidden_dim, dtype=torch.float8_e4m3fn, device="cuda"
    )
    baseline_scale = torch.empty(
        num_tokens, num_groups, dtype=torch.float32, device="cuda"
    )
    sgl_per_token_group_quant_8bit(
        baseline_intermediate,
        baseline_output,
        baseline_scale,
        128,
        1e-10,
        -448.0,
        448.0,
        False,
    )

    # --- Method 3: Our fused kernel ---
    output = torch.empty(
        num_tokens, hidden_dim, dtype=torch.float8_e4m3fn, device="cuda"
    )
    output_scale = torch.empty(
        num_tokens, num_groups, dtype=torch.float32, device="cuda"
    )
    sgl_act_mul_blockwise_quant(
        output, output_scale, input_tensor, expert_ids, expert_step, swiglu_limit
    )

    # Compare (skip filtered rows)
    valid_mask = torch.ones(num_tokens, dtype=torch.bool, device="cuda")
    for i in range(num_tokens):
        block_idx = i // expert_step
        if block_idx < len(expert_ids) and expert_ids[block_idx] == -1:
            valid_mask[i] = False

    # --- Compare fused vs reference ---
    ref_fp8_float = ref_output[valid_mask].float()
    out_fp8_float = output[valid_mask].float()
    fp8_diff = (ref_fp8_float - out_fp8_float).abs()
    fp8_max_diff = fp8_diff.max().item()
    total_elements = fp8_diff.numel()
    fp8_nonzero_count = (fp8_diff > 0).sum().item()
    fp8_nonzero_ratio = fp8_nonzero_count / total_elements * 100

    ref_scale_valid = ref_scale[valid_mask]
    out_scale_valid = output_scale[valid_mask]
    scale_diff = (ref_scale_valid - out_scale_valid).abs()
    scale_diff.max().item()
    scale_rel_diff = (scale_diff / (ref_scale_valid.abs() + 1e-8)).max().item()

    print(
        f"  vs PyTorch ref:   fp8_max_diff={fp8_max_diff:.4f}, scale_rel_diff={scale_rel_diff:.6f}, "
        f"fp8_mismatch={fp8_nonzero_count}/{total_elements} ({fp8_nonzero_ratio:.4f}%)"
    )

    # --- Compare fused vs baseline (act_and_mul_triton + quant) ---
    baseline_fp8_float = baseline_output[valid_mask].float()
    fp8_diff_vs_baseline = (baseline_fp8_float - out_fp8_float).abs()
    fp8_max_diff_baseline = fp8_diff_vs_baseline.max().item()
    total_elements_baseline = fp8_diff_vs_baseline.numel()
    fp8_nonzero_count_baseline = (fp8_diff_vs_baseline > 0).sum().item()
    fp8_nonzero_ratio_baseline = (
        fp8_nonzero_count_baseline / total_elements_baseline * 100
    )

    baseline_scale_valid = baseline_scale[valid_mask]
    scale_diff_baseline = (baseline_scale_valid - out_scale_valid).abs()
    scale_diff_baseline.max().item()
    scale_rel_diff_baseline = (
        (scale_diff_baseline / (baseline_scale_valid.abs() + 1e-8)).max().item()
    )

    print(
        f"  vs act_and_mul_triton+quant: fp8_max_diff={fp8_max_diff_baseline:.4f}, scale_rel_diff={scale_rel_diff_baseline:.6f}, "
        f"fp8_mismatch={fp8_nonzero_count_baseline}/{total_elements_baseline} ({fp8_nonzero_ratio_baseline:.4f}%)"
    )

    # Thresholds (FP8 e4m3 has coarse steps — max step is 32 at large values)
    assert fp8_max_diff <= 32.0, f"FP8 output max diff vs ref too large: {fp8_max_diff}"
    assert scale_rel_diff < 0.02, (
        f"Scale relative diff vs ref too large: {scale_rel_diff}"
    )
    assert fp8_max_diff_baseline <= 32.0, (
        f"FP8 output max diff vs baseline too large: {fp8_max_diff_baseline}"
    )
    assert scale_rel_diff_baseline < 0.02, (
        f"Scale relative diff vs baseline too large: {scale_rel_diff_baseline}"
    )
    print("  ✅ PASSED\n")


def benchmark(
    num_tokens: int = 32768,
    hidden_dim: int = 2048,
    expert_step: int = 128,
    swiglu_limit: float = 10.0,
    warmup: int = 20,
    repeat: int = 100,
):
    """Benchmark the fused kernel vs two-step baseline (act_and_mul_triton + quant).

    Simulates DeepSeek V3/V4 EP8 production scenario:
    - 256 total experts, EP8 → 32 local experts on this card
    - Each expert gets a variable number of tokens (uneven routing)
    - Tokens are sorted by expert and padded to BLOCK_SIZE_M (expert_step) boundaries
    - expert_ids marks which blocks belong to filtered (non-local) experts as -1
    """
    import sys

    sys.path.insert(0, "/sgl-workspace/sglang/python")

    from sgl_kernel import sgl_act_mul_blockwise_quant, sgl_per_token_group_quant_8bit
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        act_and_mul_triton,
    )

    # --- Simulate realistic expert routing ---
    # DeepSeek V3: 256 experts, topk=8, EP8 → 32 local experts on this card
    num_total_experts = 256
    num_local_experts = 32

    torch.manual_seed(42)

    if expert_step == 1:
        # Non-TMA mode: per-token expert_ids, no sorting, no padding
        # total_tokens = num_tokens * topk, each token has its own expert_id
        total_padded_tokens = num_tokens * 8  # topk=8
        # Random routing: each pair gets a random expert, non-local marked as -1
        expert_ids_raw = torch.randint(0, num_total_experts, (total_padded_tokens,))
        expert_ids_list = []
        for eid in expert_ids_raw.tolist():
            if eid < num_local_experts:
                expert_ids_list.append(eid)
            else:
                expert_ids_list.append(-1)
        expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32, device="cuda")
        num_blocks = total_padded_tokens
    else:
        # TMA mode: sorted by expert, padded to expert_step boundaries
        num_pairs = num_tokens * 8  # topk=8
        routed_experts = torch.randint(0, num_total_experts, (num_pairs,))
        raw_counts = torch.zeros(num_total_experts, dtype=torch.int64)
        for e in routed_experts.tolist():
            raw_counts[e] += 1

        local_expert_mask = torch.zeros(num_total_experts, dtype=torch.bool)
        local_expert_mask[:num_local_experts] = True

        expert_ids_list = []
        total_padded_tokens = 0
        for e in range(num_total_experts):
            n_tokens_e = raw_counts[e].item()
            if n_tokens_e == 0:
                continue
            n_blocks_e = (n_tokens_e + expert_step - 1) // expert_step
            eid = e if local_expert_mask[e] else -1
            expert_ids_list.extend([eid] * n_blocks_e)
            total_padded_tokens += n_blocks_e * expert_step

        expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32, device="cuda")
        num_blocks = len(expert_ids_list)

    # Actual tensor sizes based on padded layout
    input_tensor = torch.randn(
        total_padded_tokens, 2 * hidden_dim, dtype=torch.bfloat16, device="cuda"
    )

    num_groups = hidden_dim // 128
    output = torch.empty(
        total_padded_tokens, hidden_dim, dtype=torch.float8_e4m3fn, device="cuda"
    )
    output_scale = torch.empty(
        total_padded_tokens, num_groups, dtype=torch.float32, device="cuda"
    )
    intermediate_bf16 = torch.empty(
        total_padded_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda"
    )

    config = {"BLOCK_SIZE_M": expert_step, "BLOCK_SIZE_N": 128}

    local_blocks = sum(1 for eid in expert_ids_list if eid != -1)
    filter_ratio = 1.0 - local_blocks / num_blocks if num_blocks > 0 else 0

    print(
        f"Benchmark: tokens={num_tokens}, padded={total_padded_tokens}, hidden={hidden_dim}, "
        f"expert_step={expert_step}, swiglu_limit={swiglu_limit}"
    )
    print(
        f"  Experts: {num_total_experts} total, {num_local_experts} local, "
        f"{num_blocks} blocks, {filter_ratio * 100:.1f}% filtered"
    )
    print(
        f"  Data volume: input={total_padded_tokens * 2 * hidden_dim * 2 / 1e6:.1f} MB (bf16), "
        f"output={total_padded_tokens * hidden_dim / 1e6:.1f} MB (fp8)"
    )

    # =========================================================
    # Benchmark: Fused kernel (ours)
    # =========================================================
    for _ in range(warmup):
        sgl_act_mul_blockwise_quant(
            output, output_scale, input_tensor, expert_ids, expert_step, swiglu_limit
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        sgl_act_mul_blockwise_quant(
            output, output_scale, input_tensor, expert_ids, expert_step, swiglu_limit
        )
    end.record()
    torch.cuda.synchronize()
    fused_time_ms = start.elapsed_time(end) / repeat

    # =========================================================
    # Benchmark: Baseline step 1 — act_and_mul_triton
    # =========================================================
    _down_moe_use_tma = expert_step > 1
    for _ in range(warmup):
        act_and_mul_triton(
            input_tensor,
            intermediate_bf16,
            config,
            topk_ids=None if _down_moe_use_tma else expert_ids.unsqueeze(1),
            expert_ids=expert_ids if _down_moe_use_tma else None,
            down_moe_use_tma=_down_moe_use_tma,
            activation="silu",
            swiglu_limit=swiglu_limit,
        )
    torch.cuda.synchronize()

    start.record()
    for _ in range(repeat):
        act_and_mul_triton(
            input_tensor,
            intermediate_bf16,
            config,
            topk_ids=None if _down_moe_use_tma else expert_ids.unsqueeze(1),
            expert_ids=expert_ids if _down_moe_use_tma else None,
            down_moe_use_tma=_down_moe_use_tma,
            activation="silu",
            swiglu_limit=swiglu_limit,
        )
    end.record()
    torch.cuda.synchronize()
    act_time_ms = start.elapsed_time(end) / repeat

    # =========================================================
    # Benchmark: Baseline step 2 — per_token_group_quant_fp8
    # =========================================================
    for _ in range(warmup):
        sgl_per_token_group_quant_8bit(
            intermediate_bf16, output, output_scale, 128, 1e-10, -448.0, 448.0, False
        )
    torch.cuda.synchronize()

    start.record()
    for _ in range(repeat):
        sgl_per_token_group_quant_8bit(
            intermediate_bf16, output, output_scale, 128, 1e-10, -448.0, 448.0, False
        )
    end.record()
    torch.cuda.synchronize()
    quant_time_ms = start.elapsed_time(end) / repeat

    # =========================================================
    # Results
    # =========================================================
    baseline_time_ms = act_time_ms + quant_time_ms
    speedup = baseline_time_ms / fused_time_ms if fused_time_ms > 0 else 0

    print(f"\n  Results (avg over {repeat} runs):")
    print(f"    Fused kernel (ours):            {fused_time_ms:.4f} ms")
    print(f"    Baseline act_and_mul_triton:    {act_time_ms:.4f} ms")
    print(f"    Baseline quant_fp8:             {quant_time_ms:.4f} ms")
    print(f"    Baseline total (act+quant):     {baseline_time_ms:.4f} ms")
    print(f"    Speedup (fused vs act+quant):   {speedup:.2f}x")
    print()

    return (
        fused_time_ms,
        act_time_ms,
        quant_time_ms,
        baseline_time_ms,
        speedup,
        total_padded_tokens,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench", action="store_true", help="Run performance benchmark"
    )
    parser.add_argument("--tokens", type=int, default=None, help="Number of tokens")
    parser.add_argument("--hidden", type=int, default=2048, help="Hidden dim (C)")
    args = parser.parse_args()

    if not args.bench:
        # Correctness tests
        print("=" * 60)
        print("Correctness Tests")
        print("=" * 60)

        # --- TMA mode (expert_step=128, sorted by expert with padding) ---
        print("--- TMA mode (expert_step=128, sorted layout) ---")
        test_cases_tma = [
            (1024, 2048, 128, 0.0, 0.0),  # no clamp, no filter
            (1024, 2048, 128, 10.0, 0.0),  # with clamp, no filter
            (1024, 2048, 128, 10.0, 0.3),  # with clamp, 30% filtered
            (4096, 2048, 128, 10.0, 0.2),  # larger batch
            (2048, 4096, 128, 10.0, 0.2),  # larger hidden
        ]
        for tokens, hidden, step, limit, filt in test_cases_tma:
            test_correctness(tokens, hidden, step, limit, filt)

        # --- Non-TMA mode (expert_step=1, per-token expert_ids, no padding) ---
        print("--- Non-TMA mode (expert_step=1, per-token layout) ---")
        test_cases_non_tma = [
            (1024, 2048, 1, 0.0, 0.0),  # no clamp, no filter
            (1024, 2048, 1, 10.0, 0.0),  # with clamp, no filter
            (1024, 2048, 1, 10.0, 0.3),  # with clamp, 30% filtered
            (4096, 2048, 1, 10.0, 0.2),  # larger batch
            (2048, 4096, 1, 10.0, 0.2),  # larger hidden
        ]
        for tokens, hidden, step, limit, filt in test_cases_non_tma:
            test_correctness(tokens, hidden, step, limit, filt)

        print("All correctness tests passed! ✅")
    else:
        # Benchmark with production-realistic parameters:
        # DeepSeek V3/V4: moe_intermediate_size=2048, BLOCK_SIZE_M=128, swiglu_limit=10.0
        # 256 experts total, EP8 → 32 local experts per card, topk=8
        # Tokens routed unevenly across experts (realistic distribution)
        print("=" * 100)
        print("Performance Benchmark — DeepSeek V3/V4 Production Config")
        print("=" * 100)
        print("  hidden_dim=2048, swiglu_limit=10.0")
        print("  256 experts, EP8 → 32 local + 224 filtered")
        print("  Realistic: per-expert token count follows uneven distribution")
        print("=" * 100)

        if args.tokens:
            token_sizes = [args.tokens]
        else:
            token_sizes = [
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
                32768,
                65536,
            ]

        # ======================== TMA mode (expert_step=128) ========================
        print("\n" + "=" * 100)
        print("  Mode: TMA (expert_step=128, sorted by expert, with padding)")
        print("=" * 100)
        results_tma = []
        for tokens in token_sizes:
            fused_ms, act_ms, quant_ms, total_ms, speedup, padded = benchmark(
                tokens, args.hidden, expert_step=128, swiglu_limit=10.0
            )
            results_tma.append(
                (tokens, padded, fused_ms, act_ms, quant_ms, total_ms, speedup)
            )

        print("\n" + "-" * 100)
        print(
            f"{'Tokens':>8} | {'Total':>8} | {'Fused(ms)':>10} | {'Act(ms)':>10} | {'Quant(ms)':>10} | {'Total(ms)':>10} | {'Speedup':>8}"
        )
        print("-" * 100)
        for (
            tokens,
            padded,
            fused_ms,
            act_ms,
            quant_ms,
            total_ms,
            speedup,
        ) in results_tma:
            print(
                f"{tokens:>8} | {padded:>8} | {fused_ms:>10.4f} | {act_ms:>10.4f} | {quant_ms:>10.4f} | {total_ms:>10.4f} | {speedup:>7.2f}x"
            )
        print("-" * 100)

        # ======================== Non-TMA mode (expert_step=1) ========================
        print("\n" + "=" * 100)
        print("  Mode: Non-TMA (expert_step=1, per-token routing, no padding)")
        print("=" * 100)
        results_non_tma = []
        for tokens in token_sizes:
            fused_ms, act_ms, quant_ms, total_ms, speedup, padded = benchmark(
                tokens, args.hidden, expert_step=1, swiglu_limit=10.0
            )
            results_non_tma.append(
                (tokens, padded, fused_ms, act_ms, quant_ms, total_ms, speedup)
            )

        print("\n" + "-" * 100)
        print(
            f"{'Tokens':>8} | {'Total':>8} | {'Fused(ms)':>10} | {'Act(ms)':>10} | {'Quant(ms)':>10} | {'Total(ms)':>10} | {'Speedup':>8}"
        )
        print("-" * 100)
        for (
            tokens,
            padded,
            fused_ms,
            act_ms,
            quant_ms,
            total_ms,
            speedup,
        ) in results_non_tma:
            print(
                f"{tokens:>8} | {padded:>8} | {fused_ms:>10.4f} | {act_ms:>10.4f} | {quant_ms:>10.4f} | {total_ms:>10.4f} | {speedup:>7.2f}x"
            )
        print("-" * 100)
