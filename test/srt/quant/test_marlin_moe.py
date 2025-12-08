from typing import Optional

import pytest
import torch
from sgl_kernel.scalar_type import ScalarType, scalar_types

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

from .test_marlin_utils import awq_marlin_quantize, marlin_quantize

set_global_server_args_for_scheduler(object.__new__(ServerArgs))


def stack_and_dev(tensors: list[torch.Tensor]):
    dev = tensors[0].device
    return torch.stack(tensors, dim=0).to(dev)


def torch_experts(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    quant_dtype: Optional[torch.dtype] = None,
    apply_router_weights_on_input: bool = False,
) -> torch.Tensor:
    assert (
        global_num_experts == -1
        or (global_num_experts == w1.shape[0] and expert_map is None)
        or (expert_map is not None and global_num_experts == expert_map.shape[0])
    ), "Invalid expert configuration"

    M, K = a.shape
    topk = topk_ids.shape[1]

    if apply_router_weights_on_input:
        assert topk == 1, "apply_router_weights_on_input only works with topk=1"
        a = a * topk_weight.to(a.dtype)

    a = a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)

    out = torch.zeros(M * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    num_experts = w1.shape[0]

    topk_ids = topk_ids.view(-1)
    if expert_map is not None:
        topk_ids = expert_map[topk_ids]

    f32 = torch.float32

    for i in range(num_experts):
        mask = topk_ids == i
        if mask.sum():
            if quant_dtype is None:
                tmp1 = a[mask] @ w1[i].transpose(0, 1)
                tmp2 = SiluAndMul()(tmp1)
                out[mask] = tmp2 @ w2[i].transpose(0, 1)

    if apply_router_weights_on_input:
        return out
    else:
        return (
            (out.view(M, -1, w2.shape[1]).to(f32) * topk_weight.view(M, -1, 1))
            .sum(dim=1)
            .to(out.dtype)
        )


def torch_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    topk: int,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    return torch_experts(
        a, w1, w2, topk_weight, topk_ids, global_num_experts, expert_map
    )


def marlin_moe_generate_valid_test_cases():
    import itertools

    # Expanded test configurations based on vLLM
    m_list = [1, 33, 123, 256, 666]  # Added more token counts
    n_list = [128, 512, 1024, 2048]  # Added more hidden dimensions
    k_list = [128, 256, 1024, 2048]  # Added more input dimensions
    e_list = [4, 8, 12, 16]  # Added more expert counts
    topk_list = [2, 3, 4]  # Added topk=4
    dtype_list = [torch.half, torch.bfloat16]
    group_size_list = [64, 128, 256]  # Added more group sizes
    act_order_list = [True, False]
    quant_type_list = [
        scalar_types.uint4,  # AWQ-style quantization with zero points
        scalar_types.uint4b8,  # GPTQ-style quantization without zero points
    ]
    is_k_full_list = [True, False]
    ep_size_list = [1, 2]  # Expert parallelism sizes

    all_combinations = itertools.product(
        m_list,
        n_list,
        k_list,
        e_list,
        topk_list,
        dtype_list,
        group_size_list,
        act_order_list,
        quant_type_list,
        is_k_full_list,
        ep_size_list,
    )

    def is_valid(
        m, n, k, e, topk, dtype, group_size, act_order, quant_type, is_k_full, ep_size
    ):
        """
        Validate if a test case configuration is valid.
        
        Rules:
        1. act_order only works with GPTQ (uint4b8) and valid group_size
        2. Group size must divide k evenly
        3. is_k_full only makes sense when act_order is False
        4. Expert count must be divisible by ep_size
        5. Skip very large test cases to keep test time reasonable
        """
        # Group size must divide k evenly
        if group_size > 0 and k % group_size != 0:
            return False

        # Filter act_order constraints
        if act_order:
            # act_order doesn't work with these group sizes
            if group_size in (-1, k, n):
                return False
            # act_order only works with GPTQ (uint4b8)
            if quant_type not in [scalar_types.uint4b8]:
                return False
        else:
            # is_k_full only applies when act_order is False
            if not is_k_full:
                return False

        # Expert parallelism: e must be divisible by ep_size
        if ep_size > 1 and e % ep_size != 0:
            return False


        return True

    cases = []
    for case in all_combinations:
        if is_valid(*case):
            cases.append(case)
    return cases


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    (
        "m, n, k, e, topk, dtype, group_size,"
        "act_order, quant_type, is_k_full, ep_size"
    ),
    marlin_moe_generate_valid_test_cases(),
)
def test_fused_marlin_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    group_size: int,
    act_order: bool,
    quant_type: ScalarType,
    is_k_full: bool,
    ep_size: int,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")

    torch.manual_seed(0)

    has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]

    # Filter act_order
    if act_order:
        if group_size == -1:
            return
        if group_size in (k, n):
            return
        if has_zp:
            return
    else:
        if not is_k_full:
            return

    # Create input tensors
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 20
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 20

    # Setup expert parallelism if needed
    if ep_size > 1:
        local_e = e // ep_size
        # Randomly select which experts this rank handles
        e_ids = torch.randperm(e, device="cuda", dtype=torch.int32)[:local_e]
        e_map = torch.full((e,), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        # Only keep weights for local experts
        w1 = w1[e_ids]
        w2 = w2[e_ids]
    else:
        e_map = None

    w_ref1_l = []
    qweight1_l = []
    scales1_l = []
    zeros1_l = []
    g_idx1_l = []
    sort_indices1_l = []

    for i in range(w1.shape[0]):
        if has_zp:
            w_ref1, qweight1, scales1, zeros1 = awq_marlin_quantize(
                w1[i].transpose(1, 0), quant_type, group_size
            )

            w_ref1_l.append(w_ref1.T)
            qweight1_l.append(qweight1)
            scales1_l.append(scales1)
            zeros1_l.append(zeros1)
        else:
            test_perm = torch.randperm(k)
            w_ref1, qweight1, scales1, g_idx1, sort_indices1, _ = marlin_quantize(
                w1[i].transpose(1, 0), quant_type, group_size, act_order, test_perm
            )

            w_ref1_l.append(w_ref1.T)
            qweight1_l.append(qweight1)
            scales1_l.append(scales1)
            g_idx1_l.append(g_idx1)
            sort_indices1_l.append(sort_indices1)

    w_ref1 = stack_and_dev(w_ref1_l)
    qweight1 = stack_and_dev(qweight1_l).contiguous()
    scales1 = stack_and_dev(scales1_l)
    g_idx1 = stack_and_dev(g_idx1_l) if g_idx1_l else None
    zeros1 = stack_and_dev(zeros1_l) if zeros1_l else None
    sort_indices1 = stack_and_dev(sort_indices1_l) if sort_indices1_l else None

    w_ref2_l = []
    qweight2_l = []
    scales2_l = []
    zeros2_l = []
    g_idx2_l = []
    sort_indices2_l = []

    for i in range(w2.shape[0]):
        if has_zp:
            w_ref2, qweight2, scales2, zeros2 = awq_marlin_quantize(
                w2[i].transpose(1, 0), quant_type, group_size
            )

            w_ref2_l.append(w_ref2.T)
            qweight2_l.append(qweight2)
            scales2_l.append(scales2)
            zeros2_l.append(zeros2)
        else:
            test_perm = torch.randperm(n)
            w_ref2, qweight2, scales2, g_idx2, sort_indices2, _ = marlin_quantize(
                w2[i].transpose(1, 0), quant_type, group_size, act_order, test_perm
            )

            w_ref2_l.append(w_ref2.T)
            qweight2_l.append(qweight2)
            scales2_l.append(scales2)
            g_idx2_l.append(g_idx2)
            sort_indices2_l.append(sort_indices2)

    w_ref2 = stack_and_dev(w_ref2_l)
    qweight2 = stack_and_dev(qweight2_l).contiguous()
    scales2 = stack_and_dev(scales2_l)
    g_idx2 = stack_and_dev(g_idx2_l) if g_idx2_l else None
    zeros2 = stack_and_dev(zeros2_l) if zeros2_l else None
    sort_indices2 = stack_and_dev(sort_indices2_l) if sort_indices2_l else None

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    from sglang.srt.layers.moe.topk import fused_topk_torch_native

    topk_weights, topk_ids = fused_topk_torch_native(a, score, topk, False)

    # Compute reference output using PyTorch implementation
    torch_output = torch_moe(
        a, w_ref1, w_ref2, score, topk, global_num_experts=e, expert_map=e_map
    )

    # Compute output using fused Marlin MoE kernel
    marlin_output = fused_marlin_moe(
        a,
        qweight1,
        qweight2,
        scales1,
        scales2,
        score,
        topk_weights,
        topk_ids,
        global_num_experts=e,
        expert_map=e_map,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=zeros1,
        w2_zeros=zeros2,
        num_bits=4,
        is_k_full=is_k_full,
    )

    # Verify correctness with relatively loose tolerance due to quantization
    torch.testing.assert_close(marlin_output, torch_output, atol=5e-2, rtol=0)


@pytest.mark.parametrize("m", [1, 16, 128])
@pytest.mark.parametrize("e", [8, 16])
def test_fused_marlin_moe_expert_parallelism(m: int, e: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    
    torch.manual_seed(100)
    
    # Test configuration
    n, k = 256, 256
    topk = 2
    ep_size = 2
    group_size = 128
    dtype = torch.bfloat16
    quant_type = scalar_types.uint4b8
    
    # Simulate expert parallelism
    local_e = e // ep_size
    e_ids = torch.randperm(e, device="cuda", dtype=torch.int32)[:local_e]
    e_map = torch.full((e,), -1, device="cuda", dtype=torch.int32)
    e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
    
    # Create test data
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1_full = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 20
    w2_full = torch.randn((e, k, n), device="cuda", dtype=dtype) / 20
    score = torch.randn((m, e), device="cuda", dtype=dtype)
    
    # Select local experts
    w1 = w1_full[e_ids]
    w2 = w2_full[e_ids]
    
    # Quantize local expert weights
    w_ref1_l, qweight1_l, scales1_l = [], [], []
    for i in range(local_e):
        test_perm = torch.randperm(k)
        w_ref1, qweight1, scales1, _, _, _ = marlin_quantize(
            w1[i].transpose(1, 0), quant_type, group_size, False, test_perm
        )
        w_ref1_l.append(w_ref1.T)
        qweight1_l.append(qweight1)
        scales1_l.append(scales1)
    
    w_ref2_l, qweight2_l, scales2_l = [], [], []
    for i in range(local_e):
        test_perm = torch.randperm(n)
        w_ref2, qweight2, scales2, _, _, _ = marlin_quantize(
            w2[i].transpose(1, 0), quant_type, group_size, False, test_perm
        )
        w_ref2_l.append(w_ref2.T)
        qweight2_l.append(qweight2)
        scales2_l.append(scales2)
    
    w_ref1 = stack_and_dev(w_ref1_l)
    qweight1 = stack_and_dev(qweight1_l).contiguous()
    scales1 = stack_and_dev(scales1_l)
    
    w_ref2 = stack_and_dev(w_ref2_l)
    qweight2 = stack_and_dev(qweight2_l).contiguous()
    scales2 = stack_and_dev(scales2_l)
    
    from sglang.srt.layers.moe.topk import fused_topk_torch_native
    topk_weights, topk_ids = fused_topk_torch_native(a, score, topk, False)
    
    # Reference (using full expert weights)
    w_ref1_full_l, w_ref2_full_l = [], []
    for i in range(e):
        test_perm1 = torch.randperm(k)
        w_ref1_full, _, _, _, _, _ = marlin_quantize(
            w1_full[i].transpose(1, 0), quant_type, group_size, False, test_perm1
        )
        w_ref1_full_l.append(w_ref1_full.T)
        
        test_perm2 = torch.randperm(n)
        w_ref2_full, _, _, _, _, _ = marlin_quantize(
            w2_full[i].transpose(1, 0), quant_type, group_size, False, test_perm2
        )
        w_ref2_full_l.append(w_ref2_full.T)
    
    w_ref1_full = stack_and_dev(w_ref1_full_l)
    w_ref2_full = stack_and_dev(w_ref2_full_l)
    
    torch_output = torch_moe(
        a, w_ref1_full, w_ref2_full, score, topk, global_num_experts=e, expert_map=e_map
    )
    
    marlin_output = fused_marlin_moe(
        a,
        qweight1,
        qweight2,
        scales1,
        scales2,
        score,
        topk_weights,
        topk_ids,
        global_num_experts=e,
        expert_map=e_map,
        num_bits=4,
        is_k_full=True,
    )
    
    torch.testing.assert_close(marlin_output, torch_output, atol=5e-2, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
