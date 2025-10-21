from typing import Optional

import pytest
import torch
from sgl_kernel import fused_marlin_moe
from sgl_kernel.scalar_type import ScalarType, scalar_types

from sglang.srt.layers.activation import SiluAndMul
from sglang.test.test_marlin_utils import awq_marlin_quantize, marlin_quantize


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
    )

    M, K = a.shape
    topk = topk_ids.shape[1]
    print("quant_dtype", quant_dtype)
    # exit(0)
    if apply_router_weights_on_input:
        assert topk == 1
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

    m_list = [1, 123, 666]
    n_list = [128, 1024]
    k_list = [256, 2048]
    e_list = [4, 12]
    topk_list = [2, 3]
    dtype_list = [torch.half, torch.bfloat16]
    group_size_list = [128]
    act_order_list = [True, False]
    quant_type_list = [
        scalar_types.uint4,
        scalar_types.uint4b8,
    ]
    is_k_full_list = [True, False]

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
    )

    def is_invalid(
        m, n, k, e, topk, dtype, group_size, act_order, quant_type, is_k_full
    ):

        # Filter act_order
        if act_order:
            if group_size in (-1, k, n):
                return False
            if quant_type not in [scalar_types.uint4b8]:
                return False
        elif not is_k_full:
            return False

        return True

    cases = []
    for case in all_combinations:
        if is_invalid(*case):
            cases.append(case)
    return cases


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    ("m, n, k, e, topk, dtype, group_size," "act_order, quant_type, is_k_full"),
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

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 20
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 20

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

    torch_output = torch_moe(a, w_ref1, w_ref2, score, topk, expert_map=e_map)

    marlin_output = fused_marlin_moe(
        a,
        qweight1,
        qweight2,
        scales1,
        scales2,
        score,
        topk_weights,
        topk_ids,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=zeros1,
        w2_zeros=zeros2,
        num_bits=4,
        is_k_full=is_k_full,
    )

    torch.testing.assert_close(marlin_output, torch_output, atol=5e-2, rtol=0)


if __name__ == "__main__":
    # Run the specific test function directly
    pytest.main([__file__])
