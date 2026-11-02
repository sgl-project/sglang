import random
from typing import List

import torch
import triton
import triton.language as tl


def torch_validate_result(
    # validate object
    output_tensor,
    output_tensor_scale,
    output_index,
    expert_start_loc,
    # source
    recv_x,
    recv_x_scale,
    recv_topk,
    num_experts,
):
    output_tensor, output_index, expert_start_loc, recv_x, recv_topk = map(
        lambda x: x.cpu(),
        [output_tensor, output_index, expert_start_loc, recv_x, recv_topk],
    )

    assert (recv_topk >= 0).all()
    expert_token_num = torch.bincount(recv_topk.view(-1), minlength=num_experts)
    cumsum = torch.cumsum(expert_token_num, dim=0)
    expert_start_loc_golden = torch.cat([torch.tensor([0]), cumsum[:-1]])

    assert (expert_start_loc == expert_start_loc_golden).all()
    expert_start_loc = expert_start_loc_golden
    expert_end_loc = expert_start_loc + expert_token_num

    recv_topk_flatten = recv_topk.view(-1)
    output_index_flatten = output_index.view(-1)

    check_bins = torch.bincount(
        output_index_flatten, minlength=len(output_index_flatten)
    )
    assert len(check_bins) == len(output_index_flatten)
    assert (check_bins == 1).all()

    check_status = (expert_start_loc[recv_topk_flatten] <= output_index_flatten) & (
        output_index_flatten < expert_end_loc[recv_topk_flatten]
    )
    if check_status.all():
        print("output_index check pass")
    else:
        print("output_index check fail")
        return

    num_tokens = recv_x.shape[0]
    for i in range(num_tokens):
        check_status = output_tensor[output_index[i]] == recv_x[i].unsqueeze(0)
        if not check_status.all():
            print(f"output_tensor[{i}] check fail!")
            break
    else:
        print("output_tensor check pass")

    if output_tensor_scale is not None:
        for i in range(num_tokens):
            check_status = output_tensor_scale[output_index[i]] == recv_x_scale[
                i
            ].unsqueeze(0)
            if not check_status.all():
                print(f"output_tensor_scale[{i}] check fail!")
                break
        else:
            print("output_tensor_scale check pass")


@triton.jit
def _fwd_kernel_ep_scatter_2_optimal(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    with_scale: tl.constexpr,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    COPY_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    token_id = tl.program_id(0)

    expert_offsets = tl.arange(0, BLOCK_SIZE)
    expert_mask = expert_offsets < topk_num
    expert_loc = tl.load(
        recv_topk + token_id * recv_topk_stride0 + expert_offsets, mask=expert_mask
    )

    tt_mask = expert_mask & (expert_loc >= 0)
    dest_token_index_int32 = tl.atomic_add(
        expert_start_loc + expert_loc, 1, mask=tt_mask
    )
    tl.store(
        output_index + token_id * output_index_stride0 + expert_offsets,
        dest_token_index_int32,
        mask=tt_mask,
        eviction_policy="evict_last",
    )
    tl.debug_barrier()
    dest_token_index = tl.load(
        output_index + token_id * output_index_stride0 + expert_offsets, mask=tt_mask
    ).to(tl.int64)

    value_offsets = tl.arange(0, COPY_SIZE)
    for _ in tl.range(0, triton.cdiv(HIDDEN_SIZE, COPY_SIZE), num_stages=num_stages):
        copy_mask = value_offsets < HIDDEN_SIZE
        to_copy = tl.load(
            recv_x + token_id * recv_x_stride0 + value_offsets, mask=copy_mask
        )
        output_offsets = (
            dest_token_index[:, None] * output_tensor_stride0 + value_offsets[None, :]
        )
        to_copy = to_copy[None, :].broadcast_to(BLOCK_SIZE, COPY_SIZE)
        tl.store(
            output_tensor + output_offsets,
            to_copy,
            mask=(copy_mask[None, :] & tt_mask[:, None]),
        )
        value_offsets += COPY_SIZE

    if with_scale:
        scale_offsets = tl.arange(0, COPY_SIZE)
        for _ in tl.range(
            0,
            triton.cdiv(SCALE_HIDDEN_SIZE, COPY_SIZE),
        ):
            copy_mask = scale_offsets < SCALE_HIDDEN_SIZE
            to_copy_scale = tl.load(
                recv_x_scale + token_id * recv_x_scale_stride0 + scale_offsets,
                mask=copy_mask,
            )
            output_scale_offsets = (
                dest_token_index[:, None] * output_tensor_scale_stride0
                + scale_offsets[None, :]
            )
            to_copy_scale = to_copy_scale[None, :].broadcast_to(BLOCK_SIZE, COPY_SIZE)
            tl.store(
                output_tensor_scale + output_scale_offsets,
                to_copy_scale,
                mask=(copy_mask[None, :] & tt_mask[:, None]),
            )
            scale_offsets += COPY_SIZE


def recover_topk_ids(
    expert_num_tokens: List[int],
    num_tokens,
    num_experts,
    num_experts_per_tok,
) -> torch.Tensor:
    assert len(expert_num_tokens) == num_experts
    assert max(expert_num_tokens) <= num_tokens
    assert sum(expert_num_tokens) == (num_tokens * num_experts_per_tok)

    indices = list(range(len(expert_num_tokens)))
    random.shuffle(indices)
    tensor_slices = [
        torch.ones(size=(expert_num_tokens[i],), dtype=torch.int32) * i for i in indices
    ]
    topk_ids = torch.cat(tensor_slices)
    topk_ids = topk_ids.view(num_experts_per_tok, num_tokens).t().contiguous()

    rand_idx = torch.argsort(torch.rand(num_tokens, num_experts_per_tok), dim=1)
    topk_ids = torch.gather(topk_ids, 1, rand_idx)

    indices = torch.randperm(topk_ids.size(0))
    topk_ids = topk_ids[indices]
    return topk_ids.contiguous()


def base_test(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    expert_num_tokens: List,
    dtype=torch.bfloat16,
):
    hidden_size_scale = hidden_size // 128  # block_shape = [128, 128]
    recv_x = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    scales = torch.randn(
        (num_tokens, hidden_size_scale), dtype=torch.float32, device="cuda"
    )
    topk_ids = recover_topk_ids(expert_num_tokens, num_tokens, num_experts, topk)

    num_tokens_per_expert = torch.bincount(topk_ids.view(-1), minlength=num_experts)
    cumsum = torch.cumsum(num_tokens_per_expert, dim=-1)
    expert_start_loc = torch.cat([torch.tensor([0]), cumsum[:-1]])
    topk_ids = topk_ids.view(num_tokens, topk).cuda()
    expert_start_loc = expert_start_loc.cuda()

    output_tensor = torch.zeros(
        (num_tokens * topk, hidden_size), dtype=dtype, device="cuda"
    )
    output_index = torch.empty_like(topk_ids)
    output_tensor_scale = torch.zeros(
        (num_tokens * topk, hidden_size_scale), dtype=torch.float32, device="cuda"
    )

    expert_start_loc_origin = expert_start_loc.clone()
    grid = lambda meta: (recv_x.shape[0],)
    _fwd_kernel_ep_scatter_2_optimal[grid](
        num_tokens,
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        scales,
        scales.stride(0),
        scales.stride(1),
        topk_ids,
        topk_ids.stride(0),
        topk_ids.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        with_scale=(scales is not None),
        topk_num=topk,
        HIDDEN_SIZE=hidden_size,
        SCALE_HIDDEN_SIZE=hidden_size_scale,
        BLOCK_SIZE=triton.next_power_of_2(topk_ids.shape[1]),
        COPY_SIZE=512,
        num_stages=3,
        num_warps=8,
    )

    torch_validate_result(
        output_tensor,
        output_tensor_scale,
        output_index,
        expert_start_loc_origin,
        recv_x,
        scales,
        topk_ids,
        num_experts,
    )


def set_deterministic_seeds(seed: int = 2025):
    import os
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    # boost CE freq
    # BBA = torch.randn(3*1000*1000*1000, dtype=torch.float32).to(device=torch.device("cuda"))
    set_deterministic_seeds()
    num_experts = 512
    num_tokens = 884
    hidden_size = 2048
    topk = 10
    expert_num_tokens = [
        0,
        106,
        18,
        0,
        19,
        0,
        0,
        0,
        33,
        36,
        1,
        0,
        0,
        0,
        12,
        0,
        5,
        1,
        7,
        8,
        0,
        0,
        0,
        1,
        0,
        0,
        28,
        1,
        0,
        2,
        0,
        15,
        457,
        22,
        0,
        0,
        6,
        5,
        24,
        9,
        4,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        211,
        0,
        4,
        0,
        0,
        0,
        0,
        8,
        0,
        0,
        0,
        4,
        3,
        0,
        6,
        0,
        23,
        59,
        3,
        36,
        0,
        1,
        0,
        1,
        0,
        3,
        0,
        0,
        0,
        0,
        65,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        8,
        0,
        0,
        0,
        47,
        8,
        1,
        1,
        0,
        0,
        43,
        2,
        0,
        1,
        50,
        4,
        0,
        3,
        0,
        0,
        249,
        0,
        30,
        0,
        1,
        89,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        29,
        0,
        1,
        35,
        0,
        3,
        3,
        0,
        1,
        155,
        0,
        4,
        16,
        9,
        0,
        0,
        0,
        35,
        0,
        12,
        34,
        17,
        2,
        37,
        0,
        77,
        11,
        7,
        4,
        0,
        0,
        0,
        0,
        0,
        85,
        0,
        0,
        0,
        14,
        34,
        0,
        0,
        137,
        118,
        23,
        0,
        0,
        17,
        0,
        4,
        58,
        0,
        75,
        0,
        56,
        17,
        0,
        0,
        0,
        3,
        6,
        0,
        0,
        0,
        0,
        4,
        2,
        0,
        1,
        3,
        29,
        0,
        2,
        0,
        5,
        9,
        0,
        0,
        18,
        0,
        6,
        173,
        13,
        0,
        190,
        0,
        2,
        56,
        0,
        12,
        0,
        0,
        67,
        4,
        9,
        175,
        0,
        14,
        0,
        1,
        0,
        14,
        3,
        0,
        0,
        0,
        2,
        3,
        0,
        0,
        121,
        0,
        4,
        0,
        16,
        72,
        0,
        6,
        0,
        0,
        32,
        2,
        56,
        0,
        1,
        0,
        1,
        0,
        0,
        15,
        0,
        29,
        0,
        0,
        31,
        0,
        0,
        2,
        3,
        0,
        3,
        4,
        0,
        3,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        4,
        3,
        0,
        9,
        26,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        35,
        0,
        0,
        0,
        5,
        20,
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        10,
        7,
        0,
        0,
        1,
        16,
        0,
        0,
        0,
        7,
        0,
        0,
        1,
        17,
        21,
        0,
        0,
        3,
        0,
        5,
        12,
        17,
        0,
        0,
        0,
        0,
        14,
        0,
        41,
        0,
        1,
        0,
        0,
        3,
        0,
        158,
        1,
        28,
        5,
        5,
        0,
        0,
        0,
        0,
        8,
        2,
        0,
        0,
        0,
        2,
        0,
        11,
        0,
        66,
        8,
        0,
        0,
        4,
        4,
        1,
        122,
        5,
        450,
        2,
        10,
        109,
        0,
        10,
        202,
        302,
        48,
        212,
        0,
        0,
        10,
        3,
        6,
        1,
        17,
        0,
        28,
        0,
        1,
        0,
        260,
        9,
        142,
        0,
        1,
        0,
        1,
        0,
        4,
        0,
        1,
        23,
        20,
        44,
        3,
        0,
        368,
        0,
        0,
        17,
        60,
        0,
        8,
        130,
        0,
        0,
        0,
        70,
        14,
        0,
        0,
        16,
        242,
        0,
        8,
        27,
        0,
        2,
        0,
        22,
        65,
        1,
        0,
        35,
        0,
        0,
        289,
        0,
        0,
        0,
        8,
        34,
        11,
        0,
        0,
        0,
        33,
        114,
        0,
        0,
        7,
        0,
        0,
        4,
        1,
        0,
        0,
        0,
        0,
        22,
        0,
        0,
        0,
        0,
        39,
        0,
        3,
        0,
        128,
        0,
        23,
        81,
        5,
        0,
        6,
        0,
        0,
        50,
        10,
        0,
        0,
        1,
        0,
        0,
        4,
        8,
        0,
        1,
        4,
        22,
        0,
        1,
        3,
        0,
        1,
        0,
        0,
        0,
        0,
        6,
        1,
        0,
        0,
        10,
        0,
        0,
        1,
        0,
        38,
        0,
        0,
        2,
        0,
        5,
    ]
    base_test(num_tokens, hidden_size, num_experts, topk, expert_num_tokens)
