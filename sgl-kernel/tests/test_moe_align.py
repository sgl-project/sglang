import torch
from sgl_kernel import moe_align_block_size


def test_moe_align_block_size():
    num_experts = 256
    block_size = 128
    topk_ids = torch.randint(0, num_experts, (3, 4), dtype=torch.int32, device="cuda")

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    token_cnts_buffer = torch.empty(
        (num_experts + 1) * num_experts, dtype=torch.int32, device=topk_ids.device
    )
    cumsum_buffer = torch.empty(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )

    moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        token_cnts_buffer,
        cumsum_buffer,
    )


test_moe_align_block_size()
