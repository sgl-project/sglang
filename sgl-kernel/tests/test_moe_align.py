import torch
from sgl_kernel import moe_align_block_size


def test_moe_align_block_size():
    # For DeepSeek V3, we have 256 experts
    num_experts = 256

    # Test different combinations of block_size, num_tokens and topk
    for block_size in [32, 64, 128, 256]:
        print(f"\nTesting block_size={block_size}")
        for num_tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            for topk in [1, 2, 4, 8, 16, 32, 64]:
                print(
                    f"Testing block_size={block_size}, num_tokens={num_tokens}, topk={topk}"
                )

                # Create random topk_ids with shape [num_tokens, topk]
                topk_ids = torch.randint(
                    0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda"
                )

                max_num_tokens_padded = topk_ids.numel() + num_experts * (
                    block_size - 1
                )
                sorted_ids = torch.empty(
                    (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
                )
                sorted_ids.fill_(topk_ids.numel())
                max_num_m_blocks = max_num_tokens_padded // block_size
                expert_ids = torch.empty(
                    (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
                )
                num_tokens_post_pad = torch.empty(
                    (1), dtype=torch.int32, device=topk_ids.device
                )

                token_cnts_buffer = torch.empty(
                    (num_experts + 1) * num_experts,
                    dtype=torch.int32,
                    device=topk_ids.device,
                )
                cumsum_buffer = torch.empty(
                    num_experts + 1, dtype=torch.int32, device=topk_ids.device
                )

                try:
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
                except Exception as e:
                    print(
                        f"Error occurred with block_size={block_size}, num_tokens={num_tokens}, topk={topk}"
                    )
                    print(f"Error message: {str(e)}")
                    raise e


if __name__ == "__main__":
    test_moe_align_block_size()
