import torch


def generate_block_sparse_mask_for_function(h, num_blocks, k, device="cuda"):
    """
    Generate block sparse mask of shape [h, num_blocks, num_blocks].

    Args:
        h: number of heads
        num_blocks: number of blocks
        k: number of kv blocks each q block attends to
        device: device to create tensors on

    Returns:
        block_sparse_mask: [h, num_blocks, num_blocks] bool tensor
    """
    k = min(k, num_blocks)
    scores = torch.rand(h, num_blocks, num_blocks, device=device)
    _, indices = torch.topk(scores, k, dim=-1)
    block_sparse_mask = torch.zeros(
        h, num_blocks, num_blocks, dtype=torch.bool, device=device
    )

    block_sparse_mask = block_sparse_mask.scatter_(2, indices, 1).bool()
    return block_sparse_mask


def create_full_mask_from_block_mask(
    block_sparse_mask, variable_block_sizes, device="cuda"
):
    """
    Convert block-level sparse mask to full attention mask.

    Args:
        block_sparse_mask: [h, num_blocks, num_blocks] bool tensor
        variable_block_sizes: [num_blocks] tensor
        device: device to create tensors on

    Returns:
        full_mask: [h, S, S] bool tensor where S = total sequence length
    """
    h, num_blocks, _ = block_sparse_mask.shape
    total_seq_len = variable_block_sizes.sum().item()
    cumsum = torch.cat(
        [torch.tensor([0], device=device), variable_block_sizes.cumsum(dim=0)[:-1]]
    )

    full_mask = torch.zeros(
        h, total_seq_len, total_seq_len, dtype=torch.bool, device=device
    )

    for head in range(h):
        for q_block in range(num_blocks):
            q_start = cumsum[q_block]
            q_end = q_start + variable_block_sizes[q_block]

            for kv_block in range(num_blocks):
                if block_sparse_mask[head, q_block, kv_block]:
                    kv_start = cumsum[kv_block]
                    kv_end = kv_start + variable_block_sizes[kv_block]
                    full_mask[head, q_start:q_end, kv_start:kv_end] = True

    return full_mask
