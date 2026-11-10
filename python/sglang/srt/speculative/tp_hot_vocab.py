# SPDX-License-Identifier: Apache-2.0
"""Select an FP16 native-MTP hot head without losing tensor parallelism.

Penny Royal's hot_vocab.py identifies the global-ID/local-row pitfall. Here
the assembled rows are sharded again, retaining TP4 projection parallelism.
"""

import torch


def build_sharded_hot_head(head, hot_ids, layer, group):
    """Return a separate padded TP shard; never mutate the target Parameter."""
    if head.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Native-MTP hot heads require unquantized floating weights")
    ids = hot_ids.detach().cpu().long()
    vocab_size = layer.org_vocab_size
    if ids.ndim != 1 or not ids.numel() or ids.unique().numel() != ids.numel():
        raise ValueError("The speculative token map must be a nonempty unique vector")
    if ids.min().item() < 0 or ids.max().item() >= vocab_size:
        raise ValueError("Speculative token IDs are outside the target vocabulary")
    if layer.tp_size != group.world_size:
        raise ValueError(
            "Hot-head selection requires matching target and draft TP groups"
        )
    if layer.num_embeddings != layer.org_vocab_size:
        raise ValueError("Native-MTP hot heads do not support added LoRA vocabulary")
    shard = layer.shard_indices
    owned = (ids >= shard.org_vocab_start_index) & (ids < shard.org_vocab_end_index)
    positions = owned.nonzero().flatten().to(head.device)
    local = (ids[owned] - shard.org_vocab_start_index).to(head.device)
    padded = ((ids.numel() + 64 * group.world_size - 1) // (64 * group.world_size)) * (
        64 * group.world_size
    )
    selected = torch.zeros(
        (padded, head.shape[1]), device=head.device, dtype=head.dtype
    )
    selected.index_copy_(0, positions, head.index_select(0, local))
    # Exactly one rank owns each row. Summing zero-filled contributions only
    # exchanges existing values; it does not average or requantize weights.
    if group.world_size > 1:
        selected = group.all_reduce(selected)
    width = padded // group.world_size
    start = group.rank_in_group * width
    result = selected[start : start + width].clone()
    return torch.nn.Parameter(result, requires_grad=False)
