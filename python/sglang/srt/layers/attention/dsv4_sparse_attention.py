import torch


def dsv4_sparse_attn(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    sinks: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    kv_len = key_states.shape[-2]
    if kv_len == 0:
        return query_states.new_zeros(
            query_states.shape[0],
            query_states.shape[-2],
            query_states.shape[1],
            value_states.shape[-1],
        )
    if key_states.shape[1] != query_states.shape[1]:
        if query_states.shape[1] % key_states.shape[1] != 0:
            raise ValueError(
                f"Query heads ({query_states.shape[1]}) must be divisible by KV heads ({key_states.shape[1]})"
            )
        repeat_factor = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_factor, dim=1)
        value_states = value_states.repeat_interleave(repeat_factor, dim=1)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    attn_weights = attn_weights * softmax_scale

    topk_idxs = topk_idxs.to(device=query_states.device, dtype=torch.long)
    valid_topk = (topk_idxs >= 0) & (topk_idxs < kv_len)
    clamped_topk_idxs = topk_idxs.clamp(min=0, max=max(kv_len - 1, 0))
    index_mask = torch.zeros(
        (*topk_idxs.shape[:-1], kv_len), dtype=torch.int32, device=query_states.device
    )
    index_mask.scatter_add_(-1, clamped_topk_idxs, valid_topk.to(torch.int32))
    index_mask = index_mask > 0
    attn_weights = attn_weights.masked_fill(
        ~index_mask.unsqueeze(1), torch.finfo(torch.float32).min
    )

    sinks = sinks.reshape(1, -1, 1, 1).expand(
        query_states.shape[0], -1, query_states.shape[-2], -1
    )
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = torch.nn.functional.softmax(
        combined_logits, dim=-1, dtype=combined_logits.dtype
    )
    scores = probs[..., :-1].to(value_states.dtype)
    attn_output = torch.matmul(scores, value_states)
    return attn_output.transpose(1, 2).contiguous()