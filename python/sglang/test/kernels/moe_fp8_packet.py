"""Unfused Torch wire-format reference for tests and microbenchmarks only."""

import torch


def pack_reference(q, scale, weights, ids):
    rows, hidden = q.shape

    def row_bytes(tensor):
        flat = tensor.reshape(-1).contiguous()
        if flat.stride(-1) != 1:
            flat = torch.empty(tensor.numel(), device=tensor.device, dtype=tensor.dtype)
            flat.copy_(tensor.reshape(-1))
        return flat.view(torch.uint8).reshape(
            rows, tensor.shape[1] * tensor.element_size()
        )

    pieces = [q.view(torch.uint8), row_bytes(scale), row_bytes(weights), row_bytes(ids)]
    padding = (-(hidden + scale.shape[1] * 4 + weights.shape[1] * 8)) % 16
    if padding:
        pieces.append(q.new_zeros((rows, padding), dtype=torch.uint8))
    return torch.cat(pieces, dim=1)


def unpack_reference(payload, hidden, topk):
    scale_end = hidden + hidden // 128 * 4
    weights_end = scale_end + topk * 4
    q = payload[:, :hidden].contiguous().view(torch.float8_e4m3fn)
    scale = payload[:, hidden:scale_end].contiguous().view(torch.float32)
    scale = scale.t().contiguous().t()
    weights = payload[:, scale_end:weights_end].contiguous().view(torch.float32)
    ids = (
        payload[:, weights_end : weights_end + topk * 4].contiguous().view(torch.int32)
    )
    return q, scale, weights, ids
