"""Load-time intermediate padding for native MXFP4 MoE weights."""

import torch


def pad_mxfp4_moe_intermediate(w13, w2, s13, s2, alignment=128):
    """Zero-pad both gate/up halves and down-projection input channels.

    Scales for zero weights reuse existing scales from the same expert, so
    padding does not change the exponent range used by Humming preprocessing.
    The gate/up order and all checkpoint bytes in the valid region are preserved.
    """
    num_experts, twice_intermediate, packed_hidden = w13.shape
    intermediate = twice_intermediate // 2
    padded = (intermediate + alignment - 1) // alignment * alignment
    if padded == intermediate:
        return w13, w2, s13, s2
    if intermediate % 32 or alignment % 32:
        raise ValueError("MXFP4 intermediate padding must preserve 32-element groups.")
    assert w2.shape == (num_experts, packed_hidden * 2, intermediate // 2)
    assert s13.shape == (num_experts, twice_intermediate, packed_hidden // 16)
    assert s2.shape == (num_experts, packed_hidden * 2, intermediate // 32)

    padded_w13 = w13.new_zeros(num_experts, 2, padded, packed_hidden)
    padded_w13[:, :, :intermediate].copy_(
        w13.reshape(num_experts, 2, intermediate, packed_hidden)
    )
    padded_w2 = w2.new_zeros(num_experts, packed_hidden * 2, padded // 2)
    padded_w2[..., : intermediate // 2].copy_(w2)

    # Byte views support native E8M0 copies without arithmetic on float8.
    s13_data = s13.view(torch.uint8) if s13.dtype == torch.float8_e8m0fnu else s13
    s2_data = s2.view(torch.uint8) if s2.dtype == torch.float8_e8m0fnu else s2
    s13_halves = s13_data.reshape(num_experts, 2, intermediate, -1)
    padded_s13 = s13_data.new_empty(num_experts, 2, padded, s13.shape[-1])
    padded_s13[:, :, :intermediate].copy_(s13_halves)
    padded_s13[:, :, intermediate:].copy_(s13_halves[:, :, :1])
    padded_s2 = s2_data.new_empty(num_experts, packed_hidden * 2, padded // 32)
    padded_s2[..., : intermediate // 32].copy_(s2_data)
    padded_s2[..., intermediate // 32 :].copy_(s2_data[..., -1:])

    return (
        padded_w13.reshape(num_experts, 2 * padded, packed_hidden),
        padded_w2,
        padded_s13.reshape(num_experts, 2 * padded, -1).view(s13.dtype),
        padded_s2.view(s2.dtype),
    )
