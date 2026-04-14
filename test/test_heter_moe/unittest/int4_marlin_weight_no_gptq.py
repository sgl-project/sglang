"""Simple BF16 → INT4 Marlin weight conversion (no GPTQ calibration).

Uses the proven marlin_quantize path from sglang's test utils, which
does round-to-nearest quantization → Marlin weight permutation → packing.
This is the same path used by test_marlin_moe.py.
"""

import torch


def quantize_expert_weight(
    weight_nk: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one expert's [N, K] BF16 weight to Marlin-packed INT4.

    Returns:
        marlin_qweight: Marlin-format packed INT32
        marlin_scales:  Marlin-permuted scales
    """
    from sglang.srt.layers.quantization.utils import scalar_types
    from sglang.test.test_marlin_utils import marlin_quantize

    N, K = weight_nk.shape

    # marlin_quantize expects [K, N] float on CPU
    w_kn = weight_nk.t().contiguous().float()

    w_ref, marlin_qw, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
        w_kn,
        quant_type=scalar_types.uint4b8,
        group_size=group_size,
        act_order=False,
    )

    return marlin_qw, marlin_s


def fill_int4_params(layer, bf16_w13, bf16_w2, group_size):
    """Quantize BF16 expert weights to Marlin INT4 and fill layer params.

    Writes directly into the layer's Marlin-format parameters. Caller
    should NOT call repack_int4_to_marlin() after this — weights are
    already in Marlin format.
    """
    import torch.nn as nn

    E = bf16_w13.shape[0]

    qw13_list, sc13_list = [], []
    qw2_list, sc2_list = [], []

    for e in range(E):
        qw13, sc13 = quantize_expert_weight(bf16_w13[e], group_size)
        qw13_list.append(qw13)
        sc13_list.append(sc13)

        qw2, sc2 = quantize_expert_weight(bf16_w2[e], group_size)
        qw2_list.append(qw2)
        sc2_list.append(sc2)

    # Stack into [E, ...] and replace layer params directly
    device = bf16_w13.device
    dtype = bf16_w13.dtype

    layer.int4_w13_qweight = nn.Parameter(
        torch.stack(qw13_list).to(device), requires_grad=False)
    layer.int4_w2_qweight = nn.Parameter(
        torch.stack(qw2_list).to(device), requires_grad=False)
    layer.int4_w13_scales = nn.Parameter(
        torch.stack(sc13_list).to(device).to(dtype), requires_grad=False)
    layer.int4_w2_scales = nn.Parameter(
        torch.stack(sc2_list).to(device).to(dtype), requires_grad=False)

    # Empty g_idx / sort_indices (no act_order)
    layer.int4_w13_g_idx = nn.Parameter(
        torch.empty((E, 0), dtype=torch.int32, device=device), requires_grad=False)
    layer.int4_w2_g_idx = nn.Parameter(
        torch.empty((E, 0), dtype=torch.int32, device=device), requires_grad=False)
    layer.int4_w13_g_idx_sort_indices = nn.Parameter(
        torch.empty((E, 0), dtype=torch.int32, device=device), requires_grad=False)
    layer.int4_w2_g_idx_sort_indices = nn.Parameter(
        torch.empty((E, 0), dtype=torch.int32, device=device), requires_grad=False)

    layer._int4_is_k_full = True
    layer._int4_group_size = group_size
