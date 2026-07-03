# SPDX-License-Identifier: Apache-2.0
"""GPTQ int4 dense linear for Intel XPU via torch ``_weight_int4pack_mm``.

GPTQ stores ``W = (q - zp) * scale`` with codes packed 8-per-int32 sequentially:
``qweight`` packs along the K (input) dim, ``qzeros`` along the N (output) dim.
We deinterleave to the torch int4pack layout at load time. Two GPTQ subtleties:

* ``checkpoint_format`` v1 vs v2: v1 zero-points are stored off-by-one, so the
  effective zero-point is ``unpacked_qzero + 1`` (v2 uses it as-is).
* ``desc_act`` (act_order): input channels are permuted by activation order, so
  groups are non-contiguous in K. ``_weight_int4pack_mm`` requires contiguous
  grouping, so we sort the weight's K dim by ``g_idx`` at load time and gather
  the activation by the same permutation at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.hardware_backend.xpu.quantization.int4pack_utils import (
    SUPPORTED_GROUP_SIZES,
    build_qscale_and_zeros,
    pack_int4_to_uint8,
    xpu_int4pack_mm,
)
from sglang.srt.layers.quantization.utils import replace_parameter

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


def _nibble_shifts(device: torch.device) -> torch.Tensor:
    return torch.arange(0, 32, 4, device=device, dtype=torch.int32)


def _unpack_gptq_qweight(qweight: torch.Tensor) -> torch.Tensor:
    """``[K // 8, N]`` int32 packed along K -> ``[K, N]`` codes in ``[0, 15]``."""
    n = qweight.shape[1]
    shifts = _nibble_shifts(qweight.device)  # [8]
    # [K // 8, 8, N]; sub-index i selects k = row * 8 + i
    codes = (qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF
    return codes.reshape(-1, n)  # [K, N]


def _unpack_gptq_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    """``[K // gs, N // 8]`` int32 packed along N -> ``[K // gs, N]`` codes."""
    rows = qzeros.shape[0]
    shifts = _nibble_shifts(qzeros.device)  # [8]
    # [K // gs, N // 8, 8]; sub-index j selects n = col * 8 + j
    codes = (qzeros.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF
    return codes.reshape(rows, -1)  # [K // gs, N]


class GPTQXPULinearKernel:
    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        self.quant_config = quant_config
        self.use_v2_format = (
            getattr(quant_config, "checkpoint_format", "") == "gptq_v2"
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        group_size = self.quant_config.group_size
        if group_size not in SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"GPTQ on XPU requires group_size in {SUPPORTED_GROUP_SIZES}, "
                f"got {group_size}. torch _weight_int4pack_mm_xpu does not "
                "support this group size (per-channel/-1 is out of scope)."
            )

        qweight = layer.qweight.data  # [K // 8, N] int32
        qzeros = layer.qzeros.data  # [K // gs, N // 8] int32
        scales = layer.scales.data  # [K // gs, N]
        desc_act = bool(self.quant_config.desc_act)

        codes = _unpack_gptq_qweight(qweight)  # [K, N]
        k, n = codes.shape

        # qzeros -> [K // gs, N] effective zero-points (v1 is off-by-one).
        zp = _unpack_gptq_qzeros(qzeros).to(torch.int32)  # [num_groups, N]
        if not self.use_v2_format:
            zp = zp + 1

        act_perm = None
        if desc_act:
            g_idx = layer.g_idx.data
            if g_idx.numel() != k:
                raise ValueError(
                    "GPTQ act_order on XPU expects a per-channel g_idx of length "
                    f"K={k}, got {g_idx.numel()}."
                )
            # Sort K by group id so groups become contiguous gs-blocks.
            act_perm = torch.argsort(g_idx, stable=True).to(torch.int64)
            codes = codes[act_perm, :]
            sorted_g = g_idx[act_perm].to(torch.int64)
            # Each contiguous gs-block must belong to a single group, otherwise
            # this (typically row-parallel) shard splits a group across the K
            # boundary and torch's contiguous-group mm cannot represent it.
            blocks = sorted_g.view(-1, group_size)
            if not torch.equal(blocks, blocks[:, :1].expand_as(blocks)):
                raise ValueError(
                    "GPTQ act_order on XPU requires every group_size block of "
                    "input channels to map to a single group. This shard splits "
                    "a group across the K (row-parallel) boundary, which the "
                    "torch _weight_int4pack_mm path cannot represent."
                )
            # Reorder scales/zeros to follow the block group order.
            block_gid = blocks[:, 0]  # [num_blocks]
            scales = scales[block_gid]
            zp = zp[block_gid]

        codes = codes.t().contiguous()  # [N, K]
        qweight_packed = pack_int4_to_uint8(codes)  # [N, K // 2]

        qscale_and_zeros = build_qscale_and_zeros(scales, zp)  # [K // gs, N, 2]

        replace_parameter(layer, "qweight", qweight_packed)
        layer.register_parameter(
            "qscale_and_zeros",
            torch.nn.Parameter(qscale_and_zeros, requires_grad=False),
        )
        if act_perm is not None:
            layer.register_buffer(
                "xpu_act_perm", act_perm.to(qweight_packed.device), persistent=False
            )
        else:
            layer.xpu_act_perm = None
        # scales/qzeros folded into qscale_and_zeros; g_idx folded into act_perm.
        del layer.qzeros
        del layer.scales
        if hasattr(layer, "g_idx"):
            del layer.g_idx

        layer.xpu_out_features = n
        layer.xpu_group_size = group_size

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        act_perm = getattr(layer, "xpu_act_perm", None)
        if act_perm is not None:
            x = x.index_select(-1, act_perm)
        return xpu_int4pack_mm(
            x,
            layer.qweight,
            layer.xpu_group_size,
            layer.qscale_and_zeros,
            layer.xpu_out_features,
            bias,
        )
