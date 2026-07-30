# SPDX-License-Identifier: Apache-2.0
"""Dynamic blockwise (128x128) fp8 linear method for the otherwise bf16 /
Quark-excluded GLM-5.1 ATTENTION PROJECTIONS (o_proj, q_b_proj, q_a_proj).

Env-gated by SGLANG_FP8_ATTN_PROJ=1 in quark.QuarkConfig. The GLM-5.1 MLA decode
path pre-quantizes o_proj / q_b_proj inputs with group_size=128 and
``transpose_scale=_use_aiter_bpreshuffle_gfx95`` (``fused_flatten_fp8_group_quant``
/ ``fused_rms_fp8_group_quant``), handing the linear a ``(qinput, x_scale)``
tuple destined for the aiter gfx95 ``gemm_a8w8_blockscale_bpreshuffle`` GEMM. To
match that GEMM we online-quantize the bf16 weight to 128x128-block fp8 with a
plain float32 [N/128, K/128] scale and bpreshuffle the fp8 weight via
``shuffle_weight(w, (16, 16))`` (validated cos>0.999 vs bf16 on a standalone
test). On the prefill path we quantize the raw bf16 activation with the same
per-1x128 + transpose_scale aiter quant. gfx950 uses OCP e4m3fn.

Only N and K divisible by 128 are supported (o_proj 6144x16384, q_b_proj
16384x2048, q_a_proj 2048x6144). kv_a_proj_with_mqa (N=576) and the fused
qkv_a_proj (N=2624) are not 128-aligned and stay bf16.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch.nn import Parameter

from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.utils import get_bool_env_var, is_hip

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

if _use_aiter:
    import aiter
    from aiter import gemm_a8w8_blockscale_bpreshuffle, get_hip_quant
    from aiter.ops.shuffle import shuffle_weight

    _fp8 = aiter.dtypes.fp8
    _finfo = torch.finfo(_fp8)
    _per1x128 = get_hip_quant(aiter.QuantType.per_1x128)

_BLK = 128


def _block128_weight_quant(w: torch.Tensor):
    # w [N, K] bf16 -> fp8 [N, K] + float32 scale [N/128, K/128].
    n, k = w.shape
    wv = w.float().view(n // _BLK, _BLK, k // _BLK, _BLK)
    amax = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(1e-6)
    scale = amax / _finfo.max
    wq = (wv / scale).clamp(_finfo.min, _finfo.max).to(_fp8).view(n, k)
    return wq.contiguous(), scale.view(n // _BLK, k // _BLK).float().contiguous()


class Fp8AttnProjBlockMethod(LinearMethodBase):
    def __init__(self):
        self.out_dtype = torch.get_default_dtype()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        n = sum(output_partition_sizes)
        k = input_size_per_partition
        if n % _BLK != 0 or k % _BLK != 0:
            raise ValueError(
                f"Fp8AttnProjBlockMethod requires N,K % 128 == 0, got N={n} K={k}"
            )
        layer.logical_widths = output_partition_sizes
        weight = ModelWeightParameter(
            data=torch.empty(n, k, dtype=params_dtype),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)
        layer.weight_scale_inv = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        wq, w_scale = _block128_weight_quant(layer.weight.data)
        wq = shuffle_weight(wq, layout=(16, 16))  # bpreshuffle layout for the GEMM
        layer.weight = Parameter(wq, requires_grad=False)
        layer.weight_scale_inv = Parameter(w_scale, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            # decode: model already produced (qinput_fp8, x_scale) (group 128,
            # transpose_scale=_use_aiter_bpreshuffle_gfx95).
            q_input, x_scale = x
            out_lead = q_input.shape[:-1]
        else:
            x2d = x.view(-1, x.shape[-1])
            q_input, x_scale = _per1x128(
                x2d, quant_dtype=_fp8, transpose_scale=True
            )
            out_lead = x.shape[:-1]
        output = gemm_a8w8_blockscale_bpreshuffle(
            q_input,
            layer.weight,
            x_scale,
            layer.weight_scale_inv,
            dtype=torch.bfloat16,
        )
        if bias is not None:
            output = output + bias
        return output.view(*out_lead, output.shape[-1])
