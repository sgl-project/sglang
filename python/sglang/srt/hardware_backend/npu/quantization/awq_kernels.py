from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.moe_runner.torch_npu import (
    TorchNpuQuantInfo,
)
from sglang.srt.layers.quantization.utils import replace_parameter

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A16Int4MoEMethod,
)
import torch_npu
import torch.nn.functional as F


class AWQAscendLinearKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Keep scales as (groups, N) – NPU kernel expects this layout
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)
    
        raw_qweight = layer.qweight.data   # (K, N // pack_factor)
        raw_qzeros  = layer.qzeros.data    # (groups, N // pack_factor)
    
        pack_factor = self.quant_config.pack_factor
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]
    
        K = raw_qweight.shape[0]
        N = raw_qweight.shape[1] * pack_factor
        num_groups = layer.scales.shape[0]
    
        if K % num_groups != 0:
            raise RuntimeError(f"K={K} not divisible by scale groups {num_groups}")
        group_size = K // num_groups
    
        # NPU constraint
        npu_ok = False #(group_size == 0) or (group_size % 32 == 0 and 32 <= group_size < K)
    
        if npu_ok:
            # ----- NPU path (unsigned weight + raw zero point) -----
            # 1) Pack weight as unsigned nibbles (NO XOR)
            qweight_tmp = torch.zeros_like(raw_qweight)
            qzeros_list = []
            for i in range(pack_factor):
                shift_num = shifts[i] * 4
                qzeros_list.append((raw_qzeros.reshape(-1, 1) >> shift_num) & 0xF)
                qweight_tmp.bitwise_or_(
                    ((layer.qweight.data >> shift_num) & 0xF) << (4 * i)
                )

            qweight_tmp.bitwise_xor_(0x88888888)
        
            qzeros_tmp = torch.cat(qzeros_list, dim=-1).reshape(raw_qzeros.shape[0], -1)
            qzeros_tmp = -(qzeros_tmp - 8)
            qzeros_tmp = qzeros_tmp.to(layer.scales.data.dtype)
    
            layer.zeros = torch.nn.Parameter(qzeros_tmp, requires_grad=False)
            layer.weight = torch.nn.Parameter(qweight_tmp, requires_grad=False)
            
            layer.use_npu_matmul = True
            layer.npu_group_size = group_size
        else:
            # ----- Fallback: asymmetric dequantisation -----
            weight_u8 = torch.zeros((K, N), dtype=torch.int8, device=raw_qweight.device)
            zeros_u8 = torch.zeros((num_groups, N), dtype=torch.int8, device=raw_qzeros.device)
    
            for i in range(pack_factor):
                shift = shifts[i] * 4
                nib_w = (raw_qweight >> shift) & 0xF
                weight_u8[:, i::pack_factor] = nib_w.to(torch.int8)
                nib_z = (raw_qzeros >> shift) & 0xF
                zeros_u8[:, i::pack_factor] = nib_z.to(torch.int8)
    
            if group_size > 0:
                zeros_exp = zeros_u8.repeat_interleave(group_size, dim=0)
                scales_exp = layer.scales.data.repeat_interleave(group_size, dim=0)
            else:
                zeros_exp = zeros_u8
                scales_exp = layer.scales.data
    
            weight_float = (weight_u8.float() - zeros_exp.float()) * scales_exp.float()
            weight_float = weight_float.t().contiguous().to(torch.bfloat16)
    
            layer.register_parameter("weight", torch.nn.Parameter(weight_float, requires_grad=False))
            delattr(layer, "scales")
            layer.use_npu_matmul = False
    
        # Clean original packed tensors
        for attr in ("qweight", "qzeros"):
            if hasattr(layer, attr):
                delattr(layer, attr)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        pack_factor = self.quant_config.pack_factor
    
        if layer.use_npu_matmul:
            qweight = layer.weight          # (K, N//pack) int32, unsigned
            scales = layer.scales           # (groups, N)
            offset = layer.zeros            # (groups, N) raw zero point
        
            out_shape = x.shape[:-1] + (qweight.shape[1] * pack_factor,)
            if bias is not None and bias.dtype == torch.bfloat16:
                bias = bias.float()
        
            out = torch_npu.npu_weight_quant_batchmatmul(
                reshaped_x,
                qweight,
                antiquant_scale=scales,
                antiquant_offset=offset,           # raw zero point
                antiquant_group_size=layer.npu_group_size,
                bias=bias,
            )
            return out.reshape(out_shape)
        else:
            return F.linear(x, layer.weight, bias)


class AWQAscendMoEKernel:
    def __init__(self, quant_config: Optional[QuantizationConfig] = None, use_unquantized: bool = True):
        self.quant_config = quant_config
        self.use_unquantized = use_unquantized
        if not use_unquantized:
            self.w13_kernel = NPUW4A16Int4MoEMethod()
            self.w2_kernel = NPUW4A16Int4MoEMethod()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Capture original AWQ tensors (before any modification)
        w13_qweight_orig = layer.w13_qweight.data
        w13_qzeros_orig  = layer.w13_qzeros.data
        w13_scales_orig  = layer.w13_scales.data

        w2_qweight_orig = layer.w2_qweight.data
        w2_qzeros_orig  = layer.w2_qzeros.data
        w2_scales_orig  = layer.w2_scales.data

        # Packing function (quantized path)
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]
        pack_factor = self.quant_config.pack_factor

        def pack_one(qweight, qzeros):
            qweight = qweight.to(torch.int32)
            qzeros = qzeros.to(torch.int32)
            qweight_tmp = torch.zeros_like(qweight)
            qzeros_list = []
            for i in range(pack_factor):
                shift_num = shifts[i] * 4
                qzeros_list.append((qzeros.reshape(-1, 1) >> shift_num) & 0xF)
                qweight_tmp.bitwise_or_(
                    ((qweight >> shift_num) * (2 ** (4 * i))) & (0xF << (4 * i))
                )
            qweight_tmp.bitwise_xor_(0x88888888)
            qzeros_tmp = torch.cat(qzeros_list, dim=-1).reshape(qzeros.shape[0], qzeros.shape[1], -1)
            qzeros_tmp = -(qzeros_tmp - 8)
            qzeros_tmp = qzeros_tmp.to(qweight.device)
            return qweight_tmp, qzeros_tmp

        w13_qweight_packed, w13_qzeros_packed = pack_one(w13_qweight_orig, w13_qzeros_orig)
        w2_qweight_packed, w2_qzeros_packed = pack_one(w2_qweight_orig, w2_qzeros_orig)

        # Cast offsets to scale dtype
        w13_qzeros_packed = w13_qzeros_packed.to(w13_scales_orig.dtype)
        w2_qzeros_packed   = w2_qzeros_packed.to(w2_scales_orig.dtype)

        if self.use_unquantized:
            # Dequantize using the original tensors (before they are overwritten)
            self._dequantize_and_store(
                layer,
                prefix="w13",
                qweight=w13_qweight_orig,
                qzeros=w13_qzeros_orig,
                scales=w13_scales_orig,
            )
            self._dequantize_and_store(
                layer,
                prefix="w2",
                qweight=w2_qweight_orig,
                qzeros=w2_qzeros_orig,
                scales=w2_scales_orig,
            )
            # Delete original AWQ tensors and packed versions to free memory
            del w13_qweight_orig, w13_qzeros_orig, w13_scales_orig
            del w2_qweight_orig, w2_qzeros_orig, w2_scales_orig
            del w13_qweight_packed, w13_qzeros_packed
            del w2_qweight_packed, w2_qzeros_packed
            # Remove scales from the layer (they are no longer needed)
            if hasattr(layer, "w13_scales"):
                delattr(layer, "w13_scales")
            if hasattr(layer, "w2_scales"):
                delattr(layer, "w2_scales")
            # Remove original packed tensors if they still exist (they were not overwritten)
            if hasattr(layer, "w13_qweight"):
                delattr(layer, "w13_qweight")
            if hasattr(layer, "w13_qzeros"):
                delattr(layer, "w13_qzeros")
            if hasattr(layer, "w2_qweight"):
                delattr(layer, "w2_qweight")
            if hasattr(layer, "w2_qzeros"):
                delattr(layer, "w2_qzeros")
        else:
            # Quantized path: register the packed versions
            self._register_or_replace_parameter(layer, "w13_qweight", w13_qweight_packed)
            self._register_or_replace_parameter(layer, "w13_qzeros", w13_qzeros_packed)
            self._register_or_replace_parameter(layer, "w2_qweight", w2_qweight_packed)
            self._register_or_replace_parameter(layer, "w2_qzeros", w2_qzeros_packed)
            # Scales remain unchanged
            del w13_qweight_orig, w13_qzeros_orig, w13_scales_orig
            del w2_qweight_orig, w2_qzeros_orig, w2_scales_orig

        torch.npu.empty_cache()
        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})

    def _dequantize_and_store(self, layer, prefix, qweight, qzeros, scales):
        """
        Dequantize one expert group (w13 or w2) and store FP weights as
        `prefix_weight_fp` on the layer.
        """
        # Ensure integer types for bitwise operations
        qweight = qweight.to(torch.int32)
        qzeros = qzeros.to(torch.int32)
    
        E, K, N_packed = qweight.shape
        N = N_packed * 8
        groups = scales.shape[1]
        group_size = K // groups
        assert K % groups == 0, f"K={K} not divisible by groups={groups}"
    
        # Allocate as (E, K, N) – we will transpose to (E, N, K) later
        fp_weight = torch.empty((E, K, N), dtype=torch.bfloat16, device=qweight.device)
    
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]
    
        for e in range(E):
            # Unpack weight
            qw_e = qweight[e]                         # (K, N_packed)
            w_int8 = torch.empty((K, N), dtype=torch.int8, device=qweight.device)
            for i, s in enumerate(shifts):
                nib = (qw_e >> (s * 4)) & 0xF
                signed = torch.where(nib >= 8, nib - 16, nib).to(torch.int8)
                w_int8[:, i::8] = signed
    
            # Unpack zeros
            qz_e = qzeros[e]                          # (groups, N_packed)
            z_int8 = torch.empty((groups, N), dtype=torch.int8, device=qweight.device)
            for i, s in enumerate(shifts):
                nib = (qz_e >> (s * 4)) & 0xF
                signed = torch.where(nib >= 8, nib - 16, nib).to(torch.int8)
                z_int8[:, i::8] = signed
    
            # Expand scales and zeros to (K, N)
            z_exp = z_int8.repeat_interleave(group_size, dim=0)   # (K, N)
            s_exp = scales[e].repeat_interleave(group_size, dim=0) # (K, N)
    
            # Dequantize: (w - z) * scale, result shape (K, N)
            w_fp = ((w_int8.float() - z_exp.float()) * s_exp.float()).to(torch.bfloat16)
            fp_weight[e] = w_fp  # assign directly – shape (K, N) matches
    
        # Transpose to (E, N, K) for batched matmul
        fp_weight = fp_weight.transpose(-1, -2).contiguous()
        setattr(layer, f"{prefix}_qweight", torch.nn.Parameter(fp_weight, requires_grad=False))

    def _register_or_replace_parameter(self, layer, name, tensor):
        if hasattr(layer, name):
            replace_parameter(layer, name, tensor)
        else:
            layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
