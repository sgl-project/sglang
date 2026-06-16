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
    def __init__(self, quant_config: Optional[QuantizationConfig] = None):
        self.quant_config = quant_config
        self.w13_kernel = NPUW4A16Int4MoEMethod()
        self.w2_kernel = NPUW4A16Int4MoEMethod()

    @staticmethod
    def _register_or_replace_parameter(layer, name, tensor):
        if hasattr(layer, name):
            replace_parameter(layer, name, tensor)
        else:
            layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))

    # ------------------------------------------------------------------
    #  AWQ interleaved unpacking (0,4,1,5,2,6,3,7) with sign-extension
    # ------------------------------------------------------------------
    def _unpack_awq_int4(self, packed: torch.Tensor) -> torch.Tensor:
        E, K, N_packed = packed.shape
        pack_factor = 8
        N = N_packed * pack_factor
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]

        flat = packed.flatten(0, 1)          # (E*K, N_packed)
        out = torch.zeros((E * K, N), dtype=torch.int8, device=packed.device)

        for i, s in enumerate(shifts):
            nib = (flat >> (s * 4)) & 0xF
            signed = torch.where(nib >= 8, nib - 16, nib).to(torch.int8)
            out[:, i::pack_factor] = signed
        return out

    # ------------------------------------------------------------------
    #  Weight packing for W4A16 NPU kernel
    # ------------------------------------------------------------------
    def _pack_for_w4a16(self, weight_int8: torch.Tensor) -> torch.Tensor:
        """
        Convert signed int8 weight (shape E,N,K) into the packed int4 layout
        required by torch.ops.npu.npu_grouped_matmul (W4A16).
        """
        E, N, K = weight_int8.shape
        # npu_convert_weight_to_int4pack expects int32 input (M, K)
        weight_int32 = weight_int8.reshape(E * N, K).to(torch.int32)
        packed = torch.ops.npu.npu_convert_weight_to_int4pack(weight_int32)  # (E*N, K//8)
        packed = packed.view(E, N, -1)
        return packed.contiguous()

    # ------------------------------------------------------------------
    #  Full conversion from AWQ packed weight to NPU packed weight
    # ------------------------------------------------------------------
    def _convert_awq_weight_to_npu_layout(self, qweight: torch.Tensor) -> torch.Tensor:
        E, K, _ = qweight.shape
        # Unpack to signed int8 (E*K, N)
        unpacked = self._unpack_awq_int4(qweight)
        N = unpacked.shape[1]
        # Reshape to (E, K, N) and transpose to (E, N, K)
        weight_int8 = unpacked.view(E, K, N).permute(0, 2, 1).contiguous()  # (E, N, K)
        # Pack for W4A16
        return self._pack_for_w4a16(weight_int8)

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Helper to expand per-group tensors to per-channel (K)
        def expand_to_k(tensor: torch.Tensor, groups: int, group_size: int, K: int, N: int) -> torch.Tensor:
            # tensor shape: (E, groups, N)
            expanded = tensor.unsqueeze(2).expand(-1, -1, group_size, -1)  # (E, groups, group_size, N)
            return expanded.reshape(-1, K, N)  # (E, K, N)
    
        for prefix in ("w13", "w2"):
            # 1. Unpack weight to get K and N
            qweight = getattr(layer, f"{prefix}_qweight").data  # (E, K, N_packed)
            unpacked = self._unpack_awq_int4(qweight)           # (E*K, N)
            E, K = qweight.shape[0], qweight.shape[1]
            N = unpacked.shape[1]
            weight_int8 = unpacked.view(E, K, N).permute(0, 2, 1).contiguous()  # (E, N, K)
    
            # 2. Pack weight for NPU
            packed_weight = self._pack_for_w4a16(weight_int8)   # (E, N, K//8)
            self._register_or_replace_parameter(layer, f"{prefix}_weight", packed_weight)
    
            # 3. Process scales: expand from (E, groups, N) to (E, N, K)
            scales = getattr(layer, f"{prefix}_scales").data    # (E, groups, N)
            groups = scales.shape[1]
            group_size = K // groups
            assert K % groups == 0, f"K={K} not divisible by groups={groups}"
            scales_expanded = scales.unsqueeze(2).expand(-1, -1, group_size, -1).reshape(E, K, N)  # (E, K, N)
            scales_final = scales_expanded.transpose(1, 2).contiguous()  # (E, N, K)
            self._register_or_replace_parameter(layer, f"{prefix}_weight_scale", scales_final)
    
            # 4. Process offsets: unpack qzeros, expand, negate (8 - zero), transpose
            qzeros = getattr(layer, f"{prefix}_qzeros").data    # (E, groups, N_packed)
            unpacked_zeros = self._unpack_awq_int4(qzeros)      # (E*groups, N)
            offset_int8 = unpacked_zeros.view(E, groups, N)     # (zero - 8)
            offset_expanded = offset_int8.unsqueeze(2).expand(-1, -1, group_size, -1).reshape(E, K, N)  # (E, K, N)
            offset_final = -offset_expanded                     # 8 - zero
            offset_final = offset_final.transpose(1, 2).contiguous()  # (E, N, K)
            offset_final = offset_final.to(scales.dtype)        # cast to scale dtype
            self._register_or_replace_parameter(layer, f"{prefix}_weight_offset", offset_final)
    
            # Free original AWQ tensors
            for attr in (f"{prefix}_qweight", f"{prefix}_qzeros", f"{prefix}_scales"):
                if hasattr(layer, attr):
                    delattr(layer, attr)
    
        # 5. Set dispatcher output dtype for w13
        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})
