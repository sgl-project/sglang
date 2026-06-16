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
        # We still use the kernel for packing utilities and dispatcher dtype, but not for process_weights
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
        """
        Unpack AWQ int4 weight to signed int8 (E*K, N).
        packed: (E, K, N_packed) int32
        returns: (E*K, N) int8 (signed two's complement)
        """
        E, K, N_packed = packed.shape
        pack_factor = 8
        N = N_packed * pack_factor
        shifts = [0, 4, 1, 5, 2, 6, 3, 7]

        flat = packed.flatten(0, 1)          # (E*K, N_packed)
        out = torch.zeros((E * K, N), dtype=torch.int8, device=packed.device)

        for i, s in enumerate(shifts):
            nib = (flat >> (s * 4)) & 0xF      # unsigned 0..15
            # sign‑extend: values >= 8 are negative
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
        Uses npu_convert_weight_to_int4pack which expects int32 input of shape (E*N, K).
        """
        E, N, K = weight_int8.shape
        # The operator expects (M, K) int32 tensor where each int32 holds 8 int4 values.
        # Reshape to (E*N, K) and cast to int32 (it will be reinterpreted as packed int4)
        weight_int32 = weight_int8.reshape(E * N, K).to(torch.int32)
        # Convert to int4pack format (shape becomes (E*N, K//8))
        packed = torch.ops.npu.npu_convert_weight_to_int4pack(weight_int32)
        # Reshape back to (E, N, K//8)
        packed = packed.view(E, N, -1)
        return packed.contiguous()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # 1. Convert weights and register
        # w13
        w13_unpacked = self._unpack_awq_int4(layer.w13_qweight.data)  # (E*K, N13) int8
        E, K13, N13_packed = layer.w13_qweight.shape
        N13 = N13_packed * 8
        w13_int8 = w13_unpacked.view(E, K13, N13).permute(0, 2, 1).contiguous()  # (E, N13, K13)
        w13_packed = self._pack_for_w4a16(w13_int8)   # (E, N13, K13//8) int32
        self._register_or_replace_parameter(layer, "w13_weight", w13_packed)

        # w2
        w2_unpacked = self._unpack_awq_int4(layer.w2_qweight.data)
        E2, K2, N2_packed = layer.w2_qweight.shape
        N2 = N2_packed * 8
        w2_int8 = w2_unpacked.view(E2, K2, N2).permute(0, 2, 1).contiguous()
        w2_packed = self._pack_for_w4a16(w2_int8)
        self._register_or_replace_parameter(layer, "w2_weight", w2_packed)

        # 2. Scales – transpose from (E, groups, N) to (E, N, groups) and keep as is
        self._register_or_replace_parameter(
            layer, "w13_weight_scale",
            layer.w13_scales.data.transpose(1, 2).contiguous()
        )
        self._register_or_replace_parameter(
            layer, "w2_weight_scale",
            layer.w2_scales.data.transpose(1, 2).contiguous()
        )

        # 3. Offsets – correctly computed from zeros
        for w_prefix in ("w13", "w2"):
            qzeros = getattr(layer, f"{w_prefix}_qzeros").data   # (E, groups, N//8)
            unpacked_zeros = self._unpack_awq_int4(qzeros)       # (E*groups, N) int8, value = zero_point - 8
            E_z, groups, _ = qzeros.shape
            N_z = unpacked_zeros.shape[1]
            offset = unpacked_zeros.view(E_z, groups, N_z).permute(0, 2, 1).contiguous()  # (E, N, groups)
            offset = -offset  # kernel expects 8 - zero_point
            dtype = getattr(layer, f"{w_prefix}_scales").data.dtype
            self._register_or_replace_parameter(
                layer, f"{w_prefix}_weight_offset", offset.to(dtype)
            )

        # 4. Set dispatcher output dtype for w13
        self.w13_kernel._set_dispatcher_output_dtype(layer, "bf16")  # this only sets a config, safe

        # 5. Free original AWQ tensors
        for attr in ("w13_qweight", "w13_qzeros", "w13_scales",
                     "w2_qweight", "w2_qzeros", "w2_scales"):
            if hasattr(layer, attr):
                delattr(layer, attr)
