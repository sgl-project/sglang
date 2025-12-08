from __future__ import annotations

from typing import Optional
import torch

from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    permute_param_layout_,
)

from sglang.srt.layers.quantization.marlin_utils import (
    unpack_quantized_values_into_int32,
    pack_quantized_values_into_int32,
)
from sgl_kernel.gemm import machete_mm

from sglang.srt.utils import is_cuda
from sglang.srt.layers.quantization.gptq import GPTQKernel

_is_cuda = is_cuda()
if _is_cuda:
    from sgl_kernel import permute_cols


class MacheteLinearKernel(GPTQKernel):

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        c = self.kernel_config

        self.w_q_name = "qweight"
        self.w_s_name = "scales"
        self.w_zp_name = "qzeros"
        self.w_gidx_name = "g_idx"

        if c.has_g_idx:
            assert self.w_gidx_name is not None
            perm = torch.argsort(getattr(layer, self.w_gidx_name))\
                .to(torch.int)

            self.act_perm = lambda x: x[:, perm]
            # use `ops.permute_cols` if possible
            if c.act_type in [torch.float16, torch.bfloat16] \
                and c.partition_weight_shape[0] % 8 == 0:
                #self.act_perm = partial(ops.permute_cols, perm=perm)
                self.act_perm = lambda x: permute_cols(x, perm=perm)

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            if c.has_g_idx:
                x_unpacked = unpack_quantized_values_into_int32(x.data,
                                                                c.weight_type,
                                                                packed_dim=0)
                x_perm = x_unpacked[perm, :]
                x.data = pack_quantized_values_into_int32(x_perm,
                                                          c.weight_type,
                                                          packed_dim=0)
            x.data = torch.ops.sgl_kernel.machete_prepack_B(x.data.t().contiguous().t(),
                                           a_type=c.act_type,
                                           b_type=c.weight_type.id,
                                           group_scales_type=c.act_type)
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x

        # Repack weights and scales for Machete
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        if c.zero_points:
            return False, "Zero points currently not supported by Machete !"


    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c = self.kernel_config

        w_q, w_s, _, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1], )

        if c.has_g_idx:
            x_2d = self.act_perm(x_2d)

        output = machete_mm(a=x_2d,
                                b_q=w_q,
                                b_type=c.weight_type,
                                b_group_zeros=None,
                                b_group_scales=w_s,
                                b_group_size=c.group_size)

        if bias is not None:
            output.add_(bias)  # In-place add

        return output.reshape(out_shape)
