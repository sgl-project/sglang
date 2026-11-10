# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.xpu.quantization.int4pack_utils import (
    unpack_gptq_qweight,
    unpack_gptq_qzeros,
)
from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import MoeRunnerConfig

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput
    from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig

__all__ = ["GPTQIntelAMXLinearKernel", "GPTQIntelAMXMoEKernel"]


class GPTQIntelAMXLinearKernel:
    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _amx_process_weight_after_loading(
            layer, ["qweight", "qzeros", "scales"], None, "gptq"
        )
        layer.qweight = torch.nn.Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.ops.sgl_kernel.int4_scaled_mm_cpu(
            x,
            layer.qweight,
            layer.qzeros,
            layer.scales,
            bias,
        )


class GPTQIntelAMXMoEKernel:
    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config
        self.moe_runner_config: Optional[MoeRunnerConfig] = None

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    @staticmethod
    def _dequant_gptq_expert_weight(
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        use_v2_format: bool,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        # qweight: [K//8, N], qzeros/scales: [K//group_size, N]
        codes = unpack_gptq_qweight(qweight)  # [K, N], uint/int nibble codes
        zp = unpack_gptq_qzeros(qzeros).to(torch.int32)  # [K//gs, N]
        if not use_v2_format:
            zp = zp + 1

        k = codes.shape[0]
        if group_size == -1:
            group_size = k

        zp = zp.repeat_interleave(group_size, dim=0)[:k, :]
        scales = scales.repeat_interleave(group_size, dim=0)[:k, :]

        deq = (codes.to(torch.float32) - zp.to(torch.float32)) * scales.to(
            torch.float32
        )
        # F.linear expects [out_features, in_features]
        return deq.t().contiguous().to(out_dtype)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        runner_config = self.moe_runner_config or getattr(
            layer, "moe_runner_config", None
        )

        # Keep raw GPTQ tensors for relu2 fallback; the fused INT4 AMX kernel
        # still rejects relu2 and requires C++ changes we are not doing here.
        if runner_config is not None and runner_config.activation == "relu2":
            return

        _amx_process_weight_after_loading(
            layer, ["w13_qweight", "w13_qzeros", "w13_scales"], None, "gptq"
        )
        _amx_process_weight_after_loading(
            layer, ["w2_qweight", "w2_qzeros", "w2_scales"], None, "gptq"
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        assert self.moe_runner_config is not None
        moe_runner_config = self.moe_runner_config

        assert moe_runner_config.activation in ["silu", "relu2"], (
            f"Only SiLU/ReLU2 activations are supported on CPU INT4, "
            f"got {moe_runner_config.activation!r}."
        )
        if moe_runner_config.activation == "relu2":
            assert not moe_runner_config.is_gated, (
                "relu2 is only valid for non-gated MoE experts on CPU INT4."
            )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output
        x, topk_weights = apply_topk_weights_cpu(
            moe_runner_config.apply_router_weight_on_input, topk_weights, x
        )
        if moe_runner_config.activation == "relu2":
            # CPU fused INT4 MoE supports silu only. For relu2, run a Python
            # fallback on raw GPTQ tensors: dequantize selected experts and
            # evaluate relu2 MoE with torch GEMMs.
            out = x.new_zeros((x.shape[0], x.shape[1]))
            token_idx, slot_idx = torch.nonzero(topk_ids >= 0, as_tuple=True)
            if token_idx.numel() == 0:
                return StandardCombineInput(hidden_states=out)

            selected_experts = topk_ids[token_idx, slot_idx].to(torch.long)
            selected_weights = topk_weights[token_idx, slot_idx]
            group_size = int(getattr(self.quant_config, "group_size", -1))
            use_v2_format = (
                getattr(self.quant_config, "checkpoint_format", "") == "gptq_v2"
            )

            for expert_id in selected_experts.unique(sorted=False).tolist():
                expert_mask = selected_experts == expert_id
                if not torch.any(expert_mask):
                    continue

                rows = token_idx[expert_mask]
                row_weights = selected_weights[expert_mask].to(x.dtype).unsqueeze(-1)
                x_rows = x.index_select(0, rows)

                w13 = self._dequant_gptq_expert_weight(
                    layer.w13_qweight[expert_id],
                    layer.w13_qzeros[expert_id],
                    layer.w13_scales[expert_id],
                    group_size,
                    use_v2_format,
                    x.dtype,
                )
                w2 = self._dequant_gptq_expert_weight(
                    layer.w2_qweight[expert_id],
                    layer.w2_qzeros[expert_id],
                    layer.w2_scales[expert_id],
                    group_size,
                    use_v2_format,
                    x.dtype,
                )

                # FusedMoE stores w13 with two output shards even for this
                # non-gated model. Nemotron relu2 loads its sole up projection
                # into the w1 shard and intentionally leaves w3 unused.
                intermediate_size = layer.intermediate_size_per_partition
                w13 = w13[:intermediate_size]

                # The raw GPTQ down projection is loaded in transposed form:
                # dequantization yields [intermediate, hidden], whereas
                # F.linear requires [hidden, intermediate].
                if w2.shape == (intermediate_size, x.shape[-1]):
                    w2 = w2.t().contiguous()

                h = F.linear(x_rows, w13)
                h = torch.relu(h)
                h = h * h
                expert_out = F.linear(h, w2)

                out.index_add_(0, rows, (expert_out * row_weights).to(out.dtype))

            return StandardCombineInput(hidden_states=out)

        output = torch.ops.sgl_kernel.fused_experts_cpu(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights,
            topk_ids,
            False,  # inplace See [Note] inplace should be False in fused_experts.
            CPUQuantMethod.INT4_W4A8,
            layer.w13_scales,  # w1_scale
            layer.w2_scales,  # w2_scale
            layer.w13_qzeros,
            layer.w2_qzeros,
            None,  # block_size
            None,  # w1 bias
            None,  # w3 bias
            None,  # alpha
            None,  # limit
            True,  # is_vnni
            moe_runner_config.activation,  # activation
        )
        return StandardCombineInput(hidden_states=output)
