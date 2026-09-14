# SPDX-License-Identifier: Apache-2.0
"""GGUF dense weights plus streamed GGUF-MXFP4 routed experts."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from sglang.kernels.ops.moe.expert_pack_mxfp4 import (
    mxfp4_matvec,
    mxfp4_matvec_dual,
)
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe.expert_pack import (
    ExpertPackStore,
    KimiGGMLExpertPackStore,
)
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.gguf import (
    GGUFConfig,
    GGUFEmbeddingMethod,
    GGUFLinearMethod,
)


def _clamped_swiglu(
    gate: torch.Tensor, up: torch.Tensor, limit: float | None
) -> torch.Tensor:
    if limit is not None:
        if gate.is_cuda:
            from sglang.kernels.ops.attention.dsv4 import silu_and_mul_clamp

            gate_up = torch.cat((gate, up), dim=-1)
            output = torch.empty_like(gate)
            silu_and_mul_clamp(gate_up, output, float(limit))
            return output
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class ExpertPackConfig(GGUFConfig):
    """Use regular GGUF methods except for routed FusedMoE layers."""

    is_fp4_experts = True
    supports_kimi_k3_quantized_latent_projections = True
    supports_kimi_k3_split_gguf_kv_b = True

    def __init__(self, store: ExpertPackStore) -> None:
        super().__init__()
        self.store = store

    def get_name(self) -> str:
        return "expert_pack"

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

        if isinstance(layer, FusedMoE):
            return ExpertPackMoEMethod(self.store, prefix)
        if isinstance(layer, LinearBase):
            return GGUFLinearMethod(self)
        if isinstance(layer, VocabParallelEmbedding):
            return GGUFEmbeddingMethod(self)
        return None


class ExpertPackMoEMethod(FusedMoEMethodBase):
    def __init__(self, store: ExpertPackStore, prefix: str) -> None:
        self.store = store
        self.prefix = prefix
        self.layer_id: int | None = None
        self.hidden_size: int | None = None
        self.intermediate_size: int | None = None
        self.activation = "silu"
        self.swiglu_limit: float | None = None
        self.situ_beta: float | None = None
        self.situ_linear_beta: float | None = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del extra_weight_attrs
        if layer.num_fused_shared_experts:
            raise ValueError(
                "expert-pack requires --disable-shared-experts-fusion so the "
                "shared expert remains on the dense GGUF path"
            )
        if layer.moe_ep_size != 1 or layer.moe_tp_size != 1:
            raise ValueError("expert-pack v1 supports only single-GPU TP=EP=1")
        if num_experts != self.store.header.num_experts:
            raise ValueError("FusedMoE expert count does not match expert-pack")
        if params_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("expert-pack kernel requires BF16 or FP16 activations")
        gate_shape = self.store.entries[(layer.layer_id, 0, 0)].shape
        down_shape = self.store.entries[(layer.layer_id, 0, 2)].shape
        if gate_shape != (hidden_size, intermediate_size_per_partition):
            raise ValueError(
                f"expert-pack gate shape {gate_shape} does not match "
                f"{(hidden_size, intermediate_size_per_partition)}"
            )
        if down_shape != (intermediate_size_per_partition, hidden_size):
            raise ValueError(
                f"expert-pack down shape {down_shape} does not match "
                f"{(intermediate_size_per_partition, hidden_size)}"
            )
        self.layer_id = layer.layer_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size_per_partition

        # An empty, non-persistent marker makes the absence of eager expert
        # parameters visible in module/state audits without reserving VRAM.
        layer.register_buffer(
            "expert_pack_marker",
            torch.empty(0, dtype=torch.uint8),
            persistent=False,
        )

    def create_moe_runner(self, layer, moe_runner_config) -> None:
        del layer
        if isinstance(self.store, KimiGGMLExpertPackStore):
            if moe_runner_config.activation != "situ":
                raise ValueError("Kimi-K3 expert-pack requires SiTU experts")
            if (
                float(moe_runner_config.gemm1_alpha or 0.0),
                float(moe_runner_config.gemm1_clamp_limit or 0.0),
            ) != (4.0, 25.0):
                raise ValueError("Kimi-K3 SiTU constants must be exactly 4.0 and 25.0")
            self.situ_beta = 4.0
            self.situ_linear_beta = 25.0
        elif moe_runner_config.activation != "silu":
            raise ValueError("DeepSeek expert-pack requires SiLU experts")
        self.activation = moe_runner_config.activation
        self.swiglu_limit = moe_runner_config.swiglu_limit

    @staticmethod
    def _kimi_vec(
        inputs: torch.Tensor,
        weights: torch.Tensor,
        expert_ids: torch.Tensor,
        *,
        top_k: int,
        weight_type: int,
        output_size: int,
    ) -> torch.Tensor:
        # sgl_kernel.quantization is CUDA/MUSA-only; keep the import local so
        # this module stays importable on other devices (see gguf.py:44-70).
        from sgl_kernel.quantization import ggml_moe_a8_vec

        return ggml_moe_a8_vec(
            inputs,
            weights,
            expert_ids,
            top_k,
            weight_type,
            output_size,
            inputs.shape[0],
        )

    def _apply_kimi(self, hidden_states, topk_ids, topk_weights, slots):
        if self.intermediate_size is None or self.hidden_size is None:
            raise RuntimeError("Kimi-K3 expert-pack dimensions are unavailable")
        if self.situ_beta != 4.0 or self.situ_linear_beta != 25.0:
            raise RuntimeError("Kimi-K3 SiTU constants were not initialized")

        cache = self.store.device_cache
        top_k = topk_ids.shape[-1]
        role_types = {"gate": 10, "up": 10, "down": 11}
        row_bytes = {
            "gate": self.store.role_nbytes["gate"] // self.intermediate_size,
            "up": self.store.role_nbytes["up"] // self.intermediate_size,
            "down": self.store.role_nbytes["down"] // self.hidden_size,
        }

        gate_start = self.store.role_offsets["gate"]
        up_start = self.store.role_offsets["up"]
        down_start = self.store.role_offsets["down"]
        if hidden_states.shape[0] != 1:
            raise ValueError("Kimi expert-pack compact kernel expects one token")
        slot_indices = slots.long()
        compact_ids = torch.arange(
            top_k, dtype=torch.int32, device=hidden_states.device
        ).view(1, top_k)
        gate_weights = torch.index_select(
            cache[:, gate_start : gate_start + self.store.role_nbytes["gate"]],
            0,
            slot_indices,
        ).view(top_k, self.intermediate_size, row_bytes["gate"])
        up_weights = torch.index_select(
            cache[:, up_start : up_start + self.store.role_nbytes["up"]],
            0,
            slot_indices,
        ).view(top_k, self.intermediate_size, row_bytes["up"])
        gate = self._kimi_vec(
            hidden_states,
            gate_weights,
            compact_ids,
            top_k=top_k,
            weight_type=role_types["gate"],
            output_size=self.intermediate_size,
        )
        up = self._kimi_vec(
            hidden_states,
            up_weights,
            compact_ids,
            top_k=top_k,
            weight_type=role_types["up"],
            output_size=self.intermediate_size,
        )
        gate_fp32 = gate.float()
        gate = self.situ_beta * torch.tanh(gate_fp32 / self.situ_beta)
        gate = gate * torch.sigmoid(gate_fp32)
        up = self.situ_linear_beta * torch.tanh(up.float() / self.situ_linear_beta)
        activated = (gate * up).to(hidden_states.dtype)
        down_weights = torch.index_select(
            cache[:, down_start : down_start + self.store.role_nbytes["down"]],
            0,
            slot_indices,
        ).view(top_k, self.hidden_size, row_bytes["down"])
        down = self._kimi_vec(
            activated,
            down_weights,
            compact_ids.reshape(-1, 1),
            top_k=1,
            weight_type=role_types["down"],
            output_size=self.hidden_size,
        )
        down = down.view(hidden_states.shape[0], top_k, self.hidden_size)
        output = torch.zeros(
            (hidden_states.shape[0], self.hidden_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for route_index in range(top_k):
            output.add_(
                down[:, route_index].float()
                * topk_weights[:, route_index].float().unsqueeze(-1)
            )
        return output.to(hidden_states.dtype)

    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        if self.layer_id is None or self.hidden_size is None:
            raise RuntimeError("expert-pack MoE method was not initialized")
        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
        if topk_ids.shape[-1] != self.store.header.top_k:
            raise ValueError(
                f"runtime top-k {topk_ids.shape[-1]} does not match expert-pack "
                f"top-k {self.store.header.top_k}"
            )
        if hidden_states.shape[0] == 0:
            return StandardCombineInput(hidden_states=torch.empty_like(hidden_states))

        if isinstance(self.store, KimiGGMLExpertPackStore):
            token_outputs = []
            for token_index in range(topk_ids.shape[0]):
                slots, host_slots = self.store.acquire(
                    self.layer_id,
                    topk_ids[token_index : token_index + 1],
                    is_prefill=topk_ids.shape[0] > 1,
                )
                try:
                    token_outputs.append(
                        self._apply_kimi(
                            hidden_states[token_index : token_index + 1],
                            topk_ids[token_index : token_index + 1],
                            topk_weights[token_index : token_index + 1],
                            slots,
                        )
                    )
                finally:
                    self.store.mark_used(host_slots)
            return StandardCombineInput(hidden_states=torch.cat(token_outputs, dim=0))
        slots, host_slots = self.store.acquire(
            self.layer_id,
            topk_ids,
            is_prefill=topk_ids.shape[0] > 1,
        )
        cache = self.store.device_cache
        records_per_input = topk_ids.shape[-1]
        gate, up = mxfp4_matvec_dual(
            hidden_states,
            cache,
            slots,
            gate_role_offset=0,
            up_role_offset=self.store.role_bytes,
            role_bytes=self.store.role_bytes,
            input_size=self.hidden_size,
            output_size=self.intermediate_size,
            records_per_input=records_per_input,
        )
        intermediate = _clamped_swiglu(gate, up, self.swiglu_limit)
        down = mxfp4_matvec(
            intermediate,
            cache,
            slots,
            role_offset=2 * self.store.role_bytes,
            role_bytes=self.store.role_bytes,
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            records_per_input=1,
        )
        output = (
            down.view(hidden_states.shape[0], records_per_input, self.hidden_size)
            * topk_weights.unsqueeze(-1).to(down.dtype)
        ).sum(dim=1)
        self.store.mark_used(host_slots)
        return StandardCombineInput(hidden_states=output)
