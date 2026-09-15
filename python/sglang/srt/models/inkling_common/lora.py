from __future__ import annotations

import torch

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.models.inkling_common.dense_mlp import InklingBatchDenseMLP


class InklingBatchDenseMLPWithLoRA(InklingBatchDenseMLP):
    """LoRA layer for Inkling's dense shared-expert sink."""

    is_shared_fused_moe = True

    def initialize_lora(self, lora_backend: BaseLoRABackend) -> None:
        problems = []
        if (
            lora_backend.max_loras_per_batch > 1
            and getattr(lora_backend, "name", None) != "triton"
        ):
            problems.append("multi-slot dense LoRA requires the Triton backend")
        if not self._linearized_bf16_enabled:
            problems.append("the shared sink does not use linearized BF16 weights")
        if problems:
            raise ValueError(
                "InklingBatchDenseMLPWithLoRA is ineligible: " + "; ".join(problems)
            )

        self.lora_backend = lora_backend
        self.set_lora = False
        self.experts_shared_outer_loras = False
        self.register_buffer("_w1_delta", None, persistent=False)
        self.register_buffer("_a_cat", None, persistent=False)
        self._lora_routing_cache = {}
        lora_backend.is_moe_lora = True

    def set_lora_info(
        self,
        gate_up_lora_a_weights: torch.Tensor,
        gate_up_lora_b_weights: torch.Tensor,
        down_lora_a_weights: torch.Tensor,
        down_lora_b_weights: torch.Tensor,
    ) -> None:
        tensors = (
            gate_up_lora_a_weights,
            gate_up_lora_b_weights,
            down_lora_a_weights,
            down_lora_b_weights,
        )
        if any(weight.ndim != 4 for weight in tensors):
            raise ValueError("Inkling shared-sink LoRA requires four 4D MoE buffers")
        gate_outer = gate_up_lora_a_weights.shape[1]
        down_outer = down_lora_b_weights.shape[1]
        valid_outer_dims = (1, self.n_shared_experts)
        if gate_outer not in valid_outer_dims or down_outer not in valid_outer_dims:
            raise ValueError(
                "Inkling shared-sink LoRA outer factors must have expert dimension "
                f"1 or {self.n_shared_experts}"
            )
        if gate_outer != down_outer:
            raise ValueError(
                "Inkling shared-sink gate-up A and down B must use the same "
                "expert layout"
            )
        if (
            gate_up_lora_b_weights.shape[1] != self.n_shared_experts
            or down_lora_a_weights.shape[1] != self.n_shared_experts
        ):
            raise ValueError("Inkling shared-sink LoRA expert count does not match")

        max_rank = gate_up_lora_b_weights.shape[-1]
        if (
            gate_up_lora_a_weights.shape[2] != 2 * max_rank
            or down_lora_a_weights.shape[2] != max_rank
            or down_lora_b_weights.shape[-1] != max_rank
        ):
            raise ValueError("Inkling shared-sink LoRA rank dimensions do not match")

        self.set_lora = True
        self.gate_up_lora_a_weights = gate_up_lora_a_weights
        self.gate_up_lora_b_weights = gate_up_lora_b_weights
        self.down_lora_a_weights = down_lora_a_weights
        self.down_lora_b_weights = down_lora_b_weights
        self.experts_shared_outer_loras = gate_outer == 1
        self._allocate_lora_operands()
        self._refresh_lora_operands()

    def _allocate_lora_operands(self) -> None:
        if not self.experts_shared_outer_loras:
            self._w1_delta = None
            self._a_cat = None
            return
        slots, n, two_f, rank = self.gate_up_lora_b_weights.shape
        _, _, _, f = self.down_lora_a_weights.shape
        expected_gate_up = (slots, n * two_f, 2 * rank)
        expected_down = (slots, rank, n * f)
        if self._w1_delta is None:
            self._w1_delta = self.gate_up_lora_b_weights.new_empty(expected_gate_up)
            self._a_cat = self.down_lora_a_weights.new_empty(expected_down)
            return
        if (
            tuple(self._w1_delta.shape) != expected_gate_up
            or tuple(self._a_cat.shape) != expected_down
        ):
            raise RuntimeError(
                "Shared-sink LoRA pool shape changed after initialization: "
                f"gate-up {tuple(self._w1_delta.shape)} -> {expected_gate_up}, "
                f"down-A {tuple(self._a_cat.shape)} -> {expected_down}"
            )

    def on_lora_slots_updated(self, slot_ids: set[int] | None) -> None:
        self._refresh_lora_operands(slot_ids)

    def _refresh_lora_operands(self, slot_ids: set[int] | None = None) -> None:
        if not self.set_lora or self._w1_delta is None or self._a_cat is None:
            return
        b_gate_up = self.gate_up_lora_b_weights
        a_down = self.down_lora_a_weights
        slots, n, two_f, rank = b_gate_up.shape
        f = two_f // 2
        if slot_ids is None:
            slot_ids = set(range(slots))
        elif any(slot < 0 or slot >= slots for slot in slot_ids):
            raise IndexError(f"Shared-sink LoRA slot out of range: {sorted(slot_ids)}")
        with torch.no_grad():
            gate_up = self._w1_delta.view(slots, n, f, 2, 2 * rank)
            a_cat = self._a_cat.view(slots, rank, n, a_down.shape[3])
            for slot in slot_ids:
                gate_up[slot].zero_()
                gate_up[slot, :, :, 0, :rank].copy_(b_gate_up[slot, :, :f, :])
                gate_up[slot, :, :, 1, rank:].copy_(b_gate_up[slot, :, f:, :])
                a_cat[slot].copy_(a_down[slot].permute(1, 0, 2))

    def slice_moe_lora_a_weights(
        self,
        weights: torch.Tensor | dict[int, torch.Tensor],
        tp_rank: int,
        target_module: str,
    ) -> torch.Tensor | dict[int, torch.Tensor]:
        if isinstance(weights, torch.Tensor) and weights.dim() == 2:
            if target_module == "gate_up_proj_moe":
                weights = weights.unsqueeze(0)
            else:
                rank, flat_intermediate = weights.shape
                if flat_intermediate % self.n_shared_experts != 0:
                    raise ValueError(
                        "Shared-sink down LoRA-A width must be divisible by "
                        f"{self.n_shared_experts}, got {flat_intermediate}"
                    )
                weights = (
                    weights.view(
                        rank,
                        self.n_shared_experts,
                        flat_intermediate // self.n_shared_experts,
                    )
                    .transpose(0, 1)
                    .contiguous()
                )
        if self.moe_tp_size <= 1 or target_module != "down_proj_moe":
            return weights
        if isinstance(weights, dict):
            return {
                expert_id: self._slice_down_lora_a(weight, tp_rank)
                for expert_id, weight in weights.items()
            }
        return self._slice_down_lora_a(weights, tp_rank)

    def _slice_down_lora_a(self, weights: torch.Tensor, tp_rank: int) -> torch.Tensor:
        start = tp_rank * self.intermediate_size_per_partition
        end = start + self.intermediate_size_per_partition
        return weights[..., start:end].contiguous()

    def slice_moe_lora_b_weights(
        self,
        weights: torch.Tensor | dict[int, torch.Tensor],
        tp_rank: int,
        target_module: str,
    ) -> torch.Tensor | dict[int, torch.Tensor]:
        if isinstance(weights, torch.Tensor) and weights.dim() == 2:
            if target_module == "down_proj_moe":
                weights = weights.unsqueeze(0)
            else:
                flat_intermediate, rank = weights.shape
                if flat_intermediate % self.n_shared_experts != 0:
                    raise ValueError(
                        "Shared-sink gate/up LoRA-B height must be divisible by "
                        f"{self.n_shared_experts}, got {flat_intermediate}"
                    )
                weights = weights.view(
                    self.n_shared_experts,
                    flat_intermediate // self.n_shared_experts,
                    rank,
                )
        if self.moe_tp_size <= 1 or target_module != "gate_up_proj_moe":
            return weights
        if isinstance(weights, dict):
            return {
                expert_id: self._slice_gate_up_lora_b(weight, tp_rank)
                for expert_id, weight in weights.items()
            }
        if weights.dim() == 3:
            return torch.stack(
                [
                    self._slice_gate_up_lora_b(weights[i], tp_rank)
                    for i in range(weights.shape[0])
                ]
            )
        return self._slice_gate_up_lora_b(weights, tp_rank)

    def _slice_gate_up_lora_b(
        self, weights: torch.Tensor, tp_rank: int
    ) -> torch.Tensor:
        shard = self.intermediate_size_per_partition
        start = tp_rank * shard
        end = start + shard
        full_intermediate = weights.shape[0] // 2
        gate = weights[start:end]
        up = weights[full_intermediate + start : full_intermediate + end]
        return torch.cat([gate, up], dim=0).contiguous()

    def _forward_bf16_linearized(
        self,
        x_td: torch.Tensor,
        gammas_ts: torch.Tensor,
        linearized_weights: tuple[torch.Tensor, torch.Tensor],
        use_reduce_scatter: bool,
    ) -> torch.Tensor:
        if not self.set_lora:
            return super()._forward_bf16_linearized(
                x_td,
                gammas_ts,
                linearized_weights,
                use_reduce_scatter,
            )

        from sglang.srt.lora.trtllm_lora_temp.inkling_dense import forward_with_lora

        return forward_with_lora(
            self,
            x_td,
            gammas_ts,
            linearized_weights,
            use_reduce_scatter,
        )
