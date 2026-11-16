from __future__ import annotations

import torch

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.dense.plan import DenseLoraKind
from sglang.srt.lora.utils import capturing_lora_graph
from sglang.srt.models.inkling_common.dense_mlp import InklingBatchDenseMLP
from sglang.srt.models.inkling_common.kernels.comm import symm_mem_all_reduce


class InklingBatchDenseMLPWithLoRA(InklingBatchDenseMLP):
    """LoRA layer for Inkling's dense shared-expert sink."""

    set_lora = False
    is_shared_fused_moe = False

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
        self.is_shared_fused_moe = True
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


class InklingBatchDenseMLPWithLoRAV2(InklingBatchDenseMLP):
    """Dense-engine LoRA for Inkling's shared-expert sink."""

    set_lora = False
    is_shared_fused_moe = False

    def initialize_lora(self, lora_backend: BaseLoRABackend) -> None:
        if getattr(lora_backend, "runner", None) is None:
            raise ValueError(
                f"{type(self).__name__} needs a backend with the dense engine"
            )
        if not self._linearized_bf16_enabled:
            raise ValueError(
                f"{type(self).__name__} needs the linearized BF16 shared sink"
            )
        if not self._w13_gate_up_contiguous:
            raise ValueError(
                "The V2 shared sink must select contiguous W13 before loading"
            )
        self.lora_backend = lora_backend
        self.set_lora = False
        self.register_buffer("_down_cat", None, persistent=False)
        self.is_shared_fused_moe = True
        lora_backend.is_moe_lora = True

    def set_lora_info(
        self,
        gate_up_lora_a_weights: torch.Tensor,
        gate_up_lora_b_weights: torch.Tensor,
        down_lora_a_weights: torch.Tensor,
        down_lora_b_weights: torch.Tensor,
    ) -> None:
        n = self.n_shared_experts
        slots, outer, two_rank, _ = gate_up_lora_a_weights.shape
        _, _, two_f, rank = gate_up_lora_b_weights.shape
        hidden = down_lora_b_weights.shape[2]
        if (
            outer not in (1, n)
            or down_lora_b_weights.shape[1] != outer
            or gate_up_lora_b_weights.shape[1] != n
            or down_lora_a_weights.shape[1] != n
            or two_rank != 2 * rank
            or down_lora_a_weights.shape[2] != rank
            or down_lora_b_weights.shape[-1] != rank
        ):
            raise ValueError(
                "Inkling shared-sink LoRA pool buffers do not agree: expected gate-up A "
                f"[slots, 1 or {n}, 2R, H], gate-up B [slots, {n}, 2F, R], down A "
                f"[slots, {n}, R, F], down B [slots, 1 or {n}, H, R] with one layout"
            )
        shared = outer == 1
        shape = (slots, rank, n * (two_f // 2)) if shared else (slots, hidden, n * rank)
        down_cat = self._down_cat
        if down_cat is None:
            down_cat = down_lora_a_weights.new_empty(shape)
        elif tuple(down_cat.shape) != shape:
            raise RuntimeError(
                "Shared-sink LoRA pool shape changed after initialization: "
                f"{tuple(down_cat.shape)} -> {shape}"
            )
        device = gate_up_lora_b_weights.device
        # Pool factors are max-rank padded and already scaled.
        ranks = torch.full((slots,), rank, dtype=torch.int32, device=device)
        scalings = torch.ones(slots, dtype=torch.float32, device=device)

        gate_up = dict(
            kind=DenseLoraKind.SINK_GATE_UP,
            a=(
                gate_up_lora_a_weights[:, 0]
                if shared
                else gate_up_lora_a_weights.reshape(slots, n * 2 * rank, -1)
            ),
            b=gate_up_lora_b_weights.view(slots, n * two_f, rank),
            offsets=tuple(range(0, n * two_f + 1, two_f // 2)),
            lora_ranks=ranks,
            scalings=scalings,
            a_blocks=2 if shared else 2 * n,
            bridge_slices=2 if shared else 0,
        )
        if shared:
            down = dict(
                kind=DenseLoraKind.LINEAR,
                a=down_cat,
                b=down_lora_b_weights[:, 0],
                offsets=(0, hidden),
                lora_ranks=ranks,
                scalings=scalings,
            )
        else:
            # Each A block reads one expert's activation window.
            down = dict(
                kind=DenseLoraKind.SINK_DOWN,
                a=down_lora_a_weights.view(slots, n * rank, -1),
                b=down_cat,
                offsets=(0, hidden),
                lora_ranks=ranks * n,
                scalings=scalings,
                a_blocks=n,
                a_windowed=True,
            )
        source = (down_lora_a_weights if shared else down_lora_b_weights).permute(
            0, 2, 1, 3
        )
        self._down_refresh = source, down_cat.view_as(source)
        self._down_cat = down_cat
        self._gate_up, self._down = gate_up, down
        self.set_lora = True
        self.on_lora_slots_updated()

    def on_lora_slots_updated(self, slot_ids: set[int] | None = None) -> None:
        if not self.set_lora:
            return
        source, target = self._down_refresh
        # Keep the derived buffer's address stable across graph replays.
        with torch.no_grad():
            for slot in range(source.shape[0]) if slot_ids is None else slot_ids:
                target[slot].copy_(source[slot])

    # Both wrappers receive the same pool factors and TP slicing geometry.
    slice_moe_lora_a_weights = InklingBatchDenseMLPWithLoRA.slice_moe_lora_a_weights
    slice_moe_lora_b_weights = InklingBatchDenseMLPWithLoRA.slice_moe_lora_b_weights
    _slice_down_lora_a = InklingBatchDenseMLPWithLoRA._slice_down_lora_a
    _slice_gate_up_lora_b = InklingBatchDenseMLPWithLoRA._slice_gate_up_lora_b

    def _forward_bf16_linearized(
        self,
        x_td: torch.Tensor,
        gammas_ts: torch.Tensor,
        linearized_weights: tuple[torch.Tensor, torch.Tensor],
        use_reduce_scatter: bool,
    ) -> torch.Tensor:
        # Capture LoRA even without an active adapter, for later replay batches
        # (except the decode runner's LoRA-free "nolora" graph).
        if not self.set_lora or not (
            getattr(self.lora_backend.batch_info, "has_active_lora", False)
            or capturing_lora_graph()
        ):
            return super()._forward_bf16_linearized(
                x_td, gammas_ts, linearized_weights, use_reduce_scatter
            )
        w13_lin, w2_lin = linearized_weights
        t, n = x_td.shape[0], self.n_shared_experts
        y = self._lora_gemm(x_td, lambda: torch.mm(x_td, w13_lin.T), **self._gate_up)
        act_flat = self._swiglu(y.view(t, n, -1), gammas_ts).reshape(t, -1)
        out_td = self._lora_gemm(
            act_flat, lambda: torch.mm(act_flat, w2_lin), **self._down
        )
        if not use_reduce_scatter and self.tp_group is not None:
            out_td = symm_mem_all_reduce(out_td, self.tp_group)
        return out_td

    def _lora_gemm(self, x: torch.Tensor, base_fn, *, kind, **site) -> torch.Tensor:
        runner = self.lora_backend.runner
        plan = runner.plan_for(
            kind,
            site["b"].shape[-1],
            x.shape[-1],
            site["offsets"][-1],
            num_tokens=x.shape[0],
        )
        return runner.apply(x, base_fn, plan, **site)
