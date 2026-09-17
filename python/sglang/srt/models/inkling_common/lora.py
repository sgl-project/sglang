from __future__ import annotations

import torch

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.dense.plan import DenseLoraKind
from sglang.srt.lora.utils import capturing_lora_graph
from sglang.srt.models.inkling_common.dense_mlp import InklingBatchDenseMLP
from sglang.srt.models.inkling_common.kernels.comm import symm_mem_all_reduce

Weights = torch.Tensor | dict[int, torch.Tensor]


def _pool_layout(weights: Weights, target_module: str, n: int, lora_a: bool) -> Weights:
    """Add the expert axis; gate/up A and down B are shared across experts."""
    if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
        return weights
    if (target_module == "gate_up_proj_moe") == lora_a:
        return weights.unsqueeze(0)
    dim = 1 if lora_a else 0
    width = weights.shape[dim]
    if width % n:
        raise ValueError(
            f"Shared-sink {target_module} LoRA factor extent {width} is not divisible "
            f"by the {n} experts"
        )
    weights = weights.unflatten(dim, (n, width // n))
    return weights.transpose(0, 1).contiguous() if lora_a else weights


def _moe_tp_shard(
    weights: Weights, target_module: str, tp_rank: int, shard: int
) -> Weights:
    """Shard down A columns or paired gate/up B rows."""
    start, end = tp_rank * shard, (tp_rank + 1) * shard

    def cut(weight: torch.Tensor) -> torch.Tensor:
        if target_module == "down_proj_moe":
            return weight[..., start:end].contiguous()
        full = weight.shape[-2] // 2
        return torch.cat(
            [weight[..., start:end, :], weight[..., full + start : full + end, :]],
            dim=-2,
        ).contiguous()

    if isinstance(weights, dict):
        return {expert_id: cut(weight) for expert_id, weight in weights.items()}
    return cut(weights)


class InklingBatchDenseMLPWithLoRA(InklingBatchDenseMLP):
    """Experimental-path LoRA for the sink's interleaved gate/up layout."""

    is_shared_fused_moe = True

    def initialize_lora(self, lora_backend: BaseLoRABackend) -> None:
        problems = []
        if lora_backend.max_loras_per_batch > 1 and getattr(
            lora_backend, "name", None
        ) not in ("triton", "triton_v2"):
            problems.append("multi-slot dense LoRA requires a Triton backend")
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
        slots, n, two_f, rank = self.gate_up_lora_b_weights.shape
        _, _, _, f = self.down_lora_a_weights.shape
        expected = {"_w1_delta": (slots, n * two_f, 2 * rank)}
        if self.experts_shared_outer_loras:
            expected["_a_cat"] = (slots, rank, n * f)
        for name, shape in expected.items():
            current = getattr(self, name)
            if current is None:
                setattr(self, name, self.gate_up_lora_b_weights.new_empty(shape))
            elif tuple(current.shape) != shape:
                raise RuntimeError(
                    "Shared-sink LoRA pool shape changed after initialization: "
                    f"{name} {tuple(current.shape)} -> {shape}"
                )

    def on_lora_slots_updated(self, slot_ids: set[int] | None) -> None:
        self._refresh_lora_operands(slot_ids)

    def _refresh_lora_operands(self, slot_ids: set[int] | None = None) -> None:
        if not self.set_lora or self._w1_delta is None:
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
            # Gate/up outputs interleave, but their rank blocks are separate.
            gate_up = self._w1_delta.view(slots, n, f, 2, 2 * rank)
            for slot in slot_ids:
                gate_up[slot].zero_()
                gate_up[slot, :, :, 0, :rank].copy_(b_gate_up[slot, :, :f, :])
                gate_up[slot, :, :, 1, rank:].copy_(b_gate_up[slot, :, f:, :])
            if self.experts_shared_outer_loras:
                a_cat = self._a_cat.view(slots, rank, n, a_down.shape[3])
                for slot in slot_ids:
                    a_cat[slot].copy_(a_down[slot].permute(1, 0, 2))

    def slice_moe_lora_a_weights(
        self, weights: Weights, tp_rank: int, target_module: str
    ) -> Weights:
        weights = _pool_layout(weights, target_module, self.n_shared_experts, True)
        if self.moe_tp_size <= 1 or target_module != "down_proj_moe":
            return weights
        return _moe_tp_shard(
            weights, target_module, tp_rank, self.intermediate_size_per_partition
        )

    def slice_moe_lora_b_weights(
        self, weights: Weights, tp_rank: int, target_module: str
    ) -> Weights:
        weights = _pool_layout(weights, target_module, self.n_shared_experts, False)
        if self.moe_tp_size <= 1 or target_module != "gate_up_proj_moe":
            return weights
        return _moe_tp_shard(
            weights, target_module, tp_rank, self.intermediate_size_per_partition
        )

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
            self, x_td, gammas_ts, linearized_weights, use_reduce_scatter
        )


class InklingBatchDenseMLPWithLoRAV2(InklingBatchDenseMLP):
    """Dense-engine LoRA for Inkling's shared-expert sink."""

    is_shared_fused_moe = True

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
            lora_ranks=torch.full((slots,), rank, dtype=torch.int32, device=device),
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
                lora_ranks=torch.full((slots,), rank, dtype=torch.int32, device=device),
                scalings=scalings,
            )
        else:
            # Each A block reads one expert's activation window.
            down = dict(
                kind=DenseLoraKind.SINK_DOWN,
                a=down_lora_a_weights.view(slots, n * rank, -1),
                b=down_cat,
                offsets=(0, hidden),
                lora_ranks=torch.full(
                    (slots,), n * rank, dtype=torch.int32, device=device
                ),
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

    def slice_moe_lora_a_weights(
        self, weights: Weights, tp_rank: int, target_module: str
    ) -> Weights:
        weights = _pool_layout(weights, target_module, self.n_shared_experts, True)
        if self.moe_tp_size <= 1 or target_module != "down_proj_moe":
            return weights
        return _moe_tp_shard(
            weights, target_module, tp_rank, self.intermediate_size_per_partition
        )

    def slice_moe_lora_b_weights(
        self, weights: Weights, tp_rank: int, target_module: str
    ) -> Weights:
        weights = _pool_layout(weights, target_module, self.n_shared_experts, False)
        if self.moe_tp_size <= 1 or target_module != "gate_up_proj_moe":
            return weights
        return _moe_tp_shard(
            weights, target_module, tp_rank, self.intermediate_size_per_partition
        )

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
