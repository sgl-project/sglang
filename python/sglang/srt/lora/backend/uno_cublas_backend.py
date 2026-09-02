"""Single-adapter LoRA backend for UNO draft forwards.

LoRA-A is always overlapped with the base GEMM on an auxiliary CUDA stream.
Single-request batches operate only on draft rows; larger batches use one
``mm``/``addmm_`` over all rows and zero the seed-row LoRA hidden states
between the two GEMMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.lora.backend.triton_backend import TritonLoRABackend


@dataclass(frozen=True)
class _UnoSingleAdapterRoute:
    weight_index: int
    rank: int
    scaling: float
    batch_size: int
    forward_width: int
    total_rows: int
    active_rows: int

    @property
    def lora_rows(self) -> int:
        # For C=1, skipping the seed row makes both GEMMs smaller. For C>1,
        # one GEMM across C*F rows is faster than C tiny (F-1)-row GEMMs.
        return self.active_rows if self.batch_size == 1 else self.total_rows


@dataclass(frozen=True)
class _PendingLoRAA:
    x: torch.Tensor
    weights: torch.Tensor
    num_slices: int
    output: torch.Tensor
    producer_stream: torch.cuda.Stream


class UnoCublasLoRABackend(TritonLoRABackend):
    """Fast LoRA backend for UNO's draft forwards.
    Use cuBLAS and CUDA streams.

    Reuse multi-LoRA batch metadata from the Triton parent.
    """

    name = "uno_cublas"
    supports_lora_a_overlap = True
    # K2's runners prepare base-only LoRA metadata whenever any internal
    # manager exists. UNO never exposes request-selectable adapters, so those
    # prefill/warmup batches must stay on the plain base-model path.
    skip_inactive_lora_batches = True

    def __init__(
        self,
        max_loras_per_batch: int,
        device: torch.device,
        **kwargs,
    ):
        super().__init__(max_loras_per_batch, device, **kwargs)
        # Different CUDA-graph runners may capture and replay on different main
        # streams. Give each main stream its own LoRA-A side stream so concurrently
        # replayed graphs do not serialize or interfere through one shared stream.
        self._lora_a_streams: dict[torch.cuda.Stream, torch.cuda.Stream] = {}
        self._pending_lora_a: Optional[_PendingLoRAA] = None

    def reset_batch_state(self):
        self._pending_lora_a = None
        super().reset_batch_state()

    def validate_lora_targets(
        self,
        base_model: torch.nn.Module,
        target_modules: set[str],
    ) -> None:
        """Reject target layers that cannot honor UNO's token-row routing."""

        from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.utils import get_layer_id
        from sglang.srt.models.inkling_common.dense_mlp import InklingBatchDenseMLP

        unsupported: list[str] = []
        if "lm_head" in target_modules:
            lm_head = getattr(base_model, "lm_head", None)
            unsupported.append(f"lm_head ({type(lm_head).__name__})")
        if "embed_tokens" in target_modules:
            unsupported.append("embed_tokens (VocabParallelEmbedding)")

        supported = (ColumnParallelLinear, RowParallelLinear)
        target_moe = {"gate_up_proj", "down_proj"}.issubset(target_modules)
        for module_name, module in base_model.named_modules():
            parts = module_name.split(".")
            named_target = bool(parts) and (
                parts[-1] in target_modules or ".".join(parts[-2:]) in target_modules
            )
            special_moe_target = target_moe and isinstance(
                module, (FusedMoE, InklingBatchDenseMLP)
            )
            if not (named_target or special_moe_target):
                continue
            if get_layer_id(module_name) is None:
                continue
            if not isinstance(module, supported):
                unsupported.append(f"{module_name} ({type(module).__name__})")

        if unsupported:
            raise ValueError(
                "UNO's LoRA backend supports only ColumnParallelLinear and "
                "RowParallelLinear targets; unsupported target modules: "
                + ", ".join(sorted(set(unsupported)))
            )

    def prepare_lora_token_segments(
        self,
        *,
        segment_lens: list[int],
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
    ) -> None:
        super().prepare_lora_token_segments(
            segment_lens=segment_lens,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
        )

        route: Optional[_UnoSingleAdapterRoute] = None
        if (
            len(segment_lens) >= 2
            and len(segment_lens) % 2 == 0
            and len(weight_indices) == len(segment_lens)
            and segment_lens[0] == 1
            and segment_lens[1] > 0
        ):
            batch_size = len(segment_lens) // 2
            forward_width = segment_lens[1] + 1
            base_index, adapter_index = weight_indices[:2]
            adapter_rank = lora_ranks[adapter_index]
            if (
                base_index != adapter_index
                and lora_ranks[base_index] == 0
                and adapter_rank > 0
                and all(
                    segment_lens[2 * index] == 1
                    and segment_lens[2 * index + 1] == forward_width - 1
                    and weight_indices[2 * index] == base_index
                    and weight_indices[2 * index + 1] == adapter_index
                    for index in range(batch_size)
                )
            ):
                route = _UnoSingleAdapterRoute(
                    weight_index=adapter_index,
                    rank=adapter_rank,
                    scaling=float(scalings[adapter_index]),
                    batch_size=batch_size,
                    forward_width=forward_width,
                    total_rows=sum(segment_lens),
                    active_rows=batch_size * (forward_width - 1),
                )

        # Each CUDA-graph bucket retains its own batch_info. Store the immutable
        # UNO route there so switching back to a captured bucket restores the
        # route corresponding to that bucket, rather than using metadata last
        # written by another bucket.
        self.batch_info.uno_single_adapter_route = route

    def _route(self) -> _UnoSingleAdapterRoute:
        route = getattr(self.batch_info, "uno_single_adapter_route", None)
        if route is None:
            raise RuntimeError("UNO cuBLAS execution requires an active UNO route.")
        return route

    @staticmethod
    def _output_offsets(output_offset_cpu, output_offset) -> list[int]:
        offsets = output_offset_cpu if output_offset_cpu is not None else output_offset
        return [int(offset) for offset in offsets.tolist()]

    @staticmethod
    def _lora_a_input(
        x: torch.Tensor,
        route: _UnoSingleAdapterRoute,
    ) -> torch.Tensor:
        if route.batch_size == 1:
            return x[1:]
        return x

    def _compute_lora_a(
        self,
        lora_input: torch.Tensor,
        active_a: torch.Tensor,
        route: _UnoSingleAdapterRoute,
        *,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = torch.mm(lora_input, active_a.t(), out=output)
        if route.batch_size > 1:
            # LoRA must not affect each request's seed token. Zeroing C rows
            # is cheaper than masking every hidden element or running C tiny
            # strided-batched GEMMs.
            hidden.view(route.batch_size, route.forward_width, -1)[:, 0].zero_()
        return hidden

    def _accumulate_lora_b(
        self,
        *,
        hidden: torch.Tensor,
        active_b: torch.Tensor,
        base_output: torch.Tensor,
        output_start: int,
        output_end: int,
        route: _UnoSingleAdapterRoute,
    ) -> None:
        if route.batch_size == 1:
            base_output[1:, output_start:output_end].addmm_(
                hidden,
                active_b.t(),
                beta=1.0,
                alpha=route.scaling,
            )
            return

        base_output[:, output_start:output_end].addmm_(
            hidden,
            active_b.t(),
            beta=1.0,
            alpha=route.scaling,
        )

    def _run_lora_b(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        base_output: torch.Tensor,
    ) -> torch.Tensor:
        route = self._route()
        active_b = weights[route.weight_index, :, : route.rank]
        self._accumulate_lora_b(
            hidden=x[:, : route.rank],
            active_b=active_b,
            base_output=base_output,
            output_start=0,
            output_end=active_b.shape[0],
            route=route,
        )
        return base_output

    def run_lora_a_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        stack_num: int = 1,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self._consume_lora_a_overlap(
            x,
            weights,
            num_slices=stack_num,
        )

    def run_lora_b_sgemm(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        base_output: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self._run_lora_b(x, weights, base_output)

    def _run_stacked_lora(
        self,
        *,
        x: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        base_output: torch.Tensor,
        output_offset,
        output_offset_cpu,
        num_slices: int,
    ) -> torch.Tensor:
        route = self._route()
        hidden = self._consume_lora_a_overlap(
            x,
            lora_a,
            num_slices=num_slices,
        )
        offsets = self._output_offsets(output_offset_cpu, output_offset)
        for slice_index in range(num_slices):
            input_start = slice_index * route.rank
            input_end = input_start + route.rank
            output_start = offsets[slice_index]
            output_end = offsets[slice_index + 1]
            active_b = lora_b[
                route.weight_index,
                output_start:output_end,
                : route.rank,
            ]
            self._accumulate_lora_b(
                hidden=hidden[:, input_start:input_end],
                active_b=active_b,
                base_output=base_output,
                output_start=output_start,
                output_end=output_end,
                route=route,
            )
        return base_output

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: torch.Tensor,
        output_offset=None,
        output_offset_cpu=None,
        base_output: torch.Tensor = None,
        n_slices: int = 3,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self._run_stacked_lora(
            x=x,
            lora_a=qkv_lora_a,
            lora_b=qkv_lora_b,
            base_output=base_output,
            output_offset=output_offset,
            output_offset_cpu=output_offset_cpu,
            num_slices=n_slices,
        )

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: torch.Tensor,
        output_offset=None,
        output_offset_cpu=None,
        base_output: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self._run_stacked_lora(
            x=x,
            lora_a=gate_up_lora_a,
            lora_b=gate_up_lora_b,
            base_output=base_output,
            output_offset=output_offset,
            output_offset_cpu=output_offset_cpu,
            num_slices=2,
        )

    def start_lora_a_overlap(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        *,
        num_slices: int = 1,
    ) -> None:
        """Launch LoRA-A on the auxiliary stream before the base GEMM."""

        if self._pending_lora_a is not None:
            raise RuntimeError("Previous UNO LoRA-A overlap was not consumed.")

        route = self._route()
        out_dim = num_slices * route.rank
        lora_input = self._lora_a_input(x, route)
        active_a = weights[route.weight_index, :out_dim]

        main_stream = torch.cuda.current_stream(x.device)
        stream = self._lora_a_streams.get(main_stream)
        if stream is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "UNO LoRA overlap side stream was not created during "
                    "CUDA-graph warmup."
                )
            stream = torch.cuda.Stream(device=x.device)
            self._lora_a_streams[main_stream] = stream

        # Allocate on the main/consumer stream before handing the buffer to
        # the auxiliary stream. This keeps CUDA-graph allocator ownership and
        # the eventual LoRA-B consumer on the same stream.
        output = torch.empty(
            (route.lora_rows, out_dim),
            dtype=x.dtype,
            device=x.device,
        )
        stream.wait_stream(main_stream)
        with torch.cuda.stream(stream):
            self._compute_lora_a(lora_input, active_a, route, output=output)

        self._pending_lora_a = _PendingLoRAA(
            x=x,
            weights=weights,
            num_slices=num_slices,
            output=output,
            producer_stream=stream,
        )

    def _consume_lora_a_overlap(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        *,
        num_slices: int,
    ) -> torch.Tensor:
        pending = self._pending_lora_a
        if pending is None:
            raise RuntimeError("UNO LoRA-A overlap was not launched.")
        if (
            pending.x is not x
            or pending.weights is not weights
            or pending.num_slices != num_slices
        ):
            raise RuntimeError("UNO LoRA-A overlap was consumed by another layer.")

        self._pending_lora_a = None
        torch.cuda.current_stream(x.device).wait_stream(pending.producer_stream)
        return pending.output
