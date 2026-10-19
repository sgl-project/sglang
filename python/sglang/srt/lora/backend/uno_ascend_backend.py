"""Graph-safe single-adapter GEMMs for UNO on Ascend.

The SGEMMV expand kernel supports ranks 8/16/32/64 only. UNO's published
Qwen3 adapter has rank 128. Use torch's NPU matmul dispatch instead, retaining
the full rank and handling fused projections explicitly. Like UNO's CUDA
GEMM path, the intermediate uses the model dtype and scaling is applied by B.
"""

from dataclasses import dataclass

import torch

from sglang.srt.lora.backend.ascend_backend import AscendLoRABackend


@dataclass(frozen=True)
class _UnoAscendRoute:
    weight_index: int
    rank: int
    scaling: float
    batch_size: int
    forward_width: int

    @property
    def total_rows(self):
        return self.batch_size * self.forward_width


class UnoAscendLoRABackend(AscendLoRABackend):
    name = "uno_ascend"
    supports_dp_attention = False
    skip_inactive_lora_batches = True

    def prepare_lora_token_segments(
        self, *, segment_lens, weight_indices, lora_ranks, scalings
    ):
        if (
            not segment_lens
            or len(segment_lens) % 2
            or len(weight_indices) != len(segment_lens)
            or len(scalings) != len(lora_ranks)
            or any(index < 0 or index >= len(lora_ranks) for index in weight_indices)
        ):
            raise ValueError("UNO Ascend requires paired base/adapter token segments.")
        base, adapter = weight_indices[:2]
        width = segment_lens[1] + 1
        if (
            width <= 1
            or base == adapter
            or lora_ranks[base] != 0
            or lora_ranks[adapter] <= 0
            or any(
                segment_lens[i : i + 2] != [1, width - 1]
                or weight_indices[i : i + 2] != [base, adapter]
                for i in range(0, len(segment_lens), 2)
            )
        ):
            raise ValueError("UNO Ascend requires one base row then F-1 adapter rows.")
        super().prepare_lora_token_segments(
            segment_lens=segment_lens,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
        )
        # Retained by UnoCudaGraphLoRAState for each captured batch bucket.
        self.batch_info.uno_ascend_route = _UnoAscendRoute(
            weight_index=adapter,
            rank=lora_ranks[adapter],
            scaling=float(scalings[adapter]),
            batch_size=len(segment_lens) // 2,
            forward_width=width,
        )

    def _route(self, rows):
        route = getattr(self.batch_info, "uno_ascend_route", None)
        if route is None or rows != route.total_rows:
            raise RuntimeError("UNO Ascend LoRA input does not match its token route.")
        return route

    def _shrink(self, x, weights, slices):
        route = self._route(x.shape[0])
        if (
            weights.ndim != 3
            or route.weight_index >= weights.shape[0]
            or weights.shape[1] < slices * route.rank
            or weights.shape[2] != x.shape[1]
        ):
            raise ValueError("Invalid UNO Ascend LoRA-A weight geometry.")
        hidden = torch.mm(x, weights[route.weight_index, : slices * route.rank].t())
        # One GEMM across all requests; the carried seed must remain base-only.
        hidden.view(route.batch_size, route.forward_width, -1)[:, 0].zero_()
        return hidden

    def _expand(self, x, weights, base_output, offsets):
        route = self._route(x.shape[0])
        slices = len(offsets) - 1
        if (
            weights.ndim != 3
            or route.weight_index >= weights.shape[0]
            or weights.shape[2] < route.rank
            or x.shape[1] != slices * route.rank
            or offsets[0] != 0
            or offsets[-1] != weights.shape[1]
            or any(end <= start for start, end in zip(offsets, offsets[1:]))
        ):
            raise ValueError("Invalid UNO Ascend LoRA-B weight geometry.")
        if base_output is None:
            base_output = x.new_zeros((x.shape[0], weights.shape[1]))
        if base_output.shape != (x.shape[0], weights.shape[1]):
            raise ValueError("UNO Ascend LoRA output shape mismatch.")
        for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
            hidden = x[:, i * route.rank : (i + 1) * route.rank].contiguous()
            result = torch.addmm(
                base_output[:, start:end],
                hidden,
                weights[route.weight_index, start:end, : route.rank].t(),
                beta=1.0,
                alpha=route.scaling,
            )
            base_output[:, start:end].copy_(result)
        return base_output

    @staticmethod
    def _offsets(output_offset_cpu):
        if output_offset_cpu is None or output_offset_cpu.device.type != "cpu":
            raise ValueError("UNO Ascend requires CPU output offsets (no device sync).")
        return output_offset_cpu.tolist()

    def run_lora_a_sgemm(self, x, weights, *args, stack_num=1, **kwargs):
        return self._shrink(x, weights, stack_num)

    def run_lora_b_sgemm(
        self, x, weights, output_offset=None, base_output=None, *args, **kwargs
    ):
        return self._expand(x, weights, base_output, [0, weights.shape[1]])

    def run_qkv_lora(
        self,
        x,
        qkv_lora_a,
        qkv_lora_b,
        output_offset,
        base_output=None,
        n_slices=3,
        *args,
        output_offset_cpu=None,
        **kwargs,
    ):
        offsets = self._offsets(output_offset_cpu)
        if len(offsets) != n_slices + 1:
            raise ValueError("UNO Ascend QKV offsets do not match its slice count.")
        return self._expand(
            self._shrink(x, qkv_lora_a, n_slices),
            qkv_lora_b,
            base_output,
            offsets,
        )

    def run_gate_up_lora(
        self,
        x,
        gate_up_lora_a,
        gate_up_lora_b,
        output_offset,
        base_output=None,
        *args,
        output_offset_cpu=None,
        **kwargs,
    ):
        offsets = self._offsets(output_offset_cpu)
        if len(offsets) != 3:
            raise ValueError("UNO Ascend gate/up requires two output slices.")
        return self._expand(
            self._shrink(x, gate_up_lora_a, 2),
            gate_up_lora_b,
            base_output,
            offsets,
        )
