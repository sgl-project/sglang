"""The contiguous row domain: all routed rows in one compact 2-D buffer.

:mod:`masked_row_domain` gives each expert its own ``[m_max, ·]`` slab. This
module instead sorts the routed rows by expert into a single buffer. Each
expert's segment starts on a multiple of the DeepGEMM m-alignment. The buffer
then holds about ``num_tokens * top_k`` rows, not ``num_experts * m_max``.

Use this domain for prefill only. A decode port of it measured slower than the
masked domain on GB300, so decode keeps the masked domain.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProvider,
)
from sglang.srt.lora.moe.kernels.activation_delta import (
    act_delta_contiguous,
)
from sglang.srt.lora.moe.kernels.dispatch import (
    ContiguousSchedulePack,
    contiguous_m_pad_ceiling,
    dispatch_fill_contiguous,
)
from sglang.srt.lora.moe.kernels.fused_act import (
    fused_b_act_contiguous,
)
from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.moe.route_view import RouteView
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


class ContiguousRowState(msgspec.Struct, kw_only=True):
    hidden_compact: torch.Tensor  # [m_pad_ceiling, hidden] bf16
    seg_counts: torch.Tensor  # [E_local] int32
    seg_offsets: torch.Tensor  # [E_local + 1] int32 first row of each segment
    src2dst: torch.Tensor  # [num_tokens * top_k] int32 compact rows
    grouped_layout: torch.Tensor  # [m_pad_ceiling] int32; -1 = skipped rows
    m_pad_ceiling: int
    retained_inputs: bool


class ContiguousRowDomainProvider(MoeBaseProvider):
    """The prepare, activation and finalize stages over the contiguous domain.

    A subclass adds the two GEMM stages, gate/up and down. It also passes
    ``m_alignment``, the m-block alignment that its GEMM engine needs. DeepGEMM
    reports that value from ``get_m_alignment_for_contiguous_layout()``.
    """

    def __init__(self, quant_info: MoeLoraBf16QuantInfo, *, m_alignment: int):
        self.quant_info = quant_info
        if not isinstance(m_alignment, int) or m_alignment < 1:
            raise ValueError(f"m_alignment must be a positive int, got {m_alignment!r}")
        self._m_alignment = m_alignment
        if quant_info.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        expected_w2 = (
            quant_info.num_local_experts,
            quant_info.hidden_size,
            quant_info.intermediate_size,
        )
        if quant_info.w2_weight.shape != expected_w2:
            raise ValueError(
                f"w2_weight must be {expected_w2}, got "
                f"{tuple(quant_info.w2_weight.shape)}"
            )
        if (
            quant_info.w13_weight.ndim != 3
            or quant_info.w13_weight.shape[0] != quant_info.num_local_experts
            or quant_info.w13_weight.shape[2] != quant_info.hidden_size
        ):
            raise ValueError(
                "w13_weight must be [num_local_experts, slices*intermediate, hidden]"
            )
        gateup_width = quant_info.w13_weight.shape[1]
        if gateup_width % quant_info.intermediate_size:
            raise ValueError(
                "w13 output width must be an integer multiple of intermediate_size"
            )
        self._gate_up_slices = gateup_width // quant_info.intermediate_size
        if self._gate_up_slices not in (1, 2):
            raise ValueError(
                "contiguous BF16 provider supports one non-gated slice or two "
                f"gated gate/up slices, got {self._gate_up_slices}"
            )

    @property
    def m_alignment(self) -> int:
        return self._m_alignment

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: MoeLoraWorkspace | None = None,
        *,
        schedule_pack: ContiguousSchedulePack | None = None,
    ) -> ContiguousRowState:
        num_pairs = topk_ids.numel()
        num_experts = self.quant_info.num_local_experts
        m_pad_ceiling = contiguous_m_pad_ceiling(
            num_pairs, num_experts, self._m_alignment
        )
        device = hidden_states.device
        if workspace is not None:
            # The row count depends on the alignment. The tag names the
            # alignment, so a provider with another alignment gets its own
            # buffers in this workspace.
            prefix = f"contig:a{self._m_alignment}"
            seg_counts = workspace.tensor(
                f"{prefix}:seg_counts",
                (num_experts,),
                dtype=torch.int32,
                device=device,
            )
            seg_offsets = workspace.tensor(
                f"{prefix}:seg_offsets",
                (num_experts + 1,),
                dtype=torch.int32,
                device=device,
            )
            src2dst = workspace.tensor(
                f"{prefix}:src2dst",
                (num_pairs,),
                dtype=torch.int32,
                device=device,
            )
            grouped_layout = workspace.tensor(
                f"{prefix}:grouped_layout",
                (m_pad_ceiling,),
                dtype=torch.int32,
                device=device,
            )
            # Zero the buffer on the first allocation, so a GEMM tile never
            # reads uninitialized memory. Stale rows from a later pass are safe.
            # Each row is independent, and nothing reads a padding row's output.
            hidden_compact = workspace.tensor(
                f"{prefix}:hidden_compact",
                (m_pad_ceiling, hidden_states.size(1)),
                dtype=torch.bfloat16,
                device=device,
                zero_on_first_allocation=True,
            )
        else:
            seg_counts = torch.empty(num_experts, dtype=torch.int32, device=device)
            seg_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
            src2dst = torch.empty(num_pairs, dtype=torch.int32, device=device)
            grouped_layout = torch.empty(
                m_pad_ceiling, dtype=torch.int32, device=device
            )
            hidden_compact = torch.zeros(
                (m_pad_ceiling, hidden_states.size(1)),
                dtype=torch.bfloat16,
                device=device,
            )
        dispatch_fill_contiguous(
            hidden_states,
            topk_ids,
            num_experts,
            top_k,
            self._m_alignment,
            seg_counts_out=seg_counts,
            seg_offsets_out=seg_offsets,
            src2dst_out=src2dst,
            grouped_layout_out=grouped_layout,
            hidden_compact_out=hidden_compact,
            schedule_pack=schedule_pack,
        )
        return ContiguousRowState(
            hidden_compact=hidden_compact,
            seg_counts=seg_counts,
            seg_offsets=seg_offsets,
            src2dst=src2dst,
            grouped_layout=grouped_layout,
            m_pad_ceiling=m_pad_ceiling,
            retained_inputs=workspace is not None,
        )

    def release_prepared_inputs(self, row_state: ContiguousRowState) -> None:
        # Nothing reads the compact hidden rows after the gate/up GEMM. A
        # workspace tensor must keep its address for graph replay, so the runner
        # frees it later instead.
        if row_state.retained_inputs:
            return
        from sglang.srt.utils import dispose_tensor

        dispose_tensor(row_state.hidden_compact)

    def act_with_delta(
        self,
        row_state: ContiguousRowState,
        gateup_out: torch.Tensor,
        gate_up_delta: torch.Tensor | None,
        topk_ids: torch.Tensor,
        act_out: torch.Tensor,
        activation_lora_input: torch.Tensor,
        *,
        activation: str = "silu",
        consume_base_pdl: bool = False,
    ) -> None:
        act_delta_contiguous(
            gateup_out,
            gate_up_delta,
            act_out,
            activation_lora_input,
            row_state.src2dst,
            topk_ids,
            self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            activation=activation,
            consume_base_pdl=consume_base_pdl,
        )

    def fused_act(
        self,
        row_state: ContiguousRowState,
        family: str,
        *,
        activation: str,
        base_gateup: torch.Tensor,
        act_masked: torch.Tensor,
        act_pairs: torch.Tensor | None,
        routing: RouteView,
        config: Mapping[str, int],
        bridge_gateup: torch.Tensor | None = None,
        b_gate_up: torch.Tensor | None = None,
        bridge_top_k: int = 1,
        consume_base_pdl: bool = False,
    ) -> None:
        # ``act_masked`` is the parameter name in the provider interface. Here
        # it holds the compact 2-D activation buffer.
        fused_b_act_contiguous(
            family,
            activation=activation,
            base_gateup=base_gateup,
            act_compact=act_masked,
            act_pairs=act_pairs,
            src2dst=row_state.src2dst,
            routing=routing,
            num_local_experts=self.num_local_experts,
            gate_first=self.contract.gate_first,
            interleaved=self.contract.interleaved,
            config=config,
            bridge_gateup=bridge_gateup,
            b_gate_up=b_gate_up,
            bridge_top_k=bridge_top_k,
            consume_base_pdl=consume_base_pdl,
        )

    def gateup_out_shape(self, row_state: ContiguousRowState) -> tuple[int, ...]:
        return (
            row_state.m_pad_ceiling,
            self.gate_up_slices * self.quant_info.intermediate_size,
        )

    def act_out_shape(self, row_state: ContiguousRowState) -> tuple[int, ...]:
        return (row_state.m_pad_ceiling, self.quant_info.intermediate_size)

    def down_out_shape(self, row_state: ContiguousRowState) -> tuple[int, ...]:
        return (row_state.m_pad_ceiling, self.quant_info.hidden_size)
