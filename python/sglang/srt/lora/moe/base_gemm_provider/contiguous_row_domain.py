"""Routed rows packed into aligned, contiguous expert segments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgspec
import torch

from sglang.kernels.ops.lora.moe.dispatch_contiguous import (
    contiguous_m_pad_ceiling,
    dispatch_fill_rows_contiguous_bf16,
    dispatch_layout_contiguous,
)
from sglang.srt.lora.moe.base_gemm_provider.base import (
    MoeBaseProvider,
    RowBuffers,
    admit_weight_layout,
    expected_rows_per_expert,
    prepare_buffer,
)
from sglang.srt.lora.moe.quant_info import StandardLayoutQuantInfo

if TYPE_CHECKING:
    from sglang.srt.lora.workspace import LoraWorkspace


class ContiguousRowState(msgspec.Struct, kw_only=True):
    """The row domain's prepare() sets these six fields: it allocates
    ``seg_counts``, ``seg_offsets`` and ``pair_to_row`` and the layout kernel
    writes them, and it takes ``hidden_compact`` from _input_buffers() for the
    fill to write. A provider's row state adds exactly the fields its
    _input_buffers() and _schedules() hooks return.
    """

    hidden_compact: torch.Tensor  # [m_pad_ceiling, hidden], the GEMM input dtype
    seg_counts: torch.Tensor  # [E_local] int32
    seg_offsets: torch.Tensor  # [E_local + 1] int32 first row of each segment
    pair_to_row: torch.Tensor  # [num_tokens * top_k] int32 compact rows
    m_pad_ceiling: int
    input_buffer_reuse: bool


class ContiguousRowDomainProvider(MoeBaseProvider):
    """prepare() lays the segments out, takes the encoding's buffers, runs its
    fill and builds the GEMM schedules; the hooks default to BF16 rows and none.

    The hooks, in the order prepare() calls them:
      _input_buffers    allocates the GEMM input rows and the encoding's buffers
      _fill_contiguous  writes the rows and their scales (and finishes
                        pair_to_row when ``_fill_finalizes_map``)
      _token_width_for  the GEMM token width for these rows, if any
      _schedules        builds the GEMM schedules over the filled segments
    ``_row_state_cls`` is built from ContiguousRowState's fields plus what
    _input_buffers() and _schedules() return, by field name.
    """

    _row_state_cls = ContiguousRowState
    # A fill that turns the layout's in-expert slots into rows itself.
    _fill_finalizes_map = False

    def __init__(self, quant_info: StandardLayoutQuantInfo, *, m_alignment: int):
        self.quant_info = quant_info
        if not isinstance(m_alignment, int) or m_alignment < 1:
            raise ValueError(f"m_alignment must be a positive int, got {m_alignment!r}")
        self._m_alignment = m_alignment
        self._gate_up_slices = admit_weight_layout(quant_info)

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: LoraWorkspace | None = None,
    ) -> ContiguousRowState:
        num_pairs = topk_ids.numel()
        num_experts = self.quant_info.num_local_experts
        m_pad_ceiling = contiguous_m_pad_ceiling(
            num_pairs, num_experts, self._m_alignment
        )
        device = hidden_states.device
        prefix = f"contiguous:a{self._m_alignment}"
        seg_counts = prepare_buffer(
            workspace,
            f"{prefix}:seg_counts",
            (num_experts,),
            dtype=torch.int32,
            device=device,
        )
        seg_offsets = prepare_buffer(
            workspace,
            f"{prefix}:seg_offsets",
            (num_experts + 1,),
            dtype=torch.int32,
            device=device,
        )
        pair_to_row = prepare_buffer(
            workspace,
            f"{prefix}:pair_to_row",
            (num_pairs,),
            dtype=torch.int32,
            device=device,
        )
        rows, buffers = self._input_buffers(
            workspace, prefix, (m_pad_ceiling,), hidden_states
        )
        dispatch_layout_contiguous(
            hidden_states,
            topk_ids,
            num_experts,
            top_k,
            self._m_alignment,
            seg_counts_out=seg_counts,
            seg_offsets_out=seg_offsets,
            pair_to_row_out=pair_to_row,
            finalize=not self._fill_finalizes_map,
        )
        self._fill_contiguous(
            hidden_states, topk_ids, pair_to_row, seg_offsets, rows, buffers
        )
        # Distinct top-k experts bound each expert to one row per token.
        max_expert_rows = max(int(hidden_states.size(0)), 1)
        token_width = self._token_width_for(
            max_expert_rows, expected_rows_per_expert(num_pairs, num_experts)
        )
        return self._row_state_cls(
            hidden_compact=rows,
            seg_counts=seg_counts,
            seg_offsets=seg_offsets,
            pair_to_row=pair_to_row,
            m_pad_ceiling=m_pad_ceiling,
            input_buffer_reuse=workspace is not None,
            **buffers,
            **self._schedules(
                seg_counts,
                token_width,
                f"cutedsl_contiguous:a{self._m_alignment}",
                workspace,
                m_pad_ceiling=m_pad_ceiling,
                max_expert_rows=max_expert_rows,
                m_alignment=self._m_alignment,
            ),
        )

    def _input_buffers(
        self,
        workspace: LoraWorkspace | None,
        prefix: str,
        lead: tuple[int, ...],
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, RowBuffers]:
        """``(rows, buffers)``: the gate/up GEMM's input rows, shaped
        ``(*lead, width)`` with ``lead = (m_pad_ceiling,)``, and the encoding's
        other row-state fields by name (BF16 has none).

        Allocation only, through prepare_buffer() under ``<prefix>:<name>``.
        With a workspace the tensors are its buffers: they keep their addresses
        for graph replay and stay the workspace's. Without one they are fresh
        and the row state's; release_prepared_inputs() disposes of the rows.
        No input-dependent fill happens here: _fill_contiguous() writes the
        rows and their scales, and an encoding's down-input buffers are written
        later, by the activation or by down(). A first allocation is
        initialized as the buffer's policy says (``zero_on_first_allocation``:
        the rows start zeroed); a reused buffer is not reset. prepare() stores
        ``rows`` as ``hidden_compact`` and passes ``buffers`` to
        ``_row_state_cls``."""
        # Padding outputs are ignored, so reused padding needs no reset.
        rows = prepare_buffer(
            workspace,
            f"{prefix}:hidden_compact",
            (*lead, hidden_states.size(1)),
            dtype=torch.bfloat16,
            device=hidden_states.device,
            zero_on_first_allocation=True,
        )
        return rows, {}

    def _fill_contiguous(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        pair_to_row: torch.Tensor,
        seg_offsets: torch.Tensor,
        rows: torch.Tensor,
        buffers: RowBuffers,
    ) -> None:
        """Copy the routed rows into their segments in the encoding of
        ``rows``, through the ``pair_to_row`` the layout kernel wrote: writes
        ``rows`` and the input-row scales among ``buffers``. A fill that sets
        ``_fill_finalizes_map`` also turns the map's in-expert slots into rows."""
        dispatch_fill_rows_contiguous_bf16(
            hidden_states, topk_ids, pair_to_row, hidden_compact_out=rows
        )

    def _token_width_for(self, max_expert_rows: int, expected_m: int) -> int | None:
        """The GEMM token width for these rows; a bare row domain has none."""
        return None

    def _schedules(
        self,
        counts: torch.Tensor,
        token_width: int | None,
        prefix: str,
        workspace: LoraWorkspace | None,
        **rows: int,
    ) -> dict[str, Any]:
        """Row-state fields of the GEMM schedules over the filled rows; a bare
        row domain has none. ``counts`` is ``seg_counts`` as the layout kernel
        wrote it, ``token_width`` what _token_width_for() chose, ``rows`` the
        segment geometry (``m_pad_ceiling``, ``max_expert_rows``,
        ``m_alignment``), ``prefix`` the workspace name of the schedule buffers."""
        return {}

    def release_prepared_inputs(self, row_state: ContiguousRowState) -> None:
        # Workspace buffers retain their addresses for graph replay.
        if row_state.input_buffer_reuse:
            return
        from sglang.srt.utils import dispose_tensor

        dispose_tensor(row_state.hidden_compact)

    # The layout's view of rows for the GEMM launch and its compile stand-ins.
    @staticmethod
    def _rows_arg(tensor: torch.Tensor) -> torch.Tensor:
        # Flat rows occupy one expert slot in the swap_ab wrapper.
        return tensor.unsqueeze(0)

    @staticmethod
    def _input_rows(row_state: ContiguousRowState) -> torch.Tensor:
        return row_state.hidden_compact

    @staticmethod
    def _group_arg(row_state: ContiguousRowState) -> torch.Tensor:
        return row_state.seg_offsets

    def _dummy_layout(self, device) -> tuple[tuple, tuple, torch.Tensor]:
        """Leading shapes of stand-in rows and of their scales, and the groups."""
        experts = self.quant_info.num_local_experts
        return (
            (256,),
            (1, 256),
            torch.zeros(experts + 1, dtype=torch.int32, device=device),
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
