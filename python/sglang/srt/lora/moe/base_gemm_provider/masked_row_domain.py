"""Expert slabs [E_local, m_max, ...] with valid counts and pair mappings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import msgspec
import torch

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


def masked_m_max(num_tokens: int, alignment: int = 8) -> int:
    """Fit all input rows in one expert, with a nonzero aligned slab extent."""
    return max(alignment, (num_tokens + alignment - 1) // alignment * alignment)


class MaskedRowState(msgspec.Struct, kw_only=True):
    """``pair_to_row[t * top_k + k]`` is ``expert * m_max + offset``; a pair is
    valid only when ``topk_ids[t, k] >= 0``.

    The row domain's prepare() sets these six fields: it allocates
    ``masked_m`` and ``pair_to_row``, takes ``hidden_permuted`` from
    _input_buffers(), and the fill writes all three. A provider's row state
    adds exactly the fields its _input_buffers() and _schedules() hooks return.
    """

    hidden_permuted: torch.Tensor  # [E_local, m_max, hidden], the GEMM input dtype
    masked_m: torch.Tensor  # [E_local] int32
    expected_m: int
    pair_to_row: torch.Tensor  # [num_tokens * top_k] int32
    m_max: int
    input_buffer_reuse: bool


class MaskedRowDomainProvider(MoeBaseProvider):
    """prepare() plans the slabs, takes the encoding's buffers, runs its fill
    and builds the GEMM schedules; the hooks default to BF16 rows and none.

    The hooks, in the order prepare() calls them:
      _plan_rows      sizes: the slab extent and, if any, the GEMM token width
      _input_buffers  allocates the GEMM input rows and the encoding's buffers
      _fill_masked    writes masked_m, pair_to_row, the rows and their scales
      _schedules      builds the GEMM schedules over the filled slabs
    ``_row_state_cls`` is built from MaskedRowState's fields plus what
    _input_buffers() and _schedules() return, by field name.
    """

    _row_state_cls = MaskedRowState

    def __init__(self, quant_info: StandardLayoutQuantInfo):
        self.quant_info = quant_info
        self._gate_up_slices = admit_weight_layout(quant_info)

    def prepare(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        workspace: LoraWorkspace | None = None,
    ) -> MaskedRowState:
        num_experts = self.quant_info.num_local_experts
        num_pairs = topk_ids.numel()
        expected_m = expected_rows_per_expert(num_pairs, num_experts)
        token_width, m_max = self._plan_rows(hidden_states.size(0), expected_m)
        device = hidden_states.device
        masked_m = prepare_buffer(
            workspace,
            "masked:masked_m",
            (num_experts,),
            dtype=torch.int32,
            device=device,
        )
        pair_to_row = prepare_buffer(
            workspace,
            "masked:pair_to_row",
            (num_pairs,),
            dtype=torch.int32,
            device=device,
        )
        rows, buffers = self._input_buffers(
            workspace, "masked", (num_experts, m_max), hidden_states
        )
        self._fill_masked(
            hidden_states, topk_ids, top_k, masked_m, pair_to_row, rows, buffers
        )
        return self._row_state_cls(
            hidden_permuted=rows,
            masked_m=masked_m,
            expected_m=expected_m,
            pair_to_row=pair_to_row,
            m_max=m_max,
            input_buffer_reuse=workspace is not None,
            **buffers,
            **self._schedules(
                masked_m, token_width, "cutedsl_masked", workspace, m_max=m_max
            ),
        )

    def _plan_rows(self, num_tokens: int, expected_m: int) -> tuple[int | None, int]:
        """``(token_width, m_max)``: the GEMM token width, or None when rows are
        not planned around one, and the rows of one expert slab. ``m_max`` is at
        least ``num_tokens`` (one expert may take every token) and sizes every
        ``[E_local, m_max, ...]`` buffer of the row state; ``expected_m`` is the
        rounded-up mean rows per expert. Sizes only: nothing is allocated."""
        return None, masked_m_max(num_tokens)

    def _input_buffers(
        self,
        workspace: LoraWorkspace | None,
        prefix: str,
        lead: tuple[int, ...],
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, RowBuffers]:
        """``(rows, buffers)``: the gate/up GEMM's input rows, shaped
        ``(*lead, width)`` with ``lead = (E_local, m_max)``, and the encoding's
        other row-state fields by name (BF16 has none).

        Allocation only, through prepare_buffer() under ``<prefix>:<name>``.
        With a workspace the tensors are its buffers: they keep their addresses
        for graph replay and stay the workspace's. Without one they are fresh
        and the row state's; release_prepared_inputs() disposes of the rows.
        Nothing is written here: _fill_masked() writes the rows and their
        scales, and an encoding's down-input buffers are written later, by the
        activation or by down(). prepare() stores ``rows`` as
        ``hidden_permuted`` and passes ``buffers`` to ``_row_state_cls``."""
        rows = prepare_buffer(
            workspace,
            f"{prefix}:hidden_permuted",
            (*lead, hidden_states.size(1)),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        return rows, {}

    def _fill_masked(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        masked_m: torch.Tensor,
        pair_to_row: torch.Tensor,
        rows: torch.Tensor,
        buffers: RowBuffers,
    ) -> None:
        """Dispatch the routed rows into the slabs in the encoding of
        ``rows``: writes ``masked_m``, ``pair_to_row``, ``rows`` and the
        input-row scales among ``buffers``."""
        from sglang.kernels.ops.lora.moe.dispatch_masked import (
            dispatch_fill_masked_bf16,
        )

        dispatch_fill_masked_bf16(
            hidden_states,
            topk_ids,
            top_k,
            masked_m_out=masked_m,
            pair_to_row_out=pair_to_row,
            rows_out=rows,
        )

    def _schedules(
        self,
        counts: torch.Tensor,
        token_width: int | None,
        prefix: str,
        workspace: LoraWorkspace | None,
        **rows: int,
    ) -> dict[str, Any]:
        """Row-state fields of the GEMM schedules over the filled rows; a bare
        row domain has none. ``counts`` is ``masked_m`` as the fill wrote it,
        ``token_width`` what _plan_rows() chose, ``rows`` the slab geometry
        (``m_max``), ``prefix`` the workspace name of the schedule buffers."""
        return {}

    def release_prepared_inputs(self, row_state: MaskedRowState) -> None:
        # Workspace buffers retain their addresses for graph replay.
        if row_state.input_buffer_reuse:
            return
        from sglang.srt.utils import dispose_tensor

        dispose_tensor(row_state.hidden_permuted)

    # The layout's view of rows for the GEMM launch and its compile stand-ins.
    @staticmethod
    def _rows_arg(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def _input_rows(row_state: MaskedRowState) -> torch.Tensor:
        return row_state.hidden_permuted

    @staticmethod
    def _group_arg(row_state: MaskedRowState) -> torch.Tensor:
        return row_state.masked_m

    def _dummy_layout(self, device) -> tuple[tuple, tuple, torch.Tensor]:
        """Leading shapes of stand-in rows and of their scales, and the groups."""
        experts = self.quant_info.num_local_experts
        return (
            (experts, 256),
            (experts, 256),
            torch.zeros(experts, dtype=torch.int32, device=device),
        )

    def gateup_out_shape(self, row_state: MaskedRowState) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            row_state.m_max,
            self.gate_up_slices * self.quant_info.intermediate_size,
        )

    def act_out_shape(self, row_state: MaskedRowState) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            row_state.m_max,
            self.quant_info.intermediate_size,
        )

    def down_out_shape(self, row_state: MaskedRowState) -> tuple[int, ...]:
        return (
            self.quant_info.num_local_experts,
            row_state.m_max,
            self.quant_info.hidden_size,
        )
