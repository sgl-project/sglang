"""Temporary prefill shim for the Ascend Ops dispatcher.

The real prefill (TP allgather / alltoallv) path now lives in
``AscendOpsDispatcher``. This subclass keeps the DS v2 EP dispatch/combine as an
A/B control for the prefill batch, so both can be exercised from the same build
via ``SGLANG_ASCEND_OPS_MOCK_PREFILL``. Delete this module once the real path is
verified end to end.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.token_dispatcher.ascend_ops import AscendOpsDispatcher


class MockPrefillAscendOpsDispatcher(AscendOpsDispatcher):
    """Run prefill through the DS v2 EP dispatch/combine path."""

    def _dispatch_prefill_allgather(self, hidden_states, topk_weights, topk_ids):
        return self._dispatch_prefill_mock(
            hidden_states, topk_weights, topk_ids, "prefill_allgather"
        )

    def _dispatch_prefill_alltoallv(self, hidden_states, topk_weights, topk_ids):
        return self._dispatch_prefill_mock(
            hidden_states, topk_weights, topk_ids, "prefill_alltoallv"
        )

    def _combine_prefill_allgather(self, combine_input):
        return self._ds_combine(combine_input)

    def _combine_prefill_alltoallv(self, combine_input):
        return self._ds_combine(combine_input)

    def _dispatch_prefill_mock(self, hidden_states, topk_weights, topk_ids, path):
        self._dispatch_path = path
        self._dispatch_group = self.ep_group
        return self._ds_dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            padded_bs=self._prefill_padded_bs(hidden_states),
            dtype_mode="normal",
            prefill_mode=self.ascend_ops_config.prefill,
        )

    def _prefill_padded_bs(self, hidden_states: torch.Tensor) -> int:
        """Batch size shared by all EP ranks (the collective takes one BS)."""
        padded_bs = torch.tensor(
            [hidden_states.shape[0]], dtype=torch.int64, device=hidden_states.device
        )
        dist.all_reduce(padded_bs, op=dist.ReduceOp.MAX, group=self.ep_group)
        return int(padded_bs.item())
