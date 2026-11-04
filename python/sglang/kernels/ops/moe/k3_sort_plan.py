# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 MoE sort plan: the model-owned inputs of the K3 route/sort/quant
kernels, plugged into aiter ``fused_moe`` through moe_sorting_small's
SortExtension hook.

route_state: launch-A products of k3_route_sort_mid (top-k, counts, quant),
    run by the model with top-k fused in.
shared_gemm: (topk_ids, act, w_down, out) of the shared down GEMM to run in the
    small-batch sort launch (k3_moe_epi.shared_gemm_sort_quant).
Both belong to the top-k tensor they were computed for, and the sort takes them
only for that tensor; ``finish`` resolves whatever was not taken.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

from sglang.kernels.ops.moe import k3_moe_epi, k3_route_sort, k3_route_sort_mid
from sglang.kernels.ops.moe.k3_route_sort_mid import RouteState
from sglang.kernels.ops.moe.moe_sorting_small import SortExtension, sort_extension

logger = logging.getLogger(__name__)


@dataclass
class K3SortPlan(SortExtension):
    route_state: RouteState | None = None
    shared_gemm: tuple | None = None

    def max_quant_tokens(self) -> int:
        return k3_route_sort_mid.max_tokens()

    def mid_sort_supported(self, m, topk, num_experts, block_size, n_cols) -> bool:
        return k3_route_sort_mid.supported(m, topk, num_experts, block_size, n_cols)

    def mid_sort_quant(
        self,
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        block_size,
        num_experts,
        qx,
    ):
        # launch A, unless the model already ran it with top-k fused in
        st = self.take_route_state(topk_ids, qx)
        if st is None:
            st = k3_route_sort_mid.route_quant(
                qx,
                k3_route_sort_mid.counts_buffer(topk_ids.device, None, num_experts),
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                num_experts=num_experts,
            )
        elif st.topk_weights.data_ptr() != topk_weights.data_ptr():
            st.topk_weights = topk_weights
        return k3_route_sort_mid.sort_scatter(
            st,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            block_size,
            num_experts,
        )

    def small_sort_quant(
        self,
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        block_size,
        mx_quant_input,
        qout,
        qscale,
    ) -> bool:
        m, topk = topk_ids.shape
        if not k3_route_sort.sort_quant_supported(
            m, topk, block_size, mx_quant_input.shape[-1]
        ):
            return False
        args = (
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            block_size,
            mx_quant_input,
            qout,
            qscale,
        )
        # the model planned its shared down GEMM for this top-k; run it in the
        # sort's launch
        pending = self.take_shared_gemm(topk_ids)
        if pending is not None:
            k3_moe_epi.shared_gemm_sort_quant(pending, *args)
        else:
            k3_route_sort.sort_quant(*args)
        return True

    def take_route_state(self, topk_ids, qx) -> RouteState | None:
        st = self.route_state
        if (
            st is None
            or st.consumed
            or topk_ids.data_ptr() != st.topk_ids.data_ptr()
            or topk_ids.shape != st.topk_ids.shape
            or qx is None
            or qx.data_ptr() != st.qx.data_ptr()
            or qx.shape != st.qx.shape
            or qx.stride() != st.qx.stride()
        ):
            return None
        return st

    def take_shared_gemm(self, topk_ids):
        """(act, w_down, out) if the GEMM was planned for exactly this top-k."""
        p = self.shared_gemm
        if (
            p is not None
            and p[0].data_ptr() == topk_ids.data_ptr()
            and p[0].shape == topk_ids.shape
        ):
            self.shared_gemm = None
            return p[1:]
        return None

    def finish(self) -> None:
        st = self.route_state
        if st is not None and not st.consumed:
            # launch A ran but its sort launch B never did
            st.counts.zero_()
        if self.shared_gemm is not None:
            _, act, w_down, out = self.shared_gemm
            self.shared_gemm = None
            _warn_shared_gemm_untaken(act.shape[0])
            k3_moe_epi.shared_gemm_preact(act, w_down, out)


@functools.cache
def _warn_shared_gemm_untaken(m: int) -> None:
    logger.warning(
        "The MoE sort did not take the shared GEMM (M=%d); running it "
        "standalone (correct, no launch saved)",
        m,
    )


def k3_sort_plan(plan: K3SortPlan | None = None):
    """Run the enclosed aiter fused_moe calls with the K3 sort kernels and
    ``plan``'s inputs; resolves the untaken parts of the plan on exit."""
    return sort_extension(K3SortPlan() if plan is None else plan)
