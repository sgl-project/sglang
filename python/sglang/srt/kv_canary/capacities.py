from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

_MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY: int = 4_000_000


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryLaunchCapacities:
    """Pre-allocation sizes for the per-forward and sweep tensors a CanaryRunner owns. Computed
    once at install_canary from ServerArgs + ModelRunner metadata; all four fields are upper
    bounds - actual per-step usage may be smaller but never larger.

    Fields:
        per_forward_verify_capacity: VerifyPlan row capacity for the per-forward HEAD/TAIL
            launches. Sized to the total verify entries the per-forward path may produce in one
            step (= sum_r prefix_lens[r] for the FULL group), upper-bounded by max_bs *
            max_seq_len_per_req (the req_to_token table extent). install_canary refuses to silently
            cap this value: if the upper exceeds _MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY it raises with
            an actionable knob list. PerForwardOrchestrator.before_forward additionally throws on
            the per-step actual sum so an undersized capacity fails fast instead of OOB-reading
            the tail threads of the verify kernel grid.
        per_forward_write_req_capacity: WritePlan row capacity for per-forward writes, also used
            to size the static fb_* PlanInput buffers (= max batch size under cuda graph).
        per_forward_write_entry_capacity: Capacity for the expected_input_* placeholder tensors,
            one entry per token written in a single forward.
        sweep_verify_capacity: VerifyPlan row capacity for the radix sweep launch, sized to the
            pool slot count. install_canary throws when this exceeds
            _MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY for the same reason as per-forward.
    """

    per_forward_verify_capacity: int
    per_forward_write_req_capacity: int
    per_forward_write_entry_capacity: int
    sweep_verify_capacity: int

    @classmethod
    def from_args(
        cls,
        *,
        server_args: "ServerArgs",
        req_to_token_pool_size: int,
        max_seq_len_per_req: int,
        pool_slot_count: int,
    ) -> "CanaryLaunchCapacities":
        cuda_graph_max_bs = server_args.cuda_graph_max_bs or 0
        spec_num_draft_tokens = server_args.speculative_num_draft_tokens
        num_tokens_per_bs = 1
        if spec_num_draft_tokens:
            num_tokens_per_bs = max(num_tokens_per_bs, spec_num_draft_tokens)
        max_bs = max(cuda_graph_max_bs, req_to_token_pool_size)
        chunked_prefill_size = server_args.chunked_prefill_size
        max_prefill_tokens = server_args.max_prefill_tokens
        chunked_limit = (
            chunked_prefill_size
            if chunked_prefill_size is not None and chunked_prefill_size >= 0
            else math.inf
        )
        max_extend_tokens_per_forward = min(max_prefill_tokens, chunked_limit)
        write_entry_capacity = max(
            1, max(max_bs * num_tokens_per_bs, max_extend_tokens_per_forward)
        )
        # Per-forward verify entries = sum_r (prefix_lens[r] - SWA_window_start[r]); the FULL group
        # never clips with a window, so the upper bound is sum_r prefix_lens[r]. Under radix prefix
        # sharing reqs can collectively reference more tokens than the pool holds, so the hard bound
        # is the req_to_token table extent (max_bs rows * max_seq_len_per_req cols). Throw at install
        # when the safe ceiling is exceeded — refuse to silently shrink the buffer and let runtime
        # decide whether the canary actually has enough room.
        per_forward_verify_capacity_upper = max_bs * max_seq_len_per_req
        if per_forward_verify_capacity_upper > _MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY:
            raise RuntimeError(
                f"kv-canary: per-forward verify capacity "
                f"{per_forward_verify_capacity_upper} (= max_bs {max_bs} * max_seq_len_per_req "
                f"{max_seq_len_per_req}) exceeds the cuda-grid-safe ceiling "
                f"{_MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY}. To enable canary, choose one: "
                f"(a) lower --cuda-graph-max-bs / --max-running-requests so max_bs drops below "
                f"{_MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY // max_seq_len_per_req + 1}; "
                f"(b) lower the per-req sequence cap (req_to_token cols, currently "
                f"{max_seq_len_per_req}) so max_seq_len_per_req drops below "
                f"{_MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY // max_bs + 1}; "
                f"(c) raise _MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY in "
                f"python/sglang/srt/kv_canary/capacities.py if the extra device memory is acceptable."
            )

        if pool_slot_count > _MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY:
            raise RuntimeError(
                f"kv-canary: sweep verify capacity {pool_slot_count} "
                f"(= max_total_num_tokens) exceeds the cuda-grid-safe ceiling "
                f"{_MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY}. To enable canary, choose one: "
                f"(a) reduce the KV pool size (--mem-fraction-static, --max-total-tokens, or shrink "
                f"the model footprint) so max_total_num_tokens drops below "
                f"{_MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY}; "
                f"(b) raise _MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY in "
                f"python/sglang/srt/kv_canary/capacities.py if the extra device memory is acceptable."
            )

        return cls(
            per_forward_verify_capacity=max(1, per_forward_verify_capacity_upper),
            per_forward_write_req_capacity=max(1, max_bs),
            per_forward_write_entry_capacity=write_entry_capacity,
            sweep_verify_capacity=max(1, pool_slot_count),
        )
