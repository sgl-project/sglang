from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryLaunchCapacities:
    """Pre-allocation sizes for the per-forward and sweep tensors a CanaryRunner owns. Computed
    once at install_canary from ServerArgs + ModelRunner metadata; all four fields are upper
    bounds - actual per-step usage may be smaller but never larger.

    Fields:
        per_forward_verify_capacity: VerifyPlan row capacity for the per-forward HEAD/TAIL
            launches. Sized to pool_slot_count * 1.2 (radix prefix sharing across running reqs can
            cause sum_r prefix_lens[r] to exceed the pool slot count). When the per-step actual
            count exceeds this, the plan kernel sets VerifyPlan.enable=0 and the verify kernel
            skips the step; host logs a warn (no install-time raise).
        per_forward_write_req_capacity: WritePlan row capacity for per-forward writes, also used
            to size the static fb_* PlanInput buffers (= max batch size under cuda graph).
        per_forward_write_entry_capacity: Capacity for the expected_input_* placeholder tensors,
            one entry per token written in a single forward.
        sweep_verify_capacity: VerifyPlan row capacity for the radix sweep launch, sized to the
            pool slot count.
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
        # sharing, reqs can collectively reference more tokens than the pool holds, so the budget
        # is pool_slot_count with a 1.2x headroom. Overflow at runtime is handled by the
        # partial-fallback path (plan kernel disables verify for the step + host warn-logs);
        # there is no install-time raise.
        per_forward_verify_capacity = max(1, int(pool_slot_count * 1.2))

        return cls(
            per_forward_verify_capacity=per_forward_verify_capacity,
            per_forward_write_req_capacity=max(1, max_bs),
            per_forward_write_entry_capacity=write_entry_capacity,
            sweep_verify_capacity=max(1, pool_slot_count),
        )
