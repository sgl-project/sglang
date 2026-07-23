from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


@dataclass(frozen=True, slots=True, kw_only=True)
class CanaryLaunchCapacities:
    """Pre-allocation sizes for the per-forward tensors a SingleForwardManager owns. Computed
    once at install_canary from ServerArgs + ModelRunner metadata; all fields are upper
    bounds - actual per-step usage may be smaller but never larger.

    Fields:
        per_forward_verify_capacity: VerifyPlan row capacity for the per-forward HEAD/TAIL
            launches. Sized to pool_slot_count * 3 (3x headroom; radix prefix sharing across
            running reqs can cause sum_r prefix_lens[r] to exceed the pool slot count). When
            the per-step actual count exceeds this, the plan kernel sets VerifyPlan.enable=0
            and the verify kernel skips the step; host logs a warn (no install-time raise).
        per_forward_write_req_capacity: WritePlan row capacity for per-forward writes, also used
            to size the static PlanInput buffers (= max batch size under cuda graph).
        per_forward_write_entry_capacity: Capacity for the expected_input_* placeholder tensors,
            one entry per token written in a single forward.
    """

    per_forward_verify_capacity: int
    per_forward_write_req_capacity: int
    per_forward_write_entry_capacity: int

    def __post_init__(self) -> None:
        for name, value in (
            ("per_forward_verify_capacity", self.per_forward_verify_capacity),
            ("per_forward_write_req_capacity", self.per_forward_write_req_capacity),
            ("per_forward_write_entry_capacity", self.per_forward_write_entry_capacity),
        ):
            if value <= 0:
                raise ValueError(f"kv-canary: {name} must be positive, got {value}")

    @classmethod
    def from_args(
        cls,
        *,
        server_args: ServerArgs,
        req_to_token_pool_size: int,
        max_seq_len_per_req: int,
        pool_slot_count: int,
    ) -> CanaryLaunchCapacities:
        if req_to_token_pool_size <= 0:
            raise ValueError(
                "kv-canary: req_to_token_pool_size must be positive, "
                f"got {req_to_token_pool_size}"
            )
        if max_seq_len_per_req <= 0:
            raise ValueError(
                "kv-canary: max_seq_len_per_req must be positive, "
                f"got {max_seq_len_per_req}"
            )
        if pool_slot_count <= 0:
            raise ValueError(
                f"kv-canary: pool_slot_count must be positive, got {pool_slot_count}"
            )

        cuda_graph_config = server_args.cuda_graph_config
        cuda_graph_max_bs = (
            cuda_graph_config.decode.max_bs if cuda_graph_config is not None else 0
        ) or 0
        if cuda_graph_max_bs < 0:
            raise ValueError(
                f"kv-canary: cuda_graph_max_bs must be non-negative, got {cuda_graph_max_bs}"
            )

        spec_num_draft_tokens = server_args.speculative_num_draft_tokens
        if spec_num_draft_tokens is None:
            spec_num_draft_tokens = 0
        if spec_num_draft_tokens < 0:
            raise ValueError(
                "kv-canary: speculative_num_draft_tokens must be non-negative, "
                f"got {spec_num_draft_tokens}"
            )

        max_prefill_tokens = server_args.max_prefill_tokens
        if max_prefill_tokens <= 0:
            raise ValueError(
                f"kv-canary: max_prefill_tokens must be positive, got {max_prefill_tokens}"
            )

        num_tokens_per_req = 1
        if spec_num_draft_tokens:
            num_tokens_per_req = max(num_tokens_per_req, spec_num_draft_tokens)

        max_bs = max(cuda_graph_max_bs, req_to_token_pool_size)

        chunked_prefill_size = server_args.chunked_prefill_size
        chunked_limit = (
            chunked_prefill_size
            if chunked_prefill_size is not None and chunked_prefill_size >= 0
            else math.inf
        )
        max_extend_tokens_per_forward = min(max_prefill_tokens, chunked_limit)

        write_entry_capacity = max(
            max_bs * num_tokens_per_req, max_extend_tokens_per_forward
        )

        # Radix prefix sharing lets sum_r prefix_lens[r] exceed pool_slot_count; observed up to ~2x
        # on 20 parallel token-oracle prompts. 3x headroom keeps the partial-fallback path
        # (plan kernel enable=0 + host warn) exceptional. Overflow does not raise at install time.
        per_forward_verify_capacity = int(pool_slot_count * 3)

        return cls(
            per_forward_verify_capacity=per_forward_verify_capacity,
            per_forward_write_req_capacity=max_bs,
            per_forward_write_entry_capacity=write_entry_capacity,
        )
