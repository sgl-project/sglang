from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Optional

MAX_MULTI_STEP_TOPK_NUM = 8192


class HiSparseKVTransferKind(str, Enum):
    LINEAR = "linear"
    DSV4_PAGED = "dsv4_paged"


@dataclass(frozen=True)
class HiSparseSpecLayout:
    """Logical layout for speculative HiSparse.

    0                    H-1 H          H+1               H+1+E                H+T
    ┌───────────────────────┬───────────┬─────────────────┬─────────────────────┐
    │ hot buffer: H         │ canonical │ extra: E        |scratch: S           │
    │ swap in/out           │ C=1       │ draft token kv  │mult step swap buffer│
    └───────────────────────┴───────────┴─────────────────┴─────────────────────┘
    H: device_buffer_size
    T: page_size
    """

    canonical_slots: ClassVar[int] = 1

    hot_slots: int
    tail_capacity_slots: int
    compress_ratio: int
    verify_width: int
    top_k: int

    @property
    def speculative_slots(self) -> int:
        return (self.verify_width + self.compress_ratio - 1) // self.compress_ratio

    @property
    def scratch_slots(self) -> int:
        return self.tail_capacity_slots - self.canonical_slots - self.speculative_slots

    @property
    def canonical_offset(self) -> int:
        return self.hot_slots

    @property
    def speculative_offset(self) -> int:
        return self.canonical_offset + self.canonical_slots

    @property
    def scratch_offset(self) -> int:
        return self.speculative_offset + self.speculative_slots

    @property
    def active_tail_slots(self) -> int:
        return self.canonical_slots + self.speculative_slots

    @property
    def active_buffer_slots(self) -> int:
        return self.hot_slots + self.active_tail_slots

    @property
    def total_buffer_slots(self) -> int:
        return self.hot_slots + self.tail_capacity_slots

    @property
    def occurrences(self) -> int:
        return self.verify_width * self.top_k

    @property
    def hash_slots(self) -> int:
        return 1 << (2 * self.hot_slots - 1).bit_length()

    def policy_width(self, num_req_slots: int) -> int:
        return max(num_req_slots, self.hot_slots)

    def state_width(self, num_req_slots: int) -> int:
        width = max(4 * num_req_slots, 5 * self.occurrences)
        # The CUDA workspace reinterprets pairs of int32 values as uint64.
        return width + width % 2

    def fixed_state_bytes(self, *, num_layers: int, num_req_slots: int) -> int:
        """Exact bytes allocated by HiSparseCoordinator for speculative state."""

        int32_bytes = 4
        int64_bytes = 8
        cache_index = num_layers * num_req_slots * 2 * self.hash_slots * int64_bytes
        cache_policy = (
            num_layers
            * (num_req_slots + 1)
            * self.policy_width(num_req_slots)
            * int32_bytes
        )
        scratch_state = (
            num_layers
            * (num_req_slots + 1)
            * self.state_width(num_req_slots)
            * int32_bytes
        )
        verify_buffers = (
            2 * num_req_slots * self.verify_width * self.top_k * int32_bytes
        )
        finalize_snapshots = num_req_slots * (int64_bytes + int64_bytes + int32_bytes)
        return (
            cache_index
            + cache_policy
            + scratch_state
            + verify_buffers
            + finalize_snapshots
        )


@dataclass(frozen=True)
class HiSparseSpecPlan:
    layout: HiSparseSpecLayout
    transfer_kind: HiSparseKVTransferKind


def build_hisparse_spec_layout(
    *,
    hot_slots: int,
    tail_capacity_slots: int,
    compress_ratio: int,
    verify_width: Optional[int],
    top_k: int,
) -> HiSparseSpecLayout:
    if verify_width is None:
        raise ValueError(
            "HiSparse speculative verify requires speculative_num_draft_tokens."
        )

    dimensions = {
        "hot_slots": hot_slots,
        "tail_capacity_slots": tail_capacity_slots,
        "compress_ratio": compress_ratio,
        "verify_width": verify_width,
        "top_k": top_k,
    }
    invalid_types = [
        name
        for name, value in dimensions.items()
        if not isinstance(value, int) or isinstance(value, bool)
    ]
    if invalid_types:
        raise ValueError(
            "HiSparse speculative layout dimensions must be integers; invalid: "
            + ", ".join(f"{name}={dimensions[name]!r}" for name in invalid_types)
        )

    layout = HiSparseSpecLayout(
        hot_slots=hot_slots,
        tail_capacity_slots=tail_capacity_slots,
        compress_ratio=compress_ratio,
        verify_width=verify_width,
        top_k=top_k,
    )
    if not (
        layout.compress_ratio > 0
        and layout.verify_width > 0
        and layout.top_k > 0
        and layout.tail_capacity_slots > 0
        and layout.occurrences <= layout.hot_slots
        and layout.occurrences <= MAX_MULTI_STEP_TOPK_NUM
        and layout.speculative_slots > 0
        and layout.scratch_slots >= 0
    ):
        raise ValueError(
            "Invalid HiSparse speculative layout: "
            f"H={layout.hot_slots}, C={layout.canonical_slots}, "
            f"E={layout.speculative_slots}, S={layout.scratch_slots}, "
            f"T={layout.tail_capacity_slots}, W={layout.verify_width}, "
            f"K={layout.top_k}, R={layout.compress_ratio}."
        )
    return layout


def resolve_hisparse_spec_plan(
    *,
    server_args,
    hf_text_config,
    is_draft_worker: bool = False,
) -> Optional[HiSparseSpecPlan]:
    """Resolve the supported model/spec combination once during startup."""

    from sglang.srt.arg_groups.model_override_base import resolved_view

    # Defaults such as page size and speculative width live in the resolution
    # declarations rather than on the raw ServerArgs fields.
    server_args = resolved_view(server_args)

    if (
        not server_args.enable_hisparse
        or server_args.speculative_algorithm is None
        or is_draft_worker
    ):
        return None

    from sglang.srt.configs.model_config import is_deepseek_dsa, is_deepseek_v4
    from sglang.srt.mem_cache.sparsity.factory import parse_hisparse_config_json
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )
    from sglang.srt.utils import is_cuda

    algorithm = server_args.speculative_algorithm.upper()
    is_v4 = is_deepseek_v4(hf_text_config)
    is_dsa_mtp = is_deepseek_dsa(hf_text_config) and algorithm == "EAGLE"
    if not is_cuda() or not (
        (
            is_v4
            and algorithm == "DSPARK"
            and read_ragged_verify_mode() is RaggedVerifyMode.STATIC
        )
        or is_dsa_mtp
    ):
        raise ValueError(
            "HiSparse speculative verify requires NVIDIA CUDA and either "
            "DeepSeek V4 with DSpark static mode or DSA with EAGLE MTP."
        )

    if is_dsa_mtp:
        if server_args.disaggregation_mode == "null":
            raise ValueError(
                "HiSparse DSA MTP requires PD disaggregation; "
                "unified serving is not supported."
            )
        if server_args.speculative_eagle_topk != 1:
            raise ValueError("HiSparse MTP requires speculative_eagle_topk=1.")
        if (
            server_args.speculative_num_draft_tokens
            != server_args.speculative_num_steps + 1
        ):
            raise ValueError(
                "HiSparse MTP requires verify width = speculative_num_steps + 1."
            )
        if (
            server_args.speculative_adaptive
            or server_args.enable_multi_layer_eagle
            or server_args.enable_two_batch_overlap
            or server_args.enable_pdmux
            or server_args.pp_size != 1
            or (server_args.attn_cp_size or 1) != 1
            or server_args.dcp_size != 1
            or server_args.enable_prefill_cp
        ):
            raise ValueError(
                "HiSparse MTP requires fixed-width single-layer EAGLE, PP=CP=DCP=1, "
                "and no two-batch overlap or PD multiplexing."
            )
        if (
            server_args.disaggregation_mode == "decode"
            and server_args.disaggregation_transfer_backend != "mooncake"
        ):
            raise ValueError("HiSparse MTP PD transfer requires Mooncake.")
        if server_args.page_size != 64:
            raise ValueError("HiSparse DSA MTP requires page_size=64.")

    compress_ratios = 4 if is_v4 else 1
    page_size = int(server_args.page_size)

    hisparse_config = parse_hisparse_config_json(server_args.hisparse_config)
    if hisparse_config.device_buffer_size % (page_size // compress_ratios):
        raise ValueError("HiSparse speculative hot buffer must be page-aligned.")
    top_k = int(getattr(hf_text_config, "index_topk", hisparse_config.top_k))
    layout = build_hisparse_spec_layout(
        hot_slots=hisparse_config.device_buffer_size,
        tail_capacity_slots=page_size // compress_ratios,
        compress_ratio=compress_ratios,
        verify_width=server_args.speculative_num_draft_tokens,
        top_k=top_k,
    )
    return HiSparseSpecPlan(
        layout=layout,
        transfer_kind=(
            HiSparseKVTransferKind.LINEAR
            if is_dsa_mtp
            else HiSparseKVTransferKind.DSV4_PAGED
        ),
    )
