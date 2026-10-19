"""Router top-k and trtllm-gen MoE routing tables for up to 64 tokens in one launch.

The outputs are the precomputed routing metadata FlashInfer's trtllm-gen MoE runs its FC1 / FC2 body
from (``da_routing_metadata``), so FlashInfer's routing kernels are skipped. Expert ids and weights equal
FlashInfer's routing (TopK -> Softmax) bit for bit; the tables equal it up to the order of the tokens
inside one expert's rows, which FlashInfer's own routing does not fix either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

NUM_EXPERTS = 512
TOP_K = 10
MAX_TOKENS = 64
# A second kernel instance for 64 < M <= MAX_TOKENS_WIDE (SGLANG_Q35MF_K4_PARQ_MAX_TOKENS).
MAX_TOKENS_WIDE = 128


@cache_once
def _jit_module(max_tokens: int = MAX_TOKENS) -> Module:
    if max_tokens == MAX_TOKENS:
        args = make_cpp_args(is_arch_support_pdl())
        name = "qwen_router_tables"
    else:
        args = make_cpp_args(is_arch_support_pdl(), max_tokens)
        name = f"qwen_router_tables_{max_tokens}"
    return load_jit(
        name,
        *args,
        cuda_files=["moe/qwen_router_tables.cuh"],
        cuda_wrappers=[("run", f"RouterTopkTables<{args}>::run")],
        # FlashInfer builds its routing with fast math; the softmax must lower to the same instructions.
        extra_cuda_cflags=["-O3", "-use_fast_math"],
    )


@cache_once
def _jit_module_small(early_trigger: bool = False) -> Module:
    args = make_cpp_args(is_arch_support_pdl(), early_trigger)
    return load_jit(
        "qwen_router_tables_small",
        *args,
        cuda_files=["moe/qwen_router_tables_small.cuh"],
        cuda_wrappers=[("run", f"RouterTopkTablesSmall<{args}>::run")],
        # FlashInfer builds its routing with fast math; the softmax must lower to the same instructions.
        extra_cuda_cflags=["-O3", "-use_fast_math"],
    )


def max_ctas(num_tokens: int, tile_n: int) -> int:
    """FlashInfer getMaxNumCtasInBatchDim for 512 experts and top-10."""
    expanded = num_tokens * TOP_K
    filled = min(NUM_EXPERTS, expanded)
    return filled + (expanded - filled) // tile_n


class RouterTables:
    """Routing metadata of one call site, in FlashInfer's RoutingMetadataBuffers order; reuse across
    CUDA graph replays."""

    def __init__(
        self, num_tokens: int, tile_n: int, device: torch.device | str = "cuda"
    ) -> None:
        assert 0 < num_tokens <= MAX_TOKENS_WIDE
        assert tile_n & (tile_n - 1) == 0 and 8 <= tile_n <= 256
        m, ctas = num_tokens, max_ctas(num_tokens, tile_n)
        i32 = dict(dtype=torch.int32, device=device)
        self.num_tokens = m
        self.tile_n = tile_n
        self.tile_log2 = tile_n.bit_length() - 1
        self.total_num_padded_tokens = torch.zeros(1, **i32)
        self.expanded_idx_to_permuted_idx = torch.zeros(m * TOP_K, **i32)
        # One guard element past the largest padded size (FlashInfer's metadata ABI).
        self.permuted_idx_to_token_idx = torch.full((ctas * tile_n + 1,), -1, **i32)
        self.expert_weights = torch.zeros(
            (m, TOP_K), dtype=torch.bfloat16, device=device
        )
        self.expert_count_histogram = torch.zeros(2 * NUM_EXPERTS, **i32)
        self.num_tokens_per_expert = torch.zeros(NUM_EXPERTS, **i32)
        self.cta_idx_xy_to_batch_idx = torch.zeros(ctas, **i32)
        self.cta_idx_xy_to_mn_limit = torch.zeros(ctas, **i32)
        self.num_non_exiting_ctas = torch.zeros(1, **i32)
        self.topk_ids = torch.zeros((m, TOP_K), **i32)
        # Monotonic counter; only its value modulo the token count matters.
        self.arrival = torch.zeros(1, **i32)

    def metadata(self) -> List[torch.Tensor]:
        """The nine tensors FlashInfer takes as ``da_routing_metadata``."""
        return [
            self.total_num_padded_tokens,
            self.expanded_idx_to_permuted_idx,
            self.permuted_idx_to_token_idx,
            self.expert_weights,
            self.expert_count_histogram,
            self.num_tokens_per_expert,
            self.cta_idx_xy_to_batch_idx,
            self.cta_idx_xy_to_mn_limit,
            self.num_non_exiting_ctas,
        ]


def router_topk_tables(logits: torch.Tensor, out: RouterTables) -> None:
    """Top-10 experts and softmax weights of each token's router logits ([M, 512] bf16) and the routing
    tables for ``out.tile_n``-token tiles, written into ``out``."""
    _jit_module(MAX_TOKENS if out.num_tokens <= MAX_TOKENS else MAX_TOKENS_WIDE).run(
        logits,
        out.expert_weights,
        out.topk_ids,
        out.expanded_idx_to_permuted_idx,
        out.permuted_idx_to_token_idx,
        out.cta_idx_xy_to_batch_idx,
        out.cta_idx_xy_to_mn_limit,
        out.num_non_exiting_ctas,
        out.total_num_padded_tokens,
        out.num_tokens_per_expert,
        out.arrival,
        out.tile_log2,
    )


SMALL_MAX_TOKENS = 16


def router_topk_tables_small(
    logits: torch.Tensor, out: RouterTables, early_trigger: bool = False
) -> None:
    """router_topk_tables for up to 16 tokens in one CTA; the rows inside each expert's range are in ascending
    token order, as FlashInfer's single-block routing kernels write them. ``early_trigger`` lets the dependent
    launch start as soon as this kernel's producer has completed."""
    _jit_module_small(early_trigger).run(
        logits,
        out.expert_weights,
        out.topk_ids,
        out.expanded_idx_to_permuted_idx,
        out.permuted_idx_to_token_idx,
        out.cta_idx_xy_to_batch_idx,
        out.cta_idx_xy_to_mn_limit,
        out.num_non_exiting_ctas,
        out.total_num_padded_tokens,
        out.num_tokens_per_expert,
        out.tile_log2,
    )
