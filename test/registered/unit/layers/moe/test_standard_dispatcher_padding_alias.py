"""Regression test for the -1 sentinel alias in StandardDispatcher.

Reproduces the production failure from sgl-project/sglang#43324: under EP with
the standard dispatch path, a CUDA-graph padded speculative-verify batch masks
rows past num_token_non_padded to -1, and advanced indexing through
local_expert_mapping wrapped that -1 to the table's final entry (= the last
local expert of the last EP rank). The masked-count kernels then piled
top_k * num_padded_rows fake assignments onto one expert and overflowed the
masked-GEMM per-expert slab.
"""

from unittest.mock import patch

import torch

from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _build_last_rank_dispatcher() -> StandardDispatcher:
    dispatcher = object.__new__(StandardDispatcher)
    dispatcher.moe_ep_size = 2
    dispatcher.moe_ep_rank = 1
    dispatcher.num_experts = 288
    dispatcher.num_local_experts = 144
    dispatcher.num_local_routed_experts = 144
    dispatcher.num_local_shared_experts = 0
    dispatcher.skip_local_expert_mapping = False
    dispatcher.use_aiter_moe_runner = False
    dispatcher.expert_mask_gpu = None
    mapping = torch.full((288,), -1, dtype=torch.int32)
    mapping[144:288] = torch.arange(0, 144, dtype=torch.int32)
    dispatcher.local_expert_mapping = mapping
    return dispatcher


def test_padded_verify_rows_keep_negative_sentinel() -> None:
    torch.manual_seed(0)
    rows, top_k, num_valid = 512, 8, 8
    topk_ids = torch.full((rows, top_k), -1, dtype=torch.int32)
    valid = torch.randint(0, 288, (num_valid, top_k), dtype=torch.int32)
    valid[0, 1::2] = -1  # non-local experts already project to -1 as well
    valid[3, :] = 287  # routes that legitimately land on local expert 143
    topk_ids[:num_valid] = valid
    topk_output = StandardTopKOutput(
        topk_weights=torch.rand((rows, top_k), dtype=torch.float32),
        topk_ids=topk_ids,
        router_logits=None,
    )

    dispatcher = _build_last_rank_dispatcher()
    with patch(
        "sglang.srt.layers.moe.token_dispatcher.standard."
        "should_use_flashinfer_cutlass_moe_fp4_allgather",
        return_value=False,
    ):
        output = dispatcher.dispatch(torch.ones((rows, 16)), topk_output)

    local_ids = output.topk_output.topk_ids
    assert (local_ids[num_valid:] == -1).all(), (
        "every padded slot must survive translation as the -1 drop sentinel "
        "instead of aliasing to the mapping table's last entry"
    )
    counts = torch.bincount(local_ids[local_ids >= 0].to(torch.int64), minlength=144)
    assert int(counts.max()) <= num_valid * top_k, (
        f"no expert may absorb the padded rows; max count {int(counts.max())}"
    )
    # The table-tail expert only holds its legitimate assignments from the
    # valid rows (global 287 -> local 143 on the last rank).
    assert int(counts[143]) == int((valid == 287).sum())
    # Legitimate local routes are still translated.
    expect = valid.clone()
    expect[expect >= 144] -= 144
    expect[(valid >= 0) & (valid < 144)] = -1
    assert torch.equal(local_ids[:num_valid], expect)
