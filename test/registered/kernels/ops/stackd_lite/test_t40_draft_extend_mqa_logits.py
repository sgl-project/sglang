"""T40 parity: native next_n DeepGEMM paged MQA logits for draft extend v2
(SGLANG_OPT_DSA_DRAFT_EXTEND_NATIVE_MQA_LOGITS) against the split path the T38 tree runs.

Both arms use the production layouts: the schedule builder's ``[B, next_n]`` context lengths
(native) or the per-row expanded lengths (split), the expanded page table, and the indexer's
DeepGEMM wrappers. On every row's causal range the logits and the index top-k must agree; both
must match the FP32 reference.
"""

import json
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="SM 10.x only",
)

from sglang.kernels.ops.attention.dsa import (  # noqa: E402
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.attention import dsa_backend  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.test.kernels.paged_mqa import (  # noqa: E402
    BLOCK_KV,
    HEAD_DIM,
    assert_paged_mqa_matches_ref,
    generate_paged_mqa_test_data,
    ref_fp8_paged_mqa_logits,
)

NEXT_N = 4  # EAGLE 3 steps, 4 draft tokens: draft extend v2 extends every request by 4
HEADS = 32  # GLM-5.2 index_n_heads
INDEX_TOPK = 2048
GRAPH_BS = (1, 2, 4, 8, 12, 16, 20, 24)


def _record(**fields) -> None:
    path = os.environ.get("T40_PARITY_JSONL")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"test": "draft_extend_mqa_logits", **fields}) + "\n")


def _schedule_ctx(native: bool, cache_seqlens, seqlens_expanded, bs):
    """The production builder (DeepseekSparseAttnBackend._build_paged_mqa_schedule_2d_ctx_lens)."""
    fake = SimpleNamespace(speculative_num_draft_tokens=NEXT_N)
    with envs.SGLANG_OPT_DSA_DRAFT_EXTEND_NATIVE_MQA_LOGITS.override(native), mock.patch.object(
        dsa_backend, "get_platform", lambda: SimpleNamespace(is_sm100=True)
    ):
        return dsa_backend.DeepseekSparseAttnBackend._build_paged_mqa_schedule_2d_ctx_lens(
            fake, ForwardMode.DRAFT_EXTEND_V2, cache_seqlens, seqlens_expanded, bs
        )


def _valid_mask(context_lens, bs, max_len):
    pos = torch.arange(max_len, device="cuda").unsqueeze(0)
    row = torch.arange(bs * NEXT_N, device="cuda")
    end = context_lens[row // NEXT_N] - NEXT_N + row % NEXT_N
    return pos <= end.unsqueeze(1)


@pytest.mark.parametrize("bs", GRAPH_BS)
@pytest.mark.parametrize("avg_ctx", [1024, 16384])
def test_native_matches_split(bs, avg_ctx):
    import deep_gemm

    max_len = max(avg_ctx * 2, 2048)
    data = generate_paged_mqa_test_data(bs, NEXT_N, HEADS, avg_ctx, max_len)
    ctx = data["context_lens"].to(torch.int32)
    seqlens_expanded = (
        ctx.unsqueeze(-1) - NEXT_N + torch.arange(1, NEXT_N + 1, device="cuda", dtype=torch.int32)
    ).reshape(-1)
    ctx_native = _schedule_ctx(True, ctx, seqlens_expanded, bs)
    ctx_split = _schedule_ctx(False, ctx, seqlens_expanded, bs)
    assert ctx_native.shape == (bs, NEXT_N) and ctx_split.shape == (bs * NEXT_N, 1)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    q = data["q_fp8"].view(bs * NEXT_N, HEADS, HEAD_DIM)
    tables = data["block_table"].repeat_interleave(NEXT_N, dim=0)
    native = deepgemm_paged_mqa_logits_native(
        deep_gemm.fp8_paged_mqa_logits, q, data["kv_fused"], data["weights"], ctx_native,
        tables, deep_gemm.get_paged_mqa_logits_metadata(ctx_native, BLOCK_KV, num_sms),
        max_len, q_offset=bs * NEXT_N, B=bs, next_n=NEXT_N,
    )
    split = deepgemm_paged_mqa_logits_split(
        deep_gemm.fp8_paged_mqa_logits, q, data["kv_fused"], data["weights"], ctx_split,
        tables, deep_gemm.get_paged_mqa_logits_metadata(ctx_split, BLOCK_KV, num_sms),
        max_len, q_offset=bs * NEXT_N,
    )
    mask = _valid_mask(ctx, bs, max_len)
    a = native.float().masked_fill(~mask, float("-inf"))
    b = split.float().masked_fill(~mask, float("-inf"))
    finite = mask
    assert torch.isfinite(native.float()[finite]).all() and torch.isfinite(split.float()[finite]).all()
    torch.testing.assert_close(a[finite], b[finite], atol=5e-5, rtol=1e-5)
    bitwise = (a[finite] == b[finite]).float().mean().item()
    k = min(INDEX_TOPK, max_len)
    va, ia = a.topk(k, dim=-1)
    vb, ib = b.topk(k, dim=-1)
    torch.testing.assert_close(va, vb, atol=5e-5, rtol=1e-5)
    same_set = (ia.sort(-1).values == ib.sort(-1).values).all(-1).float().mean().item()
    # Index sets may differ only through ties or last-ulp value differences at the k-th value.
    assert same_set >= 0.95, same_set
    if avg_ctx <= 1024:
        ref = ref_fp8_paged_mqa_logits(
            data["q_fp8"], data["kv_fp8"], data["kv_scales"], data["weights"],
            data["context_lens"], data["block_table"], max_len, BLOCK_KV,
        )
        assert_paged_mqa_matches_ref(native, ref, data["context_lens"], bs, NEXT_N, max_len)
        assert_paged_mqa_matches_ref(split, ref, data["context_lens"], bs, NEXT_N, max_len)
    _record(bs=bs, avg_ctx=avg_ctx, bitwise_frac=bitwise, topk_same_set_frac=same_set)


def test_builder_keeps_split_layout_off_or_uneven():
    """Env off, or a row count that is not bs * next_n, keeps the per-row layout."""
    ctx = torch.tensor([100, 200], dtype=torch.int32, device="cuda")
    rows = torch.arange(1, 9, dtype=torch.int32, device="cuda")
    assert _schedule_ctx(False, ctx, rows, 2).shape == (8, 1)
    assert _schedule_ctx(True, ctx, rows, 2).shape == (2, NEXT_N)
    assert _schedule_ctx(True, ctx, rows[:7], 2).shape == (7, 1)
