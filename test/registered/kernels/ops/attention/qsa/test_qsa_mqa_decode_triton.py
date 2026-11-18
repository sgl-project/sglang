"""The Triton QSA decode scoring kernel, which ``qsa_mqa_decode`` runs on ROCm,
against the torch reference: row by row, and grouped over a target-verify step's
draft rows, which share their request's page-table row."""

import sys

import pytest
import torch

import sglang.kernels.ops.attention.qsa_mqa_decode_triton as triton_decode
from sglang.kernels.ops.attention.qsa_mqa_decode_triton import (
    can_use_qsa_mqa_decode_triton,
    qsa_mqa_decode_triton,
)
from sglang.srt.layers.attention.qsa.mqa import qsa_mqa_decode, torch_qsa_mqa_decode
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _verify_inputs(lengths, draft, page_size, heads=4, padded_heads=8, head_dim=128):
    """One request per entry of ``lengths`` (compressed keys), with ``draft`` rows
    whose lengths step by one, all over the request's page-table row. q holds
    ``heads`` heads followed by zero padding, as the indexer pads them."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch = len(lengths)
    width = -(-(max(lengths) + draft) // page_size)
    num_pages = batch * width + 1
    k_cache = torch.randn(
        num_pages, page_size, 1, head_dim, dtype=torch.bfloat16, device=device
    )
    tables = (torch.randperm(num_pages - 1)[: batch * width] + 1).view(batch, width)
    page_table = tables.repeat_interleave(draft, dim=0).to(device, torch.int32)
    steps = torch.arange(draft).repeat(batch)
    context_lens = torch.tensor(lengths).repeat_interleave(draft) + steps
    q = torch.zeros(
        batch * draft, padded_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    q[:, :heads] = torch.randn(
        batch * draft, heads, head_dim, dtype=torch.bfloat16, device=device
    )
    return (
        q,
        k_cache,
        page_table,
        context_lens.to(device, torch.int32),
        width * page_size,
    )


def _assert_scored_close(actual, expected, context_lens, min_scored_len):
    if min_scored_len == 0:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        return
    # Unscored rows and the logits past each row's length are left unset.
    checked = 0
    for row, length in enumerate(context_lens.tolist()):
        if length > min_scored_len:
            torch.testing.assert_close(
                actual[row, :length], expected[row, :length], rtol=1e-4, atol=1e-4
            )
            checked += 1
    assert checked


@needs_gpu
@pytest.mark.parametrize("min_scored_len", [0, 512])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("draft,rows_per_group", [(4, 1), (4, 4), (3, 3), (1, 1)])
def test_qsa_mqa_decode_triton_matches_torch_reference(
    draft, rows_per_group, page_size, min_scored_len
):
    q, k_cache, page_table, context_lens, max_model_len = _verify_inputs(
        [300, 509, 1290, 2047, 5000], draft, page_size
    )
    assert can_use_qsa_mqa_decode_triton(
        q, k_cache, page_table, context_lens, max_model_len, rows_per_group, 4
    )
    actual = qsa_mqa_decode_triton(
        q,
        k_cache,
        page_table,
        context_lens,
        max_model_len,
        min_scored_len=min_scored_len,
        rows_per_group=rows_per_group,
        num_heads=4,
    )
    expected = torch_qsa_mqa_decode(q, k_cache, page_table, context_lens, max_model_len)
    _assert_scored_close(actual, expected, context_lens, min_scored_len)


@needs_gpu
def test_qsa_mqa_decode_triton_reads_every_head_by_default():
    q, k_cache, page_table, context_lens, max_model_len = _verify_inputs(
        [700, 1500], draft=2, page_size=64, heads=8, padded_heads=8
    )
    actual = qsa_mqa_decode_triton(
        q, k_cache, page_table, context_lens, max_model_len, 0.125, rows_per_group=2
    )
    expected = torch_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, 0.125
    )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_hip()), reason="ROCm dispatch"
)
def test_qsa_mqa_decode_dispatches_triton_on_rocm(monkeypatch):
    q, k_cache, page_table, context_lens, max_model_len = _verify_inputs(
        [700, 1500], draft=4, page_size=64
    )
    groups = []

    def spy(*args, **kwargs):
        groups.append(kwargs["rows_per_group"])
        return qsa_mqa_decode_triton(*args, **kwargs)

    monkeypatch.setattr(triton_decode, "qsa_mqa_decode_triton", spy)
    expected = torch_qsa_mqa_decode(q, k_cache, page_table, context_lens, max_model_len)
    actual = qsa_mqa_decode(
        q,
        k_cache,
        page_table,
        context_lens,
        max_model_len,
        rows_per_request=4,
        num_heads=4,
    )
    assert groups == [4]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_qsa_mqa_decode_triton_declines_unsupported_inputs():
    q = torch.zeros(8, 8, 128, dtype=torch.bfloat16)
    k_cache = torch.zeros(10, 64, 1, 128, dtype=torch.bfloat16)
    page_table = torch.zeros(8, 4, dtype=torch.int32)
    context_lens = torch.ones(8, dtype=torch.int32)

    def can_use(q=q, k_cache=k_cache, rows_per_group=4, num_heads=4):
        return can_use_qsa_mqa_decode_triton(
            q, k_cache, page_table, context_lens, 256, rows_per_group, num_heads
        )

    assert can_use()
    # fp8 scoring stays on TileLang.
    assert not can_use(
        q=q.to(torch.float8_e4m3fn), k_cache=k_cache.to(torch.float8_e4m3fn)
    )
    assert not can_use(rows_per_group=3)
    assert not can_use(num_heads=9)
    assert not can_use(q=torch.zeros(8, 40, 128, dtype=torch.bfloat16), num_heads=40)
    assert not can_use(
        q=torch.zeros(8, 8, 96, dtype=torch.bfloat16),
        k_cache=torch.zeros(10, 64, 1, 96, dtype=torch.bfloat16),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
