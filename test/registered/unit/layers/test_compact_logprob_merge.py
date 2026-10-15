"""Focused merge contract, including IEEE ordering and padded candidates."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.compact_logprob_merge import (
    merge_compact_input_state,
    pack_compact_input_state,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="These fused kernels are NVIDIA CUDA only; other paths use Torch.",
)


def test_merge_rejects_global_int32_overflow():
    fake = SimpleNamespace(
        shape=(2, 2**30),
        is_cuda=True,
        dtype=torch.float32,
        is_contiguous=lambda: True,
        numel=lambda: 2**31,
    )
    with pytest.raises(AssertionError):
        merge_compact_input_state(fake, rows=1, scores=2**30 - 4, k=1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("valid,k,empty", [(0, 5, False), (3, 20, False), (8, 5, True)])
def test_pack(dtype, valid, k, empty):
    torch.manual_seed(314159)
    rows, start, vocab = 17, 9, 100
    logits = torch.randn(rows, 16, device="cuda", dtype=dtype)[:, ::2]
    if not valid:
        logits = torch.empty(rows, 0, device="cuda", dtype=dtype)
    maxima = torch.randn(rows, device="cuda")
    logs = torch.rand(rows, device="cuda")
    local_k = min(k, valid)
    values, ids = logits[:, :valid].topk(local_k, dim=-1) if local_k else (None, None)
    lookup_rows = torch.arange(rows, device="cuda").repeat_interleave(8)[::2]
    lookup_ids = (
        torch.tensor(
            [start - 1, start, start + valid - 1, start + valid], device="cuda"
        )
        .repeat(rows)
        .repeat_interleave(2)[::2]
    )
    if empty:
        lookup_rows, lookup_ids = lookup_rows[:0], lookup_ids[:0]
    actual = pack_compact_input_state(
        logits,
        maxima,
        logs,
        values,
        ids,
        lookup_rows,
        lookup_ids,
        start,
        start + valid,
        vocab,
        k,
    )
    selected = torch.zeros(lookup_ids.numel(), device="cuda")
    owned = (lookup_ids >= start) & (lookup_ids < start + valid)
    if valid:
        selected[owned] = logits[lookup_rows[owned], lookup_ids[owned] - start].float()
    padded_values = torch.full((rows, k), -float("inf"), device="cuda")
    padded_ids = torch.full((rows, k), vocab, device="cuda", dtype=torch.float32)
    if local_k:
        padded_values[:, :local_k] = values.float()
        padded_ids[:, :local_k] = (ids + start).float()
    expected = torch.cat(
        (maxima, logs, selected, padded_values.flatten(), padded_ids.flatten())
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize(
    "tp,k,rows", [(1, 1, 1), (2, 5, 17), (4, 20, 128), (8, 128, 3), (32, 1, 17)]
)
@pytest.mark.parametrize("normalize", [False, True])
def test_merge(tp, k, rows, normalize):
    torch.manual_seed(314159)
    scores = 0 if rows == 1 else rows * 3
    maxima = torch.randn(tp, rows, device="cuda")
    logs = torch.rand_like(maxima)
    selected = torch.zeros(tp, scores, device="cuda")
    if scores:
        selected[
            torch.arange(scores, device="cuda") % tp,
            torch.arange(scores, device="cuda"),
        ] = torch.randn(scores, device="cuda")
    values = torch.randint(-2, 3, (tp, rows, k), device="cuda").float()
    ids = torch.arange(tp * k, device="cuda").view(tp, 1, k).expand(tp, rows, k).clone()
    if k >= 3:
        values[0, 0, 0] = float("nan")
        values[0, 0, 1] = 0.0
        values[0, 0, 2] = -0.0
        values[-1, :, -1] = -float("inf")
        ids[-1, :, -1] = 2**24
        values[0, :, -1] = -float("inf")
    if rows > 2:
        values[:, 1, :] = 0.0
        values[0, 1, 0] = -0.0
        values[:, 2, :] = -float("inf")
        maxima[0, 1] = float("nan")
        maxima[:, 2] = -float("inf")
    if rows > 3:
        maxima[0, 3] = float("inf")
    packed = torch.cat(
        (maxima, logs, selected, values.flatten(1), ids.flatten(1).float()), dim=1
    )
    lookup_rows = (torch.randperm(scores, device="cuda") % rows).repeat_interleave(2)[
        ::2
    ]
    actual = merge_compact_input_state(
        packed, rows, scores, k, lookup_rows=lookup_rows if normalize else None
    )
    expected_max = maxima.amax(dim=0)
    expected_logs = ((maxima - expected_max + logs).exp().sum(dim=0)).log()
    cv = values.permute(1, 0, 2).reshape(rows, -1)
    ci = ids.permute(1, 0, 2).reshape(rows, -1)
    order = ci.argsort(dim=-1, stable=True)
    cv, ci = cv.gather(1, order), ci.gather(1, order)
    order = cv.argsort(dim=-1, descending=True, stable=True)[:, :k]
    expected = (
        expected_max,
        expected_logs,
        selected.sum(dim=0),
        cv.gather(1, order),
        ci.gather(1, order),
    )
    if normalize:
        expected = (
            expected[0],
            expected[1],
            (expected[2] - expected[0][lookup_rows]) - expected[1][lookup_rows],
            (expected[3] - expected[0][:, None]) - expected[1][:, None],
            expected[4],
        )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5, equal_nan=True)
    if rows > 2 and not normalize:
        assert not torch.signbit(actual[3][1]).any()
        assert (actual[4][2] < 2**24).all()
