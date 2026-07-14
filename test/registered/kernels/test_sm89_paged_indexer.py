from __future__ import annotations

import inspect

import pytest
import torch

from sglang.kernels.ops.attention.dsa import sm89_paged_indexer
from sglang.kernels.ops.attention.dsa.sm89_paged_indexer import (
    sm89_paged_fp8_index_logits,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

PAGE_SIZE = 64
NUM_HEADS = 32
HEAD_DIM = 128
K_BYTES_PER_PAGE = PAGE_SIZE * HEAD_DIM
SCALE_BYTES_PER_PAGE = PAGE_SIZE * torch.float32.itemsize
PACKED_BYTES_PER_PAGE = K_BYTES_PER_PAGE + SCALE_BYTES_PER_PAGE


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_q(batch_size: int, *, phase: int = 0) -> torch.Tensor:
    q = torch.ones(
        batch_size, 1, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.float32
    )
    q[:, :, NUM_HEADS // 2 :] = -1
    if phase % 2:
        q.neg_()
    if phase % 3 == 2:
        q[:, :, :, 1::2].neg_()
    return q.to(torch.float8_e4m3fn).contiguous()


def _make_weights(batch_size: int, *, phase: int = 0) -> torch.Tensor:
    first_half = torch.tensor(
        [1.0, -0.5] * (NUM_HEADS // 4), device="cuda", dtype=torch.float32
    )
    second_half = torch.tensor(
        [0.25, -0.75] * (NUM_HEADS // 4),
        device="cuda",
        dtype=torch.float32,
    )
    weights = torch.cat((first_half, second_half)).repeat(batch_size, 1)
    if phase:
        weights = torch.roll(weights, shifts=phase, dims=1)
        weights[:, ::5].neg_()
    return weights.contiguous()


def _make_raw_cache(num_pages: int, *, phase: int = 0) -> torch.Tensor:
    cache = torch.zeros(
        num_pages, PACKED_BYTES_PER_PAGE, device="cuda", dtype=torch.uint8
    )
    token_ids = torch.arange(
        num_pages * PAGE_SIZE, device="cuda", dtype=torch.int64
    ).view(num_pages, PAGE_SIZE)
    signs = torch.where(
        (token_ids + phase) % 3 == 0,
        torch.tensor(-1.0, device="cuda"),
        torch.tensor(1.0, device="cuda"),
    )
    dimension_signs = torch.ones(HEAD_DIM, device="cuda", dtype=torch.float32)
    if phase % 3 == 2:
        dimension_signs[1::2] = -1
    keys = (signs[..., None] * dimension_signs).to(torch.float8_e4m3fn)
    cache[:, :K_BYTES_PER_PAGE].view(torch.float8_e4m3fn).view(
        num_pages, PAGE_SIZE, HEAD_DIM
    ).copy_(keys)

    scales = 1.25 + (token_ids.to(torch.float32) + phase) / 1024.0
    cache[:, K_BYTES_PER_PAGE:].view(torch.float32).copy_(scales)
    return cache


def _gather_packed_cache(
    kv_cache_u8: torch.Tensor, page_table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_pages = kv_cache_u8.shape[0]
    keys = (
        kv_cache_u8[:, :K_BYTES_PER_PAGE]
        .view(torch.float8_e4m3fn)
        .view(num_pages, PAGE_SIZE, HEAD_DIM)
    )
    scales = kv_cache_u8[:, K_BYTES_PER_PAGE:].view(torch.float32)
    valid_pages = (page_table >= 0) & (page_table < num_pages)
    safe_pages = torch.where(valid_pages, page_table, 0).to(torch.long)
    return keys[safe_pages], scales[safe_pages], valid_pages


def _reference_logits(
    q_fp8: torch.Tensor,
    kv_cache_u8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    batch_size = q_fp8.shape[0]
    selected_keys, selected_scales, valid_pages = _gather_packed_cache(
        kv_cache_u8, page_table
    )
    selected_keys = selected_keys.reshape(batch_size, max_seq_len, HEAD_DIM)
    selected_scales = selected_scales.reshape(batch_size, max_seq_len)
    dots = torch.matmul(
        selected_keys.to(torch.float32),
        q_fp8[:, 0].to(torch.float32).transpose(1, 2),
    )
    logits = (torch.relu(dots) * weights[:, None, :]).sum(dim=-1) * selected_scales

    positions = torch.arange(max_seq_len, device=q_fp8.device)[None, :]
    valid_tokens = (
        valid_pages[..., None]
        .expand(-1, -1, PAGE_SIZE)
        .reshape(batch_size, max_seq_len)
    )
    valid_tokens &= positions < seq_lens.reshape(batch_size, 1)
    return logits.masked_fill(~valid_tokens, float("-inf"))


def _sorted_topk(logits: torch.Tensor, k: int = 8) -> torch.Tensor:
    return torch.topk(logits, k=k, dim=-1).indices.sort(dim=-1).values


@pytest.fixture
def valid_case() -> dict[str, object]:
    batch_size = 2
    max_pages = 3
    return {
        "q_fp8": _make_q(batch_size),
        "kv_cache_u8": _make_raw_cache(5),
        "weights": _make_weights(batch_size),
        "seq_lens": torch.tensor([130, 159], device="cuda", dtype=torch.int32),
        "page_table": torch.tensor(
            [[2, 0, -1], [1, 5, 3]], device="cuda", dtype=torch.int32
        ),
        "max_seq_len": max_pages * PAGE_SIZE,
    }


def test_matches_packed_page_reference_and_masks_invalid_positions(valid_case):
    logits = sm89_paged_fp8_index_logits(**valid_case)
    expected = _reference_logits(**valid_case)

    assert logits.shape == (2, 3 * PAGE_SIZE)
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-5)
    assert torch.equal(_sorted_topk(logits), _sorted_topk(expected))

    assert torch.isfinite(logits[0, : 2 * PAGE_SIZE]).all()
    assert torch.isneginf(logits[0, 2 * PAGE_SIZE :]).all()
    assert torch.isfinite(logits[1, :PAGE_SIZE]).all()
    assert torch.isneginf(logits[1, PAGE_SIZE : 2 * PAGE_SIZE]).all()
    assert torch.isfinite(logits[1, 2 * PAGE_SIZE : 159]).all()
    assert torch.isneginf(logits[1, 159:]).all()


def test_applies_relu_per_head_before_signed_weight(valid_case):
    q_fp8 = valid_case["q_fp8"]
    kv_cache_u8 = valid_case["kv_cache_u8"]
    weights = valid_case["weights"]
    page_table = valid_case["page_table"]
    max_seq_len = valid_case["max_seq_len"]
    selected_keys, selected_scales, valid_pages = _gather_packed_cache(
        kv_cache_u8, page_table
    )
    dots = torch.matmul(
        selected_keys.reshape(2, max_seq_len, HEAD_DIM).float(),
        q_fp8[:, 0].float().transpose(1, 2),
    )

    assert (dots > 0).any(dim=-1).all()
    assert (dots < 0).any(dim=-1).all()
    assert (weights > 0).any() and (weights < 0).any()
    assert torch.all(selected_scales > 0)
    assert not torch.any(selected_scales == 1)

    correct = (torch.relu(dots) * weights[:, None, :]).sum(dim=-1)
    wrong = torch.relu((dots * weights[:, None, :]).sum(dim=-1))
    valid = valid_pages[..., None].expand(-1, -1, PAGE_SIZE).reshape(2, max_seq_len)
    assert not torch.allclose(correct[valid], wrong[valid])


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("q_fp8", lambda case: case["q_fp8"].float(), "q_fp8.*float8_e4m3fn"),
        ("q_fp8", lambda case: case["q_fp8"][:, 0], "q_fp8.*shape"),
        ("q_fp8", lambda case: case["q_fp8"][:, :, :31], "q_fp8.*shape"),
        ("q_fp8", lambda case: case["q_fp8"][..., :64], "q_fp8.*shape"),
        (
            "q_fp8",
            lambda case: torch.empty(
                2,
                1,
                NUM_HEADS,
                HEAD_DIM * 2,
                device="cuda",
                dtype=torch.float8_e4m3fn,
            )[..., ::2],
            "q_fp8.*contiguous",
        ),
        (
            "kv_cache_u8",
            lambda case: case["kv_cache_u8"].view(torch.float8_e4m3fn),
            "kv_cache_u8.*uint8",
        ),
        (
            "kv_cache_u8",
            lambda case: case["kv_cache_u8"][:, :-1].contiguous(),
            "kv_cache_u8.*shape",
        ),
        (
            "kv_cache_u8",
            lambda case: torch.empty(
                5, PACKED_BYTES_PER_PAGE * 2, device="cuda", dtype=torch.uint8
            )[:, ::2],
            "kv_cache_u8.*contiguous",
        ),
        (
            "weights",
            lambda case: case["weights"].to(torch.bfloat16),
            "weights.*float32",
        ),
        ("weights", lambda case: case["weights"][:, :-1], "weights.*shape"),
        (
            "weights",
            lambda case: torch.empty(
                2, NUM_HEADS * 2, device="cuda", dtype=torch.float32
            )[:, ::2],
            "weights.*contiguous",
        ),
        ("seq_lens", lambda case: case["seq_lens"].long(), "seq_lens.*int32"),
        (
            "seq_lens",
            lambda case: case["seq_lens"][:, None].expand(-1, 2).contiguous(),
            "seq_lens.*shape",
        ),
        ("seq_lens", lambda case: case["seq_lens"].cpu(), "same CUDA device"),
        ("page_table", lambda case: case["page_table"].long(), "page_table.*int32"),
        ("page_table", lambda case: case["page_table"][:1], "page_table.*shape"),
        ("page_table", lambda case: case["page_table"].cpu(), "same CUDA device"),
        ("max_seq_len", lambda case: True, "max_seq_len.*integer"),
        ("max_seq_len", lambda case: 2 * PAGE_SIZE, "max_seq_len.*page_table"),
    ],
)
def test_rejects_invalid_contract(valid_case, field, replacement, message):
    invalid_case = dict(valid_case)
    invalid_case[field] = replacement(valid_case)

    with pytest.raises((TypeError, ValueError), match=message):
        sm89_paged_fp8_index_logits(**invalid_case)


def test_accepts_column_device_lengths(valid_case):
    valid_case["seq_lens"] = valid_case["seq_lens"][:, None]
    logits = sm89_paged_fp8_index_logits(**valid_case)
    expected = _reference_logits(**valid_case)
    torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-5)


def test_accepts_noncontiguous_page_table(valid_case):
    page_table_storage = torch.tensor(
        [[2, 4, 0, 4, -1, 4], [1, 4, 5, 4, 3, 4]],
        device="cuda",
        dtype=torch.int32,
    )
    valid_case["page_table"] = page_table_storage[:, ::2]
    assert not valid_case["page_table"].is_contiguous()

    logits = sm89_paged_fp8_index_logits(**valid_case)
    expected = _reference_logits(**valid_case)
    torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-5)
    assert torch.equal(_sorted_topk(logits), _sorted_topk(expected))


def test_source_has_no_host_device_reads():
    source = inspect.getsource(sm89_paged_indexer)
    for forbidden in (".item(", ".tolist(", ".cpu(", ".numpy("):
        assert forbidden not in source


def test_cuda_graph_replays_short_long_and_long_short_with_stable_pointers():
    max_seq_len = 2 * PAGE_SIZE
    static_q = _make_q(1, phase=0)
    static_cache = _make_raw_cache(4, phase=0)
    static_weights = _make_weights(1, phase=0)
    static_seq_lens = torch.tensor([37], device="cuda", dtype=torch.int32)
    static_page_table = torch.tensor([[0, -1]], device="cuda", dtype=torch.int32)
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (
            static_q,
            static_cache,
            static_weights,
            static_seq_lens,
            static_page_table,
        )
    )

    states = [
        {
            "q_fp8": _make_q(1, phase=1),
            "kv_cache_u8": _make_raw_cache(4, phase=1),
            "weights": _make_weights(1, phase=1),
            "seq_lens": torch.tensor([111], device="cuda", dtype=torch.int32),
            "page_table": torch.tensor([[2, 1]], device="cuda", dtype=torch.int32),
        },
        {
            "q_fp8": _make_q(1, phase=2),
            "kv_cache_u8": _make_raw_cache(4, phase=2),
            "weights": _make_weights(1, phase=2),
            "seq_lens": torch.tensor([19], device="cuda", dtype=torch.int32),
            "page_table": torch.tensor([[3, 0]], device="cuda", dtype=torch.int32),
        },
    ]

    sm89_paged_fp8_index_logits(
        static_q,
        static_cache,
        static_weights,
        static_seq_lens,
        static_page_table,
        max_seq_len,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_logits = sm89_paged_fp8_index_logits(
            static_q,
            static_cache,
            static_weights,
            static_seq_lens,
            static_page_table,
            max_seq_len,
        )
        replay_topk = torch.topk(replay_logits, k=8, dim=-1).indices

    graph.replay()
    torch.cuda.synchronize()
    initial_expected = _reference_logits(
        static_q,
        static_cache,
        static_weights,
        static_seq_lens,
        static_page_table,
        max_seq_len,
    )
    torch.testing.assert_close(replay_logits, initial_expected, atol=1e-4, rtol=1e-5)
    assert torch.equal(
        replay_topk.sort(dim=-1).values.to(torch.int64),
        _sorted_topk(initial_expected),
    )

    replay_snapshots = []
    for state in states:
        static_q.copy_(state["q_fp8"])
        static_cache.copy_(state["kv_cache_u8"])
        static_weights.copy_(state["weights"])
        static_seq_lens.copy_(state["seq_lens"])
        static_page_table.copy_(state["page_table"])

        graph.replay()
        torch.cuda.synchronize()
        expected = _reference_logits(
            static_q,
            static_cache,
            static_weights,
            static_seq_lens,
            static_page_table,
            max_seq_len,
        )
        actual = replay_logits.clone()
        replay_snapshots.append(actual)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)
        assert torch.equal(
            replay_topk.sort(dim=-1).values.to(torch.int64),
            _sorted_topk(expected),
        )
        assert pointers == tuple(
            tensor.data_ptr()
            for tensor in (
                static_q,
                static_cache,
                static_weights,
                static_seq_lens,
                static_page_table,
            )
        )

    assert torch.isfinite(replay_snapshots[0][0, :111]).all()
    assert torch.isneginf(replay_snapshots[1][0, 19:]).all()
