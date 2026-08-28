"""Correctness and stream tests for QSA sparse prefill."""

import sys
from unittest import mock

import pytest
import torch

from sglang.srt.layers.attention.qsa import sparse_attn
from sglang.srt.layers.attention.qsa.sparse_attn import (
    _SM120_PREFILL_CONFIGS,
    _get_prefill_config,
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.srt.utils import is_sm120_supported, is_sm121
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")
register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 256
TOPK = 2051
SCALE = HEAD_DIM**-0.5


def _clear_selector_caches():
    sparse_attn._get_prefill_device_configs.cache_clear()
    sparse_attn._triton_supports_sm120_prefill.cache_clear()
    sparse_attn._sm120_multiprocessor_count.cache_clear()


@pytest.fixture(autouse=True)
def _reset_selector_caches():
    _clear_selector_caches()
    yield
    _clear_selector_caches()


def _production_tuned_eligible():
    device_index = torch.cuda.current_device()
    return (
        is_sm120_supported()
        and not is_sm121()
        and sparse_attn._triton_supports_sm120_prefill()
        and sparse_attn._sm120_multiprocessor_count(device_index)
        in sparse_attn._SM120_PREFILL_SMS
    )


@pytest.fixture(autouse=True)
def _require_production_tuned_sm120_runner():
    if (
        torch.cuda.is_available()
        and is_sm120_supported()
        and not is_sm121()
        and sparse_attn._triton_supports_sm120_prefill()
    ):
        assert (
            _production_tuned_eligible()
        ), "SM120 sparse prefill CI requires a production-admitted GPU"


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("QSA sparse prefill requires CUDA")


def _make_case(q_lens, prefix_lens, q_heads, kv_heads, *, seed=810, selected_count=67):
    torch.manual_seed(seed + q_heads + kv_heads + sum(q_lens) + sum(prefix_lens))
    device = torch.device("cuda")
    kv_lens = [q_len + prefix for q_len, prefix in zip(q_lens, prefix_lens)]
    total_q = sum(q_lens)
    total_k = sum(kv_lens)
    q = torch.randn(total_q, q_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, kv_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    cu_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cu_k = torch.tensor(
        [0, *torch.tensor(kv_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    indices = torch.full((total_q, TOPK), -1, dtype=torch.int32, device=device)
    row = 0
    for q_len, prefix, kv_len in zip(q_lens, prefix_lens, kv_lens):
        for relative in range(q_len):
            visible = prefix + relative + 1
            count = min(visible, selected_count)
            if count:
                chosen = torch.randperm(visible, device=device, dtype=torch.int32)[
                    :count
                ]
                indices[row, :count] = chosen
            invalid_end = min(count + 4, TOPK)
            indices[row, count:invalid_end] = torch.tensor(
                [-1, -7, kv_len, kv_len + 19], dtype=torch.int32, device=device
            )[: invalid_end - count]
            row += 1
    return q, k, v, indices, cu_q, cu_k, kv_lens_tensor


def _reference(case, prefix_lens):
    q, k, v, indices, cu_q, cu_k, kv_lens = case
    output = torch.empty_like(q)
    group_size = q.shape[1] // k.shape[1]
    for batch, prefix in enumerate(prefix_lens):
        q_start = int(cu_q[batch])
        q_end = int(cu_q[batch + 1])
        k_start = int(cu_k[batch])
        kv_len = int(kv_lens[batch])
        for begin in range(q_start, q_end, 16):
            end = min(begin + 16, q_end)
            selected = indices[begin:end]
            valid = (selected >= 0) & (selected < kv_len)
            safe = selected.clamp(0, kv_len - 1).long()
            for kv_head in range(k.shape[1]):
                head_start = kv_head * group_size
                head_end = head_start + group_size
                keys = k[k_start + safe, kv_head].float()
                values = v[k_start + safe, kv_head].float()
                scores = torch.einsum(
                    "bhd,bkd->bhk",
                    q[begin:end, head_start:head_end].float(),
                    keys,
                )
                scores.mul_(SCALE).masked_fill_(~valid[:, None], -float("inf"))
                probabilities = torch.softmax(scores, dim=-1)
                output[begin:end, head_start:head_end] = torch.einsum(
                    "bhk,bkd->bhd", probabilities, values
                ).to(q.dtype)
    return output


def _run(case, q_lens, prefix_lens):
    q, k, v, indices, cu_q, cu_k, kv_lens = case
    if any(prefix_lens):
        return sparse_gqa_fwd_interface_triton_ck(
            q, k, v, indices, cu_q, cu_k, kv_lens, SCALE
        )
    return sparse_gqa_fwd_interface_triton(q, k, v, max(q_lens), indices, cu_q, SCALE)


@pytest.mark.parametrize(
    "q_lens,prefix_lens",
    [
        ([9], [0]),
        ([7, 11], [0, 0]),
        ([1, 2, 3, 4, 5, 6, 7, 8], [0] * 8),
        ([9], [3]),
        ([7, 11], [4, 7]),
        ([1, 2, 3, 4, 5, 6, 7, 8], [3, 4, 7, 8, 11, 12, 15, 16]),
    ],
)
@pytest.mark.parametrize("q_heads,kv_heads", [(3, 1), (6, 1), (12, 1), (12, 2)])
def test_qsa_sparse_prefill_matches_fp32_reference(
    q_lens, prefix_lens, q_heads, kv_heads
):
    _require_cuda()
    case = _make_case(q_lens, prefix_lens, q_heads, kv_heads)
    actual = _run(case, q_lens, prefix_lens)
    expected = _reference(case, prefix_lens)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
    repeated = _run(case, q_lens, prefix_lens)
    assert torch.equal(actual, repeated)


@pytest.mark.parametrize("prefix", [0, 3])
def test_qsa_sparse_prefill_side_stream(prefix):
    _require_cuda()
    q_lens = [13, 5]
    prefix_lens = [prefix, prefix]
    case = _make_case(q_lens, prefix_lens, 6, 1, seed=820)
    producer_done = torch.cuda.Event()
    producer_done.record(torch.cuda.current_stream())
    stream = torch.cuda.Stream()
    stream.wait_event(producer_done)
    with torch.cuda.stream(stream):
        actual = _run(case, q_lens, prefix_lens)
        actual.record_stream(stream)
    stream.synchronize()
    expected = _reference(case, prefix_lens)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("q_heads,geometry", _SM120_PREFILL_CONFIGS.items())
@pytest.mark.parametrize("prefix", [0, TOPK - 1])
def test_qsa_sparse_prefill_tuned_steady_state(q_heads, geometry, prefix):
    _require_cuda()
    if not sparse_attn._triton_supports_sm120_prefill():
        pytest.skip("tuned tuples need the Triton floor the production gate uses")
    q_lens = [1]
    prefix_lens = [prefix]
    case = _make_case(
        q_lens,
        prefix_lens,
        q_heads,
        1,
        seed=830,
        selected_count=TOPK,
    )
    with mock.patch.object(sparse_attn, "_get_prefill_config", return_value=geometry):
        actual = _run(case, q_lens, prefix_lens)
    expected = _reference(case, prefix_lens)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("q_heads", _SM120_PREFILL_CONFIGS)
def test_qsa_sparse_prefill_launch_geometry(monkeypatch, q_heads):
    _require_cuda()
    total_q = 513
    q = torch.empty(total_q, q_heads, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.empty(total_q, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.empty_like(k)
    indices = torch.empty(total_q, TOPK, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.tensor([0, total_q], dtype=torch.int32, device="cuda")
    kv_lens = torch.tensor([total_q], dtype=torch.int32, device="cuda")
    launches = []

    class LaunchRecorder:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*_args, **kwargs):
                geometry = tuple(
                    kwargs[name]
                    for name in ("BLOCK_M", "BLOCK_N", "num_warps", "num_stages")
                )
                launches.append((self.name, geometry))

            return launch

    monkeypatch.setenv("SGLANG_QSA_PREFILL_GEOMETRY", "auto")
    monkeypatch.setattr(sparse_attn, "_sparse_gqa_prefill", LaunchRecorder("non-chunk"))
    monkeypatch.setattr(
        sparse_attn, "_sparse_gqa_chunk_prefill", LaunchRecorder("chunk")
    )

    sparse_gqa_fwd_interface_triton(q, k, v, total_q, indices, cu_seqlens, SCALE)
    sparse_gqa_fwd_interface_triton_ck(
        q, k, v, indices, cu_seqlens, cu_seqlens, kv_lens, SCALE
    )

    table = sparse_attn._get_table_prefill_config(total_q, q_heads)
    chunk = _SM120_PREFILL_CONFIGS[q_heads] if _production_tuned_eligible() else table
    assert launches == [("non-chunk", table), ("chunk", chunk)]


@pytest.mark.parametrize("q_heads", _SM120_PREFILL_CONFIGS)
def test_qsa_sparse_prefill_real_tuned_selector(monkeypatch, q_heads):
    _require_cuda()
    monkeypatch.setenv("SGLANG_QSA_PREFILL_GEOMETRY", "auto")
    q_lens = [8192]
    prefix_lens = [0]
    case = _make_case(
        q_lens,
        prefix_lens,
        q_heads,
        1,
        seed=840,
        selected_count=TOPK,
    )
    if _production_tuned_eligible():
        assert (
            _get_prefill_config(
                sum(q_lens),
                q_heads,
                1,
                256,
                kernel="ordinary",
                topk=TOPK,
                num_kv_heads=1,
                max_q=max(q_lens),
            )
            == _SM120_PREFILL_CONFIGS[q_heads]
        )
    actual = _run(case, q_lens, prefix_lens)
    expected = _reference(case, prefix_lens)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "q_lens,prefix_lens,q_heads,kv_heads",
    [
        ([700, 620, 530], [0, 0, 0], 6, 1),
        ([700, 620, 530], [0, 17, 31], 6, 1),
        ([600], [0], 24, 2),
        ([600], [17], 24, 2),
    ],
)
def test_qsa_sparse_prefill_real_tuned_ragged_and_kv2(
    monkeypatch, q_lens, prefix_lens, q_heads, kv_heads
):
    _require_cuda()
    if not _production_tuned_eligible():
        pytest.skip("real tuned selector coverage requires an admitted SM120 part")
    monkeypatch.setenv("SGLANG_QSA_PREFILL_GEOMETRY", "auto")
    group_size = q_heads // kv_heads
    assert (
        _get_prefill_config(
            sum(q_lens),
            group_size,
            len(q_lens),
            256,
            kernel="chunk",
            topk=TOPK,
            num_kv_heads=kv_heads,
            max_q=max(q_lens),
        )
        == _SM120_PREFILL_CONFIGS[group_size]
    )
    case = _make_case(
        q_lens,
        prefix_lens,
        q_heads,
        kv_heads,
        seed=850,
    )
    actual = _run(case, q_lens, prefix_lens)
    expected = _reference(case, prefix_lens)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


def test_qsa_sparse_prefill_hardware_geometry(monkeypatch):
    _require_cuda()
    monkeypatch.setenv("SGLANG_QSA_PREFILL_GEOMETRY", "auto")
    for group_size, geometry in _SM120_PREFILL_CONFIGS.items():
        table = sparse_attn._get_table_prefill_config(8192, group_size)
        expected = geometry if _production_tuned_eligible() else table
        assert (
            _get_prefill_config(
                8192,
                group_size,
                1,
                256,
                kernel="ordinary",
                topk=TOPK,
                num_kv_heads=1,
                max_q=8192,
            )
            == expected
        )
        assert _get_prefill_config(
            512,
            group_size,
            1,
            256,
            kernel="chunk",
            topk=TOPK,
            num_kv_heads=1,
            max_q=512,
        ) == (
            geometry
            if _production_tuned_eligible()
            else sparse_attn._get_table_prefill_config(512, group_size)
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
