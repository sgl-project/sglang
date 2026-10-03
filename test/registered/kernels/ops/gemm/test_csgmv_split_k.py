"""Compact expand and split-K regressions, including changing graph metadata."""

import itertools
import math

import pytest
import torch

from sglang.kernels.ops.gemm.chunked_sgmv_expand import chunked_sgmv_lora_expand_forward
from sglang.kernels.ops.gemm.chunked_sgmv_shrink import chunked_sgmv_lora_shrink_forward
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_batch(tokens=33, active=True):
    info = LoRABatchInfo(
        use_cuda_graph=True,
        bs=tokens,
        num_segments=1,
        max_len=16,
        seg_lens=None,
        seg_indptr=torch.zeros(tokens + 1, device="cuda", dtype=torch.int32),
        weight_indices=torch.zeros(tokens, device="cuda", dtype=torch.int32),
        lora_ranks=torch.zeros(4, device="cuda", dtype=torch.int32),
        scalings=torch.tensor([0.3, -0.7, 1.0, 1.5], device="cuda"),
        permutation=torch.zeros(tokens, device="cuda", dtype=torch.int32),
    )
    ids = set_batch(info, active)
    return info, ids


def set_batch(info, active):
    ids = [i % 4 if active else 0 for i in range(info.bs)]
    counts = [ids.count(adapter) for adapter in range(4)]
    lengths = [min(16, n - j) for n in counts for j in range(0, n, 16)]
    adapters = [a for a, n in enumerate(counts) for _ in range(0, n, 16)]
    boundaries = [0, *itertools.accumulate(lengths)]
    info.num_segments = len(adapters)
    info.seg_indptr.fill_(info.bs)
    info.weight_indices.zero_()
    info.seg_indptr[: len(boundaries)] = torch.tensor(boundaries, device="cuda")
    info.weight_indices[: len(adapters)] = torch.tensor(adapters, device="cuda")
    info.permutation.copy_(
        torch.tensor(sorted(range(info.bs), key=ids.__getitem__), device="cuda")
    )
    info.lora_ranks.copy_(
        torch.tensor([0, 8, 16, 32] if active else [0] * 4, device="cuda")
    )
    return ids


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("slices", [1, 2, 3])
@pytest.mark.parametrize("split_k,k", [(2, 4096), (8, 4096), (16, 12288), (8, 12345)])
def test_split_k_reference(dtype, slices, split_k, k, monkeypatch):
    monkeypatch.setenv("SGLANG_CSGMV_SPLIT_K", "0")
    monkeypatch.setenv("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "0")
    torch.manual_seed(5)
    info, ids = make_batch()
    x = torch.randn(info.bs, k, device="cuda", dtype=dtype)
    a = torch.randn(4, slices * 32, k, device="cuda", dtype=dtype) / math.sqrt(k)
    actual = chunked_sgmv_lora_shrink_forward(x, a, info, slices, split_k=split_k)
    for adapter in (1, 2, 3):
        rows = [i for i, value in enumerate(ids) if value == adapter]
        width = slices * (8, 16, 32)[adapter - 1]
        expected = (x[rows].float() @ a[adapter, :width].float().T).to(dtype)
        torch.testing.assert_close(
            actual[rows, :width], expected, rtol=0.025, atol=0.04
        )


@pytest.mark.parametrize(
    "offsets", [(0, 65), (0, 65, 82), (0, 65, 82, 211), (0, 128, 128, 160)]
)
def test_compact_expand_matches_rectangular(offsets):
    torch.manual_seed(6)
    info, ids = make_batch()
    slices, n = len(offsets) - 1, offsets[-1]
    h = torch.randn(info.bs, slices * 32, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4, n, 32, device="cuda", dtype=torch.bfloat16) / math.sqrt(32)
    initial = torch.randn(info.bs, n + 17, device="cuda", dtype=torch.bfloat16)
    old, new = initial.clone(), initial.clone()
    offsets_cpu = torch.tensor(offsets, dtype=torch.int32)
    offsets_gpu = offsets_cpu.cuda()
    width = max(hi - lo for lo, hi in zip(offsets, offsets[1:]))
    chunked_sgmv_lora_expand_forward(h, b, info, offsets_gpu, width, old[:, :n])
    chunked_sgmv_lora_expand_forward(
        h, b, info, offsets_gpu, width, new[:, :n], offsets_cpu
    )
    assert torch.equal(old, new)
    assert torch.equal(new[:, n:], initial[:, n:])
    for adapter in range(4):
        rows = [i for i, value in enumerate(ids) if value == adapter]
        rank = [0, 8, 16, 32][adapter]
        if not rank:
            assert torch.equal(new[rows], initial[rows])
            continue
        for s, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
            delta = (
                h[rows, s * rank : (s + 1) * rank].float()
                @ b[adapter, lo:hi, :rank].float().T
            )
            expected = (
                initial[rows, lo:hi] + (delta * info.scalings[adapter]).bfloat16()
            )
            torch.testing.assert_close(new[rows, lo:hi], expected, rtol=0.03, atol=0.08)


def test_split_k_compact_graph_metadata_changes(monkeypatch):
    monkeypatch.setenv("SGLANG_CSGMV_SPLIT_K", "1")
    monkeypatch.setenv("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "0")
    torch.manual_seed(7)
    info, _ = make_batch(active=False)
    x = torch.randn(info.bs, 4096, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(4, 96, 4096, device="cuda", dtype=torch.bfloat16) / 64
    b = torch.randn(4, 211, 32, device="cuda", dtype=torch.bfloat16) / math.sqrt(32)
    offsets_cpu = torch.tensor([0, 65, 82, 211], dtype=torch.int32)
    offsets_gpu = offsets_cpu.cuda()
    base = torch.randn(info.bs, 211, device="cuda", dtype=torch.bfloat16)

    def run():
        h = chunked_sgmv_lora_shrink_forward(x, a, info, 3)
        return chunked_sgmv_lora_expand_forward(
            h, b, info, offsets_gpu, 129, base.clone(), offsets_cpu
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for active in (True, False, True):
        set_batch(info, active)
        expected = run()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_deterministic_mode_disables_split_k(monkeypatch):
    info, ids = make_batch()
    x = torch.randn(info.bs, 4096, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(4, 32, 4096, device="cuda", dtype=torch.bfloat16) / 64
    monkeypatch.setenv("SGLANG_CSGMV_SPLIT_K", "0")
    expected = chunked_sgmv_lora_shrink_forward(x, a, info, 1)
    monkeypatch.setenv("SGLANG_CSGMV_SPLIT_K", "1")
    monkeypatch.setenv("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "1")
    actual = chunked_sgmv_lora_shrink_forward(x, a, info, 1)
    for adapter in (1, 2, 3):
        rows = [i for i, value in enumerate(ids) if value == adapter]
        rank = [0, 8, 16, 32][adapter]
        assert torch.equal(actual[rows, :rank], expected[rows, :rank])
