import sys
from collections.abc import Iterator
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.batch_invariant_ops import (
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
)
from sglang.srt.layers.attention.qsa import qsa_indexer as qsa_indexer_module
from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="nightly", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def batch_invariant_mode() -> Iterator[None]:
    enable_batch_invariant_mode()
    try:
        yield
    finally:
        disable_batch_invariant_mode()


@pytest.mark.parametrize("topk", [512, 2048])
def test_qsa_topk_ragged_ties_and_graph(topk: int) -> None:
    torch.manual_seed(42)
    lengths = [0, 1, 31, 511, 512, 513, 2047, 2048, 2049]
    logits = torch.randint(0, 32, (len(lengths), 4096), device="cuda").float()
    starts = torch.arange(len(lengths), dtype=torch.int32, device="cuda") * 37
    ends = starts + torch.tensor(lengths, dtype=torch.int32, device="cuda")
    expected = torch.full((len(lengths), topk), -1, dtype=torch.int32, device="cuda")
    for row, length in enumerate(lengths):
        start = row * 37
        order = torch.argsort(
            logits[row, start : start + length], descending=True, stable=True
        )
        count = min(length, topk)
        expected[row, :count] = order[:count].sort().values.to(torch.int32)
    actual = qsa_fast_topk(logits, starts, ends, topk)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qsa_fast_topk(logits, starts, ends, topk)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_qsa_decode_topk_repeatability(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(42)
    logits = torch.randn(16, 2048, device="cuda").round()

    def mock_mqa_decode(*args: object, min_scored_len: int) -> torch.Tensor:
        assert min_scored_len == 0
        return logits

    monkeypatch.setattr(qsa_indexer_module, "qsa_mqa_decode", mock_mqa_decode)
    indexer = SimpleNamespace(block_topk=512)
    lengths = torch.full((16,), 2048, dtype=torch.int32, device="cuda")
    results = [
        QSAIndexer.select_decode_tokens(
            indexer,
            q=logits,
            compressed_cache=logits,
            compressed_page_table=lengths,
            compressed_lengths=lengths,
            max_model_len=2048,
            query_positions=lengths,
            sequence_lengths=lengths,
            defer_expansion=True,
        )
        for _ in range(4)
    ]
    for actual in results[1:]:
        torch.testing.assert_close(actual, results[0], rtol=0, atol=0)


def test_qsa_prefill_batch_invariance() -> None:
    torch.manual_seed(42)
    seq, heads, dim = 256, 3, 256
    q = torch.randn(seq, heads, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(seq, 1, dim, dtype=q.dtype, device=q.device)
    v = torch.randn_like(k)
    positions = torch.arange(seq, device=q.device)
    indices = positions.expand(seq, -1).to(torch.int32).contiguous()
    indices.masked_fill_(positions[None, :] > positions[:, None], -1)
    cu = torch.tensor([0, seq], dtype=torch.int32, device=q.device)
    reference = sparse_gqa_fwd_interface_triton(q, k, v, seq, indices, cu, dim**-0.5)
    for copies in (4, 8):
        actual = sparse_gqa_fwd_interface_triton(
            q.repeat(copies, 1, 1),
            k.repeat(copies, 1, 1),
            v.repeat(copies, 1, 1),
            seq,
            indices.repeat(copies, 1),
            torch.arange(copies + 1, dtype=torch.int32, device=q.device) * seq,
            dim**-0.5,
        )
        torch.testing.assert_close(actual[:seq], reference, rtol=0, atol=0)


def test_qsa_fp8_chunk_prefill_batch_invariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8-capable CUDA GPU required")

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA H20")
    torch.manual_seed(42)
    q_lens = (4096, 1024)
    kv_lens = (16384, 4096)
    heads, dim, topk = 3, 256, 1024
    q = torch.randn(sum(q_lens), heads, dim, dtype=torch.bfloat16, device="cuda")
    k, v = [
        torch.randn(sum(kv_lens), 1, dim, dtype=torch.bfloat16, device="cuda").to(
            torch.float8_e4m3fn
        )
        for _ in range(2)
    ]
    indices = torch.cat(
        [
            torch.randint(0, kv_len, (q_len, topk), dtype=torch.int32, device="cuda")
            for q_len, kv_len in zip(q_lens, kv_lens)
        ]
    )
    cu_q = torch.tensor([0, *accumulate(q_lens)], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, *accumulate(kv_lens)], dtype=torch.int32, device="cuda")

    def run(num_requests):
        total_q = sum(q_lens[:num_requests])
        total_k = sum(kv_lens[:num_requests])
        return sparse_gqa_fwd_interface_triton_ck(
            q[:total_q],
            k[:total_k],
            v[:total_k],
            indices[:total_q],
            cu_q[: num_requests + 1],
            cu_k[: num_requests + 1],
            torch.tensor(kv_lens[:num_requests], dtype=torch.int32, device="cuda"),
            dim**-0.5,
            max_q=q_lens[0],
        )

    reference = run(1)
    actual = run(2)
    torch.testing.assert_close(actual[: q_lens[0]], reference, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
