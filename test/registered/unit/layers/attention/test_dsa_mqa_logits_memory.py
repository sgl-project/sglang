"""Check the lifetime of chunked prefill logits without loading CUDA kernels."""

import ast
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize("ragged_offsets", [False, True])
def test_prefill_releases_logits_before_allocating_next_chunk(ragged_offsets):
    root = Path(__file__).resolve().parents[5]
    path = root / "python/sglang/srt/layers/attention/dsa/dsa_indexer.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Indexer"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_get_topk_ragged"
    )
    num_q, num_k, topk = 5, 64, 2
    logits_refs = []
    chunk_sizes = []

    def kernel(q, kv, weights, ks, ke, clean_logits):
        assert all(ref() is None for ref in logits_refs), (
            "Previous logits chunk is still live when the next chunk is allocated"
        )
        logits = torch.arange(num_k, dtype=torch.float32).repeat(len(q), 1)
        logits_refs.append(weakref.ref(logits))
        chunk_sizes.append(len(q))
        return logits

    pool = SimpleNamespace(
        page_size=64,
        get_index_k_scale_buffer=lambda *args: (
            torch.zeros(num_k, 1, dtype=torch.uint8),
            torch.ones(num_k, 1, dtype=torch.float32),
        ),
    )
    namespace = dict(
        torch=torch,
        TYPE_CHECKING=False,
        _is_hip=False,
        _is_xpu=False,
        _is_fp8_fnuz=False,
        get_token_to_kv_pool=lambda: pool,
        deep_gemm=SimpleNamespace(fp8_mqa_logits=kernel),
    )
    exec("from __future__ import annotations\n" + ast.unparse(method), namespace)

    class IndexedCPUQuery:
        # The real method requires an indexed device; cpu:0 keeps allocation
        # calls on CPU while exercising the unmodified chunk loop.
        device = torch.device("cpu:0")
        shape = (num_q, 1, 1)

        def __getitem__(self, index):
            return torch.zeros(num_q, 1, 1)[index]

    lengths = torch.full((num_q,), num_k, dtype=torch.int32)
    metadata = SimpleNamespace(
        get_page_table_64=lambda: torch.tensor([[1]], dtype=torch.int32),
        get_indexer_kvcache_range=lambda: (torch.zeros_like(lengths), lengths),
        get_indexer_seq_len_cpu=lambda: torch.tensor([num_k]),
        get_indexer_seq_len=lambda: torch.tensor([num_k]),
        get_seqlens_expanded=lambda: lengths,
        get_token_to_batch_idx=lambda: torch.zeros_like(lengths),
        attn_metadata=SimpleNamespace(
            topk_indices_offset=torch.zeros_like(lengths) if ragged_offsets else None
        ),
        topk_transform=lambda logits, k, **kwargs: logits.topk(k).indices.to(
            torch.int32
        ),
    )
    indexer = SimpleNamespace(
        index_topk=topk,
        _MQA_LOGITS_BYTES_PER_ELEM=4,
        _should_chunk_mqa_logits=lambda *args: (True, 2 * num_k * 4),
        _with_real_sm_count=nullcontext,
        _pad_heads_for_deep_gemm=lambda q, w: (q, w, None),
        _mask_init_and_local_tokens=lambda *args: None,
    )
    result = namespace["_get_topk_ragged"](
        indexer,
        False,
        SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
            seq_lens_cpu=torch.tensor([num_k]),
            extend_seq_lens_cpu=torch.tensor([num_q]),
        ),
        0,
        IndexedCPUQuery(),
        torch.ones(num_q, 1, 1),
        metadata,
    )
    assert chunk_sizes == [2, 2, 1]
    assert all(ref() is None for ref in logits_refs)
    torch.testing.assert_close(
        result, torch.tensor([[63, 62]], dtype=torch.int32).repeat(num_q, 1)
    )
