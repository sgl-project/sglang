import os
import tempfile
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.models import qwen4_exp as qwen4_exp_module
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpPinnedHostEmbedding,
    Qwen4ExpPLELayer,
)
from sglang.srt.utils import set_weight_attrs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test."
)


def _make_source_embedding(
    *,
    dtype=torch.bfloat16,
    embedding_dim=7,
    vocab_start=0,
    vocab_end=8,
    org_vocab_size=8,
    tp_size=1,
    num_added_embeddings=0,
):
    local_rows = vocab_end - vocab_start
    weight = nn.Parameter(
        torch.empty((local_rows, embedding_dim), dtype=dtype, device="cuda"),
        requires_grad=False,
    )
    set_weight_attrs(
        weight,
        {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": lambda *_args, **_kwargs: None,
        },
    )
    shard_indices = VocabParallelEmbeddingShardIndices(
        padded_org_vocab_start_index=vocab_start,
        padded_org_vocab_end_index=vocab_end,
        padded_added_vocab_start_index=org_vocab_size,
        padded_added_vocab_end_index=org_vocab_size,
        org_vocab_start_index=vocab_start,
        org_vocab_end_index=vocab_end,
        added_vocab_start_index=org_vocab_size,
        added_vocab_end_index=org_vocab_size,
    )
    return SimpleNamespace(
        weight=weight,
        quant_config=None,
        enable_tp=True,
        use_attn_tp_group=False,
        tp_size=tp_size,
        num_embeddings=org_vocab_size + num_added_embeddings,
        org_vocab_size=org_vocab_size,
        padding_size=1,
        num_added_embeddings=num_added_embeddings,
        use_presharded_weights=False,
        org_vocab_size_padded=org_vocab_size,
        num_embeddings_padded=org_vocab_size + num_added_embeddings,
        shard_indices=shard_indices,
        embedding_dim=embedding_dim,
        quant_method=UnquantizedEmbeddingMethod(),
        # The offloaded table keeps the source's per-tensor scale (1.0 for bf16).
        weight_scale=torch.ones(1, dtype=torch.bfloat16, device="cuda"),
        num_embeddings_per_partition=local_rows,
        num_org_embeddings_per_partition=local_rows,
        num_added_embeddings_per_partition=0,
    )


def _load_rows(offloaded, rows, *, pinned=True):
    pointer = offloaded.weight.data_ptr()
    offloaded.weight_loader(offloaded.weight, rows)
    assert offloaded.weight.data_ptr() == pointer
    assert offloaded.weight.is_pinned() == pinned
    assert offloaded.weight.weight_loader.__self__ is offloaded
    assert offloaded.quant_method is None


@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("embedding_dim", [7, 64, 257])
def test_qwen4_ple_pinned_gather_tp1(input_dtype, embedding_dim):
    source = _make_source_embedding(embedding_dim=embedding_dim)
    offloaded = Qwen4ExpPinnedHostEmbedding(source)
    rows = torch.arange(8 * embedding_dim, dtype=torch.bfloat16, device="cuda").reshape(
        8, embedding_dim
    )
    _load_rows(offloaded, rows)

    ids = torch.tensor([[0, 7, 3], [4, 1, 6]], dtype=input_dtype, device="cuda")
    expected = rows.index_select(0, ids.long().flatten()).reshape(
        *ids.shape, embedding_dim
    )
    actual = offloaded(ids)

    assert actual.shape == expected.shape
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qwen4_ple_pinned_gather_shard_boundaries_and_out_buffer():
    embedding_dim = 13
    source = _make_source_embedding(
        embedding_dim=embedding_dim,
        vocab_start=4,
        vocab_end=8,
        org_vocab_size=8,
        tp_size=2,
    )
    offloaded = Qwen4ExpPinnedHostEmbedding(source)
    rows = torch.arange(8 * embedding_dim, dtype=torch.bfloat16, device="cuda").reshape(
        8, embedding_dim
    )
    _load_rows(offloaded, rows)

    ids = torch.tensor([[-1, 3, 4], [7, 8, 100]], device="cuda")
    output = torch.full(
        (*ids.shape, embedding_dim),
        torch.nan,
        dtype=torch.bfloat16,
        device="cuda",
    )
    actual = offloaded.gather(ids, out=output)
    expected = torch.zeros_like(output)
    expected[0, 2] = rows[4]
    expected[1, 0] = rows[7]

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qwen4_ple_pinned_gather_empty_input():
    offloaded = Qwen4ExpPinnedHostEmbedding(_make_source_embedding())
    _load_rows(offloaded, torch.zeros((8, 7), dtype=torch.bfloat16, device="cuda"))
    ids = torch.empty((0, 3), dtype=torch.int64, device="cuda")
    actual = offloaded.gather(ids)
    assert actual.shape == (0, 3, 7)
    assert actual.numel() == 0


def test_qwen4_ple_pinned_embedding_rejects_unsupported_weights():
    with pytest.raises(TypeError, match="requires bfloat16"):
        Qwen4ExpPinnedHostEmbedding(_make_source_embedding(dtype=torch.float16))
    with pytest.raises(NotImplementedError, match="added vocabulary"):
        Qwen4ExpPinnedHostEmbedding(_make_source_embedding(num_added_embeddings=1))


def test_qwen4_ple_prefetch_buffer_lifecycle(monkeypatch):
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.ple_embed_dim = 7
    layer.ple_embedding = SimpleNamespace(
        ngram_embedding=Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(embedding_dim=layer.ple_embed_dim)
        )
    )
    layer._graph_prefetch_buffers = {}
    layer._eager_prefetch_buffer = None
    lookup_ids = torch.empty((0,), dtype=torch.int64, device="cuda")

    monkeypatch.setattr(qwen4_exp_module, "get_is_capture_mode", lambda: False)
    eager_large = layer._get_prefetch_buffer(8, lookup_ids)
    eager_small = layer._get_prefetch_buffer(3, lookup_ids)
    assert eager_small.data_ptr() == eager_large.data_ptr()
    assert layer._eager_prefetch_buffer.shape == (8, layer.ple_embed_dim)

    eager_grown = layer._get_prefetch_buffer(12, lookup_ids)
    eager_grown_small = layer._get_prefetch_buffer(4, lookup_ids)
    assert eager_grown_small.data_ptr() == eager_grown.data_ptr()
    assert layer._eager_prefetch_buffer.shape == (12, layer.ple_embed_dim)

    monkeypatch.setattr(qwen4_exp_module, "get_is_capture_mode", lambda: True)
    graph_three = layer._get_prefetch_buffer(3, lookup_ids)
    graph_five = layer._get_prefetch_buffer(5, lookup_ids)
    graph_three_reused = layer._get_prefetch_buffer(3, lookup_ids)
    assert graph_three_reused.data_ptr() == graph_three.data_ptr()
    assert graph_five.data_ptr() != graph_three.data_ptr()
    assert set(layer._graph_prefetch_buffers) == {3, 5}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))


def _file_backend_supported() -> bool:
    from sglang.srt.models.qwen4_exp_ple_table import device_uses_host_page_tables

    return device_uses_host_page_tables(torch.cuda.current_device()) is True


@pytest.mark.skipif(
    not _file_backend_supported(),
    reason="the file backend needs pageable host memory reachable through host page tables",
)
@pytest.mark.parametrize("embedding_dim", [7, 160])
def test_qwen4_ple_file_backend_matches_pinned(embedding_dim):
    with tempfile.TemporaryDirectory() as table_dir:
        pinned = Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(embedding_dim=embedding_dim)
        )
        filed = Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(embedding_dim=embedding_dim),
            backend="file",
            table_dir=table_dir,
        )
        assert pinned._file_prefetcher is None and filed._file_prefetcher is not None
        (name,) = os.listdir(table_dir)
        assert "rows0-8" in name  # this rank's vocabulary shard
        rows = torch.arange(
            8 * embedding_dim, dtype=torch.bfloat16, device="cuda"
        ).reshape(8, embedding_dim)
        _load_rows(pinned, rows)
        _load_rows(filed, rows, pinned=False)
        ids = torch.tensor([[0, 7, 3], [4, 1, 6]], dtype=torch.int64, device="cuda")
        torch.testing.assert_close(filed(ids), pinned(ids), rtol=0, atol=0)
        # A prefill-sized gather goes through the page-cache hint path.
        big = torch.randint(0, 8, (4096,), device="cuda")
        torch.testing.assert_close(
            filed(big), rows.index_select(0, big), rtol=0, atol=0
        )


@pytest.mark.skipif(
    not _file_backend_supported(),
    reason="the file backend needs pageable host memory reachable through host page tables",
)
def test_qwen4_ple_file_backend_fp8_table():
    embedding_dim = 160
    with tempfile.TemporaryDirectory() as table_dir:
        filed = Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(
                embedding_dim=embedding_dim, dtype=torch.float8_e4m3fn
            ),
            backend="file",
            table_dir=table_dir,
        )
        assert filed.weight.dtype == torch.float8_e4m3fn
        rows = (
            torch.arange(8 * embedding_dim, dtype=torch.float32, device="cuda").reshape(
                8, embedding_dim
            )
            / 64
        ).to(torch.float8_e4m3fn)
        _load_rows(filed, rows, pinned=False)
        ids = torch.tensor([[0, 7, 3]], dtype=torch.int64, device="cuda")
        expected = (
            rows.index_select(0, ids.flatten())
            .to(torch.bfloat16)
            .reshape(1, 3, embedding_dim)
        )
        torch.testing.assert_close(filed(ids), expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not _file_backend_supported(),
    reason="the file backend needs pageable host memory reachable through host page tables",
)
@pytest.mark.parametrize("vocab_start", [0, 64])
def test_qwen4_ple_file_prefetch_tp_shard(vocab_start):
    with tempfile.TemporaryDirectory() as table_dir:
        filed = Qwen4ExpPinnedHostEmbedding(
            _make_source_embedding(
                embedding_dim=160,
                vocab_start=vocab_start,
                vocab_end=vocab_start + 64,
                org_vocab_size=128,
                tp_size=2,
            ),
            backend="file",
            table_dir=table_dir,
        )
        rows = torch.arange(64, device="cuda", dtype=torch.bfloat16)[:, None].expand(
            64, 160
        )
        filed.weight.copy_(rows)
        ids = torch.tensor([0, 63, 64, 76, 127], device="cuda").repeat(512)
        out = torch.empty((ids.numel(), 160), device="cuda", dtype=torch.bfloat16)
        owned = (ids >= vocab_start) & (ids < vocab_start + 64)
        expected = torch.zeros_like(out)
        expected[owned] = rows[ids[owned] - vocab_start]
        try:
            with mock.patch("os.posix_fadvise") as fadvise:
                filed.gather(ids, out=out)
                torch.cuda.synchronize()
                filed._file_prefetcher._pool.shutdown(wait=True)
                offsets = sorted(c.args[1] for c in fadvise.call_args_list)
                assert offsets == ([0, 16384] if vocab_start == 0 else [0, 4096, 16384])
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        finally:
            filed._file_rss_trimmer.close()
            filed._file_prefetcher.close()
