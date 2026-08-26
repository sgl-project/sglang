from types import SimpleNamespace

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
        num_embeddings_per_partition=local_rows,
        num_org_embeddings_per_partition=local_rows,
        num_added_embeddings_per_partition=0,
    )


def _load_rows(offloaded, rows):
    pointer = offloaded.weight.data_ptr()
    offloaded.weight_loader(offloaded.weight, rows)
    assert offloaded.weight.data_ptr() == pointer
    assert offloaded.weight.is_pinned()
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
