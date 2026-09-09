from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.kernels.ops.qwen4_ple import fused_qwen4_ngram_gather
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.models import qwen4_exp as qwen4_exp_module
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPinnedHostEmbedding,
    Qwen4ExpPLELayer,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import set_weight_attrs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

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
        weight_scale=None,
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


@pytest.mark.parametrize("is_fp8", [False, True])
def test_qwen4_ple_nonowner_does_not_access_backing(is_fp8):
    ids = torch.tensor([0, 3, 8, 99], dtype=torch.long, device="cuda")
    output = torch.full((4, 160), torch.nan, dtype=torch.bfloat16, device="cuda")
    # All IDs are outside [4, 8), so even an unmapped backing must not be read.
    qwen4_exp_module._gather_ple_embedding_from_pinned_kernel[(4,)](
        0,
        ids,
        output,
        embedding_dim=160,
        tp_vocab_start=4,
        tp_vocab_end=8,
        is_fp8=is_fp8,
        BLOCK_D=256,
    )
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)


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


def _make_ngram_inputs(num_tokens):
    contexts = (
        torch.tensor(
            [[1, 2, 3], [0, 2, 3], [1, 0, 3], [0, 0, 0], [3, 2, 1]],
            dtype=torch.long,
            device="cuda",
        )
        .repeat((num_tokens + 4) // 5, 1)[:num_tokens]
        .contiguous()
    )
    multipliers = torch.tensor(
        [190734863281251, 953674316406251, 4768371582031251],
        dtype=torch.long,
        device="cuda",
    )
    sizes = torch.tensor(
        [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79],
        dtype=torch.long,
        device="cuda",
    )
    offsets = sizes.cumsum(0) - sizes
    return contexts, multipliers, sizes, offsets


def _reference_ngram_ids(contexts, multipliers, sizes, offsets):
    previous = torch.where(
        (contexts[:, 0] == 0) | (contexts[:, 1] == 0), 0, contexts[:, 0]
    )
    mixed = (contexts[:, 2] * multipliers[0]) ^ (contexts[:, 1] * multipliers[1])
    mixed_three = mixed ^ (previous * multipliers[2])
    return torch.cat(
        (
            mixed[:, None].remainder(sizes[:8]) + offsets[:8],
            mixed_three[:, None].remainder(sizes[8:]) + offsets[8:],
        ),
        dim=1,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("num_tokens", [0, 1, 5, 128])
@pytest.mark.parametrize("vocab_start,vocab_end", [(0, 850), (100, 400)])
def test_qwen4_fused_ngram_gather(dtype, num_tokens, vocab_start, vocab_end):
    contexts, multipliers, sizes, offsets = _make_ngram_inputs(num_tokens)
    weight = torch.empty((vocab_end - vocab_start, 160), dtype=dtype, pin_memory=True)
    weight.copy_(
        (torch.arange(weight.numel()).reshape(weight.shape) % 31 - 15).to(dtype)
    )
    output = torch.empty((num_tokens, 16, 160), dtype=torch.bfloat16, device="cuda")
    actual = fused_qwen4_ngram_gather(
        contexts, multipliers, sizes, offsets, 0, weight, vocab_start, vocab_end, output
    )
    ids = _reference_ngram_ids(contexts, multipliers, sizes, offsets)
    in_range = (ids >= vocab_start) & (ids < vocab_end)
    local_ids = torch.where(in_range, ids - vocab_start, 0)
    rows = weight.to(device="cuda", dtype=torch.bfloat16)
    expected = torch.where(in_range[..., None], rows[local_ids], 0)
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qwen4_fused_ngram_gather_validates_output():
    contexts, multipliers, sizes, offsets = _make_ngram_inputs(1)
    weight = torch.empty((850, 160), dtype=torch.bfloat16, pin_memory=True)
    out = torch.empty((1, 16, 159), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="invalid output buffer"):
        fused_qwen4_ngram_gather(
            contexts, multipliers, sizes, offsets, 0, weight, 0, 850, out
        )


@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("gather_dp_tokens", [False, True])
@pytest.mark.parametrize(
    "mode", [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.EXTEND]
)
def test_qwen4_ple_prefetch_stream_and_graph(
    monkeypatch, fusion, gather_dp_tokens, mode
):
    contexts, multipliers, sizes, offsets = _make_ngram_inputs(5)
    offloaded = Qwen4ExpPinnedHostEmbedding(
        _make_source_embedding(embedding_dim=160, vocab_end=850, org_vocab_size=850)
    )
    rows = (torch.arange(850 * 160).reshape(850, 160) % 31).to(
        device="cuda", dtype=torch.bfloat16
    )
    _load_rows(offloaded, rows)
    offloaded.weight_scale = torch.ones(1, device="cuda", dtype=torch.bfloat16)
    embedding = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(embedding)
    embedding.ngram_embedding = offloaded
    embedding.gather_dp_tokens = gather_dp_tokens
    embedding.ngram_size = 3
    embedding.ngram_heads = 16
    embedding.heads_per_ngram = 8
    embedding.eos_token_id = 0
    embedding.enable_ple_fusion = fusion
    embedding.layer_multipliers = multipliers
    embedding.ngram_heads_vocab_sizes = sizes
    embedding.ngram_heads_offsets = offsets
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.ple_embedding = embedding
    layer.ple_embed_dim = 2560
    layer._prefetch_stream = torch.cuda.Stream()
    layer._graph_prefetch_buffers = {}
    layer._eager_prefetch_buffer = None
    layer._prefetch_state = None
    pool = SimpleNamespace(ple_window_cache=None)
    monkeypatch.setattr(qwen4_exp_module, "get_req_to_token_pool", lambda: pool)
    capturing = False
    monkeypatch.setattr(qwen4_exp_module, "get_is_capture_mode", lambda: capturing)
    forward_batch = SimpleNamespace(global_dp_buffer_len=5)
    main_stream = torch.cuda.current_stream()
    collectives = []

    def gather(out, src, batch):
        assert torch.cuda.current_stream() == main_stream
        collectives.append("gather")
        out.copy_(src)

    def scatter(out, src, batch):
        assert torch.cuda.current_stream() == main_stream
        collectives.append("scatter")
        out.copy_(src)

    monkeypatch.setattr(qwen4_exp_module, "dp_gather_replicate", gather)
    monkeypatch.setattr(qwen4_exp_module, "dp_scatter", scatter)
    batch = SimpleNamespace(
        ngram_context=contexts,
        physical_tokens=5,
        use_decode_fast_path=mode == ForwardMode.DECODE,
        req_indices=torch.arange(5, device="cuda"),
        token_offsets=torch.zeros(5, dtype=torch.long, device="cuda"),
        mode=mode,
    )

    from sglang.kernels.ops import qwen4_ple

    original = qwen4_ple.fused_qwen4_ngram_gather
    streams = []

    def checked_gather(*args, **kwargs):
        streams.append(torch.cuda.current_stream())
        return original(*args, **kwargs)

    monkeypatch.setattr(qwen4_ple, "fused_qwen4_ngram_gather", checked_gather)

    def run():
        pool.ple_window_cache = None
        layer.start_prefetch(batch, forward_batch)
        return layer._consume_prefetched_embeddings(forward_batch)

    actual = run()
    expected_ids = _reference_ngram_ids(contexts, multipliers, sizes, offsets)
    torch.testing.assert_close(actual, rows[expected_ids].flatten(1), rtol=0, atol=0)
    if fusion and not gather_dp_tokens:
        assert streams == [layer._prefetch_stream]
    else:
        assert streams == []
    assert collectives == (["gather", "scatter"] if gather_dp_tokens else [])
    assert layer._prefetch_state is None

    capturing = True
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        main_stream = torch.cuda.current_stream()
        graph_output = run()
    contexts.copy_(contexts.flip(0).clone())
    graph.replay()
    expected_ids = _reference_ngram_ids(contexts, multipliers, sizes, offsets)
    torch.testing.assert_close(
        graph_output, rows[expected_ids].flatten(1), rtol=0, atol=0
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
