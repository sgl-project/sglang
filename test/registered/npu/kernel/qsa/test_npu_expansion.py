"""Model-contract expansion dispatch and real QSA metadata/graph integration."""
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa import kernel, mqa
from sglang.srt.layers.attention.qsa.metadata import build_qsa_row_ranges
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")
pytestmark = pytest.mark.skipif(not is_npu(), reason="NPU is required")


def _expected(blocks, positions, lengths, ratio, token_topk):
    # CPU only: never captured or used as a production fallback.
    return kernel.torch_expand_qsa_block_indices(
        blocks.cpu(), positions.cpu(), lengths.cpu(), ratio, token_topk
    )


def _assert_prefix(blocks, positions, lengths, ratio):
    b, p, n = blocks.cpu().long(), positions.cpu().long(), lengths.cpu().long()
    valid = b >= 0
    assert torch.equal(valid, torch.arange(b.shape[1])[None, :] < valid.sum(1)[:, None])
    assert torch.all(b[~valid] == -1)
    limit = torch.minimum(p + 1, n)[:, None].expand_as(b)
    assert torch.all(((b + 1) * ratio)[valid] <= limit[valid])


def _assert_topk(logits, starts, lengths, blocks):
    """Test-only exact selection oracle; ties do not require equal indices."""
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert starts.is_contiguous() and lengths.is_contiguous()
    scores, begin, counts, indices = (x.cpu() for x in (logits, starts, lengths, blocks))
    width = indices.shape[1]
    for row, (start, length) in enumerate(zip(begin.tolist(), counts.tolist())):
        valid_scores = scores[row, start:start + length]
        assert torch.isfinite(valid_scores).all()
        count = min(width, length)
        chosen = indices[row, :count].long()
        assert ((chosen >= 0) & (chosen < length)).all()
        assert chosen.unique().numel() == count
        assert (indices[row, count:] == -1).all()
        if length <= width:
            assert torch.equal(chosen, torch.arange(length))
        else:
            torch.testing.assert_close(valid_scores[chosen].sort().values,
                                       valid_scores.topk(width).values.sort().values,
                                       atol=0, rtol=0)


@pytest.mark.parametrize("rows", [0, 1, 127, 128, 129])
@pytest.mark.parametrize("block_topk", [512, 2048])
def test_dispatch_model_contract(rows, block_topk, monkeypatch):
    blocks = torch.full((rows, block_topk), -1, dtype=torch.int32, device="npu")
    blocks[:, :3] = torch.tensor([2, 0, 1], device="npu")
    positions = torch.full((rows,), 14, dtype=torch.int64, device="npu")
    lengths = positions + 1
    expected = _expected(blocks, positions, lengths, 4, block_topk * 4)

    def forbidden(*args, **kwargs):
        raise AssertionError("NPU production must not use the generic reference")

    monkeypatch.setattr(kernel, "torch_expand_qsa_block_indices", forbidden)
    actual = kernel.expand_qsa_block_indices(blocks, positions, lengths, 4, block_topk * 4)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    assert actual.is_contiguous() and actual.dtype == torch.int32


@pytest.mark.parametrize("kind", ["budget", "ratio", "block_topk", "cpu", "rank", "dtype"])
def test_unsupported_metadata_raises_without_reference(kind, monkeypatch):
    args = [torch.zeros((2, 512), dtype=torch.int32, device="npu"),
            torch.full((2,), 15, dtype=torch.int64, device="npu"),
            torch.full((2,), 16, dtype=torch.int32, device="npu"), 4, 2048]
    if kind == "budget": args[4] = 2047
    elif kind == "ratio": args[3], args[4] = 1, 512
    elif kind == "block_topk": args[0], args[4] = args[0][:, :256], 1024
    elif kind == "cpu": args[2] = args[2].cpu()
    elif kind == "rank": args[1] = args[1].reshape(1, 2)
    elif kind == "dtype": args[0] = args[0].float()

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported metadata must not fall back")

    monkeypatch.setattr(kernel, "torch_expand_qsa_block_indices", forbidden)
    with pytest.raises(ValueError):
        kernel.expand_qsa_block_indices(*args)


@pytest.mark.parametrize("rows", [8, 140])
@pytest.mark.parametrize("block_topk", [512, 2048])
def test_packed_mqa_current_topk_expansion_graph(rows, block_topk):
    # Two packed sequences, nonzero starts, and a chunk crossing request rows.
    q = torch.randn(rows, 4, 128, device="npu", dtype=torch.bfloat16)
    keys = torch.randn(1280, 1, 128, device="npu", dtype=q.dtype)
    sequence_lengths = torch.full((2,), 2560, device="npu", dtype=torch.int32)
    ids = (torch.arange(rows, device="npu") % 2).int()
    positions = torch.full((rows,), 2402, device="npu", dtype=torch.int32)

    def forward():
        starts, ends, _ = build_qsa_row_ranges(sequence_lengths, positions, ids, 4)
        logits = mqa.qsa_mqa_prefill(q, keys, starts, ends)
        blocks = kernel.qsa_fast_topk(logits, starts, ends, block_topk)
        tokens = kernel.expand_qsa_block_indices(
            blocks, positions, sequence_lengths[ids.long()], 4, block_topk * 4)
        return blocks, tokens, logits, starts, ends - starts

    for _ in range(2): forward()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured_blocks, captured_tokens, scores, starts, lengths = forward()
    for step in range(4):
        if step == 1:
            q.zero_()
            positions.copy_((torch.arange(rows, device="npu") % 20).int())
        elif step == 2:
            keys.normal_()
            sequence_lengths.copy_(torch.tensor([2407, 2411], device="npu"))
            positions.fill_(2398)
        elif step == 3:
            ids.copy_(1 - ids)
            q.normal_()
            positions.fill_(2401)
        graph.replay()
        torch.npu.synchronize()
        row_lengths = sequence_lengths[ids.long()]
        # Validate the newly integrated MQA as well as selection of its scores.
        # Reference work stays outside capture; near ties need not choose the
        # same IDs as an independently rounded reference score matrix.
        reference_scores = mqa.torch_qsa_mqa_prefill(q, keys, starts, starts + lengths)
        torch.testing.assert_close(scores, reference_scores, atol=2e-5, rtol=2e-5)
        assert torch.equal(torch.isneginf(scores), torch.isneginf(reference_scores))
        _assert_prefix(captured_blocks, positions, row_lengths, 4)
        _assert_topk(scores, starts, lengths, captured_blocks)
        expected = _expected(captured_blocks, positions, row_lengths, 4, block_topk * 4)
        torch.testing.assert_close(captured_tokens.cpu(), expected, atol=0, rtol=0)
        eager_blocks, eager_tokens, eager_scores, eager_starts, eager_lengths = forward()
        _assert_topk(eager_scores, eager_starts, eager_lengths, eager_blocks)
        torch.testing.assert_close(captured_blocks, eager_blocks, atol=0, rtol=0)
        torch.testing.assert_close(captured_tokens, eager_tokens, atol=0, rtol=0)


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2])
@pytest.mark.parametrize("batch_size", [2, 32])
def test_backend_graph_metadata_current_topk_expansion(mode, batch_size):
    # Real backend methods/persistent buffers; synthetic pool and Q/K.
    # Includes sparse attention; not projection, compression, scheduler or service.
    raw_width = 4096
    rows = batch_size if mode.is_decode() else batch_size * 4
    requests = batch_size + 1
    cache = torch.randn(requests * 1024, 1, 128, device="npu", dtype=torch.bfloat16)
    attention_q = torch.randn(rows, 3, 256, device="npu", dtype=torch.bfloat16)
    attention_k = torch.randn(requests * raw_width, 1, 256, device="npu", dtype=torch.bfloat16)
    attention_v = torch.randn_like(attention_k)
    pool = SimpleNamespace(
        qsa_compress_ratio=4, qsa_block_topk=512,
        qsa_index_kv_heads=1, qsa_index_head_dim=128, qsa_compressed_page_size=16,
        get_qsa_compressed_k_buffer=lambda layer_id: cache,
        get_key_buffer=lambda layer_id: attention_k.reshape(-1, 64, 1, 256),
        get_value_buffer=lambda layer_id: attention_v.reshape(-1, 64, 1, 256))
    layer = SimpleNamespace(layer_id=0, scaling=1 / 16)
    req_to_token = torch.arange(requests * raw_width, device="npu", dtype=torch.int32).reshape(requests, raw_width)
    backend = QwenSparseAttnBackend()
    backend.device = torch.device("npu")
    backend.token_to_kv_pool, backend.req_to_token = pool, req_to_token
    backend.max_context_len = raw_width
    backend.init_cuda_graph_state(batch_size, rows)
    q = torch.randn(rows, 4, 128, device="npu", dtype=torch.bfloat16)
    lengths_cpu = torch.full((batch_size,), 2405, dtype=torch.int32, device="cpu")
    request_ids = torch.arange(1, requests, device="npu", dtype=torch.int32)
    spec = SimpleNamespace(topk=1, draft_token_num=4, extend_seq_lens_cpu=[4] * batch_size)
    batch = SimpleNamespace(
        forward_mode=mode, batch_size=batch_size,
        input_ids=torch.zeros(rows, dtype=torch.int64, device="npu"),
        req_pool_indices=request_ids, seq_lens=lengths_cpu.to("npu"),
        seq_lens_cpu=lengths_cpu, spec_info=spec, num_padding=0)
    backend.init_forward_metadata_out_graph(batch, in_capture=True)
    backend.init_forward_metadata_out_graph(batch, in_capture=False)
    metadata = backend.forward_metadata
    indexer = metadata.indexer_metadata

    def pointers():
        return [x.data_ptr() for x in (
            metadata.sequence_lengths, metadata.row_req_pool_indices,
            indexer.decode_logical_positions, indexer.graph_compressed_page_table,
            indexer.graph_compressed_lengths)]

    addresses = pointers()

    def forward():
        paged_cache, table, lengths, width = indexer.get_decode_mqa_inputs(0)
        logits = mqa.qsa_mqa_decode(q, paged_cache, table, lengths, width)
        blocks = kernel.qsa_fast_topk(logits, torch.zeros_like(lengths), lengths, 512)
        tokens = kernel.expand_qsa_block_indices(
            blocks, indexer.decode_logical_positions, indexer.sequence_lengths, 4, 2048)
        slots = backend._logical_to_physical(tokens, metadata)
        output = backend._forward_paged_attention(attention_q, layer, batch, tokens)
        return blocks, tokens, slots, logits, lengths, output

    for _ in range(2): forward()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = forward()
    for step in range(4):
        if step == 1:
            q.zero_()
            attention_q.neg_()
            batch.seq_lens_cpu.fill_(5)
        elif step == 2:
            cache.normal_()
            attention_k.neg_()
            batch.seq_lens_cpu.fill_(2411)
            batch.req_pool_indices.copy_(request_ids.flip(0))
            if not mode.is_decode():
                batch.num_padding = 1
                spec.extend_seq_lens_cpu = [3] * batch_size
        elif step == 3:
            # Permute physical pages without breaking token-group/page alignment.
            req_to_token.copy_(req_to_token.roll(64, dims=1))
            attention_v.neg_()
            q.normal_()
            batch.seq_lens_cpu.fill_(2307)
            batch.num_padding = 0
        batch.seq_lens.copy_(batch.seq_lens_cpu.to("npu"))
        backend.init_forward_metadata_out_graph(batch, in_capture=False)
        assert pointers() == addresses
        graph.replay()
        torch.npu.synchronize()
        blocks, tokens, slots, logits, lengths, output = outputs
        reference_output = kernel.qsa_sparse_attention_reference(
            attention_q.cpu(), attention_k.cpu(), attention_v.cpu(), slots.cpu(), 1 / 16)
        torch.testing.assert_close(output.cpu(), reference_output.flatten(1), atol=0.02, rtol=0.02)
        paged_cache, page_table, valid_lengths, width = indexer.get_decode_mqa_inputs(0)
        reference_scores = mqa.torch_qsa_mqa_decode(q, paged_cache, page_table, valid_lengths, width)
        torch.testing.assert_close(logits, reference_scores, atol=2e-5, rtol=2e-5)
        assert torch.equal(torch.isneginf(logits), torch.isneginf(reference_scores))
        _assert_topk(logits, torch.zeros_like(lengths), lengths, blocks)
        p, n = indexer.decode_logical_positions, indexer.sequence_lengths
        _assert_prefix(blocks, p, n, 4)
        expected = _expected(blocks, p, n, 4, 2048)
        torch.testing.assert_close(tokens.cpu(), expected, atol=0, rtol=0)
        table_cpu = req_to_token.cpu()[metadata.row_req_pool_indices.cpu().long()]
        valid = (expected >= 0) & (expected < n.cpu()[:, None])
        physical = table_cpu.gather(1, expected.clamp_min(0).long())
        physical[~valid] = -1
        torch.testing.assert_close(slots.cpu(), physical, atol=0, rtol=0)
        for captured, eager in zip(outputs, forward()):
            torch.testing.assert_close(captured, eager, atol=0, rtol=0)


@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("rows,width,path", [
    (4, 256, "shortcut"), (4, 8193, "tiled"),
    (4, 262145, "hybrid"), (129, 262145, "hybrid"),
])
def test_final_topk_expansion_all_paths_graph(k, rows, width, path):
    from sgl_kernel_npu.qwen3_8_flash_next.qsa_topk import select_implementation

    assert select_implementation(rows, width, k) == path
    logits = torch.randn(rows, width, device="npu", dtype=torch.float32)
    starts = (torch.arange(rows, device="npu") % 8).int()
    lengths = torch.full_like(starts, width - 16)
    positions = lengths * 4 + 2
    sequence_lengths = torch.full_like(starts, width * 4)

    def forward():
        blocks = kernel.qsa_fast_topk(logits, starts, starts + lengths, k)
        tokens = kernel.expand_qsa_block_indices(blocks, positions, sequence_lengths, 4, k * 4)
        return blocks, tokens

    def check(outputs):
        blocks, tokens = outputs
        _assert_topk(logits, starts, lengths, blocks)
        _assert_prefix(blocks, positions, sequence_lengths, 4)
        torch.testing.assert_close(tokens.cpu(), _expected(blocks, positions, sequence_lengths, 4, k * 4),
                                   atol=0, rtol=0)

    for _ in range(2):
        check(forward())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = forward()
    for step in range(4):
        if step == 1:
            logits.zero_()
            starts.fill_(7)
        elif step == 2:
            lengths.copy_((torch.arange(rows, device="npu") % 3).int())
            positions.copy_(lengths * 4)
        elif step == 3:
            logits.normal_()
            lengths.fill_(width - 16)
            positions.copy_(lengths * 4 + 2)
        graph.replay()
        torch.npu.synchronize()
        check(outputs)
        check(forward())


def test_topk_dispatch_errors_are_not_hidden(monkeypatch):
    from sgl_kernel_npu.qwen3_8_flash_next import qsa_topk

    logits = torch.zeros(2, 4096, device="npu")
    starts = torch.zeros(2, dtype=torch.int32, device="npu")
    ends = starts + 1024
    with pytest.raises(ValueError, match="K=512 or 2048"):
        kernel.qsa_fast_topk(logits, starts, ends, 8)
    with pytest.raises(ValueError, match="contiguous in columns"):
        kernel.qsa_fast_topk(logits[:, ::2], starts, ends, 512)

    def fail(*args, **kwargs):
        raise RuntimeError("intentional Top-K failure")

    monkeypatch.setattr(qsa_topk, "fast_topk", fail)
    with pytest.raises(RuntimeError, match="intentional Top-K failure"):
        kernel.qsa_fast_topk(logits, starts, ends, 512)
