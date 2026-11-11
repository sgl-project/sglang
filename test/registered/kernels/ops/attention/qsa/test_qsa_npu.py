"""NPU QSA selection, KV layout and graph metadata updates."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend_module
from sglang.srt.layers.attention.qsa import kernel, mqa
from sglang.srt.layers.attention.qsa.metadata import build_qsa_row_ranges
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")
pytestmark = pytest.mark.skipif(not is_npu(), reason="NPU is required")


@pytest.fixture
def runtime_context():
    publish(ServerArgs(model_path="dummy", speculative_num_draft_tokens=4), role="test")
    try:
        yield
    finally:
        reset_context()


def _index_keys(count):
    # Two exactly representable BF16 channels give distinct scores at the cutoff.
    positions = torch.arange(count, device="npu")
    keys = torch.zeros(count, 1, 128, dtype=torch.bfloat16, device="npu")
    keys[:, 0, 0] = positions // 128
    keys[:, 0, 1] = (positions % 128) / 128
    return keys


def _check_selected(tokens, scores, starts, ends, positions, lengths):
    blocks = torch.full((scores.shape[0], 512), -1, dtype=torch.int32)
    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        count = min(512, end - start)
        blocks[row, :count] = scores[row, start:end].topk(count).indices.int()
    expected = kernel.torch_expand_qsa_block_indices(
        blocks, positions.cpu(), lengths.cpu(), 4, 2048
    )
    # Top-k order is unspecified. Token membership and padding must agree.
    torch.testing.assert_close(
        tokens.cpu().sort(1).values, expected.sort(1).values, atol=0, rtol=0
    )


def test_qwen38_packed_prefill_selection():
    # Two requests share a packed cache; the second starts at compressed key 602.
    lengths = torch.tensor([2411, 31], dtype=torch.int32, device="npu")
    positions = torch.cat([torch.arange(2403, 2411), torch.arange(24, 31)]).to("npu")
    request_ids = torch.tensor([0] * 8 + [1] * 7, device="npu", dtype=torch.int32)
    q = torch.ones(15, 4, 128, device="npu", dtype=torch.bfloat16)
    keys = _index_keys(609)
    starts, ends, _ = build_qsa_row_ranges(lengths, positions, request_ids, 4)
    row_lengths = lengths[request_ids.long()]
    indexer = SimpleNamespace(token_topk=2048, block_topk=512, compress_ratio=4)
    actual = QSAIndexer.select_prefill_tokens(
        indexer,
        q,
        keys,
        starts,
        ends,
        positions,
        row_lengths,
        max_row_keys=602,
    )
    scores = mqa.torch_qsa_mqa_prefill(q.cpu(), keys.cpu(), starts.cpu(), ends.cpu())
    _check_selected(actual, scores, starts.cpu(), ends.cpu(), positions, row_lengths)


@pytest.mark.parametrize(
    "mode,batch_size,heads",
    [
        pytest.param(
            ForwardMode.DECODE,
            2,
            6,
            id="qwen3.8-flash-next-decode-b2-attn-tp4-cap65536",
        ),
        pytest.param(
            ForwardMode.TARGET_VERIFY,
            1,
            3,
            id="qwen3.8-flash-next-verify-t4-attn-tp8-cap65536",
        ),
    ],
)
def test_qsa_graph_request_updates(mode, batch_size, heads, runtime_context):
    torch.manual_seed(23)
    rows = batch_size if mode.is_decode() else 4 * batch_size
    requests, width = 3, 4096
    cache = _index_keys(requests * width // 4)
    k = torch.randn(requests * width, 1, 256, device="npu", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn(rows, heads, 256, device="npu", dtype=k.dtype)
    index_q = torch.ones(rows, 4, 128, device="npu", dtype=k.dtype)
    pool = SimpleNamespace(
        qsa_compress_ratio=4,
        qsa_block_topk=512,
        qsa_index_kv_heads=1,
        qsa_index_head_dim=128,
        qsa_compressed_page_size=16,
        get_qsa_compressed_k_buffer=lambda _: cache,
        get_key_buffer=lambda _: k.reshape(-1, 64, 1, 256),
        get_value_buffer=lambda _: v.reshape(-1, 64, 1, 256),
        start_layer=0,
        layer_transfer_counter=None,
    )
    pool.layer_router = PlainPoolRouter(pool, "full")
    req_to_token = torch.arange(
        requests * width, device="npu", dtype=torch.int32
    ).reshape(requests, width)
    backend = QwenSparseAttnBackend()
    backend.device, backend.token_to_kv_pool = torch.device("npu"), pool
    backend.req_to_token = req_to_token
    # Graph capacity follows 262144 / 4 even when the live request is short.
    backend.max_context_len = 262144
    backend.init_cuda_graph_state(batch_size, rows)
    lengths_cpu = torch.full((batch_size,), 2405, dtype=torch.int32)
    batch = SimpleNamespace(
        forward_mode=mode,
        batch_size=batch_size,
        input_ids=torch.zeros(rows, dtype=torch.int64, device="npu"),
        req_pool_indices=torch.arange(
            1, batch_size + 1, dtype=torch.int32, device="npu"
        ),
        seq_lens=lengths_cpu.to("npu"),
        seq_lens_cpu=lengths_cpu,
        spec_info=SimpleNamespace(topk=1, draft_token_num=4),
        num_padding=0,
    )
    backend.init_forward_metadata_out_graph(batch, in_capture=True)
    backend.init_forward_metadata_out_graph(batch, in_capture=False)
    metadata = backend.forward_metadata
    index_metadata = metadata.indexer_metadata
    indexer = SimpleNamespace(token_topk=2048, block_topk=512, compress_ratio=4)
    layer = SimpleNamespace(
        layer_id=0, scaling=1 / 16, tp_q_head_num=heads, head_dim=256
    )

    def forward():
        paged_cache, table, lengths, capacity = index_metadata.get_decode_mqa_inputs(0)
        tokens = QSAIndexer.select_decode_tokens(
            indexer,
            index_q,
            paged_cache,
            table,
            lengths,
            capacity,
            index_metadata.decode_logical_positions,
            metadata.sequence_lengths,
        )
        attention = (
            backend.forward_decode if mode.is_decode() else backend.forward_extend
        )
        output = attention(
            q, None, None, layer, batch, save_kv_cache=False, topk_indices=tokens
        )
        return tokens, output

    for _ in range(2):
        forward()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        tokens, output = forward()
    for step in range(2):
        if step:
            # Reuse the captured graph for different requests and physical pages.
            batch.req_pool_indices.copy_(3 - batch.req_pool_indices)
            batch.seq_lens_cpu.fill_(7)
            batch.seq_lens.copy_(batch.seq_lens_cpu.to("npu"))
            req_to_token.copy_(req_to_token.roll(64, dims=1))
            v.neg_()
            backend.init_forward_metadata_out_graph(batch, in_capture=False)
        graph.replay()
        torch.npu.synchronize()
        paged_cache, table, lengths, capacity = index_metadata.get_decode_mqa_inputs(0)
        assert capacity == 65536
        row_lengths = batch.seq_lens_cpu
        if mode.is_target_verify():
            row_lengths = row_lengths.repeat_interleave(4) + torch.arange(
                1, 5, dtype=torch.int32
            )
        torch.testing.assert_close(lengths.cpu(), row_lengths // 4, atol=0, rtol=0)
        scores = mqa.torch_qsa_mqa_decode(
            index_q.cpu(), paged_cache.cpu(), table.cpu(), lengths.cpu(), capacity
        )
        _check_selected(
            tokens,
            scores,
            torch.zeros_like(lengths.cpu()),
            lengths.cpu(),
            row_lengths - 1,
            row_lengths,
        )
        # Gather from the live table independently of the backend's mapping helper.
        logical = tokens.cpu().long()
        row_requests = (
            batch.req_pool_indices.cpu().long().repeat_interleave(rows // batch_size)
        )
        slots = req_to_token.cpu()[row_requests[:, None], logical.clamp_min(0)]
        slots[logical < 0] = -1
        expected = kernel.qsa_sparse_attention_reference(
            q.cpu(), k.cpu(), v.cpu(), slots, layer.scaling
        )
        torch.testing.assert_close(
            output.cpu(), expected.flatten(1), atol=2e-2, rtol=2e-2
        )


@pytest.mark.parametrize("fia", [False, True], ids=["paged", "fia"])
def test_qsa_pool_write_and_padding(monkeypatch, fia):
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
    from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype

    monkeypatch.setenv("ASCEND_USE_FIA", "True" if fia else "False")
    _, dtype = configure_kv_cache_dtype(
        server_args_kv_cache_dtype="auto",
        model=SimpleNamespace(quant_config=None),
        model_dtype=torch.bfloat16,
        is_draft_worker=False,
        is_dflash=False,
        speculative_draft_attention_backend="ascend",
    )
    pool = NPUMHATokenToKVPool(
        size=128,
        page_size=64,
        dtype=dtype,
        head_num=1,
        head_dim=256,
        layer_num=2,
        device="npu",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    backend = QwenSparseAttnBackend()
    backend.token_to_kv_pool = pool
    backend.forward_metadata = SimpleNamespace(
        is_cuda_graph=False,
        token_to_batch_idx=torch.zeros(3, dtype=torch.int32, device="npu"),
        sequence_lengths=torch.tensor([128], device="npu", dtype=torch.int32),
        token_slot_table=torch.arange(128, device="npu", dtype=torch.int32)[None],
    )
    layer = SimpleNamespace(layer_id=1, scaling=1 / 16, tp_q_head_num=3, head_dim=256)
    locations = torch.tensor([1, 2, 64, 65], dtype=torch.int32, device="npu")
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, out_cache_loc=locations)
    q = torch.ones(4, 3, 256, dtype=dtype, device="npu")
    new_k = torch.zeros(4, 1, 256, dtype=dtype, device="npu")
    new_v = (
        torch.arange(1, 5, device="npu", dtype=dtype)[:, None, None]
        .expand_as(new_k)
        .contiguous()
    )
    tokens = torch.full((3, 2051), -1, dtype=torch.int32, device="npu")
    tokens[0, :2] = locations[:2]
    tokens[1, :4] = locations
    for shift in (0, 2):
        if shift:
            new_v.add_(shift)
        output = backend.forward_extend(
            q, new_k, new_v, layer, batch, topk_indices=tokens
        )
        # One fully masked semantic row and one DP padding row both return zero.
        expected = torch.zeros_like(output)
        expected[0] = 1.5 + shift
        expected[1] = 2.5 + shift
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        for get_buffer, updates in (
            (pool.get_key_buffer, new_k),
            (pool.get_value_buffer, new_v),
        ):
            cache = get_buffer(1)
            flat = backend_module._flatten_qsa_kv_cache(cache, "cache")
            assert flat.data_ptr() == cache.data_ptr()
            torch.testing.assert_close(flat[locations.long()], updates, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
