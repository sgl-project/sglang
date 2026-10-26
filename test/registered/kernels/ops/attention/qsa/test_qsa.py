import argparse
import os
import socket
import sys
from itertools import accumulate
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.kernels.ops.attention import qwen38_qsa_sm121_varlen
from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig
from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_backend_module
from sglang.srt.layers.attention.qsa import qsa_indexer as qsa_indexer_module
from sglang.srt.layers.attention.qsa.cache_sharding import (
    QSACacheShardingStaticBuffers,
    resolve_qsa_cache_sharding_runtime,
    validate_qsa_cache_sharding,
)
from sglang.srt.layers.attention.qsa.distributed_topk import (
    fused_publish_and_merge_qsa_topk_candidates,
    gather_and_merge_qsa_topk_candidates,
    merge_qsa_topk_candidates,
)
from sglang.srt.layers.attention.qsa.kernel import (
    expand_qsa_block_indices,
    qsa_fast_topk,
    qsa_sparse_attention,
    torch_expand_qsa_block_indices,
    triton_expand_qsa_block_indices,
)
from sglang.srt.layers.attention.qsa.metadata import (
    QSAIndexerMetadata,
    build_qsa_row_ranges,
)
from sglang.srt.layers.attention.qsa.mqa import (
    qsa_mqa_decode,
    qsa_mqa_prefill,
)
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qsa.sparse_attn import (
    pack_qsa_prefill_kv,
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    QwenSparseMultiStepDraftBackend,
)
from sglang.srt.mem_cache.qsa_kv_pool import (
    QSACompressedBlockSharding,
    QSARawKVSharding,
    QSATokenToKVPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

COMPRESS_RATIO = 4
TOKEN_TOPK = 2048
BLOCK_TOPK = TOKEN_TOPK // COMPRESS_RATIO
FINAL_TOPK = TOKEN_TOPK + COMPRESS_RATIO - 1


def test_qsa_cache_sharding_layers_share_one_static_arena():
    buffers = QSACacheShardingStaticBuffers(
        max_rows=8,
        topk=16,
        owner_topk=32,
        num_heads=8,
        head_dim=128,
        world_size=4,
        device="cpu",
        dtype=torch.bfloat16,
        layer_ids=(2, 5, 9),
    )

    layer_buffers = [buffers.for_layer(layer_id) for layer_id in (2, 5, 9)]
    assert all(layer_buffer is buffers for layer_buffer in layer_buffers)
    assert len({layer_buffer.q_send.data_ptr() for layer_buffer in layer_buffers}) == 1
    assert (
        len({layer_buffer.candidate_recv.data_ptr() for layer_buffer in layer_buffers})
        == 1
    )
    with pytest.raises(KeyError, match="no static transport slice"):
        buffers.for_layer(10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qsa_cache_sharding_max_graph_bucket_matches_budget():
    dimensions = {
        "max_rows": 1024,
        "topk": 512,
        "owner_topk": 2051,
        "num_heads": 32,
        "head_dim": 128,
        "world_size": 4,
        "dtype": torch.bfloat16,
    }
    buffers = QSACacheShardingStaticBuffers(
        **dimensions,
        device="cuda",
        layer_ids=tuple(range(48)),
    )

    tensors = [
        value for value in vars(buffers).values() if isinstance(value, torch.Tensor)
    ]
    actual_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    assert actual_bytes < 100 * 1024**2
    assert all(buffers.for_layer(layer_id) is buffers for layer_id in range(48))


class _QSAIntegrationGroup:
    def __init__(self, rank, world_size):
        self.rank_in_group = rank
        self.world_size = world_size
        self.device_group = dist.group.WORLD

    def all_gather_into_tensor(self, output, value):
        dist.all_gather_into_tensor(output, value, group=self.device_group)
        return output

    def all_to_all_single(self, output, value):
        dist.all_to_all_single(output, value, group=self.device_group)
        return output

    def all_gather(self, value, dim=0):
        gathered = [torch.empty_like(value) for _ in range(self.world_size)]
        dist.all_gather(gathered, value, group=self.device_group)
        return torch.cat(gathered, dim=dim)


def _assert_qsa_integration_close(
    selected,
    output,
    lse,
    expected_selected,
    expected_output,
    expected_lse,
    control_group,
):
    local_status = torch.tensor(
        [
            torch.equal(
                selected.sort(dim=-1).values,
                expected_selected.sort(dim=-1).values,
            ),
            torch.allclose(output, expected_output, rtol=2e-2, atol=2e-2),
            torch.allclose(lse, expected_lse, rtol=2e-3, atol=2e-3),
        ],
        dtype=torch.int32,
    )
    global_status = local_status.clone()
    dist.all_reduce(global_status, op=dist.ReduceOp.MIN, group=control_group)
    if not bool(torch.all(global_status).item()):
        raise AssertionError(
            "QSA multi-rank integration mismatch: "
            f"local={local_status.tolist()}, global={global_status.tolist()}"
        )


def _qsa_integration_inputs(world_size, rank, device):
    rows, local_heads, head_dim = 2, 2, 128
    local_blocks, compress_ratio, token_topk = 640, 4, 2048
    global_blocks = local_blocks * world_size
    global_tokens = global_blocks * compress_ratio
    generator = torch.Generator().manual_seed(20261012)
    index_q = torch.randn(
        rows, 8, head_dim, dtype=torch.bfloat16, generator=generator
    ).to(device)
    compressed = torch.randn(
        global_blocks, 1, head_dim, dtype=torch.bfloat16, generator=generator
    ).to(device)
    full_q = torch.randn(
        rows,
        local_heads * world_size,
        head_dim,
        dtype=torch.bfloat16,
        generator=generator,
    ).to(device)
    full_k = torch.randn(
        global_tokens, 1, head_dim, dtype=torch.bfloat16, generator=generator
    ).to(device)
    full_v = torch.randn(
        global_tokens, 1, head_dim, dtype=torch.bfloat16, generator=generator
    ).to(device)
    return SimpleNamespace(
        rows=rows,
        local_heads=local_heads,
        head_dim=head_dim,
        local_blocks=local_blocks,
        global_blocks=global_blocks,
        global_tokens=global_tokens,
        compress_ratio=compress_ratio,
        token_topk=token_topk,
        index_q=index_q,
        local_compressed=compressed[rank::world_size].contiguous(),
        global_compressed=compressed,
        local_q=full_q[:, rank * local_heads : (rank + 1) * local_heads].contiguous(),
        full_q=full_q,
        local_k=full_k[rank::world_size].contiguous(),
        local_v=full_v[rank::world_size].contiguous(),
        full_k=full_k,
        full_v=full_v,
    )


def _qsa_integration_tokens(
    inputs,
    indexer,
    mode,
    compressed,
    *,
    logical_positions=None,
    static_buffers=None,
):
    query_positions = torch.full(
        (inputs.rows,),
        inputs.global_tokens - 1,
        dtype=torch.int64,
        device=inputs.index_q.device,
    )
    sequence_lengths = query_positions + 1
    local_blocks = compressed.shape[0]
    if mode == "prefill":
        kwargs = {}
        if indexer.distributed_topk_group is not None:
            kwargs = {
                "collective_keys": inputs.global_blocks * inputs.rows,
                "collective_world_size": indexer.distributed_topk_group.world_size,
            }
        return indexer.select_prefill_tokens(
            inputs.index_q,
            compressed,
            torch.zeros(inputs.rows, dtype=torch.int32, device=inputs.index_q.device),
            torch.full(
                (inputs.rows,),
                local_blocks,
                dtype=torch.int32,
                device=inputs.index_q.device,
            ),
            query_positions,
            sequence_lengths,
            logical_block_positions=logical_positions,
            static_buffers=static_buffers,
            **kwargs,
        )

    page_size = 16
    compressed_cache = compressed.reshape(-1, page_size, 1, inputs.head_dim)
    page_table = (
        torch.arange(
            compressed_cache.shape[0],
            dtype=torch.int32,
            device=inputs.index_q.device,
        )
        .unsqueeze(0)
        .expand(inputs.rows, -1)
    )
    return indexer.select_decode_tokens(
        inputs.index_q,
        compressed_cache,
        page_table,
        torch.full(
            (inputs.rows,),
            local_blocks,
            dtype=torch.int32,
            device=inputs.index_q.device,
        ),
        local_blocks,
        query_positions,
        sequence_lengths,
        logical_block_positions=logical_positions,
        static_buffers=static_buffers,
    )


def _run_qsa_multirank_production_integration(rank, world_size, port):
    from sglang.srt.layers.attention.qsa.owner_sparse_attn import (
        owner_sparse_attention,
    )

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["SGLANG_QSA_CANDIDATE_SYMM_MEM"] = "0"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        control_group = dist.new_group(backend="gloo")
        group = _QSAIntegrationGroup(rank, world_size)
        inputs = _qsa_integration_inputs(world_size, rank, device)
        indexer = QSAIndexer.__new__(QSAIndexer)
        indexer.block_topk = inputs.token_topk // inputs.compress_ratio
        indexer.token_topk = inputs.token_topk
        indexer.compress_ratio = inputs.compress_ratio
        indexer.distributed_topk_group = group
        reference_indexer = QSAIndexer.__new__(QSAIndexer)
        reference_indexer.block_topk = indexer.block_topk
        reference_indexer.token_topk = indexer.token_topk
        reference_indexer.compress_ratio = indexer.compress_ratio
        reference_indexer.distributed_topk_group = None
        logical_positions = (
            torch.arange(inputs.local_blocks, device=device) * world_size + rank
        ).to(torch.int32)
        logical_positions = logical_positions.unsqueeze(0).expand(inputs.rows, -1)
        buffers = QSACacheShardingStaticBuffers(
            max_rows=inputs.rows,
            topk=indexer.block_topk,
            owner_topk=inputs.token_topk + inputs.compress_ratio - 1,
            num_heads=inputs.local_heads * world_size,
            head_dim=inputs.head_dim,
            world_size=world_size,
            device=device,
            dtype=torch.bfloat16,
            layer_ids=(0, 1),
            candidate_group=None,
        )
        sharding = QSARawKVSharding(
            local_capacity=inputs.local_k.shape[0],
            world_size=world_size,
            rank=rank,
        )
        reference_sharding = QSARawKVSharding(
            local_capacity=inputs.full_k.shape[0],
            world_size=1,
            rank=0,
        )

        def run(mode):
            selected = _qsa_integration_tokens(
                inputs,
                indexer,
                mode,
                inputs.local_compressed,
                logical_positions=logical_positions,
                static_buffers=buffers.for_layer(0),
            )
            output, lse = owner_sparse_attention(
                inputs.local_q,
                inputs.local_k,
                inputs.local_v,
                selected,
                sharding,
                group=group,
                static_buffers=buffers.for_layer(1),
            )
            return selected, output, lse

        for mode in ("prefill", "decode"):
            expected_selected = _qsa_integration_tokens(
                inputs, reference_indexer, mode, inputs.global_compressed
            )
            expected_output, expected_lse = owner_sparse_attention(
                inputs.full_q,
                inputs.full_k,
                inputs.full_v,
                expected_selected,
                reference_sharding,
            )
            head_slice = slice(
                rank * inputs.local_heads, (rank + 1) * inputs.local_heads
            )
            expected_output = expected_output[:, head_slice]
            expected_lse = expected_lse[:, head_slice]

            actual = run(mode)
            _assert_qsa_integration_close(
                actual[0],
                actual[1],
                actual[2],
                expected_selected,
                expected_output,
                expected_lse,
                control_group,
            )

            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                run(mode)
            torch.cuda.current_stream().wait_stream(warmup_stream)
            torch.cuda.synchronize()
            dist.barrier(device_ids=[rank])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_outputs = run(mode)
            inputs.index_q.add_(
                torch.tensor(0.03125, dtype=torch.bfloat16, device=device)
            )
            inputs.local_q.add_(
                torch.tensor(0.015625, dtype=torch.bfloat16, device=device)
            )
            inputs.full_q.add_(
                torch.tensor(0.015625, dtype=torch.bfloat16, device=device)
            )
            expected_selected = _qsa_integration_tokens(
                inputs, reference_indexer, mode, inputs.global_compressed
            )
            expected_output, expected_lse = owner_sparse_attention(
                inputs.full_q,
                inputs.full_k,
                inputs.full_v,
                expected_selected,
                reference_sharding,
            )
            expected_selected = expected_selected.cpu()
            expected_output = expected_output[:, head_slice].cpu()
            expected_lse = expected_lse[:, head_slice].cpu()
            graph.replay()
            torch.cuda.synchronize()
            replay_selected = graph_outputs[0].cpu()
            replay_output = graph_outputs[1].cpu()
            replay_lse = graph_outputs[2].cpu()
            _assert_qsa_integration_close(
                replay_selected,
                replay_output,
                replay_lse,
                expected_selected,
                expected_output,
                expected_lse,
                control_group,
            )
            del graph_outputs, graph
            torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_qsa_multimem_candidate_transport(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["SGLANG_QSA_CANDIDATE_SYMM_MEM"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        group = _QSAIntegrationGroup(rank, world_size)
        rows, topk, page_size = 2, 512, 16
        global_ids = torch.arange(topk * world_size, dtype=torch.int32, device=device)
        logical_positions = (
            global_ids[(global_ids // page_size) % world_size == rank]
            .unsqueeze(0)
            .expand(rows, -1)
            .contiguous()
        )
        local_indices = (
            torch.arange(topk, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .expand(rows, -1)
            .contiguous()
        )
        row_starts = torch.zeros(rows, dtype=torch.int32, device=device)
        logits = -logical_positions.to(torch.float32)
        logits = (
            logits
            + torch.arange(rows, dtype=torch.float32, device=device).unsqueeze(1) * 0.25
        )
        buffers = QSACacheShardingStaticBuffers(
            max_rows=rows,
            topk=topk,
            num_heads=world_size,
            head_dim=128,
            world_size=world_size,
            device=device,
            dtype=torch.bfloat16,
            candidate_group=group,
        )
        if buffers.candidate_symm_state is None:
            raise AssertionError("H20 multimem candidate transport was not enabled")

        def reference():
            return gather_and_merge_qsa_topk_candidates(
                logits,
                logical_positions,
                group=group,
                topk=topk,
            )

        expected_scores, expected_indices = reference()
        gathered_scores, gathered_indices = gather_and_merge_qsa_topk_candidates(
            logits,
            logical_positions,
            group=group,
            topk=topk,
            static_buffers=buffers,
        )
        torch.testing.assert_close(gathered_scores, expected_scores, rtol=0, atol=0)
        torch.testing.assert_close(gathered_indices, expected_indices, rtol=0, atol=0)

        fused_scores, fused_indices = fused_publish_and_merge_qsa_topk_candidates(
            logits,
            local_indices,
            row_starts,
            logical_positions,
            group=group,
            topk=topk,
            static_buffers=buffers,
        )
        torch.testing.assert_close(fused_scores, expected_scores, rtol=0, atol=0)
        torch.testing.assert_close(fused_indices, expected_indices, rtol=0, atol=0)

        dist.barrier(device_ids=[rank])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_scores, graph_indices = fused_publish_and_merge_qsa_topk_candidates(
                logits,
                local_indices,
                row_starts,
                logical_positions,
                group=group,
                topk=topk,
                static_buffers=buffers,
            )
        logits.add_(rank * 4096.0)
        expected_scores, expected_indices = reference()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_scores, expected_scores, rtol=0, atol=0)
        torch.testing.assert_close(graph_indices, expected_indices, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_qsa_multirank_production_pipeline_eager_and_cuda_graph(world_size):
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"{world_size} CUDA devices are required")
    mp.spawn(
        _run_qsa_multirank_production_integration,
        args=(world_size, _find_free_port()),
        nprocs=world_size,
        join=True,
    )


def test_qsa_multimem_candidate_transport_eager_and_cuda_graph():
    world_size = 4
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"{world_size} CUDA devices are required")
    mp.spawn(
        _run_qsa_multimem_candidate_transport,
        args=(world_size, _find_free_port()),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("capture", [False, True])
def test_qsa_graph_compression_fallback_local_writes(
    monkeypatch, rank, cache_dtype, capture
):
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.layers.rotary_embedding import get_rope

    device = "cuda"
    indexer = QSAIndexer.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.layer_id = 0
    indexer.index_head_dim = 128
    indexer.index_kv_heads = 1
    indexer.compress_ratio = COMPRESS_RATIO
    indexer.rotary_emb = get_rope(
        head_size=128,
        rotary_dim=64,
        max_position=64,
        base=10000,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [8, 12, 12],
            "mrope_interleaved_glm": True,
        },
        dtype=torch.float32,
    ).to(device)
    indexer.k_layernorm = GemmaRMSNorm(128, eps=1e-6).to(
        device=device, dtype=torch.bfloat16
    )
    # Graph runners publish capture mode before invoking the indexer.
    monkeypatch.setattr(qsa_indexer_module, "get_is_capture_mode", lambda: True)
    generator = torch.Generator(device=device).manual_seed(42)
    keys = torch.randn(
        16, 1, 128, dtype=torch.bfloat16, device=device, generator=generator
    )
    cache = torch.full((128, 1, 128), -3, dtype=cache_dtype, device=device)
    pool = SimpleNamespace(
        qsa_compressed_sharding=QSACompressedBlockSharding(256, 16, 2, rank),
        qsa_rope_position_buffer=torch.zeros(16, 3, dtype=torch.int64, device=device),
        get_qsa_key_state_buffer=lambda _: keys,
        get_qsa_compressed_k_buffer=lambda _: cache,
    )
    pool.set_qsa_compressed_k_buffer = MethodType(
        QSATokenToKVPool.set_qsa_compressed_k_buffer, pool
    )
    assert not indexer._use_fused_compress(pool)
    local_locs = torch.tensor([0, 33, 0, 0], dtype=torch.int32, device=device)
    owners = torch.tensor([True, True, False, False], device=device)
    metadata = SimpleNamespace(
        token_to_kv_pool=pool,
        graph_ring_group_locs=torch.arange(16, device=device).reshape(4, 4),
        graph_write_locs=local_locs,
        write_owner_mask=owners,
    )
    pooled = keys.float().reshape(4, 4, 1, 128).mean(1).to(keys.dtype)
    compressed = indexer.normalize_compressed_keys(
        pooled, torch.zeros(3, 4, dtype=torch.int64, device=device)
    ).to(cache_dtype)
    graph = None
    if capture:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            indexer._compress_decode_cuda_graph(metadata)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            indexer._compress_decode_cuda_graph(metadata)
    for slots, mask in (
        ([0, 33, 0, 0], [True, True, False, False]),
        ([0, 0, 65, 0], [False, False, True, True]),
        ([0, 0, 0, 0], [False, False, False, False]),
    ):
        local_locs.copy_(torch.tensor(slots, device=device))
        owners.copy_(torch.tensor(mask, device=device))
        cache.fill_(-3)
        expected = cache.float()
        for row, (slot, owned) in enumerate(zip(slots, mask)):
            if owned:
                expected[slot] = compressed[row].float()
        if graph is None:
            indexer._compress_decode_cuda_graph(metadata)
        else:
            graph.replay()
        torch.testing.assert_close(cache.float(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("is_graph", [False, True])
@pytest.mark.parametrize("expanded", [False, True])
def test_owner_fallback_expands_blocks_and_pending_tail(
    monkeypatch, device, is_graph, expanded
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.compress_ratio = COMPRESS_RATIO
    backend.qsa_profile = SimpleNamespace(block_topk=BLOCK_TOPK, budget=TOKEN_TOPK)
    backend._qsa_cache_sharding_buffers = None
    positions = torch.arange(11, 15, device=device)
    lengths = positions + 1
    slots = torch.arange(63, -1, -1, device=device).repeat(4, 1)
    backend.req_to_token = slots.flip(0)
    metadata = SimpleNamespace(
        token_to_batch_idx=torch.arange(4, device=device),
        sequence_lengths=lengths,
        is_cuda_graph=is_graph,
        row_req_pool_indices=torch.arange(3, -1, -1, device=device),
        token_slot_table=slots,
        indexer_metadata=SimpleNamespace(
            decode_logical_positions=positions,
            get_seqlens_int32=lambda: lengths.to(torch.int32),
        ),
    )
    backend.forward_metadata = metadata
    k = torch.zeros(64, 1, 8, device=device)
    v = torch.arange(64, device=device, dtype=torch.float32)[:, None, None].expand(
        64, 1, 8
    )
    backend.token_to_kv_pool = SimpleNamespace(
        qsa_raw_kv_sharding=object(),
        qsa_owner_group=object(),
        get_key_buffer=lambda _: k,
        get_value_buffer=lambda _: v,
        set_kv_buffer=lambda *args: None,
    )
    backend._select_attention_kv_heads = lambda k, v, layer: (k, v)
    backend._fused_kv_pool_eligible = True
    monkeypatch.setattr(
        qsa_backend_module, "_resolve_trtllm_sparse_decode", lambda: None
    )
    expected_slots = torch.full((4, FINAL_TOPK), -1, device=device, dtype=torch.int32)
    for row in range(4):
        expected_slots[row, :4] = slots[row, 8:12]
        expected_slots[row, 4 : 4 + row] = slots[row, 12 : 12 + row]

    def owner(q, k, v, actual_slots, runtime, **kwargs):
        torch.testing.assert_close(actual_slots, expected_slots)
        return qsa_sparse_attention(q, k, v, actual_slots), None

    monkeypatch.setattr(qsa_backend_module, "owner_sparse_attention", owner)
    blocks = torch.full((4, BLOCK_TOPK), -1, device=device, dtype=torch.int32)
    blocks[:, 0] = 2
    indices = (
        expand_qsa_block_indices(blocks, positions, lengths, COMPRESS_RATIO, TOKEN_TOPK)
        if expanded
        else blocks
    )
    q = torch.zeros(4, 1, 8, device=device)
    layer = SimpleNamespace(layer_id=0, scaling=1.0, tp_q_head_num=1, head_dim=8)
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, out_cache_loc=None)
    actual = backend.forward_decode(q, k[:4], v[:4], layer, batch, topk_indices=indices)
    expected = qsa_sparse_attention(q, k, v, expected_slots).reshape(4, -1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("tp_size", "kv_heads", "sharding_size", "valid"),
    [
        (4, 2, 4, False),
        (4, 2, 2, True),
        (8, 2, 4, True),
        (8, 2, 8, False),
        (8, 4, 2, True),
        (8, 4, 4, False),
    ],
)
def test_qsa_sharding_stays_inside_kv_replicas(tp_size, kv_heads, sharding_size, valid):
    kwargs = dict(
        device="cuda",
        is_qsa=True,
        tp_size=tp_size,
        sharding_size=sharding_size,
        pp_size=1,
        attn_cp_size=1,
        enable_prefill_cp=False,
        ep_size=tp_size,
        moe_dp_size=1,
        disaggregation_mode="null",
        num_attention_heads=24,
        num_key_value_heads=kv_heads,
    )
    if valid:
        validate_qsa_cache_sharding(**kwargs)
    else:
        with pytest.raises(ValueError, match="KV-head replica"):
            validate_qsa_cache_sharding(**kwargs)


def _pack_global_topk_candidates(scores, *, world_size, page_size, topk):
    ranks = []
    for rank in range(world_size):
        global_ids = [
            block
            for block in range(scores.shape[1])
            if (block // page_size) % world_size == rank
        ]
        rank_scores = torch.full((scores.shape[0], topk), float("-inf"))
        rank_ids = torch.full((scores.shape[0], topk), -1, dtype=torch.int32)
        for row in range(scores.shape[0]):
            selected = sorted(
                global_ids, key=lambda block: (-float(scores[row, block]), block)
            )[:topk]
            if selected:
                selected_tensor = torch.tensor(selected)
                width = len(selected)
                rank_scores[row, :width] = scores[row, selected_tensor]
                rank_ids[row, :width] = selected_tensor
        packed = torch.empty(scores.shape[0], topk, 2, dtype=torch.float32)
        packed[..., 0] = rank_scores
        packed[..., 1].view(torch.int32).copy_(rank_ids)
        ranks.append(packed)
    return torch.stack(ranks)


def _select_page_sharded_topk_candidates(scores, *, world_size, page_size, topk):
    global_ids = torch.arange(scores.shape[1], device=scores.device)
    packed_ranks = []
    for rank in range(world_size):
        rank_ids = global_ids[(global_ids // page_size) % world_size == rank]
        rank_scores = scores.index_select(1, rank_ids)
        row_starts = torch.zeros(
            scores.shape[0], dtype=torch.int32, device=scores.device
        )
        row_ends = torch.full_like(row_starts, rank_scores.shape[1])
        local_indices = qsa_fast_topk(
            rank_scores,
            row_starts,
            row_ends,
            topk=topk,
        )
        selected_ids = rank_ids[local_indices.long()].to(torch.int32)
        selected_scores = rank_scores.gather(1, local_indices.long())
        packed = torch.empty(
            (*local_indices.shape, 2),
            dtype=torch.float32,
            device=scores.device,
        )
        packed[..., 0].copy_(selected_scores)
        packed[..., 1].view(torch.int32).copy_(selected_ids)
        packed_ranks.append(packed)
    return merge_qsa_topk_candidates(torch.stack(packed_ranks), topk=topk)


def test_distributed_topk_matches_test_oracle_for_ties_and_ragged_shards():
    scores = torch.tensor(
        [
            [0.1, 9.0, 2.0, 8.0, 7.0, 3.0, 6.0, 4.0, 5.0, 0.0],
            [4.0, 1.0, 4.0, 3.0, 2.0, 4.0, -1.0, 0.0, 4.0, 2.0],
        ]
    )
    actual_scores, actual_indices = merge_qsa_topk_candidates(
        _pack_global_topk_candidates(scores, world_size=3, page_size=2, topk=5),
        topk=5,
    )
    expected_indices = torch.tensor(
        [[1, 3, 4, 6, 8], [0, 2, 5, 8, 3]], dtype=torch.int32
    )
    torch.testing.assert_close(actual_indices, expected_indices)
    torch.testing.assert_close(actual_scores, scores.gather(1, expected_indices.long()))

    ragged = _pack_global_topk_candidates(
        torch.tensor([[2.0, 1.0]]), world_size=4, page_size=1, topk=4
    )
    actual_scores, actual_indices = merge_qsa_topk_candidates(ragged, topk=4)
    torch.testing.assert_close(
        actual_indices, torch.tensor([[0, 1, -1, -1]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        actual_scores, torch.tensor([[2.0, 1.0, float("-inf"), float("-inf")]])
    )


@pytest.mark.parametrize("topk", [512, 2048])
def test_page_sharded_local_topk_matches_global_ids_without_ties(topk):
    torch.manual_seed(20261013 + topk)
    scores = torch.randn(1, topk * 8, dtype=torch.float32, device="cuda")
    _, actual_indices = _select_page_sharded_topk_candidates(
        scores,
        world_size=4,
        page_size=16,
        topk=topk,
    )
    expected_indices = torch.topk(scores, topk, dim=1, sorted=False).indices.to(
        torch.int32
    )
    torch.testing.assert_close(
        actual_indices.sort(dim=1).values,
        expected_indices.sort(dim=1).values,
    )


@pytest.mark.parametrize("topk", [512, 2048])
def test_page_sharded_local_topk_preserves_score_multiset_at_boundary_ties(topk):
    scores = torch.full(
        (1, topk * 8),
        -1.0,
        dtype=torch.float32,
        device="cuda",
    )
    strict_count = topk - 12
    scores[0, :strict_count] = torch.arange(
        strict_count,
        0,
        -1,
        dtype=torch.float32,
        device=scores.device,
    )
    scores[0, strict_count : strict_count + 64] = 0.0

    actual_scores, _ = _select_page_sharded_topk_candidates(
        scores,
        world_size=4,
        page_size=16,
        topk=topk,
    )
    expected_scores = torch.topk(scores, topk, dim=1, sorted=False).values
    torch.testing.assert_close(
        actual_scores.sort(dim=1).values,
        expected_scores.sort(dim=1).values,
    )


def test_distributed_topk_requires_explicit_logical_block_positions():
    indexer = QSAIndexer.__new__(QSAIndexer)
    indexer.block_topk = 2
    indexer.distributed_topk_group = SimpleNamespace(
        world_size=2,
        rank_in_group=0,
    )

    with pytest.raises(
        ValueError,
        match="distributed QSA top-k requires logical block positions",
    ):
        indexer._maybe_merge_distributed_topk(
            logits=torch.tensor([[3.0, 2.0, 1.0]]),
            local_indices=torch.tensor([[0, 1]], dtype=torch.int32),
            row_starts=torch.tensor([0], dtype=torch.int32),
        )


def test_fused_publish_requires_explicit_logical_block_positions():
    with pytest.raises(
        ValueError,
        match="fused candidate publish requires logical block positions",
    ):
        fused_publish_and_merge_qsa_topk_candidates(
            logits=torch.empty(1, 2),
            local_indices=torch.empty(1, 2, dtype=torch.int32),
            row_starts=torch.zeros(1, dtype=torch.int32),
            logical_block_positions=None,
            group=SimpleNamespace(world_size=2, rank_in_group=0),
            topk=2,
            static_buffers=None,
        )


@pytest.mark.parametrize("option", ["--qsa-cache-sharding-size", "--state-cp-size"])
def test_qsa_cache_sharding_cli_keeps_state_cp_alias(option):
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(["--model", "dummy", option, "4"])
    assert ServerArgs.from_cli_args(parsed).qsa_cache_sharding_size == 4


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"device": "cpu"}, "CUDA"),
        ({"is_qsa": False}, "QSA"),
        ({"tp_size": 3, "sharding_size": 2}, "divide"),
        ({"pp_size": 2}, "pipeline"),
        ({"attn_cp_size": 2}, "attention context"),
        ({"enable_prefill_cp": True}, "prefill"),
        ({"disaggregation_mode": "decode"}, "PD disaggregation"),
        ({"dcp_size": 2}, "decode context"),
    ],
)
def test_qsa_cache_sharding_rejects_unsupported_combinations(overrides, message):
    kwargs = {
        "device": "cuda",
        "is_qsa": True,
        "tp_size": 4,
        "sharding_size": 2,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "pp_size": 1,
        "attn_cp_size": 1,
        "enable_prefill_cp": False,
        "ep_size": 2,
        "moe_dp_size": 1,
        "disaggregation_mode": "null",
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        validate_qsa_cache_sharding(**kwargs)


@pytest.mark.parametrize("sharding_size", [2, 4])
def test_qsa_cache_sharding_accepts_supported_degrees(sharding_size):
    validate_qsa_cache_sharding(
        device="cuda",
        is_qsa=True,
        tp_size=8,
        sharding_size=sharding_size,
        num_attention_heads=24,
        num_key_value_heads=2,
        pp_size=1,
        attn_cp_size=1,
        enable_prefill_cp=False,
        ep_size=8,
        moe_dp_size=1,
        disaggregation_mode="null",
    )


def test_qsa_chunk_prefill_accepts_fp8_cached_prefix():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8-capable CUDA GPU required")

    torch.manual_seed(42)
    device = torch.device("cuda")
    num_requests, num_q_heads, num_kv_heads, head_dim, topk = 2, 4, 1, 128, 16
    q = torch.randn(
        num_requests, num_q_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        num_requests * topk,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    v = torch.randn(
        num_requests * topk,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    indices = torch.arange(topk, dtype=torch.int32, device=device).repeat(
        num_requests, 1
    )
    cu_q = torch.arange(num_requests + 1, dtype=torch.int32, device=device)
    cu_k = torch.arange(
        0,
        (num_requests + 1) * topk,
        topk,
        dtype=torch.int32,
        device=device,
    )
    kv_lens = torch.full((num_requests,), topk, dtype=torch.int32, device=device)
    scale = head_dim**-0.5

    actual = sparse_gqa_fwd_interface_triton_ck(
        q, k, v, indices, cu_q, cu_k, kv_lens, scale
    )
    expected = sparse_gqa_fwd_interface_triton_ck(
        q,
        k.to(torch.bfloat16),
        v.to(torch.bfloat16),
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((12, 0), True), ((12, 1), False), ((10, 0), False)],
)
def test_is_sm120_matches_exact_capability(monkeypatch, capability, expected):
    from sglang.srt.utils import common

    common.is_sm120.cache_clear()
    monkeypatch.setattr(common, "is_cuda", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)

    try:
        assert common.is_sm120() is expected
    finally:
        common.is_sm120.cache_clear()


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((12, 1), True), ((12, 0), False), ((10, 0), False)],
)
def test_is_sm121_matches_exact_capability(monkeypatch, capability, expected):
    from sglang.srt.utils import common

    common.is_sm121.cache_clear()
    monkeypatch.setattr(common, "is_cuda", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)

    try:
        assert common.is_sm121() is expected
    finally:
        common.is_sm121.cache_clear()


@pytest.mark.parametrize(
    ("sm100", "sm120", "expected_enabled"),
    [(False, True, True), (True, False, True), (False, False, False)],
    ids=["sm120", "sm100", "other-sm12x"],
)
def test_qsa_trtllm_sparse_decode_arch_gate(
    monkeypatch, sm100, sm120, expected_enabled
):
    resolver = qsa_backend_module._resolve_trtllm_sparse_decode
    resolver.cache_clear()

    trtllm_decode_func = object()
    flashinfer_decode = ModuleType("flashinfer.decode")
    flashinfer_decode.trtllm_batch_decode_with_kv_cache = trtllm_decode_func

    monkeypatch.setattr(qsa_backend_module, "is_sm100_supported", lambda: sm100)
    monkeypatch.setattr(qsa_backend_module, "is_sm120", lambda: sm120)
    monkeypatch.setitem(sys.modules, flashinfer_decode.__name__, flashinfer_decode)

    try:
        expected = trtllm_decode_func if expected_enabled else None
        assert resolver() is expected
    finally:
        resolver.cache_clear()


def test_qsa_sm121_resolves_kda_varlen_kernel(monkeypatch):
    resolver = qsa_backend_module._resolve_flash_attn_varlen_func
    resolver.cache_clear()
    monkeypatch.setattr("sglang.srt.utils.is_sm121", lambda: True)

    try:
        assert resolver() is qwen38_qsa_sm121_varlen
    finally:
        resolver.cache_clear()


def test_qsa_sm121_compaction_and_attention_match_sparse_reference():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("SM121-only kernel")
    torch.manual_seed(2028)
    device = torch.device("cuda")
    batch, topk = 3, FINAL_TOPK
    num_q_heads, num_kv_heads, head_dim = 24, 2, 256
    sequence_lengths = torch.tensor([17, 1050, 4096], dtype=torch.int32)
    valid_counts_cpu = [17, 911, topk]
    max_sequence_length = int(sequence_lengths.max())

    req_to_token = torch.arange(
        batch * max_sequence_length, dtype=torch.int32, device=device
    ).reshape(batch, max_sequence_length)
    indices = torch.full((batch, topk), -1, dtype=torch.int32, device=device)
    for row, valid_count in enumerate(valid_counts_cpu):
        indices[row, :valid_count] = torch.randperm(
            int(sequence_lengths[row]), device=device, dtype=torch.int64
        )[:valid_count].to(torch.int32)

    slots = torch.full_like(indices, -1)
    for row, valid_count in enumerate(valid_counts_cpu):
        slots[row, :valid_count] = req_to_token[row, indices[row, :valid_count].long()]

    pool_size = batch * max_sequence_length
    q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(
        pool_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn_like(k_cache)
    valid_counts = torch.empty(batch, dtype=torch.int32, device=device)
    cu_k = torch.empty(batch + 1, dtype=torch.int32, device=device)
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
    qwen_sparse_fa2_cu_seqlens_triton(
        sequence_lengths.to(device),
        indices,
        valid_counts,
        cu_k,
        batch,
        topk,
    )
    packed_k = torch.empty(
        batch * topk,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    packed_v = torch.empty_like(packed_k)
    qwen_sparse_kv_extraction_compact_triton(
        k_cache,
        v_cache,
        req_to_token,
        torch.arange(batch, dtype=torch.int32, device=device),
        indices,
        sequence_lengths.to(device),
        cu_k,
        packed_k,
        packed_v,
        batch,
        topk,
    )

    scale = head_dim**-0.5
    actual = qwen38_qsa_sm121_varlen(
        q,
        packed_k,
        packed_v,
        cu_q,
        cu_k,
        max_seqlen_k=topk,
        softmax_scale=scale,
    )
    expected = qsa_sparse_attention(q, k_cache, v_cache, slots, scale)
    assert valid_counts.tolist() == valid_counts_cpu
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _compressed_config_namespace(**overrides):
    fields = dict(
        model_type="qwen4_exp",
        indexer_n_heads=8,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=TOKEN_TOPK,
        indexer_compress_ratio=COMPRESS_RATIO,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_qsa_profile_parses_compressed_qwen4_exp_schema():
    from sglang.srt.layers.attention.qsa.config import (
        is_qwen_qsa,
        parse_qsa_profile,
    )

    wrapped = Qwen4ExpConfig(
        text_config={
            "hc_count": 4,
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "indexer_n_heads": 8,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": TOKEN_TOPK,
            "indexer_compress_ratio": COMPRESS_RATIO,
        },
    )
    for config in (wrapped, _compressed_config_namespace()):
        profile = parse_qsa_profile(config)
        assert profile.n_heads == 8
        assert profile.kv_heads == 1
        assert profile.head_dim == 128
        assert profile.budget == TOKEN_TOPK
        assert profile.compress_ratio == COMPRESS_RATIO
        assert profile.block_topk == BLOCK_TOPK
        assert is_qwen_qsa(config)
    # The legacy backend module keeps re-exporting the shared detector.
    assert qsa_backend_module.is_qwen_qsa is is_qwen_qsa
    assert not is_qwen_qsa(SimpleNamespace())
    assert parse_qsa_profile(None) is None


def test_qsa_profile_rejects_malformed_compressed_schema():
    from sglang.srt.layers.attention.qsa.config import parse_qsa_profile

    bad_configs = {
        "missing": _compressed_config_namespace(indexer_budget=None),
        "ratio_one": _compressed_config_namespace(indexer_compress_ratio=1),
        "indivisible": _compressed_config_namespace(indexer_budget=TOKEN_TOPK - 2),
        "bad_block_topk": _compressed_config_namespace(
            indexer_budget=1024, indexer_compress_ratio=4
        ),
        "kv_heads": _compressed_config_namespace(indexer_kv_heads=2),
    }
    for name, config in bad_configs.items():
        try:
            parse_qsa_profile(config)
        except ValueError:
            continue
        raise AssertionError(f"{name} compressed config must be rejected")


def test_qsa_glue_builds_compressed_indexer(monkeypatch):
    from sglang.srt.layers.attention.qsa.glue import build_qsa_indexer

    recorded = {}

    class _FakeIndexer:
        def __init__(
            self,
            config,
            layer_id,
            quant_config=None,
            prefix="",
            rotary_emb=None,
            cache_sharding_runtime=None,
        ):
            recorded.update(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=prefix,
                rotary_emb=rotary_emb,
            )

    monkeypatch.setattr(qsa_indexer_module, "QSAIndexer", _FakeIndexer)
    rotary = object()
    config = _compressed_config_namespace()
    indexer = build_qsa_indexer(
        config, layer_id=7, quant_config="qc", prefix="p", rotary_emb=rotary
    )
    assert isinstance(indexer, _FakeIndexer)
    assert recorded == dict(
        config=config, layer_id=7, quant_config="qc", prefix="p", rotary_emb=rotary
    )

    try:
        build_qsa_indexer(SimpleNamespace(), layer_id=0, rotary_emb=rotary)
    except ValueError as exc:
        assert "QSA indexer schema" in str(exc)
    else:
        raise AssertionError("non-QSA configs must be rejected")


def test_qsa_glue_fetches_indexer_metadata_without_model_unwrap():
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.qsa.glue import get_qsa_indexer_metadata

    full = SimpleNamespace(
        get_indexer_metadata=lambda layer_id, forward_batch: f"meta-{layer_id}",
        token_to_kv_pool=None,
        req_to_token_pool=None,
        kv_index_translator=None,
        needs_cpu_seq_lens=True,
    )
    hybrid = HybridLinearAttnBackend(
        full_attn_backend=full,
        linear_attn_backend=SimpleNamespace(needs_cpu_seq_lens=True),
        full_attn_layers=[3],
    )
    # Full-attention layers fetch through the hybrid wrapper directly.
    assert get_qsa_indexer_metadata(hybrid, 3, object()) == "meta-3"
    # A backend that provides no indexer metadata anywhere must surface an
    # explicit error instead of silently running without sparse selection.
    empty = SimpleNamespace(get_indexer_metadata=lambda layer_id, batch: None)
    try:
        get_qsa_indexer_metadata(empty, 3, object())
    except RuntimeError as exc:
        assert "indexer metadata" in str(exc)
    else:
        raise AssertionError("QSA must fail when no indexer metadata exists")


def test_qsa_draft_extend_backend_decision_follows_profile():
    from sglang.srt.layers.attention.qsa.config import parse_qsa_profile
    from sglang.srt.speculative.draft_utils import DraftBackendFactory

    def factory(config):
        runner = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=config),
            draft_attention_backend=None,
        )
        return DraftBackendFactory(
            draft_model_runner=runner,
            topk=1,
            speculative_num_steps=3,
            qsa_profile=parse_qsa_profile(config),
        )

    compressed = factory(_compressed_config_namespace())
    backend = compressed.create_draft_extend_backend()
    assert isinstance(backend, QwenSparseAttnBackend)
    assert backend.runner is compressed.draft_model_runner
    assert backend.decode_attention_backend_str == "qsa"


def _make_mtp_draft_batch(steps: int, seq_lens=(8, 16), loc_base: int = 40):
    bs = len(seq_lens)
    pool = _FakeQSAPool(capacity=bs * steps + 256)
    req_to_token = torch.stack(
        [torch.arange(i * 64, (i + 1) * 64, dtype=torch.int32) for i in range(bs)]
    )
    runner = SimpleNamespace(
        device="cpu",
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        model_config=SimpleNamespace(
            context_len=64,
            hf_config=SimpleNamespace(indexer_compress_ratio=COMPRESS_RATIO),
        ),
    )
    backend = QwenSparseMultiStepDraftBackend(
        runner, topk=1, speculative_num_steps=steps
    )
    full_out_cache_loc = loc_base + torch.arange(bs * steps, dtype=torch.int32)
    forward_batch = SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=runner.req_to_token_pool,
        batch_size=bs,
        req_pool_indices=torch.arange(bs, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        positions=torch.tensor([length - 1 for length in seq_lens], dtype=torch.int64),
        out_cache_loc=full_out_cache_loc,
        forward_mode=ForwardMode.DECODE,
    )
    return backend, forward_batch, pool


def test_qsa_cuda_graph_pads_dynamic_draft_extend_rows():
    row_lengths, row_req_pool_indices, row_prefix_lengths = (
        QwenSparseAttnBackend._graph_speculative_layout(
            bs=2,
            num_tokens=8,
            req_pool_indices=torch.tensor([3, 7], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([11, 21], dtype=torch.int32),
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            spec_info=SimpleNamespace(extend_seq_lens_cpu=[1, 2]),
        )
    )
    assert row_lengths.tolist() == [11, 20, 21, 1, 1, 1, 1, 1]
    assert row_req_pool_indices.tolist() == [3, 7, 7, 3, 3, 3, 3, 3]
    assert row_prefix_lengths.tolist() == [10, 19, 19, 0, 0, 0, 0, 0]


def test_qsa_cuda_graph_target_verify_ignores_capture_bucket_requests():
    row_lengths, row_req_pool_indices, row_prefix_lengths = (
        QwenSparseAttnBackend._graph_speculative_layout(
            bs=2,
            num_tokens=8,
            req_pool_indices=torch.tensor([3, 0], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([9, 1], dtype=torch.int32),
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(draft_token_num=4),
            num_padding=1,
        )
    )
    assert row_lengths.tolist() == [10, 11, 12, 13, 1, 1, 1, 1]
    assert row_req_pool_indices.tolist() == [3] * 8
    assert row_prefix_lengths.tolist() == [9, 9, 9, 9, 0, 0, 0, 0]


def test_qsa_target_verify_rejects_branching_speculation():
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.compress_ratio = 4
    try:
        backend._require_chain_speculation(
            ForwardMode.TARGET_VERIFY, SimpleNamespace(topk=2)
        )
    except NotImplementedError as exc:
        assert "topk=1" in str(exc)
    else:
        raise AssertionError("QSA target verification must reject tree branches")
    # The pending-group ring keys state by position % ratio: a verify window
    # wider than the ratio would collide within one forward.
    try:
        backend._require_chain_speculation(
            ForwardMode.TARGET_VERIFY, SimpleNamespace(topk=1, draft_token_num=5)
        )
    except NotImplementedError as exc:
        assert "compress ratio" in str(exc)
    else:
        raise AssertionError("QSA must reject draft windows wider than the ratio")
    backend._require_chain_speculation(
        ForwardMode.TARGET_VERIFY, SimpleNamespace(topk=1, draft_token_num=4)
    )


def test_qsa_mtp_cuda_graph_padding_stays_below_compression_boundary():
    backend, forward_batch, _ = _make_mtp_draft_batch(steps=4, seq_lens=(3, 1))

    class Recorder:
        def __init__(self):
            self.capture_lengths = None
            self.replay_lengths = None

        def _capture_cuda_graph_metadata(self, **kwargs):
            self.capture_lengths = kwargs["seq_lens"].clone()

        def _replay_cuda_graph_metadata(self, *, bs, seq_lens, **kwargs):
            self.replay_lengths = seq_lens[:bs].clone()

    recorders = [Recorder() for _ in backend.attn_backends]
    backend.attn_backends = recorders
    forward_batch.spec_info = SimpleNamespace()

    backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)
    forward_batch.num_padding = 1
    backend.init_forward_metadata_out_graph(forward_batch)

    assert [r.capture_lengths.tolist() for r in recorders] == [[1, 1]] * 3
    assert [r.replay_lengths.tolist() for r in recorders] == [
        [4, 1],
        [5, 1],
        [6, 1],
    ]


def test_qsa_indexer_ignores_dp_attention_token_padding():
    calls = {}

    def run(num_rows):
        mapping = torch.zeros(15, dtype=torch.int32)
        metadata = SimpleNamespace(
            is_cuda_graph=False,
            token_to_kv_pool=None,
            compress_member_rows=None,
            decode_logical_positions=None,
            pending_ring_slots=None,
            get_token_to_batch_idx=lambda: mapping,
            get_prefill_mqa_inputs=lambda layer_id, logical_positions: (
                torch.empty(0, 1, 128),
                torch.zeros(15, dtype=torch.int32),
                torch.zeros(15, dtype=torch.int32),
                torch.tensor([15], dtype=torch.int32),
            ),
        )
        indexer = SimpleNamespace(
            layer_id=0,
            project_qk=lambda hidden, rope_positions, **kwargs: (
                hidden.reshape(-1, 1, 2),
                hidden.reshape(-1, 1, 2),
                False,
            ),
            _pending_ring_slots=lambda metadata, logical_positions, is_extend: (
                torch.zeros(logical_positions.numel(), dtype=torch.long)
            ),
            update_key_state_and_compress=lambda token_k, logical, rope, meta, state_slots=None, state_stored=False: (
                calls.update(
                    token_rows=token_k.shape[0],
                    logical_rows=logical.numel(),
                    rope_rows=rope.numel(),
                )
            ),
            select_prefill_tokens=lambda q, keys, starts, ends, logical, lengths, **kwargs: (
                q.squeeze(1)
            ),
        )
        indexer._forward_impl = MethodType(QSAIndexer._forward_impl, indexer)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            positions=torch.cat(
                [torch.arange(15), torch.zeros(num_rows - 15, dtype=torch.long)]
            ),
            out_cache_loc=torch.arange(num_rows, dtype=torch.int32),
        )
        hidden = torch.arange(num_rows * 2, dtype=torch.float32).reshape(num_rows, 2)
        return QSAIndexer.forward_cuda(
            indexer, hidden, forward_batch.positions, forward_batch, metadata
        )

    unpadded = run(15)
    padded = run(16)
    torch.testing.assert_close(padded, unpadded)
    assert padded.shape[0] == 15
    assert calls == {"token_rows": 15, "logical_rows": 15, "rope_rows": 15}


def test_qsa_cuda_extend_ignores_dp_attention_padding(monkeypatch):
    if not torch.cuda.is_available():
        return

    kernel_shapes = []

    def fake_sparse_gqa(q, k, v, max_seqlen_k, indices, cu_seqlens, scale):
        kernel_shapes.append((q.shape[0], k.shape[0], v.shape[0], indices.shape[0]))
        return q + 1

    monkeypatch.setattr(
        qsa_backend_module, "sparse_gqa_fwd_interface_triton", fake_sparse_gqa
    )
    backend = QwenSparseAttnBackend()

    class Pool:
        def set_kv_buffer(self, layer, loc, k, v):
            pass

    pool = Pool()
    backend.token_to_kv_pool = pool
    layer = SimpleNamespace(tp_q_head_num=1, head_dim=2, layer_id=0, scaling=1.0)
    topk = torch.zeros(15, 3, dtype=torch.int32, device="cuda")

    def run(num_rows):
        values = torch.arange(num_rows * 2, dtype=torch.float32, device="cuda").reshape(
            num_rows, 2
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            token_to_kv_pool=pool,
            out_cache_loc=torch.arange(num_rows, dtype=torch.int32, device="cuda"),
            extend_seq_lens=torch.tensor([15], dtype=torch.int32, device="cuda"),
            extend_seq_lens_cpu=[15],
            seq_lens_cpu=[15],
        )
        return backend.forward_extend(
            values, values.clone(), values.clone(), layer, batch, topk_indices=topk
        )

    unpadded = run(15)
    padded = run(16)
    torch.testing.assert_close(padded[:15], unpadded)
    torch.testing.assert_close(padded[15], torch.zeros(2, device="cuda"))
    assert kernel_shapes == [(15, 15, 15, 15), (15, 15, 15, 15)]


def _make_paged_extend_backend():
    metadata = SimpleNamespace(
        is_cuda_graph=False,
        token_to_batch_idx=torch.zeros(3, dtype=torch.int32),
        sequence_lengths=torch.tensor([8], dtype=torch.int32),
        token_slot_table=torch.arange(16, dtype=torch.int32).reshape(1, 16),
    )
    backend = QwenSparseAttnBackend()
    backend.forward_metadata = metadata
    backend.owner_sparse_attn_group = None
    backend.owner_sparse_attn_runtime = None
    backend._select_attention_kv_heads = lambda k, v, layer: (k, v)

    class Pool:
        def set_kv_buffer(self, layer, loc, k, v):
            pass

        def get_key_buffer(self, layer_id):
            return torch.zeros(16, 1, 2)

        def get_value_buffer(self, layer_id):
            return torch.zeros(16, 1, 2)

    layer = SimpleNamespace(tp_q_head_num=1, head_dim=2, layer_id=0, scaling=1.0)
    pool = Pool()
    backend.token_to_kv_pool = pool
    return backend, pool, layer


def test_qsa_paged_extend_trims_padding_rows_and_restores_output(monkeypatch):
    kernel_rows = []

    def fake_sparse_attention(q, k, v, slots, scale):
        kernel_rows.append((q.shape[0], slots.shape[0]))
        return q + 1

    monkeypatch.setattr(
        qsa_backend_module, "qsa_sparse_attention", fake_sparse_attention
    )
    backend, pool, layer = _make_paged_extend_backend()
    topk = torch.zeros(3, 2, dtype=torch.int32)

    def run(num_rows, mode):
        values = torch.arange(num_rows * 2, dtype=torch.float32).reshape(num_rows, 2)
        batch = SimpleNamespace(
            forward_mode=mode,
            token_to_kv_pool=pool,
            out_cache_loc=torch.arange(num_rows, dtype=torch.int32),
        )
        return backend.forward_extend(
            values, values.clone(), values.clone(), layer, batch, topk_indices=topk
        )

    # DP attention pads the physical q rows past the semantic draft rows in
    # every speculative paged mode; padding must round-trip through the kernel.
    for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
        unpadded = run(3, mode)
        padded = run(5, mode)
        torch.testing.assert_close(padded[:3], unpadded)
        torch.testing.assert_close(padded[3:], torch.zeros(2, 2))
        assert padded.shape == (5, 2)
    # The reference path (and any kernel behind it) only ever saw valid rows.
    assert kernel_rows == [(3, 3)] * 4

    # More semantic rows than physical q rows is a bug and must not silently
    # truncate the top-k table.
    short_q = torch.zeros(2, 2, dtype=torch.float32)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        token_to_kv_pool=pool,
        out_cache_loc=torch.arange(2, dtype=torch.int32),
    )
    try:
        backend.forward_extend(
            short_q, short_q.clone(), short_q.clone(), layer, batch, topk_indices=topk
        )
    except ValueError as exc:
        assert "top-k rows exceed query rows" in str(exc)
    else:
        raise AssertionError("QSA must reject top-k rows beyond the query rows")


def test_qsa_indexer_rejects_shorter_source_than_request_mapping():
    metadata = SimpleNamespace(
        get_token_to_batch_idx=lambda: torch.zeros(16, dtype=torch.int32)
    )
    indexer = SimpleNamespace()
    indexer._forward_impl = MethodType(QSAIndexer._forward_impl, indexer)
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, positions=torch.arange(15))
    try:
        QSAIndexer.forward_cuda(
            indexer, torch.zeros(15, 2), batch.positions, batch, metadata
        )
    except ValueError as exc:
        assert "logical positions are shorter" in str(exc)
    else:
        raise AssertionError("QSA must reject a request mapping longer than positions")


def _make_qsa_runner_and_pool(num_reqs=4):
    pool = _FakeQSAPool(capacity=256)
    req_to_token = torch.stack(
        [torch.arange(i * 64, (i + 1) * 64, dtype=torch.int32) for i in range(num_reqs)]
    )
    runner = SimpleNamespace(
        device="cpu",
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        model_config=SimpleNamespace(
            context_len=64,
            hf_config=SimpleNamespace(indexer_compress_ratio=COMPRESS_RATIO),
        ),
    )
    return runner, pool, runner.req_to_token_pool


def test_qsa_idle_metadata_builds_empty_rows():
    runner, pool, req_pool = _make_qsa_runner_and_pool()
    idle_batch = SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=req_pool,
        req_pool_indices=torch.empty(0, dtype=torch.int32),
        seq_lens=torch.empty(0, dtype=torch.int32),
        seq_lens_cpu=torch.empty(0, dtype=torch.int64),
        positions=torch.empty(0, dtype=torch.int64),
        out_cache_loc=torch.empty(0, dtype=torch.int32),
        # The MTP wrapper forwards idle batches as zero-row DECODE steps.
        forward_mode=ForwardMode.DECODE,
    )
    backend = QwenSparseAttnBackend(runner)
    backend.init_forward_metadata(idle_batch)
    metadata = backend.forward_metadata
    assert metadata.sequence_lengths.numel() == 0
    assert metadata.token_to_batch_idx.numel() == 0
    assert metadata.row_req_pool_indices.numel() == 0
    assert metadata.token_slot_table.shape[0] == 0
    assert metadata.indexer_metadata.out_cache_loc.numel() == 0

    # The MTP multi-step wrapper forwards idle batches as zero-row DECODE
    # steps; those must not fall into the empty extend/max() path either.
    draft = QwenSparseMultiStepDraftBackend(runner, topk=1, speculative_num_steps=2)
    draft.init_forward_metadata(idle_batch)
    step_metadata = draft.attn_backends[0].forward_metadata
    assert step_metadata.sequence_lengths.numel() == 0
    assert step_metadata.row_req_pool_indices.numel() == 0
    # Per-step out_cache_loc slicing must stay empty without allocating rows.
    for attn_backend in draft.attn_backends:
        assert attn_backend.forward_metadata.indexer_metadata.out_cache_loc.numel() == 0


def test_qsa_decode_requires_one_query_row_per_request():
    runner, pool, req_pool = _make_qsa_runner_and_pool()
    backend = QwenSparseAttnBackend(runner)
    forward_batch = SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=req_pool,
        req_pool_indices=torch.tensor([1, 3], dtype=torch.int32),
        seq_lens=torch.tensor([8, 16], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([8, 16], dtype=torch.int32),
        # 2 request rows but 3 query rows: this layout is ambiguous under DP.
        positions=torch.tensor([7, 15, 0], dtype=torch.int64),
        out_cache_loc=torch.arange(3, dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
    )
    try:
        backend.init_forward_metadata(forward_batch)
    except ValueError as exc:
        assert "exactly one query row per request" in str(exc)
    else:
        raise AssertionError("QSA decode must reject multi-row requests")


def test_qsa_extend_rope_matrix_uses_mrope_coordinates():
    """extend_rope_matrix must be built from forward_batch.mrope_positions ([3, N])
    when set, so prefill- and decode-compressed keys RoPE with the same coordinates."""
    runner, pool, req_pool = _make_qsa_runner_and_pool()
    backend = QwenSparseAttnBackend(runner)
    num_tokens = 8
    flat = torch.arange(num_tokens, dtype=torch.int64)
    mrope = torch.stack([flat, flat + 100, flat + 200])
    forward_batch = SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=req_pool,
        req_pool_indices=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([num_tokens], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([num_tokens], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=torch.tensor([num_tokens], dtype=torch.int32),
        extend_prefix_lens=torch.zeros(1, dtype=torch.int32),
        positions=flat,
        mrope_positions=mrope,
        input_ids=torch.zeros(num_tokens, dtype=torch.int32),
        out_cache_loc=torch.arange(num_tokens, dtype=torch.int32),
        _original_forward_mode=None,
    )
    backend.init_forward_metadata(forward_batch)
    got = backend.forward_metadata.indexer_metadata.extend_rope_matrix
    assert got is not None
    assert torch.equal(got, mrope.transpose(0, 1))


def test_qsa_speculative_pseudo_extend_is_rejected():
    runner, pool, req_pool = _make_qsa_runner_and_pool()
    backend = QwenSparseAttnBackend(runner)
    for original_mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
        forward_batch = SimpleNamespace(
            token_to_kv_pool=pool,
            req_to_token_pool=req_pool,
            req_pool_indices=torch.tensor([1], dtype=torch.int32),
            seq_lens=torch.tensor([10], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([10], dtype=torch.int32),
            # DP MAX_LEN pseudo-extend rewrites the mode and loses the
            # per-request draft fan-out of the original speculative mode.
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.ones(1, dtype=torch.int32),
            positions=torch.tensor([9], dtype=torch.int64),
            out_cache_loc=torch.arange(1, dtype=torch.int32),
            _original_forward_mode=original_mode,
        )
        try:
            backend.init_forward_metadata(forward_batch)
        except ValueError as exc:
            assert "pseudo-extend" in str(exc)
        else:
            raise AssertionError(f"QSA must reject pseudo-extend of {original_mode}")


class _FakeQSAPool:
    cache_sharding_runtime = resolve_qsa_cache_sharding_runtime(SimpleNamespace())
    qsa_index_kv_heads = 1
    qsa_index_head_dim = 128
    qsa_compressed_page_size = 64
    qsa_compress_ratio = COMPRESS_RATIO
    qsa_token_topk = TOKEN_TOPK
    qsa_block_topk = BLOCK_TOPK

    def __init__(self, capacity=32):
        self.token_k = torch.zeros(capacity, 1, 128, dtype=torch.bfloat16)
        self.compressed_k = torch.zeros(3 * 64, 1, 128, dtype=torch.bfloat16)
        self.mapping = torch.full((capacity,), -1, dtype=torch.int32)
        self.qsa_index_loc_map = self.mapping
        self.qsa_page_loc_map = torch.full((capacity,), -1, dtype=torch.int32)
        self.qsa_free_index_slots = torch.empty(0, dtype=torch.int32)
        self.rope_positions = torch.zeros(capacity, 3, dtype=torch.int64)
        self.qsa_page_table = None
        self.qsa_page_owned = None
        self.next_page = 1
        self.bulk_alloc_calls = 0

    def set_qsa_key_state_buffer(self, layer_id, loc, token_k):
        self.token_k[loc.long()] = token_k

    def get_qsa_key_state_buffer(self, layer_id):
        return self.token_k

    def clear_qsa_key_state_buffer(self, layer_id, loc):
        self.token_k[loc.long()] = 0

    def set_qsa_rope_position_buffer(self, loc, positions):
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        self.rope_positions[loc.long()] = positions.transpose(0, 1)

    def get_qsa_rope_position_buffer(self, loc):
        return self.rope_positions[loc.long()]

    def get_or_alloc_qsa_index_locs(self, full_locs, page_anchor_locs, page_offsets):
        result = torch.empty_like(full_locs, dtype=torch.int32)
        for index, (full_loc, anchor, offset) in enumerate(
            zip(full_locs.long(), page_anchor_locs.long(), page_offsets.int())
        ):
            if self.mapping[anchor] < 0:
                self.mapping[anchor] = self.next_page * 64
                self.next_page += 1
            result[index] = self.mapping[anchor] + offset
            self.mapping[full_loc] = result[index]
        return result

    def alloc_qsa_compressed_pages(self, num_pages):
        self.bulk_alloc_calls += 1
        pages = torch.arange(
            self.next_page, self.next_page + num_pages, dtype=torch.int32
        )
        self.next_page += num_pages
        return pages

    def copy_qsa_compressed_page_prefixes(
        self, source_pages, destination_pages, copy_blocks
    ):
        for source, destination, blocks in zip(
            source_pages.tolist(), destination_pages.tolist(), copy_blocks.tolist()
        ):
            src = source * 64
            dst = destination * 64
            self.compressed_k[dst : dst + blocks] = self.compressed_k[
                src : src + blocks
            ]

    def alloc_qsa_compressed_page(self, source_page=-1, copy_blocks=0):
        page = int(self.alloc_qsa_compressed_pages(1)[0])
        if source_page >= 0 and copy_blocks > 0:
            self.copy_qsa_compressed_page_prefixes(
                torch.tensor([source_page]),
                torch.tensor([page]),
                torch.tensor([copy_blocks]),
            )
        return page

    def set_qsa_index_locs(self, full_locs, page, page_offsets):
        result = page * 64 + page_offsets.int()
        self.mapping[full_locs.long()] = result
        return result

    def get_qsa_index_locs(self, full_locs):
        result = self.mapping[full_locs.long()]
        assert torch.all(result >= 0)
        return result

    def get_qsa_compressed_page_locs(self, page_anchor_locs):
        return self.get_qsa_index_locs(page_anchor_locs) // 64

    def set_qsa_compressed_k_buffer(self, layer_id, loc, compressed_k):
        self.compressed_k[loc.long()] = compressed_k

    def get_qsa_compressed_k_buffer(self, layer_id):
        return self.compressed_k


class _DispatchIndexer:
    layer_id = 3
    index_n_heads = 4
    compress_ratio = 4
    _pending_ring_slots = QSAIndexer._pending_ring_slots
    # forward_cuda (the code under test) delegates to _forward_impl; bind the
    # real implementation so this mock indexer can be dispatched through it.
    _forward_impl = QSAIndexer._forward_impl

    def __init__(self):
        self.selected = None
        self.logical_positions = None

    @staticmethod
    def project_qk(hidden_states, positions, **kwargs):
        rows = hidden_states.shape[0]
        return torch.zeros(rows, 4, 128), torch.zeros(rows, 1, 128), False

    def select_prefill_tokens(self, *args, **kwargs):
        self.selected = "prefill"
        return torch.tensor([1])

    def select_decode_tokens(self, *args, defer_expansion=False, **kwargs):
        self.selected = "decode"
        return torch.tensor([2])

    def update_key_state_and_compress(
        self, token_k, logical_positions, rope_positions, metadata, **kwargs
    ):
        self.logical_positions = logical_positions.clone()


class _DispatchMetadata:
    is_cuda_graph = False
    token_to_kv_pool = None
    out_cache_loc = None
    compress_member_rows = None
    decode_logical_positions = None
    pending_ring_slots = None
    # Consumed by the real _pending_ring_slots helper the dispatch indexer
    # borrows: one token row owned by request slot 1.
    token_to_batch_idx = torch.zeros(2, dtype=torch.int32)
    req_pool_indices = torch.ones(2, dtype=torch.int32)
    sequence_lengths = torch.ones(2, dtype=torch.int32)

    @staticmethod
    def get_decode_mqa_inputs(layer_id):
        return (
            torch.zeros(2, 1, 1, 128),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            1,
        )

    @staticmethod
    def get_prefill_mqa_inputs(layer_id, positions):
        return (
            torch.zeros(1, 1, 128),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
        )

    @staticmethod
    def get_token_to_batch_idx():
        return torch.zeros(1, dtype=torch.int32)

    @staticmethod
    def get_seqlens_int32():
        return torch.ones(1, dtype=torch.int32)

    @staticmethod
    def get_seqlens_expanded():
        return torch.tensor([7], dtype=torch.int32)


class _ForwardMode:
    def __init__(self, decode):
        self.decode = decode

    def is_decode(self):
        return self.decode


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_qsa_prefill_mqa_inputs_gather_all_compressed_keys_at_once(dtype):
    """One launch packs every request's complete compressed keys, byte-equal
    to the per-request index_select it replaces, without a host sync when the
    host lengths are known."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8-capable CUDA GPU required")
    torch.manual_seed(11)
    page, lengths = 64, [131, 3, 4096 + 7, 64, 257]
    max_len = max(lengths)
    pages_per_request = -(-max_len // page)
    pages = torch.randperm(len(lengths) * pages_per_request * 2)[
        : len(lengths) * pages_per_request
    ].reshape(len(lengths), pages_per_request, 1)
    # Page-aligned token slots, as the QSA allocator hands them out.
    token_slot_table = (
        (pages * page + torch.arange(page)).reshape(len(lengths), -1)[:, :max_len]
    ).to(device="cuda", dtype=torch.int32)
    capacity = int(token_slot_table.max()) // COMPRESS_RATIO + 1
    compressed = torch.randn(capacity, 1, 128, device="cuda").to(dtype)
    pool = SimpleNamespace(
        get_qsa_compressed_k_buffer=lambda layer_id: compressed,
        qsa_index_kv_heads=1,
        qsa_index_head_dim=128,
    )
    rows = torch.repeat_interleave(torch.arange(len(lengths)), 2).to(torch.int32)
    positions = torch.tensor(
        [p for n in lengths for p in (n - 2, n - 1)], device="cuda"
    )
    metadata = QSAIndexerMetadata(
        sequence_lengths=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        token_to_batch_idx=rows.cuda(),
        token_slot_table=token_slot_table,
        out_cache_loc=torch.empty(0, dtype=torch.int32, device="cuda"),
        token_to_kv_pool=pool,
        compress_ratio=COMPRESS_RATIO,
        block_topk=BLOCK_TOPK,
        prefill_sequence_lengths_cpu=tuple(lengths),
    )
    metadata.get_prefill_mqa_inputs(0, positions)  # compile outside the check
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        keys, starts, ends, _ = metadata.get_prefill_mqa_inputs(0, positions)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    expected = torch.cat(
        [
            compressed.index_select(
                0,
                token_slot_table[i, : (n // COMPRESS_RATIO) * COMPRESS_RATIO].long()[
                    ::COMPRESS_RATIO
                ]
                // COMPRESS_RATIO,
            )
            for i, n in enumerate(lengths)
        ]
    )
    assert keys.dtype == dtype
    assert torch.equal(keys.view(torch.uint8), expected.view(torch.uint8))
    blocks = [n // COMPRESS_RATIO for n in lengths]
    cu = [0, *accumulate(blocks)]
    assert starts.tolist() == [cu[i] for i in rows.tolist()]
    assert ends.tolist() == [
        cu[i] + min((p + 1) // COMPRESS_RATIO, blocks[i])
        for i, p in zip(rows.tolist(), positions.tolist())
    ]


def test_qsa_extend_metadata_carries_host_sequence_lengths():
    runner, pool, req_pool = _make_qsa_runner_and_pool()
    backend = QwenSparseAttnBackend(runner)
    forward_batch = SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=req_pool,
        req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([8, 6], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([8, 6], dtype=torch.int32),
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens=torch.tensor([4, 6], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([4, 0], dtype=torch.int32),
        positions=torch.tensor([4, 5, 6, 7, 0, 1, 2, 3, 4, 5]),
        mrope_positions=None,
        input_ids=torch.zeros(10, dtype=torch.int32),
        out_cache_loc=torch.arange(10, dtype=torch.int32),
        _original_forward_mode=None,
    )
    backend.init_forward_metadata(forward_batch)
    indexer = backend.forward_metadata.indexer_metadata
    assert indexer.prefill_sequence_lengths_cpu == (8, 6)


def test_qsa_row_ranges_do_not_cross_sequences():
    sequence_lengths = torch.tensor([10, 7], dtype=torch.int32)
    query_positions = torch.tensor([8, 9, 4, 6], dtype=torch.int32)
    query_sequence_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    starts, ends, compressed_cu = build_qsa_row_ranges(
        sequence_lengths, query_positions, query_sequence_ids, COMPRESS_RATIO
    )
    assert compressed_cu.tolist() == [0, 2, 3]
    assert starts.tolist() == [0, 0, 2, 2]
    assert ends.tolist() == [2, 2, 3, 3]


def test_qsa_weight_free_mqa_logits_matches_explicit_formula():
    torch.manual_seed(1)
    q = torch.randn(3, 4, 128, dtype=torch.bfloat16)
    k = torch.randn(5, 1, 128, dtype=torch.bfloat16)
    starts = torch.tensor([0, 1, 3], dtype=torch.int32)
    ends = torch.tensor([2, 4, 5], dtype=torch.int32)
    actual = qsa_mqa_prefill(q, k, starts, ends)
    broadcast_k = k.expand(-1, 4, -1)
    expected = torch.einsum("mhd,nhd->mnh", q.float(), broadcast_k.float())
    expected = torch.relu(expected).sum(-1) / (128**0.5)
    columns = torch.arange(5).unsqueeze(0)
    expected.masked_fill_(
        (columns < starts[:, None]) | (columns >= ends[:, None]), -float("inf")
    )
    torch.testing.assert_close(actual, expected)


def test_qsa_prefill_selection_microchunks_rows(monkeypatch):
    rows, keys, heads, head_dim = 65, 64, 4, 8
    token_topk, compress_ratio = 8, 4
    indexer = SimpleNamespace(
        token_topk=token_topk,
        compress_ratio=compress_ratio,
        block_topk=token_topk // compress_ratio,
    )
    torch.manual_seed(2)
    q = torch.randn(rows, heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn(keys, 1, head_dim, dtype=torch.bfloat16)
    starts = torch.zeros(rows, dtype=torch.int32)
    ends = torch.full((rows,), keys, dtype=torch.int32)
    positions = torch.full((rows,), keys * compress_ratio - 1, dtype=torch.long)
    sequence_lengths = torch.full((rows,), keys * compress_ratio, dtype=torch.int32)

    monkeypatch.setattr(
        qsa_indexer_module,
        "_QSA_PREFILL_LOGITS_BUDGET_BYTES",
        32 * keys * torch.float32.itemsize,
    )
    assert qsa_indexer_module._qsa_prefill_row_chunk_size(rows, keys, heads) == 32
    actual = QSAIndexer.select_prefill_tokens(
        indexer, q, k, starts, ends, positions, sequence_lengths
    )

    logits = qsa_mqa_prefill(q, k, starts, ends)
    blocks = qsa_fast_topk(logits, starts, ends, topk=indexer.block_topk)
    expected = expand_qsa_block_indices(
        blocks,
        positions,
        sequence_lengths,
        compress_ratio=compress_ratio,
        token_topk=token_topk,
    )
    torch.testing.assert_close(actual, expected)


def test_qsa_prefill_logical_positions_are_gathered_per_microchunk(monkeypatch):
    rows, keys, heads, head_dim = 65, 64, 4, 8
    indexer = SimpleNamespace(token_topk=8, compress_ratio=4, block_topk=2)
    q = torch.zeros(rows, heads, head_dim, dtype=torch.bfloat16)
    k = torch.zeros(keys, 1, head_dim, dtype=torch.bfloat16)
    starts = torch.zeros(rows, dtype=torch.int32)
    ends = torch.full((rows,), keys, dtype=torch.int32)
    positions = torch.full((rows,), keys * 4 - 1, dtype=torch.long)
    sequence_lengths = torch.full((rows,), keys * 4, dtype=torch.int32)
    logical_by_sequence = torch.stack((torch.arange(keys), torch.arange(keys) + 100))
    sequence_ids = torch.arange(rows).remainder(2)
    seen = []

    monkeypatch.setattr(
        qsa_indexer_module,
        "_QSA_PREFILL_LOGITS_BUDGET_BYTES",
        32 * keys * torch.float32.itemsize,
    )
    monkeypatch.setattr(
        qsa_indexer_module,
        "qsa_mqa_prefill",
        lambda q, *_args: torch.zeros(q.shape[0], keys),
    )
    monkeypatch.setattr(
        qsa_indexer_module,
        "qsa_fast_topk",
        lambda logits, *_args, **_kwargs: torch.zeros(
            logits.shape[0], 2, dtype=torch.int32
        ),
    )
    monkeypatch.setattr(
        qsa_indexer_module,
        "expand_qsa_block_indices",
        lambda blocks, *_args, **_kwargs: torch.zeros(
            blocks.shape[0], 11, dtype=torch.int32
        ),
    )

    def record_logical(
        _self, _logits, block_indices, _starts, logical, _static_buffers
    ):
        seen.append(logical.clone())
        return block_indices

    monkeypatch.setattr(QSAIndexer, "_maybe_merge_distributed_topk", record_logical)
    QSAIndexer.select_prefill_tokens(
        indexer,
        q,
        k,
        starts,
        ends,
        positions,
        sequence_lengths,
        logical_block_positions=logical_by_sequence,
        logical_block_sequence_ids=sequence_ids,
    )

    assert [chunk.shape[0] for chunk in seen] == [32, 32, 1]
    torch.testing.assert_close(torch.cat(seen), logical_by_sequence[sequence_ids])


def test_qsa_prefill_collective_schedule_is_rank_invariant():
    rows = 8192
    schedules = []
    for local_keys in (5590, 5600):
        chunk = qsa_indexer_module._qsa_prefill_row_chunk_size(
            rows,
            local_keys,
            heads=8,
            collective_keys=22400,
            collective_world_size=4,
        )
        schedules.append([min(chunk, rows), rows - min(chunk, rows)])
    assert schedules[0] == schedules[1]


def test_qsa_forward_cuda_dispatches_prefill_and_decode_mqa():
    indexer = _DispatchIndexer()
    metadata = _DispatchMetadata()
    inputs = torch.zeros(1, 16)
    positions = torch.zeros(1, dtype=torch.int32)

    prefill_result = QSAIndexer.forward_cuda(
        indexer,
        inputs,
        positions,
        SimpleNamespace(forward_mode=_ForwardMode(False)),
        metadata,
    )
    assert indexer.selected == "prefill"
    assert indexer.logical_positions.tolist() == [0]
    assert prefill_result.item() == 1

    positions.fill_(99)
    decode_result = QSAIndexer.forward_cuda(
        indexer,
        inputs,
        positions,
        SimpleNamespace(forward_mode=_ForwardMode(True)),
        metadata,
    )
    assert indexer.selected == "decode"
    assert indexer.logical_positions.tolist() == [6]
    assert decode_result.item() == 2


def test_qsa_block_expansion_adds_only_incomplete_tail():
    blocks = torch.full((2, BLOCK_TOPK), -1, dtype=torch.int32)
    blocks[0, 0] = 0
    blocks[1, :2] = torch.tensor([1, 0])
    result = expand_qsa_block_indices(
        blocks,
        query_positions=torch.tensor([5, 10]),
        sequence_lengths=torch.tensor([6, 11]),
        compress_ratio=COMPRESS_RATIO,
        token_topk=TOKEN_TOPK,
    )
    assert result.shape == (2, FINAL_TOPK)
    assert result[0, :6].tolist() == [0, 1, 2, 3, 4, 5]
    assert sorted(result[1, :11].tolist()) == list(range(11))
    assert torch.all(result[0, 6:] == -1)
    assert torch.all(result[1, 11:] == -1)


def test_qsa_triton_block_expansion_matches_torch_reference():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(9)
    # Regression coverage for the largest default CUDA graph batch.  The old
    # 2-D Triton expansion exceeded the per-program element limit at bs=512.
    rows = 512
    blocks = torch.full((rows, BLOCK_TOPK), -1, dtype=torch.int32)
    query_positions = torch.randint(0, 3000, (rows,), dtype=torch.int64)
    sequence_lengths = query_positions + 1
    complete_blocks = (query_positions + 1) // COMPRESS_RATIO
    valid_counts = torch.randint(0, BLOCK_TOPK + 1, (rows,))
    for row in range(rows):
        width = min(int(valid_counts[row]), int(complete_blocks[row]))
        if width:
            blocks[row, :width] = torch.randperm(int(complete_blocks[row]))[:width].to(
                torch.int32
            )

    expected = torch_expand_qsa_block_indices(
        blocks,
        query_positions,
        sequence_lengths,
        COMPRESS_RATIO,
        TOKEN_TOPK,
    )
    actual = triton_expand_qsa_block_indices(
        blocks.cuda(),
        query_positions.cuda(),
        sequence_lengths.cuda(),
        COMPRESS_RATIO,
        TOKEN_TOPK,
    ).cpu()
    assert torch.equal(actual, expected)


def test_qsa_decode_mqa_reads_paged_cache():
    torch.manual_seed(7)
    q = torch.randn(2, 4, 128, dtype=torch.bfloat16)
    cache = torch.randn(8, 64, 1, 128, dtype=torch.bfloat16)
    page_table = torch.tensor([[3, 1, 5], [4, 2, 0]], dtype=torch.int32)
    context_lens = torch.tensor([130, 67], dtype=torch.int32)
    actual = qsa_mqa_decode(
        q,
        cache,
        page_table,
        context_lens,
        max_model_len=192,
    )

    gathered = cache[page_table.long(), :, 0].reshape(2, 192, 128)
    expected = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    expected = torch.relu(expected).sum(-1) / (128**0.5)
    positions = torch.arange(192).unsqueeze(0)
    expected.masked_fill_(positions >= context_lens[:, None], -float("inf"))
    torch.testing.assert_close(actual, expected)


def test_qsa_mtp_step_out_cache_loc_matches_draft_forward_layout():
    """EagleDraftWorker.draft_forward gives each MTP draft step an out_cache_loc slice;
    the step's metadata must reference that slice, not the first batch_size slots.
    """
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseMultiStepDraftBackend,
    )

    backend = QwenSparseMultiStepDraftBackend.__new__(QwenSparseMultiStepDraftBackend)
    backend.topk, backend.speculative_num_steps = 1, 3
    bs, topk, steps = 4, 1, 3
    flat = torch.arange(bs * topk * steps, dtype=torch.int64)
    fb = SimpleNamespace(out_cache_loc=flat, batch_size=bs, seq_lens=torch.ones(bs))
    # Reference: the exact draft_forward expression chain.
    reference = flat.reshape(bs, topk, steps).permute(2, 0, 1).reshape(steps, -1)
    for step in range(steps):
        got = backend._step_out_cache_loc(fb, step)
        assert torch.equal(got, reference[step]), step
    # Steps must address disjoint slots -- the guarded regression is every
    # step landing on the first bs slots.
    step_slices = [backend._step_out_cache_loc(fb, i) for i in range(steps)]
    all_slots = torch.cat(step_slices)
    assert all_slots.unique().numel() == bs * topk * steps
    # Non-draft layouts pass through untouched instead of inventing slots.
    odd = SimpleNamespace(
        out_cache_loc=torch.arange(5), batch_size=bs, seq_lens=torch.ones(bs)
    )
    assert torch.equal(backend._step_out_cache_loc(odd, 1), odd.out_cache_loc)
    backend.speculative_num_steps = 1
    assert torch.equal(backend._step_out_cache_loc(fb, 0), flat)


@pytest.mark.parametrize("max_pages", [32, 129, 2048, 8193])
def test_qsa_graph_metadata_kernels_match_legacy_host_path(max_pages):
    """For decode rows and target-verify fan-out (boundary and non-boundary),
    replay kernels and the host refresh must build identical graph buffers."""
    from sglang.srt.layers.attention.qsa.graph_metadata import launch_graph_metadata

    device = "cuda"
    ratio, full_page = 4, 64

    class _Pool:
        qsa_compress_ratio = ratio
        # compressed slots per full-KV page (page 64 tokens / ratio 4)
        qsa_compressed_page_size = full_page // ratio
        qsa_compressed_sharding = None
        qsa_block_topk = 512

    def run_case(mode, bs, num_rows, seq_lens_list, extend_len, extend_lens=None):
        pool = _Pool()
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        # A page-aligned token table: request r's token i lives in full page
        # (r * max_pages + i // 64) at offset i % 64, mirroring the paged allocator.
        rows = torch.arange(8, dtype=torch.int32, device=device)[:, None]
        cols = torch.arange(max_pages * full_page, dtype=torch.int32, device=device)[
            None, :
        ]
        backend.req_to_token = (
            (rows * max_pages + cols // full_page) * full_page + cols % full_page
        ).contiguous()
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        req_pool_indices = torch.arange(bs, dtype=torch.int32, device=device)

        def make_metadata():
            indexer = QSAIndexerMetadata(
                sequence_lengths=torch.zeros(
                    num_rows, dtype=torch.int32, device=device
                ),
                token_to_batch_idx=torch.arange(
                    num_rows, dtype=torch.int32, device=device
                ),
                token_slot_table=torch.zeros(
                    (num_rows, 1), dtype=torch.int32, device=device
                ),
                out_cache_loc=torch.zeros(num_rows, dtype=torch.int64, device=device),
                token_to_kv_pool=pool,
                compress_ratio=ratio,
                block_topk=512,
                req_pool_indices=torch.zeros(
                    num_rows, dtype=torch.int32, device=device
                ),
                is_cuda_graph=True,
                graph_write_locs=torch.zeros(
                    num_rows, dtype=torch.int32, device=device
                ),
                write_owner_mask=torch.zeros(num_rows, dtype=torch.bool, device=device),
                graph_compressed_page_table=torch.zeros(
                    (num_rows, max_pages), dtype=torch.int32, device=device
                ),
                decode_block_positions=torch.full(
                    (num_rows, max_pages * (full_page // ratio)),
                    -1,
                    dtype=torch.int32,
                    device=device,
                ),
                graph_compressed_lengths=torch.zeros(
                    num_rows, dtype=torch.int32, device=device
                ),
                graph_prefix_lengths=torch.zeros(
                    num_rows, dtype=torch.int32, device=device
                ),
                decode_logical_positions=torch.zeros(
                    num_rows, dtype=torch.int32, device=device
                ),
                pending_ring_slots=torch.zeros(
                    num_rows, dtype=torch.int64, device=device
                ),
                graph_ring_group_locs=torch.zeros(
                    (num_rows, ratio), dtype=torch.int32, device=device
                ),
            )
            return qsa_backend_module.QwenSparseAttnMetadata(
                sequence_lengths=indexer.sequence_lengths,
                token_to_batch_idx=indexer.token_to_batch_idx,
                token_slot_table=indexer.token_slot_table,
                indexer_metadata=indexer,
                row_req_pool_indices=torch.zeros(
                    num_rows, dtype=torch.int32, device=device
                ),
                is_cuda_graph=True,
            )

        # Path 1: recorded kernels.
        kernel_metadata = make_metadata()
        launch_graph_metadata(
            mode=mode,
            bs=bs,
            num_rows=num_rows,
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            extend_lens=(
                None
                if extend_lens is None
                else torch.tensor(extend_lens, dtype=torch.int32, device=device)
            ),
            extend_len=extend_len,
            num_padding=0,
            metadata=kernel_metadata,
            req_to_token=backend.req_to_token,
            pool=pool,
        )
        # Path 2: legacy host refresh over the layout the kernels produced.
        host_metadata = make_metadata()
        host_metadata.sequence_lengths.copy_(kernel_metadata.sequence_lengths)
        host_metadata.row_req_pool_indices.copy_(kernel_metadata.row_req_pool_indices)
        backend._update_qsa_cuda_graph_metadata(
            host_metadata.indexer_metadata, host_metadata.row_req_pool_indices
        )
        for field in (
            "graph_write_locs",
            "graph_compressed_lengths",
            "decode_logical_positions",
            "pending_ring_slots",
            "graph_ring_group_locs",
        ):
            kernel_buf = getattr(kernel_metadata.indexer_metadata, field)
            host_buf = getattr(host_metadata.indexer_metadata, field)
            assert torch.equal(kernel_buf, host_buf), (mode, field)
        for row, compressed_len in enumerate(
            kernel_metadata.indexer_metadata.graph_compressed_lengths.tolist()
        ):
            visible_pages = (
                compressed_len + pool.qsa_compressed_page_size - 1
            ) // pool.qsa_compressed_page_size
            assert torch.equal(
                kernel_metadata.indexer_metadata.graph_compressed_page_table[
                    row, :visible_pages
                ],
                host_metadata.indexer_metadata.graph_compressed_page_table[
                    row, :visible_pages
                ],
            ), (mode, "graph_compressed_page_table", row)

    # Decode: lengths straddling boundaries (256 is a boundary, others not).
    run_case(mode=0, bs=4, num_rows=4, seq_lens_list=[255, 256, 257, 512], extend_len=0)
    # Target verify: 2 requests x 3 draft rows, one request crossing a boundary.
    run_case(mode=1, bs=2, num_rows=6, seq_lens_list=[254, 300], extend_len=3)
    # Draft extend: ragged per-request rows plus a dummy-tail row.
    run_case(
        mode=2,
        bs=2,
        num_rows=6,
        seq_lens_list=[256, 303],
        extend_len=0,
        extend_lens=[3, 2],
    )


def _qsa_expected_graph_layout(
    *, mode, bs, num_rows, seq_lens, req_pool, extend_lens, extend_len, num_padding
):
    """Independent host oracle for the graph row layout and padded dummy tail."""
    real_reqs = bs - num_padding
    row_lens, row_prefix, row_reqs = [], [], []
    for pid in range(bs):
        base = seq_lens[pid]
        if mode == 0:
            base = base if pid < real_reqs else 1
            row_lens.append(base)
            row_prefix.append(max(base - 1, 0))
            row_reqs.append(req_pool[pid] if pid < real_reqs else 0)
            continue
        eff = (
            (extend_len if pid < real_reqs else 0)
            if mode == 1
            else (extend_lens[pid] if pid < real_reqs else 0)
        )
        prefix, limit = (base, base + eff) if mode == 1 else (max(base - eff, 0), base)
        for j in range(eff):
            row_lens.append(min(prefix + 1 + j, limit))
            row_prefix.append(prefix)
            row_reqs.append(req_pool[pid])
    # Dummy tail: length 1, prefix 0, aliased to request slot 0 (never
    # allocated, so its pending-ring rows are the inert store dump).
    while len(row_lens) < num_rows:
        row_lens.append(1)
        row_prefix.append(0)
        row_reqs.append(0)
    return row_lens, row_prefix, row_reqs


@pytest.mark.parametrize("seq_offset", [0, 1, 3])
def test_qsa_graph_layout_covers_speculative_rows_and_padded_tail(seq_offset):
    """The layout kernel must rebuild speculative row fan-out and the padded dummy tail;
    the row-metadata kernel must derive compressed slots from those rows."""
    from sglang.srt.layers.attention.qsa.graph_metadata import launch_graph_metadata

    device = "cuda"
    ratio, full_page = 4, 64

    class _Pool:
        qsa_compress_ratio = ratio
        qsa_compressed_page_size = full_page // ratio
        qsa_compressed_sharding = None
        qsa_block_topk = 512

    def run_case(mode, bs, num_rows, seq_lens, extend_lens, extend_len, num_padding):
        pool = _Pool()
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        rows = torch.arange(8, dtype=torch.int32, device=device)[:, None]
        cols = torch.arange(4096, dtype=torch.int32, device=device)[None, :]
        backend.req_to_token = (
            (rows * 64 + cols // full_page) * full_page + cols % full_page
        ).contiguous()
        req_pool = list(range(bs))
        max_pages = 64
        indexer = QSAIndexerMetadata(
            sequence_lengths=torch.zeros(num_rows, dtype=torch.int32, device=device),
            token_to_batch_idx=torch.arange(num_rows, dtype=torch.int32, device=device),
            token_slot_table=torch.zeros(
                (num_rows, 1), dtype=torch.int32, device=device
            ),
            out_cache_loc=torch.zeros(num_rows, dtype=torch.int64, device=device),
            token_to_kv_pool=pool,
            compress_ratio=ratio,
            block_topk=512,
            req_pool_indices=torch.zeros(num_rows, dtype=torch.int32, device=device),
            is_cuda_graph=True,
            graph_write_locs=torch.zeros(num_rows, dtype=torch.int32, device=device),
            write_owner_mask=torch.zeros(num_rows, dtype=torch.bool, device=device),
            graph_compressed_page_table=torch.zeros(
                (num_rows, max_pages), dtype=torch.int32, device=device
            ),
            decode_block_positions=torch.full(
                (num_rows, max_pages * (full_page // ratio)),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            graph_compressed_lengths=torch.zeros(
                num_rows, dtype=torch.int32, device=device
            ),
            graph_prefix_lengths=torch.zeros(
                num_rows, dtype=torch.int32, device=device
            ),
            decode_logical_positions=torch.zeros(
                num_rows, dtype=torch.int32, device=device
            ),
            pending_ring_slots=torch.zeros(num_rows, dtype=torch.int64, device=device),
            graph_ring_group_locs=torch.zeros(
                (num_rows, ratio), dtype=torch.int32, device=device
            ),
        )
        metadata = qsa_backend_module.QwenSparseAttnMetadata(
            sequence_lengths=indexer.sequence_lengths,
            token_to_batch_idx=indexer.token_to_batch_idx,
            token_slot_table=indexer.token_slot_table,
            indexer_metadata=indexer,
            row_req_pool_indices=torch.zeros(
                num_rows, dtype=torch.int32, device=device
            ),
            is_cuda_graph=True,
        )
        launch_graph_metadata(
            mode=mode,
            bs=bs,
            num_rows=num_rows,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
            req_pool_indices=torch.tensor(req_pool, dtype=torch.int32, device=device),
            extend_lens=(
                None
                if extend_lens is None
                else torch.tensor(extend_lens, dtype=torch.int32, device=device)
            ),
            extend_len=extend_len,
            num_padding=num_padding,
            metadata=metadata,
            req_to_token=backend.req_to_token,
            pool=pool,
            seq_offset=seq_offset if mode == 0 else 0,
        )
        torch.cuda.synchronize()

        exp_lens, exp_prefix, exp_reqs = _qsa_expected_graph_layout(
            mode=mode,
            bs=bs,
            num_rows=num_rows,
            seq_lens=[x + seq_offset for x in seq_lens] if mode == 0 else seq_lens,
            req_pool=req_pool,
            extend_lens=extend_lens,
            extend_len=extend_len,
            num_padding=num_padding,
        )
        assert metadata.sequence_lengths.tolist() == exp_lens, (mode, "lengths")
        assert indexer.graph_prefix_lengths.tolist() == exp_prefix, (mode, "prefix")
        assert metadata.row_req_pool_indices.tolist() == exp_reqs, (mode, "req rows")

        r2t = backend.req_to_token.cpu()
        for row, (length, req) in enumerate(zip(exp_lens, exp_reqs)):
            assert int(indexer.graph_compressed_lengths[row]) == length // ratio
            if length > 0 and length % ratio == 0:
                expect = int(r2t[req, length - 1]) // ratio
            else:
                expect = 0  # non-boundary rows keep the inert reserved slot
            assert int(indexer.graph_write_locs[row]) == expect, (row, length)

    # Target verify: uniform 4-token window, 2 padded request slots.
    run_case(
        mode=1,
        bs=6,
        num_rows=32,
        seq_lens=[254, 255, 256, 300, 7, 7],
        extend_lens=None,
        extend_len=4,
        num_padding=2,
    )
    # Draft extend: per-request extend lengths (accept-count dependent), padded.
    run_case(
        mode=2,
        bs=5,
        num_rows=24,
        seq_lens=[260, 512, 257, 9, 9],
        extend_lens=[3, 1, 4, 0, 0],
        extend_len=0,
        num_padding=2,
    )
    run_case(
        mode=0,
        bs=4,
        num_rows=4,
        seq_lens=[255, 1024, 4096, 4096],
        extend_lens=None,
        extend_len=0,
        num_padding=2,
    )
    # Decode with a padded tail (dummy rows alias request slot 0).
    run_case(
        mode=0,
        bs=4,
        num_rows=4,
        seq_lens=[256, 1024, 1025, 4080],
        extend_lens=None,
        extend_len=0,
        num_padding=0,
    )


@pytest.mark.parametrize("bs", [1, 3, 128])
@pytest.mark.parametrize("padding", [0, 1])
def test_qsa_draft_metadata_multi_step_graph(bs, padding):
    from types import SimpleNamespace

    from sglang.srt.layers.attention.qsa.graph_metadata import (
        launch_draft_graph_metadata,
        launch_graph_metadata,
        prepare_draft_graph_metadata,
    )

    pages, ratio, full_page = 129, 4, 64
    pool = SimpleNamespace(qsa_compress_ratio=ratio, qsa_compressed_page_size=16)
    table = torch.arange(
        (bs + 2) * pages * full_page, dtype=torch.int32, device="cuda"
    ).reshape(bs + 2, -1)
    seq = torch.full((bs,), 8190, dtype=torch.int64, device="cuda")
    reqs = torch.arange(1, bs + 1, dtype=torch.int32, device="cuda")

    def metadata():
        indexer = SimpleNamespace(
            graph_compressed_lengths=torch.empty(bs, dtype=torch.int32, device="cuda"),
            graph_write_locs=torch.empty(bs, dtype=torch.int32, device="cuda"),
            write_owner_mask=torch.empty(bs, dtype=torch.bool, device="cuda"),
            graph_compressed_page_table=torch.empty(
                (bs, pages), dtype=torch.int32, device="cuda"
            ),
            decode_block_positions=torch.empty(
                (bs, pages * (full_page // ratio)),
                dtype=torch.int32,
                device="cuda",
            ),
            decode_logical_positions=torch.empty(bs, dtype=torch.int32, device="cuda"),
            pending_ring_slots=torch.empty(bs, dtype=torch.int64, device="cuda"),
            graph_ring_group_locs=torch.empty(
                (bs, ratio), dtype=torch.int32, device="cuda"
            ),
            graph_prefix_lengths=torch.empty(bs, dtype=torch.int32, device="cuda"),
            compress_ratio=ratio,
        )
        return SimpleNamespace(
            sequence_lengths=torch.empty(bs, dtype=torch.int32, device="cuda"),
            row_req_pool_indices=torch.empty(bs, dtype=torch.int32, device="cuda"),
            indexer_metadata=indexer,
        )

    outputs = [metadata() for _ in range(3)]
    references = [metadata() for _ in range(3)]
    args = prepare_draft_graph_metadata(outputs, table, pool)
    launch_draft_graph_metadata(args, seq, reqs, bs, padding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_draft_graph_metadata(args, seq, reqs, bs, padding)
    for replay in range(3):
        seq.add_(replay)
        reqs.copy_(reqs.roll(1))
        graph.replay()
        for step, (out, ref) in enumerate(zip(outputs, references)):
            launch_graph_metadata(
                mode=0,
                bs=bs,
                num_rows=bs,
                seq_lens=seq,
                req_pool_indices=reqs,
                extend_lens=None,
                extend_len=0,
                num_padding=padding,
                metadata=ref,
                req_to_token=table,
                pool=pool,
                seq_offset=step + 1,
            )
            torch.testing.assert_close(
                out.sequence_lengths, ref.sequence_lengths, rtol=0, atol=0
            )
            torch.testing.assert_close(
                out.row_req_pool_indices, ref.row_req_pool_indices, rtol=0, atol=0
            )
            for key, value in vars(out.indexer_metadata).items():
                if not isinstance(value, torch.Tensor):
                    continue
                reference = getattr(ref.indexer_metadata, key)
                if key == "graph_compressed_page_table":
                    for row, length in enumerate(
                        out.indexer_metadata.graph_compressed_lengths.tolist()
                    ):
                        visible_pages = (length + 15) // 16
                        torch.testing.assert_close(
                            value[row, :visible_pages],
                            reference[row, :visible_pages],
                            rtol=0,
                            atol=0,
                        )
                elif key == "decode_block_positions":
                    for row, length in enumerate(
                        out.indexer_metadata.graph_compressed_lengths.tolist()
                    ):
                        torch.testing.assert_close(
                            value[row, :length],
                            reference[row, :length],
                            rtol=0,
                            atol=0,
                        )
                else:
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "dtype,output_dtype",
    [
        (torch.bfloat16, None),
        (torch.float8_e4m3fn, None),
        (torch.float8_e4m3fn, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("head_dim,num_kv_heads,group_size", [(128, 2, 4), (256, 1, 6)])
@pytest.mark.parametrize(
    "query_lens,prefix_lens",
    [
        ([1, 7], [31, 19]),
        ([17, 33], [23, 0]),
        ([2049, 3], [7, 40]),
        ([3, 1, 2], [1018, 128, 128]),
        ([3, 1, 2], [253, 63, 62]),
        ([3, 1, 2], [254, 63, 62]),
        ([3, 1, 2], [8189, 2047, 2046]),
        ([3, 1, 2], [8190, 2047, 2046]),
    ],
)
def test_qsa_prefill_kv_packing(
    dtype,
    output_dtype,
    head_dim,
    num_kv_heads,
    group_size,
    query_lens,
    prefix_lens,
    strided,
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8-capable CUDA GPU required")
    torch.manual_seed(42)
    device = "cuda"
    lens = [q + p for q, p in zip(query_lens, prefix_lens)]
    request_ids = [3, 1, 0][: len(lens)]
    replay_ids = [2, 0, 3][: len(lens)]
    table_width = max(lens) + 11
    pool_size = 16 * table_width
    table = torch.randperm(pool_size, device=device).reshape(8, table_width * 2)[
        ::2, ::2
    ]
    if strided:
        table = table.to(torch.int32)
    req_indices = torch.tensor(request_ids, device=device, dtype=torch.int64)
    k = torch.randn(
        pool_size, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    ).to(dtype)
    v = torch.randn_like(k.to(torch.bfloat16)).to(dtype)
    if strided:
        k_storage = torch.empty(
            pool_size, num_kv_heads, head_dim * 2, device=device, dtype=dtype
        )
        v_storage = torch.empty_like(k_storage)
        k = k_storage[..., ::2].copy_(k)
        v = v_storage[..., ::2].copy_(v)
    q = torch.randn(
        sum(query_lens),
        num_kv_heads * group_size,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    cu_q = torch.tensor([0, *accumulate(query_lens)], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, *accumulate(lens)], device=device, dtype=torch.int32)
    kv_lens = torch.tensor(lens, device=device, dtype=torch.int32)
    indices = torch.full((sum(query_lens), 32), -1, device=device, dtype=torch.int32)
    row = 0
    for count, prefix in zip(query_lens, prefix_lens):
        for offset in range(count):
            visible = prefix + offset + 1
            selected = torch.randperm(visible, device=device)[:32]
            indices[row, : selected.numel()] = selected
            row += 1

    def reference(request_ids):
        slots = torch.cat([table[request_ids[i], :n] for i, n in enumerate(lens)])
        return sparse_gqa_fwd_interface_triton_ck(
            q,
            k.index_select(0, slots).to(output_dtype or k.dtype),
            v.index_select(0, slots).to(output_dtype or v.dtype),
            indices,
            cu_q,
            cu_k,
            kv_lens,
            head_dim**-0.5,
        )

    def packed():
        packed_k, packed_v = pack_qsa_prefill_kv(
            k=k,
            v=v,
            req_to_token=table,
            req_indices=req_indices,
            cu_k=cu_k,
            total_k=sum(lens),
            max_k=max(lens),
            output_dtype=output_dtype,
        )
        if not torch.cuda.is_current_stream_capturing():
            slots = torch.cat([table[req_indices[i], :n] for i, n in enumerate(lens)])
            torch.testing.assert_close(
                packed_k.float(), k.index_select(0, slots).float(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                packed_v.float(), v.index_select(0, slots).float(), rtol=0, atol=0
            )
        return packed_k, packed_v

    def attend(packed_k, packed_v):
        return sparse_gqa_fwd_interface_triton_ck(
            q,
            packed_k,
            packed_v,
            indices,
            cu_q,
            cu_k,
            kv_lens,
            head_dim**-0.5,
        )

    expected = reference(request_ids)
    actual = attend(*packed())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = packed()
    req_indices.copy_(torch.tensor(replay_ids, device=device))
    expected = reference(replay_ids)
    graph.replay()
    torch.testing.assert_close(attend(*captured), expected, rtol=0, atol=0)


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("heads", [1, 3, 6, 12])
def test_qsa_large_fp8_prefill_float_reference(head_dim, heads):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8-capable CUDA GPU required")
    torch.manual_seed(19)
    rows, prefix, topk = 2053, 64, 64
    q = torch.randn(rows, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(rows + prefix, 1, head_dim, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    v = torch.randn_like(k.to(torch.bfloat16)).to(k.dtype)
    visible = torch.arange(rows, device="cuda") + prefix + 1
    indices = (
        torch.arange(topk, device="cuda")[None, :]
        + torch.arange(rows, device="cuda")[:, None] * 7
    ) % visible[:, None]
    indices = indices.to(torch.int32)
    cu_q = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, rows + prefix], device="cuda", dtype=torch.int32)
    lengths = cu_k[1:]
    actual = sparse_gqa_fwd_interface_triton_ck(
        q, k, v, indices, cu_q, cu_k, lengths, head_dim**-0.5
    )
    for row in [0, 17, rows // 2, rows - 1]:
        keys = k.index_select(0, indices[row].long()).squeeze(1).double()
        values = v.index_select(0, indices[row].long()).squeeze(1).double()
        expected = (
            torch.softmax(q[row].double() @ keys.T * head_dim**-0.5, dim=-1) @ values
        )
        torch.testing.assert_close(actual[row].double(), expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
