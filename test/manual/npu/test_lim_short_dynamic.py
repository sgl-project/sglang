"""Real-manager dense/sparse mixed attention, replay and residency lifecycle."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from sgl_kernel_npu.sparsity_driven_kv_offload import free_shm

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_attention import (
    forward_sparsity_driven_kv_offload,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget import (
    resolve_sparse_kv_budget,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager import (
    SparseKVCacheManager,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_parallel


@pytest.fixture(autouse=True)
def _bounded_cpu_oracle():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class _IndexPool(MLATokenToKVPool):
    def __init__(self, key):
        self.key = key
        self.kv_lora_rank, self.qk_rope_head_dim = 512, 64
        self.store_dtype = torch.bfloat16
        self.start_layer, self.layer_num, self.page_size = 0, 2, 128
        self.size = key.shape[0] * 128 - self.page_size

    def get_index_k_buffer(self, layer):
        return self.key


def _reference(query, rope, payload, ids, scale):
    kv = payload[ids].float()
    scores = (query.float() @ kv[:, :512].T + rope.float() @ kv[:, 512:].T) * scale
    return scores.softmax(-1) @ kv[:, :512]


def test_prefill_attention_and_short_residency():
    torch.npu.set_device(0)
    torch.manual_seed(8117)
    context, capacity, heads = 8448, 8192, 4
    key = torch.zeros(
        2 * context // 128, 128, 1, 128, device="npu", dtype=torch.bfloat16
    )
    index_pool = _IndexPool(key)
    pool = ReqToTokenPool(2, context, "npu", False)
    pages = torch.randperm(2 * context // 128).view(2, -1)
    mapping = (pages[..., None] * 128 + torch.arange(128)).view(2, context)
    pool.req_to_token[1:3] = mapping.to("npu")
    budget = resolve_sparse_kv_budget(
        model_config=SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            hf_config=SimpleNamespace(
                architectures=["GlmMoeDsaForCausalLM"],
                index_head_dim=128,
                index_n_heads=32,
                index_topk=2048,
            ),
        ),
        page_size=128,
        request_rows=3,
        graph_batch_capacity=3,
        context_len=context,
        num_layers=2,
        top_k=2048,
        kv_bytes_per_token=1152,
        cache_tokens=capacity,
        copy_sfa_query_heads=heads,
    )
    index_pool.sparse_kv_budget_plan = budget
    manager = None
    with get_parallel().override(attn_cp_size=1):
        try:
            manager = SparseKVCacheManager(
                pool,
                SimpleNamespace(get_kvcache=lambda: index_pool),
                2048,
                copy_sfa_padded_query_heads=heads,
                paged_kv_cache=index_pool,
            )
            lengths, positions = [129, 65, 0], [127, 128, 63, 64]
            physical = torch.cat((mapping[0, 127:129], mapping[1, 63:65]))
            inputs = SimpleNamespace(
                batch_size=3,
                forward_mode=ForwardMode.EXTEND,
                req_pool_indices=torch.tensor(
                    [1, 2, 0], device="npu", dtype=torch.int32
                ),
                req_pool_indices_cpu=torch.tensor([1, 2, 0]),
                seq_lens=torch.tensor(lengths, device="npu", dtype=torch.int32),
                seq_lens_cpu=torch.tensor(lengths),
                extend_seq_lens=torch.tensor(
                    [2, 2, 0], device="npu", dtype=torch.int32
                ),
                extend_prefix_lens=torch.tensor(
                    [127, 63, 0], device="npu", dtype=torch.int32
                ),
                extend_seq_lens_cpu=[2, 2, 0],
                out_cache_loc=physical.to("npu"),
                attn_cp_metadata=None,
            )
            backend = SimpleNamespace(
                device="npu",
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                sparse_kv_manager=manager,
                forward_metadata=SimpleNamespace(
                    actual_seq_lengths_q=None, actual_seq_lengths_kv=None
                ),
            )
            query = torch.randn(4, heads, 512, device="npu", dtype=torch.bfloat16)
            rope = torch.randn(4, heads, 64, device="npu", dtype=torch.bfloat16)
            sources = torch.arange(2048, device="npu", dtype=torch.int32).expand(4, -1)
            sources = sources.masked_fill(
                sources > torch.tensor(positions, device="npu")[:, None], -1
            )
            for layer_id, host in enumerate(manager.host_kv_buffer):
                payload = torch.randn(2 * context, 576, dtype=torch.bfloat16)
                fresh = torch.randn(4, 1, 576, device="npu", dtype=torch.bfloat16)
                host.view(-1, 576).copy_(payload)
                payload[physical] = fresh.cpu().view(4, 576)
                layer = SimpleNamespace(
                    layer_id=layer_id,
                    tp_k_head_num=1,
                    tp_q_head_num=heads,
                    scaling=576**-0.5,
                )
                output = forward_sparsity_driven_kv_offload(
                    backend,
                    query,
                    fresh[..., :512].contiguous(),
                    None,
                    layer,
                    inputs,
                    q_rope=rope,
                    k_rope=fresh[..., 512:].contiguous(),
                    topk_indices=sources,
                )
                torch.npu.synchronize()
                for token, pos in enumerate(positions):
                    row = token // 2
                    expected = _reference(
                        query[token].cpu(),
                        rope[token].cpu(),
                        payload,
                        mapping[row, : pos + 1],
                        layer.scaling,
                    )
                    torch.testing.assert_close(
                        output[token].cpu().float(), expected, rtol=0.025, atol=0.025
                    )
                for row, length in enumerate(lengths[:2]):
                    torch.testing.assert_close(
                        manager.device_kv_buffer[layer_id][row + 1, :length, 0].cpu(),
                        payload[mapping[row, :length]],
                        rtol=0,
                        atol=0,
                    )
        finally:
            if manager is not None:
                torch.npu.synchronize()
                free_shm(torch.npu.current_device())


@pytest.mark.parametrize(
    "width,capacity",
    [(1, 8192), (2, 8192), (3, 8192), (4, 8192), (5, 12288), (8, 16384), (14, 28672)],
)
@pytest.mark.parametrize("graph", [False, True])
def test_short_sparse_graph_lifecycle(
    width, capacity, graph, record_property, measure=False
):
    torch.npu.set_device(0)
    torch.manual_seed(1729 + width)
    batch, heads, context = 3, 4, capacity + 1024
    pages = context // 128
    key = torch.randn(2 * pages, 128, 1, 128, dtype=torch.bfloat16, device="npu")
    table = torch.cat(
        (torch.arange(2 * pages).view(2, pages), torch.zeros(1, pages)), dim=0
    ).to(device="npu", dtype=torch.int32)
    pool = ReqToTokenPool(2, context, "npu", False)
    mapping = torch.arange(2 * context).view(2, context)
    pool.req_to_token[1, :context] = mapping[0].to("npu")
    pool.req_to_token[2, :context] = mapping[1].to("npu")
    index_pool = _IndexPool(key)
    budget = resolve_sparse_kv_budget(
        model_config=SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            hf_config=SimpleNamespace(
                architectures=["GlmMoeDsaForCausalLM"],
                index_head_dim=128,
                index_n_heads=32,
                index_topk=2048,
            ),
        ),
        page_size=128,
        request_rows=3,
        graph_batch_capacity=3,
        context_len=context,
        num_layers=2,
        top_k=2048,
        kv_bytes_per_token=1152,
        cache_tokens=capacity,
        copy_sfa_query_heads=heads,
        max_query_width=width,
    )
    index_pool.sparse_kv_budget_plan = budget
    manager = None
    with get_parallel().override(attn_cp_size=1):
        try:
            manager = SparseKVCacheManager(
                pool,
                SimpleNamespace(get_kvcache=lambda: index_pool),
                2048,
                copy_sfa_padded_query_heads=heads,
                paged_kv_cache=index_pool,
            )
            allocations = [
                tensor
                for owner in (manager, manager.lim_adapter)
                for value in vars(owner).values()
                for tensor in (value if isinstance(value, (list, tuple)) else [value])
                if isinstance(tensor, torch.Tensor)
                and tensor.device.type == "npu"
                and tensor is not manager.lim_adapter.key_scale
            ]
            assert (
                sum(t.numel() * t.element_size() for t in allocations)
                == budget.managed_hbm_bytes
            )
            payloads = [
                torch.randn(2 * context, 576, dtype=torch.bfloat16) for _ in range(2)
            ]
            for host, payload in zip(manager.host_kv_buffer, payloads):
                host.view(-1, 576).copy_(payload)
            tokens = batch * width + 3  # TP suffix is separate from request padding.
            query = torch.randn(tokens, heads, 512, dtype=torch.bfloat16, device="npu")
            rope = torch.randn(tokens, heads, 64, dtype=torch.bfloat16, device="npu")
            iq = torch.randn(tokens, 32, 128, dtype=torch.bfloat16, device="npu")
            weights = torch.randn(tokens, 32, dtype=torch.bfloat16, device="npu")
            fresh = [
                torch.randn(tokens, 1, 576, dtype=torch.bfloat16, device="npu")
                for _ in range(2)
            ]
            inputs = SimpleNamespace(
                batch_size=batch,
                forward_mode=ForwardMode.DECODE
                if width == 1
                else ForwardMode.TARGET_VERIFY,
                req_pool_indices=torch.tensor(
                    [1, 2, 0], dtype=torch.int32, device="npu"
                ),
                seq_lens=torch.empty(batch, dtype=torch.int32, device="npu"),
                positions=torch.empty(tokens, dtype=torch.int64, device="npu"),
                out_cache_loc=torch.empty(tokens, dtype=torch.int64, device="npu"),
                spec_info=None
                if width == 1
                else SimpleNamespace(num_tokens_per_req=width),
            )
            layers = [
                SimpleNamespace(
                    layer_id=i, tp_k_head_num=1, tp_q_head_num=heads, scaling=576**-0.5
                )
                for i in range(2)
            ]
            backend = SimpleNamespace(
                device="npu",
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                sparse_kv_manager=manager,
                forward_metadata=SimpleNamespace(sparse_kv_lim_plan=None),
            )
            lengths = [max(129, width + 1), capacity + 257, 0]
            max_abs = 0.0

            def set_inputs(short_length):
                lengths[0] = short_length
                pos = [
                    p
                    for length in lengths
                    for p in (range(length - width, length) if length else [0] * width)
                ]
                loc = [
                    int(mapping[r, p]) if lengths[r] else -1
                    for r in range(batch)
                    for p in pos[r * width : (r + 1) * width]
                ]
                pos.extend([0] * 3)
                loc.extend([-1] * 3)
                inputs.positions.copy_(torch.tensor(pos, device="npu"))
                inputs.out_cache_loc.copy_(torch.tensor(loc, device="npu"))
                inputs.seq_lens.copy_(
                    torch.tensor(
                        [n if width == 1 else max(0, n - width) for n in lengths],
                        device="npu",
                    )
                )
                for layer_id, value in enumerate(fresh):
                    cpu = value.cpu().view(tokens, 576)
                    for r in range(2):
                        payloads[layer_id][
                            mapping[r, lengths[r] - width : lengths[r]]
                        ] = cpu[r * width : (r + 1) * width]

            def run():
                plan = manager.lim_adapter.select(
                    manager,
                    query=iq,
                    weights=weights,
                    key=key,
                    block_table=table,
                    forward_batch=inputs,
                    layer_id=0,
                )
                backend.forward_metadata.sparse_kv_lim_plan = plan
                outputs = []
                for layer, value in zip(layers, fresh):
                    outputs.append(
                        forward_sparsity_driven_kv_offload(
                            backend,
                            query,
                            value[..., :512].contiguous(),
                            None,
                            layer,
                            inputs,
                            q_rope=rope,
                            k_rope=value[..., 512:].contiguous(),
                            topk_indices=plan.source_ids,
                        ).clone()
                    )
                return plan, outputs

            def check(plan, outputs):
                nonlocal max_abs
                torch.npu.synchronize()
                source = plan.source_ids.cpu()
                q, qr = query.cpu(), rope.cpu()
                for layer_id, output in enumerate(outputs):
                    actual = output.cpu().view(tokens, heads, 512)
                    for r in range(2):
                        for col in range(width):
                            t = r * width + col
                            visible = lengths[r] - width + col + 1
                            ids = (
                                torch.arange(visible)
                                if lengths[r] < capacity
                                else source[t].long()
                            )
                            ids = ids[(ids >= 0) & (ids < visible)]
                            expected = _reference(
                                q[t],
                                qr[t],
                                payloads[layer_id],
                                mapping[r, ids],
                                layers[layer_id].scaling,
                            )
                            torch.testing.assert_close(
                                actual[t].float(), expected, rtol=0.025, atol=0.025
                            )
                            max_abs = max(
                                max_abs,
                                (actual[t].float() - expected).abs().max().item(),
                            )
                    assert torch.count_nonzero(actual[2 * width :]) == 0
                assert inputs.sparse_kv_decode_path == "lim_copy_sfa"
                assert manager.planner_num_cache_tokens[:batch].cpu().tolist() == [
                    capacity if lengths[0] >= capacity else 0,
                    capacity,
                    0,
                ]

            set_inputs(lengths[0])
            plan, outputs = run()
            check(plan, outputs)
            if graph:
                # Serving captures before real requests arrive. Replaying a
                # graph captured with only padding must admit a cold mixed batch.
                inputs.req_pool_indices.zero_()
                inputs.seq_lens.zero_()
                inputs.out_cache_loc.fill_(-1)
                captured = torch.npu.NPUGraph()
                with torch.npu.graph(captured):
                    plan, outputs = run()
                manager.reset_requests([1, 2])
                inputs.req_pool_indices.copy_(torch.tensor([1, 2, 0], device="npu"))
                set_inputs(lengths[0])
                captured.replay()
                check(plan, outputs)
            for short_length in (
                1 if width == 1 else width + 1,
                127,
                128,
                129,
                2047,
                2048,
                2049,
                capacity - 1,
                capacity + 1,
                capacity - width - 1,
                capacity - width,
            ):
                set_inputs(short_length)
                if graph:
                    captured.replay()
                else:
                    plan, outputs = run()
                check(plan, outputs)
                # Repeating the same step rewrites current candidates but must
                # not recopy an already resident dense prefix from host.
                if graph:
                    captured.replay()
                else:
                    plan, outputs = run()
                check(plan, outputs)
            # Ragged prefill publishes host KV and invalidates resident extents.
            # The following decode is responsible for restoring its prefix.
            manager.reset_requests([1, 2])
            chunk_counts = [65, 128, 0]
            chunk_prefix = [64, 129, 0]
            chunk_tokens = sum(chunk_counts)
            chunk_values = [
                torch.randn(chunk_tokens, 1, 576, device="npu", dtype=torch.bfloat16)
                for _ in layers
            ]
            prefill = SimpleNamespace(
                batch_size=batch,
                forward_mode=ForwardMode.EXTEND,
                req_pool_indices=inputs.req_pool_indices,
                seq_lens=torch.empty(batch, device="npu", dtype=torch.int32),
                extend_seq_lens=torch.tensor(
                    chunk_counts, device="npu", dtype=torch.int32
                ),
                extend_prefix_lens=torch.empty(batch, device="npu", dtype=torch.int32),
                extend_seq_lens_cpu=chunk_counts,
                out_cache_loc=torch.empty(
                    chunk_tokens, device="npu", dtype=torch.int64
                ),
            )

            def set_chunk(prefixes):
                prefill.extend_prefix_lens.copy_(torch.tensor(prefixes, device="npu"))
                ends = [p + n for p, n in zip(prefixes, chunk_counts)]
                prefill.seq_lens.copy_(torch.tensor(ends, device="npu"))
                physical = torch.cat(
                    [
                        mapping[r, p : p + n]
                        for r, (p, n) in enumerate(zip(prefixes, chunk_counts))
                        if n
                    ]
                )
                prefill.out_cache_loc.copy_(physical.to("npu"))
                for layer_id, values in enumerate(chunk_values):
                    payloads[layer_id][physical] = values.cpu().view(chunk_tokens, 576)
                return ends

            def publish_chunk():
                for layer, values in zip(layers, chunk_values):
                    manager.offload_v2(
                        values[..., :512].contiguous(),
                        values[..., 512:].contiguous(),
                        layer,
                        prefill,
                        None,
                    )

            def check_chunk(ends):
                torch.npu.synchronize()
                for layer_id, host in enumerate(manager.host_kv_buffer):
                    for r in range(2):
                        torch.testing.assert_close(
                            host.view(-1, 576)[mapping[r, : ends[r]]],
                            payloads[layer_id][mapping[r, : ends[r]]],
                            rtol=0,
                            atol=0,
                        )
                    assert manager.resident_begin[layer_id, 1:3].cpu().tolist() == [
                        -1,
                        -1,
                    ]
                    assert manager.resident_end[layer_id, 1:3].cpu().tolist() == [
                        -1,
                        -1,
                    ]

            ends = set_chunk(chunk_prefix)
            publish_chunk()
            check_chunk(ends)
            if graph:
                prefill_graph = torch.npu.NPUGraph()
                with torch.npu.graph(prefill_graph):
                    publish_chunk()
            ends = set_chunk(ends)
            if graph:
                prefill_graph.replay()
            else:
                publish_chunk()
            check_chunk(ends)

            # Refill the neighbor after the independent prefill reset, then
            # verify that reusing only request 1 preserves request 2's plan.
            set_inputs(2049)
            if graph:
                captured.replay()
            else:
                plan, outputs = run()
            check(plan, outputs)

            # A radix hit attaches a physically shared prefix to a reused row.
            manager.reset_requests([1])
            mapping[0, :128] = mapping[1, :128]
            pool.req_to_token[1, :128] = mapping[0, :128].to("npu")
            table[0, 0] = table[1, 0]
            set_inputs(2049)
            if graph:
                captured.replay()
            else:
                plan, outputs = run()
            check(plan, outputs)
            assert int(plan.miss_counts[1].cpu()) < capacity
            if measure:
                import statistics

                step = captured.replay if graph else run
                for _ in range(5):
                    step()
                samples = []
                for _ in range(5):
                    begin, end = (
                        torch.npu.Event(enable_timing=True),
                        torch.npu.Event(enable_timing=True),
                    )
                    begin.record()
                    for _ in range(20):
                        step()
                    end.record()
                    end.synchronize()
                    samples.append(begin.elapsed_time(end) * 1000 / 20)
                record_property(
                    "mixed_two_layer_manager_us", statistics.median(samples)
                )
                record_property(
                    "timing_scope",
                    "one LIM owner, two KV publications/residency stages and CopySFA consumers, output clones; fixed resident inputs",
                )
            record_property("max_abs_vs_fp32", max_abs)
            record_property("query_width", width)
            record_property("cache_capacity", capacity)
            record_property("graph", graph)
        finally:
            if manager is not None:
                torch.npu.synchronize()
                free_shm(torch.npu.current_device())
