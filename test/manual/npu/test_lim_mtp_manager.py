"""Real-manager chain-MTP LIM/CopySFA integration on one Ascend device."""

import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torch_npu")

from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_attention import (
    forward_fused_sparse_kv_offload,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget import (
    resolve_sparse_kv_budget,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager import (
    FusedSparseKVCacheManager,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel

K = 2048


@pytest.fixture(autouse=True)
def _single_rank_attention_group():
    group = SimpleNamespace(
        rank_in_group=0,
        world_size=1,
        broadcast_object=lambda obj: obj,
        barrier=lambda: None,
        broadcast=lambda tensor, src: tensor,
    )
    with get_parallel().override(attn_tp_group=group):
        yield


class _IndexPool(MLATokenToKVPool):
    """Isolate index-cache allocation while retaining the real manager path."""

    def __init__(self, key, num_layers):
        self.key = key
        self.kv_lora_rank, self.qk_rope_head_dim = 512, 64
        self.store_dtype = torch.bfloat16
        self.start_layer, self.layer_num = 0, num_layers
        self.page_size = 128
        self.size = key.shape[0] * 128 - self.page_size

    def get_index_k_buffer(self, layer_id):
        return self.key


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize(
    "query_width,capacity",
    [
        (2, 4096),
        (3, 6144),
        (4, 11392),
        (4, 12288),
        (4, 16384),
        (4, 16768),
        (5, 12288),
        (8, 16384),
        (14, 28672),
    ],
)
def test_real_manager_mtp_owner_share_lifecycle(graph, heads, query_width, capacity):
    _run_manager_lifecycle(graph, heads, query_width, capacity)


@pytest.mark.parametrize("graph", [False, True])
def test_real_manager_mtp_compact_pd_lifecycle(graph):
    _run_manager_lifecycle(graph, 4, 2, 4096, compact_pd=True)


@pytest.mark.parametrize("graph", [False, True])
def test_real_manager_mtp_multiple_owners(graph):
    _run_manager_lifecycle(graph, 4, 4, 12288, num_layers=4)


def _run_manager_lifecycle(
    graph, heads, query_width, capacity, compact_pd=False, num_layers=2
):
    if not torch.npu.is_available():
        pytest.skip("NPU is required")
    device_id = torch.npu.current_device()
    device = torch.device("npu", device_id)
    C = capacity
    Q = query_width
    # Keep the rollback inside the sparse branch for both capacities. Falling
    # below C correctly produces zero resident-cache misses instead.
    L = max(16389, C + 261)
    CONTEXT = (L + 127) // 128 * 128
    batch = 3
    keys = torch.zeros(batch, CONTEXT, 128, dtype=torch.bfloat16, device=device)
    keys[:, :, 0] = -8
    keys[0, :K, 0] = 4
    keys[1, K : 2 * K, 0] = 4
    keys[:, L - Q : L, 0] = 16
    keys = keys.view(-1, 128, 1, 128)
    table = torch.arange(batch * CONTEXT // 128, dtype=torch.int32, device=device).view(
        batch, -1
    )
    # Graph metadata may reserve an extra speculative page outside host context.
    table = torch.nn.functional.pad(table, (0, 1))
    token_rows = batch * Q + 4  # Include a TP padding suffix.
    query = torch.zeros(token_rows, 32, 128, dtype=torch.bfloat16, device=device)
    query[:, :, 0] = 1
    weights = torch.full((token_rows, 32), 1 / 32, dtype=query.dtype, device=device)
    if compact_pd:
        pool = DecodeReqToTokenPool(2, CONTEXT, "npu", False, pre_alloc_size=4)
        request_ids = [5, 4, 0]
    else:
        pool = ReqToTokenPool(2, CONTEXT, "npu", False)
        request_ids = [1, 2, 0]
    pool.req_to_token[request_ids[0], :CONTEXT] = torch.arange(CONTEXT, device=device)
    pool.req_to_token[request_ids[1], :CONTEXT] = torch.arange(
        CONTEXT, 2 * CONTEXT, device=device
    )
    if compact_pd:
        pool.req_to_token[6, :CONTEXT] = pool.req_to_token[5, :CONTEXT]
    index_pool = _IndexPool(keys, num_layers)
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
        request_rows=7 if compact_pd else 3,
        resident_rows=3,
        graph_batch_capacity=3,
        context_len=CONTEXT,
        cache_tokens=C,
        num_layers=num_layers,
        top_k=K,
        kv_bytes_per_token=1152,
        copy_sfa_query_heads=heads,
        max_query_width=Q,
    )
    budget.bind_index_k_pool(index_pool)
    index_pool.sparse_kv_budget_plan = budget
    inputs = SimpleNamespace(
        batch_size=batch,
        forward_mode=ForwardMode.TARGET_VERIFY,
        req_pool_indices=torch.tensor(request_ids, dtype=torch.int32, device=device),
        seq_lens=torch.tensor([L - Q, L - Q, 0], dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor([L - Q, L - Q, 0], dtype=torch.int32),
        positions=torch.tensor(
            list(range(L - Q, L)) * 2 + [0] * (token_rows - 2 * Q),
            dtype=torch.int64,
            device=device,
        ),
        out_cache_loc=torch.tensor(
            list(range(L - Q, L))
            + list(range(CONTEXT + L - Q, CONTEXT + L))
            + [-1] * (token_rows - 2 * Q),
            dtype=torch.int32,
            device=device,
        ),
        spec_info=SimpleNamespace(num_tokens_per_req=Q),
    )
    layer = [
        SimpleNamespace(
            layer_id=i, tp_k_head_num=1, tp_q_head_num=heads, scaling=576**-0.5
        )
        for i in range(num_layers)
    ]
    generator = torch.Generator().manual_seed(123)
    q = torch.randn(token_rows, heads, 512, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    rope = torch.randn(token_rows, heads, 64, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    fresh = [
        torch.randn(token_rows, 1, 576, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        for _ in layer
    ]
    spec = SimpleNamespace(speculative_eagle_topk=1, speculative_num_draft_tokens=Q)
    manager = None
    try:
        with ExitStack() as stack:
            stack.enter_context(
                get_context().override_server_args(disaggregation_mode="null")
            )
            stack.enter_context(
                patch("sglang.srt.runtime_context.get_spec", return_value=spec)
            )
            manager = FusedSparseKVCacheManager(
                pool,
                K,
                index_k_pool=index_pool,
            )
            if compact_pd:
                pool.resident_row_lifecycle.admit([4, 5])
                assert manager.resident_pool.request_to_resident.cpu().tolist() == [
                    0,
                    0,
                    0,
                    0,
                    1,
                    2,
                    0,
                ]
            for i in range(num_layers):
                for component in (manager.host_k_buffer[i], manager.host_v_buffer[i]):
                    component.copy_(
                        torch.randn(component.shape, generator=generator).to(
                            torch.bfloat16
                        )
                    )
            backend = SimpleNamespace(
                graph_mode=graph,
                device=device,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                sparse_kv_manager=manager,
                forward_metadata=SimpleNamespace(
                    sparse_kv_lim_plan=None, sparse_kv_offload_rows=None
                ),
            )

            def run():
                outputs = []
                for i in range(num_layers):
                    if i % 2 == 0:
                        plan = manager.lim_adapter.select(
                            query=query,
                            weights=weights,
                            key=keys,
                            block_table=table,
                            forward_batch=inputs,
                            slot_pool=manager.resident_pool.device_slot_map[
                                i - manager.start_layer
                            ],
                            reuse_metadata=i > 0,
                        )
                        manager.prepare_copy_sfa_plan(plan)
                        backend.forward_metadata.sparse_kv_lim_plan = plan
                    outputs.append(
                        forward_fused_sparse_kv_offload(
                            backend,
                            q,
                            fresh[i][..., :512].contiguous(),
                            None,
                            layer[i],
                            inputs,
                            q_rope=rope,
                            k_rope=fresh[i][..., 512:].contiguous(),
                            topk_indices=plan.source_ids,
                        ).clone()
                    )
                return plan, outputs

            adapter = manager.lim_adapter
            with (
                patch.object(
                    adapter, "_prepare_forward", wraps=adapter._prepare_forward
                ) as prepare_lim,
                patch.object(
                    manager,
                    "prepare_copy_sfa_plan",
                    wraps=manager.prepare_copy_sfa_plan,
                ) as prepare_copy,
                patch.object(
                    manager, "prepare_offload_rows", wraps=manager.prepare_offload_rows
                ) as prepare_offload,
            ):
                plan, outputs = run()
                assert prepare_lim.call_count == 1
                assert prepare_copy.call_count == num_layers // 2
                assert prepare_offload.call_count == 1
            torch.npu.synchronize()
            assert plan.miss_counts.cpu().tolist() == [C, C, 0]
            if compact_pd:
                assert plan.resident_safe_slots.cpu().tolist() == [2, 1, 3]
            if graph:
                captured = torch.npu.NPUGraph()
                with torch.npu.graph(captured):
                    plan, outputs = run()
            # Rewrite resident positions after a rejected draft. A warm plan
            # must refresh those slots without cold-filling the whole cache.
            for value in fresh:
                value.add_(1)
            if graph:
                captured.replay()
            else:
                plan, outputs = run()
            torch.npu.synchronize()
            assert plan.miss_counts.cpu().tolist() == [Q, Q, 0]
            # Reusing one request row must cold-fill only that row.
            if compact_pd:
                pool.resident_row_lifecycle.release(5)
                pool.resident_row_lifecycle.admit([6])
                inputs.req_pool_indices[0] = 6
                assert manager.resident_pool.request_to_resident.cpu().tolist() == [
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    2,
                ]
            else:
                manager.resident_pool.reset_rows([1])
            for value in fresh:
                value.add_(1)
            if graph:
                captured.replay()
            else:
                plan, outputs = run()
            torch.npu.synchronize()
            assert plan.miss_counts.cpu().tolist() == [C, Q, 0]
            ids = plan.source_ids.cpu()
            for i in range(num_layers):
                host = torch.cat(
                    (manager.host_k_buffer[i], manager.host_v_buffer[i]), dim=-1
                )
                actual = outputs[i].cpu().float().view(token_rows, heads, 512)
                errors = []
                for row in range(2):
                    for col in range(Q):
                        idx = row * Q + col
                        assert torch.all((ids[idx] >= 0) & (ids[idx] < L - Q + col + 1))
                        assert L - Q + col in ids[idx].tolist()
                        in_prefix = (ids[idx] >= row * K) & (ids[idx] < (row + 1) * K)
                        in_tail = ids[idx] >= L - Q
                        assert torch.all(in_prefix | in_tail)
                        selected = host.view(-1, 576)[
                            row * CONTEXT + ids[idx].long()
                        ].float()
                        score = (
                            q[idx].cpu().float() @ selected[:, :512].T
                            + rope[idx].cpu().float() @ selected[:, 512:].T
                        ) * (576**-0.5)
                        expected = torch.softmax(score, dim=-1) @ selected[:, :512]
                        errors.append(actual[idx] - expected)
                        torch.testing.assert_close(
                            actual[idx], expected, rtol=0.01, atol=0.002
                        )
                errors = torch.stack(errors)
                padding_max = actual[2 * Q :].abs().max().item()
                print(
                    "FRAMEWORK_MTP_METRICS "
                    + json.dumps(
                        {
                            "metadata_fused": True,
                            "graph": graph,
                            "heads": heads,
                            "query_width": Q,
                            "capacity": C,
                            "phase": "reuse",
                            "layer": i,
                            "max_abs": errors.abs().max().item(),
                            "rms": errors.square().mean().sqrt().item(),
                            "values": errors.numel(),
                            "padding_max_abs": padding_max,
                        }
                    )
                )
                assert padding_max == 0

            # Rollback below the previous visible length must cold-fill. Poison
            # old speculative keys so any future-token leakage is observable.
            rolled_length = L - 128
            inputs.positions[: 2 * Q].sub_(128)
            inputs.out_cache_loc[: 2 * Q].sub_(128)
            inputs.seq_lens[:2].sub_(128)
            inputs.seq_lens_cpu[:2].sub_(128)
            key_rows = keys.view(batch, CONTEXT, 1, 128)
            key_rows[:2, L - Q : L, 0, 0] = 32
            key_rows[:2, rolled_length - Q : rolled_length, 0, 0] = 16
            if graph:
                captured.replay()
            else:
                plan, outputs = run()
            torch.npu.synchronize()
            assert plan.miss_counts.cpu().tolist() == [C, C, 0]
            ids = plan.source_ids.cpu()
            for i in range(num_layers):
                host = torch.cat(
                    (manager.host_k_buffer[i], manager.host_v_buffer[i]), dim=-1
                )
                actual = outputs[i].cpu().float().view(token_rows, heads, 512)
                errors = []
                for row in range(2):
                    for col in range(Q):
                        idx = row * Q + col
                        visible = rolled_length - Q + col + 1
                        assert torch.all((ids[idx] >= 0) & (ids[idx] < visible))
                        assert rolled_length - Q + col in ids[idx].tolist()
                        selected = host.view(-1, 576)[
                            row * CONTEXT + ids[idx].long()
                        ].float()
                        score = (
                            q[idx].cpu().float() @ selected[:, :512].T
                            + rope[idx].cpu().float() @ selected[:, 512:].T
                        ) * (576**-0.5)
                        expected = torch.softmax(score, dim=-1) @ selected[:, :512]
                        errors.append(actual[idx] - expected)
                        torch.testing.assert_close(
                            actual[idx], expected, rtol=0.01, atol=0.002
                        )
                errors = torch.stack(errors)
                padding_max = actual[2 * Q :].abs().max().item()
                print(
                    "FRAMEWORK_MTP_METRICS "
                    + json.dumps(
                        {
                            "metadata_fused": True,
                            "graph": graph,
                            "heads": heads,
                            "query_width": Q,
                            "capacity": C,
                            "phase": "rollback",
                            "layer": i,
                            "max_abs": errors.abs().max().item(),
                            "rms": errors.square().mean().sqrt().item(),
                            "values": errors.numel(),
                            "padding_max_abs": padding_max,
                        }
                    )
                )
                assert padding_max == 0
    finally:
        if manager is not None:
            torch.npu.synchronize()
            for raw in manager.host_pool._allocations:
                manager.host_pool.offload.free(raw.data_ptr())
            manager.host_pool.offload.uninitialize()
