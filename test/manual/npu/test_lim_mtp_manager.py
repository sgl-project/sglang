"""Real-manager Q4 cold LIM/CopySFA integration on one Ascend device."""

import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torch_npu")

from sgl_kernel_npu.sparsity_driven_kv_offload import free_shm

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.attention import (
    forward_sparsity_driven_kv_offload,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.budget import (
    resolve_sparse_kv_budget,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
    SparseKVCacheManager,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode

K, C, Q, L = 2048, 12288, 4, 16389
CONTEXT = (L + 127) // 128 * 128


class _IndexPool(MLATokenToKVPool):
    """Isolate index-cache allocation while retaining the real manager path."""

    def __init__(self, key):
        self.key = key
        self.kv_lora_rank, self.qk_rope_head_dim = 512, 64
        self.store_dtype = torch.bfloat16
        self.start_layer, self.layer_num = 0, 2

    def get_index_k_buffer(self, layer_id):
        return self.key


@pytest.mark.parametrize("fused_metadata", [False, True])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("async_checks", [False, True])
@pytest.mark.parametrize("heads", [4, 8])
def test_real_manager_mtp_owner_share_lifecycle(
    fused_metadata, graph, async_checks, heads
):
    if not torch.npu.is_available():
        pytest.skip("NPU is required")
    device_id = torch.npu.current_device()
    device = torch.device("npu", device_id)
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
    token_rows = 16  # 12 logical rows plus a TP padding suffix.
    query = torch.zeros(token_rows, 32, 128, dtype=torch.bfloat16, device=device)
    query[:, :, 0] = 1
    weights = torch.full((token_rows, 32), 1 / 32, dtype=query.dtype, device=device)
    pool = ReqToTokenPool(2, CONTEXT, "npu", False)
    index_pool = _IndexPool(keys)
    budget = resolve_sparse_kv_budget(
        request_rows=3,
        graph_batch_capacity=3,
        context_len=CONTEXT,
        cache_tokens=C,
        num_layers=2,
        top_k=K,
        kv_bytes_per_token=1152,
        indexer_impl="lim",
        copy_sfa=True,
        copy_sfa_query_heads=heads,
        mtp_full_visible=True,
        fused_copy_sfa_metadata=fused_metadata,
    )
    inputs = SimpleNamespace(
        batch_size=batch,
        forward_mode=ForwardMode.TARGET_VERIFY,
        req_pool_indices=torch.tensor([1, 2, 0], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([L - Q, L - Q, 0], dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor([L - Q, L - Q, 0], dtype=torch.int32),
        positions=torch.tensor(
            list(range(L - Q, L)) * 2 + [0] * (token_rows - 2 * Q),
            dtype=torch.int64,
            device=device,
        ),
        out_cache_loc=torch.tensor(
            [1] * (2 * Q) + [-1] * (token_rows - 2 * Q),
            dtype=torch.int32,
            device=device,
        ),
        spec_info=SimpleNamespace(num_tokens_per_req=Q),
    )
    layer = [
        SimpleNamespace(
            layer_id=i, tp_k_head_num=1, tp_q_head_num=heads, scaling=576**-0.5
        )
        for i in range(2)
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
    spec = SimpleNamespace(speculative_eagle_topk=1, speculative_num_draft_tokens=4)
    try:
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_NPU_ENABLE_COPY_SFA.override(True))
            stack.enter_context(envs.SGLANG_ENABLE_ASYNC_ASSERT.override(async_checks))
            stack.enter_context(
                envs.SGLANG_ENABLE_NPU_LIM_MTP_FULL_VISIBLE.override(True)
            )
            stack.enter_context(
                envs.SGLANG_NPU_FUSED_LIM_COPY_SFA_METADATA.override(fused_metadata)
            )
            stack.enter_context(envs.SGLANG_NPU_FUSED_LIM_PRE_METADATA.override(False))
            stack.enter_context(
                patch("sglang.srt.runtime_context.get_spec", return_value=spec)
            )
            stack.enter_context(
                patch(
                    "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager.get_sparse_kv_budget",
                    return_value=budget,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager.get_sparse_kv_indexer_impl",
                    return_value="lim",
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager.get_eager_max_batch_size",
                    return_value=3,
                )
            )
            manager = SparseKVCacheManager(
                pool,
                SimpleNamespace(get_kvcache=lambda: index_pool),
                K,
                copy_sfa_padded_query_heads=heads,
                paged_kv_cache=index_pool,
            )
            for i in range(2):
                manager.host_kv_buffer[i].copy_(
                    torch.randn(
                        manager.host_kv_buffer[i].shape, generator=generator
                    ).to(torch.bfloat16)
                )
            backend = SimpleNamespace(
                graph_mode=graph,
                device=device,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                sparse_kv_manager=manager,
                forward_metadata=SimpleNamespace(sparse_kv_lim_plan=None),
            )

            def run():
                plan = manager.lim_adapter.select_mtp(
                    manager,
                    query=query,
                    weights=weights,
                    key=keys,
                    block_table=table,
                    forward_batch=inputs,
                    layer_id=0,
                )
                backend.forward_metadata.sparse_kv_lim_plan = plan
                outputs = []
                for i in range(2):
                    outputs.append(
                        forward_sparsity_driven_kv_offload(
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

            plan, outputs = run()
            torch.npu.synchronize()
            assert plan.miss_counts.cpu().tolist() == [C, C, 0]
            if graph:
                captured = torch.npu.NPUGraph()
                if async_checks:
                    # Ascend's torch._assert_async performs a synchronous copy
                    # during capture. The diagnostic must reject this mode.
                    with pytest.raises(
                        RuntimeError, match="async validation is unsupported"
                    ):
                        with torch.npu.graph(captured):
                            run()
                    return
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
            manager.reset_requests([1])
            for value in fresh:
                value.add_(1)
            if graph:
                captured.replay()
            else:
                plan, outputs = run()
            torch.npu.synchronize()
            assert plan.miss_counts.cpu().tolist() == [C, Q, 0]
            ids = plan.source_ids.cpu()
            for i in range(2):
                host = manager.host_kv_buffer[i]
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
                        selected = host[row + 1, ids[idx].long(), 0].float()
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
                            "metadata_fused": fused_metadata,
                            "graph": graph,
                            "async_checks": async_checks,
                            "heads": heads,
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
            for i in range(2):
                host = manager.host_kv_buffer[i]
                actual = outputs[i].cpu().float().view(token_rows, heads, 512)
                errors = []
                for row in range(2):
                    for col in range(Q):
                        idx = row * Q + col
                        visible = rolled_length - Q + col + 1
                        assert torch.all((ids[idx] >= 0) & (ids[idx] < visible))
                        assert rolled_length - Q + col in ids[idx].tolist()
                        selected = host[row + 1, ids[idx].long(), 0].float()
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
                            "metadata_fused": fused_metadata,
                            "graph": graph,
                            "async_checks": async_checks,
                            "heads": heads,
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
        torch.npu.synchronize()
        free_shm(device_id)
