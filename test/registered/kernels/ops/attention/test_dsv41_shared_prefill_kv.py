"""Shared compressed-KV workspace validity across layers and forwards."""

import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.attention import deepseek_v4_backend as backend
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    WORKSPACE_DIM,
    CompressedGather,
    SparsePrefillKVBuffer,
)
from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_shared_compressed_dequant_lifetime():
    device = "cuda"
    obj = object.__new__(backend.DeepseekV4AttnBackend)
    obj.sparse_prefill_kv_buffer = SparsePrefillKVBuffer(device)
    obj.shared_compressed_prefill_workspaces = {
        ratio: SparsePrefillKVBuffer(device) for ratio in (1, 2)
    }
    obj.softmax_scale = 0.1
    obj.head_dim_v = WORKSPACE_DIM
    sources = {0: 0, 1: 1, 2: 0, 3: 1, 4: 4, 5: 4, 6: 6, 7: 7}
    ratios = {0: 1, 1: 2, 2: 1, 3: 2, 4: 1, 5: 1, 6: 0, 7: 4}
    compressed = {
        source: torch.full(
            (128, 1, WORKSPACE_DIM),
            float(source + 1),
            device=device,
            dtype=torch.bfloat16,
        )
        for source in (0, 1, 4, 7)
    }
    swa = torch.empty((16, 1, WORKSPACE_DIM), device=device, dtype=torch.bfloat16)
    pool = NS(
        source_layer_of=lambda layer: sources[layer],
        get_extra_key_page_size=lambda layer: 1,
        get_extra_key_buffer=lambda layer: compressed[sources[layer]],
        get_extra_key_layout=lambda layer: None,
        get_swa_key_buffer_radix=lambda layer: swa,
        get_swa_key_layout=lambda: None,
    )
    calls = []
    active = [True]

    def dequant(src, indices, *, out, **kwargs):
        calls.append("swa" if src is swa else sources_by_ptr[src.data_ptr()])
        out.copy_(src.index_select(0, indices.long()))

    sources_by_ptr = {v.data_ptr(): k for k, v in compressed.items()}

    def make_cache(n):
        gathers = {
            ratio: CompressedGather(
                flat_token_ids=torch.arange(
                    n // ratio, device=device, dtype=torch.int32
                ),
                compressed_base=torch.zeros(1, device=device, dtype=torch.int32),
                swa_base=torch.zeros(1, device=device, dtype=torch.int32),
            )
            for ratio in (1, 2, 4)
        }
        indices = torch.zeros((4, 128), device=device, dtype=torch.int32)
        lengths = torch.full((4,), 3, device=device, dtype=torch.int32)
        cache = NS(
            compressed=gathers,
            swa_token_ids=torch.arange(3, device=device),
            swa_page_size=1,
            c0_combined_indices=indices,
            c0_combined_lens=lengths,
        )
        cache.layer_inputs = lambda ratio, core, page: (
            gathers[ratio].flat_token_ids,
            indices,
            lengths,
        )
        return cache

    def forward(layer):
        if not active[0]:
            obj.sparse_prefill_kv_buffer.forget_head()
        return obj._forward_prefill_sparse(
            torch.empty((4, 1, 1, WORKSPACE_DIM), device=device, dtype=torch.bfloat16),
            layer,
            ratios[layer],
            NS(),
            pool,
            NS(),
            torch.zeros(1, device=device),
        )

    with (
        patch.object(backend, "is_cp_active", side_effect=lambda _: active[0]),
        patch.object(backend, "dequantize_k_cache_paged_dsv4", side_effect=dequant),
        patch(
            "sglang.kernels.ops.attention.flash_mla_interface.flash_mla_sparse_fwd",
            side_effect=lambda **kw: (kw["kv"].clone(), None, None),
        ),
    ):
        # Same-size replay, then growth and shrink must all read freshly written KV.
        for step, n in enumerate((8, 8, 40, 4)):
            obj.forward_metadata = NS(sparse_prefill_cache=make_cache(n))
            for source, tensor in compressed.items():
                tensor.fill_(source + 1 + step * 10)
            for layer, should_dequant in (
                (0, True),
                (1, True),
                (2, False),
                (3, False),
                (4, True),
                (5, False),
                (2, True),
                (3, False),
                (6, False),
                (7, True),
                (7, False),
            ):
                swa.fill_(100 + layer + step)
                before = len(calls)
                active[0] = True
                actual = forward(layer)
                actual_calls = calls[before:]
                assert actual_calls.count("swa") == 1
                assert len(actual_calls) == 1 + int(should_dequant)
                active[0] = False
                expected = forward(layer)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            # Workspace replacement invalidates even an unchanged source identity.
            obj.shared_compressed_prefill_workspaces[1].get(256 + step * 256)
            active[0] = True
            before = len(calls)
            actual = forward(2)
            assert calls[before:] == [0, "swa"]
            active[0] = False
            torch.testing.assert_close(actual, forward(2), rtol=0, atol=0)

            # Re-executing a producer may mutate the same cache address in place.
            compressed[0].add_(1)
            active[0] = True
            before = len(calls)
            actual = forward(0)
            assert calls[before:] == [0, "swa"]
            active[0] = False
            torch.testing.assert_close(actual, forward(0), rtol=0, atol=0)


def _h200_cp_backend():
    obj = object.__new__(backend.DeepseekV4AttnBackend)
    obj.mtp_enabled = False
    obj.trtllm_attn = False
    obj.is_dsv41 = True
    obj.mega_attn = False
    obj.dsv4_prefill_backend = "auto"
    obj.token_to_kv_pool = Mock(spec=backend.DeepSeekV4TokenToKVPool)
    obj.token_to_kv_pool.request_window = None
    obj.forward_metadata = NS(late_layer_tail=None)
    batch = NS(
        forward_mode=ForwardMode.EXTEND,
        attn_cp_metadata=InterleaveContextParallelMetadata(),
    )
    return obj, batch


def test_h200_cp_real_forward_dispatches_to_bf16_sparse_prefill():
    obj, batch = _h200_cp_backend()
    n, page, width = 3, 64, 584
    pool = obj.token_to_kv_pool
    pool.get_swa_key_buffer_radix.return_value = torch.zeros(1, page * width)
    pool.get_swa_key_page_size.return_value = page
    pool.get_swa_key_bytes_per_token.return_value = width
    pool.get_extra_key_buffer.return_value = torch.zeros(1, page * width)
    pool.get_extra_key_page_size.return_value = page
    pool.get_extra_key_bytes_per_token.return_value = width
    core = Mock(spec=backend.DSV4AttnMetadata)
    core.swa_page_indices = torch.zeros(n, 64, dtype=torch.int32)
    core.swa_topk_lengths = torch.ones(n, dtype=torch.int32)
    core.sparse_page_indices.return_value = torch.zeros(n, 64, dtype=torch.int32)
    core.sparse_topk_lengths.return_value = torch.ones(n, dtype=torch.int32)
    obj.forward_metadata.core_attn_metadata = core
    result = object()
    obj._forward_prefill_sparse = Mock(return_value=result)
    obj._forward_prefill_sparse_q8kv8 = Mock(side_effect=AssertionError("Q8 bypass"))
    with (
        patch.object(backend, "is_cp_active", return_value=True),
        patch.object(
            backend, "get_platform", return_value=NS(is_sm90=True, is_sm120=False)
        ),
        patch.object(backend, "use_dsv4_q8kv8_sparse_prefill", return_value=False),
    ):
        actual = obj.forward(
            q=torch.zeros(n, 16, 512),
            k=None,
            v=None,
            layer=NS(layer_id=0),
            forward_batch=batch,
            compress_ratio=1,
            save_kv_cache=False,
            attn_sink=torch.zeros(16),
        )
    assert actual is result
    obj._forward_prefill_sparse.assert_called_once()
    assert obj._forward_prefill_sparse.call_args.kwargs["forward_batch"] is batch


@pytest.mark.parametrize(
    "unsupported",
    ["request_window", "tail", "local_index", "mega", "sm100", "q8", "non_v41"],
)
def test_h200_cp_sparse_gate_keeps_unsupported_combinations_paged(unsupported):
    obj, batch = _h200_cp_backend()
    if unsupported == "request_window":
        obj.token_to_kv_pool.request_window = object()
    if unsupported == "tail":
        obj.forward_metadata.late_layer_tail = object()
    if unsupported == "local_index":
        batch.attn_cp_metadata.local_index = torch.tensor([0])
    if unsupported == "mega":
        obj.mega_attn = True
    if unsupported == "non_v41":
        obj.is_dsv41 = False
    with (
        patch.object(backend, "is_cp_active", return_value=True),
        patch.object(
            backend,
            "get_platform",
            return_value=NS(is_sm90=unsupported != "sm100", is_sm120=False),
        ),
        patch.object(
            backend, "use_dsv4_q8kv8_sparse_prefill", return_value=unsupported == "q8"
        ),
    ):
        assert not obj._use_sparse_prefill(batch, num_qo_tokens=16)


@pytest.mark.parametrize("cp,rank", [(2, 0), (2, 1), *[(8, rank) for rank in range(8)]])
def test_cp_cache_uses_local_queries_and_global_prefix_positions(cp, rank):
    device = "cuda"
    obj = object.__new__(backend.DeepseekV4AttnBackend)
    obj.sparse_prefill_kv_buffer = SparsePrefillKVBuffer(device)
    obj.forward_metadata = NS(late_layer_tail=None)
    obj.req_to_token = torch.arange(6 * 64, device=device, dtype=torch.int32).view(
        6, 64
    )
    obj.token_to_kv_pool = NS(
        full_to_swa_index_mapping=torch.arange(6 * 64, device=device),
        get_swa_key_page_size=lambda: 256,
    )
    req_ids = torch.tensor([2, 4, 1], dtype=torch.int32, device=device)
    lengths = [3, 0, 9]
    seq_lens = [10, 0, 20]
    owners = torch.repeat_interleave(req_ids, torch.tensor(lengths, device=device))
    positions = torch.cat(
        (torch.arange(7, 10, device=device), torch.arange(11, 20, device=device))
    )
    local_owners, local_pos = owners[rank::cp], positions[rank::cp]
    counts = [(local_owners == r).sum().item() for r in req_ids]
    physical_rows = local_pos.numel() + 2
    batch = NS(
        forward_mode=ForwardMode.EXTEND,
        attn_cp_metadata=InterleaveContextParallelMetadata(),
        seq_lens_cpu=torch.tensor(seq_lens),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        extend_seq_lens_cpu=lengths,
        extend_seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
        req_pool_indices=req_ids,
    )
    core = NS(
        request_window_layout=None, seq_lens_casual=(local_pos + 1).to(torch.int32)
    )
    with (
        patch.object(backend, "is_cp_active", return_value=True),
        patch.object(
            backend, "get_parallel", return_value=NS(attn_cp_size=cp, attn_cp_rank=rank)
        ),
    ):
        cache = obj._build_sparse_prefill_chunk_cache(
            batch, core, num_qo_tokens=physical_rows
        )
    assert cache.query_lens_cpu == counts
    assert cache.query_start_loc.cpu().tolist() == [
        0,
        counts[0],
        counts[0] + counts[1],
        sum(counts),
    ]
    torch.testing.assert_close(
        cache.query_pos[: local_pos.numel()], local_pos.to(torch.int32)
    )
    for ratio in (1, 2):
        table = torch.full(
            (physical_rows, max(seq_lens) // ratio),
            -1,
            dtype=torch.int32,
            device=device,
        )
        for row, r in enumerate(local_owners.tolist()):
            table[row] = (
                obj.req_to_token[r, torch.arange(table.shape[1], device=device) * ratio]
                // ratio
            )
        gather = cache.ensure_compressed(ratio, table, 1)
        expected = torch.cat(
            [
                obj.req_to_token[
                    int(r), torch.arange(length // ratio, device=device) * ratio
                ]
                // ratio
                for r, length, count in zip(req_ids, seq_lens, counts)
                if count
            ]
        ).to(torch.int32)
        torch.testing.assert_close(gather.flat_token_ids, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
