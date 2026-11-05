from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
    AscendGDNAttnBackend,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")

requires_pool_prefill_op = pytest.mark.skipif(
    not hasattr(torch.ops.npu, "chunk_gated_delta_rule_pool"),
    reason="requires sgl-kernel-npu with chunk_gated_delta_rule_pool",
)


@requires_pool_prefill_op
def test_chunk_gdn_pool_matches_temporal_cow_gather_scatter():
    torch.npu.set_device("npu:0")
    torch.manual_seed(11)
    device = "npu:0"
    num_heads = 4
    head_dim = 128
    lengths = [65, 7]
    total = sum(lengths)
    q = F.normalize(torch.randn(total, num_heads, head_dim, device=device), dim=-1).to(
        torch.bfloat16
    )
    k = F.normalize(torch.randn(total, num_heads, head_dim, device=device), dim=-1).to(
        torch.bfloat16
    )
    v = torch.randn(total, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    g = F.logsigmoid(torch.randn(total, num_heads, device=device)).float()
    beta = torch.rand(total, num_heads, dtype=torch.bfloat16, device=device)
    pool = 0.05 * torch.randn(
        32, num_heads, head_dim, head_dim, dtype=torch.bfloat16, device=device
    )
    before = pool.clone()
    reference_pool = pool.clone()
    src_slots = torch.tensor([5, 5], dtype=torch.int32, device=device)
    dst_slots = torch.tensor([17, 29], dtype=torch.int32, device=device)
    has_initial = torch.tensor([False, True], dtype=torch.bool, device=device)
    cu_seqlens = torch.tensor([0, 65, 72], dtype=torch.int32, device=device)

    backend = object.__new__(AscendGDNAttnBackend)
    backend.actual_seq_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)

    reference_pool[dst_slots.long()] = reference_pool[src_slots.long()]
    reference_pool[dst_slots[0]] = 0
    expected_out, expected_final, expected_h = backend.chunk_gdn(
        q,
        k,
        v,
        g,
        beta,
        reference_pool,
        dst_slots,
        dst_slots,
        has_initial,
        cu_seqlens,
        use_pool=False,
    )
    reference_pool[dst_slots.long()] = expected_final

    actual_out, actual_final, actual_h = backend.chunk_gdn(
        q,
        k,
        v,
        g,
        beta,
        pool,
        src_slots,
        dst_slots,
        has_initial,
        cu_seqlens,
        use_pool=True,
    )
    torch.npu.synchronize()

    assert actual_final is None
    torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(actual_h, expected_h, rtol=0, atol=0)
    torch.testing.assert_close(pool, reference_pool, rtol=0, atol=0)
    torch.testing.assert_close(pool[5], before[5], rtol=0, atol=0)
    untouched = [slot for slot in range(pool.shape[0]) if slot not in (17, 29)]
    torch.testing.assert_close(pool[untouched], before[untouched], rtol=0, atol=0)


def _production_prefill_case(layer_pad=0):
    device = "npu:0"
    layers = 2 if layer_pad else 1
    lengths = [65, 7, 5]
    prefix_lens = [64, 32, 0]
    slots = 64
    allocator = MambaSlotAllocator(slots, device)
    source = allocator.alloc(1)
    destinations = allocator.alloc(3)
    assert source.dtype == destinations.dtype == torch.int64
    reqs = []
    for index, cached in enumerate((True, True, False)):
        reqs.append(
            SimpleNamespace(
                kv=SimpleNamespace(
                    mamba_pool_idx=destinations[index],
                    mamba_cow_src_index=source if cached else None,
                    mamba_needs_clear=not cached,
                )
            )
        )
    batch = object.__new__(ScheduleBatch)
    ScheduleBatch._collect_deferred_mamba_cow_and_clear(batch, reqs)
    assert batch.mamba_prefill_src_indices.dtype == torch.int64
    assert batch.mamba_prefill_src_indices.tolist() == [1, 1, 4]

    total = sum(lengths)
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(lengths),
        input_ids=torch.arange(total, device=device),
        req_pool_indices=torch.arange(len(lengths), device=device),
        seq_lens=torch.tensor(
            [p + n for p, n in zip(prefix_lens, lengths)], device=device
        ),
        out_cache_loc=torch.arange(total, device=device),
        seq_lens_sum=total,
        extend_seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
        extend_prefix_lens=torch.tensor(prefix_lens, device=device),
        extend_start_loc=torch.tensor(
            [0, lengths[0], sum(lengths[:2])], dtype=torch.int32, device=device
        ),
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens_cpu=prefix_lens,
        mamba_prefill_src_indices=batch.mamba_prefill_src_indices,
    )

    shape = (layers, slots + 1, 4, 128, 128)
    slot_stride = 4 * 128 * 128
    layer_stride = (slots + 1) * slot_stride + layer_pad
    storage = torch.empty(
        layers * layer_stride + slot_stride, dtype=torch.bfloat16, device=device
    )
    temporal = storage.as_strided(shape, (layer_stride, slot_stride, 128 * 128, 128, 1))
    temporal.copy_(0.05 * torch.randn(shape, dtype=torch.bfloat16, device=device))
    request_pool = object.__new__(HybridReqToTokenPool)
    request_pool.gdn_prefill_pool_io = True
    request_pool.state = SimpleNamespace(
        ckpt_pool=None,
        pool=SimpleNamespace(
            supports_temporal_cow_elision=True,
            mamba_cache=SimpleNamespace(temporal=temporal),
        ),
    )
    request_pool.get_mamba_indices = lambda rows: destinations[rows]

    backend = object.__new__(AscendGDNAttnBackend)
    backend.device = device
    backend.topk = 1
    backend.req_to_token_pool = request_pool
    backend.state_pool = None
    backend._state_ids = IdSpace(key=("mamba", None))
    metadata = backend._forward_metadata(forward_batch)
    backend.forward_metadata = metadata
    backend.prepare_gdn_inputs(len(lengths), ForwardMode.EXTEND, None)
    assert metadata.mamba_cache_indices.dtype == torch.int32
    assert metadata.mamba_cache_indices.is_contiguous()
    if metadata.use_gdn_pool_prefill:
        assert metadata.mamba_prefill_src_indices.dtype == torch.int32
        assert metadata.mamba_prefill_src_indices.is_contiguous()
    else:
        assert metadata.mamba_prefill_src_indices is None

    torch.manual_seed(19)
    inputs = (
        torch.randn(total, 4, 128, dtype=torch.bfloat16, device=device),
        torch.randn(total, 4, 128, dtype=torch.bfloat16, device=device),
        torch.randn(total, 4, 128, dtype=torch.bfloat16, device=device),
        F.logsigmoid(torch.randn(total, 4, device=device)).float(),
        torch.rand(total, 4, dtype=torch.bfloat16, device=device),
    )
    return SimpleNamespace(
        backend=backend,
        forward_batch=forward_batch,
        metadata=metadata,
        temporal=temporal,
        raw_sources=batch.mamba_prefill_src_indices,
        raw_destinations=destinations,
        inputs=inputs,
    )


def _check_production_path(case, expect_pool, temporal_restored=False):
    assert case.metadata.use_gdn_pool_prefill is expect_pool
    backend = case.backend
    forward_batch = case.forward_batch
    destinations = case.metadata.mamba_cache_indices
    has_initial = forward_batch.extend_prefix_lens > 0
    raw_sources = case.raw_sources
    raw_destinations = case.raw_destinations
    if expect_pool:
        sources = case.metadata.mamba_prefill_src_indices
        reference = case.temporal[-1].clone()
        reference[raw_destinations] = reference[raw_sources]
    else:
        assert case.metadata.mamba_prefill_src_indices is None
        sources = destinations
        if not temporal_restored:
            case.temporal[:, raw_destinations] = case.temporal[:, raw_sources]
        reference = case.temporal[-1].clone()
    reference[raw_destinations[~has_initial]] = 0
    expected_out, expected_final, expected_h = backend.chunk_gdn(
        *case.inputs,
        reference,
        raw_destinations,
        raw_destinations,
        has_initial,
        case.metadata.query_start_loc,
        use_pool=False,
    )
    reference[raw_destinations] = expected_final

    actual_out, actual_final, actual_h = backend.chunk_gdn(
        *case.inputs,
        case.temporal[-1],
        sources,
        destinations,
        has_initial,
        case.metadata.query_start_loc,
        use_pool=expect_pool,
    )
    if actual_final is not None:
        case.temporal[-1, destinations.long()] = actual_final
    torch.npu.synchronize()

    assert (actual_final is None) is expect_pool
    torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(actual_h, expected_h, rtol=0, atol=0)
    torch.testing.assert_close(case.temporal[-1], reference, rtol=0, atol=0)


def _refresh_metadata(case):
    case.metadata = case.backend._forward_metadata(case.forward_batch)
    case.backend.forward_metadata = case.metadata
    case.backend.prepare_gdn_inputs(
        case.forward_batch.batch_size, case.forward_batch.forward_mode, None
    )


@requires_pool_prefill_op
def test_allocator_schedule_metadata_backend_uses_contiguous_int32():
    torch.npu.set_device("npu:0")
    case = _production_prefill_case()
    _check_production_path(case, expect_pool=True)


def test_unaligned_layer_falls_back_to_restored_destination():
    torch.npu.set_device("npu:0")
    case = _production_prefill_case(layer_pad=16)
    _check_production_path(case, expect_pool=False)


def test_int8_checkpoint_hit_fallback_uses_restored_active_destinations():
    from sglang.srt.model_executor.model_runner import ModelRunner

    torch.npu.set_device("npu:0")
    case = _production_prefill_case()
    request_pool = case.backend.req_to_token_pool
    state = request_pool.state
    active_pool = state.pool

    def clear_slots(indices):
        active_pool.mamba_cache.temporal.index_fill_(1, indices.long(), 0)

    active_pool.clear_slots = clear_slots
    checkpoint = SimpleNamespace(calls=[])

    def load_to_active(pool, src_slots, dst_slots):
        checkpoint.calls.append((src_slots.clone(), dst_slots.clone()))
        pool.mamba_cache.temporal[:, dst_slots.long()] = 42

    checkpoint.load_to_active = load_to_active
    state.ckpt_pool = checkpoint
    forward_batch = case.forward_batch
    forward_batch.mamba_clear_indices = case.raw_destinations[2:]
    forward_batch.mamba_cow_src_indices = case.raw_sources[:2]
    forward_batch.mamba_cow_dst_indices = case.raw_destinations[:2]
    runner = object.__new__(ModelRunner)
    runner.req_to_token_pool = request_pool
    runner.token_to_kv_pool_allocator = SimpleNamespace(state=state)
    runner.token_to_kv_pool = SimpleNamespace()
    runner.runner_ids = SimpleNamespace(state_ids=lambda _pool: case.backend._state_ids)
    runner.is_draft_worker = False

    runner._maybe_execute_deferred_mamba_cow_and_clear(forward_batch)

    assert len(checkpoint.calls) == 1
    torch.testing.assert_close(checkpoint.calls[0][0], case.raw_sources[:2])
    torch.testing.assert_close(checkpoint.calls[0][1], case.raw_destinations[:2])
    torch.testing.assert_close(
        case.temporal[:, case.raw_destinations[:2]],
        torch.full_like(case.temporal[:, case.raw_destinations[:2]], 42),
    )
    torch.testing.assert_close(
        case.temporal[:, case.raw_destinations[2:]],
        torch.zeros_like(case.temporal[:, case.raw_destinations[2:]]),
    )
    _refresh_metadata(case)
    _check_production_path(case, expect_pool=False, temporal_restored=True)


def _merge_batch(reqs, destinations, forward_mode):
    from sglang.srt.model_executor.forward_mode import CaptureHiddenMode

    count = len(reqs)
    return ScheduleBatch(
        reqs=reqs,
        forward_mode=forward_mode,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        sampling_info=SimpleNamespace(merge_batch=lambda other: None),
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        req_pool_indices=torch.arange(count, device=destinations.device),
        req_pool_indices_cpu=torch.arange(count),
        seq_lens=torch.ones(count, dtype=torch.int64, device=destinations.device),
        seq_lens_cpu=torch.ones(count, dtype=torch.int64),
        orig_seq_lens=torch.ones(count, dtype=torch.int32, device=destinations.device),
    )


def test_mixed_prefill_decode_fallback_uses_all_active_destinations():
    torch.npu.set_device("npu:0")
    device = "npu:0"
    case = _production_prefill_case()
    running_destinations = torch.tensor([30, 31], dtype=torch.int64, device=device)
    destinations = torch.cat((case.raw_destinations, running_destinations))
    prefill_batch = _merge_batch(
        [SimpleNamespace() for _ in range(3)],
        case.raw_destinations,
        ForwardMode.MIXED,
    )
    prefill_batch.mamba_prefill_src_indices = case.raw_sources
    running_batch = _merge_batch(
        [SimpleNamespace() for _ in range(2)],
        running_destinations,
        ForwardMode.DECODE,
    )

    prefill_batch.merge_batch(running_batch)

    assert prefill_batch.mamba_prefill_src_indices.shape == (3,)
    assert prefill_batch.batch_size() == 5
    request_pool = case.backend.req_to_token_pool
    request_pool.get_mamba_indices = lambda rows: destinations[rows]
    case.forward_batch = ForwardBatch(
        forward_mode=ForwardMode.MIXED,
        batch_size=5,
        input_ids=torch.arange(5, device=device),
        req_pool_indices=torch.arange(5, device=device),
        seq_lens=torch.ones(5, dtype=torch.int64, device=device),
        out_cache_loc=torch.arange(5, device=device),
        seq_lens_sum=5,
        extend_seq_lens=torch.ones(5, dtype=torch.int32, device=device),
        extend_prefix_lens=torch.zeros(5, dtype=torch.int64, device=device),
        extend_start_loc=torch.arange(5, dtype=torch.int32, device=device),
        extend_seq_lens_cpu=[1] * 5,
        extend_prefix_lens_cpu=[0] * 5,
        mamba_prefill_src_indices=prefill_batch.mamba_prefill_src_indices,
    )
    case.raw_destinations = destinations
    case.inputs = (
        torch.randn(5, 4, 128, dtype=torch.bfloat16, device=device),
        torch.randn(5, 4, 128, dtype=torch.bfloat16, device=device),
        torch.randn(5, 4, 128, dtype=torch.bfloat16, device=device),
        F.logsigmoid(torch.randn(5, 4, device=device)).float(),
        torch.rand(5, 4, dtype=torch.bfloat16, device=device),
    )
    _refresh_metadata(case)
    _check_production_path(case, expect_pool=False, temporal_restored=True)
