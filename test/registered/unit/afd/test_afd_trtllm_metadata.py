"""S2 metadata storage must stay private, stable and fail closed on drift."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd.contracts import AFDError
from sglang.srt.afd.metadata import DSAMetadataGuard
from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention import trtllm_mha_backend as trtllm
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.trtllm_mha_interleaved import (
    TRTLLMInterleavedGraphMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _backend():
    backend = object.__new__(trtllm.TRTLLMHAAttnBackend)
    backend.is_xqa_impl = backend.is_nvfp4_kvcache = False
    backend.use_sliding_window_kv_pool = backend.expand_encoder_only_verify = False
    backend.speculative_num_draft_tokens = None
    backend.max_num_pages = 3
    backend.kv_index_translator = SimpleNamespace(
        is_translating=True, copy_page_table=lambda plan, out: out.copy_(plan)
    )
    backend.workspace_buffer = torch.zeros(8, dtype=torch.uint8)
    backend._multi_ctas_kv_counter_buffer = torch.zeros(4, dtype=torch.int32)
    backend.decode_cuda_graph_metadata = {}
    backend.forward_metadata = None
    return backend


def _batch(rows=2, offset=0):
    return SimpleNamespace(
        batch_size=rows,
        forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True),
        spec_info=None,
        seq_lens=torch.arange(rows, dtype=torch.int32) + 1,
        kv_loc_plan=torch.arange(rows * 3, dtype=torch.int32).reshape(rows, 3) + offset,
    )


def _snapshot(backend, batch, stage=0):
    return TRTLLMInterleavedGraphMetadata(
        backend, batch, stage_index=stage, metadata_type=trtllm.TRTLLMMHAMetadata
    )


def test_stages_isolate_scratch_and_reuse_storage_across_refresh():
    backend = _backend()
    assert backend._active_workspace_buffer() is backend.workspace_buffer
    first = _snapshot(backend, _batch())
    second = _snapshot(backend, _batch(offset=10), stage=1)
    original = first.metadata.page_table.clone()
    pointer = first.metadata.page_table.data_ptr()
    assert (
        first.metadata.workspace_buffer
        is _snapshot(backend, _batch(rows=1)).metadata.workspace_buffer
    )
    assert first.metadata.workspace_buffer is not second.metadata.workspace_buffer
    assert (
        first.metadata.multi_ctas_kv_counter_buffer
        is not second.metadata.multi_ctas_kv_counter_buffer
    )
    second.refresh(_batch(offset=20))
    torch.testing.assert_close(first.metadata.page_table, original)
    for offset in (30, 0):
        first.refresh(_batch(offset=offset))
        assert first.metadata.page_table.data_ptr() == pointer
        torch.testing.assert_close(
            first.metadata.page_table, _batch(offset=offset).kv_loc_plan
        )
    assert backend.decode_cuda_graph_metadata == {}
    first.metadata.page_table = first.metadata.page_table.clone()
    with pytest.raises(RuntimeError, match="METADATA_DRIFT"):
        first.refresh(_batch())


def test_unsupported_features_and_changed_batch_fail_before_execution():
    backend = _backend()
    backend.is_xqa_impl = True
    with pytest.raises(RuntimeError, match="REQUIRES_GEN_MHA"):
        _snapshot(backend, _batch())
    backend.is_xqa_impl = False
    snapshot = _snapshot(backend, _batch())
    with pytest.raises(RuntimeError, match="NON_SPEC_DECODE"):
        snapshot.refresh(_batch(rows=1))
    speculative = _batch()
    speculative.spec_info = object()
    with pytest.raises(RuntimeError, match="NON_SPEC_DECODE"):
        snapshot.refresh(speculative)


@pytest.fixture
def dsa_backend_fixture(monkeypatch):
    # Avoid the full GPU/ModelRunner constructor. The actual CPU metadata
    # allocation, capture prep, page-table refresh and owner lookup still run.
    monkeypatch.setattr(dsa_backend, "is_cuda", lambda: False)
    monkeypatch.setattr(dsa_backend, "_is_hip", False)
    value = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
    value.__dict__.update(
        _memory_saver_adapter=dsa_backend.TorchMemorySaverAdapter.create(enable=False),
        device=torch.device("cpu"),
        workspace_buffer=None,
        prefill_attention_backend_str="nsa",
        decode_attention_backend_str="nsa",
        req_to_token=torch.arange(16 * 8, dtype=torch.int32).reshape(16, 8),
        physical_page_size=1,
        dsa_index_kpool=1,
        dsa_index_topk=4,
        dsa_decode_impl="triton",
        dsa_topk_backend=DSATopKBackend.TORCH,
        experimental_kpool_metadata_fusion=False,
        enable_auto_select_prefill_impl=False,
        _arange_buf=torch.arange(32, dtype=torch.int32),
        _is_in_breakable_cuda_graph=lambda: False,
        _is_in_tc_piecewise_cuda_graph=lambda: False,
        _get_device_sm=lambda: 0,
        _is_blackwell=lambda: False,
        decode_cuda_graph_metadata={"original": object()},
        forward_metadata=object(),
    )
    return value


def _dsa_batch(
    slots=(1, 2),
    seq_lens=(3, 6),
    *,
    mode=ForwardMode.DECODE,
    non_padded=None,
    scalar=False,
):
    lengths = torch.tensor(seq_lens, dtype=torch.int32)
    count = torch.tensor(non_padded, dtype=torch.int32) if scalar else None
    return ForwardBatch(
        forward_mode=mode,
        batch_size=len(slots),
        input_ids=torch.arange(len(slots), dtype=torch.int64),
        req_pool_indices=torch.tensor(slots, dtype=torch.int64),
        seq_lens=lengths,
        seq_lens_cpu=lengths.clone(),
        out_cache_loc=torch.tensor(slots, dtype=torch.int64),
        seq_lens_sum=sum(seq_lens),
        positions=lengths.to(torch.int64) - 1,
        global_num_token_non_padded_cpu=non_padded,
        global_num_token_non_padded=count,
        num_token_non_padded=count,
    )


def _dsa_guard(dsa_backend_fixture, batch, *, bucket_rows=4):
    stage = SimpleNamespace(index=0, forward_batch=batch, attention_metadata=object())
    return DSAMetadataGuard(
        backend=dsa_backend_fixture, stage=stage, bucket_rows=bucket_rows
    )


def _tensor_storage(value):
    return {
        field.name: getattr(value, field.name).data_ptr()
        for field in dataclasses.fields(value)
        if torch.is_tensor(getattr(value, field.name))
    }


def _check_metadata(dsa_backend_fixture, metadata, batch):
    lengths = batch.seq_lens.to(torch.int32)
    torch.testing.assert_close(metadata.cache_seqlens_int32, lengths)
    torch.testing.assert_close(metadata.cu_seqlens_k[1:], lengths.cumsum(0).int())
    torch.testing.assert_close(metadata.dsa_cache_seqlens_int32, lengths.clamp(max=4))
    torch.testing.assert_close(
        metadata.dsa_cu_seqlens_k[1:], lengths.clamp(max=4).cumsum(0).int()
    )
    torch.testing.assert_close(
        metadata.page_table_1, dsa_backend_fixture.req_to_token[batch.req_pool_indices]
    )
    if dsa_backend_fixture.physical_page_size == 1:
        assert metadata.real_page_table is metadata.page_table_1
    else:
        torch.testing.assert_close(
            metadata.real_page_table,
            metadata.page_table_1[:, :: dsa_backend_fixture.physical_page_size]
            // dsa_backend_fixture.physical_page_size,
        )


@pytest.mark.parametrize("bucket_rows", [(4, 4), (4, 8)])
@pytest.mark.parametrize("page_size", [1, 2])
def test_dsa_stages_refresh_real_metadata_without_replacing_storage(
    dsa_backend_fixture, bucket_rows, page_size, monkeypatch
):
    dsa_backend_fixture.physical_page_size = page_size
    original_state = dsa_backend_fixture.decode_cuda_graph_metadata
    original_metadata = dsa_backend_fixture.forward_metadata
    batches = [_dsa_batch(), _dsa_batch((7,), (2,))]
    guards = [
        _dsa_guard(dsa_backend_fixture, batch, bucket_rows=rows)
        for batch, rows in zip(batches, bucket_rows)
    ]
    captured = []
    storage = []
    batch_storage = []
    for guard, batch in zip(guards, batches):
        guard.capture(batch)
        metadata = dsa_backend_fixture.forward_metadata
        assert isinstance(metadata, dsa_backend.DSAMetadata)
        captured.append(metadata)
        storage.append(_tensor_storage(metadata))
        padded = guard.graph_forward_batch
        batch_storage.append(_tensor_storage(padded))
        guard.activate_in_graph()
        assert guard._stage.forward_batch is padded
        assert guard._stage.attention_metadata is metadata
        guard.restore()
        assert dsa_backend_fixture.decode_cuda_graph_metadata is original_state
        assert dsa_backend_fixture.forward_metadata is original_metadata
        assert guard._stage.forward_batch is batch

    # Exercise scratch binding on the real frozen metadata, not a namespace.
    dsa_backend_fixture.workspace_buffer = torch.zeros(8)
    dsa_backend_fixture._multi_ctas_kv_counter_buffer = torch.zeros(4)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for index, metadata in enumerate(captured):
        dsa_backend_fixture.bind_interleaved_stage_workspace(
            metadata, stage_index=index
        )
    for first, second in zip(
        captured[0].interleaved_stage_scratch, captured[1].interleaved_stage_scratch
    ):
        assert first.data_ptr() != second.data_ptr()

    assert captured[0] is not captured[1]
    for field in ("page_table_1", "cache_seqlens_int32", "dsa_cache_seqlens_int32"):
        assert storage[0][field] != storage[1][field]

    for updates in (
        [_dsa_batch((3, 5, 8), (7, 2, 4)), _dsa_batch((9, 6), (5, 3))],
        [_dsa_batch((2,), (1,)), _dsa_batch((1, 4, 10), (4, 6, 2))],
    ):
        for index, (guard, live_batch) in enumerate(zip(guards, updates)):
            peer = captured[1 - index]
            peer_snapshot = peer.page_table_1.clone()
            guard.prepare_replay(live_batch)
            guard.assert_stable()
            padded = guard.graph_forward_batch
            _check_metadata(dsa_backend_fixture, captured[index], padded)
            assert _tensor_storage(captured[index]) == storage[index]
            assert _tensor_storage(padded) == batch_storage[index]
            torch.testing.assert_close(peer.page_table_1, peer_snapshot)
            assert padded.seq_lens.tolist()[live_batch.batch_size :] == [1] * (
                padded.batch_size - live_batch.batch_size
            )
            assert padded.req_pool_indices.tolist()[live_batch.batch_size :] == [0] * (
                padded.batch_size - live_batch.batch_size
            )
            guard.restore()
            assert dsa_backend_fixture.decode_cuda_graph_metadata is original_state
            assert dsa_backend_fixture.forward_metadata is original_metadata


@pytest.mark.parametrize("failure", ["missing", "wrong_size"])
def test_dsa_guard_rejects_lost_owner_before_replay(dsa_backend_fixture, failure):
    batch = _dsa_batch()
    guard = _dsa_guard(dsa_backend_fixture, batch)
    guard.capture(batch)
    metadata = dsa_backend_fixture.forward_metadata
    guard.restore()
    if failure == "missing":
        dsa_backend_fixture._afd_private_graph_states.pop(id(metadata))
    else:
        dsa_backend_fixture._afd_private_graph_states[id(metadata)]["batch_size"] = 7
    with pytest.raises(AFDError, match="AFD_DSA_METADATA_OWNERSHIP_LOST"):
        guard.prepare_replay(batch)


def test_private_state_release_preserves_peer_and_is_idempotent(dsa_backend_fixture):
    import weakref

    batch = _dsa_batch()
    guards = [_dsa_guard(dsa_backend_fixture, batch) for _ in range(2)]
    for guard in guards:
        guard.capture(batch)
        guard.restore()
    metadata = guards[0]._capture_metadata
    state = dsa_backend_fixture._afd_private_graph_states[id(metadata)]["graph_state"]
    refs = [weakref.ref(metadata), weakref.ref(state["cache_seqlens"])]
    del metadata, state
    guards[0].close()
    guards[0].close()
    assert all(ref() is None for ref in refs)
    assert len(dsa_backend_fixture._afd_private_graph_states) == 1
    guards[1].prepare_replay(batch)
    guards[1].close()
    assert dsa_backend_fixture._afd_private_graph_states == {}


def test_warmup_and_replay_preserve_stage_counts_and_restore_metadata(
    dsa_backend_fixture,
):
    from sglang.srt.afd.model_adapters.base import stage_forward_context
    from sglang.srt.model_executor.forward_context import get_forward_context

    original = dsa_backend_fixture.forward_metadata
    batch = _dsa_batch(non_padded=2, scalar=True)
    guards = [_dsa_guard(dsa_backend_fixture, batch) for _ in range(2)]
    for guard in guards:
        guard.capture(batch)
    counts = [guard.graph_forward_batch.num_token_non_padded for guard in guards]
    assert (
        len({value.data_ptr() for value in [batch.num_token_non_padded, *counts]}) == 3
    )
    for _ in range(3):
        for guard in guards:
            guard.activate_in_graph()
            padded = guard.graph_forward_batch
            with stage_forward_context(guard._stage.forward_batch):
                assert get_forward_context().moe_num_token_non_padded.__self__ is padded
                assert padded.global_num_token_non_padded_cpu == 2
    for guard in reversed(guards):
        guard.restore()
    assert dsa_backend_fixture.forward_metadata is original
    for index, guard in enumerate(guards):
        live = _dsa_batch(
            (1,),
            (2,),
            mode=ForwardMode.DECODE if index else ForwardMode.IDLE,
            non_padded=index,
            scalar=True,
        )
        peer = counts[1 - index].clone()
        guard.prepare_replay(live)
        guard.assert_stable()
        padded = guard.graph_forward_batch
        assert padded.num_token_non_padded is counts[index]
        assert padded.global_num_token_non_padded is counts[index]
        assert padded.global_num_token_non_padded_cpu == counts[index].item() == index
        torch.testing.assert_close(counts[1 - index], peer)
        guard.restore()
        assert guard._stage.forward_batch is batch
    assert batch.num_token_non_padded.item() == 2
    for guard in reversed(guards):
        guard.close()
    assert dsa_backend_fixture._afd_private_graph_states == {}


def test_trtllm_guard_owns_private_padded_batch_and_restores_parent():
    from sglang.srt.afd.metadata import TRTLLMMHAMetadataGuard

    backend = _backend()
    backend.prefill_attention_backend_str = backend.decode_attention_backend_str = (
        "trtllm_mha"
    )
    backend.kv_index_translator.is_translating = False
    batch = _dsa_batch(non_padded=2, scalar=True)
    stage = SimpleNamespace(index=0, forward_batch=batch, attention_metadata=object())
    original = stage.attention_metadata
    guard = TRTLLMMHAMetadataGuard(backend=backend, stage=stage, bucket_rows=4)
    with pytest.raises(AFDError, match="PADDED_BATCH_MISSING"):
        _ = guard.graph_forward_batch
    guard.capture(batch)
    first = guard.graph_forward_batch
    metadata = backend.forward_metadata
    assert first.batch_size == 4 and first.num_padding == 2
    assert first.input_ids.tolist() == [0, 1, 0, 0]
    with pytest.raises(AFDError, match="GUARD_REENTERED"):
        guard.capture(batch)
    guard.restore()
    assert backend.forward_metadata is None and stage.attention_metadata is original
    guard.capture(batch)
    assert guard.graph_forward_batch is first and backend.forward_metadata is metadata
    guard.restore()
    guard.prepare_replay(_dsa_batch((5,), (7,), non_padded=1, scalar=True))
    guard.assert_stable()
    assert first.req_pool_indices.tolist() == [5, 0, 0, 0]
    assert first.seq_lens.tolist() == [7, 1, 1, 1]
    guard.close()
    guard.close()
    assert backend.forward_metadata is None and stage.forward_batch is batch


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
