"""Real CPU DSA metadata generation with AFD stage ownership contracts.

These tests run the non-fused Torch implementation, not CUDA capture or attention
kernels. Device/capture probes are fixed to CPU; metadata generation is real.
"""

from __future__ import annotations

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.afd.contracts import AFDError
from sglang.srt.afd.metadata import DSAMetadataGuard
from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


@pytest.fixture
def backend(monkeypatch):
    # Avoid the full GPU/ModelRunner constructor. The actual CPU metadata
    # allocation, capture prep, page-table refresh and owner lookup still run.
    monkeypatch.setattr(dsa_backend, "is_cuda", lambda: False)
    monkeypatch.setattr(dsa_backend, "_is_hip", False)
    value = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
    value.__dict__.update(
        _memory_saver_adapter=dsa_backend.TorchMemorySaverAdapter.create(enable=False),
        device=torch.device("cpu"),
        prefill_attention_backend_str="nsa",
        decode_attention_backend_str="nsa",
        req_to_token=torch.arange(16 * 8, dtype=torch.int32).reshape(16, 8),
        real_page_size=1,
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


def _batch(slots=(1, 2), seq_lens=(3, 6), *, mode=ForwardMode.DECODE):
    lengths = torch.tensor(seq_lens, dtype=torch.int32)
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
    )


def _guard(backend, batch, *, bucket_rows=4):
    stage = SimpleNamespace(forward_batch=batch, attention_metadata=object())
    return DSAMetadataGuard(backend=backend, stage=stage, bucket_rows=bucket_rows)


def _tensor_storage(value):
    return {
        field.name: getattr(value, field.name).data_ptr()
        for field in dataclasses.fields(value)
        if torch.is_tensor(getattr(value, field.name))
    }


def _check_metadata(backend, metadata, batch):
    lengths = batch.seq_lens.to(torch.int32)
    torch.testing.assert_close(metadata.cache_seqlens_int32, lengths)
    torch.testing.assert_close(metadata.cu_seqlens_k[1:], lengths.cumsum(0).int())
    torch.testing.assert_close(metadata.dsa_cache_seqlens_int32, lengths.clamp(max=4))
    torch.testing.assert_close(
        metadata.dsa_cu_seqlens_k[1:], lengths.clamp(max=4).cumsum(0).int()
    )
    torch.testing.assert_close(
        metadata.page_table_1, backend.req_to_token[batch.req_pool_indices]
    )
    if backend.real_page_size == 1:
        assert metadata.real_page_table is metadata.page_table_1
    else:
        torch.testing.assert_close(
            metadata.real_page_table,
            metadata.page_table_1[:, :: backend.real_page_size]
            // backend.real_page_size,
        )


@pytest.mark.parametrize("bucket_rows", [(4, 4), (4, 8)])
@pytest.mark.parametrize("page_size", [1, 2])
def test_dsa_stages_refresh_real_metadata_without_replacing_storage(
    backend, bucket_rows, page_size
):
    backend.real_page_size = page_size
    original_state = backend.decode_cuda_graph_metadata
    original_metadata = backend.forward_metadata
    batches = [_batch(), _batch((7,), (2,))]
    guards = [
        _guard(backend, batch, bucket_rows=rows)
        for batch, rows in zip(batches, bucket_rows)
    ]
    captured = []
    storage = []
    batch_storage = []
    for guard, batch in zip(guards, batches):
        guard.capture(batch)
        metadata = backend.forward_metadata
        assert isinstance(metadata, dsa_backend.DSAMetadata)
        captured.append(metadata)
        storage.append(_tensor_storage(metadata))
        padded = guard.graph_forward_batch
        batch_storage.append(_tensor_storage(padded))
        guard.activate_in_graph()
        assert guard._stage.forward_batch is padded
        assert guard._stage.attention_metadata is metadata
        guard.restore()
        assert backend.decode_cuda_graph_metadata is original_state
        assert backend.forward_metadata is original_metadata
        assert guard._stage.forward_batch is batch

    assert captured[0] is not captured[1]
    for field in ("page_table_1", "cache_seqlens_int32", "dsa_cache_seqlens_int32"):
        assert storage[0][field] != storage[1][field]

    for updates in (
        [_batch((3, 5, 8), (7, 2, 4)), _batch((9, 6), (5, 3))],
        [_batch((2,), (1,)), _batch((1, 4, 10), (4, 6, 2))],
    ):
        for index, (guard, live_batch) in enumerate(zip(guards, updates)):
            peer = captured[1 - index]
            peer_snapshot = peer.page_table_1.clone()
            guard.prepare_replay(live_batch)
            guard.assert_stable()
            padded = guard.graph_forward_batch
            _check_metadata(backend, captured[index], padded)
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
            assert backend.decode_cuda_graph_metadata is original_state
            assert backend.forward_metadata is original_metadata


@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.IDLE])
def test_dsa_owner_checks_shape_without_exposing_state_bank(backend, mode):
    batch = _batch(mode=mode)
    metadata = backend.init_forward_metadata_for_afd_capture(batch)
    assert backend.owns_afd_capture_metadata(metadata, batch_size=2)
    assert not backend.owns_afd_capture_metadata(metadata, batch_size=3)
    assert not backend.owns_afd_capture_metadata(object(), batch_size=2)


@pytest.mark.parametrize("failure", ["missing", "wrong_size"])
def test_dsa_guard_rejects_lost_owner_before_replay(backend, failure):
    batch = _batch()
    guard = _guard(backend, batch)
    guard.capture(batch)
    metadata = backend.forward_metadata
    guard.restore()
    if failure == "missing":
        backend._afd_private_graph_states.pop(id(metadata))
    else:
        backend._afd_private_graph_states[id(metadata)]["batch_size"] = 7
    with pytest.raises(AFDError, match="AFD_DSA_METADATA_OWNERSHIP_LOST"):
        guard.prepare_replay(batch)


@pytest.mark.parametrize(
    "live_rows,static_rows,error",
    [
        (2, None, "STATIC_BATCH_MISSING"),
        (2, 3, "SHAPE_CHANGED"),
        (3, 2, "SHAPE_CHANGED"),
    ],
)
def test_dsa_replay_rejects_changed_or_missing_static_shape(
    backend, live_rows, static_rows, error
):
    batch = _batch()
    metadata = backend.init_forward_metadata_for_afd_capture(batch)
    original_state = backend.decode_cuda_graph_metadata
    live = _batch(tuple(range(live_rows)), (2,) * live_rows)
    static = (
        None
        if static_rows is None
        else _batch(tuple(range(static_rows)), (2,) * static_rows)
    )
    with pytest.raises(RuntimeError, match=f"AFD_DSA_{error}"):
        backend.prepare_forward_metadata_for_afd_replay(
            metadata, live, static_forward_batch=static
        )
    assert backend.decode_cuda_graph_metadata is original_state


def test_dsa_replay_rejects_metadata_from_another_owner(backend):
    batch = _batch()
    with pytest.raises(RuntimeError, match="AFD_DSA_METADATA_NOT_CAPTURED"):
        backend.prepare_forward_metadata_for_afd_replay(
            object(), batch, static_forward_batch=batch
        )


def test_dsa_replay_accepts_a_live_prefix_of_the_static_shape(backend):
    static = _batch((1, 2, 0, 0), (3, 6, 1, 1))
    metadata = backend.init_forward_metadata_for_afd_capture(static)
    storage = _tensor_storage(metadata)
    live = _batch((8,), (7,))
    static.req_pool_indices.copy_(torch.tensor([8, 0, 0, 0]))
    static.seq_lens.copy_(torch.tensor([7, 1, 1, 1]))
    backend.prepare_forward_metadata_for_afd_replay(
        metadata, live, static_forward_batch=static
    )
    _check_metadata(backend, metadata, static)
    assert _tensor_storage(metadata) == storage


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("forward_mode", ForwardMode.EXTEND, "AFD_DSA_MODE_UNSUPPORTED"),
        ("forward_mode", ForwardMode.TARGET_VERIFY, "AFD_DSA_MODE_UNSUPPORTED"),
        ("spec_info", object(), "AFD_MTP_SPECULATIVE_UNSUPPORTED"),
        ("positions", None, "AFD_DSA_SHAPE_MISMATCH"),
        ("positions", torch.arange(3), "AFD_DSA_SHAPE_MISMATCH"),
        ("batch_size", 0, "AFD_DSA_SHAPE_MISMATCH"),
    ],
)
def test_dsa_capture_rejects_unsupported_batch_before_allocating(
    backend, field, value, error
):
    batch = _batch()
    setattr(batch, field, value)
    original_state = backend.decode_cuda_graph_metadata
    original_metadata = backend.forward_metadata
    with pytest.raises(RuntimeError, match=error):
        backend.init_forward_metadata_for_afd_capture(batch)
    assert backend.decode_cuda_graph_metadata is original_state
    assert backend.forward_metadata is original_metadata
    assert not hasattr(backend, "_afd_private_graph_states")


def test_dsa_replay_rejects_capture_token_extent_drift(backend):
    batch = _batch()
    metadata = backend.init_forward_metadata_for_afd_capture(batch)
    backend._afd_private_graph_states[id(metadata)]["num_tokens"] += 1
    with pytest.raises(RuntimeError, match="AFD_DSA_SHAPE_CHANGED"):
        backend.prepare_forward_metadata_for_afd_replay(
            metadata, batch, static_forward_batch=batch
        )


@pytest.mark.parametrize("failure", ["exception", "identity"])
def test_dsa_replay_failure_restores_guard_and_backend_state(
    backend, monkeypatch, failure
):
    batch = _batch()
    guard = _guard(backend, batch)
    original_state = backend.decode_cuda_graph_metadata
    original_metadata = backend.forward_metadata
    original_stage_metadata = guard._stage.attention_metadata
    guard.capture(batch)
    captured = backend.forward_metadata
    guard.restore()
    original_apply = backend._apply_cuda_graph_metadata

    def fail_after_refresh(*args, **kwargs):
        original_apply(*args, **kwargs)
        if failure == "exception":
            raise RuntimeError("injected metadata refresh failure")
        backend.forward_metadata = dataclasses.replace(backend.forward_metadata)

    monkeypatch.setattr(backend, "_apply_cuda_graph_metadata", fail_after_refresh)
    expected = (
        "injected metadata refresh failure"
        if failure == "exception"
        else "IDENTITY_CHANGED"
    )
    with pytest.raises(RuntimeError, match=expected):
        guard.prepare_replay(_batch((5,), (4,)))
    assert backend.decode_cuda_graph_metadata is original_state
    assert backend.forward_metadata is original_metadata
    assert guard._stage.forward_batch is batch
    assert guard._stage.attention_metadata is original_stage_metadata
    assert backend.owns_afd_capture_metadata(captured, batch_size=4)
    guard.restore()
    monkeypatch.setattr(backend, "_apply_cuda_graph_metadata", original_apply)
    guard.prepare_replay(batch)
    _check_metadata(backend, captured, guard.graph_forward_batch)
    guard.restore()


@pytest.mark.parametrize("initial_metadata", [None, "existing"])
def test_dsa_capture_failure_restores_guard_and_backend_state(
    backend, monkeypatch, initial_metadata
):
    batch = _batch()
    backend.forward_metadata = initial_metadata
    original_state = backend.decode_cuda_graph_metadata
    guard = _guard(backend, batch)
    original_init = backend.init_forward_metadata_out_graph

    def fail_after_metadata(*args, **kwargs):
        original_init(*args, **kwargs)
        raise RuntimeError("injected metadata capture failure")

    monkeypatch.setattr(backend, "init_forward_metadata_out_graph", fail_after_metadata)
    with pytest.raises(RuntimeError, match="injected metadata capture failure"):
        guard.capture(batch)
    assert backend.decode_cuda_graph_metadata is original_state
    assert backend.forward_metadata is initial_metadata
    assert guard._stage.forward_batch is batch
    assert not getattr(backend, "_afd_private_graph_states", {})
    guard.restore()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))


def test_private_state_release_preserves_peer_and_is_idempotent(backend):
    import weakref

    batch = _batch()
    guards = [_guard(backend, batch) for _ in range(2)]
    for guard in guards:
        guard.capture(batch)
        guard.restore()
    metadata = guards[0]._capture_metadata
    state = backend._afd_private_graph_states[id(metadata)]["graph_state"]
    refs = [weakref.ref(metadata), weakref.ref(state["cache_seqlens"])]
    del metadata, state
    guards[0].close()
    guards[0].close()
    assert all(ref() is None for ref in refs)
    assert len(backend._afd_private_graph_states) == 1
    guards[1].prepare_replay(batch)
    guards[1].close()
    assert backend._afd_private_graph_states == {}


def test_split_is_metadata_free_and_eager_initializes_each_stage_once():
    from sglang.srt.afd.contracts import AFDRole
    from sglang.srt.afd.model_adapters.base import AFDDecoderAdapter

    calls = []
    backend = SimpleNamespace(forward_metadata=None)

    def initialize(batch):
        calls.append(batch.batch_size)
        backend.forward_metadata = object()

    backend.init_forward_metadata = initialize
    adapter = object.__new__(AFDDecoderAdapter)
    adapter.role = AFDRole.ATTENTION
    adapter.attention_backend = backend
    batch = _batch((1, 2, 3), (3, 6, 4))
    stages = adapter.split_step(
        hidden_states=torch.ones(3, 4),
        residual=None,
        positions=batch.positions,
        forward_batch=batch,
        stages=2,
    )
    assert calls == []
    for _ in range(3):
        for stage in stages:
            adapter.prepare_stage(stage=stage)
            assert backend.forward_metadata is stage.attention_metadata
    assert calls == [2, 1]
    assert stages[0].attention_metadata is not stages[1].attention_metadata


def test_repeated_warmup_activation_restores_original_batch_and_metadata(backend):
    original = backend.forward_metadata
    batch = _batch()
    guards = [_guard(backend, batch) for _ in range(2)]
    stages = [guard._stage for guard in guards]
    for guard in guards:
        guard.capture(batch)
    for _ in range(3):
        for guard in guards:
            guard.activate_in_graph()
            assert guard._stage.forward_batch is guard.graph_forward_batch
    for guard in reversed(guards):
        guard.restore()
    assert backend.forward_metadata is original
    assert all(stage.forward_batch is batch for stage in stages)
    for guard in reversed(guards):
        guard.close()
    assert backend._afd_private_graph_states == {}
