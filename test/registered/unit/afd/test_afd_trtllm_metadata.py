"""CPU ownership checks for interleaved TRTLLM state; no CUDA kernel emulation."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import trtllm_mha_backend as trtllm
from sglang.srt.layers.attention.trtllm_mha_interleaved import (
    TRTLLMInterleavedGraphMetadata,
)
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


def test_common_adapter_selects_trtllm_without_model_binding_or_query_mutation():
    from sglang.srt.afd.contracts import AFDRole
    from sglang.srt.afd.model_adapters.base import AFDDecoderAdapter

    backend = _backend()
    backend.prefill_attention_backend_str = "trtllm_mha"
    backend.decode_attention_backend_str = "trtllm_mha"
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[], config=SimpleNamespace(hidden_size=4))
    )
    attention = AFDDecoderAdapter(
        role=AFDRole.ATTENTION, model=model, attention_backend=backend
    )
    ffn = AFDDecoderAdapter(role=AFDRole.FFN, model=model, attention_backend=None)
    expected = attention.attention_capability(configured_backend="trtllm_mha")
    assert ffn.attention_capability(configured_backend="trtllm_mha") == expected
    with pytest.raises(RuntimeError, match="CONFIG_UNSUPPORTED"):
        ffn.attention_capability(configured_backend="nsa")

    def init_metadata(batch):
        backend.forward_metadata = trtllm.TRTLLMMHAMetadata()

    backend.init_forward_metadata = init_metadata
    stages = [
        SimpleNamespace(index=index, attention_metadata=None, forward_batch=_batch())
        for index in range(2)
    ]
    for stage in stages:
        attention.prepare_stage(stage=stage)
    first, second = (stage.attention_metadata for stage in stages)
    assert first.workspace_buffer is not second.workspace_buffer
    assert first.multi_ctas_kv_counter_buffer is not second.multi_ctas_kv_counter_buffer
    attention.prepare_stage(stage=stages[0])
    assert backend.forward_metadata is first
    assert backend._active_workspace_buffer() is first.workspace_buffer
    assert backend._active_kv_counter_buffer() is first.multi_ctas_kv_counter_buffer
    ffn.prepare_stage(stage=SimpleNamespace(index=0))


def test_sparse_backends_keep_stage_scratch_private_and_native_fallback(monkeypatch):
    from sglang.srt.layers.attention import deepseek_v4_trtllm_backend as v4
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.workspace_buffer = torch.zeros(8)
    backend._multi_ctas_kv_counter_buffer = torch.zeros(4)
    first, second, bucket = (SimpleNamespace() for _ in range(3))
    for metadata, index in ((first, 0), (second, 1), (bucket, 0)):
        backend.bind_interleaved_stage_workspace(metadata, stage_index=index)
    assert first.interleaved_stage_scratch is bucket.interleaved_stage_scratch
    for a, b in zip(first.interleaved_stage_scratch, second.interleaved_stage_scratch):
        assert a.data_ptr() != b.data_ptr()
    backend.forward_metadata = first
    assert backend._active_workspace_buffer() is first.interleaved_stage_scratch[0]
    backend.forward_metadata = SimpleNamespace()
    assert backend._active_workspace_buffer() is backend.workspace_buffer

    sparse = object.__new__(v4.DeepseekV4TrtllmAttnBackend)
    monkeypatch.setattr(
        v4, "trtllm_batch_decode_sparse_mla_dsv4", lambda **kw: kw, raising=False
    )
    sparse.forward_metadata = SimpleNamespace(
        trtllm_stage_scratch=first.interleaved_stage_scratch
    )
    call = sparse._call_trtllm(workspace_buffer=backend.workspace_buffer)
    assert call["workspace_buffer"] is first.interleaved_stage_scratch[0]
    assert call["multi_ctas_kv_counter_buffer"] is first.interleaved_stage_scratch[1]
    sparse.forward_metadata = SimpleNamespace()
    assert sparse._call_trtllm(workspace_buffer=backend.workspace_buffer) == {
        "workspace_buffer": backend.workspace_buffer
    }
