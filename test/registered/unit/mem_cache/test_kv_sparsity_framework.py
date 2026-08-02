from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.kv_sparsity_hook import validate_kv_cache_sparsity
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.sparsity.backend.visibility_adaptor import (
    FlashAttentionVisibilityAdaptor,
    HBMResidentPlacement,
)
from sglang.srt.mem_cache.sparsity.config import KVSparsityConfig
from sglang.srt.mem_cache.sparsity.contracts import (
    Granularity,
    RequestIdentity,
    SelectionContext,
    SelectionResult,
)
from sglang.srt.mem_cache.sparsity.core.kv_sparsity_controller import (
    KVSparsityController,
)
from sglang.srt.mem_cache.sparsity.factory import parse_kv_sparsity_config
from sglang.srt.mem_cache.sparsity.policies.streaming_llm import (
    StreamingLLMPolicy,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_parse_kv_sparsity_config_is_independent_from_hisparse():
    args = SimpleNamespace(
        page_size=1,
        kv_cache_sparsity_config=(
            '{"policy":"streaming_llm","backend":"flashattention",'
            '"min_sparse_tokens":16,"policy_config":{'
            '"sink_pages":2,"recent_pages":4}}'
        ),
    )

    config = parse_kv_sparsity_config(args)

    assert config.policy == "streaming_llm"
    assert config.backend == "fa3"
    assert config.page_size == 1
    assert config.min_sparse_tokens == 16
    assert config.policy_config == {"sink_pages": 2, "recent_pages": 4}


def test_parse_kv_sparsity_config_rejects_unknown_fields():
    args = SimpleNamespace(
        page_size=1,
        kv_cache_sparsity_config='{"policy":"streaming_llm","evict":true}',
    )
    with pytest.raises(ValueError, match="Unknown KV sparsity config"):
        parse_kv_sparsity_config(args)


def test_validation_uses_resolved_fa3_default_and_disables_graphs(monkeypatch):
    hf_text_config = SimpleNamespace(num_kv_shared_layers=0)
    hf_config = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        model_type="qwen3",
        get_text_config=lambda: hf_text_config,
    )
    model_config = SimpleNamespace(
        attention_arch=AttentionArch.MHA,
        is_encoder_decoder=False,
        is_multimodal=False,
        is_generation=True,
        num_attention_layers=2,
        num_hidden_layers=2,
        sliding_window_size=None,
        is_hybrid_swa=False,
        attention_chunk_size=None,
        hf_text_config=hf_text_config,
        hf_config=hf_config,
        is_draft_model=False,
        linear_attn_registry_result=None,
    )
    args = SimpleNamespace(
        enable_kv_cache_sparsity=True,
        kv_cache_sparsity_config=None,
        page_size=1,
        enable_hisparse=False,
        attention_backend=None,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        speculative_algorithm=None,
        dllm_algorithm=None,
        enable_pdmux=False,
        disaggregation_mode="null",
        enable_two_batch_overlap=False,
        enable_mixed_chunk=False,
        enable_torch_compile=False,
        enable_dp_attention=False,
        enable_prefill_cp=False,
        attn_cp_size=1,
        dcp_size=1,
        cuda_graph_config=CudaGraphConfig(),
        _cuda_graph_config_locked=set(),
        _resolved_overrides=[("test", {"attention_backend": "fa3"})],
        get_model_config=lambda: model_config,
    )
    monkeypatch.setattr("sglang.srt.server_args.is_cuda", lambda: True)
    monkeypatch.setattr("sglang.srt.server_args.is_hip", lambda: False)

    validate_kv_cache_sparsity(args)

    assert args.cuda_graph_config[Phase.DECODE].backend == Backend.DISABLED
    assert args.cuda_graph_config[Phase.PREFILL].backend == Backend.DISABLED


def test_streaming_llm_rejects_unknown_policy_fields():
    config = KVSparsityConfig(policy_config={"recent_pages": 4, "typo": 1})
    with pytest.raises(ValueError, match="Unknown StreamingLLM policy field"):
        StreamingLLMPolicy(config, torch.device("cpu"))


def test_streaming_llm_selects_sink_and_recent_pages_without_host_reads():
    config = KVSparsityConfig(
        page_size=1,
        min_sparse_tokens=6,
        policy_config={"sink_pages": 2, "recent_pages": 3},
    )
    policy = StreamingLLMPolicy(config, torch.device("cpu"))
    seq_lens = torch.tensor([4, 6, 10], dtype=torch.int64)

    result = policy.select(
        SelectionContext(
            query=None,
            layer_id=0,
            req_pool_indices=torch.tensor([1, 2, 3]),
            seq_lens=seq_lens,
        )
    )

    assert result.granularity == Granularity.PAGE
    assert result.capacity == 5
    torch.testing.assert_close(result.sparse_mask, torch.tensor([False, True, True]))
    torch.testing.assert_close(
        result.logical_indices,
        torch.tensor(
            [
                [-1, -1, -1, -1, -1],
                [0, 1, 3, 4, 5],
                [0, 1, 7, 8, 9],
            ],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        result.valid_lengths, torch.tensor([0, 5, 5], dtype=torch.int32)
    )
    torch.testing.assert_close(
        result.visible_kv_lens, torch.tensor([0, 5, 5], dtype=torch.int32)
    )


def test_streaming_llm_accounts_for_a_partial_final_page():
    config = KVSparsityConfig(
        page_size=4,
        min_sparse_tokens=0,
        policy_config={"sink_pages": 1, "recent_pages": 2},
    )
    policy = StreamingLLMPolicy(config, torch.device("cpu"))

    result = policy.select(
        SelectionContext(
            query=None,
            layer_id=0,
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([12, 13]),
        )
    )

    torch.testing.assert_close(result.sparse_mask, torch.tensor([False, True]))
    torch.testing.assert_close(
        result.logical_indices[1], torch.tensor([0, 2, 3], dtype=torch.int32)
    )
    assert result.visible_kv_lens[1].item() == 9


def test_selection_result_rejects_mismatched_batch_shapes():
    with pytest.raises(ValueError, match="valid_lengths"):
        SelectionResult(
            granularity=Granularity.PAGE,
            logical_indices=torch.zeros((2, 4), dtype=torch.int32),
            valid_lengths=torch.zeros(1, dtype=torch.int32),
            visible_kv_lens=torch.zeros(2, dtype=torch.int32),
            sparse_mask=torch.zeros(2, dtype=torch.bool),
        )


def test_fa3_adaptor_rewrites_sparse_rows_and_restores_dense_metadata():
    req_to_token = torch.zeros((3, 8), dtype=torch.int32)
    req_to_token[1] = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])
    req_to_token[2] = torch.tensor([20, 21, 22, 23, 24, 25, 26, 27])
    placement = HBMResidentPlacement(req_to_token=req_to_token, page_size=1)
    adaptor = FlashAttentionVisibilityAdaptor(placement)
    metadata = SimpleNamespace(
        page_table=torch.tensor(
            [[10, 11, 12, 13, 14, 15, 16, 17], [20, 21, 22, 23, 24, 25, 26, 27]],
            dtype=torch.int32,
        ),
        cache_seqlens_int32=torch.tensor([8, 4], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 8, 12], dtype=torch.int32),
        max_seq_len_k=8,
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        scheduler_metadata=torch.ones(1),
    )
    dense_page_table = metadata.page_table.clone()
    forward_batch = SimpleNamespace(req_pool_indices=torch.tensor([1, 2]))
    result = SelectionResult(
        granularity=Granularity.PAGE,
        logical_indices=torch.tensor([[0, 1, 6, 7], [-1, -1, -1, -1]]),
        valid_lengths=torch.tensor([4, 0], dtype=torch.int32),
        visible_kv_lens=torch.tensor([4, 0], dtype=torch.int32),
        sparse_mask=torch.tensor([True, False]),
        max_visible_kv_len=4,
    )

    adaptor.capture_dense_metadata(metadata)
    adaptor.apply(result, metadata, forward_batch)

    torch.testing.assert_close(
        metadata.page_table[0, :4],
        torch.tensor([10, 11, 16, 17], dtype=torch.int32),
    )
    torch.testing.assert_close(metadata.page_table[1], dense_page_table[1])
    torch.testing.assert_close(
        metadata.cache_seqlens_int32, torch.tensor([4, 4], dtype=torch.int32)
    )
    torch.testing.assert_close(
        metadata.cu_seqlens_k, torch.tensor([0, 4, 8], dtype=torch.int32)
    )
    assert metadata.max_seq_len_k == 4
    assert metadata.scheduler_metadata is None

    adaptor.restore_dense_metadata(metadata)
    torch.testing.assert_close(metadata.page_table, dense_page_table)
    torch.testing.assert_close(
        metadata.cache_seqlens_int32, torch.tensor([8, 4], dtype=torch.int32)
    )
    torch.testing.assert_close(
        metadata.cu_seqlens_k, torch.tensor([0, 8, 12], dtype=torch.int32)
    )
    assert metadata.max_seq_len_k == 8


def test_fa3_adaptor_rebuilds_sparse_scheduler_metadata_once_per_step():
    req_to_token = torch.arange(8, dtype=torch.int32).unsqueeze(0)
    adaptor = FlashAttentionVisibilityAdaptor(
        HBMResidentPlacement(req_to_token=req_to_token, page_size=1)
    )
    calls = []

    def build_scheduler(batch_size, max_seq_len_k, cache_seqlens, cu_seqlens_q):
        calls.append(
            (
                batch_size,
                max_seq_len_k,
                cache_seqlens.clone(),
                cu_seqlens_q.clone(),
            )
        )
        return torch.tensor([17])

    adaptor.bind_attention_backend(
        SimpleNamespace(_compute_scheduler_metadata=build_scheduler)
    )
    metadata = SimpleNamespace(
        page_table=req_to_token.clone(),
        cache_seqlens_int32=torch.tensor([8], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 8], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len_k=8,
        scheduler_metadata=torch.tensor([99]),
    )
    result = SelectionResult(
        granularity=Granularity.PAGE,
        logical_indices=torch.tensor([[0, 1, 6, 7]], dtype=torch.int32),
        valid_lengths=torch.tensor([4], dtype=torch.int32),
        visible_kv_lens=torch.tensor([4], dtype=torch.int32),
        sparse_mask=torch.tensor([True]),
        max_visible_kv_len=4,
    )
    forward_batch = SimpleNamespace(req_pool_indices=torch.tensor([0]))

    adaptor.capture_dense_metadata(metadata)
    adaptor.apply(result, metadata, forward_batch)
    adaptor.apply(result, metadata, forward_batch)

    assert len(calls) == 1
    assert calls[0][0:2] == (1, 4)
    torch.testing.assert_close(calls[0][2], torch.tensor([4], dtype=torch.int32))
    torch.testing.assert_close(metadata.scheduler_metadata, torch.tensor([17]))

    adaptor.restore_dense_metadata(metadata)
    assert metadata.max_seq_len_k == 8
    torch.testing.assert_close(metadata.scheduler_metadata, torch.tensor([99]))


def test_fa3_adaptor_keeps_short_dense_rows_when_capacity_exceeds_page_table():
    req_to_token = torch.arange(4, dtype=torch.int32).unsqueeze(0)
    adaptor = FlashAttentionVisibilityAdaptor(
        HBMResidentPlacement(req_to_token=req_to_token, page_size=1)
    )
    metadata = SimpleNamespace(
        page_table=req_to_token.clone(),
        cache_seqlens_int32=torch.tensor([4], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 4], dtype=torch.int32),
        max_seq_len_k=4,
    )
    result = SelectionResult(
        granularity=Granularity.PAGE,
        logical_indices=torch.full((1, 8), -1, dtype=torch.int32),
        valid_lengths=torch.tensor([0], dtype=torch.int32),
        visible_kv_lens=torch.tensor([0], dtype=torch.int32),
        sparse_mask=torch.tensor([False]),
        max_visible_kv_len=8,
    )

    adaptor.capture_dense_metadata(metadata)
    adaptor.apply(
        result,
        metadata,
        SimpleNamespace(req_pool_indices=torch.tensor([0])),
    )

    torch.testing.assert_close(metadata.page_table, req_to_token)
    torch.testing.assert_close(
        metadata.cache_seqlens_int32, torch.tensor([4], dtype=torch.int32)
    )
    assert metadata.max_seq_len_k == 4


def test_hbm_placement_maps_logical_pages_to_physical_pages():
    req_to_token = torch.zeros((2, 16), dtype=torch.int32)
    req_to_token[1] = torch.arange(40, 56, dtype=torch.int32)
    placement = HBMResidentPlacement(req_to_token=req_to_token, page_size=4)

    physical = placement.logical_to_physical_pages(
        logical_pages=torch.tensor([[0, 2, 3, -1]], dtype=torch.int32),
        req_pool_indices=torch.tensor([1]),
    )

    torch.testing.assert_close(
        physical, torch.tensor([[10, 12, 13, 0]], dtype=torch.int32)
    )


def test_controller_reuses_step_selection_and_restores_after_layer_range():
    class CountingStreamingPolicy(StreamingLLMPolicy):
        def __init__(self, config, device):
            super().__init__(config, device)
            self.select_calls = 0

        def select(self, context):
            self.select_calls += 1
            return super().select(context)

    config = KVSparsityConfig(
        page_size=1,
        min_sparse_tokens=0,
        policy_config={"sink_pages": 1, "recent_pages": 2},
    )
    req_to_token = torch.zeros((2, 8), dtype=torch.int32)
    req_to_token[1] = torch.arange(10, 18, dtype=torch.int32)
    pool = SimpleNamespace(
        req_to_token=req_to_token,
        req_generation=torch.tensor([0, 1], dtype=torch.int64),
    )
    adaptor = FlashAttentionVisibilityAdaptor(
        HBMResidentPlacement(req_to_token=req_to_token, page_size=1)
    )
    policy = CountingStreamingPolicy(config, torch.device("cpu"))
    controller = KVSparsityController(
        config=config,
        policy=policy,
        adaptor=adaptor,
        req_to_token_pool=pool,
        start_layer=0,
        end_layer=2,
    )
    metadata = SimpleNamespace(
        page_table=req_to_token[1:2].clone(),
        cache_seqlens_int32=torch.tensor([8], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 8], dtype=torch.int32),
        max_seq_len_k=8,
        scheduler_metadata=None,
    )
    forward_batch = SimpleNamespace(
        req_pool_indices=torch.tensor([1]),
        seq_lens=torch.tensor([8]),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
    )
    query = torch.zeros((1, 1, 1))
    layer_0 = SimpleNamespace(layer_id=0)
    layer_1 = SimpleNamespace(layer_id=1)

    controller.before_attention(query, None, None, layer_0, forward_batch, metadata)
    controller.after_attention(
        query, None, None, query, layer_0, forward_batch, metadata
    )
    controller.before_attention(query, None, None, layer_1, forward_batch, metadata)

    assert policy.select_calls == 1
    torch.testing.assert_close(
        metadata.page_table[0, :3],
        torch.tensor([10, 16, 17], dtype=torch.int32),
    )
    controller.after_attention(
        query, None, None, query, layer_1, forward_batch, metadata
    )
    torch.testing.assert_close(metadata.page_table, req_to_token[1:2])
    torch.testing.assert_close(
        metadata.cache_seqlens_int32, torch.tensor([8], dtype=torch.int32)
    )


def test_controller_observes_prefill_without_rewriting_dense_metadata():
    class TrackingStreamingPolicy(StreamingLLMPolicy):
        def __init__(self, config, device):
            super().__init__(config, device)
            self.forward_contexts = []
            self.completed_contexts = []

        def begin_forward(self, context):
            self.forward_contexts.append(context)

        def on_attention_complete(self, context):
            self.completed_contexts.append(context)

    config = KVSparsityConfig(
        min_sparse_tokens=0,
        policy_config={"sink_pages": 1, "recent_pages": 2},
    )
    req_to_token = torch.zeros((2, 8), dtype=torch.int32)
    req_to_token[1] = torch.arange(10, 18, dtype=torch.int32)
    policy = TrackingStreamingPolicy(config, torch.device("cpu"))
    controller = KVSparsityController(
        config=config,
        policy=policy,
        adaptor=FlashAttentionVisibilityAdaptor(
            HBMResidentPlacement(req_to_token=req_to_token, page_size=1)
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=req_to_token,
            req_generation=torch.tensor([0, 1], dtype=torch.int64),
        ),
        start_layer=0,
        end_layer=1,
    )
    metadata = SimpleNamespace(page_table=req_to_token[1:2].clone())
    dense_page_table = metadata.page_table.clone()
    forward_batch = SimpleNamespace(
        req_pool_indices=torch.tensor([1]),
        seq_lens=torch.tensor([8]),
        forward_mode=SimpleNamespace(is_decode=lambda: False),
    )
    query = torch.zeros((8, 1, 1))
    key = torch.ones((8, 1, 1))
    output = torch.full((8, 1, 1), 2.0)
    layer = SimpleNamespace(layer_id=0)

    controller.before_attention(query, key, None, layer, forward_batch, metadata)
    controller.after_attention(query, key, None, output, layer, forward_batch, metadata)

    torch.testing.assert_close(metadata.page_table, dense_page_table)
    assert len(policy.forward_contexts) == 1
    assert len(policy.completed_contexts) == 1
    assert policy.completed_contexts[0].key is key
    assert policy.completed_contexts[0].output is output


def test_controller_resets_state_when_a_pool_slot_generation_changes():
    class StatefulStreamingPolicy(StreamingLLMPolicy):
        _CAPABILITIES = replace(
            StreamingLLMPolicy._CAPABILITIES, requires_request_state=True
        )

        def __init__(self, config, device):
            super().__init__(config, device)
            self.begun = []
            self.ended = []

        def on_request_begin(self, identity):
            self.begun.append(identity)

        def on_request_end(self, identity):
            self.ended.append(identity)

    config = KVSparsityConfig(
        min_sparse_tokens=0,
        policy_config={"sink_pages": 1, "recent_pages": 2},
    )
    req_to_token = torch.zeros((2, 8), dtype=torch.int32)
    req_to_token[1] = torch.arange(10, 18, dtype=torch.int32)
    pool = SimpleNamespace(
        req_to_token=req_to_token,
        req_generation=torch.tensor([0, 7], dtype=torch.int64),
    )
    policy = StatefulStreamingPolicy(config, torch.device("cpu"))
    controller = KVSparsityController(
        config=config,
        policy=policy,
        adaptor=FlashAttentionVisibilityAdaptor(
            HBMResidentPlacement(req_to_token=req_to_token, page_size=1)
        ),
        req_to_token_pool=pool,
        start_layer=0,
        end_layer=1,
    )
    forward_batch = SimpleNamespace(
        req_pool_indices=torch.tensor([1]),
        seq_lens=torch.tensor([8]),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
    )
    query = torch.zeros((1, 1, 1))
    layer = SimpleNamespace(layer_id=0)

    def run_forward():
        metadata = SimpleNamespace(
            page_table=req_to_token[1:2].clone(),
            cache_seqlens_int32=torch.tensor([8], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 8], dtype=torch.int32),
            max_seq_len_k=8,
            scheduler_metadata=None,
        )
        controller.before_attention(query, None, None, layer, forward_batch, metadata)
        controller.after_attention(
            query, None, None, query, layer, forward_batch, metadata
        )

    run_forward()
    pool.req_generation[1] = 8
    run_forward()

    assert policy.begun == [RequestIdentity(1, 7), RequestIdentity(1, 8)]
    assert policy.ended == [RequestIdentity(1, 7)]


@pytest.mark.parametrize("is_decode", [False, True])
def test_radix_attention_runs_sparsity_hooks_around_existing_backend(is_decode):
    calls = []
    metadata = object()

    class Backend:
        forward_metadata = metadata

        def forward(self, query, key, value, layer, batch, save_kv_cache, **kwargs):
            calls.append(("backend", key, value))
            return query + 1

    class Controller:
        def before_attention(self, query, key, value, layer, batch, attn_metadata):
            calls.append(("before", attn_metadata))

        def after_attention(
            self, query, key, value, output, layer, batch, attn_metadata
        ):
            calls.append(("after", output, attn_metadata))

    class ForwardMode:
        def is_decode(self):
            return is_decode

        def is_extend(self, include_draft_extend_v2=False):
            return not is_decode

    layer = RadixAttention(
        num_heads=2,
        head_dim=4,
        scaling=0.5,
        num_kv_heads=1,
        layer_id=0,
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode(),
        spec_info=None,
        req_pool_indices=torch.tensor([1]),
        seq_lens=torch.tensor([8]),
    )
    query = torch.zeros((1, 2, 4))
    key = torch.ones((1, 1, 4))
    value = torch.full((1, 1, 4), 2.0)

    with forward_context(
        ForwardContext(attn_backend=Backend(), kv_sparsity_controller=Controller())
    ):
        output = layer(query, key, value, batch)

    assert [call[0] for call in calls] == ["before", "backend", "after"]
    assert calls[0][1] is metadata
    assert calls[1][1].shape == (1, 1, 4)
    torch.testing.assert_close(output, query + 1)
    assert calls[2][1] is output


def test_radix_attention_dense_path_does_not_require_a_controller():
    class Backend:
        def forward(self, query, key, value, layer, batch, save_kv_cache, **kwargs):
            return query + 3

    class ForwardMode:
        def is_decode(self):
            return True

        def is_extend(self, include_draft_extend_v2=False):
            return False

    layer = RadixAttention(
        num_heads=2,
        head_dim=4,
        scaling=0.5,
        num_kv_heads=1,
        layer_id=0,
    )
    batch = SimpleNamespace(forward_mode=ForwardMode())
    query = torch.zeros((1, 2, 4))
    key = torch.ones((1, 1, 4))
    value = torch.full((1, 1, 4), 2.0)

    with forward_context(ForwardContext(attn_backend=Backend())):
        output = layer(query, key, value, batch)

    torch.testing.assert_close(output, query + 3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
