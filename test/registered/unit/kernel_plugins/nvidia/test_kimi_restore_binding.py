# SPDX-License-Identifier: Apache-2.0
"""Typed K3 cache/restore checks with real CPU tensors and the native state router.

GPU backend constructors are isolated; no CUDA or distributed operation runs.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.kernel_plugins.nvidia import binding as mb
from sglang.srt.mem_cache.kv.router import STATE_PART, KVLayerRouter
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


@pytest.fixture
def pools(monkeypatch):
    class HybridBackend:
        pass

    class FullBackend:
        pass

    class LinearBackend:
        pass

    class SplitBackend:
        spec_attn_is_decode = True
        spec_attn_is_prefill = False

        def _select_backend(self, mode):
            return self.decode_backend if mode.is_decode() else self.prefill_backend

    class TokenPool:
        pass

    class RequestPool:
        pass

    class StatePool:
        class SpeculativeState:
            pass

    modules = {
        "sglang.srt.layers.attention.hybrid_linear_attn_backend": dict(
            HybridLinearAttnBackend=HybridBackend
        ),
        "sglang.srt.layers.attention.trtllm_mla_backend": dict(
            TRTLLMMLABackend=FullBackend
        ),
        "sglang.srt.layers.attention.hybrid_attn_backend": dict(
            HybridAttnBackend=SplitBackend
        ),
        "sglang.srt.layers.attention.linear.kda_backend": dict(
            KDAAttnBackend=LinearBackend
        ),
        "sglang.srt.mem_cache.memory_pool": dict(
            HybridLinearKVPool=TokenPool,
            HybridReqToTokenPool=RequestPool,
            MambaPool=StatePool,
        ),
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    def construct(*, split=False):
        waits = []
        counter = SimpleNamespace(consumer_index=0)
        counter.wait_until = lambda layer: waits.append((layer, counter.consumer_index))
        token = TokenPool()
        raw = [torch.zeros(128, 1, 576, dtype=torch.uint8)]
        full_pool = SimpleNamespace(
            page_size=64,
            dtype=torch.float8_e4m3fn,
            store_dtype=torch.uint8,
            _write_loc_dcp_span=8,
            kv_buffer=raw,
        )
        token.use_mla = True
        token.full_kv_pool, token.size = full_pool, 64
        token.post_capture_active = False
        token.full_attention_layer_id_mapping = {0: 0}
        token.layer_transfer_counter, token.start_layer = counter, 0

        def get_key(layer):
            counter.wait_until(layer)
            return full_pool.kv_buffer[0].view(full_pool.dtype)

        token.get_key_buffer = get_key
        state = SimpleNamespace(
            conv=[torch.zeros(4, 3, 4608, dtype=torch.bfloat16)],
            temporal=torch.zeros(4, 12, 128, 128),
            replayssm_d=None,
            replayssm_rawv=None,
        )
        request = RequestPool()
        request.mamba_map = {1: 0}
        request.mamba_pool = SimpleNamespace(
            size=3,
            mamba2_layer_cache=lambda layer: state,
        )
        router = KVLayerRouter.from_layer_maps(token, [])
        router.bind_part(
            sub_pool="state",
            part=STATE_PART,
            view=request.mamba_pool,
            indices=request.mamba_map,
        )
        router.load_parts_layer_wise()
        token.layer_router = router
        full, linear, backend = FullBackend(), LinearBackend(), HybridBackend()
        for child in (full, linear, backend):
            child.token_to_kv_pool, child.req_to_token_pool = token, request
        full.forward_decode_metadata = SimpleNamespace(
            block_kv_indices=torch.zeros(2, 4, dtype=torch.int32),
            seq_lens_k=torch.tensor([0, 1], dtype=torch.int32),
            global_seq_lens_k=torch.tensor([1, 8], dtype=torch.int32),
        )
        linear.state_layer_cache = lambda layer: router.read(
            layer, STATE_PART
        ).view.mamba2_layer_cache(0)
        linear.forward_metadata = SimpleNamespace(
            mamba_cache_indices=torch.tensor([3, 1], dtype=torch.int32),
        )
        linear._track_mamba_state_decode = Mock()
        owner = full
        if split:
            owner = SplitBackend()
            owner.token_to_kv_pool, owner.req_to_token_pool = token, request
            owner.decode_backend = full
            owner.prefill_backend = FullBackend()
            owner.prefill_backend.token_to_kv_pool = token
            owner.prefill_backend.req_to_token_pool = request
        backend.full_attn_backend, backend.linear_attn_backend = owner, linear
        backend.attn_backend_list = [owner, linear]
        runner = SimpleNamespace(attn_backend=backend)
        binding = mb.MegakernelBinding(model=torch.nn.Module())
        monkeypatch.setattr(
            mb, "get_parallel", lambda: SimpleNamespace(attn_dcp_rank=3)
        )
        prepared = binding.preparation_view(runner)
        return SimpleNamespace(
            binding=binding,
            runner=runner,
            prepared=prepared,
            token=token,
            state=state,
            full_pool=full_pool,
            full=full,
            owner=owner,
            linear=linear,
            waits=waits,
            counter=counter,
        )

    return construct


def test_prepared_write_topology_and_replay_follow_the_live_restore_router(pools):
    f = pools()
    assert (
        f.prepared.hybrid.mla.write_shard_size,
        f.prepared.hybrid.mla.write_shard_rank,
    ) == (8, 3)
    for consumer in (0, 2):
        f.waits.clear()
        f.counter.consumer_index = consumer
        f.binding.prepare_replay()
        assert {(0, consumer), (1, consumer)} <= set(f.waits)
        assert all(index == consumer for _, index in f.waits)


@pytest.mark.parametrize("which", ["mla", "kda"])
def test_replay_rejects_storage_replacement_before_using_the_graph(pools, which):
    f = pools()
    if which == "mla":
        f.full_pool.kv_buffer[0] = f.full_pool.kv_buffer[0].clone()
    else:
        f.state.conv[0] = f.state.conv[0].clone()
    with pytest.raises(RuntimeError, match="mla_cache|kda_conv"):
        f.binding.prepare_replay()


def test_split_backend_uses_and_seals_the_native_per_forward_metadata(pools):
    f = pools(split=True)
    metadata = SimpleNamespace(
        block_kv_indices=torch.zeros(2, 4, dtype=torch.int32),
        seq_lens_k=torch.tensor([0, 2], dtype=torch.int32),
        global_seq_lens_k=torch.tensor([1, 16], dtype=torch.int32),
    )
    batch = SimpleNamespace(
        batch_size=2,
        out_cache_loc=torch.tensor([11, 19]),
        out_cache_loc_is_physical=True,
        decode_trtllm_mla_metadata=metadata,
        forward_mode=SimpleNamespace(
            is_decode=lambda: True, is_target_verify=lambda: False
        ),
    )
    view = mb.ForwardView(
        f.runner.attn_backend,
        hybrid=f.binding._hybrid_step(f.runner.attn_backend, batch),
        legacy=False,
    )
    assert view.hybrid.mla.sequence_lengths is metadata.seq_lens_k
    assert view.hybrid.mla.global_sequence_lengths is metadata.global_seq_lens_k
    f.binding.seal_capture_view(batch, view)
    metadata.global_seq_lens_k = metadata.global_seq_lens_k.clone()
    with pytest.raises(RuntimeError, match="global_sequence_lengths"):
        f.binding.prepare_replay()


def test_split_backend_child_replacement_is_refused_at_eager_and_replay(pools):
    f = pools(split=True)
    f.owner.decode_backend = object()
    with pytest.raises(RuntimeError, match="mla_backend"):
        f.binding._current_hybrid_sources()
    with pytest.raises(RuntimeError, match="mla_backend"):
        f.binding.prepare_replay()


def test_replay_hook_keeps_absent_and_other_bindings_unchanged():
    mb.prepare_megakernel_replay(torch.nn.Module())
    other = SimpleNamespace(prepare_replay=Mock())
    model = torch.nn.Module()
    model.__dict__["_megakernel"] = SimpleNamespace(binding=other)
    mb.prepare_megakernel_replay(model)
    other.prepare_replay.assert_not_called()


@pytest.mark.parametrize("q", [1, 6])
@pytest.mark.parametrize("configured", [None, [24, 37, 96]])
def test_install_keeps_request_capture_buckets_distinct_from_token_rows(
    monkeypatch, q, configured
):
    configured = None if configured is None else list(configured)
    contexts = []

    def build(model, hf_config, context):
        contexts.append(context)
        return SimpleNamespace(
            widths=context.widths,
            draft_sampler=None,
            draft_loop=None,
            draft_extend=None,
            begin_draft_round=None,
            graph_variants=None,
            speculative_runtime_hooks=None,
            summary=lambda: "CPU capture-width contract",
            plan=lambda *args, **kwargs: None,
        )

    spec = SimpleNamespace(
        name="CPU capture-width contract",
        max_rows=64,
        build=build,
        prepare=None,
        numerics=SimpleNamespace(kind=SimpleNamespace(value="test")),
    )
    graph = SimpleNamespace(decode=SimpleNamespace(bs=configured))
    monkeypatch.setattr(mb, "megakernel_enabled", lambda: True)
    monkeypatch.setattr(mb, "_find_spec_on_every_rank", lambda **kwargs: spec)
    monkeypatch.setattr(mb, "_requirement_reasons", lambda **kwargs: [])
    monkeypatch.setattr(mb, "_model_disable_reasons", lambda **kwargs: [])
    monkeypatch.setattr(mb, "_gather_reasons", lambda reasons: reasons)
    monkeypatch.setattr(mb, "_tokens_per_request", lambda: q)
    monkeypatch.setattr(mb, "get_device", lambda: SimpleNamespace(watchdog_timeout=30))
    monkeypatch.setattr(
        mb,
        "get_exec",
        lambda: SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=graph)),
    )
    monkeypatch.setattr(mb, "megakernel_agree", lambda *args, **kwargs: None)
    monkeypatch.setattr(mb, "_decode_graphs_enabled", lambda: False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (0, 0))
    mb.install_megakernel(
        model=torch.nn.Module(),
        hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"]),
        capture_tail_hooks=[],
        is_draft=False,
    )
    context = contexts[0]
    assert context.tokens_per_request == q
    assert context.widths == tuple(range(q, 65, q))
    assert context.capture_widths == (None if configured is None else (24, 37, 96))
    if configured is not None:
        configured.append(48)
        assert context.capture_widths == (24, 37, 96)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
