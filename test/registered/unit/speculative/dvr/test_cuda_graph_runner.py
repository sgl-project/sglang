from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.speculative.dvr.cuda_graph_runner as graph_module
from sglang.srt.layers.attention.dvr.gdn_backend import DVRGDNAttnBackend
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.speculative.dvr.cuda_graph_runner import (
    DVRDraftDecodeCudaGraphRunner,
    DVRTargetVerifyCudaGraphRunner,
    _draft_custom_allreduce_enabled,
    _fast_decode_overrides,
    _resolve_dvr_backends,
    dvr_draft_decode_context,
    validate_dvr_attention_backend,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize("draft_custom_all_reduce", [True, False])
def test_draft_capture_is_fast_and_restores_target_state(
    monkeypatch, draft_custom_all_reduce
):
    events = []

    backend = SimpleNamespace(enable_deterministic=True)
    server_args = SimpleNamespace(
        enable_deterministic_inference=True,
        dvr_enable_draft_custom_all_reduce=draft_custom_all_reduce,
        flashinfer_allreduce_fusion_backend=None,
    )
    global_server_args = SimpleNamespace(enable_deterministic_inference=True)
    model_runner = SimpleNamespace(
        attn_backend=backend,
        server_args=server_args,
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
    )

    class FakeEnv:
        def __init__(self):
            self.value = True
            self.present = True

        def is_set(self):
            return self.present

        def get(self):
            return self.value

        def set(self, value):
            self.value = value
            events.append(("deterministic_env", value))

        def clear(self):
            self.present = False

    class FakeGroup:
        pass

    @contextmanager
    def custom_allreduce_enabled(_group, **kwargs):
        events.append(("custom_all_reduce", True, kwargs))
        try:
            yield True
        finally:
            events.append(("custom_all_reduce", False, kwargs))

    monkeypatch.setattr(
        graph_module,
        "envs",
        SimpleNamespace(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=FakeEnv()),
    )
    monkeypatch.setattr(graph_module, "get_server_args", lambda: global_server_args)
    monkeypatch.setattr(
        graph_module,
        "_clear_draft_kernel_policy_caches",
        lambda: events.append(("kernel_policy_cache", "clear")),
    )
    monkeypatch.setattr(
        graph_module,
        "_fast_decode_overrides",
        lambda *_args, **_kwargs: [(backend, "enable_deterministic", False)],
    )
    monkeypatch.setattr(
        graph_module, "_iter_decode_custom_all_reduce_groups", lambda _: [FakeGroup()]
    )
    monkeypatch.setattr(
        graph_module, "_draft_custom_allreduce_enabled", custom_allreduce_enabled
    )
    import sglang.srt.batch_invariant_ops as batch_invariant_ops

    batch_invariant_state = {"enabled": True}
    monkeypatch.setattr(
        batch_invariant_ops,
        "is_batch_invariant_mode_enabled",
        lambda: batch_invariant_state["enabled"],
    )

    def disable_batch_invariant_mode():
        batch_invariant_state["enabled"] = False
        events.append(("batch_invariant", False))

    def enable_batch_invariant_mode():
        batch_invariant_state["enabled"] = True
        events.append(("batch_invariant", True))

    monkeypatch.setattr(
        batch_invariant_ops,
        "disable_batch_invariant_mode",
        disable_batch_invariant_mode,
    )
    monkeypatch.setattr(
        batch_invariant_ops,
        "enable_batch_invariant_mode",
        enable_batch_invariant_mode,
    )

    with pytest.raises(RuntimeError, match="capture failure"):
        with dvr_draft_decode_context(model_runner, {}, capture=True):
            assert not backend.enable_deterministic
            assert not server_args.enable_deterministic_inference
            assert not global_server_args.enable_deterministic_inference
            assert not batch_invariant_state["enabled"]
            assert server_args.flashinfer_allreduce_fusion_backend is None
            assert (
                model_runner.spec_algorithm
                == SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
            )
            if draft_custom_all_reduce:
                assert any(
                    event[0:2] == ("custom_all_reduce", True) for event in events
                )
            raise RuntimeError("capture failure")

    assert backend.enable_deterministic
    assert server_args.enable_deterministic_inference
    assert global_server_args.enable_deterministic_inference
    assert batch_invariant_state["enabled"]
    assert model_runner.spec_algorithm == SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK
    assert ("deterministic_env", True) in events
    assert events.count(("kernel_policy_cache", "clear")) == 2
    assert events[-1] == ("kernel_policy_cache", "clear")
    assert (("custom_all_reduce", False) in [event[:2] for event in events]) == (
        draft_custom_all_reduce
    )


def test_draft_replay_changes_only_backend_local_metadata(monkeypatch):
    backend = SimpleNamespace(enable_deterministic=True)
    model_runner = SimpleNamespace(
        attn_backend=backend,
        server_args=SimpleNamespace(
            enable_deterministic_inference=True,
            dvr_enable_draft_custom_all_reduce=True,
        ),
        spec_algorithm=SpeculativeAlgorithm.DECODE_VERIFY_ROLLBACK,
    )

    monkeypatch.setattr(
        graph_module,
        "_fast_decode_overrides",
        lambda *_args, **_kwargs: [(backend, "enable_deterministic", False)],
    )
    monkeypatch.setattr(
        graph_module,
        "_iter_decode_custom_all_reduce_groups",
        lambda _: pytest.fail("replay must not change collective policy"),
    )
    monkeypatch.setattr(
        graph_module,
        "get_server_args",
        lambda: pytest.fail("replay must not change global ServerArgs"),
    )
    monkeypatch.setattr(
        graph_module,
        "_clear_draft_kernel_policy_caches",
        lambda: pytest.fail("replay must not clear process-wide kernel caches"),
    )
    monkeypatch.setattr(
        graph_module,
        "envs",
        SimpleNamespace(
            SGLANG_ENABLE_DETERMINISTIC_INFERENCE=SimpleNamespace(
                is_set=lambda: pytest.fail(
                    "replay must not change deterministic environment state"
                )
            )
        ),
    )

    with dvr_draft_decode_context(model_runner, {}):
        assert not backend.enable_deterministic
        assert model_runner.server_args.enable_deterministic_inference

    assert backend.enable_deterministic


def test_draft_policy_cache_clear_uses_upstream_cache_boundaries(monkeypatch):
    from sglang.kernels.ops.moe import fused_moe_triton_kernels as kernel_module
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe_triton_config as config_module,
    )

    config_module.get_moe_configs.cache_clear()
    kernel_module.should_enable_swap_ab.cache_clear()
    monkeypatch.setattr(
        config_module,
        "get_server_args",
        lambda: SimpleNamespace(enable_deterministic_inference=True),
    )
    monkeypatch.setattr(kernel_module, "_is_cuda", False)

    assert config_module.get_moe_configs(2, 3, None) is None
    assert not kernel_module.should_enable_swap_ab(32, 64)
    assert config_module.get_moe_configs.cache_info().currsize == 1
    assert kernel_module.should_enable_swap_ab.cache_info().currsize == 1

    graph_module._clear_draft_kernel_policy_caches()

    assert config_module.get_moe_configs.cache_info().currsize == 0
    assert kernel_module.should_enable_swap_ab.cache_info().currsize == 0


def test_triton_fast_decode_override_contract(monkeypatch):
    """Detect upstream Triton field changes at DVR's narrow graph boundary."""

    monkeypatch.setenv("SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false")
    model_runner = SimpleNamespace(
        server_args=SimpleNamespace(
            triton_attention_split_tile_size=None,
            triton_attention_num_kv_splits=8,
        ),
        kv_cache_dtype=torch.bfloat16,
        tp_size=2,
        model_config=SimpleNamespace(
            num_attention_heads=16,
            get_num_kv_heads=lambda tp_size: 8 // tp_size,
        ),
    )
    backend = object.__new__(TritonAttnBackend)
    backend.max_context_len = 8192
    backend.max_kv_splits = 32
    backend.split_tile_size = 256
    backend.static_kv_splits = False
    backend.enable_deterministic = True
    backend.cuda_graph_attn_logits = torch.zeros(4, 2, 32, 8)
    backend.cuda_graph_attn_lse = torch.zeros(4, 2, 32)
    backend.cuda_graph_swa_attn_logits = None
    backend.cuda_graph_num_kv_splits = torch.full((4,), 32, dtype=torch.int32)

    for owner, name, value in _fast_decode_overrides(backend, model_runner, {}):
        setattr(owner, name, value)

    assert not backend.enable_deterministic
    assert backend.max_kv_splits == 8
    assert backend.cuda_graph_attn_logits.shape[2] == 8
    assert torch.equal(
        backend.cuda_graph_num_kv_splits,
        torch.full((4,), 8, dtype=torch.int32),
    )


def test_fa3_fast_decode_override_contract():
    backend = object.__new__(FlashAttentionBackend)
    backend.fa_impl_ver = 3
    backend.num_splits = 1
    with dvr_draft_decode_context(
        SimpleNamespace(
            attn_backend=backend,
            server_args=SimpleNamespace(
                enable_deterministic_inference=True,
                dvr_enable_draft_custom_all_reduce=False,
            ),
        ),
        {},
    ):
        assert backend.num_splits == 0

    assert backend.num_splits == 1


def test_backend_resolution_returns_attention_and_linear_state_once():
    class Backend:
        token_to_kv_pool = object()
        req_to_token_pool = object()
        needs_cpu_seq_lens = False

    adapter = object()
    linear_backend = object.__new__(DVRGDNAttnBackend)
    linear_backend.dvr_state_adapter = adapter
    linear_backend.needs_cpu_seq_lens = False
    full_attention = Backend()
    hybrid_linear = HybridLinearAttnBackend(
        full_attention, linear_backend, full_attn_layers=[]
    )
    children = [Backend(), Backend()]
    tbo = TboAttnBackend(primary=hybrid_linear, children=children)
    model_runner = SimpleNamespace(
        kv_cache_dtype=torch.bfloat16,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        server_args=SimpleNamespace(speculative_attention_mode="prefill"),
    )
    hybrid = HybridAttnBackend(
        model_runner=model_runner,
        prefill_backend=Backend(),
        decode_backend=tbo,
    )

    leaves, resolved_adapter = _resolve_dvr_backends(hybrid)

    assert leaves == [full_attention, *children]
    assert resolved_adapter is adapter


def test_backend_resolution_flattens_upstream_multi_step_wrappers():
    triton = object.__new__(TritonAttnBackend)
    fa3 = object.__new__(FlashAttentionBackend)
    fa3.fa_impl_ver = 3
    wrapper = SimpleNamespace(attn_backends=[triton, fa3])

    leaves, adapter = _resolve_dvr_backends(wrapper)

    assert leaves == [triton, fa3]
    assert adapter is None


def test_hybrid_backend_propagates_cpu_sequence_length_requirement():
    class Backend:
        token_to_kv_pool = object()
        req_to_token_pool = object()

        def __init__(self, needs_cpu_seq_lens):
            self.needs_cpu_seq_lens = needs_cpu_seq_lens

    model_runner = SimpleNamespace(
        kv_cache_dtype=torch.bfloat16,
        token_to_kv_pool=object(),
        req_to_token_pool=object(),
        server_args=SimpleNamespace(speculative_attention_mode="prefill"),
    )

    hybrid = HybridAttnBackend(
        model_runner=model_runner,
        prefill_backend=Backend(True),
        decode_backend=Backend(False),
    )

    assert hybrid.needs_cpu_seq_lens


def test_backend_resolution_rejects_distinct_linear_state_adapters():
    class Backend:
        token_to_kv_pool = object()
        req_to_token_pool = object()
        needs_cpu_seq_lens = False

    def hybrid_with_adapter(adapter):
        linear_backend = object.__new__(DVRGDNAttnBackend)
        linear_backend.dvr_state_adapter = adapter
        linear_backend.needs_cpu_seq_lens = False
        return HybridLinearAttnBackend(Backend(), linear_backend, full_attn_layers=[])

    backend = TboAttnBackend(
        primary=hybrid_with_adapter(object()),
        children=[hybrid_with_adapter(object())],
    )

    with pytest.raises(RuntimeError, match="multiple linear-state adapters"):
        _resolve_dvr_backends(backend)


def test_backend_validation_accepts_supported_attention_backends():
    fa3 = object.__new__(FlashAttentionBackend)
    fa3.fa_impl_ver = 3
    for backend in (object.__new__(TritonAttnBackend), fa3):
        leaves, adapter = validate_dvr_attention_backend(backend)
        assert leaves == [backend] and adapter is None

    fa4 = object.__new__(FlashAttentionBackend)
    fa4.fa_impl_ver = 4
    with pytest.raises(RuntimeError, match="requires FlashAttention 3"):
        validate_dvr_attention_backend(fa4)


def test_draft_custom_allreduce_context_restores_target_policy():
    communicator = SimpleNamespace(
        disabled=True,
        original_disabled=True,
        full_nvlink=True,
    )
    group = SimpleNamespace(ca_comm=communicator, world_size=2)

    with _draft_custom_allreduce_enabled(group) as enabled:
        assert enabled
        assert not communicator.disabled

    assert communicator.disabled
    assert communicator.original_disabled

    communicator.full_nvlink = False
    with _draft_custom_allreduce_enabled(group) as enabled:
        assert not enabled
        assert communicator.disabled


def test_dvr_graph_layouts_separate_draft_decode_from_target_verify():
    draft_runner = object.__new__(DVRDraftDecodeCudaGraphRunner)
    verify_runner = object.__new__(DVRTargetVerifyCudaGraphRunner)
    verify_runner.speculative_num_draft_tokens = 16
    verify_runner.model_runner = SimpleNamespace(
        is_draft_worker=False,
        decode_num_tokens_per_req=lambda *, num_draft_tokens: num_draft_tokens,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: True),
    )

    assert draft_runner._resolve_capture_layout() == (ForwardMode.DECODE, 1)
    assert not draft_runner.owns_attention_graph_state
    assert verify_runner._resolve_capture_layout() == (ForwardMode.TARGET_VERIFY, 16)
    assert ForwardMode.TARGET_VERIFY.is_extend()
    assert not ForwardMode.TARGET_VERIFY.is_decode()


@pytest.mark.parametrize(
    (
        "forward_mode",
        "dvr_self_draft",
        "dvr_dflash",
        "dflash",
        "records_event",
    ),
    [
        (ForwardMode.DECODE, False, False, False, True),
        (ForwardMode.DECODE, True, False, False, False),
        (ForwardMode.TARGET_VERIFY, False, False, True, True),
        (ForwardMode.TARGET_VERIFY, True, False, False, True),
        (ForwardMode.TARGET_VERIFY, False, True, False, True),
    ],
)
def test_cuda_graph_war_event_skips_provisional_self_draft(
    forward_mode, dvr_self_draft, dvr_dflash, dflash, records_event
):
    events = []

    class Event:
        def record(self):
            events.append("record")

    class Backend:
        @contextmanager
        def replay_session(self):
            yield

        def replay(self, _key, _forward_batch):
            return PPProxyTensors({})

    runner = object.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        device_timer=None,
        spec_algorithm=SimpleNamespace(
            is_dvr_self_draft=lambda: dvr_self_draft,
            is_dvr_dflash=lambda: dvr_dflash,
            is_dflash_family=lambda: dflash,
        ),
        war_fastpath_read_done_event=None,
    )
    runner.attn_backend = SimpleNamespace(
        use_captured_forward_metadata_for_breakable_cuda_graph=False
    )
    runner.device_module = SimpleNamespace(Event=Event)
    runner.backend = Backend()
    runner.bs = 1

    def load_batch(_forward_batch, _pp_proxy_tensors):
        runner._replay_graph_key = 1

    runner.load_batch = load_batch
    runner.execute(SimpleNamespace(forward_mode=forward_mode))

    assert bool(events) is records_event
    assert (
        runner.model_runner.war_fastpath_read_done_event is not None
    ) is records_event
