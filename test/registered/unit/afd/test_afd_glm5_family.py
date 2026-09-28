"""GLM-5.2 family seams: identity, DSA guard, shared-topk carry."""

from __future__ import annotations

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys
from types import SimpleNamespace

import pytest

from sglang.srt.afd import contracts as contracts
from sglang.srt.afd import metadata as metadata
from sglang.srt.afd import profiles as profiles
from sglang.srt.afd.model_adapters import glm5 as glm5


def _config(**overrides):
    fields = {
        "index_topk": 2048,
        "index_topk_freq": 4,
        "num_hidden_layers": 8,
        "hidden_size": 64,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _model(*, name="GlmMoeDsaForCausalLM", config=None, layers=None):
    inner = SimpleNamespace(
        config=config if config is not None else _config(),
        layers=layers if layers is not None else [],
        start_layer=0,
        end_layer=0 if layers is None else len(layers),
    )
    return type(name, (SimpleNamespace,), {})(model=inner)


def test_identity_admits_dsa_glm_and_rejects_everything_else():
    assert glm5.matches_glm_moe_dsa(_model())
    assert glm5.matches_glm_moe_dsa(
        _model(config=_config(index_topk_pattern=["I", "S", "S", "S"]))
    )
    assert not glm5.matches_glm_moe_dsa(_model(name="Qwen3MoeForCausalLM"))
    assert not glm5.matches_glm_moe_dsa(_model(config=_config(index_topk=0)))
    assert not glm5.matches_glm_moe_dsa(_model(config=_config(index_topk_freq=0)))
    assert not glm5.matches_glm_moe_dsa(
        _model(config=_config(index_topk_pattern=["I", "X"]))
    )
    assert not glm5.matches_glm_moe_dsa(SimpleNamespace())


def test_identity_does_not_pin_a_layer_count():
    """Reduced configs must resolve, unlike the exact-size Qwen entry."""

    assert glm5.matches_glm_moe_dsa(_model(config=_config(num_hidden_layers=2)))
    assert glm5.matches_glm_moe_dsa(_model(config=_config(num_hidden_layers=78)))


def test_adapter_binds_the_dsa_guard_and_its_private_contract():
    adapter = glm5.Glm5AFDAdapter(
        role=contracts.AFDRole.FFN,
        model=_model(
            layers=[
                SimpleNamespace(),
                SimpleNamespace(),
            ]
        ),
        attention_backend=None,
    )
    assert adapter.guard_class is metadata.DSAMetadataGuard
    assert adapter.metadata_contract is contracts.MetadataContract.PRIVATE_DSA
    assert metadata.DSAMetadataGuard.backends == ("nsa",)
    # Private state is created per capture, so nothing is reserved up front.
    assert (
        metadata.DSAMetadataGuard.initialize_shared_state(
            backend=object(),
            max_rows=32,
        )
        == 0
    )


@pytest.mark.parametrize(
    "field,value",
    [("attn_cp_size", 2), ("attn_dcp_size", 2), ("enable_prefill_cp", True)],
)
def test_context_parallel_uses_published_runtime_config(monkeypatch, field, value):
    base = sys.modules[glm5.Glm5AFDAdapter.__mro__[1].__module__]
    cfg = SimpleNamespace(attn_cp_size=1, attn_dcp_size=1, enable_prefill_cp=False)
    monkeypatch.setattr(base, "get_parallel", lambda: cfg)
    layers = [SimpleNamespace(), SimpleNamespace()]
    adapter = glm5.Glm5AFDAdapter(
        role=contracts.AFDRole.FFN, model=_model(layers=layers), attention_backend=None
    )
    adapter.validate_context_parallel()
    setattr(cfg, field, value)
    with pytest.raises(contracts.AFDError, match="AFD_CONTEXT_PARALLEL_UNSUPPORTED"):
        adapter.validate_context_parallel()


def test_glm_attention_uses_native_tp_scope_and_layout(routing):
    from contextlib import contextmanager

    import torch

    events = []
    indices = torch.tensor([2])
    hidden, residual = torch.ones(1, 4), torch.zeros(1, 4)
    batch = object()

    class Attention:
        @contextmanager
        def maybe_use_decode_attn_tp(self, forward_batch):
            assert forward_batch is batch
            events.append("enter")
            yield
            events.append("exit")

        def __call__(
            self,
            *,
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
            input_on_attention_tp_slices,
            prev_topk_indices,
        ):
            assert events == ["enter"]
            assert not input_on_attention_tp_slices
            assert prev_topk_indices is indices
            return hidden_states, indices

    layer = SimpleNamespace(
        self_attn=Attention(),
        layer_communicator=SimpleNamespace(
            input_on_attention_tp_slices=False,
            prepare_attn_and_capture_last_layer_outputs=lambda h, r, b, **kw: (h, r),
            prepare_mlp=lambda h, r, b: (h, r),
        ),
    )
    routing["afd_execution_mode"] = lambda: "attention"
    routing["get_attn_tp_context"] = lambda: SimpleNamespace(
        clear_attn_inputs=lambda: events.append("clear"),
    )
    actual = routing["GlmMoeDsaAFDDecoderLayer"].forward_attention_for_afd(
        layer,
        positions=torch.tensor([0]),
        hidden_states=hidden,
        forward_batch=batch,
        residual=residual,
        zero_allocator=None,
        prev_topk_indices=indices,
    )
    assert actual[0] is hidden and actual[1] is residual and actual[2] is indices
    assert events == ["enter", "exit", "clear"]


class _Layer:
    """One DSA layer that records the indices it was handed."""

    def __init__(self, *, skip_topk, next_skip_topk, produced):
        self.self_attn = SimpleNamespace(
            skip_topk=skip_topk,
            next_skip_topk=next_skip_topk,
        )
        self._produced = produced
        self.seen = []

    def forward_attention_for_afd(
        self,
        *,
        positions,
        hidden_states,
        forward_batch,
        residual,
        zero_allocator,
        prev_topk_indices=None,
    ):
        del positions, forward_batch, zero_allocator
        self.seen.append(prev_topk_indices)
        return hidden_states, residual, self._produced


def _carry_adapter(layers):
    adapter = object.__new__(glm5.Glm5AFDAdapter)
    adapter.role = contracts.AFDRole.ATTENTION
    adapter.inner = SimpleNamespace(layers=layers)
    adapter.num_layers = len(layers)
    adapter._stage_states = {}
    adapter._step_id = 1
    return adapter


def _stage(index):
    return SimpleNamespace(index=index, positions=None, forward_batch=None)


def _install_state(adapter, index):
    state = object.__new__(glm5._StageState)
    state.zero_allocator = SimpleNamespace(reset=lambda: None)
    state.topk_indices = None
    state.topk_step = -1
    adapter._stage_states[index] = state
    return state


def test_owner_hands_its_indices_to_the_following_sharing_layers():
    indices = object()
    layers = [
        # freq=4 -> layer 1 owns, layers 2..4 reuse.
        _Layer(skip_topk=False, next_skip_topk=False, produced=None),
        _Layer(skip_topk=False, next_skip_topk=True, produced=indices),
        _Layer(skip_topk=True, next_skip_topk=True, produced=indices),
        _Layer(skip_topk=True, next_skip_topk=False, produced=indices),
    ]
    adapter = _carry_adapter(layers)
    _install_state(adapter, 0)
    stage = _stage(0)
    for layer in range(len(layers)):
        adapter.local_compute(
            layer=layer,
            stage=stage,
            hidden_states=object(),
            residual=None,
        )
    assert layers[0].seen == [None]
    assert layers[1].seen == [None]
    assert layers[2].seen == [indices]
    assert layers[3].seen == [indices]


def test_stages_never_share_one_carry():
    first = object()
    second = object()
    layers = [
        _Layer(skip_topk=False, next_skip_topk=True, produced=first),
        _Layer(skip_topk=True, next_skip_topk=False, produced=second),
    ]
    adapter = _carry_adapter(layers)
    _install_state(adapter, 0)
    _install_state(adapter, 1)
    for stage_index in (0, 1):
        stage = _stage(stage_index)
        for layer in range(len(layers)):
            adapter.local_compute(
                layer=layer,
                stage=stage,
                hidden_states=object(),
                residual=None,
            )
    assert adapter._stage_states[0] is not adapter._stage_states[1]
    assert adapter._stage_states[0].topk_indices is second
    assert adapter._stage_states[1].topk_indices is second
    assert layers[1].seen == [first, first]


def test_a_sharing_layer_refuses_a_carry_from_an_earlier_step():
    """A replay that fails mid-step leaves the python carry a step behind.

    The owner's indices then live only in graph memory, so serving the stale
    slot would produce wrong tokens instead of an error.
    """

    indices = object()
    layers = [
        _Layer(skip_topk=False, next_skip_topk=True, produced=indices),
        _Layer(skip_topk=True, next_skip_topk=False, produced=indices),
    ]
    adapter = _carry_adapter(layers)
    state = _install_state(adapter, 0)
    stage = _stage(0)
    adapter.local_compute(
        layer=0,
        stage=stage,
        hidden_states=object(),
        residual=None,
    )
    assert state.topk_step == adapter._step_id
    # The owner replayed from graph memory this step; only its graph wrote.
    adapter._step_id += 1
    with pytest.raises(
        contracts.AFDError,
        match="AFD_GLM_DSA_TOPK_CARRY_STALE",
    ):
        adapter.local_compute(
            layer=1,
            stage=stage,
            hidden_states=object(),
            residual=None,
        )


def test_a_missing_stage_state_fails_closed():
    layers = [_Layer(skip_topk=False, next_skip_topk=False, produced=None)]
    adapter = _carry_adapter(layers)
    with pytest.raises(
        contracts.AFDError,
        match="AFD_GLM_DSA_STAGE_STATE_MISSING",
    ):
        adapter.local_compute(
            layer=0,
            stage=_stage(0),
            hidden_states=object(),
            residual=None,
        )


def test_dense_dsa_layers_never_touch_the_carry():
    layers = [
        _Layer(skip_topk=False, next_skip_topk=False, produced=object()),
        _Layer(skip_topk=False, next_skip_topk=False, produced=object()),
    ]
    adapter = _carry_adapter(layers)
    state = _install_state(adapter, 0)
    stage = _stage(0)
    for layer in range(len(layers)):
        adapter.local_compute(
            layer=layer,
            stage=stage,
            hidden_states=object(),
            residual=None,
        )
    assert state.topk_indices is None
    assert state.topk_step == -1
    assert layers[0].seen == [None]
    assert layers[1].seen == [None]


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_stage_allocators_are_built_outside_capture_and_only_rewound(
    monkeypatch, stages
):
    import torch

    from sglang.srt.afd.model_adapters.base import AFDDecoderAdapter

    adapter = _carry_adapter(
        [_Layer(skip_topk=False, next_skip_topk=False, produced=None)]
    )
    monkeypatch.setattr(AFDDecoderAdapter, "split_step", lambda self, **kwargs: [])
    hidden = torch.ones((stages, 8))
    adapter.split_step(hidden_states=hidden, stages=stages)
    buffers = [state.zero_allocator._buffer for state in adapter._stage_states.values()]
    assert len({buffer.data_ptr() for buffer in buffers}) == stages
    for state in adapter._stage_states.values():
        state.zero_allocator.allocate(1)
    adapter.split_step(hidden_states=hidden, stages=stages)
    assert [
        state.zero_allocator._buffer.data_ptr()
        for state in adapter._stage_states.values()
    ] == [buffer.data_ptr() for buffer in buffers]
    assert all(
        state.zero_allocator._pointer == 0 for state in adapter._stage_states.values()
    )
    # Local compute must reuse the state created by split_step, including eager replay.
    monkeypatch.setattr(
        glm5,
        "_StageState",
        lambda **kwargs: pytest.fail("allocation inside local_compute"),
    )
    for index in range(stages):
        adapter.local_compute(
            layer=0, stage=_stage(index), hidden_states=hidden, residual=None
        )


def test_the_glm_profile_is_registered_with_its_own_digest():
    assert profiles.GLM5_PAIRED_C1_ID in profiles.AFD_PROFILE_REGISTRY
    factories = profiles.AFD_PROFILE_REGISTRY[profiles.GLM5_PAIRED_C1_ID]
    assert factories.adapter_factory is glm5.Glm5AFDAdapter
    assert factories.model_matcher is glm5.matches_glm_moe_dsa
    assert factories.profile.digest != profiles.QWEN3_PAIRED_C1.digest


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
