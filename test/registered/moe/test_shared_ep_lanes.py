"""CPU-only coverage for SharedEP lane ownership and staged dispatch."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.shared_ep.backend import (
    SharedEpDispatcher,
    SharedEpLaneDispatcher,
    _get_shared_state,
)
from sglang.srt.layers.moe.shared_ep.lanes import (
    SHARED_EP_MAX_STATE_LANES,
    SharedEpLaneProtocol,
    compute_shared_ep_lane_protocol,
    shared_ep_state_resource_key,
)
from sglang.srt.layers.moe.shared_ep.profiles import select_profile
from sglang.srt.layers.moe.shared_ep.runtime import (
    SharedEpRuntimeCapability,
    SharedEpRuntimeHooks,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")


def _server_args(**overrides):
    values = dict(
        enable_two_batch_overlap=False,
        speculative_algorithm=None,
        speculative_num_draft_tokens=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _config() -> MoeRunnerConfig:
    return MoeRunnerConfig(
        hidden_size=4096,
        intermediate_size_per_partition=2048,
        top_k=6,
        num_experts=256,
        num_local_experts=32,
        params_dtype=torch.bfloat16,
        shared_ep_model_namespace="target",
    )


def test_lane_protocol_separates_tbo_and_speculative_generations():
    assert compute_shared_ep_lane_protocol(_server_args()).lane_count == 1

    protocol = compute_shared_ep_lane_protocol(
        _server_args(
            enable_two_batch_overlap=True,
            speculative_algorithm="EAGLE",
            speculative_num_draft_tokens=4,
        )
    )
    assert protocol == SharedEpLaneProtocol(tbo_width=2, generation_width=4)
    assert protocol.lane_count == SHARED_EP_MAX_STATE_LANES
    assert {
        protocol.lane_id(generation_index=generation, tbo_subbatch_index=subbatch)
        for generation in range(4)
        for subbatch in range(2)
    } == set(range(8))

    with pytest.raises(ValueError, match="fixed release cap"):
        compute_shared_ep_lane_protocol(
            _server_args(
                enable_two_batch_overlap=True,
                speculative_algorithm="EAGLE",
                speculative_num_draft_tokens=5,
            )
        )


def test_state_resource_keys_are_deterministic_and_namespaced():
    kwargs = dict(
        runtime_name="rocm",
        profile_name="dsv4_pro_mxfp4",
        ep_size=8,
        model_namespace="target",
        lane_id=3,
    )
    key = shared_ep_state_resource_key(**kwargs)
    assert key == shared_ep_state_resource_key(**kwargs)
    assert "model=target" in key
    assert "lane=3" in key
    assert key != shared_ep_state_resource_key(
        **{**kwargs, "model_namespace": "draft-nextn"}
    )
    assert key != shared_ep_state_resource_key(**{**kwargs, "lane_id": 4})


def test_state_lookup_reuses_one_lane_but_not_another():
    profile = select_profile(
        _config(),
        capability=(9, 0),
        ep_size=8,
        block_shape=(128, 128),
        max_tokens_per_rank=32,
        platform="cuda",
    )
    resources = SimpleNamespace(buffers={})

    def create_state(**kwargs):
        return SimpleNamespace(
            layout=kwargs["layout"],
            input_allocation=object(),
            output_allocation=object(),
            input_epoch=object(),
            output_epoch=object(),
        )

    runtime = SharedEpRuntimeHooks(
        name="test",
        platform="cuda",
        create_state=Mock(side_effect=create_state),
        capabilities=frozenset(SharedEpRuntimeCapability),
    )
    parallel = SimpleNamespace(moe_ep_group=SimpleNamespace(cpu_group="cpu"))

    with (
        patch(
            "sglang.srt.layers.moe.shared_ep.backend.get_resources",
            return_value=resources,
        ),
        patch(
            "sglang.srt.layers.moe.shared_ep.backend.get_parallel",
            return_value=parallel,
        ),
        patch(
            "sglang.srt.layers.moe.shared_ep.backend.torch.cuda.current_device",
            return_value=0,
        ),
    ):
        lane0 = _get_shared_state(
            _config(),
            profile,
            runtime,
            model_namespace="target",
            lane_id=0,
        )
        lane0_again = _get_shared_state(
            _config(),
            profile,
            runtime,
            model_namespace="target",
            lane_id=0,
        )
        lane1 = _get_shared_state(
            _config(),
            profile,
            runtime,
            model_namespace="target",
            lane_id=1,
        )

    assert lane0 is lane0_again
    assert lane0 is not lane1
    assert lane0.input_allocation is not lane1.input_allocation
    assert lane0.output_allocation is not lane1.output_allocation
    assert lane0.input_epoch is not lane1.input_epoch
    assert lane0.output_epoch is not lane1.output_epoch
    assert runtime.create_state.call_count == 2


def test_lane_wrapper_routes_staged_tbo_calls_to_disjoint_inners():
    inner0 = Mock()
    inner1 = Mock()
    wrapper = SharedEpLaneDispatcher(
        [inner0, inner1],
        SharedEpLaneProtocol(tbo_width=2, generation_width=1),
    )

    with patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_forward",
        return_value=SimpleNamespace(shared_ep_generation=0),
    ):
        wrapper.dispatch_a(
            hidden_states="h0",
            topk_output="t0",
            tbo_subbatch_index=0,
        )
        wrapper.dispatch_a(
            hidden_states="h1",
            topk_output="t1",
            tbo_subbatch_index=1,
        )
        wrapper.dispatch_b(tbo_subbatch_index=0)
        wrapper.dispatch_b(tbo_subbatch_index=1)
        wrapper.combine_a(
            combine_input="c0",
            tbo_subbatch_index=0,
        )
        wrapper.combine_a(
            combine_input="c1",
            tbo_subbatch_index=1,
        )
        wrapper.combine_b(tbo_subbatch_index=0)
        wrapper.combine_b(tbo_subbatch_index=1)

    inner0.dispatch_a.assert_called_once_with(hidden_states="h0", topk_output="t0")
    inner1.dispatch_a.assert_called_once_with(hidden_states="h1", topk_output="t1")
    inner0.dispatch_b.assert_called_once_with()
    inner1.dispatch_b.assert_called_once_with()
    inner0.combine_a.assert_called_once_with(combine_input="c0")
    inner1.combine_a.assert_called_once_with(combine_input="c1")
    inner0.combine_b.assert_called_once_with()
    inner1.combine_b.assert_called_once_with()


def test_lane_wrapper_routes_generation_to_a_disjoint_state_lane():
    inners = [Mock() for _ in range(4)]
    protocol = SharedEpLaneProtocol(tbo_width=2, generation_width=2)
    wrapper = SharedEpLaneDispatcher(inners, protocol)
    forward_flags = SimpleNamespace(shared_ep_generation=1)
    inners[2].dispatch.return_value = "generation-one"
    inners[2].combine.return_value = "combined-generation-one"

    with patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_forward",
        return_value=forward_flags,
    ):
        assert wrapper.dispatch("hidden", "topk") == "generation-one"
        assert wrapper.combine("combine-input") == "combined-generation-one"

    inners[2].dispatch.assert_called_once_with(
        hidden_states="hidden",
        topk_output="topk",
    )
    inners[2].combine.assert_called_once_with(combine_input="combine-input")
    for index, inner in enumerate(inners):
        if index != 2:
            inner.dispatch.assert_not_called()
            inner.combine.assert_not_called()


def test_shared_lane_stages_conservative_dispatch_and_combine():
    dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
    dispatcher.lane_id = 0
    dispatcher._stage = "initial"
    dispatcher._active_uses_shared_ep = None
    dispatcher.fallback_dispatcher = Mock()
    dispatch_output = object()
    dispatcher._dispatch_shared_ep = Mock(return_value=dispatch_output)
    hidden_states = torch.ones((2, 4))

    with patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_forward",
        return_value=SimpleNamespace(shared_ep_is_decode=True),
    ):
        dispatcher.dispatch_a("hidden", "topk")
        assert dispatcher.dispatch_b() is dispatch_output
        dispatcher.combine_a(StandardCombineInput(hidden_states=hidden_states))
        assert dispatcher.combine_b() is hidden_states

    dispatcher._dispatch_shared_ep.assert_called_once_with(
        "hidden",
        "topk",
        phase="decode",
    )
    dispatcher.fallback_dispatcher.dispatch_a.assert_not_called()
    assert dispatcher._stage == "initial"


def test_non_decode_stages_route_entire_transaction_through_fallback():
    fallback = Mock()
    dispatch_output = object()
    hidden_states = torch.zeros((2, 4))
    combined = torch.arange(16).view(4, 4)
    fallback.dispatch_b.return_value = dispatch_output
    fallback.combine_b.return_value = combined
    dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
    dispatcher.lane_id = 1
    dispatcher._stage = "initial"
    dispatcher._active_uses_shared_ep = None
    dispatcher.fallback_dispatcher = fallback
    dispatcher._dispatch_shared_ep = Mock()

    with patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_forward",
        return_value=SimpleNamespace(shared_ep_is_decode=False),
    ):
        dispatcher.dispatch_a(hidden_states, "verify-topk")
        assert dispatcher.dispatch_b() is dispatch_output
        dispatcher.combine_a("fallback-combine-input")
        torch.testing.assert_close(dispatcher.combine_b(), combined[:2])

    fallback.dispatch_a.assert_called_once_with(
        hidden_states=hidden_states,
        topk_output="verify-topk",
    )
    fallback.combine_a.assert_called_once_with(combine_input="fallback-combine-input")
    dispatcher._dispatch_shared_ep.assert_not_called()
    assert dispatcher._stage == "initial"


def test_same_lane_rejects_a_second_writer_before_combine():
    dispatcher = SharedEpDispatcher.__new__(SharedEpDispatcher)
    dispatcher.lane_id = 7
    dispatcher._stage = "initial"
    dispatcher._active_uses_shared_ep = None
    dispatcher.fallback_dispatcher = Mock()
    dispatcher._dispatch_shared_ep = Mock(return_value=object())

    with patch(
        "sglang.srt.layers.moe.shared_ep.backend.get_forward",
        return_value=SimpleNamespace(shared_ep_is_decode=True),
    ):
        dispatcher.dispatch_a("hidden", "topk")
        with pytest.raises(RuntimeError, match="Concurrent writers"):
            dispatcher.dispatch_a("other", "other-topk")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
