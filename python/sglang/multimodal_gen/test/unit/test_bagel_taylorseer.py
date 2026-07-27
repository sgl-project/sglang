# SPDX-License-Identifier: Apache-2.0

import asyncio
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from fastapi import HTTPException
from starlette.requests import Request

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import generations
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import build_sampling_params
from sglang.multimodal_gen.runtime.models.dits.bagel_taylorseer import (
    BagelTaylorSeerContext,
    TaylorSeerConfig,
    TaylorSeerState,
)


class _CapturedSampling(Exception):
    pass


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_order", 0),
        ("fresh_threshold", -1),
        ("first_enhance", True),
    ],
)
def test_taylorseer_config_rejects_invalid_values(field: str, value: object) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=field):
        TaylorSeerConfig(**kwargs)


def test_taylorseer_schedule_matches_bagel_defaults() -> None:
    state = TaylorSeerState(num_layers=1, num_steps=8)
    step_types = []

    for step in range(8):
        step_types.append(state.begin_step(step))
        if state.should_compute(0):
            state.update_cache(0, torch.tensor([float(step)]))
        else:
            assert torch.isfinite(state.approximate(0)).all()
        state.end_step()

    assert step_types == ["full"] * 5 + ["Taylor", "Taylor", "full"]
    assert state.get_stats() == {
        "total_steps": 8,
        "full_steps": 6,
        "taylor_steps": 2,
    }


def test_taylorseer_fifty_step_schedule_has_twenty_refreshes() -> None:
    state = TaylorSeerState(num_layers=1, num_steps=50)
    for _ in range(50):
        state.begin_next_step()
        state.end_step()

    assert state.get_stats() == {
        "total_steps": 50,
        "full_steps": 20,
        "taylor_steps": 30,
    }


def test_taylorseer_49_updates_match_official_nominal_fifty_schedule() -> None:
    state = TaylorSeerState(num_layers=1, num_steps=49)
    for _ in range(49):
        state.begin_next_step()
        state.end_step()

    assert state.get_stats() == {
        "total_steps": 49,
        "full_steps": 19,
        "taylor_steps": 30,
    }


def test_taylorseer_forecasts_linear_layer_output() -> None:
    state = TaylorSeerState(
        num_layers=1,
        num_steps=3,
        config=TaylorSeerConfig(
            max_order=2,
            fresh_threshold=3,
            first_enhance=2,
        ),
    )
    for step in (0, 1):
        assert state.begin_step(step) == "full"
        state.update_cache(0, torch.tensor([float(step)]))
        state.end_step()

    assert state.begin_step(2) == "Taylor"
    torch.testing.assert_close(state.approximate(0), torch.tensor([2.0]))
    state.end_step()


def test_taylorseer_cache_is_detached_and_preserves_tensor_metadata() -> None:
    state = TaylorSeerState(
        num_layers=1,
        num_steps=2,
        config=TaylorSeerConfig(
            max_order=1,
            fresh_threshold=3,
            first_enhance=1,
        ),
    )
    feature = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    state.begin_next_step()
    state.update_cache(0, feature)
    state.end_step()

    state.begin_next_step()
    forecast = state.approximate(0)
    state.end_step()

    assert forecast.shape == feature.shape
    assert forecast.dtype == feature.dtype
    assert forecast.device == feature.device
    assert not forecast.requires_grad


def test_cfg_branches_keep_independent_taylor_caches() -> None:
    context = BagelTaylorSeerContext.create(
        num_layers=1,
        num_steps=2,
        has_secondary=True,
        config=TaylorSeerConfig(
            max_order=1,
            fresh_threshold=2,
            first_enhance=1,
        ),
    )
    assert context.secondary_unconditional is not None
    context.validate_branch_count(has_secondary=True)

    states = (
        context.conditional,
        context.unconditional,
        context.secondary_unconditional,
    )
    for state, value in zip(states, (1.0, 2.0, 3.0), strict=True):
        state.begin_next_step()
        state.update_cache(0, torch.tensor([value]))
        state.end_step()

    for state in states:
        state.begin_next_step()
    assert context.conditional.approximate(0).item() == 1.0
    assert context.unconditional.approximate(0).item() == 2.0
    assert context.secondary_unconditional.approximate(0).item() == 3.0
    for state in states:
        state.end_step()

    assert context.get_stats()["conditional"] == {
        "total_steps": 2,
        "full_steps": 1,
        "taylor_steps": 1,
    }
    with pytest.raises(ValueError, match="branch count"):
        context.validate_branch_count(has_secondary=False)


def test_taylorseer_requires_sequential_request_steps() -> None:
    context = BagelTaylorSeerContext.create(
        num_layers=1,
        num_steps=1,
        has_secondary=False,
    )
    context.conditional.begin_next_step()
    with pytest.raises(RuntimeError, match="prior one"):
        context.conditional.begin_next_step()
    context.conditional.end_step()
    with pytest.raises(ValueError, match="within the request"):
        context.conditional.begin_next_step()


def test_partial_layer_failure_poison_releases_cache_and_blocks_retry() -> None:
    context = BagelTaylorSeerContext.create(
        num_layers=2,
        num_steps=2,
        has_secondary=False,
    )
    context.conditional.begin_next_step()
    context.conditional.update_cache(0, torch.ones(2, 3))

    context.conditional.poison()

    assert context.is_failed
    assert context.conditional.is_failed
    assert context.unconditional.is_failed
    assert context.conditional._layers == []
    with pytest.raises(RuntimeError, match="invalid after a failed"):
        context.unconditional.begin_next_step()


def test_release_drops_cached_tensors_before_vae_decode() -> None:
    context = BagelTaylorSeerContext.create(
        num_layers=1,
        num_steps=1,
        has_secondary=False,
    )
    context.conditional.begin_next_step()
    context.conditional.update_cache(0, torch.ones(2, 3))
    context.conditional.end_step()
    cached_tensor_ref = weakref.ref(context.conditional._layers[0].derivatives[0])

    context.release()
    context.release()

    assert cached_tensor_ref() is None
    assert context.conditional._layers == []
    assert len(context.conditional._run_health.states) == 0
    with pytest.raises(RuntimeError, match="released after denoising"):
        context.conditional.begin_next_step()
    with pytest.raises(ValueError, match="only before evaluation"):
        BagelTaylorSeerContext(
            context.conditional,
            TaylorSeerState(num_layers=1, num_steps=1),
        )


def test_request_context_has_no_cache_retaining_reference_cycle() -> None:
    def build_refs() -> (
        tuple[weakref.ReferenceType[object], weakref.ReferenceType[torch.Tensor]]
    ):
        context = BagelTaylorSeerContext.create(
            num_layers=1,
            num_steps=1,
            has_secondary=False,
        )
        context.conditional.begin_next_step()
        context.conditional.update_cache(0, torch.ones(2, 3))
        context.conditional.end_step()
        return (
            weakref.ref(context),
            weakref.ref(context.conditional._layers[0].derivatives[0]),
        )

    context_ref, cached_tensor_ref = build_refs()

    # Refcount cleanup must be immediate; this intentionally does not call gc.collect().
    assert context_ref() is None
    assert cached_tensor_ref() is None


def test_http_generation_forwards_taylorseer_control() -> None:
    request = ImageGenerationsRequest(
        prompt="draw a blue fox",
        response_format="b64_json",
        enable_taylorseer=True,
    )
    raw_request = Request({"type": "http", "headers": []})

    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
            "get_global_server_args",
            return_value=SimpleNamespace(model_path="", output_path=None),
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
            "build_sampling_params",
            side_effect=_CapturedSampling,
        ) as sampling_mock,
    ):
        with pytest.raises(_CapturedSampling):
            asyncio.run(generations(request, raw_request))

    assert sampling_mock.call_args.kwargs["enable_taylorseer"] is True


def test_http_generation_normalizes_explicit_false_taylorseer_control() -> None:
    request = ImageGenerationsRequest(
        prompt="draw a blue fox",
        response_format="b64_json",
        enable_taylorseer=False,
    )
    raw_request = Request({"type": "http", "headers": []})

    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
            "get_global_server_args",
            return_value=SimpleNamespace(model_path="", output_path=None),
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
            "build_sampling_params",
            side_effect=_CapturedSampling,
        ) as sampling_mock,
    ):
        with pytest.raises(_CapturedSampling):
            asyncio.run(generations(request, raw_request))

    assert sampling_mock.call_args.kwargs["enable_taylorseer"] is None


def test_http_generation_rejects_unsupported_taylorseer_with_400() -> None:
    request = ImageGenerationsRequest(
        prompt="draw a blue fox",
        enable_taylorseer=True,
    )
    raw_request = Request({"type": "http", "headers": []})

    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
            "get_global_server_args",
            return_value=SimpleNamespace(model_path="other-model", output_path=None),
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
            "build_sampling_params",
            side_effect=TypeError(
                "SamplingParams got an unexpected keyword argument 'enable_taylorseer'"
            ),
        ),
    ):
        with pytest.raises(HTTPException) as error:
            asyncio.run(generations(request, raw_request))

    assert error.value.status_code == 400
    assert "enable_taylorseer is not supported" in error.value.detail


def test_default_http_control_is_filtered_for_non_bagel_models() -> None:
    request = ImageGenerationsRequest(prompt="draw a blue fox")
    assert request.enable_taylorseer is None
    resolved = SamplingParams(prompt="draw a blue fox")

    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.utils."
            "get_global_server_args",
            return_value=SimpleNamespace(model_path="other-model"),
        ),
        patch.object(
            SamplingParams,
            "from_user_sampling_params_args",
            return_value=resolved,
        ) as sampling_builder,
    ):
        build_sampling_params(
            "request-id",
            prompt=request.prompt,
            enable_taylorseer=request.enable_taylorseer,
        )

    assert "enable_taylorseer" not in sampling_builder.call_args.kwargs
