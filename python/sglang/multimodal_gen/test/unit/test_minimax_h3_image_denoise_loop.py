# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the full H3 image loop, using real packed layouts."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
    get_forward_context_or_none,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    PackedBranchInputs,
    minimax_h3_image_denoise_loop,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)


def _inputs(text_len, value, mode="t2i", *, latent_t=1, audio_t=0):
    common = dict(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=4,
        latent_w=4,
        audio_t=audio_t,
    )
    if mode == "reference":
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            **common,
            ref_blocks=[
                {"kind": "image", "latent_h": 2, "latent_w": 2},
                {"kind": "image", "latent_h": 4, "latent_w": 4},
            ],
        )
    else:
        packed = minimax_h3_packed_sequence(
            **common,
            include_keyframe_cond=mode == "keyframe",
            keyframe_frame_indices=[0] if mode == "keyframe" else None,
            frame_count=1 if mode == "keyframe" else None,
        )
    return PackedBranchInputs(
        packed=packed,
        text_embeddings=torch.full((text_len, 8), float(value)),
        token_tags=packed["token_tags"],
        attn_metadata=AttentionMetadata(current_timestep=-1),
    )


class _BufferModel:
    """A deterministic packed model that reuses its output buffer.

    Predictions depend on the text, and optionally on current latent and
    timestep. Records preserve actual packed inputs, not mocked results.
    """

    def __init__(
        self, *, dtype=torch.float32, state_scale=0.0, time_scale=0.0, cached=True
    ):
        self.buffer = torch.empty(4, 96, dtype=dtype)
        self.state_scale = state_scale
        self.time_scale = time_scale
        self.cached = cached
        self.prepare_calls = 0
        self.calls = []
        self.plans = []

    def prepare_adaln_plans(self, plans):
        self.prepare_calls += 1
        assert self.prepare_calls == 1, "the loop reserves all AdaLN slots at once"
        self.plans = [p.clone() for p in plans]
        return torch.arange(len(plans), dtype=torch.long) if self.cached else None

    def __call__(self, **kwargs):
        context = get_forward_context()
        assert not torch.is_grad_enabled()
        assert kwargs["audio_pos_info"]["position_ids"].numel() == 0
        assert kwargs["audio_x"].shape == (1, kwargs["x"].shape[1], 32)
        assert not kwargs["audio_x"].any()
        if self.cached:
            slot = kwargs["adaln_cache_slot"]
            assert slot.ndim == 0 and slot.dtype == torch.long
            torch.testing.assert_close(
                kwargs["unique_timesteps"], self.plans[int(slot)], rtol=0, atol=0
            )
        else:
            assert "adaln_cache_slot" not in kwargs
        target_pos = kwargs["img_pos_for_infer_output_info"]["position_ids"]
        target_rows = kwargs["x"][0, target_pos]
        times = kwargs["unique_timesteps"][kwargs["inverse_indices"]]
        self.buffer.resize_(target_rows.shape)
        self.buffer.copy_(
            kwargs["prompt_embeds"][0, 0]
            + self.state_scale * target_rows
            + self.time_scale * times[target_pos, None]
        )
        if not kwargs["skip_mask_out_condition"]:
            self.buffer.mul_(kwargs["update_mask"][:, None])
        self.calls.append(
            {
                "step": context.current_timestep,
                "metadata": context.attn_metadata,
                "metadata_step": None
                if context.attn_metadata is None
                else context.attn_metadata.current_timestep,
                "kwargs": kwargs,
                "skip_mask_out_condition": kwargs["skip_mask_out_condition"],
                "rows": kwargs["x"][0, kwargs["img_pos_info"]["position_ids"]].clone(),
                "times": times.clone(),
                "prediction": self.buffer.clone(),
            }
        )
        return self.buffer, torch.empty(0, 32)


def _run(model, positive, **overrides):
    args = dict(
        model=model,
        positive=positive,
        initial_video_rows=torch.full((positive.packed["img_pos"].numel(), 96), 2.0),
        keyframe_cond_rows=None,
        sigmas=[1.0, 0.5, 0.0],
        device=torch.device("cpu"),
    )
    args.update(overrides)
    return minimax_h3_image_denoise_loop(**args)


def test_image_loop_routes_each_step_through_native_runner():
    calls = []

    def forward(model, kwargs, step):
        assert get_forward_context().current_timestep == step
        calls.append(step)
        return model(**kwargs)

    positive = _inputs(3, 2)
    expected = _run(_BufferModel(state_scale=0.5, time_scale=0.25), positive)
    result = _run(
        _BufferModel(state_scale=0.5, time_scale=0.25),
        positive,
        model_forward=forward,
    )
    assert calls == [0, 1]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_static_branch_preparation_is_request_local_and_runs_once():
    prepared = []

    def prepare(branch):
        assert not model.calls
        kwargs = branch.static_kwargs
        prepared.append(kwargs["prompt_embeds"].clone())
        kwargs["refined_prompt_embeds_length"] = kwargs["prompt_embeds"].shape[0]

    for length, value in [(3, 2), (9, 7)]:
        model = _BufferModel()
        _run(model, _inputs(length, value), prepare_branch=prepare)
        assert len(model.calls) == 2
        assert all(
            call["kwargs"]["refined_prompt_embeds_length"] == length
            for call in model.calls
        )
    assert [tuple(x.shape) for x in prepared] == [(3, 8), (9, 8)]
    assert prepared[0][0, 0] == 2 and prepared[1][0, 0] == 7


@pytest.mark.parametrize("mode", ["t2i", "reference"])
@pytest.mark.parametrize(
    "capacity,width,both",
    [
        (None, 2, True),
        (1, 2, True),
        (2, 2, True),
        (3, 2, True),
        (6, 1, True),
        (6, 2, False),
        (6, 2, True),
    ],
)
def test_adaln_prewarm_keeps_denoising_prefix_and_respects_capacity(
    mode, capacity, width, both
):
    positive = _inputs(3, 2, mode)
    n_cond = int((~positive.packed["update_mask"]).sum())
    args = dict(
        sigmas=[1.0, 0.8],
        keyframe_cond_rows=torch.zeros(n_cond, 96) if n_cond else None,
    )
    baseline = _run(_BufferModel(), positive, **args)
    model = _BufferModel(cached=capacity is not None)
    if capacity is not None:
        model.adaln_cache = SimpleNamespace(max_plans=capacity, max_plan_width=width)
    completed = []
    grid = [1.0, 0.8, 0.3, 0.0]
    result = _run(
        model,
        positive,
        **args,
        adaln_sigmas=grid,
        prewarm_both_tasks=both,
        on_step=completed.append,
    )
    expanded = capacity == 6 and width == 2 and both
    expected_count = 3 if capacity is not None and capacity >= 3 else 1
    if expanded:
        expected_count *= 2
    assert len(model.plans) == expected_count
    assert len(model.calls) == 1 and completed == [0]
    torch.testing.assert_close(result, baseline, rtol=0, atol=0)
    if expanded:
        other = _inputs(3, 2, "reference" if mode == "t2i" else "t2i")
        n_other = int((~other.packed["update_mask"]).sum())
        expected = _BufferModel()
        _run(
            expected,
            other,
            sigmas=grid,
            keyframe_cond_rows=torch.zeros(n_other, 96) if n_other else None,
        )
        # The added plans must match a real request of the opposite task,
        # including its FP32 timestep bits and one/two-row GEMM geometry.
        assert all(
            torch.equal(got, want) for got, want in zip(model.plans[3:], expected.plans)
        )


@pytest.mark.parametrize("grid", [[1.0], [1.0, 0.25, 0.0]])
def test_adaln_prewarm_requires_the_same_denoising_prefix(grid):
    with pytest.raises(ValueError, match="must extend the denoising prefix"):
        _run(_BufferModel(), _inputs(3, 2), adaln_sigmas=grid)


@pytest.mark.parametrize("cached", [False, True])
def test_t2i_positive_velocity_and_forward_inputs(cached):
    positive = _inputs(3, 3)
    model = _BufferModel(cached=cached)
    result = _run(model, positive)
    # Two Euler steps over sigmas 1 -> 0.5 -> 0 add the velocity once.
    torch.testing.assert_close(result, torch.full((4, 96), 2.0 + 3), rtol=0, atol=0)
    assert model.prepare_calls == 1 and len(model.plans) == 2
    assert [c["step"] for c in model.calls] == [0, 1]
    assert [c["metadata_step"] for c in model.calls] == [0, 1]
    for i, call in enumerate(model.calls):
        # A t2i request has no reference rows to mask out of the output.
        assert call["skip_mask_out_condition"]
        assert call["metadata"] is positive.attn_metadata
        kwargs = call["kwargs"]
        assert kwargs["refiner_packed_seq_params"]["cu_seqlens_q_host"] == (0, 3, 3)
        assert kwargs["packed_seq_params"]["cu_seqlens_q_host"] == (
            0,
            3 + 4,
            positive.packed["seq_len"],
        )
        assert kwargs["x"].shape[1] == positive.packed["seq_len"]
        torch.testing.assert_close(kwargs["prompt_embeds"], positive.text_embeddings)
        if cached:
            assert int(kwargs["adaln_cache_slot"]) == i
    assert get_forward_context_or_none() is None


@pytest.mark.parametrize(
    "sigmas",
    [
        # The 6-step PDD release's pdd_config.json sigmas (rounded) and a
        # degenerate 2-step grid.
        [1.0, 0.9675237, 0.8875485, 0.75, 0.5327739, 0.2320089, 0.0],
        [1.0, 0.0005, 0.0],
    ],
)
def test_edit_steps_targets_and_keeps_references_pinned(sigmas):
    positive = _inputs(3, 3, "reference")
    n_cond = int((~positive.packed["update_mask"]).sum())
    anchor = torch.arange(n_cond * 96, dtype=torch.float32).reshape(n_cond, 96)
    initial = torch.linspace(-1, 1, (n_cond + 4) * 96).reshape(n_cond + 4, 96)
    model = _BufferModel(dtype=torch.bfloat16, state_scale=0.125, time_scale=0.75)
    result = _run(
        model,
        positive,
        initial_video_rows=initial,
        keyframe_cond_rows=anchor,
        sigmas=sigmas,
    )
    expected = initial[n_cond:].clone()
    sigma_values = torch.tensor(sigmas)
    for step, call in enumerate(model.calls):
        torch.testing.assert_close(call["rows"][:n_cond], anchor, rtol=0, atol=0)
        torch.testing.assert_close(call["rows"][n_cond:], expected, rtol=0, atol=0)
        positions = positive.packed["img_pos"][:n_cond]
        torch.testing.assert_close(
            call["times"][positions],
            torch.full((n_cond,), 0.999),
            rtol=0,
            atol=0,
        )
        assert not call["skip_mask_out_condition"]
        time = torch.tensor(1.0 - sigmas[step])
        velocity = (3.0 + 0.125 * expected + 0.75 * time).bfloat16().float()
        expected = expected + (sigma_values[step] - sigma_values[step + 1]) * velocity
    torch.testing.assert_close(result[:n_cond], anchor, rtol=0, atol=0)
    torch.testing.assert_close(result[n_cond:], expected, rtol=0, atol=0)
    assert len(model.calls) == len(sigmas) - 1


@pytest.mark.parametrize("num_points", [51, 52])
def test_sigma_points_are_not_forward_step_count(num_points):
    model = _BufferModel()
    sigmas = torch.linspace(1, 0, num_points).tolist()
    _run(model, _inputs(3, 3), sigmas=sigmas)
    assert [c["step"] for c in model.calls] == list(range(num_points - 1))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_state_dependent_predictions_use_fp32_euler_deltas(dtype):
    sigmas = [0.93, 0.61, 0.17, 0.0]
    initial = torch.linspace(-1, 1, 384).reshape(4, 96).to(dtype)
    model = _BufferModel(dtype=dtype, state_scale=0.125, time_scale=0.75)
    actual = _run(model, _inputs(3, 2.125), initial_video_rows=initial, sigmas=sigmas)
    expected = initial.float().clone()
    sigma_values = torch.tensor(sigmas, dtype=torch.float32)
    for step in range(len(sigmas) - 1):
        time = torch.tensor(1.0 - sigmas[step], dtype=torch.float32)
        # The model-dtype prediction is upcast before the fp32 update.
        velocity = (2.125 + 0.125 * expected + 0.75 * time).to(dtype).float()
        expected = expected + (sigma_values[step] - sigma_values[step + 1]) * velocity
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(model.calls) == len(sigmas) - 1
