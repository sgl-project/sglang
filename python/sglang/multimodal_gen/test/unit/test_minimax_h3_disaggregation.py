# SPDX-License-Identifier: Apache-2.0
"""H3 role loading and the real encoder/denoiser/decoder wire boundaries."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import (
    RoleType,
    filter_modules_for_role,
)
from sglang.multimodal_gen.runtime.disaggregation.scheduler_mixin import (
    SchedulerDisaggMixin,
    extract_transfer_fields,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import (
    pack_tensors,
    unpack_tensors,
)
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY,
    MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY,
    MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY,
    MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY,
    MINIMAX_H3_SIGMAS_EXTRA_KEY,
    MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
    minimax_h3_prepare_for_queue,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY,
    minimax_h3_plan_from_batch,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    minimax_h3_condition_noise_aug,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.latent_preparation import (
    MiniMaxH3LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


@pytest.mark.parametrize("variant", ["fl2va", "ref2va"])
@pytest.mark.parametrize("role", list(RoleType))
def test_h3_role_admission(variant, role):
    args = ServerArgs.from_kwargs(
        model_path="MiniMaxAI/MiniMax-H3",
        model_variant=variant,
        disagg_role=role,
        num_gpus=1,
        log_level="error",
    )
    assert isinstance(args.pipeline_config, MiniMaxH3PipelineConfig)
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.validate_disagg_role(role)


@pytest.mark.parametrize(
    ("role", "expected_modules", "expected_stages"),
    [
        (
            RoleType.ENCODER,
            {"processor", "tokenizer", "text_encoder", "video_vae", "audio_vae"},
            [
                "InputValidationStage",
                "MiniMaxH3TextEncodingStage",
                "MiniMaxH3VisualEncodingStage",
                "MiniMaxH3AudioEncodingStage",
                "MiniMaxH3LatentPreparationStage",
                "MiniMaxH3TimestepPreparationStage",
            ],
        ),
        (RoleType.DENOISER, {"transformer"}, ["MiniMaxH3DenoisingStage"]),
        (RoleType.DECODER, {"video_vae", "audio_vae"}, ["MiniMaxH3DecodingStage"]),
    ],
)
def test_h3_role_modules_and_stages(role, expected_modules, expected_stages):
    pipeline = object.__new__(MiniMaxH3Pipeline)
    modules = filter_modules_for_role(
        pipeline._required_config_modules,
        role,
        extra_allowed_modules=pipeline._get_extra_allowed_modules_for_role(
            role, "ti2v"
        ),
    )
    assert set(modules) == expected_modules
    pipeline.modules = {name: torch.nn.Module() for name in modules}
    pipeline._disagg_role = role
    pipeline._stages = []
    pipeline._stage_name_mapping = {}
    pipeline.create_pipeline_stages(
        SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig())
    )
    assert [type(stage).__name__ for stage in pipeline._stages] == expected_stages


def _encoded_req(task):
    with_video = task == "ref2va_video"
    if with_video:
        task = "ref2va"
    conditions = []
    target = {"short_edge": 64, "aspect_ratio": "1:1", "duration_seconds": 4.0}
    if task == "fl2va":
        conditions = [
            {
                "type": "image",
                "uri": "/encoder/first.png",
                "role": "keyframe",
                "frame_index": 0,
            }
        ]
        target["aspect_ratio"] = "auto"
    elif task == "ref2va":
        conditions = [
            {"type": "image", "uri": "/encoder/ref.png", "role": "reference"},
            {"type": "audio", "uri": "/encoder/ref.wav", "role": "reference"},
        ]
        if with_video:
            conditions.append(
                {"type": "video", "uri": "/encoder/ref.mp4", "role": "reference"}
            )
        else:
            target.pop("duration_seconds")
    sp = MiniMaxH3SamplingParams(
        prompt="A musician plays a piano",
        task=task,
        conditions=conditions,
        target=target,
        seed=123,
        num_inference_steps=3,
        imgvid_cond_noise_aug_for_inference=0.8,
        audio_cond_noise_aug_for_inference=0.6,
        flow_shift=7.0,
        audio_flow_shift=2.0,
    )
    req = Req(sampling_params=sp, request_id=f"h3-{task}")
    req.extra.update(sp.build_request_extra())

    def probe(_batch, _uri, *, condition_type, condition_index):
        return dict(
            display_width=64,
            display_height=96,
            has_audio=condition_type == "audio",
            audio_duration_seconds=4.5,
            video_duration_seconds=4.5,
        )

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "minimax_h3.prequeue.minimax_h3_probe_material",
        side_effect=probe,
    ):
        minimax_h3_prepare_for_queue(req)
    MiniMaxH3LatentPreparationStage().forward(req, None)
    MiniMaxH3TimestepPreparationStage().forward(req, None)
    req.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY] = {
        "positive": {
            "hidden_states": torch.randn(3, 5120, dtype=torch.bfloat16),
            "text_len": 3,
            "text_token_tags": torch.zeros(3, dtype=torch.long),
            "text_video_token_mask": torch.zeros(3, dtype=torch.bool),
        }
    }
    if task == "fl2va":
        req.extra[MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY] = {
            "rows": torch.randn(6, 96),
            "latent_h": 6,
            "latent_w": 4,
            "semantic_frame_indices": [0],
            "pixel_frame_indices": [0],
            "frame_count": req.num_frames,
            "keyframes": [{"frame_index": 0, "resolved_frame_index": 0}],
        }
    elif task == "ref2va":
        req.extra[MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY] = {
            0: {"rows": torch.randn(4, 96), "latent_h": 4, "latent_w": 4}
        }
        req.extra[MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY] = {
            1: {"rows": torch.randn(6, 32), "audio_t": 3}
        }
        if with_video:
            req.extra[MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY] = {
                2: {
                    "rows": torch.randn(8, 96),
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                }
            }
    return req


def _transfer(req, device):
    tensors, scalars = extract_transfer_fields(req)
    metadata, buffers = pack_tensors(tensors, scalars)
    tensors, scalars = unpack_tensors(
        [metadata, *[buffer._view for buffer in buffers]], device=device
    )
    scheduler = SimpleNamespace(worker=SimpleNamespace(pipeline=MiniMaxH3Pipeline))
    return SchedulerDisaggMixin._build_disagg_req(scheduler, scalars, tensors)


@pytest.mark.parametrize("task", ["t2va", "fl2va", "ref2va", "ref2va_video"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_h3_transfer_preserves_frozen_plan_sampling_and_cpu_payloads(task, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required to exercise receiver GPU placement")
    req = _encoded_req(task)
    original_plan = minimax_h3_plan_from_batch(req)
    rebuilt = _transfer(req, device)
    assert isinstance(rebuilt.sampling_params, MiniMaxH3SamplingParams)
    assert minimax_h3_plan_from_batch(rebuilt) == original_plan
    assert rebuilt.num_frames == req.num_frames
    assert rebuilt.height == req.height and rebuilt.width == req.width
    assert minimax_h3_condition_noise_aug(rebuilt.sampling_params) == (0.8, 0.6)
    assert rebuilt.sampling_params.audio_flow_shift == 2.0
    assert (
        rebuilt.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]
        == req.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]
    )
    from torch.utils._pytree import tree_flatten

    for key, value in req.extra.items():
        if key == MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY:
            continue
        expected, _ = tree_flatten(value)
        actual, _ = tree_flatten(rebuilt.extra[key])
        for old, new in zip(expected, actual, strict=True):
            if isinstance(old, torch.Tensor):
                assert new.device == old.device
                torch.testing.assert_close(new, old, rtol=0, atol=0)

    # The second boundary carries decoded-layout video AND stereo audio latents.
    rebuilt.latents = torch.randn(1, 24, 2, 4, 4, device=device)
    rebuilt.audio_latents = torch.randn(2, 32, 3, device=device)
    decoded_req = _transfer(rebuilt, device)
    torch.testing.assert_close(decoded_req.latents, rebuilt.latents, rtol=0, atol=0)
    torch.testing.assert_close(
        decoded_req.audio_latents, rebuilt.audio_latents, rtol=0, atol=0
    )
    assert minimax_h3_plan_from_batch(decoded_req) == original_plan


def test_subclass_transfer_keeps_explicit_base_default():
    req = _encoded_req("t2va")
    # Base and H3 defaults differ. This explicit choice must not be elided
    # just because it equals the base default.
    req.width = SamplingParams().width
    rebuilt = _transfer(req, "cpu")
    assert rebuilt.width == req.width


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA stream ordering")
def test_prefetch_cpu_extras_wait_for_async_tensor_load():
    req = Req(request_id="async-cpu-extra", prompt="test")
    req.extra["tags"] = torch.full((8,), 42, dtype=torch.int64)
    _, scalars = extract_transfer_fields(req)
    stream = torch.cuda.Stream()
    payload = torch.zeros(8, device="cuda", dtype=torch.int64)
    stream.wait_stream(torch.cuda.current_stream())

    def load(*_args, **_kwargs):
        with torch.cuda.stream(stream):
            # Make the cross-stream race observable even for this tiny payload.
            torch.cuda._sleep(20_000_000)
            payload.fill_(42)
            event = stream.record_event()
        return {"_extra_tensor_tree_tags": [payload]}, event

    scheduler = SimpleNamespace(
        worker=SimpleNamespace(local_rank=torch.cuda.current_device()),
        _disagg_metrics=None,
        _transfer_stream=stream,
        _transfer_manager=SimpleNamespace(load_tensors_async=load),
    )
    scheduler._build_disagg_req = lambda scalars, tensors: (
        SchedulerDisaggMixin._build_disagg_req(scheduler, scalars, tensors)
    )
    rebuilt, _, _, _ = SchedulerDisaggMixin._prefetch_transfer_ready(
        scheduler, {"request_id": req.request_id, "scalar_fields": scalars}
    )
    torch.testing.assert_close(rebuilt.extra["tags"], req.extra["tags"], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend selection")
@pytest.mark.parametrize("ambient_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("backend_name", ["FA", "TORCH_SDPA"])
@pytest.mark.parametrize("component", ["video", "audio"])
def test_h3_vae_backend_does_not_depend_on_dit_loading(
    ambient_dtype, backend_name, component
):
    from sglang.multimodal_gen.runtime.layers.attention import layer, selector
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_audio_vae.audio_vae import (
        CausalAttention,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
        Attention,
    )
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang.multimodal_gen.runtime.server_args import server_args

    args = ServerArgs.from_kwargs(
        model_path="MiniMaxAI/MiniMax-H3", model_variant="fl2va", num_gpus=1
    )
    backend = AttentionBackendEnum[backend_name]
    with (
        patch.object(layer, "get_compute_dtype", return_value=ambient_dtype),
        patch.object(selector, "forced_attn_backend", backend),
        patch.object(server_args, "get_global_server_args", return_value=args),
    ):
        attention = (
            Attention(heads=2, dim_head=64)
            if component == "video"
            else CausalAttention(in_dim=128, out_dim=128, num_heads=2)
        )
    assert attention.attn.backend == backend
