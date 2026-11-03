# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
    Flux3ImageSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.task_type import ModelTaskType
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
from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import build_layout
from sglang.multimodal_gen.runtime.pipelines.qwen_image21 import QwenImage21Pipeline
from sglang.multimodal_gen.runtime.pipelines_core import Req


@pytest.mark.parametrize("edit", [False, True])
@pytest.mark.parametrize("flow_map", [False, True])
def test_flux3_encoding_survives_each_transfer_hop(edit, flow_map):
    """Preserve nested embeddings, reference coordinates, crop, and sampling type."""
    params = Flux3ImageSamplingParams(
        flux3_mode="i2i" if edit else None, guidance_scale=1.0
    )
    params._explicit_fields = {"width"} if edit else set()
    req = Req(request_id="flux3-transfer", prompt="test", sampling_params=params)
    params.apply_request_extra(req)
    text = [
        (torch.randn(3, 8), torch.arange(12).reshape(3, 4))
        for _ in range(1 if flow_map else 2)
    ]
    condition = (torch.randn(4, 8), torch.zeros(4, 4)) if edit else None
    req.extra.update(
        flux3_text=text,
        flux3_condition=condition,
        flux3_crop=(0, 0, 63, 61) if edit else None,
    )
    scheduler = SimpleNamespace(_disagg_sampling_params_cls=Flux3ImageSamplingParams)
    for _ in range(3):
        tensors, scalars = extract_transfer_fields(req)
        metadata, buffers = pack_tensors(tensors, scalars)
        received, scalars = unpack_tensors([metadata, *[w._view for w in buffers]])
        req = SchedulerDisaggMixin._build_disagg_req(scheduler, scalars, received)
        assert type(req.sampling_params) is Flux3ImageSamplingParams
        assert req.flux3_mode == params.flux3_mode
        assert req.guidance_scale == 1.0
        assert req.sampling_params._explicit_fields == params._explicit_fields
        torch.testing.assert_close(req.extra["flux3_text"], text, atol=0, rtol=0)
        torch.testing.assert_close(
            req.extra["flux3_condition"], condition, atol=0, rtol=0
        )
        assert req.extra["flux3_crop"] == ([0, 0, 63, 61] if edit else None)


def test_task_type_survives_disagg_transfer():
    """A GLM request must cross the JSON transport after AR completes."""
    req = Req(
        request_id="glm-transfer",
        prompt="test",
        sampling_params=SamplingParams(task_type=ModelTaskType.TI2I),
    )

    tensors, scalars = extract_transfer_fields(req)
    metadata, buffers = pack_tensors(tensors, scalars)
    received, scalars = unpack_tensors([metadata, *[w._view for w in buffers]])
    rebuilt = SchedulerDisaggMixin._build_disagg_req(None, scalars, received)

    assert ModelTaskType.parse(rebuilt.task_type) == ModelTaskType.TI2I


def test_qwen21_disagg_encoder_loads_condition_vae():
    pipeline = object.__new__(QwenImage21Pipeline)
    modules = filter_modules_for_role(
        pipeline._required_config_modules,
        RoleType.ENCODER,
        extra_allowed_modules=pipeline._get_extra_allowed_modules_for_role(
            RoleType.ENCODER, "ti2i"
        ),
    )
    assert set(modules) == {"processor", "text_encoder", "vae", "scheduler"}


@pytest.mark.parametrize("edit", [False, True])
def test_qwen21_conditioning_survives_disagg_transfer(edit):
    slots = [False, True, False] if edit else [False, False, False]
    shapes = [(1, 2, 4), (1, 4, 4)] if edit else [(1, 4, 4)]
    condition = dict(
        layouts=[build_layout(slots, shapes, (8, 12, 12), "cpu")],
        condition_latents=torch.randn(1, 8, 4) if edit else None,
        prefix_caches=[[{}, {}]],
    )
    req = Req(request_id="qwen21-transfer", prompt="test")
    req.extra = dict(qwen21_positive=condition, qwen21_negative=condition, mu=0.7)
    req.extra["_local"] = object()
    tensors, scalars = extract_transfer_fields(req)
    metadata, buffers = pack_tensors(tensors, scalars)
    received, scalars = unpack_tensors([metadata, *[w._view for w in buffers]])
    rebuilt = SchedulerDisaggMixin._build_disagg_req(None, scalars, received)
    assert "_local" not in rebuilt.extra
    assert rebuilt.extra["mu"] == 0.7
    for name in ("qwen21_positive", "qwen21_negative"):
        restored = rebuilt.extra[name]
        for key, expected in condition["layouts"][0].items():
            actual = restored["layouts"][0][key]
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            else:
                assert actual == expected
        assert restored["prefix_caches"] == [[{}, {}]]
        if edit:
            torch.testing.assert_close(
                restored["condition_latents"],
                condition["condition_latents"],
                atol=0,
                rtol=0,
            )
        else:
            assert restored["condition_latents"] is None
