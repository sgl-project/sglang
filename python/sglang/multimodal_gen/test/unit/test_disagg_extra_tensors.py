# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

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
