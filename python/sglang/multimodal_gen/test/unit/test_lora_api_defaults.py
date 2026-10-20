# SPDX-License-Identifier: Apache-2.0
"""Merge strength keeps its 1.0 default; None is an explicit opt-in."""

import inspect
import logging
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
    MergeLoraWeightsReq,
)
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
    DiffGenerator,
)
from sglang.multimodal_gen.runtime.entrypoints.openai import common_api
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.pipelines_core.lora import format_adapter


def test_default_merge_request_uses_strength_one():
    assert MergeLoraWeightsReq().strength == 1.0


def test_explicit_none_merge_request_keeps_adapter_strength():
    assert MergeLoraWeightsReq(strength=None).strength is None


def test_public_merge_entrypoints_default_to_one():
    endpoint_default = (
        inspect.signature(common_api.merge_lora_weights).parameters["strength"].default
    )
    assert endpoint_default.default == 1.0
    for fn in (DiffGenerator.merge_lora_weights, GPUWorker.merge_lora_weights):
        assert inspect.signature(fn).parameters["strength"].default == 1.0


def test_diffusers_conversion_util_is_tried_first():
    sd = {"a.lora_down.weight": torch.zeros(1)}
    converted = {"a.lora_A.weight": torch.zeros(1)}
    log = logging.getLogger(__name__)
    with patch.object(
        format_adapter.lcu,
        "maybe_convert_state_dict",
        create=True,
        return_value=converted,
    ) as mock_convert:
        out = format_adapter.convert_lora_state_dict_by_format(
            sd, format_adapter.LoRAFormat.STANDARD, log
        )
    mock_convert.assert_called_once()
    assert out == converted


def test_diffusers_conversion_failure_falls_back_to_internal():
    sd = {"a.lora_A.weight": torch.zeros(1)}
    log = logging.getLogger(__name__)
    with patch.object(
        format_adapter.lcu,
        "maybe_convert_state_dict",
        create=True,
        side_effect=RuntimeError("boom"),
    ):
        out = format_adapter.convert_lora_state_dict_by_format(
            sd, format_adapter.LoRAFormat.STANDARD, log
        )
    assert set(out) == set(sd)
