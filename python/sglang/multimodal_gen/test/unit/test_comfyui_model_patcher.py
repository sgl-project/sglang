# SPDX-License-Identifier: Apache-2.0
"""ComfyUI must keep tracking the model when a temporary SGLD clone expires."""

import gc
import weakref

import pytest
import torch

model_management = pytest.importorskip("comfy.model_management")

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.model_patcher import (
    SGLDModelPatcher,
)


def test_expired_clone_returns_loaded_model_to_live_parent():
    model = torch.nn.Linear(1, 1)
    device = torch.device("cpu")
    parent = SGLDModelPatcher(model, device, device, model_type="minimax_h3")
    tracked = model_management.LoadedModel(parent.clone().clone())
    tracked.real_model = weakref.ref(model)

    gc.collect()

    assert not tracked.is_dead()
    assert tracked.model is parent
    assert tracked.real_model() is model
