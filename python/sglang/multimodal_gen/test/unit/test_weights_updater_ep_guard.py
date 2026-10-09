# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.post_training.weights_updater import (
    LORA_MERGE_WEIGHT_UPDATE_MODE,
    WeightsUpdater,
    _reject_expert_parallel_modules,
)


class _EpTransformer(torch.nn.Module):
    def __init__(self, enabled: bool):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.ep_info = SimpleNamespace(enabled=enabled)


def _updater(enabled: bool) -> WeightsUpdater:
    pipeline = SimpleNamespace(modules={"transformer": _EpTransformer(enabled)})
    return WeightsUpdater(pipeline)


def test_update_weights_from_disk_rejects_expert_parallel(tmp_path):
    success, message = _updater(True).update_weights_from_disk(str(tmp_path))

    assert not success
    assert "expert parallelism" in message


def test_update_weights_from_tensor_rejects_expert_parallel():
    success, message = _updater(True).update_weights_from_tensor(
        [("transformer.linear.weight", torch.zeros(2, 2))]
    )

    assert not success
    assert "expert parallelism" in message


def test_lora_merge_updates_bypass_expert_parallel_guard():
    updater = _updater(True)
    with patch.object(
        WeightsUpdater,
        "_update_lora_from_tensor",
        return_value=(True, "lora ok"),
    ) as update_lora:
        success, message = updater.update_weights_from_tensor(
            [("transformer.linear.weight", torch.zeros(2, 2))],
            weight_update_mode=LORA_MERGE_WEIGHT_UPDATE_MODE,
        )

    update_lora.assert_called_once()
    assert success, message


def test_reject_expert_parallel_modules_allows_dense_and_ep_free_modules():
    disabled = SimpleNamespace(ep_info=SimpleNamespace(enabled=False))
    ep_free = object()
    _reject_expert_parallel_modules([("transformer", disabled), ("vae", ep_free)])
