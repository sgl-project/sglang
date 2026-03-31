import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)


class _FakeModule:
    def __init__(self, device: str):
        self._device = torch.device(device)
        self.to_calls = []

    def parameters(self):
        return iter([SimpleNamespace(device=self._device)])

    def to(self, device):
        self.to_calls.append(device)
        self._device = torch.device(device)
        return self


class TestDenoisingStageDevicePlacement(unittest.TestCase):
    def test_offloads_active_xpu_model_before_loading_cpu_model(self):
        model_to_use = _FakeModule("cpu")
        model_to_offload = _FakeModule("xpu:0")
        server_args = SimpleNamespace(
            dit_cpu_offload=True,
            use_fsdp_inference=False,
        )

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising.get_local_torch_device",
            return_value=torch.device("xpu:0"),
        ):
            DenoisingStage._manage_device_placement(
                SimpleNamespace(),
                model_to_use,
                model_to_offload,
                server_args,
            )

        self.assertEqual(model_to_offload.to_calls, ["cpu"])
        self.assertEqual(model_to_use.to_calls, [torch.device("xpu:0")])


if __name__ == "__main__":
    unittest.main()