import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
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


class TestTextEncodingStageOffload(unittest.TestCase):
    def test_offloads_text_encoders_after_forward_when_requested(self):
        text_encoder = _FakeModule("xpu:0")
        stage = TextEncodingStage(text_encoders=[text_encoder], tokenizers=[object()])
        stage.server_args = SimpleNamespace(
            text_encoder_cpu_offload=True,
            use_fsdp_inference=False,
        )

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding.current_platform.is_xpu",
            return_value=True,
        ), patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding.torch.xpu.empty_cache"
        ) as empty_cache:
            stage.offload_model()

        self.assertEqual(text_encoder.to_calls, ["cpu"])
        empty_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()