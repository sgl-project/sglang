import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline as diffusers_pipeline
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline import (
    DiffusersExecutionStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


class _FakeParam:
    def __init__(self, device: str):
        self.device = device


class _FakeModule:
    def __init__(self, device: str):
        self._param = _FakeParam(device)

    def parameters(self):
        return iter([self._param])


class _FakeDiffusersPipe:
    def __init__(self, device: str):
        self.transformer = _FakeModule(device)


class TestDiffusersGeneratorDevice(unittest.TestCase):
    def setUp(self):
        # Simulate an offloaded diffusers pipeline whose modules currently live on CPU.
        self.server_args_patcher = patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args",
            return_value=SimpleNamespace(comfyui_mode=False),
        )
        self.server_args_patcher.start()
        self.stage = DiffusersExecutionStage(_FakeDiffusersPipe(device="cpu"))

    def tearDown(self):
        self.server_args_patcher.stop()

    @patch.object(diffusers_pipeline.current_platform, "device_type", "cuda")
    @patch("sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline.torch.Generator")
    def test_seeded_generator_uses_requested_device_not_component_device(
        self, mock_generator_ctor
    ):
        mock_generator = MagicMock()
        mock_generator.manual_seed.return_value = "cuda-seeded-generator"
        mock_generator_ctor.return_value = mock_generator

        batch = Req(
            sampling_params=SamplingParams(
                prompt="test prompt", seed=1234, generator_device="cuda"
            )
        )

        kwargs = self.stage._build_pipeline_kwargs(batch)

        mock_generator_ctor.assert_called_once_with(device="cuda")
        mock_generator.manual_seed.assert_called_once_with(1234)
        self.assertEqual(kwargs["generator"], "cuda-seeded-generator")

    @patch.object(diffusers_pipeline.current_platform, "device_type", "cuda")
    @patch("sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline.torch.Generator")
    def test_seeded_generator_respects_explicit_cpu_generator_device(
        self, mock_generator_ctor
    ):
        mock_generator = MagicMock()
        mock_generator.manual_seed.return_value = "cpu-seeded-generator"
        mock_generator_ctor.return_value = mock_generator

        batch = Req(
            sampling_params=SamplingParams(
                prompt="test prompt", seed=5678, generator_device="cpu"
            )
        )

        kwargs = self.stage._build_pipeline_kwargs(batch)

        mock_generator_ctor.assert_called_once_with(device="cpu")
        mock_generator.manual_seed.assert_called_once_with(5678)
        self.assertEqual(kwargs["generator"], "cpu-seeded-generator")


if __name__ == "__main__":
    unittest.main()
