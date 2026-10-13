"""CPU offload must preserve device buffers and cached convolution views."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.utils.offloader import OffloaderV1
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")


class CachedConvolution(nn.Module):
    def __init__(self, projection_name: str, device: torch.device) -> None:
        super().__init__()
        self.add_module(
            projection_name,
            nn.Conv1d(2, 2, 3, groups=2, bias=False, device=device),
        )
        self.projection = nn.Linear(2, 2, bias=False, device=device)
        self.attn = nn.Module()
        convolution = getattr(self, projection_name)
        with torch.no_grad():
            convolution.weight.zero_()
        self.attn.conv_weights = convolution.weight.view(2, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        convolution = F.conv1d(
            inputs, self.attn.conv_weights.unsqueeze(1), groups=2
        ).squeeze(-1)
        return self.projection(convolution)


class TestCpuOffloader(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.accelerator.current_accelerator(check_available=True)
        if self.device is None:
            self.skipTest("An accelerator is required to exercise CPU offload")

    def test_cpu_offloaded_gemma_weight_fills_device_buffer(self) -> None:
        norm = GemmaRMSNorm(16).to(self.device)
        norm = OffloaderV1(cpu_offload_max_bytes=1024**2).maybe_offload_to_cpu(norm)
        buffer_address = norm.gemma_weight.data_ptr()
        loaded_weights = torch.randn(16, device="cpu")

        norm.weight.weight_loader(norm.weight, loaded_weights)

        self.assertEqual(norm.weight.device.type, "cpu")
        self.assertEqual(norm.gemma_weight.device.type, self.device.type)
        self.assertEqual(norm.gemma_weight.data_ptr(), buffer_address)
        torch.testing.assert_close(norm.gemma_weight.cpu(), loaded_weights + 1.0)

    def test_checkpoint_load_updates_cached_convolution(self) -> None:
        for projection_name in ("qkv_conv1d", "conv1d"):
            with self.subTest(projection_name=projection_name), torch.no_grad():
                model = CachedConvolution(projection_name, self.device)
                model = OffloaderV1(
                    cpu_offload_max_bytes=1024**2
                ).maybe_offload_to_cpu(model)
                getattr(model, projection_name).weight.fill_(1)
                model.projection.weight.fill_(1)

                output = model(torch.ones((1, 2, 3), device=self.device))

                torch.testing.assert_close(
                    output, torch.full((1, 2), 6.0, device=self.device)
                )
                self.assertEqual(model.projection.weight.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
