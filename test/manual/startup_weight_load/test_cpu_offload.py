"""Check mixed CPU/CUDA parameter storage across capture and real weight loading."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.model_executor.model_runner_components.startup_weight_load import (
    ModelStorageManifest,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_context
from sglang.srt.utils.offloader import OffloaderV1
from sglang.test.test_utils import CustomTestCase


class _SmallModel(nn.Module):
    def __init__(self, quantized=False):
        super().__init__()
        quant = (
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=[128, 128],
            )
            if quantized
            else None
        )
        with torch.device("cuda"):
            self.linear = ReplicatedLinear(
                256,
                256,
                bias=not quantized,
                params_dtype=torch.bfloat16,
                quant_config=quant,
            )
        # Offload the first parameter, leaving bias/scales on CUDA.
        self.offloader = OffloaderV1(cpu_offload_max_bytes=1)
        self.offloader.maybe_offload_to_cpu(self.linear)

    def forward(self, x):
        return self.linear(x)[0]

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, value in weights:
            param = params[name]
            param.weight_loader(param, value)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestCPUOffloadReload(CustomTestCase):
    @torch.no_grad()
    def _check_replay(self, quantized):
        torch.manual_seed(2026)
        self.addCleanup(
            setattr,
            fp8_utils,
            "FP8_GEMM_RUNNER_BACKEND",
            fp8_utils.FP8_GEMM_RUNNER_BACKEND,
        )
        with get_context().override_server_args(fp8_gemm_runner_backend="triton"):
            fp8_utils.initialize_fp8_gemm_config()
            model, serial = _SmallModel(quantized), _SmallModel(quantized)
            loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
            model_config = SimpleNamespace(dtype=torch.bfloat16)
            loader.prepare_model_for_capture(
                model=model,
                model_config=model_config,
                target_device=torch.device("cuda"),
            )
            model.offloader.post_init()
            manifest = ModelStorageManifest.capture(model)
            self.assertEqual(model.linear.weight.device.type, "cpu")
            self.assertTrue(model.linear.weight.is_pinned())
            input_ = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    model(input_)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = model(input_)
            graph.replay()
            sentinel_output = output.clone()
            torch.cuda.synchronize()

            weights = []
            for name, param in model.named_parameters():
                value = torch.randn(param.shape, dtype=torch.float32)
                if "scale" in name:
                    value = value.abs() * 0.01 + 0.01
                weights.append((name, value.to(param.dtype)))
            with patch.object(
                loader, "_get_weights_iterator", return_value=iter(weights)
            ):
                loader.commit_model_weights(
                    model=model,
                    model_config=model_config,
                    resolved_sources=(SimpleNamespace(source=None),),
                    target_device=torch.device("cuda"),
                    startup_prefetch_active=False,
                )
            loader.load_weights_and_postprocess(
                serial, iter(weights), torch.device("cuda")
            )
            expected = serial(input_)
            for _ in range(3):
                graph.replay()
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
            self.assertFalse(torch.equal(output, sentinel_output))
            self.assertEqual(manifest.changed_names(model), ())
            self.assertEqual(manifest.unchanged_parameter_names(1e-3), ())

    def test_unquantized_partial_offload_replay(self):
        self._check_replay(False)

    def test_block_fp8_partial_offload_replay(self):
        if torch.cuda.get_device_capability() < (8, 9):
            self.skipTest("Block FP8 requires SM89 or newer")
        self._check_replay(True)

    @torch.no_grad()
    def test_capture_postprocess_uses_normal_device_staging(self):
        model = _SmallModel()
        loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
        method = model.linear.quant_method
        original = method.process_weights_after_loading
        devices = []

        def observe(layer):
            devices.append(layer.weight.device.type)
            return original(layer)

        with patch.object(method, "process_weights_after_loading", side_effect=observe):
            loader.prepare_model_for_capture(
                model=model,
                model_config=SimpleNamespace(dtype=torch.bfloat16),
                target_device=torch.device("cuda"),
            )
        self.assertEqual(devices, ["cuda"])
        self.assertEqual(model.linear.weight.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
