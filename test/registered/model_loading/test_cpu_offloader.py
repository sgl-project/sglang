"""CPU offload must preserve device buffers, cached views, and eager indexers."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear
from sglang.srt.layers.offloader import OffloaderV1, OffloaderV2
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_loader.loader import PreshardedModelLoader
from sglang.srt.weight_cache.ipc_loader import IpcModelLoader
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
# backend-specific: XPU CPU-offload loads host Gemma weights into device buffers
# and must retain cached GDN/KDA views after checkpoint loading.
register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")


class CachedConvolution(nn.Module):
    def __init__(
        self,
        projection_name: str,
        device: torch.device,
        cache_name: str = "conv_weights",
    ) -> None:
        super().__init__()
        if projection_name == "qkv_conv1d":
            convolution = MergedColumnParallelLinear(
                3, [1, 1, 1], bias=False, parallel_group="replicated"
            )
        else:
            convolution = ColumnParallelLinear(
                3, 3, bias=False, parallel_group="replicated"
            )
        convolution.to(device)
        convolution.weight.data = convolution.weight.data.unsqueeze(1)
        self.add_module(projection_name, convolution)
        self.projection = nn.Linear(3, 2, bias=False, device=device)
        with torch.no_grad():
            convolution.weight.zero_()
        # Match GDN's view and KDA's squeeze before handing weights to attention.
        conv_weights = (
            convolution.weight.squeeze(1)
            if projection_name == "qkv_conv1d"
            else convolution.weight.view(3, 3)
        )
        self.attn = RadixLinearAttention(
            layer_id=0,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=1,
            head_q_dim=1,
            head_k_dim=1,
            head_v_dim=1,
            conv_weights=conv_weights,
        )
        self.cache_name = cache_name
        if cache_name != "conv_weights":
            setattr(self.attn, cache_name, self.attn.conv_weights)
            self.attn.conv_weights = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        convolution = F.conv1d(
            inputs, getattr(self.attn, self.cache_name).unsqueeze(1), groups=3
        ).squeeze(-1)
        return self.projection(convolution)


class TestCpuOffloader(CustomTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = torch.accelerator.current_accelerator(check_available=True)
        if self.device is None:
            self.skipTest("An accelerator is required to exercise CPU offload")

    @unittest.skipUnless(
        torch.cuda.is_available() and not torch.version.hip,
        "CUDA GDN input projections",
    )
    def test_gdn_input_projections_after_cpu_offload(self) -> None:
        """Offloaded GDN projections must not retain a host-only fused weight (#47896)."""
        from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

        class InputProjections(nn.Module):
            forward = Qwen3_5GatedDeltaNet._forward_input_proj

            def __init__(self, device):
                super().__init__()
                self._fused_in_proj_weight = None
                self._fused_input_proj_cpu_enabled = SimpleNamespace(value=False)
                self.alt_stream = None
                for name, width in (("in_proj_qkvz", 8), ("in_proj_ba", 4)):
                    projection = MergedColumnParallelLinear(
                        4,
                        [width],
                        bias=False,
                        params_dtype=torch.bfloat16,
                        parallel_group="replicated",
                    ).to(device)
                    projection.weight.data.normal_()
                    setattr(self, name, projection)

        for offloaded in (
            ("in_proj_qkvz",),
            ("in_proj_ba",),
            ("in_proj_qkvz", "in_proj_ba"),
        ):
            with self.subTest(offloaded=offloaded), torch.no_grad():
                model = InputProjections(self.device)
                inputs = torch.randn((2, 4), dtype=torch.bfloat16, device=self.device)
                expected = model(inputs)
                offloader = OffloaderV1(cpu_offload_max_bytes=1024**2)
                projections = [model.get_submodule(name) for name in offloaded]
                for projection in projections:
                    offloader.maybe_offload_to_cpu(projection)

                with patch(
                    "sglang.srt.models.qwen3_5.get_lora",
                    return_value=SimpleNamespace(enable_lora=False, lora_paths=None),
                ):
                    Qwen3_5GatedDeltaNet.finalize_fused_in_proj(model)

                self.assertIsNone(model._fused_in_proj_weight)
                torch.testing.assert_close(model(inputs), expected)
                for projection in projections:
                    self.assertEqual(projection.weight.device.type, "cpu")

    def test_indexer_runs_outside_offloaded_forward(self) -> None:
        """Graph replay must run indexer projections after the layer restores CPU weights."""
        with torch.no_grad():
            attention = nn.Sequential()
            attention.add_module(
                "indexer",
                nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 3)),
            )
            model = nn.Sequential(attention, nn.Linear(3, 2)).to(self.device)
            inputs = torch.randn(2, 3, device=self.device)
            expected = model(inputs)
            expected_indexer = attention.indexer(inputs)

            model = OffloaderV1(cpu_offload_max_bytes=1024**2).maybe_offload_to_cpu(
                model
            )
            torch.testing.assert_close(model(inputs), expected)
            self.assertEqual(model[1].weight.device.type, "cpu")

            # DSA eager breaks resolve the indexer from the forward context and
            # invoke it after the offloaded layer's functional_call has returned.
            torch.testing.assert_close(attention.indexer(inputs), expected_indexer)

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

    def test_presharded_rebind_refreshes_cpu_offloaded_gemma_buffer(self) -> None:
        norm = GemmaRMSNorm(16).to(self.device)
        norm = OffloaderV1(cpu_offload_max_bytes=1024**2).maybe_offload_to_cpu(norm)
        buffer_address = norm.gemma_weight.data_ptr()
        loaded_weights = torch.randn(16, device="cpu")
        norm.weight.data.copy_(loaded_weights)
        norm.gemma_weight.zero_()

        PreshardedModelLoader._rebind_parameter_aliases(norm)

        self.assertEqual(norm.weight.device.type, "cpu")
        self.assertEqual(norm.gemma_weight.device.type, self.device.type)
        self.assertEqual(norm.gemma_weight.data_ptr(), buffer_address)
        torch.testing.assert_close(norm.gemma_weight.cpu(), loaded_weights + 1.0)

    def test_checkpoint_load_updates_cached_convolution(self) -> None:
        for projection_name, cache_name in (
            ("qkv_conv1d", "conv_weights"),
            ("conv1d", "conv_weights"),
            ("temporal_projection", "cached_kernel"),
        ):
            with self.subTest(projection_name=projection_name), torch.no_grad():
                model = CachedConvolution(projection_name, self.device, cache_name)
                model = OffloaderV1(cpu_offload_max_bytes=1024**2).maybe_offload_to_cpu(
                    model
                )
                weight = getattr(model, projection_name).weight
                weight.weight_loader(weight, torch.ones_like(weight, device="cpu"))
                model.projection.weight.fill_(1)

                output = model(torch.ones((1, 3, 3), device=self.device))

                torch.testing.assert_close(
                    output, torch.full((1, 2), 9.0, device=self.device)
                )
                self.assertEqual(model.projection.weight.device.type, "cpu")

    def test_v2_whitelist_preserves_cached_convolution_storage(self) -> None:
        with torch.no_grad():
            model = CachedConvolution("conv1d", self.device)
            convolution = model.conv1d
            convolution.bias = nn.Parameter(torch.zeros(3, device=self.device))
            # Check V2's parameter selection before its CUDA-only stream transfers.
            # The cached view is held outside the selected projection submodule.
            with (
                patch(
                    "sglang.srt.layers.offloader.get_parallel",
                    return_value=SimpleNamespace(dp_rank=0, num_dp_ranks=1),
                ),
                patch("sglang.srt.layers.offloader.get_stream", return_value=None),
                patch("torch.cuda.memory_allocated", return_value=0),
                patch.dict(os.environ, {"SGLANG_RUN_ID": "test-cpu-offloader"}),
            ):
                offloader = OffloaderV2(
                    group_size=1, num_in_group=1, prefetch_step=1, mode="cpu"
                )
                offloader.wrap_modules(
                    iter([model]),
                    submodule_accessor=lambda module: module.conv1d,
                    whitelist_param_names_creator=lambda module: ["weight", "bias"],
                )
            convolution.weight.weight_loader(
                convolution.weight, torch.ones((3, 1, 3), device="cpu")
            )
            output = F.conv1d(
                torch.ones((1, 3, 3), device=self.device),
                model.attn.conv_weights.unsqueeze(1),
                groups=3,
            )

            torch.testing.assert_close(
                output, torch.full((1, 3, 1), 3.0, device=self.device)
            )
            self.assertEqual(convolution.weight.device.type, self.device.type)
            self.assertEqual(convolution.bias.device.type, "cpu")

    def test_loaders_rebuild_replaced_convolution_view(self) -> None:
        for rebind in (
            PreshardedModelLoader._rebind_parameter_aliases,
            IpcModelLoader._rebuild_stale_views,
        ):
            with self.subTest(loader=rebind.__qualname__), torch.no_grad():
                model = CachedConvolution("conv1d", self.device)
                model.conv1d.weight = nn.Parameter(torch.ones_like(model.conv1d.weight))
                model.projection.weight.fill_(1)

                rebind(model)
                output = model(torch.ones((1, 3, 3), device=self.device))

                torch.testing.assert_close(
                    output, torch.full((1, 2), 9.0, device=self.device)
                )


if __name__ == "__main__":
    unittest.main()
