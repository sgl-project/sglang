"""Reproduce the SM90 MegaMoE reload limitation on one Hopper GPU.

PASS means the known wrong-result case was reproduced, not that overlap is
supported. This uses real loader prepare/commit and FP8 postprocess, with a
captured PyTorch consumer of the packed weights, not the fused MegaMoE kernel.
No checkpoint download or engine launch is needed. Run this file explicitly;
it is intentionally excluded from default test discovery.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.model_executor.model_runner_components.startup_weight_load import (
    ModelStorageManifest,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_flags
from sglang.test.test_utils import CustomTestCase


class _SmallExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_method = Fp8MoEMethod(
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                weight_block_size=[128, 128],
            )
        )
        for name, shape, dtype in (
            ("w13_weight", (2, 256, 128), torch.float8_e4m3fn),
            ("w2_weight", (2, 128, 128), torch.float8_e4m3fn),
            ("w13_weight_scale_inv", (2, 2, 1), torch.float32),
            ("w2_weight_scale_inv", (2, 1, 1), torch.float32),
        ):
            self.register_parameter(
                name,
                nn.Parameter(
                    torch.empty(shape, device="cuda", dtype=dtype),
                    requires_grad=False,
                ),
            )

    def load_weights(self, weights):
        parameters = dict(self.named_parameters())
        for name, value in weights:
            parameters[name].copy_(value)


def _checkpoint(model):
    generator = torch.Generator(device="cuda").manual_seed(35259)
    values = []
    for name, parameter in model.named_parameters():
        if "scale" in name:
            value = 0.25 + torch.rand(
                parameter.shape, device="cuda", generator=generator
            )
        else:
            value = 0.125 * torch.randn(
                parameter.shape, device="cuda", generator=generator
            )
        values.append((name, value.to(parameter.dtype)))
    return tuple(values)


def _reference_packed_consumer(model, inputs):
    packed, scale13 = model.mega_l1_weights
    down, scale2 = model.mega_l2_weights
    experts, rows, hidden = packed.shape
    interleaved = packed.float().reshape(experts, rows // 16, 2, 8, hidden)
    gate = interleaved[:, :, 0].reshape(experts, rows // 2, hidden)
    up = interleaved[:, :, 1].reshape(experts, rows // 2, hidden)
    canonical = torch.cat((gate, up), dim=1)
    expanded13 = scale13.repeat_interleave(128, 1).repeat_interleave(128, 2)
    expanded2 = scale2.repeat_interleave(128, 1).repeat_interleave(128, 2)
    gate, up = torch.bmm(inputs, (canonical * expanded13).transpose(1, 2)).chunk(
        2, dim=-1
    )
    return torch.bmm(
        torch.nn.functional.silu(gate) * up,
        (down.float() * expanded2).transpose(1, 2),
    )


def _capture(model, inputs):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _reference_packed_consumer(model, inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _reference_packed_consumer(model, inputs)
    return graph, output


def _cache_pointers(model):
    return tuple(
        tensor.data_ptr()
        for weights in (model.mega_l1_weights, model.mega_l2_weights)
        for tensor in weights
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestMegaMoEReloadLimitation(CustomTestCase):
    def setUp(self):
        super().setUp()
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("This repro targets the SM90 postprocess branch")
        self.enterContext(
            get_flags().moe.override(
                a2a_backend=MoeA2ABackend.MEGAMOE,
                runner_backend=MoeRunnerBackend.AUTO,
            )
        )

    @torch.no_grad()
    def test_reproduces_wrong_packing_despite_stable_addresses(self):
        candidate = _SmallExperts()
        loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
        config = SimpleNamespace(dtype=torch.bfloat16)
        loader.prepare_model_for_capture(
            model=candidate,
            model_config=config,
            target_device=torch.device("cuda"),
        )
        self.assertTrue(candidate._mega_moe_sm90_fp8_weights)
        inputs = torch.zeros((2, 3, 128), device="cuda")
        graph, output = _capture(candidate, inputs)
        manifest = ModelStorageManifest.capture(candidate)
        cache_pointers = _cache_pointers(candidate)

        checkpoint = _checkpoint(candidate)
        serial = _SmallExperts()
        DefaultModelLoader.load_weights_and_postprocess(
            serial, iter(checkpoint), torch.device("cuda")
        )
        # Replace only file I/O; commit and postprocess are production code.
        with patch.object(
            loader, "_get_weights_iterator", return_value=iter(checkpoint)
        ):
            loader.commit_model_weights(
                model=candidate,
                model_config=config,
                resolved_sources=(SimpleNamespace(source=None),),
                target_device=torch.device("cuda"),
                startup_prefetch_active=False,
            )

        self.assertEqual(manifest.changed_names(candidate), ())
        self.assertEqual(_cache_pointers(candidate), cache_pointers)
        self.assertFalse(
            torch.equal(candidate.w13_weight.float(), serial.w13_weight.float())
        )
        for seed in (1, 2, 3):
            inputs.normal_(generator=torch.Generator(device="cuda").manual_seed(seed))
            expected = _reference_packed_consumer(serial, inputs)
            graph.replay()
            self.assertTrue(torch.isfinite(output).all().item())
            self.assertGreater((output - expected).abs().max().item(), 1e-3)
        print("LIMITATION REPRODUCED: stable pointers but wrong packed-weight replay")


if __name__ == "__main__":
    unittest.main()
