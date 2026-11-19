"""Unit tests for ShardedStateLoader state-dict export guards."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod  # noqa: E402
from sglang.srt.model_loader import loader as loader_module  # noqa: E402
from sglang.srt.model_loader.loader import (  # noqa: E402
    PreshardedModelLoader,
    ShardedStateLoader,
)


class _MoeWithMxfp4(torch.nn.Module):
    """A FusedMoE after triton_kernels post-processing: only the bias is a Parameter."""

    def __init__(self, use_triton_kernels: bool, use_mega_moe: bool):
        super().__init__()
        self.w13_weight_bias = torch.nn.Parameter(torch.zeros(4))
        quant_method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
        quant_method.use_triton_kernels = use_triton_kernels
        quant_method.use_mega_moe = use_mega_moe
        self.quant_method = quant_method


def _model(use_triton_kernels: bool, use_mega_moe: bool = False) -> torch.nn.Module:
    model = torch.nn.Module()
    model.experts = _MoeWithMxfp4(use_triton_kernels, use_mega_moe)
    return model


class TestShardedStateMxfp4Guard(CustomTestCase):
    def test_save_rejects_mxfp4_triton_kernels(self):
        """A sharded_state export must fail, not silently omit the expert weights."""
        with tempfile.TemporaryDirectory() as path, get_parallel().override(tp_rank=0):
            with self.assertRaisesRegex(NotImplementedError, "experts"):
                ShardedStateLoader.save_model(_model(use_triton_kernels=True), path)
            self.assertEqual(os.listdir(path), [])

    def test_save_allows_mxfp4_that_keeps_weights_as_parameters(self):
        """Backends that keep the expert weights as Parameters must still export."""
        for kwargs in (
            dict(use_triton_kernels=False),
            dict(use_triton_kernels=True, use_mega_moe=True),
        ):
            with self.subTest(**kwargs), tempfile.TemporaryDirectory() as path:
                with get_parallel().override(tp_rank=0):
                    ShardedStateLoader.save_model(_model(**kwargs), path)
                self.assertEqual(os.listdir(path), ["model-rank-0-part-0.safetensors"])


class TestPreshardedMxfp4Guard(CustomTestCase):
    def test_dump_and_reload_reject_mxfp4_triton_kernels(self):
        """The presharded dump never saves quant-method tensors either; dump and
        reload must both fail before any weight is read."""
        presharded = PreshardedModelLoader.__new__(PreshardedModelLoader)
        presharded.load_config = SimpleNamespace()
        config = SimpleNamespace(dtype=torch.float32)
        device = SimpleNamespace(device="cpu")
        with (
            mock.patch.object(
                loader_module, "_get_quantization_config", return_value=None
            ),
            mock.patch.object(
                loader_module, "_initialize_model", return_value=_model(True)
            ),
            mock.patch.object(PreshardedModelLoader, "_ensure_presharded_dir_writable"),
            mock.patch.object(
                PreshardedModelLoader, "load_weights_and_postprocess"
            ) as load_weights,
        ):
            with self.assertRaisesRegex(NotImplementedError, "experts"):
                presharded._first_time_load_and_dump(
                    config, device, "/nonexistent", shard_config={}
                )
            load_weights.assert_not_called()
            with self.assertRaisesRegex(NotImplementedError, "experts"):
                presharded._load_from_presharded(config, device, "/nonexistent")


if __name__ == "__main__":
    unittest.main()
