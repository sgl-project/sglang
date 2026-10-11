# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 MLP graph replay must use live inputs and live weight dispatch."""

import types
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.multimodal_gen.runtime.breakable_cuda_graph.cosmos3_mlp import (
    Cosmos3MLPCudaGraphs,
)
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    Cosmos3DenseMLP,
    Cosmos3GatedMLP,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DenoisingStage,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class _ScaledProjection(nn.Module):
    """An adapter-like projection with request-dependent Python dispatch."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.scale = 2

    def forward(self, x):
        y, bias = self.base(x)
        return y * self.scale, bias


class _StepQuantMethod(UnquantizedLinearMethod):
    scale = 2

    def apply(self, layer, x, bias=None):
        return super().apply(layer, x, bias) * self.scale


class TestCosmos3MlpCudaGraphs(CustomTestCase):
    def test_stage_only_enables_supported_residency_and_execution_mode(self):
        cases = (
            (True, False, 1, "resident", True),
            (False, False, 1, "resident", False),
            (True, True, 1, "resident", False),
            (True, False, 2, "resident", False),
            (True, False, 1, "component-offload", False),
            (True, False, 1, "layerwise-offload", False),
        )
        for enabled, compile_, gpus, residency, expected in cases:
            with self.subTest(case=(enabled, compile_, gpus, residency)):
                stage = Cosmos3DenoisingStage.__new__(Cosmos3DenoisingStage)
                stage._mlp_cuda_graphs = None
                args = types.SimpleNamespace(
                    enable_breakable_cuda_graph=enabled,
                    enable_torch_compile=compile_,
                    num_gpus=gpus,
                    canonical_residency_mode=lambda _: residency,
                )
                with patch(
                    "sglang.multimodal_gen.runtime.breakable_cuda_graph."
                    "cosmos3_mlp.Cosmos3MLPCudaGraphs"
                ) as constructor:
                    stage._maybe_enable_mlp_cuda_graphs(object(), args)
                    self.assertEqual(constructor.called, expected)

    def test_rollout_does_not_activate_graphs(self):
        stage = Cosmos3DenoisingStage.__new__(Cosmos3DenoisingStage)
        stage._mlp_cuda_graphs = Mock()
        stage._mlp_cuda_graphs.activate.side_effect = AssertionError
        stage.transformer = object()
        stage.use_declared_component = Mock(return_value=nullcontext())
        batch = types.SimpleNamespace(rollout=True, extra={})
        stage._denoise_once = Mock(return_value=batch)
        self.assertIs(stage.forward(batch, None), batch)
        stage._mlp_cuda_graphs.activate.assert_not_called()

    def make_graphs(self, dtype=torch.bfloat16):
        # No collectives are used by TP=1; keep this a kernel-sized test.
        group = types.SimpleNamespace(world_size=1, rank_in_group=0)
        with patch(
            "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
            return_value=group,
        ):
            mlps = [Cosmos3GatedMLP(128, 256), Cosmos3DenseMLP(128, 256)]
        for mlp in mlps:
            mlp.to(device="cuda", dtype=dtype)
            for weight in mlp.parameters():
                weight.normal_(std=0.03)
        transformer = types.SimpleNamespace(
            gen_layers=[types.SimpleNamespace(mlp=mlp) for mlp in mlps]
        )
        eager = [mlp.forward for mlp in mlps]
        graphs = Cosmos3MLPCudaGraphs(transformer, torch.device("cuda"))
        self.addCleanup(lambda: [runner.reset() for runner in graphs.runners])
        return mlps, eager, graphs

    def capture(self, mlps, graphs, x):
        with graphs.activate(is_warmup=True):
            for mlp in mlps:
                mlp(x)
        self.assertTrue(all(len(r.entries) == 1 for r in graphs.runners))

    @torch.inference_mode()
    def test_live_inputs_and_retained_outputs(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                mlps, eager, graphs = self.make_graphs(dtype)
                names = [tuple(mlp.state_dict()) for mlp in mlps]
                x = torch.randn(1, 64, 128, device="cuda", dtype=dtype)
                self.capture(mlps, graphs, x)
                retained = []
                for _ in range(3):
                    x.normal_()
                    expected = [fn(x) for fn in eager]
                    with graphs.activate(is_warmup=False):
                        outputs = [mlp(x) for mlp in mlps]
                    for actual, reference in zip(outputs, expected):
                        self.assertTrue(torch.equal(actual, reference))
                        retained.append((actual, reference))
                for actual, reference in retained:
                    self.assertTrue(torch.equal(actual, reference))
                self.assertEqual(names, [tuple(m.state_dict()) for m in mlps])

    @torch.inference_mode()
    def test_serving_miss_and_exception_do_not_capture(self):
        mlps, eager, graphs = self.make_graphs()
        x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
        self.capture(mlps, graphs, x)
        x = torch.randn(2, 33, 128, device="cuda", dtype=torch.bfloat16)
        with patch.object(graphs.runners[0], "capture", side_effect=AssertionError):
            with graphs.activate(is_warmup=False):
                self.assertTrue(torch.equal(mlps[0](x), eager[0](x)))
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                with graphs.activate(is_warmup=True):
                    raise RuntimeError("request failed")
            # An exception must not leave capture enabled outside the request.
            self.assertTrue(torch.equal(mlps[0](x), eager[0](x)))
        self.assertTrue(all(len(r.entries) == 1 for r in graphs.runners))

    @torch.inference_mode()
    def test_replaced_weight_storage_invalidates_graph(self):
        mlps, eager, graphs = self.make_graphs()
        x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
        self.capture(mlps, graphs, x)
        mlps[0].down_proj.weight = nn.Parameter(
            mlps[0].down_proj.weight.clone() * 2, requires_grad=False
        )
        with graphs.activate(is_warmup=False):
            self.assertTrue(torch.equal(mlps[0](x), eager[0](x)))
        self.assertTrue(all(not r.entries for r in graphs.runners))

    @torch.inference_mode()
    def test_inplace_weight_update_is_visible(self):
        mlps, eager, graphs = self.make_graphs()
        x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
        self.capture(mlps, graphs, x)
        mlps[0].down_proj.weight.mul_(2)
        with graphs.activate(is_warmup=False):
            self.assertTrue(torch.equal(mlps[0](x), eager[0](x)))
        self.assertEqual(len(graphs.runners[0].entries), 1)

    @torch.inference_mode()
    def test_adapter_and_step_dispatch_fall_back(self):
        for replace in ("adapter", "quant_method"):
            with self.subTest(replace=replace):
                mlps, eager, graphs = self.make_graphs()
                x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
                self.capture(mlps, graphs, x)
                if replace == "adapter":
                    dispatch = _ScaledProjection(mlps[0].down_proj)
                    mlps[0].down_proj = dispatch
                else:
                    dispatch = _StepQuantMethod()
                    mlps[0].down_proj.quant_method = dispatch
                for scale in (2, 3):
                    dispatch.scale = scale
                    with graphs.activate(is_warmup=False):
                        self.assertTrue(torch.equal(mlps[0](x), eager[0](x)))
                self.assertTrue(all(not r.entries for r in graphs.runners))


if __name__ == "__main__":
    unittest.main()
