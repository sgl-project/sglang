"""CPU coverage for GLM projection collection and its constructor boundary."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import gc
import types
import unittest
import weakref
from unittest.mock import patch

import torch

from sglang.srt.afd import projection_gc
from sglang.srt.afd.config import AFDExecutionMode
from sglang.srt.models import glm4_moe
from sglang.test.test_utils import CustomTestCase


class TestProjectionGarbageCollection(CustomTestCase):
    def test_native_no_parameters_or_gc(self):
        with patch.object(
            projection_gc.gc, "collect", side_effect=AssertionError("native GC")
        ):
            self.assertIsNone(projection_gc.collect_projection_garbage(object(), "off"))

    def test_cuda_order_and_receipt(self):
        calls = []
        cuda = types.SimpleNamespace(
            synchronize=lambda device: calls.append("sync"),
            memory_allocated=lambda device: 100,
            memory_reserved=lambda device: 200,
            mem_get_info=lambda device: (300, 600),
            device=lambda device: contextlib.nullcontext(),
            empty_cache=lambda: calls.append("empty"),
        )
        model = types.SimpleNamespace(
            parameters=lambda: iter(
                [types.SimpleNamespace(device=torch.device("cuda"))]
            )
        )
        with (
            patch.object(projection_gc.torch, "cuda", cuda),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_rank", return_value=3),
            patch.object(
                projection_gc.gc, "collect", side_effect=lambda: calls.append("gc") or 7
            ),
            patch.object(projection_gc.logger, "info") as log,
        ):
            result = projection_gc.collect_projection_garbage(model, "attention")
        self.assertEqual(calls, ["sync", "gc", "sync", "empty"])
        self.assertEqual(result["rank"], 3)
        self.assertEqual(result["gc_collected_objects"], 7)
        self.assertEqual(result["before"], result["after"])
        self.assertEqual(
            result["before"],
            dict(allocated_bytes=100, reserved_bytes=200, free_bytes=300),
        )
        self.assertEqual(log.call_args.args[0], "AFD_GLM_PROJECTION_GC %s")

    def test_unreachable_bound_loader_cycle_collected_live_kept(self):
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.empty(1, device="cpu"))
                self.param.weight_loader = self.load

            def load(self):
                return 42

        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            discarded = Layer()
            ref = weakref.ref(discarded)
            del discarded
            live = Layer()
            self.assertIsNotNone(ref())
            with patch.object(torch.distributed, "is_initialized", return_value=False):
                receipt = projection_gc.collect_projection_garbage(live, "ffn")
            self.assertIsNone(ref())
            self.assertEqual(live.param.weight_loader(), 42)
            self.assertIs(live.param.weight_loader.__self__, live)
            self.assertGreater(receipt["gc_collected_objects"], 0)
            self.assertIsNone(receipt["before"]["free_bytes"])
        finally:
            if was_enabled:
                gc.enable()

    def test_exact_constructor_native_and_role_boundary(self):
        events = []

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.start_layer = 0
                self.end_layer = 2
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Identity(), torch.nn.Identity()]
                )
                self.embed_tokens = torch.nn.Identity()

        class Projected(Model):
            pass

        def initialize_model(model, **kwargs):
            torch.nn.Module.__init__(model)
            events.append("base")
            model.model = Model()
            model.lm_head = torch.nn.Identity()

        for mode in AFDExecutionMode:
            with self.subTest(mode=mode):
                events.clear()
                with (
                    patch.object(
                        glm4_moe.DeepseekV2ForCausalLM, "__init__", initialize_model
                    ),
                    patch.object(glm4_moe, "afd_execution_mode", return_value=mode),
                    patch.object(glm4_moe, "DeepseekV2Model", Model),
                    patch.object(glm4_moe, "GlmMoeDsaAFDModel", Projected),
                    patch.object(
                        glm4_moe.GlmMoeDsaAFDDecoderLayer,
                        "install",
                        side_effect=lambda layer: events.append("install"),
                    ),
                    patch.object(
                        projection_gc,
                        "collect_projection_garbage",
                        side_effect=lambda *args: events.append("collect"),
                    ) as collect,
                ):
                    obj = glm4_moe.GlmMoeDsaForCausalLM(
                        types.SimpleNamespace(num_hidden_layers=2)
                    )
                self.assertEqual(
                    events,
                    (
                        ["base"]
                        if mode == AFDExecutionMode.OFF
                        else ["base", "install", "install", "collect"]
                    ),
                )
                if mode == AFDExecutionMode.OFF:
                    collect.assert_not_called()
                else:
                    collect.assert_called_once_with(obj, mode.value)
                if mode == AFDExecutionMode.FFN:
                    self.assertIsInstance(obj.lm_head, glm4_moe.AFDProxyMLP)
                    self.assertIsInstance(obj.model.embed_tokens, glm4_moe.AFDProxyMLP)


if __name__ == "__main__":
    unittest.main()
