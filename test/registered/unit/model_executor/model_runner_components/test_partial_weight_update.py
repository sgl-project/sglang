"""Unit tests for srt/model_executor/model_runner_components/partial_weight_update.

Critical-path bookkeeping of partial weight updates: which modules a load
touched (and therefore which get re-post-processed), whether wrapped weight
loaders are restored exactly, and which checkpoint tensors a prefix filter
admits.
"""

import unittest
from unittest.mock import MagicMock

import torch
from torch import nn

from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.model_executor.model_runner_components.partial_weight_update import (
    ModuleTouchRecorder,
    filter_weights_by_names,
    filter_weights_by_prefix,
    postprocess_touched_modules,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _TwoLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.second = nn.Linear(2, 2, bias=False)


def _load(param, weight):
    getattr(param, "weight_loader", default_weight_loader)(param, weight)


class TestModuleTouchRecorder(CustomTestCase):
    def test_load_marks_only_touched_module(self):
        model = _TwoLinearModel()
        weight = torch.full((2, 2), 7.0)

        with ModuleTouchRecorder(model) as recorder:
            _load(model.first.weight, weight)

        touched = recorder.touched_modules()
        self.assertEqual([name for name, _ in touched], ["first"])
        self.assertIs(touched[0][1], model.first)
        self.assertTrue(torch.equal(model.first.weight, weight))

    def test_loaders_are_restored_after_context(self):
        model = _TwoLinearModel()
        sentinel = object()
        model.second.weight.weight_loader = sentinel

        with ModuleTouchRecorder(model):
            self.assertNotIn(model.first.weight.weight_loader, (None, sentinel))

        self.assertNotIn("weight_loader", model.first.weight.__dict__)
        self.assertIs(model.second.weight.weight_loader, sentinel)

    def test_property_backed_parameter_is_wrapped_and_restored(self):
        loads = []

        def custom_loader(param, weight):
            loads.append(weight)
            param.data.copy_(weight)

        param = ModelWeightParameter(
            data=torch.zeros(2, 2),
            input_dim=1,
            output_dim=0,
            weight_loader=custom_loader,
        )
        model = nn.Module()
        model.proj = nn.Module()
        model.proj.register_parameter("weight", param)
        weight = torch.ones(2, 2)

        with ModuleTouchRecorder(model) as recorder:
            param.weight_loader(param, weight)

        self.assertEqual([name for name, _ in recorder.touched_modules()], ["proj"])
        self.assertEqual(len(loads), 1)
        self.assertIs(param.weight_loader, custom_loader)

    def test_touched_survives_context_for_rollback(self):
        model = _TwoLinearModel()
        recorder = ModuleTouchRecorder(model)

        with recorder:
            _load(model.first.weight, torch.ones(2, 2))
        with recorder:
            _load(model.second.weight, torch.ones(2, 2))

        self.assertEqual(
            sorted(name for name, _ in recorder.touched_modules()),
            ["first", "second"],
        )

    def test_untouched_model_reports_nothing(self):
        model = _TwoLinearModel()
        with ModuleTouchRecorder(model) as recorder:
            pass
        self.assertEqual(recorder.touched_modules(), [])


class TestWeightFilters(CustomTestCase):
    def _weights(self):
        return [
            ("model.layers.0.mlp.up_proj.weight", torch.zeros(1)),
            ("model.layers.1.mlp.up_proj.weight", torch.zeros(1)),
            ("lm_head.weight", torch.zeros(1)),
        ]

    def test_prefix_filter_records_matching_names(self):
        seen = []
        out = list(
            filter_weights_by_prefix(
                self._weights(), ["model.layers.0.", "lm_head."], seen
            )
        )
        self.assertEqual(seen, ["model.layers.0.mlp.up_proj.weight", "lm_head.weight"])
        self.assertEqual([name for name, _ in out], seen)

    def test_empty_prefix_selects_everything(self):
        seen = []
        out = list(filter_weights_by_prefix(self._weights(), [""], seen))
        self.assertEqual(len(out), 3)
        self.assertEqual(len(seen), 3)

    def test_name_filter_selects_exact_names(self):
        out = list(filter_weights_by_names(self._weights(), {"lm_head.weight"}))
        self.assertEqual([name for name, _ in out], ["lm_head.weight"])


class TestPostprocessTouchedModules(CustomTestCase):
    def test_postprocesses_only_touched_quant_modules(self):
        model = _TwoLinearModel()
        model.first.quant_method = MagicMock()
        model.second.quant_method = MagicMock()

        count = postprocess_touched_modules(
            [("first", model.first)], torch.device("cpu")
        )

        self.assertEqual(count, 1)
        model.first.quant_method.process_weights_after_loading.assert_called_once_with(
            model.first
        )
        model.second.quant_method.process_weights_after_loading.assert_not_called()

    def test_modules_without_quant_method_are_skipped(self):
        model = _TwoLinearModel()
        count = postprocess_touched_modules(
            [("first", model.first), ("second", model.second)],
            torch.device("cpu"),
        )
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
