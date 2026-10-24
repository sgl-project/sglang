"""Humming MoE weight preparation must free the tensors each sublayer transform replaces."""

from __future__ import annotations

import contextlib
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.quantization import humming as humming_mod  # noqa: E402
from sglang.srt.layers.quantization import humming_utils  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_EXPERTS, HIDDEN, INTERMEDIATE = 4, 64, 96


class _ConvertedSchema:
    """Stands in for the Humming-format weight schema returned by the conversion."""


class _TransformRecorder:
    """Fakes the Humming conversion and transform and records what is alive when
    the CUDA cache is emptied."""

    def __init__(self):
        self.events: list[str] = []
        # Converted tensors and the parameters wrapping them; all of these are
        # replaced by the transform and must be unreferenced afterwards.
        self.replaced: list[weakref.ref] = []

    def convert_humming(self, tensors, **kwargs):
        out = {name: tensor.clone() for name, tensor in tensors.items()}
        self.replaced += [weakref.ref(tensor) for tensor in out.values()]
        return _ConvertedSchema(), out

    def convert_input(self, tensors, **kwargs):
        return SimpleNamespace(), None

    def transform_humming_layer(self, layer, sublayer_name=None):
        # Like the real transform: every sublayer tensor is swapped for a padded
        # copy of a different size.
        for name, param in list(layer.named_parameters()):
            if name.startswith(f"{sublayer_name}_"):
                self.replaced.append(weakref.ref(param))
                padded = torch.zeros(
                    param.shape[0], param.shape[1] + 32, *param.shape[2:]
                )
                delattr(layer, name)
                setattr(layer, name, torch.nn.Parameter(padded, requires_grad=False))
        del param
        self.events.append(f"transform:{sublayer_name}")

    def empty_cache(self):
        alive = sum(ref() is not None for ref in self.replaced)
        self.events.append(f"empty_cache(alive={alive})")

    def patches(self):
        not_deepep = SimpleNamespace(is_deepep=lambda: False)
        humming_method = SimpleNamespace(
            prepare_layer_meta=lambda **kwargs: None,
            transform_humming_layer=self.transform_humming_layer,
        )
        return [
            patch.object(humming_utils, "get_moe_a2a_backend", return_value=not_deepep),
            patch.object(humming_utils, "HummingMethod", humming_method),
            # humming.py imports Humming lazily; HummingMethod is None until then.
            patch.object(humming_mod, "HummingMethod", humming_method),
            patch.object(torch.cuda, "empty_cache", side_effect=self.empty_cache),
        ]


def _make_moe_layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    shapes = {
        "w13_weight": (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
        "w13_weight_scale": (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 32),
        "w2_weight": (NUM_EXPERTS, HIDDEN, INTERMEDIATE // 2),
        "w2_weight_scale": (NUM_EXPERTS, HIDDEN, INTERMEDIATE // 32),
    }
    for name, shape in shapes.items():
        param = torch.nn.Parameter(torch.ones(shape), requires_grad=False)
        layer.register_parameter(name, param)
    return layer


EXPECTED_EVENTS = [
    "transform:w13",
    "empty_cache(alive=0)",
    "transform:w2",
    "empty_cache(alive=0)",
]


class TestHummingMoeReleaseReplaced(CustomTestCase):
    def _run(self, recorder: _TransformRecorder, fn):
        with contextlib.ExitStack() as stack:
            for p in recorder.patches():
                stack.enter_context(p)
            fn()

    def test_prepare_humming_moe_layer(self):
        recorder = _TransformRecorder()
        layer = _make_moe_layer()
        layer.num_local_experts = NUM_EXPERTS
        layer.hidden_size = HIDDEN
        layer.intermediate_size_per_partition = INTERMEDIATE
        layer.params_dtype = torch.bfloat16
        layer.with_bias = False
        layer.layer_name = "model.layers.0.mlp.experts"
        schema = SimpleNamespace(convert_humming=recorder.convert_humming)

        with patch.object(
            humming_utils.BaseWeightSchema, "from_config", return_value=schema
        ):
            self._run(
                recorder,
                lambda: humming_utils.prepare_humming_moe_layer(
                    layer, {"quant_method": "mxfp4"}
                ),
            )

        self.assertEqual(recorder.events, EXPECTED_EVENTS)

    def test_humming_moe_method(self):
        recorder = _TransformRecorder()
        layer = _make_moe_layer()
        layer.sublayer_configs = {
            "w13": {"shape_n": 2 * INTERMEDIATE, "shape_k": HIDDEN},
            "w2": {"shape_n": HIDDEN, "shape_k": INTERMEDIATE},
        }
        layer.param_dtype = torch.bfloat16
        layer.num_experts = NUM_EXPERTS
        layer.with_bias = False

        method = humming_mod.HummingMoEMethod.__new__(humming_mod.HummingMoEMethod)
        method.weight_schema = SimpleNamespace(convert_humming=recorder.convert_humming)
        method.input_schema = SimpleNamespace(convert_humming=recorder.convert_input)
        method.force_weight_schema = None

        with patch.object(humming_mod, "HummingWeightSchema", _ConvertedSchema):
            self._run(recorder, lambda: method.process_weights_after_loading(layer))

        self.assertEqual(recorder.events, EXPECTED_EVENTS)


if __name__ == "__main__":
    unittest.main()
