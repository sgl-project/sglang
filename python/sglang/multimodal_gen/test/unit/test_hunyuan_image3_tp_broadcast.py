"""Unit tests for HunyuanImage-3 tensor-parallel input broadcasts."""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
    ar_stage,
)
from sglang.test.test_utils import CustomTestCase


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def contiguous(self):
        return self

    def reshape(self, *_shape):
        return self


class _FakeTPGroup:
    world_size = 2

    def __init__(self):
        self.broadcast_calls = []

    def broadcast(self, tensor, src):
        self.broadcast_calls.append((tensor, src))
        return tensor


class TestHunyuanImage3TPBroadcast(CustomTestCase):
    def test_static_broadcast_happens_outside_denoising_loop(self):
        source = textwrap.dedent(
            inspect.getsource(ar_stage.HunyuanImage3AR._forward_batched)
        )
        function = ast.parse(source).body[0]
        static_broadcast_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_broadcast_static_inputs"
        ]
        self.assertEqual(len(static_broadcast_calls), 1)

        denoising_loops = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.For)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "enumerate"
                for child in ast.walk(node.iter)
            )
        ]
        self.assertEqual(len(denoising_loops), 1)
        self.assertLess(
            static_broadcast_calls[0].lineno,
            denoising_loops[0].lineno,
            "request-static inputs must be broadcast before the denoising loop",
        )

    def test_static_inputs_are_broadcast_once(self):
        stage = ar_stage.HunyuanImage3AR.__new__(ar_stage.HunyuanImage3AR)
        tp_group = _FakeTPGroup()
        attention_mask = _FakeTensor((2, 1, 8, 8))
        cos = _FakeTensor((2, 8, 4))
        sin = _FakeTensor((2, 8, 4))

        with (
            patch.object(
                ar_stage, "model_parallel_is_initialized", return_value=True
            ),
            patch.object(ar_stage, "get_tp_group", return_value=tp_group),
        ):
            result = stage._broadcast_static_inputs(attention_mask, (cos, sin))

        self.assertEqual(result, (attention_mask, (cos, sin)))
        self.assertEqual(
            tp_group.broadcast_calls,
            [(attention_mask, 0), (cos, 0), (sin, 0)],
        )

    def test_backbone_forward_only_broadcasts_dynamic_hidden_states(self):
        stage = ar_stage.HunyuanImage3AR.__new__(ar_stage.HunyuanImage3AR)
        tp_group = _FakeTPGroup()
        hidden_states = _FakeTensor((2, 3, 4))
        attention_mask = _FakeTensor((2, 1, 3, 3))
        cos = _FakeTensor((2, 3, 2))
        sin = _FakeTensor((2, 3, 2))
        output = MagicMock()
        output.shape = (6, 4)
        output.view.return_value = "reshaped-output"
        stage._cache_dit_adapter = None
        stage.ar_model = SimpleNamespace(
            forward_block=MagicMock(return_value=output)
        )

        with (
            patch.object(
                ar_stage, "model_parallel_is_initialized", return_value=True
            ),
            patch.object(ar_stage, "get_tp_group", return_value=tp_group),
        ):
            result = stage._backbone_forward(
                num_image_tokens=2,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                custom_pos_emb=(cos, sin),
                first_step=True,
            )

        self.assertEqual(result, "reshaped-output")
        self.assertEqual(tp_group.broadcast_calls, [(hidden_states, 0)])


if __name__ == "__main__":
    unittest.main()
