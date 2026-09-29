"""Dump deferred layer inputs without completing their pending reduction."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.debug_utils.dumper import _NonIntrusiveDumper
from sglang.srt.layers.layer_boundary.output import HandoffOutput, UnreducedOutput
from sglang.srt.layers.layer_boundary.residual.add_norm import ADD
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDumperLayerOutput(CustomTestCase):
    def test_pre_hook_preserves_partial_layer_inputs(self):
        def unexpected_reduce(_):
            self.fail("dumping must not complete a pending reduction")

        partial = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        unreduced = UnreducedOutput(partial, reduce_and_redistribute=unexpected_reduce)
        stream = ResidualStream(torch.zeros_like(partial))
        owed = stream.leave(unreduced, ADD)

        for value in (unreduced, owed):
            for keyword in (False, True):
                with self.subTest(wrapper=type(value).__name__, keyword=keyword):
                    captured = {}
                    hooks = _NonIntrusiveDumper.__new__(_NonIntrusiveDumper)
                    hooks._dumper = SimpleNamespace(
                        dump=lambda name, tensor: captured.update(
                            {name: tensor.clone()}
                        )
                    )
                    hooks._mode = "all"
                    hooks._core_fields = frozenset()
                    hook = hooks._make_forward_pre_hook(
                        module_name="model.layers.1", is_root=False
                    )
                    if keyword:
                        hook(None, (), {"hidden_states": value})
                        suffix = "hidden_states"
                    else:
                        hook(None, (None, value), {})
                        suffix = "1"
                    name = f"non_intrusive__model.layers.1.inputs.{suffix}"
                    self.assertEqual(set(captured), {name})
                    torch.testing.assert_close(
                        captured[name], torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                    )
        self.assertIs(stream.pending, owed.contribution)
        self.assertIs(owed.contribution.value, partial)
        self.assertIsNotNone(owed.contribution.owed)

    def test_opaque_and_consumed_outputs_have_no_tensor_to_dump(self):
        class OpaqueOutput(HandoffOutput):
            def complete(self):
                raise AssertionError("dumping must not finalize an opaque output")

        opaque = OpaqueOutput()
        stream = ResidualStream()
        owed = stream.leave(opaque, ADD)
        for value in (opaque, owed):
            self.assertEqual(_NonIntrusiveDumper._convert_value(value), {})

        stream = ResidualStream()
        consumed = stream.leave(UnreducedOutput(torch.ones(2, 2)), ADD)
        stream.write(torch.zeros(2, 2))
        self.assertEqual(_NonIntrusiveDumper._convert_value(consumed), {})


if __name__ == "__main__":
    unittest.main()
