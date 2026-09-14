"""MMEncoder must forward attn_cp_size to initialize_model_parallel, not just
tp_size.

Real 2-GPU hardware confirmed a live mismatch this test guards against
statically: with `tp_size=2, attn_cp_size=2` published, calling
`initialize_model_parallel(tensor_model_parallel_size=2)` alone builds the
live attention-TP group at width 2, while `get_parallel().attn_tp_size`
(derived from the published config) answers 1 -- `VisionAttention`
(`layers/attention/vision.py`) reads that derived value as its own
weight-sharding width, so the mismatch is a real, silent wrong-sharding bug,
not just a reporting discrepancy. Forwarding
`attention_context_model_parallel_size=get_parallel().attn_cp_size` too
makes the two agree (confirmed on the same hardware).

A full `MMEncoder` instantiation needs real weights and a live process
group, so this checks the one line that matters statically: the call passes
`attention_context_model_parallel_size` as well as
`tensor_model_parallel_size`. Not a substitute for testing `MMEncoder`
end-to-end under `--attn-cp-size > 1` on real hardware, but cheap enough to
run everywhere and catches the specific regression class (a future edit
that reverts to the tp-only call).
"""

import ast
import os

import sglang.srt.disaggregation.encoder.server as encoder_server_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMMEncoderForwardsAttnCpSize(CustomTestCase):
    def test_initialize_model_parallel_call_forwards_attn_cp_size(self):
        path = encoder_server_module.__file__
        tree = ast.parse(open(path).read())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "initialize_model_parallel"
        ]
        self.assertEqual(
            len(calls),
            1,
            f"expected exactly one initialize_model_parallel(...) call in "
            f"{os.path.basename(path)}, found {len(calls)} -- update this "
            "test if that's now intentional",
        )
        kwarg_names = {kw.arg for kw in calls[0].keywords}
        self.assertIn(
            "attention_context_model_parallel_size",
            kwarg_names,
            "MMEncoder's initialize_model_parallel(...) call must forward "
            "attention_context_model_parallel_size (not just "
            "tensor_model_parallel_size), or get_parallel().attn_tp_size "
            "silently disagrees with the group actually built whenever "
            "--attn-cp-size > 1 -- confirmed on real 2-GPU hardware",
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
