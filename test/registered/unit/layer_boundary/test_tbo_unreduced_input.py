"""The two-batch-overlap entry completes the sum its input still owes before it
splits the batch, so the split and the merge see the reduced tensor. CPU-only.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap import two_batch_overlap as tbo
from sglang.srt.layers.layer_boundary import PLAIN_ADD, Layout, UnreducedOutput
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.utils import empty_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTboEntryReducesItsInput(CustomTestCase):
    def test_each_microbatch_owns_its_contribution_and_merge_keeps_the_update(self):
        group = SimpleNamespace(all_reduce=lambda x: x * 2)
        residual = torch.ones(4, 3)
        stream = ResidualStream(residual)
        hidden = stream.record(
            UnreducedOutput(torch.ones(4, 3), group=group), PLAIN_ADD
        )
        batch = SimpleNamespace(residual_stream=stream, global_forward_mode=None)
        parts_seen = []

        def split(hidden_states, residual, **kwargs):
            return [
                dict(
                    hidden_states=hidden_states[a:b],
                    residual=residual[a:b],
                    forward_batch=SimpleNamespace(tbo_parent_token_range=(a, b)),
                )
                for a, b in ((0, 2), (2, 4))
            ]

        def execute(inputs_arr, **kwargs):
            parts_seen.extend(inputs_arr)
            self.assertIsNot(
                inputs_arr[0]["forward_batch"].residual_stream,
                inputs_arr[1]["forward_batch"].residual_stream,
            )
            for part in inputs_arr:
                self.assertNotIn("residual", part)
                state = part["forward_batch"].residual_stream
                value = part["hidden_states"]
                self.assertIsNot(state.pending, stream.pending)
                self.assertIs(state.pending.update, PLAIN_ADD)
                with self.assertRaises(RuntimeError):
                    state.input(
                        inputs_arr[1 if part is inputs_arr[0] else 0]["hidden_states"]
                    )
                value, old_residual = state.input(value)
                torch.testing.assert_close(value, torch.full((2, 3), 2.0))
                state.write(value + old_residual)
                part["hidden_states"] = state.record(value * 5, PLAIN_ADD)
            return inputs_arr

        with (
            patch.object(
                tbo.OperationsStrategy,
                "init_new_tbo",
                return_value=SimpleNamespace(
                    deep_gemm_num_sms=None, operations=[], tbo_delta_stages=0
                ),
            ),
            patch.object(tbo, "_model_forward_tbo_split_inputs", split),
            patch.object(tbo, "execute_overlapped_operations", execute),
            patch.object(
                tbo.deep_gemm_wrapper,
                "configure_deep_gemm_num_sms",
                lambda _: empty_context(),
            ),
        ):
            merged = tbo.model_forward_stages(
                layers=[
                    SimpleNamespace(
                        attn_boundary=SimpleNamespace(
                            incoming_residual_rows=Layout(frozenset())
                        )
                    )
                ],
                enable_tbo=True,
                positions=None,
                hidden_states=hidden,
                forward_batch=batch,
            )
        output = batch.residual_stream
        self.assertEqual(len(parts_seen), 2)
        self.assertTrue(
            all(part["forward_batch"].residual_stream is None for part in parts_seen)
        )
        self.assertIs(output.pending.update, PLAIN_ADD)
        self.assertIs(output.pending.value, merged)
        torch.testing.assert_close(merged, torch.full((4, 3), 10.0))
        torch.testing.assert_close(output.residual, torch.full((4, 3), 3.0))


if __name__ == "__main__":
    unittest.main()
