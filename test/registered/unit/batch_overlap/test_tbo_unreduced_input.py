"""The two-batch-overlap entry completes the sum its input still owes before it
splits the batch, so the split and the merge see the reduced tensor. CPU-only.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap import two_batch_overlap as tbo
from sglang.srt.layers.communicator import ScatterMode, UnreducedOutput
from sglang.srt.utils import empty_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTboEntryReducesItsInput(CustomTestCase):
    def test_split_and_merge_see_the_reduced_tensor(self):
        # Under attention DP the partial sum spans every DP rank's tokens; its
        # reduction also brings it back to this rank's three.
        reduced = torch.full((3, 4), 2.0)
        hidden_states = UnreducedOutput(
            torch.ones(6, 4), reduce_and_redistribute=lambda partial: reduced
        )
        seen = {}

        def split(**kwargs):
            seen["split"] = kwargs["hidden_states"]
            return [{}, {}]

        def merge(output_a, output_b, original_len):
            seen["original_len"] = original_len
            return None, None

        with (
            patch.object(tbo, "_model_forward_tbo_split_inputs", split),
            patch.object(
                tbo, "execute_overlapped_operations", lambda **kwargs: [{}, {}]
            ),
            patch.object(tbo, "_model_forward_tbo_merge_outputs", merge),
            patch.object(
                tbo.deep_gemm_wrapper,
                "configure_deep_gemm_num_sms",
                lambda num_sms: empty_context(),
            ),
        ):
            tbo._model_forward_tbo(
                inputs=dict(
                    hidden_states=hidden_states,
                    residual=torch.zeros(3, 4),
                    positions=None,
                    forward_batch=None,
                    zero_allocator=None,
                ),
                operations_strategy=SimpleNamespace(
                    deep_gemm_num_sms=None, operations=[], tbo_delta_stages=0
                ),
                input_data_scatter_mode=ScatterMode.TP_ATTN_FULL,
                layer_input_scatter_mode=ScatterMode.TP_ATTN_FULL,
            )

        self.assertIs(seen["split"], reduced)
        self.assertEqual(seen["original_len"], 3)


if __name__ == "__main__":
    unittest.main()
