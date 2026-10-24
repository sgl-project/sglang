"""CPU regression for the FlashKDA extend return contract (issue #39925)."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kernels import kda_flashkda
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFlashKDAReturnContract(CustomTestCase):
    def test_extend_return_contract(self):
        # Mock only the compute kernels; exercise the real routing and return
        # handling without CUDA or the optional flash_kda package.
        output = torch.zeros(1, 128, 1, 128)
        states = torch.ones(1, 1, 128, 128)
        inputs = [torch.empty(0)] * 5
        kwargs = dict(
            ssm_states=states,
            cache_indices=torch.tensor([0]),
            query_start_loc=torch.tensor([0, 128]),
            lower_bound=-5.0,
            extend_seq_lens_cpu=[128],
        )
        cases = [
            ("fused", {}, False, False),
            ("unbounded", {"lower_bound": None}, True, False),
            ("speculative", {"is_spec_decode": True}, True, False),
            ("short", {"extend_seq_lens_cpu": [32]}, True, False),
            ("long", {"extend_seq_lens_cpu": [4096]}, True, False),
            ("tracked", {"return_intermediate_states": True}, True, True),
        ]
        for name, overrides, fallback, tracked in cases:
            with self.subTest(name=name):
                expected = (output, states) if tracked else output
                with (
                    patch.object(
                        kda_flashkda.FlashKDAKernel,
                        "_flashkda_extend",
                        return_value=output,
                    ) as fused,
                    patch.object(
                        kda_flashkda, "_triton_fallback", return_value=expected
                    ) as triton,
                ):
                    result = kda_flashkda.FlashKDAKernel().extend(
                        *inputs, **(kwargs | overrides)
                    )
                self.assertIs(result, expected)
                if tracked:
                    out, h = result
                    self.assertIs(h, states)
                else:
                    self.assertIsInstance(result, torch.Tensor)
                    out = result
                self.assertEqual(out.shape, output.shape)
                if fallback:
                    fused.assert_not_called()
                    triton.assert_called_once()
                    self.assertEqual(
                        triton.call_args.kwargs["return_intermediate_states"], tracked
                    )
                else:
                    fused.assert_called_once()
                    triton.assert_not_called()


if __name__ == "__main__":
    unittest.main()
