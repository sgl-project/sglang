"""Unit tests for the Ascend sampling dispatch path."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAscendSamplerDispatch(unittest.TestCase):
    def test_top_k_dispatch_does_not_read_device_values_on_cpu(self):
        logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        top_ks = torch.tensor([2], dtype=torch.int32)
        top_ps = torch.ones(1)
        min_ps = torch.zeros(1)
        positions = torch.zeros(1, dtype=torch.int64)

        for eligible in (True, False):
            with self.subTest(eligible=eligible):
                npu_top_k_top_p = MagicMock(return_value=logits)
                with (
                    patch.object(
                        sampler_module,
                        "torch_npu",
                        SimpleNamespace(npu_top_k_top_p=npu_top_k_top_p),
                        create=True,
                    ),
                    patch.object(
                        sampler_module.torch,
                        "all",
                        side_effect=AssertionError("device predicate was inspected"),
                    ),
                ):
                    result = (
                        sampler_module.top_k_top_p_min_p_sampling_from_logits_ascend(
                            logits.clone(),
                            top_ks.clone(),
                            top_ps,
                            min_ps,
                            False,
                            None,
                            positions,
                            npu_top_k_top_p_eligible=eligible,
                        )
                    )

                self.assertEqual(tuple(result.shape), (1,))
                self.assertEqual(npu_top_k_top_p.called, eligible)


if __name__ == "__main__":
    unittest.main()
