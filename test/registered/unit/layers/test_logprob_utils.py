from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest

import torch

from sglang.srt.layers.utils.logprob import (
    LogprobStage,
    get_token_ids_logprobs_raw,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class LogprobUtilsTestCase(unittest.TestCase):
    def test_prefill_token_ids_logprobs_none_advances_position(self):
        logprobs = torch.tensor(
            [
                [0.0, 0.1, 0.2, 0.3],
                [1.0, 1.1, 1.2, 1.3],
                [2.0, 2.1, 2.2, 2.3],
                [3.0, 3.1, 3.2, 3.3],
            ]
        )

        vals, idxs = get_token_ids_logprobs_raw(
            logprobs,
            [[0, 2], None, [1]],
            stage=LogprobStage.PREFILL,
            extend_logprob_pruned_lens_cpu=[1, 2, 1],
        )

        self.assertEqual(len(vals[0]), 1)
        self.assertAlmostEqual(vals[0][0][0], 0.0)
        self.assertAlmostEqual(vals[0][0][1], 0.2, places=6)
        self.assertEqual(idxs[0], [[0, 2]])
        self.assertEqual(vals[1], [])
        self.assertEqual(idxs[1], [])
        self.assertEqual(len(vals[2]), 1)
        self.assertAlmostEqual(vals[2][0][0], 3.1, places=6)
        self.assertEqual(idxs[2], [[1]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
