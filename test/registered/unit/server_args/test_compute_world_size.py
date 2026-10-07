"""Unit tests for compute_world_size -- no server, no model loading."""

import unittest

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.server_args import compute_world_size
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestComputeWorldSize(unittest.TestCase):
    """The gpu count a fleet manager sizes against, from the launch widths."""

    def test_a_single_gpu_server_holds_one_gpu(self):
        self.assertEqual(compute_world_size(tp_size=1, pp_size=1, dp_size=1), 1)

    def test_tensor_and_pipeline_stages_multiply(self):
        """Each (pp_rank, tp_rank) pair is its own scheduler process on its own gpu."""
        self.assertEqual(compute_world_size(tp_size=2, pp_size=3, dp_size=1), 6)

    def test_plain_data_parallel_replicas_each_hold_their_own_gpus(self):
        """Without dp attention every replica launches a full tp group of its own."""
        self.assertEqual(compute_world_size(tp_size=2, pp_size=1, dp_size=2), 4)

    def test_data_parallel_attention_shares_the_tensor_parallel_gpus(self):
        """With dp attention the dp ranks live inside the tp world, not beside it,
        so the resolved leaves must not multiply out to 8."""
        with get_context().override_server_args(tp_size=4, attn_dp_size=2):
            parallel = get_parallel()
            world_size = compute_world_size(
                tp_size=parallel.tp_size,
                pp_size=parallel.pp_size,
                dp_size=parallel.dp_size,
            )

        self.assertEqual(world_size, 4)


if __name__ == "__main__":
    unittest.main()
