import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.server_args import compute_world_size

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _shape(*, tp_size: int, pp_size: int, dp_size: int) -> dict:
    """The three `parallel` leaves the world size is computed from."""
    return {"tp_size": tp_size, "pp_size": pp_size, "dp_size": dp_size}


class TestComputeWorldSize(unittest.TestCase):
    def test_a_single_gpu_server_holds_one_gpu(self):
        """The default shape has to come out as one, or every consumer is off by a factor."""
        shape = _shape(tp_size=1, pp_size=1, dp_size=1)

        self.assertEqual(compute_world_size(**shape), 1)

    def test_tensor_and_pipeline_stages_multiply(self):
        """Each (pp_rank, tp_rank) pair is its own scheduler process on its own gpu."""
        shape = _shape(tp_size=2, pp_size=3, dp_size=1)

        self.assertEqual(compute_world_size(**shape), 6)

    def test_plain_data_parallel_replicas_each_hold_their_own_gpus(self):
        """Without dp attention every replica launches a full tensor-parallel group of its own."""
        shape = _shape(tp_size=2, pp_size=1, dp_size=2)

        self.assertEqual(compute_world_size(**shape), 4)

    def test_data_parallel_attention_shares_the_tensor_parallel_gpus(self):
        """With dp attention the dp ranks live inside the tensor-parallel world, not beside it."""
        with get_context().override_server_args(tp_size=4, attn_dp_size=2):
            parallel = get_parallel()
            shape = _shape(
                tp_size=parallel.tp_size,
                pp_size=parallel.pp_size,
                dp_size=parallel.dp_size,
            )

        self.assertEqual(compute_world_size(**shape), 4)


if __name__ == "__main__":
    unittest.main()
