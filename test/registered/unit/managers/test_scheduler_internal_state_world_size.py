import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GetInternalStateReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_server_args(
    *, tp_size: int, pp_size: int, dp_size: int, enable_dp_attention: bool
) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.tp_size = tp_size
    server_args.pp_size = pp_size
    server_args.dp_size = dp_size
    server_args.enable_dp_attention = enable_dp_attention
    return server_args


class TestServerArgsWorldSize(unittest.TestCase):
    def test_a_single_gpu_server_holds_one_gpu(self):
        """The default shape has to come out as one, or every consumer is off by a factor."""
        server_args = _make_server_args(
            tp_size=1, pp_size=1, dp_size=1, enable_dp_attention=False
        )

        self.assertEqual(server_args.world_size, 1)

    def test_tensor_and_pipeline_stages_multiply(self):
        """Each (pp_rank, tp_rank) pair is its own scheduler process on its own gpu."""
        server_args = _make_server_args(
            tp_size=2, pp_size=3, dp_size=1, enable_dp_attention=False
        )

        self.assertEqual(server_args.world_size, 6)

    def test_plain_data_parallel_replicas_each_hold_their_own_gpus(self):
        """Without dp attention every replica launches a full tensor-parallel group of its own."""
        server_args = _make_server_args(
            tp_size=2, pp_size=1, dp_size=2, enable_dp_attention=False
        )

        self.assertEqual(server_args.world_size, 4)

    def test_data_parallel_attention_shares_the_tensor_parallel_gpus(self):
        """With dp attention the dp ranks live inside the tensor-parallel world, so multiplying
        would double count."""
        server_args = _make_server_args(
            tp_size=4, pp_size=1, dp_size=2, enable_dp_attention=True
        )

        self.assertEqual(server_args.world_size, 4)


class TestSchedulerInternalStateWorldSize(unittest.TestCase):
    def _get_internal_state(self, server_args: ServerArgs) -> dict:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = server_args
        scheduler.metrics_reporter = SimpleNamespace(
            last_gen_throughput=1.0,
            spec_total_num_forward_ct=0,
            spec_total_num_accept_tokens=0,
            step_time_dict={},
        )
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                weight_load_mem_usage=1.0,
                graph_mem_usage=2.0,
            )
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(mem_usage=3.0)
        )
        scheduler.max_total_num_tokens = 100
        scheduler.max_running_requests = 8
        scheduler.spec_algorithm = SimpleNamespace(
            is_none=lambda: True,
            is_dspark=lambda: False,
        )
        scheduler.draft_worker = None

        with patch(
            "sglang.srt.managers.scheduler.get_server_args",
            return_value=server_args,
        ):
            output = scheduler.get_internal_state(recv_req=GetInternalStateReq())

        return output.internal_state

    def test_the_internal_state_reports_the_whole_server(self):
        """A consumer sizing an external fleet has to read the gpus the server occupies, which the
        declared parallel sizes alone do not give it without repeating the formula."""
        server_args = _make_server_args(
            tp_size=2, pp_size=1, dp_size=2, enable_dp_attention=False
        )

        internal_state = self._get_internal_state(server_args)

        self.assertEqual(internal_state["world_size"], 4)

    def test_the_reported_size_is_not_one_replica_of_a_data_parallel_server(self):
        """Each plain dp replica has a torch process group of its own, so no scheduler can observe
        the whole server and report it from its own group."""
        server_args = _make_server_args(
            tp_size=2, pp_size=1, dp_size=2, enable_dp_attention=False
        )

        internal_state = self._get_internal_state(server_args)

        self.assertNotEqual(
            internal_state["world_size"], server_args.tp_size * server_args.pp_size
        )


if __name__ == "__main__":
    unittest.main()
