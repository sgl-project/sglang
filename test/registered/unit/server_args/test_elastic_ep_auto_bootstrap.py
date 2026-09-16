from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.arg_groups.parallel_hook import handle_elastic_ep
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _args(replica_index: int) -> ServerArgs:
    return ServerArgs(
        model_path="dummy",
        elastic_ep_backend="mooncake",
        ep_join_mode="auto",
        elastic_ep_replica_index=replica_index,
        tp_size=4,
        dp_size=4,
        ep_size=4,
        max_ep_size=16,
        elastic_ep_initial_size=4,
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        moe_a2a_backend="nixl",
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.DISABLED),
            prefill=PhaseConfig(backend=Backend.DISABLED),
        ),
        load_balance_method="round_robin",
    )


class TestElasticEPAutoBootstrap(CustomTestCase):
    def test_replica_zero_resolves_to_primary(self):
        server_args = _args(0)

        handle_elastic_ep(server_args)

        cfg = resolved_view(server_args)
        self.assertIsNone(cfg.ep_join_mode)
        self.assertEqual(cfg.ep_join_rank_offset, 0)
        self.assertEqual(cfg.node_rank, 0)
        self.assertEqual(cfg.nnodes, 1)

    def test_later_replica_resolves_to_scale_joiner(self):
        server_args = _args(2)

        handle_elastic_ep(server_args)

        cfg = resolved_view(server_args)
        self.assertEqual(cfg.ep_join_mode, "scale")
        self.assertEqual(cfg.ep_join_rank_offset, 8)
        self.assertEqual(cfg.node_rank, 1)
        self.assertEqual(cfg.nnodes, 2)


if __name__ == "__main__":
    import unittest

    unittest.main()
