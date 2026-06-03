"""Unit tests for MultiDetokenizerRouter and DP-rank detokenizer IPC wiring.

Integration coverage (multi detokenizer workers end-to-end) lives in
test_multi_detokenizer.py.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.server_args import (
    DETOKENIZER_ROUTER_TCP_PORT_DELTA,
    ZMQ_TCP_PORT_DELTA,
    PortArgs,
    ServerArgs,
    _build_detokenizer_router_ipc_names,
    use_dp_rank_detokenizer_router,
)
from sglang.srt.utils.network import NetworkAddress
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    BatchStrOutput,
    ClearHiCacheReqInput,
    FreezeGCReq,
)
from sglang.srt.managers.multi_tokenizer_mixin import MultiDetokenizerRouter

register_cpu_ci(est_time=8, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="base-b-test-cpu")

# dp_size used in dp_rank router tests (must be > 2).
DP_SIZE_THREE = 3
DP_SIZE_FOUR = 4


def _make_batch_str_output(
    rids=None,
    http_worker_ipcs=None,
) -> BatchStrOutput:
    n = len(rids) if rids else 0
    return BatchStrOutput(
        rids=rids or [],
        spec_verify_ct=[0] * n,
        spec_num_correct_drafts=[0] * n,
        spec_correct_drafts_histogram=[[] for _ in range(n)],
        finished_reasons=[None] * n,
        output_strs=[""] * n,
        output_ids=[[] for _ in range(n)],
        prompt_tokens=[0] * n,
        completion_tokens=[0] * n,
        reasoning_tokens=[0] * n,
        cached_tokens=[0] * n,
        cached_tokens_details=[{} for _ in range(n)],
        input_token_logprobs_val=[[] for _ in range(n)],
        input_token_logprobs_idx=[[] for _ in range(n)],
        output_token_logprobs_val=[[] for _ in range(n)],
        output_token_logprobs_idx=[[] for _ in range(n)],
        input_top_logprobs_val=[[] for _ in range(n)],
        input_top_logprobs_idx=[[] for _ in range(n)],
        output_top_logprobs_val=[[] for _ in range(n)],
        output_top_logprobs_idx=[[] for _ in range(n)],
        input_token_ids_logprobs_val=[[] for _ in range(n)],
        input_token_ids_logprobs_idx=[[] for _ in range(n)],
        output_token_ids_logprobs_val=[[] for _ in range(n)],
        output_token_ids_logprobs_idx=[[] for _ in range(n)],
        output_token_entropy_val=[0.0] * n,
        output_hidden_states=[None] * n,
        routed_experts=[None] * n,
        indexer_topk=[None] * n,
        placeholder_tokens_idx=[None] * n,
        placeholder_tokens_val=[None] * n,
        retraction_counts=[0] * n,
        http_worker_ipcs=http_worker_ipcs,
    )


class TestUseDpRankDetokenizerRouter(unittest.TestCase):
    def test_default_single_sharding_is_disabled(self):
        server_args = ServerArgs(
            model_path="dummy",
            detokenizer_worker_num=4,
            dp_size=2,
        )
        self.assertEqual(server_args.detokenizer_router_sharding, "single")
        self.assertFalse(use_dp_rank_detokenizer_router(server_args))

    def test_dp_rank_sharding_requires_multi_workers_and_dp(self):
        server_args = ServerArgs(
            model_path="dummy",
            detokenizer_worker_num=4,
            dp_size=2,
            detokenizer_router_sharding="dp_rank",
        )
        self.assertTrue(use_dp_rank_detokenizer_router(server_args))

    def test_dp_rank_sharding_disabled_with_single_worker(self):
        server_args = ServerArgs(
            model_path="dummy",
            detokenizer_worker_num=1,
            dp_size=2,
            detokenizer_router_sharding="dp_rank",
        )
        self.assertFalse(use_dp_rank_detokenizer_router(server_args))

    def test_dp_rank_sharding_disabled_with_dp_size_one(self):
        server_args = ServerArgs(
            model_path="dummy",
            detokenizer_worker_num=4,
            dp_size=1,
            detokenizer_router_sharding="dp_rank",
        )
        self.assertFalse(use_dp_rank_detokenizer_router(server_args))

    def test_dp_rank_sharding_enabled_with_dp_size_three(self):
        server_args = ServerArgs(
            model_path="dummy",
            detokenizer_worker_num=4,
            dp_size=DP_SIZE_THREE,
            detokenizer_router_sharding="dp_rank",
        )
        self.assertTrue(use_dp_rank_detokenizer_router(server_args))

    def test_dp_rank_sharding_enabled_with_dp_size_four(self):
        server_args = ServerArgs(
            model_path="dummy",
            detokenizer_worker_num=4,
            dp_size=DP_SIZE_FOUR,
            detokenizer_router_sharding="dp_rank",
        )
        self.assertTrue(use_dp_rank_detokenizer_router(server_args))


class TestPortArgsSchedulerDetokenizerIpc(unittest.TestCase):
    def test_falls_back_to_global_detokenizer_ipc(self):
        port_args = PortArgs(
            tokenizer_ipc_name="ipc://tok",
            scheduler_input_ipc_name="ipc://sched",
            detokenizer_ipc_name="ipc://detok",
            nccl_port=12345,
            rpc_ipc_name="ipc://rpc",
            metrics_ipc_name="ipc://metrics",
            tokenizer_worker_ipc_name=None,
        )
        self.assertEqual(port_args.scheduler_detokenizer_ipc(0), "ipc://detok")
        self.assertEqual(port_args.scheduler_detokenizer_ipc(1), "ipc://detok")

    def test_uses_per_dp_router_ipc_when_configured(self):
        port_args = PortArgs(
            tokenizer_ipc_name="ipc://tok",
            scheduler_input_ipc_name="ipc://sched",
            detokenizer_ipc_name="ipc://detok",
            nccl_port=12345,
            rpc_ipc_name="ipc://rpc",
            metrics_ipc_name="ipc://metrics",
            tokenizer_worker_ipc_name=None,
            detokenizer_router_ipc_names=["ipc://router0", "ipc://router1"],
        )
        self.assertEqual(port_args.scheduler_detokenizer_ipc(0), "ipc://router0")
        self.assertEqual(port_args.scheduler_detokenizer_ipc(1), "ipc://router1")

    def test_scheduler_detokenizer_ipc_for_all_ranks_when_dp_size_four(self):
        router_ipcs = [f"ipc://router{i}" for i in range(DP_SIZE_FOUR)]
        port_args = PortArgs(
            tokenizer_ipc_name="ipc://tok",
            scheduler_input_ipc_name="ipc://sched",
            detokenizer_ipc_name="ipc://detok",
            nccl_port=12345,
            rpc_ipc_name="ipc://rpc",
            metrics_ipc_name="ipc://metrics",
            tokenizer_worker_ipc_name=None,
            detokenizer_router_ipc_names=router_ipcs,
        )
        for dp_rank, expected in enumerate(router_ipcs):
            self.assertEqual(port_args.scheduler_detokenizer_ipc(dp_rank), expected)


class TestPortArgsInitNewDpRankRouter(unittest.TestCase):
    @patch("sglang.srt.server_args.get_free_port", return_value=29999)
    def test_allocates_per_dp_router_ipc_names(self, _mock_port):
        server_args = ServerArgs(
            model_path="dummy",
            detokenizer_worker_num=4,
            dp_size=3,
            detokenizer_router_sharding="dp_rank",
        )
        port_args = PortArgs.init_new(server_args)
        self.assertIsNotNone(port_args.detokenizer_router_ipc_names)
        self.assertEqual(len(port_args.detokenizer_router_ipc_names), 3)
        for dp_rank in range(3):
            self.assertEqual(
                port_args.scheduler_detokenizer_ipc(dp_rank),
                port_args.detokenizer_router_ipc_names[dp_rank],
            )

    @patch("sglang.srt.server_args.get_free_port", return_value=29999)
    def test_single_sharding_uses_global_detokenizer_ipc(self, _mock_port):
        server_args = ServerArgs(
            model_path="dummy",
            detokenizer_worker_num=4,
            dp_size=2,
            detokenizer_router_sharding="single",
        )
        port_args = PortArgs.init_new(server_args)
        self.assertIsNone(port_args.detokenizer_router_ipc_names)
        self.assertEqual(
            port_args.scheduler_detokenizer_ipc(0), port_args.detokenizer_ipc_name
        )

    @patch("sglang.srt.server_args.get_free_port", return_value=29999)
    def test_per_rank_init_does_not_allocate_router_ipc_names(self, _mock_port):
        server_args = ServerArgs(
            model_path="dummy",
            detokenizer_worker_num=4,
            dp_size=2,
            detokenizer_router_sharding="dp_rank",
        )
        port_args = PortArgs.init_new(server_args, dp_rank=0, worker_ports=[30001])
        self.assertIsNone(port_args.detokenizer_router_ipc_names)

    @patch("sglang.srt.server_args.get_free_port", return_value=29999)
    def test_allocates_four_per_dp_router_ipc_names(self, _mock_port):
        server_args = ServerArgs(
            model_path="dummy",
            detokenizer_worker_num=4,
            dp_size=DP_SIZE_FOUR,
            detokenizer_router_sharding="dp_rank",
        )
        port_args = PortArgs.init_new(server_args)
        self.assertEqual(len(port_args.detokenizer_router_ipc_names), DP_SIZE_FOUR)
        self.assertEqual(len(set(port_args.detokenizer_router_ipc_names)), DP_SIZE_FOUR)


class TestBuildDetokenizerRouterIpcNames(unittest.TestCase):
    def test_ipc_mode_allocates_one_socket_per_dp_rank(self):
        server_args = ServerArgs(model_path="dummy", dp_size=2)
        names = _build_detokenizer_router_ipc_names(server_args)
        self.assertEqual(len(names), 2)
        self.assertTrue(all(n.startswith("ipc://") for n in names))
        self.assertEqual(len(set(names)), 2)

    def test_dp_attention_tcp_mode_uses_dist_init_port_offset(self):
        server_args = ServerArgs(
            model_path="dummy",
            dp_size=2,
            enable_dp_attention=True,
            port=10000,
            dist_init_addr="127.0.0.1:20000",
            nnodes=1,
        )
        names = _build_detokenizer_router_ipc_names(server_args)
        base = 20000 + 1 + DETOKENIZER_ROUTER_TCP_PORT_DELTA
        expected = [NetworkAddress("127.0.0.1", base + i).to_tcp() for i in range(2)]
        self.assertEqual(names, expected)

    def test_single_node_dp_attention_without_dist_init_addr(self):
        server_args = ServerArgs(
            model_path="dummy",
            dp_size=2,
            enable_dp_attention=True,
            port=10000,
            nnodes=1,
        )
        names = _build_detokenizer_router_ipc_names(server_args)
        base = 10000 + ZMQ_TCP_PORT_DELTA + 1 + DETOKENIZER_ROUTER_TCP_PORT_DELTA
        expected = [NetworkAddress("127.0.0.1", base + i).to_tcp() for i in range(2)]
        self.assertEqual(names, expected)

    def test_ipc_mode_allocates_one_socket_per_dp_rank_when_dp_size_four(self):
        server_args = ServerArgs(model_path="dummy", dp_size=DP_SIZE_FOUR)
        names = _build_detokenizer_router_ipc_names(server_args)
        self.assertEqual(len(names), DP_SIZE_FOUR)
        self.assertEqual(len(set(names)), DP_SIZE_FOUR)

    def test_dp_attention_tcp_mode_dp_size_three(self):
        server_args = ServerArgs(
            model_path="dummy",
            dp_size=DP_SIZE_THREE,
            enable_dp_attention=True,
            port=10000,
            dist_init_addr="127.0.0.1:20000",
            nnodes=1,
        )
        names = _build_detokenizer_router_ipc_names(server_args)
        base = 20000 + 1 + DETOKENIZER_ROUTER_TCP_PORT_DELTA
        expected = [
            NetworkAddress("127.0.0.1", base + i).to_tcp() for i in range(DP_SIZE_THREE)
        ]
        self.assertEqual(names, expected)


class TestLaunchDetokenizerSubprocessesDpRank(unittest.TestCase):
    """Verify router process count scales with dp_size when sharding=dp_rank."""

    @patch("sglang.srt.entrypoints.engine.mp.Process")
    def test_dp_size_four_launches_four_routers(self, mock_process_cls):
        from sglang.srt.entrypoints.engine import Engine

        mock_process = MagicMock()
        mock_process_cls.return_value = mock_process

        server_args = ServerArgs(
            model_path="dummy",
            detokenizer_worker_num=4,
            dp_size=DP_SIZE_FOUR,
            detokenizer_router_sharding="dp_rank",
        )
        port_args = PortArgs(
            tokenizer_ipc_name="ipc://tok",
            scheduler_input_ipc_name="ipc://sched",
            detokenizer_ipc_name="ipc://detok",
            nccl_port=12345,
            rpc_ipc_name="ipc://rpc",
            metrics_ipc_name="ipc://metrics",
            tokenizer_worker_ipc_name=None,
            detokenizer_router_ipc_names=[
                f"ipc://router{i}" for i in range(DP_SIZE_FOUR)
            ],
        )

        processes, names = Engine._launch_detokenizer_subprocesses(
            server_args,
            port_args,
            MagicMock(),
        )

        self.assertEqual(mock_process_cls.call_count, 4 + DP_SIZE_FOUR)
        router_names = [n for n in names if n.startswith("detokenizer_router_")]
        self.assertEqual(len(router_names), DP_SIZE_FOUR)
        self.assertEqual(
            router_names,
            [f"detokenizer_router_{i}" for i in range(DP_SIZE_FOUR)],
        )
        self.assertEqual(len(processes), 4 + DP_SIZE_FOUR)


class TestMultiDetokenizerRouterRouting(unittest.TestCase):
    def _make_router(self, worker_ipcs=None):
        worker_ipcs = worker_ipcs or ["ipc://worker0", "ipc://worker1"]
        port_args = PortArgs(
            tokenizer_ipc_name="ipc://tok",
            scheduler_input_ipc_name="ipc://sched",
            detokenizer_ipc_name="ipc://router-in",
            nccl_port=12345,
            rpc_ipc_name="ipc://rpc",
            metrics_ipc_name="ipc://metrics",
            tokenizer_worker_ipc_name=None,
        )
        with patch(
            "sglang.srt.managers.multi_tokenizer_mixin.get_zmq_socket"
        ) as mock_zmq:
            mock_zmq.return_value = MagicMock()
            router = MultiDetokenizerRouter(worker_ipcs, port_args)
        router.socket_mapping = MagicMock()
        return router

    def test_pick_is_deterministic_and_within_worker_pool(self):
        router = self._make_router()
        key = "tokenizer-worker-7"
        self.assertEqual(router._pick(key), router._pick(key))
        self.assertIn(router._pick(key), router.ipc_name_list)

    def test_route_single_request_by_http_worker_ipc(self):
        router = self._make_router()
        req = ClearHiCacheReqInput(rid="rid-0", http_worker_ipc="worker-key-a")
        expected_ipc = router._pick("worker-key-a")

        router._send = MagicMock()
        router._send(router._pick(req.http_worker_ipc), req)

        router._send.assert_called_once_with(expected_ipc, req)

    def test_route_batch_splits_per_item_by_ipc(self):
        router = self._make_router(["ipc://w0", "ipc://w1", "ipc://w2"])
        batch = _make_batch_str_output(
            rids=["r0", "r1"],
            http_worker_ipcs=["ipc-a", "ipc-b"],
        )
        router._send = MagicMock()
        router._route_batch(batch)

        self.assertEqual(router._send.call_count, 2)
        sent_ipcs = {call.args[0] for call in router._send.call_args_list}
        self.assertEqual(
            sent_ipcs,
            {router._pick("ipc-a"), router._pick("ipc-b")},
        )

    def test_route_batch_invalid_ipcs_falls_back_to_first_worker(self):
        router = self._make_router(["ipc://w0", "ipc://w1"])
        batch = _make_batch_str_output(
            rids=["r0"],
            http_worker_ipcs=None,
        )
        router._send = MagicMock()
        router._route_batch(batch)

        router._send.assert_called_once_with("ipc://w0", batch)

    def test_freeze_gc_broadcasts_to_all_workers(self):
        router = self._make_router(["ipc://w0", "ipc://w1"])
        req = FreezeGCReq(rid="freeze")
        router._send = MagicMock()
        router.recv_from_scheduler = MagicMock()
        router.recv_from_scheduler.recv_pyobj.side_effect = [req, KeyboardInterrupt]

        with self.assertRaises(KeyboardInterrupt):
            router.event_loop()

        self.assertEqual(router._send.call_count, 2)
        router._send.assert_any_call("ipc://w0", req)
        router._send.assert_any_call("ipc://w1", req)


if __name__ == "__main__":
    unittest.main()
