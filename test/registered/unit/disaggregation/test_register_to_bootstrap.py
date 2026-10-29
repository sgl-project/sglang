"""Unit tests for register_to_bootstrap: retry logic, and the bootstrap server
listening before the launcher starts the schedulers that register to it."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


import socket
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import requests

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.test_utils import CustomTestCase

_PREFILL_RANK_REGISTRATION = {
    "attn_tp_size": 1,
    "attn_tp_rank": 0,
    "attn_cp_size": 1,
    "attn_cp_rank": 0,
    "attn_dp_size": 1,
    "attn_dp_rank": 0,
    "pp_size": 1,
    "pp_rank": 0,
    "system_dp_size": 1,
    "system_dp_rank": 0,
    "rank_ip": "127.0.0.1",
    "rank_port": 12345,
    "page_size": 16,
    "kv_cache_dtype": "auto",
}


def _register_prefill_rank(port):
    # The bootstrap server binds on a background thread, so retry briefly
    # the way the real registration does.
    deadline = time.monotonic() + 2.0
    while True:
        try:
            return requests.put(
                f"http://127.0.0.1:{port}/route",
                json=_PREFILL_RANK_REGISTRATION,
                timeout=5,
            ).status_code
        except requests.ConnectionError:
            if time.monotonic() > deadline:
                return "connection refused"
            time.sleep(0.05)


class TestRegisterToBootstrap(CustomTestCase):
    """Tests for CommonKVManager.register_to_bootstrap retry/backoff behavior."""

    def setUp(self):
        # register_to_bootstrap reads get_parallel().load_balance_method /
        # .enable_dsa_cache_layer_split and get_serving().port from the
        # published config.
        override = get_context().override_server_args(
            load_balance_method="follow_bootstrap_room", port=30000
        )
        override.install()
        self.addCleanup(override.restore)

    def test_sender_dp_rank_registration(self):
        from sglang.srt.disaggregation.base.conn import KVPoll
        from sglang.srt.disaggregation.common.conn import CommonKVSender

        for force_query in (False, True):
            for dp_rank in (0, 1):
                with self.subTest(force_query=force_query, dp_rank=dp_rank):
                    mgr = MagicMock(
                        is_dummy_cp_rank=False,
                        attn_dp_rank=dp_rank,
                        deferred_bootstrap=None,
                    )
                    sender = MagicMock(spec=CommonKVSender)
                    sender._register_prefill_dp_rank = (
                        CommonKVSender._register_prefill_dp_rank.__get__(sender)
                    )
                    with (
                        patch(
                            "sglang.srt.disaggregation.common.conn.requests.post"
                        ) as mock_post,
                        get_context().override_server_args(
                            dp_size=4, load_balance_method="follow_bootstrap_room"
                        ),
                        envs.SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK.override(
                            force_query
                        ),
                    ):
                        mock_post.return_value.status_code = 200
                        CommonKVSender.__init__(
                            sender,
                            mgr=mgr,
                            bootstrap_addr="127.0.0.1:8765",
                            bootstrap_room=4,
                        )

                    self.assertEqual(
                        mock_post.call_args_list,
                        [
                            call(
                                "http://127.0.0.1:8765/register_dp_rank",
                                json={"bootstrap_room": 4, "dp_rank": dp_rank},
                                timeout=5,
                            )
                        ]
                        if force_query
                        else [],
                    )
                    conflict = not force_query and dp_rank != 0
                    self.assertEqual(mgr.record_failure.call_count, int(conflict))
                    mgr.update_status.assert_called_with(
                        4, KVPoll.Failed if conflict else KVPoll.Bootstrapping
                    )

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_succeeds_on_first_attempt(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_put.return_value = mock_response

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        mock_put.assert_called_once()
        mock_time.sleep.assert_not_called()

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_succeeds_after_retries(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.side_effect = [fail_resp, fail_resp, success_resp]

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 3)
        self.assertEqual(mock_time.sleep.call_count, 2)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_all_retries_exhausted(self, mock_put, mock_time):
        """Prefill never re-registers, so running out of retries must fail
        startup instead of leaving the rank unregistered."""
        mock_time.monotonic.return_value = 0.0
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        mock_put.return_value = fail_resp

        mgr = self._make_manager()
        with self.assertRaisesRegex(RuntimeError, "status 503"):
            mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 5)
        # Sleep is only called between attempts, not after the final failure
        self.assertEqual(mock_time.sleep.call_count, 4)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_exhausted_retries_report_root_cause(self, mock_put, mock_time):
        """Only the final error survives an error-level log config, so it must
        name the root cause of the last failed attempt."""
        mock_time.monotonic.return_value = 0.0
        wrapped = Exception("Max retries exceeded with url: /route")
        wrapped.__cause__ = ConnectionRefusedError("connection refused")
        mock_put.side_effect = wrapped

        mgr = self._make_manager()
        with self.assertRaisesRegex(RuntimeError, "connection refused"):
            mgr.register_to_bootstrap()

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_exception_with_nested_cause(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0

        root_exc = ConnectionRefusedError("connection refused")
        inner_exc = OSError("os error")
        inner_exc.__cause__ = root_exc
        outer_exc = Exception("wrapped")
        outer_exc.__cause__ = inner_exc

        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.side_effect = [outer_exc, success_resp]

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 2)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_exception_with_no_cause(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0

        exc = ConnectionError("plain connection error")
        exc.__cause__ = None

        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.side_effect = [exc, success_resp]

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 2)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_backoff_delay_exponential(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        mock_put.return_value = fail_resp

        mgr = self._make_manager()
        with self.assertRaises(RuntimeError):
            mgr.register_to_bootstrap()

        # With monotonic() = 0.0, jitter factor = 0.75 + 0.25 * (0.0 % 1) = 0.75
        # delay = min(1.0 * 2^attempt, 30.0) * 0.75
        # Sleep happens only between attempts (attempt 0..3), not after the final failure
        expected_calls = [call(0.75), call(1.5), call(3.0), call(6.0)]
        self.assertEqual(mock_time.sleep.call_args_list, expected_calls)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_jitter_never_exceeds_max_delay(self, mock_put, mock_time):
        """Guard against operator-precedence regressions in the jitter factor.

        The jitter factor must stay in [0.75, 1.0), so a delay capped at
        max_delay must never exceed max_delay after applying jitter.
        """
        # monotonic() returns a value whose fractional part is close to 1.
        # If the parentheses around `time.monotonic() % 1` were dropped, the
        # jitter factor could grow up to ~1.75 and blow past max_delay.
        mock_time.monotonic.return_value = 999.9999
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        mock_put.return_value = fail_resp

        mgr = self._make_manager()
        with self.assertRaises(RuntimeError):
            mgr.register_to_bootstrap()

        max_delay = 30.0
        for sleep_call in mock_time.sleep.call_args_list:
            actual_delay = sleep_call[0][0]
            self.assertLess(actual_delay, max_delay)
            self.assertGreaterEqual(actual_delay, 0.75)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_payload_contains_required_fields(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        call_kwargs = mock_put.call_args
        payload = call_kwargs[1]["json"]
        required_fields = [
            "attn_tp_size",
            "attn_tp_rank",
            "attn_cp_size",
            "attn_cp_rank",
            "attn_dp_size",
            "attn_dp_rank",
            "pp_size",
            "pp_rank",
            "system_dp_size",
            "system_dp_rank",
            "rank_ip",
            "rank_port",
            "page_size",
            "kv_cache_dtype",
            # Decode pulls from every shard-group rank when this is set.
            "enable_kv_cache_sharding",
            # Self-registered HTTP API port used to derive the PD retract
            # rebootstrap /generate URL on the decode side.
            "prefill_http_port",
        ]
        for field in required_fields:
            self.assertIn(field, payload)
        self.assertEqual(payload["prefill_http_port"], 30000)

    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_rust_attention_dp_replicates_complete_topology_across_hosts(
        self, mock_put
    ):
        mock_world_group = MagicMock()
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        schedulers = (
            (0, 0, "10.0.0.1", 17000, 8765),
            (0, 1, "10.0.0.1", 17001, None),
            (1, 0, "10.0.0.2", 17002, 8766),
            (1, 1, "10.0.0.2", 17003, None),
        )

        def gather_topology(payload):
            return [
                {
                    **payload,
                    "attn_dp_rank": dp_rank,
                    "attn_tp_rank": tp_rank,
                    "rank_ip": host,
                    "rank_port": rank_port,
                }
                for dp_rank, tp_rank, host, rank_port, _ in schedulers
            ]

        mock_world_group.all_gather_object.side_effect = gather_topology

        with (
            get_parallel().override(world_group=mock_world_group),
            envs.SGLANG_RUST_SERVER.override(True),
        ):
            for dp_rank, tp_rank, local_ip, _, rust_http_port in schedulers:
                manager = self._make_manager()
                manager.attn_dp_size = 2
                manager.attn_dp_rank = dp_rank
                manager.attn_tp_size = 2
                manager.attn_tp_rank = tp_rank
                manager.local_ip = local_ip
                manager.bootstrap_host = local_ip
                manager.kv_args.rust_http_port = rust_http_port
                manager.register_to_bootstrap()

        topology_by_registry = {}
        for put_call in mock_put.call_args_list:
            payload = put_call.kwargs["json"]
            topology_by_registry.setdefault(put_call.args[0], {})[
                (payload["attn_dp_rank"], payload["attn_tp_rank"])
            ] = (payload["rank_ip"], payload["rank_port"])
        complete_topology = {
            (dp, tp): (host, rank_port) for dp, tp, host, rank_port, _ in schedulers
        }
        self.assertEqual(
            topology_by_registry,
            {
                "http://10.0.0.1:8765/route": complete_topology,
                "http://10.0.0.2:8766/route": complete_topology,
            },
        )
        self.assertEqual(mock_put.call_count, 8)
        self.assertEqual(
            {
                (put_call.args[0], put_call.kwargs["json"]["prefill_http_port"])
                for put_call in mock_put.call_args_list
            },
            {
                ("http://10.0.0.1:8765/route", 8765),
                ("http://10.0.0.2:8766/route", 8766),
            },
        )
        self.assertEqual(
            [
                (
                    gather_call.args[0]["attn_dp_rank"],
                    gather_call.args[0]["attn_tp_rank"],
                )
                for gather_call in mock_world_group.all_gather_object.call_args_list
            ],
            [(dp, tp) for dp, tp, _, _, _ in schedulers],
        )

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_wildcard_host_0000_uses_ipv4_loopback(self, mock_put, mock_time):
        """When --host 0.0.0.0 is used, the PUT must target IPv4 loopback.

        Scenario: cross-node P/D disagg where each role runs on a single node
        (tp=1).  Each machine runs its own SGLang instance with --host 0.0.0.0
        to accept remote connections.  dist_init_addr is None because tp=1
        needs no multi-node rendezvous, so register_to_bootstrap takes the
        else-branch and would use bootstrap_host="0.0.0.0" as the PUT target.
        aiohttp >=3.9 rejects that with HTTP 403 because 0.0.0.0 is not a
        valid Host header value.

        Fix: substitute same-family loopback when bootstrap_host is a wildcard.
        """
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager()
        mgr.bootstrap_host = "0.0.0.0"
        mgr.local_ip = "192.168.1.10"
        mgr.register_to_bootstrap()

        url_used = mock_put.call_args[0][0]
        self.assertNotIn("0.0.0.0", url_used)
        self.assertIn("127.0.0.1", url_used)

    def _make_manager(self, dist_init_addr=None):
        """Create a lightweight mock manager that has the attributes needed
        by register_to_bootstrap, without going through CommonKVManager.__init__
        (which requires zmq, ServerArgs model resolution, etc.)."""
        from sglang.srt.disaggregation.common.conn import CommonKVManager

        mgr = MagicMock(spec=CommonKVManager)
        # Bind the real method to the mock
        mgr.register_to_bootstrap = CommonKVManager.register_to_bootstrap.__get__(
            mgr, CommonKVManager
        )
        mgr._register_topology_row = CommonKVManager._register_topology_row.__get__(
            mgr, CommonKVManager
        )

        # Set attributes that register_to_bootstrap reads
        mgr.dist_init_addr = dist_init_addr
        mgr.bootstrap_host = "127.0.0.1"
        mgr.bootstrap_port = 8765
        mgr.attn_tp_size = 1
        mgr.attn_tp_rank = 0
        mgr.attn_cp_size = 1
        mgr.attn_cp_rank = 0
        mgr.attn_dp_size = 1
        mgr.attn_dp_rank = 0
        mgr.pp_size = 1
        mgr.pp_rank = 0
        mgr.system_dp_size = 1
        mgr.system_dp_rank = 0
        mgr.local_ip = "127.0.0.1"
        mgr.rank_port = 12345

        mgr.kv_args = MagicMock()
        mgr.kv_args.page_size = 16
        mgr.kv_args.rust_http_port = None
        # Resolved per-runner value threaded through KVArgs (the payload field).
        mgr.kv_cache_dtype_str = "auto"

        return mgr


class TestBootstrapServerLaunchOrder(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.srt.entrypoints import engine as engine_module
        from sglang.srt.entrypoints.engine import Engine, SchedulerInitResult
        from sglang.srt.runtime_context import reset_context
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.utils.network import get_free_port

        self.port = get_free_port()
        self.server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="prefill",
            disaggregation_bootstrap_port=self.port,
        )
        # Validation is stubbed so the dummy model path still launches.
        self.server_args.check_server_args = MagicMock()
        self.addCleanup(reset_context)
        self.servers = []
        start_service = engine_module.start_disagg_service

        def start_bootstrap():
            server = start_service()
            self.servers.append(server)
            self.addCleanup(server.close)
            return server

        self.registration_statuses = []
        self.scheduler_init_result = SchedulerInitResult(
            scheduler_infos=[{"max_req_input_len": 1024}]
        )

        def launch_prefill_schedulers(*args, **kwargs):
            self.registration_statuses.append(_register_prefill_rank(self.port))
            return self.scheduler_init_result, []

        self.launch_schedulers = MagicMock(side_effect=launch_prefill_schedulers)
        self.init_tokenizer = MagicMock(return_value=(MagicMock(), MagicMock()))
        for patcher in (
            patch.object(engine_module, "configure_logger"),
            patch.object(engine_module, "_set_envs_and_config"),
            patch.object(engine_module, "load_plugins"),
            patch.object(
                engine_module, "start_disagg_service", side_effect=start_bootstrap
            ),
            patch.object(
                Engine,
                "_launch_scheduler_processes",
                self.launch_schedulers,
            ),
            patch.object(
                Engine, "_launch_detokenizer_subprocesses", return_value=([], [])
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _launch(self):
        from sglang.srt.entrypoints.engine import Engine

        result = Engine._launch_subprocesses(
            server_args=self.server_args,
            init_tokenizer_manager_func=self.init_tokenizer,
            run_scheduler_process_func=MagicMock(),
            run_detokenizer_process_func=MagicMock(),
            port_args=SimpleNamespace(),
        )
        self.addCleanup(result[4].stop)
        return result

    def _assert_bootstrap_stopped(self):
        self.assertFalse(self.servers[-1].thread.is_alive())
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", self.port))

    def test_prefill_ranks_can_register_while_schedulers_launch(self):
        """Prefill schedulers register to the bootstrap server once, during their
        own init, so it must already be listening when the launcher starts them."""
        self._launch()
        self.assertEqual(self.registration_statuses, [200])
        # The launcher hands the server to its caller, so it must outlive the launch.
        self.assertEqual(_register_prefill_rank(self.port), 200)

    def test_rust_server_backend_bootstrap_listens_before_schedulers_launch(self):
        """In rust-server mode a transfer backend's own bootstrap server must also
        be listening when the launcher starts the schedulers that register to it."""
        from sglang.srt.entrypoints import engine as engine_module
        from sglang.srt.managers import disagg_service

        with (
            envs.SGLANG_RUST_SERVER.override(True),
            patch.object(
                disagg_service, "rust_server_serves_kv_bootstrap", return_value=False
            ),
            # Route the rust-server path through setUp's recording wrapper.
            patch.object(
                disagg_service,
                "start_disagg_service",
                side_effect=engine_module.start_disagg_service,
            ),
        ):
            self._launch()

        self.assertEqual(self.registration_statuses, [200])
        self.assertEqual(_register_prefill_rank(self.port), 200)

    def test_startup_failure_releases_bootstrap_port(self):
        """A failed launch must leave the bootstrap port reusable by a retry."""
        for stage in ("scheduler", "tokenizer", "readiness"):
            with self.subTest(stage=stage):
                failure = RuntimeError(f"{stage} initialization failed")

                def fail_schedulers(*args, **kwargs):
                    self.assertEqual(_register_prefill_rank(self.port), 200)
                    raise failure

                target, name = {
                    "scheduler": (self.launch_schedulers, "side_effect"),
                    "tokenizer": (self.init_tokenizer, "side_effect"),
                    "readiness": (self.scheduler_init_result, "wait_for_ready"),
                }[stage]
                effect = (
                    fail_schedulers
                    if stage == "scheduler"
                    else MagicMock(side_effect=failure)
                )
                try:
                    with patch.object(target, name, effect):
                        with self.assertRaisesRegex(RuntimeError, str(failure)):
                            self._launch()
                    self._assert_bootstrap_stopped()
                finally:
                    self.servers[-1].close()

    def test_config_store_failure_releases_bootstrap_port(self):
        """Config-store startup failure must not leave its bootstrap listener alive."""
        from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer
        from sglang.srt.managers import disagg_service

        failure = RuntimeError("config store initialization failed")
        init_bootstrap = CommonKVBootstrapServer.__init__

        def record_bootstrap(server, *args, **kwargs):
            init_bootstrap(server, *args, **kwargs)
            self.servers.append(server)
            self.addCleanup(server.close)

        def fail_config_store(**kwargs):
            self.assertEqual(_register_prefill_rank(self.port), 200)
            raise failure

        with (
            patch.object(CommonKVBootstrapServer, "__init__", record_bootstrap),
            patch.object(
                disagg_service,
                "maybe_create_ascend_config_store",
                side_effect=fail_config_store,
            ),
        ):
            with self.assertRaises(RuntimeError) as raised:
                self._launch()

        self.assertIs(raised.exception, failure)
        self._assert_bootstrap_stopped()

    def test_engine_shutdown_releases_bootstrap_port(self):
        """Shutdown must release the listener, including when called twice."""
        from sglang.srt.entrypoints import engine as engine_module
        from sglang.srt.entrypoints.engine import Engine

        self._launch()
        engine = object.__new__(Engine)
        engine.tokenizer_manager = None
        engine._scheduler_init_result = self.scheduler_init_result

        with patch.object(engine_module, "kill_process_tree"):
            engine.shutdown()
            engine.shutdown()

        self._assert_bootstrap_stopped()
        self.assertIsNone(self.scheduler_init_result.disaggregation_bootstrap_server)

    def test_close_before_bootstrap_thread_starts(self):
        """An immediate launch error may close the server before its thread runs."""
        from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer

        release = threading.Event()
        run_server = CommonKVBootstrapServer._run_server

        def delayed_run(server):
            if release.wait(timeout=5):
                run_server(server)

        with patch.object(CommonKVBootstrapServer, "_run_server", delayed_run):
            server = CommonKVBootstrapServer("127.0.0.1", self.port)
        self.servers.append(server)
        self.addCleanup(server.close)
        self.addCleanup(server.thread.join, timeout=5)
        self.addCleanup(release.set)
        join = server.thread.join

        def release_and_join(timeout):
            release.set()
            join(timeout)

        # Hold startup until close has requested shutdown and joins the thread.
        with patch.object(server.thread, "join", side_effect=release_and_join):
            server.close()

        self._assert_bootstrap_stopped()


if __name__ == "__main__":
    unittest.main()
