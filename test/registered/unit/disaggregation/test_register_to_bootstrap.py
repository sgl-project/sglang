"""Unit tests for srt/disaggregation/common/conn — register_to_bootstrap retry logic."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


import unittest
from unittest.mock import MagicMock, call, patch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase


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
        mock_time.monotonic.return_value = 0.0
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        mock_put.return_value = fail_resp

        mgr = self._make_manager()
        mgr.register_to_bootstrap()

        self.assertEqual(mock_put.call_count, 5)
        # Sleep is only called between attempts, not after the final failure
        self.assertEqual(mock_time.sleep.call_count, 4)

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
            # Self-registered HTTP API port used to derive the PD retract
            # rebootstrap /generate URL on the decode side.
            "prefill_http_port",
        ]
        for field in required_fields:
            self.assertIn(field, payload)
        self.assertEqual(payload["prefill_http_port"], 30000)

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_url_with_dist_init_addr(self, mock_put, mock_time):
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager(dist_init_addr="10.0.0.1:12345")
        mgr.register_to_bootstrap()

        url_used = mock_put.call_args[0][0]
        self.assertIn("10.0.0.1", url_used)

    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    @patch("sglang.srt.disaggregation.common.conn.get_world_group")
    def test_rust_attention_dp_replicates_complete_topology_across_hosts(
        self, mock_world_group, mock_put
    ):
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

        mock_world_group.return_value.all_gather_object.side_effect = gather_topology

        with envs.SGLANG_RUST_SERVER.override(True):
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
                for gather_call in mock_world_group.return_value.all_gather_object.call_args_list
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

    @patch("sglang.srt.disaggregation.common.conn.time")
    @patch("sglang.srt.disaggregation.common.conn.requests.put")
    def test_wildcard_host_ipv6_uses_ipv6_loopback(self, mock_put, mock_time):
        """Same fix for the IPv6 wildcard \"::\": must use IPv6 loopback."""
        mock_time.monotonic.return_value = 0.0
        success_resp = MagicMock()
        success_resp.status_code = 200
        mock_put.return_value = success_resp

        mgr = self._make_manager()
        mgr.bootstrap_host = "::"
        mgr.local_ip = "fd00::1"
        mgr.register_to_bootstrap()

        url_used = mock_put.call_args[0][0]
        # "::" bracketed as "[::]:port" should not appear; loopback should.
        self.assertNotIn("[::]", url_used)
        self.assertIn("[::1]", url_used)

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


if __name__ == "__main__":
    unittest.main()
