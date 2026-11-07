"""Regression tests for Decode bootstrap endpoint validation."""

import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.decode import DecodePreallocQueue, _bootstrap_addr
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBootstrapAddrValidation(CustomTestCase):
    def test_request_bootstrap_address_is_normalized(self):
        req = SimpleNamespace(bootstrap_host="Prefill-A", bootstrap_port=8998)

        self.assertEqual(_bootstrap_addr(req), "prefill-a:8998")

    def _manager(self, allowed_addrs):
        manager = CommonKVManager.__new__(CommonKVManager)
        manager.prefill_bootstrap_allowlist = set(allowed_addrs)
        manager._empty_bootstrap_allowlist_error_logged = False
        manager._empty_bootstrap_allowlist_log_lock = threading.Lock()
        manager.prefill_info_table = {}
        manager.kv_args = SimpleNamespace(page_size=16, state_types=[])
        manager.kv_cache_dtype_str = "auto"
        manager.kv_layout = None
        manager.dsv41_spec_layout = None
        manager.attn_tp_size = 1
        manager.attn_cp_size = 1
        manager.dcp_size = 1
        manager.is_mla_backend = False
        manager.is_hybrid_mla_backend = False
        manager._resolve_rank_mapping = Mock()
        return manager

    @staticmethod
    def _response():
        response = Mock(status_code=200)
        response.iter_content.return_value = [
            json.dumps(
                {
                    "attn_tp_size": 1,
                    "attn_cp_size": 1,
                    "dp_size": 1,
                    "pp_size": 1,
                    "page_size": 16,
                    "kv_cache_dtype": "auto",
                    "follow_bootstrap_room": False,
                }
            ).encode()
        ]
        return response

    @patch("sglang.srt.disaggregation.common.conn.requests.get")
    def test_unregistered_bootstrap_address_is_rejected_without_http(self, get):
        manager = self._manager({"prefill-a:8998"})

        self.assertFalse(manager.try_ensure_parallel_info("attacker:8998"))

        get.assert_not_called()
        self.assertEqual(manager.prefill_info_table, {})

    @patch("sglang.srt.disaggregation.common.conn.requests.get")
    def test_malformed_bootstrap_address_is_rejected_without_http(self, get):
        manager = self._manager({"prefill-a:8998"})

        self.assertFalse(manager.try_ensure_parallel_info("not-an-address"))

        get.assert_not_called()
        self.assertEqual(manager.prefill_info_table, {})

    @patch("sglang.srt.disaggregation.common.conn.requests.get")
    def test_empty_allowlist_allows_lookup_and_logs_recommendation_once(self, get):
        manager = self._manager(set())
        get.return_value = self._response()

        with self.assertLogs(
            "sglang.srt.disaggregation.common.conn", level="WARNING"
        ) as captured:
            self.assertTrue(manager.try_ensure_parallel_info("prefill-a:8998"))
            self.assertTrue(manager.try_ensure_parallel_info("prefill-a:8998"))

        self.assertEqual(len(captured.output), 1)
        self.assertIn("--disaggregation-bootstrap-allowlist", captured.output[0])
        get.assert_called_once_with(
            "http://prefill-a:8998/route?"
            "prefill_dp_rank=-1&prefill_cp_rank=-1&"
            "target_tp_rank=-1&target_pp_rank=-1",
            timeout=(2, 2),
            allow_redirects=False,
            stream=True,
        )
        self.assertIn("prefill-a:8998", manager.prefill_info_table)

    @patch("sglang.srt.disaggregation.common.conn.requests.get")
    def test_registered_bootstrap_address_is_queried_with_timeout(self, get):
        manager = self._manager({"prefill-a:8998"})
        get.return_value = self._response()

        self.assertTrue(manager.try_ensure_parallel_info("prefill-a:8998"))

        get.assert_called_once_with(
            "http://prefill-a:8998/route?"
            "prefill_dp_rank=-1&prefill_cp_rank=-1&"
            "target_tp_rank=-1&target_pp_rank=-1",
            timeout=(2, 2),
            allow_redirects=False,
            stream=True,
        )
        self.assertIn("prefill-a:8998", manager.prefill_info_table)

    @patch("sglang.srt.disaggregation.common.conn.requests.get")
    def test_registered_lookup_is_cached_and_bounded(self, get):
        manager = self._manager({"prefill-a:8998"})
        get.return_value = self._response()

        self.assertTrue(manager.try_ensure_parallel_info("prefill-a:8998"))
        self.assertTrue(manager.try_ensure_parallel_info("prefill-a:8998"))

        get.assert_called_once()
        self.assertLessEqual(
            len(manager.prefill_info_table), len(manager.prefill_bootstrap_allowlist)
        )

    @patch("sglang.srt.disaggregation.common.conn.time.monotonic", side_effect=[0, 3])
    @patch("sglang.srt.disaggregation.common.conn.requests.get")
    def test_slow_response_body_expires_at_total_deadline(self, get, monotonic):
        manager = self._manager({"prefill-a:8998"})
        get.return_value = self._response()

        self.assertFalse(manager.try_ensure_parallel_info("prefill-a:8998"))

        self.assertEqual(manager.prefill_info_table, {})
        get.return_value.close.assert_called_once()

    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_unregistered_address_is_aborted_without_retry(self, prepare_abort):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.kv_manager = self._manager({"prefill-a:8998"})
        queue.scheduler = SimpleNamespace(
            metrics_reporter=SimpleNamespace(enable_metrics=False)
        )
        queue._ensure_retry_count = {}
        queue._ensure_last_attempt_time = {}
        decode_req = SimpleNamespace(req=SimpleNamespace(), kv_receiver=None)

        ready, remaining = queue._ensure_prefill_info({"attacker:8998": [decode_req]})

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])
        self.assertEqual(queue._ensure_retry_count, {})
        self.assertEqual(queue._ensure_last_attempt_time, {})
        prepare_abort.assert_called_once_with(
            decode_req.req,
            "Unregistered Prefill bootstrap address: attacker:8998",
            status_code=400,
        )

    def test_address_case_variants_share_one_retry_budget(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.kv_manager = self._manager({"prefill-a:8998"})
        queue.kv_manager.try_ensure_parallel_info = Mock(return_value=False)
        queue.scheduler = SimpleNamespace(
            metrics_reporter=SimpleNamespace(enable_metrics=False)
        )
        queue._ensure_retry_count = {}
        queue._ensure_last_attempt_time = {}
        queue._max_ensure_retries = 15
        queue._ensure_retry_interval = 1.0
        requests = [
            SimpleNamespace(req=SimpleNamespace(), kv_receiver=None),
            SimpleNamespace(req=SimpleNamespace(), kv_receiver=None),
        ]

        ready, remaining = queue._ensure_prefill_info(
            {"Prefill-A:8998": requests[:1], "prefill-a:8998": requests[1:]}
        )

        self.assertEqual(ready, {})
        self.assertEqual(remaining, requests)
        self.assertEqual(queue._ensure_retry_count, {"prefill-a:8998": 1})
        queue.kv_manager.try_ensure_parallel_info.assert_called_once_with(
            "prefill-a:8998"
        )

    @patch("sglang.srt.disaggregation.decode.time.monotonic", side_effect=[1, 2, 3])
    def test_registered_lookup_stops_after_retry_limit(self, monotonic):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.kv_manager = self._manager({"prefill-a:8998"})
        queue.kv_manager.try_ensure_parallel_info = Mock(return_value=False)
        queue.scheduler = SimpleNamespace(
            metrics_reporter=SimpleNamespace(enable_metrics=False)
        )
        queue._ensure_retry_count = {}
        queue._ensure_last_attempt_time = {}
        queue._max_ensure_retries = 3
        queue._ensure_retry_interval = 0
        decode_req = SimpleNamespace(req=SimpleNamespace(), kv_receiver=None)

        for _ in range(queue._max_ensure_retries):
            _, remaining = queue._ensure_prefill_info({"prefill-a:8998": [decode_req]})

        self.assertEqual(remaining, [])
        self.assertEqual(queue._ensure_retry_count, {})
        self.assertEqual(queue._ensure_last_attempt_time, {})
        self.assertEqual(
            queue.kv_manager.try_ensure_parallel_info.call_count,
            queue._max_ensure_retries,
        )
        self.assertEqual(monotonic.call_count, queue._max_ensure_retries)


if __name__ == "__main__":
    unittest.main()
