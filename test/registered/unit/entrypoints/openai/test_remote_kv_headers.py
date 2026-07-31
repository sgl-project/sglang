import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.serving_chat import (
    _extract_remote_kv_headers,
    _remote_kv_request_id,
    _select_remote_kv_metadata,
    _validated_source_bootstrap_addr,
)
from sglang.srt.managers.io_struct import P2PKVTransferReqInput
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeRequest:
    def __init__(self, headers):
        self.headers = headers
        self.url = SimpleNamespace(scheme="http")


class TestRemoteKVHeaders(CustomTestCase):
    def test_missing_headers_leave_remote_kv_disabled(self):
        source, matched_tokens, reason, target, bootstrap_addr = (
            _extract_remote_kv_headers(_FakeRequest({"host": "127.0.0.1:30001"}))
        )

        self.assertIsNone(source)
        self.assertIsNone(matched_tokens)
        self.assertIsNone(reason)
        self.assertIsNone(target)
        self.assertIsNone(bootstrap_addr)

    def test_remote_kv_headers_extract_source_tokens_reason_and_target(self):
        source, matched_tokens, reason, target, bootstrap_addr = (
            _extract_remote_kv_headers(
                _FakeRequest(
                    {
                        "host": "127.0.0.1:30001",
                        "x-sgl-remote-kv-source": "http://127.0.0.1:30000",
                        "x-sgl-remote-kv-matched-tokens": "128",
                        "x-sgl-remote-kv-reason": "load_imbalance",
                        "x-sgl-remote-kv-source-bootstrap-addr": "127.0.0.1:32400",
                    }
                )
            )
        )

        self.assertEqual(source, "http://127.0.0.1:30000")
        self.assertEqual(matched_tokens, 128)
        self.assertEqual(reason, "load_imbalance")
        self.assertEqual(target, "http://127.0.0.1:30001")
        self.assertEqual(bootstrap_addr, "127.0.0.1:32400")

    def test_explicit_target_header_overrides_host_inference(self):
        source, matched_tokens, reason, target, bootstrap_addr = (
            _extract_remote_kv_headers(
                _FakeRequest(
                    {
                        "host": "router.internal:30100",
                        "x-sgl-remote-kv-source": "http://127.0.0.1:31201",
                        "x-sgl-remote-kv-target": "http://127.0.0.1:31200",
                        "x-sgl-remote-kv-matched-tokens": "128",
                        "x-sgl-remote-kv-reason": "load_imbalance",
                    }
                )
            )
        )

        self.assertEqual(source, "http://127.0.0.1:31201")
        self.assertEqual(matched_tokens, 128)
        self.assertEqual(reason, "load_imbalance")
        self.assertEqual(target, "http://127.0.0.1:31200")
        self.assertIsNone(bootstrap_addr)

    def test_malformed_matched_tokens_disables_remote_kv(self):
        source, matched_tokens, reason, target, bootstrap_addr = (
            _extract_remote_kv_headers(
                _FakeRequest(
                    {
                        "host": "127.0.0.1:30001",
                        "x-sgl-remote-kv-source": "http://127.0.0.1:30000",
                        "x-sgl-remote-kv-matched-tokens": "not-an-int",
                    }
                )
            )
        )

        self.assertIsNone(source)
        self.assertIsNone(matched_tokens)
        self.assertIsNone(reason)
        self.assertIsNone(target)
        self.assertIsNone(bootstrap_addr)

    def test_conflicting_header_and_body_control_bundle_disables_remote_kv(self):
        request = SimpleNamespace(
            remote_kv_source_url="http://127.0.0.1:39999",
            remote_kv_target_url="http://127.0.0.1:30001",
            remote_kv_matched_tokens=128,
            remote_kv_reason="load_imbalance",
            remote_kv_source_bootstrap_addr="127.0.0.1:32400",
            remote_kv_token_ids=[1, 2, 3],
        )
        metadata = _select_remote_kv_metadata(
            _FakeRequest(
                {
                    "host": "127.0.0.1:30001",
                    "x-sgl-remote-kv-source": "http://127.0.0.1:30000",
                    "x-sgl-remote-kv-target": "http://127.0.0.1:30001",
                    "x-sgl-remote-kv-matched-tokens": "128",
                    "x-sgl-remote-kv-reason": "load_imbalance",
                    "x-sgl-remote-kv-source-bootstrap-addr": "127.0.0.1:32400",
                }
            ),
            request,
        )

        self.assertEqual(metadata, (None, None, None, None, None, None))

    def test_matching_header_and_body_bundle_keeps_json_token_ids(self):
        request = SimpleNamespace(
            remote_kv_source_url="http://127.0.0.1:30000",
            remote_kv_target_url="http://127.0.0.1:30001",
            remote_kv_matched_tokens=128,
            remote_kv_reason="load_imbalance",
            remote_kv_source_bootstrap_addr="127.0.0.1:32400",
            remote_kv_token_ids=[1, 2, 3],
        )
        metadata = _select_remote_kv_metadata(
            _FakeRequest(
                {
                    "host": "127.0.0.1:30001",
                    "x-sgl-remote-kv-source": "http://127.0.0.1:30000",
                    "x-sgl-remote-kv-target": "http://127.0.0.1:30001",
                    "x-sgl-remote-kv-matched-tokens": "128",
                    "x-sgl-remote-kv-reason": "load_imbalance",
                    "x-sgl-remote-kv-source-bootstrap-addr": "127.0.0.1:32400",
                }
            ),
            request,
        )

        self.assertEqual(
            metadata,
            (
                "http://127.0.0.1:30000",
                128,
                "load_imbalance",
                "http://127.0.0.1:30001",
                "127.0.0.1:32400",
                [1, 2, 3],
            ),
        )

    def test_source_bootstrap_addr_must_match_source_host_and_nonzero_port(self):
        self.assertEqual(
            _validated_source_bootstrap_addr(
                "http://[fd00::10]:30000", "[fd00::10]:32400"
            ),
            "[fd00::10]:32400",
        )
        self.assertIsNone(
            _validated_source_bootstrap_addr(
                "http://127.0.0.1:30000", "127.0.0.2:32400"
            )
        )
        self.assertIsNone(
            _validated_source_bootstrap_addr("http://127.0.0.1:30000", "127.0.0.1:0")
        )

    def test_remote_kv_request_id_preserves_existing_id(self):
        self.assertEqual(_remote_kv_request_id("request-123"), "request-123")

    def test_remote_kv_request_id_generates_stable_log_correlation_value(self):
        first = _remote_kv_request_id("")
        second = _remote_kv_request_id(None)

        self.assertTrue(first.startswith("p2p-"))
        self.assertTrue(second.startswith("p2p-"))
        self.assertNotEqual(first, second)


class TestRemoteKVSchedulerControl(CustomTestCase):
    def _scheduler_with_tp(self, tp_size):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ps = SimpleNamespace(tp_size=tp_size)
        return scheduler

    def test_disabled_feature_falls_back(self):
        scheduler = self._scheduler_with_tp(1)
        scheduler.server_args = SimpleNamespace(enable_prefill_p2p_kv_transfer=False)
        ret = scheduler.handle_p2p_kv_transfer(
            P2PKVTransferReqInput(
                source_url="http://127.0.0.1:30000",
                target_url="http://127.0.0.1:30001",
                token_ids=[1, 2, 3, 4],
                matched_tokens=3,
                dry_run=True,
            )
        )

        self.assertFalse(ret.success)
        self.assertTrue(ret.fallback_recompute)
        self.assertIn("--enable-prefill-p2p-kv-transfer", ret.message)

    def test_dry_run_accepts_tp1_transfer(self):
        ret = self._scheduler_with_tp(1).handle_p2p_kv_transfer(
            P2PKVTransferReqInput(
                source_url="http://127.0.0.1:30000",
                target_url="http://127.0.0.1:30001",
                token_ids=[1, 2, 3, 4],
                matched_tokens=3,
                dry_run=True,
            )
        )

        self.assertTrue(ret.success)
        self.assertFalse(ret.fallback_recompute)
        self.assertIn("identical_tp_pp_layout_supported", ret.experimental_limitations)

    def test_non_tp1_without_transfer_engine_falls_back(self):
        ret = self._scheduler_with_tp(2).handle_p2p_kv_transfer(
            P2PKVTransferReqInput(
                source_url="http://127.0.0.1:30000",
                target_url="http://127.0.0.1:30001",
                token_ids=[1, 2, 3, 4],
                matched_tokens=3,
            )
        )

        self.assertFalse(ret.success)
        self.assertTrue(ret.fallback_recompute)
        self.assertIn("unavailable on this worker", ret.message)

    def test_non_tp1_delegates_to_transfer_engine(self):
        scheduler = self._scheduler_with_tp(2)
        scheduler.p2p_kv_transfer_engine = SimpleNamespace(
            start_transfer=lambda req: SimpleNamespace(
                success=True,
                message="transferred",
                source_url=req.source_url,
                target_url=req.target_url,
                matched_tokens=req.matched_tokens,
                transferred_tokens=req.matched_tokens,
                fallback_recompute=False,
                experimental_limitations=["identical_tp_pp_layout_supported"],
            )
        )

        ret = scheduler.handle_p2p_kv_transfer(
            P2PKVTransferReqInput(
                source_url="http://127.0.0.1:30000",
                target_url="http://127.0.0.1:30001",
                token_ids=[1, 2, 3, 4],
                matched_tokens=3,
            )
        )

        self.assertTrue(ret.success)
        self.assertFalse(ret.fallback_recompute)
        self.assertEqual(ret.transferred_tokens, 3)

    def test_non_dry_run_delegates_to_transfer_engine(self):
        scheduler = self._scheduler_with_tp(1)
        scheduler.p2p_kv_transfer_engine = SimpleNamespace(
            start_transfer=lambda req: SimpleNamespace(
                success=True,
                message="transferred",
                source_url=req.source_url,
                target_url=req.target_url,
                matched_tokens=req.matched_tokens,
                transferred_tokens=req.matched_tokens,
                fallback_recompute=False,
                experimental_limitations=[],
            )
        )

        ret = scheduler.handle_p2p_kv_transfer(
            P2PKVTransferReqInput(
                source_url="http://127.0.0.1:30000",
                target_url="http://127.0.0.1:30001",
                token_ids=[1, 2, 3, 4],
                matched_tokens=3,
            )
        )

        self.assertTrue(ret.success)
        self.assertFalse(ret.fallback_recompute)
        self.assertEqual(ret.transferred_tokens, 3)


if __name__ == "__main__":
    unittest.main()
