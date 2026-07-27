import unittest
from unittest.mock import AsyncMock, MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestP2PKVTransferEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from sglang.srt.entrypoints.http_server import app, set_global_state
            from sglang.srt.managers.io_struct import P2PKVTransferReqOutput
            from starlette.testclient import TestClient
        except (ImportError, OSError):
            raise unittest.SkipTest(
                "http_server import requires runtime libraries not available on CPU"
            )

        mock_state = MagicMock()
        tokenizer_manager = mock_state.tokenizer_manager
        tokenizer_manager.p2p_kv_transfer = AsyncMock(
            return_value=P2PKVTransferReqOutput(
                success=True,
                message="ok",
                source_url="http://source:30000",
                target_url="http://target:30000",
                matched_tokens=4,
                transferred_tokens=4,
                fallback_recompute=False,
            )
        )

        set_global_state(mock_state)
        cls.client = TestClient(app)
        cls.tokenizer_manager = tokenizer_manager

    def test_p2p_kv_transfer_accepts_json_body_and_returns_msgspec_output(self):
        response = self.client.post(
            "/experimental/p2p_kv_transfer",
            json={
                "source_url": "http://source:30000",
                "target_url": "http://target:30000",
                "token_ids": [1, 2, 3, 4],
                "matched_tokens": 4,
                "reason": "load_imbalance",
                "p2p_bootstrap_room": 123,
                "p2p_source_send": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["transferred_tokens"], 4)
        req = self.tokenizer_manager.p2p_kv_transfer.call_args.args[0]
        self.assertEqual(req.source_url, "http://source:30000")
        self.assertEqual(req.target_url, "http://target:30000")
        self.assertEqual(req.token_ids, [1, 2, 3, 4])
        self.assertEqual(req.p2p_bootstrap_room, 123)
        self.assertTrue(req.p2p_source_send)


if __name__ == "__main__":
    unittest.main()
