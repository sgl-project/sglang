"""Regression tests for elastic prefill topology metadata."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVSender,
    PrefillServerInfo,
)
from sglang.srt.disaggregation.decode import DecodePreallocQueue

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestElasticPrefillTopology(CustomTestCase):
    def test_bootstrap_server_updates_the_advertised_dp_size(self):
        server = CommonKVBootstrapServer.__new__(CommonKVBootstrapServer)
        server.dp_size = 2
        server.max_dp_size = 4
        server.dynamic_dp_size = False
        server.lock = asyncio.Lock()
        request = MagicMock()
        request.json = AsyncMock(return_value={"dp_size": 4})

        response = asyncio.run(server._handle_update_dp_size(request))

        self.assertEqual(response.status, 200)
        self.assertEqual(server.dp_size, 4)
        self.assertTrue(server.dynamic_dp_size)

        request.json = AsyncMock(return_value={"dp_size": 5})
        response = asyncio.run(server._handle_update_dp_size(request))
        self.assertEqual(response.status, 400)

    def test_dynamic_prefill_info_never_uses_cached_modulo_routing(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        bootstrap_addr = "127.0.0.1:8998"
        queue.kv_manager = SimpleNamespace(
            prefill_info_table={
                bootstrap_addr: PrefillServerInfo(
                    attn_tp_size=1,
                    attn_cp_size=1,
                    dp_size=4,
                    pp_size=1,
                    page_size=None,
                    kv_cache_dtype=None,
                    follow_bootstrap_room=True,
                    dynamic_dp_size=True,
                )
            }
        )
        req = SimpleNamespace(
            bootstrap_host="127.0.0.1",
            bootstrap_port=8998,
            bootstrap_room=2,
            disagg_prefill_dp_rank=None,
        )

        self.assertIsNone(queue._resolve_prefill_dp_rank(req))

    def test_dynamic_prefill_sender_registers_its_exact_rank(self):
        for dp_size, dp_rank in ((1, 0), (4, 2)):
            with self.subTest(dp_size=dp_size):
                manager = SimpleNamespace(
                    is_dummy_cp_rank=False,
                    dynamic_dp_size=True,
                    attn_dp_rank=dp_rank,
                    update_status=MagicMock(),
                )
                parallel = SimpleNamespace(
                    dp_size=dp_size, load_balance_method="follow_bootstrap_room"
                )

                with (
                    patch(
                        "sglang.srt.disaggregation.common.conn.get_parallel",
                        return_value=parallel,
                    ),
                    patch.object(
                        CommonKVSender, "_register_prefill_dp_rank"
                    ) as register,
                ):
                    CommonKVSender(
                        manager,
                        bootstrap_addr="127.0.0.1:8998",
                        bootstrap_room=1,
                        dest_tp_ranks=[],
                        pp_rank=0,
                    )

                register.assert_called_once()


if __name__ == "__main__":
    unittest.main()
