import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.common.conn import PrefillServerInfo
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BOOTSTRAP_HOST = "127.0.0.1"
BOOTSTRAP_PORT = 8998
BOOTSTRAP_ADDR = f"{BOOTSTRAP_HOST}:{BOOTSTRAP_PORT}"


def _make_queue(num_reqs, dp_rank_query_interval):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.pending_reqs = [
        SimpleNamespace(
            req=SimpleNamespace(
                rid=f"r{room}",
                bootstrap_room=room,
                bootstrap_host=BOOTSTRAP_HOST,
                bootstrap_port=BOOTSTRAP_PORT,
                disagg_prefill_dp_rank=None,
            ),
            kv_receiver=MagicMock(),
        )
        for room in range(num_reqs)
    ]
    queue._ensure_retry_count = {}
    queue._max_ensure_retries = 15
    queue._ensure_last_attempt_time = {}
    queue._ensure_retry_interval = 1.0
    queue._dp_rank_query_last_attempt_time = {}
    queue._dp_rank_query_interval = dp_rank_query_interval

    # Prefill topology is already cached, and the prefill does not use
    # follow_bootstrap_room, so every request has to resolve its dp_rank
    # through the bootstrap server.
    info = PrefillServerInfo(
        attn_tp_size=8,
        attn_cp_size=1,
        dp_size=8,
        pp_size=1,
        page_size=1,
        kv_cache_dtype="auto",
        follow_bootstrap_room=False,
        enable_dsa_cache_layer_split=False,
    )
    queue.kv_manager = SimpleNamespace(
        prefill_info_table={BOOTSTRAP_ADDR: info},
        try_ensure_parallel_info=lambda bootstrap_addr: True,
    )
    return queue


class TestDecodeDpRankQueryThrottle(CustomTestCase):
    def test_unresolved_reqs_query_bootstrap_once_per_interval(self):
        # The prefill registers a room's dp_rank only once it picks the request
        # up. Until then the query returns nothing, and the scheduling loop must
        # not open one bootstrap connection per cycle (issue #33088).
        queue = _make_queue(num_reqs=8, dp_rank_query_interval=10.0)

        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks",
            return_value={},
        ) as query:
            for _ in range(100):
                queue._resolve_pending_reqs()

        self.assertEqual(query.call_count, 1)
        self.assertEqual(len(queue.pending_reqs), 8)

    def test_first_query_is_not_delayed_and_resolves_reqs(self):
        queue = _make_queue(num_reqs=2, dp_rank_query_interval=10.0)
        decode_reqs = list(queue.pending_reqs)

        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks",
            return_value={"0": 3, "1": 5},
        ) as query:
            queue._resolve_pending_reqs()

        self.assertEqual(query.call_count, 1)
        self.assertEqual(queue.pending_reqs, [])
        decode_reqs[0].kv_receiver.init.assert_called_once_with(3)
        decode_reqs[1].kv_receiver.init.assert_called_once_with(5)


if __name__ == "__main__":
    unittest.main()
