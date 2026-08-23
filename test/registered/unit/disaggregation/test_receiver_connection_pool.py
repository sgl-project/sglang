import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _receiver(connection_pool, entries):
    receiver = object.__new__(CommonKVReceiver)
    receiver.kv_mgr = SimpleNamespace(
        connection_pool=connection_pool,
        connection_lock=threading.Lock(),
    )
    receiver._connection_pool_entries = entries
    return receiver


class _FetchingReceiver(CommonKVReceiver):
    def _get_bootstrap_info_from_server(
        self, prefill_dp_rank, prefill_cp_rank, target_tp_rank, target_pp_rank
    ):
        self.fetch_count += 1
        return {"rank_ip": "10.0.0.1", "rank_port": 2001, "pp_rank": target_pp_rank}

    def _register_kv_args(self):
        return True


def _fetching_receiver(connection_pool):
    receiver = object.__new__(_FetchingReceiver)
    receiver.kv_mgr = SimpleNamespace(
        connection_pool=connection_pool,
        connection_lock=threading.Lock(),
        is_mla_backend=False,
    )
    receiver.bootstrap_addr = "prefill:8998"
    receiver.bootstrap_room = 1
    receiver.prefill_dp_rank = 0
    receiver.target_cp_ranks = [0]
    receiver.target_tp_rank = 0
    receiver.target_tp_ranks = [0]
    receiver.target_pp_ranks = [0]
    receiver._connection_pool_entries = {}
    receiver.fetch_count = 0
    return receiver


def test_invalidate_removes_matching_generation():
    stale = [
        {"rank_ip": "10.0.0.1", "rank_port": 1001},
        {"rank_ip": "10.0.0.1", "rank_port": 1002},
    ]
    retained = [{"rank_ip": "10.0.0.2", "rank_port": 2001}]
    receiver = _receiver(
        {"stale": stale, "retained": retained},
        {"stale": stale},
    )

    receiver.invalidate_cached_bootstrap_infos()

    assert receiver.kv_mgr.connection_pool == {"retained": retained}
    assert receiver._connection_pool_entries == {}


def test_invalidate_preserves_concurrent_replacement_generation():
    stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
    replacement = [{"rank_ip": "10.0.0.1", "rank_port": 2001}]
    receiver = _receiver(
        {"key": replacement},
        {"key": stale},
    )

    receiver.invalidate_cached_bootstrap_infos()

    assert receiver.kv_mgr.connection_pool == {"key": replacement}


def test_invalidate_removes_all_matching_cp_entries():
    stale_cp0 = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
    stale_cp1 = [{"rank_ip": "10.0.0.1", "rank_port": 1002}]
    receiver = _receiver(
        {"cp0": stale_cp0, "cp1": stale_cp1},
        {"cp0": stale_cp0, "cp1": stale_cp1},
    )

    receiver.invalidate_cached_bootstrap_infos()

    assert receiver.kv_mgr.connection_pool == {}


def test_next_receiver_refetches_after_invalidation():
    stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
    connection_pool = {"prefill:8998_0_0_0": stale}
    stale_receiver = _receiver(
        connection_pool,
        {"prefill:8998_0_0_0": stale},
    )
    stale_receiver.invalidate_cached_bootstrap_infos()

    receiver = _fetching_receiver(connection_pool)
    receiver._setup_bootstrap_infos()

    assert receiver.fetch_count == 1
    assert receiver.bootstrap_infos[0]["rank_port"] == 2001
    assert (
        connection_pool["prefill:8998_0_0_0"]
        is receiver._connection_pool_entries["prefill:8998_0_0_0"]
    )


@patch("sglang.srt.disaggregation.common.conn.time.time", return_value=3.0)
def test_waiting_timeout_invalidates_cached_generation(_mock_time):
    stale = [{"rank_ip": "10.0.0.1", "rank_port": 1001}]
    receiver = _receiver({"key": stale}, {"key": stale})
    receiver.bootstrap_room = 1
    receiver.bootstrap_infos = stale
    receiver.init_time = 1.0
    receiver.abort_notified = True
    receiver.kv_mgr.waiting_timeout = 1.0
    receiver.kv_mgr.record_failure = Mock()
    receiver.kv_mgr.update_status = Mock()

    assert receiver._check_waiting_timeout() == KVPoll.Failed
    assert receiver.kv_mgr.connection_pool == {}
