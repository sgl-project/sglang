import concurrent.futures
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("raises", [False, True])
def test_failed_future_waits_for_running_sibling_before_releasing_ownership(raises):
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.enable_deferred_decode_kv_release = False
    manager._staging_outstanding = {}
    manager._staging_outstanding_lock = threading.Lock()
    chunk = SimpleNamespace(room=7, staging_counted=False)
    manager._count_transfer_chunk(chunk)
    sibling_started = threading.Event()
    release_sibling = threading.Event()
    await_finished = threading.Event()
    results = []
    errors = []

    def blocked_transfer():
        sibling_started.set()
        release_sibling.wait()
        return 0

    def failed_transfer():
        sibling_started.wait()
        if raises:
            raise RuntimeError("layer transfer failed")
        return 17

    def await_and_reconcile(futures):
        try:
            results.append(manager._await_transfer_futures(futures))
        except Exception as error:
            errors.append(error)
        finally:
            manager._finish_counted_transfer_chunk(chunk)
            await_finished.set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(blocked_transfer),
            executor.submit(failed_transfer),
        ]
        waiter = threading.Thread(target=await_and_reconcile, args=(futures,))
        waiter.start()
        try:
            assert sibling_started.wait(timeout=1)
            assert not await_finished.wait(timeout=0.1)
            assert not manager.is_transfer_quiesced(chunk.room)
        finally:
            release_sibling.set()
        waiter.join(timeout=1)

    assert await_finished.is_set()
    assert manager.is_transfer_quiesced(chunk.room)
    if raises:
        assert results == []
        assert [str(error) for error in errors] == ["layer transfer failed"]
    else:
        assert results == [17]
        assert errors == []


def test_worker_exception_releases_ownership_and_continues_polling():
    room = 8
    failed_req = SimpleNamespace(
        is_dummy=False,
        endpoint="127.0.0.1",
        dst_port=9000,
        room=room,
    )
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.enable_trace = False
    manager.request_status = {room: KVPoll.WaitingForInput}
    manager.check_status = Mock(side_effect=RuntimeError("worker failed"))
    manager.bootstrap_port = 8998
    manager.transfer_infos = {room: {0: failed_req}}
    manager.req_to_decode_prefix_len = {room: 0}
    manager.record_failure = Mock()
    manager.update_status = Mock()
    manager.sync_status_to_decode_endpoint = Mock()
    manager.attn_tp_rank = 0
    manager.attn_dp_rank = 0
    manager.pp_size = 1
    manager.attn_cp_size = 1
    manager.pp_rank = 0
    manager.attn_cp_rank = 0
    manager.enable_staging = False
    manager.enable_deferred_decode_kv_release = False
    manager._staging_outstanding = {}
    manager._staging_outstanding_lock = threading.Lock()
    chunk = SimpleNamespace(room=room, staging_counted=False)
    queue = SimpleNamespace(get=Mock(side_effect=[chunk, SystemExit()]))

    with pytest.raises(SystemExit):
        manager.transfer_worker(queue, executor=Mock())

    assert queue.get.call_count == 2
    assert manager.is_transfer_quiesced(room)
    manager.record_failure.assert_called_once()
    manager.update_status.assert_called_once_with(room, KVPoll.Failed)
    manager.sync_status_to_decode_endpoint.assert_called_once()
