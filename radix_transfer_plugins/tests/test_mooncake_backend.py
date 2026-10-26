import json
import threading

import pytest
from conftest import USING_FAKE_MOONCAKE, FakeMooncakeStore

from radix_transfer_plugins import available_backends, create_backend
from radix_transfer_plugins.backends.mooncake import (
    MooncakeBackend,
    MooncakeConfig,
    _LocalBufferGate,
    parse_size,
)

pytestmark = pytest.mark.skipif(
    not USING_FAKE_MOONCAKE, reason="unit tests run against the fake mooncake.store"
)

CONFIG = {
    "master_server_address": "10.0.0.1:50051",
    "local_hostname": "10.0.0.2",
    "local_buffer_size": "64mb",
    "key_prefix": "run-1/output-store",
}


def make_backend(**overrides) -> tuple[MooncakeBackend, FakeMooncakeStore]:
    backend = create_backend("mooncake", json.dumps({**CONFIG, **overrides}))
    return backend, FakeMooncakeStore.instances[-1]


def parts(*sizes):
    return {f"f{i}": memoryview(bytes([i]) * size) for i, size in enumerate(sizes)}


def test_client_holds_no_segment():
    _, store = make_backend(protocol="tcp")
    assert store.setup_config == {
        "local_hostname": "10.0.0.2",
        "metadata_server": "P2PHANDSHAKE",
        "global_segment_size": 0,
        "local_buffer_size": 64 * 1024**2,
        "protocol": "tcp",
        "rdma_devices": "",
        "master_server_addr": "10.0.0.1:50051",
    }


def test_one_pinned_put_batch_per_call_with_a_key_per_part():
    backend, store = make_backend(replica_num=2)
    keys = backend.put_many([parts(3, 5), parts(7)])

    assert len(store.put_calls) == 1
    put_keys, config = store.put_calls[0]
    assert put_keys == [f"{keys[0]}/f0", f"{keys[0]}/f1", f"{keys[1]}/f0"]
    assert all(key.startswith("run-1/output-store/") for key in keys)
    assert (config.replica_num, config.with_hard_pin) == (2, True)


def test_a_failed_put_leaves_nothing_behind():
    backend, store = make_backend()
    store.put_script = [
        (-200, 2)
    ]  # Mooncake keeps 2 of the 3 keys, then fails the call
    with pytest.raises(RuntimeError, match="-200"):
        backend.put_many([parts(3, 5), parts(7)])
    assert store.objects == {}
    assert store.remove_calls[-1][1] is True


def test_reads_land_in_memory_the_caller_owns():
    backend, store = make_backend()
    (key,) = backend.put_many([parts(3, 300)])
    (got,) = backend.get_many([key], parts=[[("f1", 300)]])
    assert list(got) == ["f1"]
    assert got["f1"].format == "B" and bytes(got["f1"]) == bytes([1]) * 300
    assert not got["f1"].readonly
    assert store.registered == {}  # registered only while the read runs


def test_reads_take_one_get_into_per_part_into_one_registered_array():
    backend, store = make_backend()
    keys = backend.put_many([parts(3, 5), parts(7)])
    registers = []
    register = store.register_buffer

    def counting_register(ptr, size):
        registers.append(size)
        return register(ptr, size)

    store.register_buffer = counting_register

    got = backend.get_many(keys, parts=[[("f0", 3), ("f1", 5)], [("f0", 7)]])

    assert registers == [15]
    assert sorted(store.get_calls) == sorted(
        [f"{keys[0]}/f0", f"{keys[0]}/f1", f"{keys[1]}/f0"]
    )
    assert [bytes(got[0]["f1"]), bytes(got[1]["f0"])] == [bytes([1]) * 5, bytes(7)]


def test_a_missing_or_resized_part_fails_the_read():
    backend, _ = make_backend()
    (key,) = backend.put_many([parts(3)])
    with pytest.raises(KeyError, match="f9"):
        backend.get_many([key], parts=[[("f9", 3)]])
    with pytest.raises(ValueError, match="has 3 bytes, its ref says 4"):
        backend.get_many([key], parts=[[("f0", 4)]])
    with pytest.raises(ValueError, match="more than 2 bytes, its ref says 2"):
        backend.get_many([key], parts=[[("f0", 2)]])


def test_a_stored_part_that_was_not_read_is_not_reported_missing():
    backend, store = make_backend()
    (key,) = backend.put_many([parts(3, 5)])
    store.get_into = lambda key, ptr, size: -707  # the read outlasted its lease
    with pytest.raises(RuntimeError, match="could not read 2 of the stored parts"):
        backend.get_many([key], parts=[[("f0", 3), ("f1", 5)]])
    assert store.registered == {}


def test_a_failing_read_still_unregisters_its_memory():
    backend, store = make_backend()
    (key,) = backend.put_many([parts(3)])

    def broken(key, ptr, size):
        raise OSError("transfer engine gone")

    store.get_into = broken
    with pytest.raises(OSError):
        backend.get_many([key], parts=[[("f0", 3)]])
    assert store.registered == {}


def test_delete_is_forced_and_idempotent():
    backend, store = make_backend()
    (key,) = backend.put_many([parts(3, 5)])
    backend.delete_many([key], parts=[["f0", "f1"]])
    backend.delete_many([key], parts=[["f0", "f1"]])  # already gone: not an error
    assert store.objects == {}
    assert all(force for _, force in store.remove_calls)


def test_delete_reports_other_failures():
    backend, store = make_backend()
    (key,) = backend.put_many([parts(3)])
    store.batch_remove = lambda keys, force=False: [-706 for _ in keys]
    with pytest.raises(RuntimeError, match="-706"):
        backend.delete_many([key], parts=[["f0"]])


def test_config_is_strict():
    with pytest.raises(ValueError, match="unknown field"):
        make_backend(namespace="default")
    with pytest.raises(ValueError, match="key_prefix"):
        create_backend(
            "mooncake",
            json.dumps({k: v for k, v in CONFIG.items() if k != "key_prefix"}),
        )


def test_unknown_backend_names_list_the_registry():
    assert available_backends() == ["mooncake"]
    with pytest.raises(ValueError, match=r"available: \['mooncake'\]"):
        create_backend("mooncak", None)


def test_sizes():
    assert parse_size(1024) == 1024
    assert parse_size("2gb") == 2 * 1024**3
    assert parse_size("512mb") == 512 * 1024**2
    assert parse_size("4096") == 4096
    with pytest.raises(ValueError):
        parse_size(True)
    assert MooncakeConfig(**CONFIG).protocol == "rdma"


def test_full_local_buffer_waits_for_another_put():
    gate = _LocalBufferGate()
    holder_started, release_holder = threading.Event(), threading.Event()
    statuses = iter([-600, 0])
    undo_calls = []

    def holder():
        def put():
            holder_started.set()
            release_holder.wait(5)
            return 0

        gate.run(put=put, undo=lambda: None)

    thread = threading.Thread(target=holder)
    thread.start()
    holder_started.wait(5)
    threading.Timer(0.2, release_holder.set).start()
    assert gate.run(put=lambda: next(statuses), undo=lambda: undo_calls.append(1)) == 0
    thread.join(5)
    assert undo_calls == [1]


def test_concurrent_puts_that_all_find_the_buffer_full_do_not_wait_on_each_other():
    """Two puts that each fit alone but fail together must both retry and succeed."""
    gate = _LocalBufferGate()
    both_running = threading.Barrier(2)
    results = {}

    def writer(name):
        statuses = iter([-600, 0])

        def put():
            status = next(statuses)
            if status == -600:
                both_running.wait(5)
            return status

        results[name] = gate.run(put=put, undo=lambda: None)

    threads = [threading.Thread(target=writer, args=(n,), daemon=True) for n in "ab"]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert results == {"a": 0, "b": 0}


def test_full_local_buffer_with_nothing_in_flight_fails_at_once():
    gate = _LocalBufferGate()
    assert gate.run(put=lambda: -600, undo=lambda: None) == -600
