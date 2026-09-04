import inspect
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

mori_module = ModuleType("mori")
mori_cpp = ModuleType("mori.cpp")
mori_io = ModuleType("mori.io")
mori_cpp.TransferStatus = type("TransferStatus", (), {})
for name in (
    "BackendType",
    "EngineDesc",
    "IOEngine",
    "IOEngineConfig",
    "MemoryDesc",
    "MemoryLocationType",
    "PollCqMode",
    "RdmaBackendConfig",
    "StatusCode",
):
    setattr(mori_io, name, type(name, (), {}))
sys.modules.setdefault("mori", mori_module)
sys.modules.setdefault("mori.cpp", mori_cpp)
sys.modules.setdefault("mori.io", mori_io)

from sglang.srt.disaggregation.ascend.conn import AscendKVSender  # noqa: E402
from sglang.srt.disaggregation.common.conn import CommonKVSender  # noqa: E402
from sglang.srt.disaggregation.fake.conn import FakeKVSender  # noqa: E402
from sglang.srt.disaggregation.mooncake.conn import (  # noqa: E402
    MooncakeKVManager,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.mori.conn import (  # noqa: E402
    MoriKVManager,
    MoriKVSender,
)
from sglang.srt.disaggregation.nixl.conn import (  # noqa: E402
    NixlKVManager,
    NixlKVSender,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _assert_sender_tracks_chunk_ownership(manager_type, sender_type, counter_name):
    manager = manager_type.__new__(manager_type)
    setattr(manager, counter_name, {})
    manager._staging_outstanding_lock = threading.Lock()
    chunk = SimpleNamespace(room=11, staging_counted=False)
    sender = sender_type.__new__(sender_type)
    sender.bootstrap_room = chunk.room
    sender.kv_mgr = manager

    manager._count_transfer_chunk(chunk)
    assert not sender.is_transfer_quiesced()

    manager._finish_counted_transfer_chunk(chunk)
    assert sender.is_transfer_quiesced()


def test_every_concrete_sender_implements_quiescence_contract():
    assert inspect.isabstract(CommonKVSender)
    for sender_type in (
        FakeKVSender,
        MooncakeKVSender,
        AscendKVSender,
        NixlKVSender,
        MoriKVSender,
    ):
        assert not inspect.isabstract(sender_type), sender_type.__name__


def test_fake_sender_is_synchronously_quiesced():
    assert FakeKVSender.__new__(FakeKVSender).is_transfer_quiesced()


def test_mooncake_sender_tracks_worker_ownership_until_completion():
    _assert_sender_tracks_chunk_ownership(
        MooncakeKVManager, MooncakeKVSender, "_staging_outstanding"
    )


def test_nixl_sender_tracks_worker_ownership_until_completion():
    _assert_sender_tracks_chunk_ownership(
        NixlKVManager, NixlKVSender, "_staging_outstanding"
    )


def test_nixl_error_waits_for_every_sibling_handle_to_finish():
    manager = NixlKVManager.__new__(NixlKVManager)
    states = {
        "failed": ["ERR", "ERR"],
        "sibling": ["PENDING", "DONE"],
    }
    manager.agent = SimpleNamespace(
        check_xfer_state=Mock(side_effect=lambda handle: states[handle].pop(0))
    )

    error = manager._wait_for_transfer_handles(["failed", "sibling"], room=12)

    assert isinstance(error, RuntimeError)
    assert "room=12" in str(error)
    assert manager.agent.check_xfer_state.call_count == 4


def test_mori_sender_tracks_worker_ownership_until_completion():
    manager = MoriKVManager.__new__(MoriKVManager)
    manager._transfer_outstanding = {}
    manager._transfer_outstanding_lock = threading.Lock()
    chunk = SimpleNamespace(room=13, staging_counted=False)
    sender = MoriKVSender.__new__(MoriKVSender)
    sender.bootstrap_room = chunk.room
    sender.kv_mgr = manager

    manager._count_transfer_chunk(chunk)
    assert not sender.is_transfer_quiesced()

    manager._finish_counted_transfer_chunk(chunk)
    assert sender.is_transfer_quiesced()


def test_mori_abort_after_submission_still_waits_for_completion():
    manager = MoriKVManager.__new__(MoriKVManager)
    manager._should_skip_transfer = Mock(side_effect=[False, True])
    manager._submit_kv_transfer = Mock(return_value=(["status"], ["target"]))
    manager._wait_transfer_completion = Mock(return_value=None)
    chunk = SimpleNamespace(
        room=14,
        wait_event=None,
        prefill_kv_indices=[],
        index_slice=slice(0, 0),
        is_last_chunk=False,
        prefill_aux_index=None,
        state_indices=None,
    )

    manager._process_transfer_chunk(chunk)

    manager._wait_transfer_completion.assert_called_once_with(["status"])
