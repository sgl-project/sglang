"""CPU-only checks for the real sender/worker room-ownership protocol."""

import __future__

import ast
import importlib.util
import queue
import sys
import threading
import time
import types
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


ROOT = Path(__file__).resolve().parents[4]
DISAGG = ROOT / "python/sglang/srt/disaggregation"
spec = importlib.util.spec_from_file_location(
    "_kvserve_lifecycle_helpers", DISAGG / "mooncake/kvserve_pipeline.py"
)
helpers = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {spec.name: helpers}):
    spec.loader.exec_module(helpers)
POLL = types.SimpleNamespace(
    Failed="failed", Success="success", Transferring="transferring"
)


def load_class(path, name, names, globals_):
    tree = ast.parse(path.read_text())
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == name)
    node.body = [
        x for x in node.body if isinstance(x, ast.FunctionDef) and x.name in names
    ]
    for func in node.body:
        func.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    ns = dict(globals_)
    exec(
        compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag),
        ns,
    )
    return ns[name]


class BaseSender:
    def _poll_with_status(self, check_status):
        if self.conclude_state is not None:
            return self.conclude_state
        status = check_status(self.bootstrap_room)
        if status in (POLL.Success, POLL.Failed):
            self.conclude_state = status
        return status


Sender = load_class(
    DISAGG / "mooncake/conn.py",
    "MooncakeKVSender",
    {"is_source_release_safe", "_poll_with_status"},
    {"CommonKVSender": BaseSender, "KVPoll": POLL},
)
CommonSender = load_class(
    DISAGG / "common/conn.py",
    "CommonKVSender",
    {"send"},
    {"BaseKVSender": object, "KVPoll": POLL, "time": time},
)


class Chunk:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.staging_counted = False


Manager = load_class(
    DISAGG / "mooncake/conn.py",
    "MooncakeKVManager",
    {"add_transfer_request", "transfer_worker"},
    {
        "StagingManagerMixin": type("Staging", (), {}),
        "CommonKVManager": type("ManagerBase", (), {}),
        "DisaggregationMode": types.SimpleNamespace(PREFILL="prefill"),
        "TransferKVChunk": Chunk,
        "TraceNullContext": object,
        "KVPoll": POLL,
        "torch": types.SimpleNamespace(Tensor=type("Tensor", (), {})),
        "logger": types.SimpleNamespace(debug=lambda *args: None),
    },
)


class BlockingQueue:
    def __init__(self):
        self.items = queue.Queue()
        self.entered = threading.Event()
        self.resume = threading.Event()

    def put(self, item):
        self.entered.set()
        if not self.resume.wait(2):
            raise RuntimeError("enqueue barrier timed out")
        self.items.put(item)

    def get(self):
        return self.items.get(timeout=2)


class TestSourceOwnership(CustomTestCase):
    def make_sender(self):
        sender = object.__new__(Sender)
        sender.bootstrap_room = 1
        sender.conclude_state = None
        sender._kvserve_source_gate = helpers.SourceRoomGate()
        sender.kv_mgr = types.SimpleNamespace(kvserve_enabled=True, kvserve=None)
        return sender

    def test_failed_poll_closes_gate_after_status_races_reader_admission(self):
        sender = self.make_sender()
        gate = sender._kvserve_source_gate

        def status(_):
            # Reproduce a transfer becoming runnable between the old initial
            # safety query and the subsequent Failed status read.
            self.assertTrue(gate.acquire())
            return POLL.Failed

        self.assertEqual(sender._poll_with_status(status), POLL.Transferring)
        self.assertFalse(gate.acquire())
        gate.release()
        self.assertEqual(sender._poll_with_status(status), POLL.Failed)

    def test_enqueued_owner_survives_failure_before_worker_dequeue(self):
        sender = self.make_sender()
        mgr = object.__new__(Manager)
        mgr.disaggregation_mode = "prefill"
        mgr.kvserve_enabled = True
        mgr.request_status = {1: POLL.Transferring}
        mgr.check_status = lambda room: mgr.request_status[room]
        mgr.transfer_infos = {1: {"host:1": object()}}
        mgr.transfer_queues = [BlockingQueue()]
        mgr.enable_trace = False
        mgr.enable_deferred_decode_kv_release = False
        mgr._staging_outstanding = defaultdict(int)
        sender.kv_mgr = mgr
        gate = sender._kvserve_source_gate
        errors = []

        def enqueue():
            try:
                mgr.add_transfer_request(
                    1,
                    [10],
                    slice(0, 1),
                    True,
                    aux_index=1,
                    trace_ctx=object(),
                    source_gate=gate,
                )
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=enqueue)
        thread.start()
        self.assertTrue(mgr.transfer_queues[0].entered.wait(1))
        mgr.request_status[1] = POLL.Failed
        self.assertEqual(sender._poll_with_status(mgr.check_status), POLL.Transferring)
        mgr.transfer_queues[0].resume.set()
        thread.join(1)
        self.assertEqual(errors, [])
        self.assertFalse(thread.is_alive())
        mgr.transfer_queues[0].items.put(None)
        mgr.transfer_worker(mgr.transfer_queues[0], None)
        self.assertEqual(sender._poll_with_status(mgr.check_status), POLL.Failed)
        self.assertFalse(gate.acquire())

    def test_open_room_is_not_closed_by_nonterminal_poll(self):
        sender = self.make_sender()
        self.assertEqual(
            sender._poll_with_status(lambda _: POLL.Transferring), POLL.Transferring
        )
        self.assertTrue(sender._kvserve_source_gate.acquire())
        sender._kvserve_source_gate.release()

    def test_native_poll_does_not_require_a_gate(self):
        sender = self.make_sender()
        sender.kv_mgr.kvserve_enabled = False
        del sender._kvserve_source_gate
        self.assertEqual(sender._poll_with_status(lambda _: POLL.Failed), POLL.Failed)

    def test_common_sender_passes_its_gate_to_real_enqueue(self):
        sender = object.__new__(CommonSender)
        sender.conclude_state = None
        sender._kvserve_source_gate = helpers.SourceRoomGate()
        sender._prepare_send_indices = lambda ids, states: (
            ids,
            slice(0, 1),
            True,
            False,
        )
        sender._transfer_start_time = None
        sender._early_send_wait_event = None
        sender._record_transfer_indices = lambda *args: None
        sender.bootstrap_room = 1
        sender.aux_index = 9
        sender.trace_ctx = types.SimpleNamespace(copy_for_thread=lambda: object())
        captured = {}
        sender.kv_mgr = types.SimpleNamespace(
            prepare_send_state_indices=lambda states: states,
            add_transfer_request=lambda *args, **kwargs: captured.update(kwargs),
        )
        sender.send([10])
        self.assertIs(captured["source_gate"], sender._kvserve_source_gate)


if __name__ == "__main__":
    unittest.main()
