"""
CPU-only unit tests for the weight cache daemon's record of its expert layout
(sglang.srt.weight_cache.expert_layout) and its users:

  - the record: the initial layout, begin/commit/abort and versions, the rule
    that only an engine alone on the daemon moves experts, and the fetch_state
    check
  - the daemon's socket handling, including an engine killed while it moves
    experts
  - the startup agreement across ranks, with threads standing in for ranks
  - building the expert location metadata from the daemon's map, which must
    equal --init-expert-location with that map
  - daemons and engines that predate the record

No CUDA, no model, no daemon process.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.eplb.expert_location import (
    ModelConfigForExpertLocation,
    compute_initial_expert_location_metadata,
)
from sglang.srt.runtime_context import get_context, get_parallel, get_resources
from sglang.srt.weight_cache import expert_layout, ipc_loader
from sglang.srt.weight_cache.daemon import CLIENT_CONNECTION_TIMEOUT, WeightCacheDaemon
from sglang.srt.weight_cache.expert_layout import (
    ABSENT,
    ATTACH,
    BEGIN,
    BUSY,
    COMMIT,
    LAYOUT_REFUSED,
    LEGACY,
    OK,
    QUERY,
    REFUSED,
    TRACKED,
    UNTRACKED,
    ExpertLayoutClient,
    ExpertLayoutPresence,
    ExpertLayoutRegistry,
    adopt_weight_cache_expert_layout,
    decide_expert_layout,
    get_expert_layout_session,
    layout_digest,
    mark_expert_weights_shared,
    start_expert_layout_session,
)
from sglang.srt.weight_cache.protocol import CacheConfig, recv_msg, send_msg
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

NUM_LAYERS = 3
NUM_LOGICAL = 8
NUM_PHYSICAL = 10  # two redundant experts over EP 2


def _initial_rows():
    # The trivial layout: physical slot p holds logical expert p % 8.
    return [[p % NUM_LOGICAL for p in range(NUM_PHYSICAL)] for _ in range(NUM_LAYERS)]


def _moved_row(layer_id, shift):
    # Another layout that still holds every logical expert.
    return [(3 * p + layer_id + shift) % NUM_LOGICAL for p in range(NUM_PHYSICAL)]


def _fetch_state(**fields):
    """A fetch_state request; the test daemon matches every engine config."""
    config = CacheConfig(
        model_path="/models/demo",
        model_arch="Qwen3MoeForCausalLM",
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        ep_size=2,
        moe_dp_size=1,
        moe_dp_rank=0,
        moe_ep_rank=0,
        attn_dp_size=1,
        enable_dp_lm_head=False,
        attn_cp_size=1,
        moe_dense_tp_size=None,
        moe_a2a_backend="none",
        quant_method="",
        quant_config_hash="",
        dtype="torch.bfloat16",
        revision="",
        device_capability="",
        torch_version="",
    )
    return {"type": "fetch_state", "config": config.to_dict(), **fields}


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the daemon")
        time.sleep(0.02)


class _DaemonServer:
    """A WeightCacheDaemon's request handling on a real Unix socket, served the
    way serve() serves it, without a model, a ready file or signal handlers."""

    def __init__(self, rows, transport_name="torch_ipc"):
        self.dir = tempfile.mkdtemp(prefix="wcl")
        self.path = os.path.join(self.dir, "d.sock")
        daemon = WeightCacheDaemon.__new__(WeightCacheDaemon)
        daemon.gpu_id = 0
        # Every engine config matches: these tests are about the expert layout.
        daemon.config = mock.Mock()
        daemon.config.matches.return_value = True
        daemon.config.to_dict.return_value = {}
        daemon.state_entries = {"w": {"shape": [1]}}
        daemon.preloaded_weights_bytes = 0
        daemon.transport_backend = mock.Mock()
        daemon.transport_backend.name = transport_name
        daemon.transport_backend.send_fetch_state_response.side_effect = (
            lambda conn, **fields: send_msg(
                conn, {"status": "ok", "transport_backend": transport_name, **fields}
            )
        )
        daemon.expert_layout = ExpertLayoutRegistry(rows)
        daemon.expert_layout_presence = ExpertLayoutPresence(daemon.expert_layout)
        daemon.expert_layout_presence.start()
        self.daemon = daemon
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(self.path)
        self._listener.listen(8)
        self._listener.settimeout(0.1)
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def registry(self) -> ExpertLayoutRegistry:
        return self.daemon.expert_layout

    def layout(self):
        return self.registry.query()["layout"]

    def _serve(self):
        while not self._stopping.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            conn.settimeout(CLIENT_CONNECTION_TIMEOUT)
            self.daemon._serve_connection(conn)

    def request(self, req):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(self.path)
            send_msg(sock, req)
            return recv_msg(sock)

    def close(self):
        self._stopping.set()
        self._thread.join(timeout=5)
        self._listener.close()
        self.daemon.expert_layout_presence.stop()
        shutil.rmtree(self.dir, ignore_errors=True)


class _FakeWorld:
    """all_gather over threads that stand in for the ranks of one engine."""

    def __init__(self, size):
        self._slots = [None] * size
        self._barrier = threading.Barrier(size)

    def gather(self, rank):
        def all_gather(obj):
            self._slots[rank] = obj
            self._barrier.wait(timeout=30)
            gathered = list(self._slots)
            self._barrier.wait(timeout=30)
            return gathered

        return all_gather


def _run_ranks(*fns):
    """Run one function per rank in threads; return (results, errors)."""
    results = [None] * len(fns)
    errors = [None] * len(fns)

    def run(rank):
        try:
            results[rank] = fns[rank]()
        except BaseException as e:
            errors[rank] = e

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(len(fns))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    return results, errors


def _move(path, moves, label="previous-engine"):
    """An engine that attaches alone, moves (layer_id, row) pairs one by one,
    and exits."""
    client = ExpertLayoutClient(path, label)
    _, layout = client.attach()
    version = layout["version"]
    for layer_id, row in moves:
        assert (
            client.request(BEGIN, base_version=version, layer_ids=[layer_id])["status"]
            == OK
        )
        reply = client.request(
            COMMIT, base_version=version, layer_ids=[layer_id], rows=[row]
        )
        assert reply["status"] == OK, reply
        version = reply["version"]
    client.close()


# A separate engine process: attaches, starts a move of layers 1-2, says so,
# and waits to be killed.
_DOOMED_ENGINE = f"""
import pickle, socket, struct, sys, time

def send(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(struct.pack("!I", len(data)) + data)

def recv(sock):
    def exact(n):
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("closed")
            buf += chunk
        return buf
    return pickle.loads(exact(struct.unpack("!I", exact(4))[0]))

presence = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
presence.connect(sys.argv[1])
send(presence, {{"type": "{ATTACH}", "engine_id": "doomed", "label": "doomed-engine"}})
assert recv(presence)["status"] == "{OK}"
move = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
move.connect(sys.argv[1])
send(move, {{"type": "{BEGIN}", "engine_id": "doomed", "base_version": 0, "layer_ids": [1, 2]}})
print(recv(move)["status"], flush=True)
time.sleep(120)
"""


class TestExpertLayoutRegistry(CustomTestCase):
    def _registry(self):
        registry = ExpertLayoutRegistry(_initial_rows())
        registry.attach("a", "engine-a")
        return registry

    def test_initial_layout(self):
        self.assertFalse(ExpertLayoutRegistry.from_metadata(None).tracked)
        metadata = SimpleNamespace(
            physical_to_logical_map_cpu=torch.tensor(_initial_rows())
        )
        registry = ExpertLayoutRegistry.from_metadata(metadata)
        layout = registry.query()["layout"]
        self.assertTrue(layout["tracked"])
        self.assertEqual(layout["version"], 0)
        self.assertEqual(layout["physical_to_logical_map"], _initial_rows())
        self.assertEqual(layout["digest"], layout_digest(_initial_rows()))
        self.assertIsNone(layout["pending"])
        self.assertIsNone(layout["dirty"])
        # A snapshot is a copy.
        layout["physical_to_logical_map"][0][0] = 7
        self.assertEqual(
            registry.query()["layout"]["physical_to_logical_map"], _initial_rows()
        )

    def test_begin_commit_and_versions(self):
        registry = self._registry()
        self.assertEqual(registry.begin("a", 0, [0])["status"], OK)
        pending = registry.query()["layout"]["pending"]
        self.assertEqual(
            (pending["engine"], pending["base_version"], pending["layer_ids"]),
            ("engine-a", 0, [0]),
        )
        expected = _initial_rows()
        expected[0] = _moved_row(0, 1)
        reply = registry.commit("a", 0, [0], [_moved_row(0, 1)])
        self.assertEqual((reply["status"], reply["version"]), (OK, 1))
        self.assertEqual(reply["digest"], layout_digest(expected))

        # The next chunk starts from the new version.
        self.assertEqual(registry.begin("a", 1, [1, 2])["status"], OK)
        reply = registry.commit("a", 1, [1, 2], [_moved_row(1, 1), _moved_row(2, 1)])
        self.assertEqual(reply["version"], 2)
        expected[1:] = [_moved_row(1, 1), _moved_row(2, 1)]
        layout = registry.query()["layout"]
        self.assertEqual(layout["physical_to_logical_map"], expected)
        self.assertIsNone(layout["pending"])

        # An aborted move leaves the version and the rows alone.
        self.assertEqual(registry.begin("a", 2, [0])["status"], OK)
        self.assertEqual(registry.abort("a", 2, [0])["status"], OK)
        layout = registry.query()["layout"]
        self.assertEqual((layout["version"], layout["pending"]), (2, None))
        self.assertEqual(layout["physical_to_logical_map"], expected)

    def test_malformed_moves_are_refused(self):
        registry = self._registry()
        for base_version, layer_ids in (
            (1, [0]),
            (0, []),
            (0, [3]),
            (0, [0, 0]),
            (0, [True]),
            (0, "0"),
        ):
            with self.subTest(base_version=base_version, layer_ids=layer_ids):
                reply = registry.begin("a", base_version, layer_ids)
                self.assertEqual(reply["status"], REFUSED)
        # Nothing is pending yet.
        reply = registry.commit("a", 0, [0], [_moved_row(0, 1)])
        self.assertEqual(reply["status"], REFUSED)

        self.assertEqual(registry.begin("a", 0, [0, 1])["status"], OK)
        # One move at a time; a commit must match it and fit its layers.
        self.assertEqual(registry.begin("a", 0, [2])["status"], REFUSED)
        for layer_ids, rows in (
            ([0], [_moved_row(0, 1)]),
            ([0, 1], [_moved_row(0, 1)]),
            ([0, 1], [_moved_row(0, 1), [0] * 4]),
            ([0, 1], [_moved_row(0, 1), [0.5] * NUM_PHYSICAL]),
        ):
            with self.subTest(layer_ids=layer_ids, rows=rows):
                self.assertEqual(
                    registry.commit("a", 0, layer_ids, rows)["status"], REFUSED
                )
        # Still pending: the weights moved to a layout the daemon could not
        # record, and the engine going away will leave the daemon dirty.
        layout = registry.query()["layout"]
        self.assertIsNotNone(layout["pending"])
        self.assertEqual(layout["version"], 0)

    def test_only_an_engine_alone_on_the_daemon_moves_experts(self):
        registry = self._registry()
        registry.attach("b", "engine-b")
        self.assertEqual(registry.begin("a", 0, [0])["status"], BUSY)
        self.assertEqual(registry.begin("b", 0, [0])["status"], BUSY)
        registry.detach("b")
        self.assertEqual(registry.begin("a", 0, [0])["status"], OK)

        # An engine attaching during a move sees it pending ...
        reply = registry.attach("c", "engine-c")
        self.assertEqual(reply["layout"]["pending"]["layer_ids"], [0])
        self.assertEqual(registry.begin("c", 0, [1])["status"], BUSY)
        # ... the move ends, and then nothing moves while both are attached.
        self.assertEqual(registry.commit("a", 0, [0], [_moved_row(0, 1)])["status"], OK)
        self.assertEqual(registry.begin("a", 1, [1])["status"], BUSY)
        self.assertEqual(registry.begin("c", 1, [1])["status"], BUSY)

        # Takeover: once the first engine is gone, the second one may move.
        registry.detach("a")
        self.assertIsNone(registry.query()["layout"]["dirty"])
        self.assertEqual(registry.begin("c", 1, [1])["status"], OK)
        # An engine that is not attached never moves experts.
        self.assertEqual(registry.begin("d", 1, [2])["status"], REFUSED)

    def test_engine_gone_mid_move_leaves_the_layout_unknown(self):
        registry = self._registry()
        self.assertEqual(registry.begin("a", 0, [1, 2])["status"], OK)
        registry.detach("a")
        layout = registry.query()["layout"]
        self.assertIsNone(layout["pending"])
        self.assertIn("engine-a went away", layout["dirty"])
        self.assertIn("layers 1-2", layout["dirty"])

        registry.attach("b", "engine-b")
        self.assertEqual(registry.begin("b", 0, [0])["status"], REFUSED)
        for fetch in (
            None,
            {"engine_id": "b", "version": 0, "digest": layout["digest"]},
        ):
            refusal = registry.check_fetch(fetch)
            self.assertIn("is unknown", refusal)
            self.assertIn("Restart the weight cache daemons", refusal)

    def test_fetch_check(self):
        registry = self._registry()
        current = {
            "engine_id": "a",
            "version": 0,
            "digest": layout_digest(_initial_rows()),
        }
        self.assertIsNone(registry.check_fetch(current))
        self.assertIn(
            "is not the one its weights are in",
            registry.check_fetch(dict(current, digest="stale")),
        )
        # While a move is in progress, no engine gets the weights.
        registry.begin("a", 0, [0])
        self.assertIn("is moving", registry.check_fetch(current))
        registry.commit("a", 0, [0], [_moved_row(0, 1)])
        # An engine without the expert_layout field cannot follow a moved layout.
        self.assertIn("Upgrade the engine", registry.check_fetch(None))
        # A daemon without EPLB tracks, and so checks, nothing.
        self.assertIsNone(ExpertLayoutRegistry(None).check_fetch(None))

    def test_engines_that_do_not_follow_the_layout_stop_all_moves(self):
        unattached = {
            "engine_id": "stranger",
            "version": 0,
            "digest": layout_digest(_initial_rows()),
        }
        for fetch in (None, unattached):
            with self.subTest(fetch=fetch):
                registry = self._registry()
                self.assertIsNone(registry.check_fetch(fetch))
                reply = registry.begin("a", 0, [0])
                self.assertEqual(reply["status"], REFUSED)
                self.assertIn("until the weight cache daemon restarts", reply["reason"])


class TestDaemonExpertLayoutRequests(CustomTestCase):
    def setUp(self):
        self.server = _DaemonServer(_initial_rows())
        self.addCleanup(self.server.close)

    def _client(self, label="engine"):
        client = ExpertLayoutClient(self.server.path, label)
        self.addCleanup(client.close)
        return client

    def test_attach_move_and_query_over_the_socket(self):
        client = self._client()
        kind, layout = client.attach()
        self.assertEqual(kind, "attached")
        self.assertEqual((layout["version"], layout["attached"]), (0, 1))
        reply = client.request(BEGIN, base_version=0, layer_ids=[2])
        self.assertEqual(reply["status"], OK)
        reply = client.request(
            COMMIT, base_version=0, layer_ids=[2], rows=[_moved_row(2, 5)]
        )
        self.assertEqual((reply["status"], reply["version"]), (OK, 1))
        layout = client.request(QUERY)["layout"]
        self.assertEqual(layout["physical_to_logical_map"][2], _moved_row(2, 5))

    def test_closing_the_attach_connection_detaches(self):
        client = self._client()
        client.attach()
        self.assertEqual(self.server.layout()["attached"], 1)
        client.close()
        _wait_for(lambda: self.server.layout()["attached"] == 0)
        self.assertIsNone(self.server.layout()["dirty"])

    def test_engine_killed_mid_move_leaves_the_daemon_dirty(self):
        # A separate process, killed with SIGKILL while its move is pending:
        # the kernel closing its attach connection is all the daemon sees.
        child = subprocess.Popen(
            [sys.executable, "-c", _DOOMED_ENGINE, self.server.path],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(child.stdout.readline().strip(), OK)
            self.assertIsNotNone(self.server.layout()["pending"])
        finally:
            child.kill()
            child.wait(timeout=10)
            child.stdout.close()
        _wait_for(lambda: self.server.layout()["dirty"] is not None)
        layout = self.server.layout()
        self.assertIn("doomed-engine went away", layout["dirty"])
        self.assertEqual(layout["version"], 0)

        # No engine gets the weights, old or new ...
        for fields in (
            {},
            {
                "expert_layout": {
                    "engine_id": "x",
                    "version": 0,
                    "digest": layout["digest"],
                }
            },
        ):
            reply = self.server.request(_fetch_state(**fields))
            self.assertEqual(reply["status"], LAYOUT_REFUSED)
            self.assertIn("Restart the weight cache daemons", reply["message"])
        self.server.daemon.transport_backend.send_fetch_state_response.assert_not_called()
        # ... and a starting engine stops with the reason.
        with self.assertRaisesRegex(RuntimeError, "doomed-engine went away"):
            start_expert_layout_session(
                socket_path=self.server.path,
                label="next-engine",
                all_gather=lambda obj: [obj],
            )

    def test_layout_check_comes_before_the_transport(self):
        for transport_name in ("torch_ipc", "vmm_fd"):
            with self.subTest(transport=transport_name):
                server = _DaemonServer(_initial_rows(), transport_name=transport_name)
                self.addCleanup(server.close)
                client = ExpertLayoutClient(server.path, "engine")
                self.addCleanup(client.close)
                client.attach()
                send = server.daemon.transport_backend.send_fetch_state_response
                fields = {
                    "engine_id": client.engine_id,
                    "version": 0,
                    "digest": "stale",
                }
                reply = server.request(_fetch_state(expert_layout=fields))
                self.assertEqual(reply["status"], LAYOUT_REFUSED)
                send.assert_not_called()
                fields["digest"] = layout_digest(_initial_rows())
                reply = server.request(_fetch_state(expert_layout=fields))
                self.assertEqual(
                    (reply["status"], reply["transport_backend"]),
                    ("ok", transport_name),
                )
                send.assert_called_once()


class TestStartupAgreement(CustomTestCase):
    @staticmethod
    def _tracked(version=0, digest="d", dirty=None):
        return {"kind": TRACKED, "version": version, "digest": digest, "dirty": dirty}

    @staticmethod
    def _other(kind):
        return {"kind": kind, "version": None, "digest": None, "dirty": None}

    def test_decide(self):
        tracked, other = self._tracked, self._other
        self.assertEqual(decide_expert_layout([tracked(3), tracked(3)]), (True, None))
        for summaries, reason in (
            ([tracked(), other(LEGACY)], "older version"),
            ([other(UNTRACKED), other(UNTRACKED)], "--enable-eplb"),
            ([other(ABSENT), other(ABSENT)], "did not attach"),
            ([tracked(), other(ABSENT)], "not every rank"),
        ):
            with self.subTest(kinds=[s["kind"] for s in summaries]):
                adopt, off_reason = decide_expert_layout(summaries)
                self.assertFalse(adopt)
                self.assertIn(reason, off_reason)
        for summaries, error in (
            ([tracked(3), tracked(4)], "disagree"),
            ([tracked(3, "a"), tracked(3, "b")], "disagree"),
            ([tracked(), tracked(dirty="engine x went away")], "engine x went away"),
            ([tracked(2), other(UNTRACKED)], "not every rank's daemon"),
        ):
            with self.subTest(summaries=summaries):
                with self.assertRaisesRegex(RuntimeError, error):
                    decide_expert_layout(summaries)

    def _start(self, servers, **kwargs):
        world = _FakeWorld(len(servers))
        return _run_ranks(
            *(
                lambda rank=rank: start_expert_layout_session(
                    socket_path=servers[rank].path,
                    label=f"rank{rank}",
                    all_gather=world.gather(rank),
                    **kwargs,
                )
                for rank in range(len(servers))
            )
        )

    def _servers(self, count):
        servers = [_DaemonServer(_initial_rows()) for _ in range(count)]
        for server in servers:
            self.addCleanup(server.close)
        return servers

    def test_every_rank_adopts_the_daemons_layout(self):
        servers = self._servers(2)
        moved = [(layer_id, _moved_row(layer_id, 2)) for layer_id in range(NUM_LAYERS)]
        for server in servers:
            _move(server.path, moved)
        results, errors = self._start(servers)
        self.assertEqual(errors, [None, None])
        for session, rows in results:
            self.addCleanup(session.client.close)
            self.assertEqual(rows, [row for _, row in moved])
            self.assertEqual((session.kind, session.version), (TRACKED, NUM_LAYERS))

    def test_daemons_at_different_versions_stop_every_rank(self):
        servers = self._servers(2)
        # Rank 1's daemon recorded a move that rank 0's did not, as after an
        # engine died between the two commits.
        _move(servers[1].path, [(0, _moved_row(0, 1))])
        _, errors = self._start(servers)
        for error in errors:
            self.assertIsInstance(error, RuntimeError)
            self.assertIn("disagree", str(error))
        # The failed start leaves nothing attached.
        for server in servers:
            _wait_for(lambda server=server: server.layout()["attached"] == 0)

    def test_a_starting_engine_waits_for_a_move_in_progress(self):
        (server,) = self._servers(1)
        mover = ExpertLayoutClient(server.path, "mover")
        self.addCleanup(mover.close)
        mover.attach()
        self.assertEqual(
            mover.request(BEGIN, base_version=0, layer_ids=[0])["status"], OK
        )
        commit = threading.Timer(
            0.5,
            lambda: mover.request(
                COMMIT, base_version=0, layer_ids=[0], rows=[_moved_row(0, 3)]
            ),
        )
        commit.start()
        self.addCleanup(commit.cancel)
        session, rows = start_expert_layout_session(
            socket_path=server.path, label="next-engine", all_gather=lambda obj: [obj]
        )
        self.addCleanup(session.client.close)
        self.assertEqual(session.version, 1)
        self.assertEqual(rows[0], _moved_row(0, 3))

    def test_a_stuck_move_stops_the_start_with_the_reason(self):
        (server,) = self._servers(1)
        mover = ExpertLayoutClient(server.path, "stuck-engine")
        self.addCleanup(mover.close)
        mover.attach()
        mover.request(BEGIN, base_version=0, layer_ids=[1])
        with self.assertRaisesRegex(RuntimeError, "stuck-engine has been moving"):
            start_expert_layout_session(
                socket_path=server.path,
                label="next-engine",
                all_gather=lambda obj: [obj],
                pending_move_timeout=0.3,
            )

    def test_a_rank_that_cannot_read_its_daemon_stops_every_rank(self):
        (server,) = self._servers(1)
        # Rank 1's socket path holds a plain file, which an engine refuses.
        not_a_socket = os.path.join(server.dir, "not-a-socket")
        open(not_a_socket, "w").close()
        world = _FakeWorld(2)
        _, errors = _run_ranks(
            lambda: start_expert_layout_session(
                socket_path=server.path, label="rank0", all_gather=world.gather(0)
            ),
            lambda: start_expert_layout_session(
                socket_path=not_a_socket, label="rank1", all_gather=world.gather(1)
            ),
        )
        for error in errors:
            self.assertIsInstance(error, RuntimeError)
            self.assertIn("Rank 1 could not read", str(error))
        _wait_for(lambda: server.layout()["attached"] == 0)


class TestCoordinatedMoves(CustomTestCase):
    def setUp(self):
        self.servers = [_DaemonServer(_initial_rows()) for _ in range(2)]
        for server in self.servers:
            self.addCleanup(server.close)
        world = _FakeWorld(2)
        results, errors = _run_ranks(
            *(
                lambda rank=rank: start_expert_layout_session(
                    socket_path=self.servers[rank].path,
                    label=f"rank{rank}",
                    all_gather=world.gather(rank),
                )
                for rank in range(2)
            )
        )
        self.assertEqual(errors, [None, None])
        self.sessions = [session for session, _ in results]
        for session in self.sessions:
            self.addCleanup(session.client.close)
            session.digest = layout_digest(_initial_rows())
            session.weights_shared = True

    def _on_every_rank(self, fn):
        results, errors = _run_ranks(*(lambda s=s: fn(s) for s in self.sessions))
        self.assertEqual(errors, [None, None])
        return results

    def test_a_move_is_recorded_by_every_rank_daemon(self):
        new_map = torch.tensor(_initial_rows())
        new_map[1] = torch.tensor(_moved_row(1, 2))
        results = self._on_every_rank(
            lambda s: (s.rebalance_mode(), s.begin([1]), s.commit([1], new_map))
        )
        for (mode, begin_reason, commit_reason), session, server in zip(
            results, self.sessions, self.servers
        ):
            self.assertEqual(mode, ("coordinated", None))
            self.assertIsNone(begin_reason)
            self.assertIsNone(commit_reason)
            self.assertEqual(session.version, 1)
            self.assertEqual(session.digest, layout_digest(new_map.tolist()))
            layout = server.layout()
            self.assertEqual(layout["version"], 1)
            self.assertEqual(layout["physical_to_logical_map"], new_map.tolist())

    def test_another_engine_on_one_rank_postpones_the_move_everywhere(self):
        # A warm standby attaches to rank 1's daemon only.
        standby = ExpertLayoutClient(self.servers[1].path, "standby")
        self.addCleanup(standby.close)
        standby.attach()
        reasons = self._on_every_rank(lambda s: s.begin([0]))
        for reason in reasons:
            self.assertIn("rank 1: 1 other engine(s) map these weights", reason)
        # Rank 0 began, then aborted: nothing is pending, nothing moved.
        for server in self.servers:
            self.assertIsNone(server.layout()["pending"])
            self.assertEqual(server.layout()["version"], 0)
        # Busy only postpones; the move goes ahead once the standby is gone.
        modes = self._on_every_rank(lambda s: s.rebalance_mode())
        self.assertEqual(modes, [("coordinated", None)] * 2)
        standby.close()
        _wait_for(lambda: self.servers[1].layout()["attached"] == 1)
        self.assertEqual(self._on_every_rank(lambda s: s.begin([0])), [None, None])

    def test_a_refusal_turns_rebalancing_off_for_good(self):
        # An engine that does not follow the layout fetched rank 0's weights.
        self.servers[0].request(_fetch_state())
        reasons = self._on_every_rank(lambda s: s.begin([0]))
        for reason in reasons:
            self.assertIn("rank 0: ", reason)
            self.assertIn("until the weight cache daemon restarts", reason)
        modes = self._on_every_rank(lambda s: s.rebalance_mode())
        self.assertEqual([mode for mode, _ in modes], ["off", "off"])

    def test_a_commit_the_daemons_do_not_agree_with_turns_rebalancing_off(self):
        self.assertEqual(self._on_every_rank(lambda s: s.begin([0])), [None, None])
        maps = [torch.tensor(_initial_rows()) for _ in range(2)]
        for engine_map in maps:
            engine_map[0] = torch.tensor(_moved_row(0, 1))
        # Rank 1's engine believes another layer moved too.
        maps[1][2] = torch.tensor(_moved_row(2, 1))
        results, errors = _run_ranks(
            *(
                lambda rank=rank: self.sessions[rank].commit([0], maps[rank])
                for rank in range(2)
            )
        )
        self.assertEqual(errors, [None, None])
        for reason in results:
            self.assertIn("rank 1: the daemon recorded another layout", reason)
        modes = self._on_every_rank(lambda s: s.rebalance_mode())
        self.assertEqual([mode for mode, _ in modes], ["off", "off"])

    def test_rebalance_mode_is_agreed_by_every_rank(self):
        self.sessions[0].weights_shared = False
        modes = self._on_every_rank(lambda s: s.rebalance_mode())
        self.assertEqual(
            modes,
            [
                (
                    "off",
                    "some ranks loaded the weights from disk and some from the daemon",
                )
            ]
            * 2,
        )

    def test_weights_loaded_from_disk_rebalance_without_the_daemons(self):
        for session in self.sessions:
            session.weights_shared = False
        modes = self._on_every_rank(lambda s: s.rebalance_mode())
        self.assertEqual(modes, [("private", None)] * 2)


class TestAdoptTheDaemonLayout(CustomTestCase):
    def setUp(self):
        self.server = _DaemonServer(_initial_rows())
        self.addCleanup(self.server.close)
        for patcher in (
            mock.patch.object(
                ModelConfigForExpertLocation,
                "from_model_config",
                return_value=ModelConfigForExpertLocation(
                    num_layers=NUM_LAYERS, num_logical_experts=NUM_LOGICAL
                ),
            ),
            # One rank.
            mock.patch.object(expert_layout, "_world_all_gather", lambda obj: [obj]),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(expert_layout.set_expert_layout_session, None)

    def _move_all_layers(self):
        moved = [(layer_id, _moved_row(layer_id, 4)) for layer_id in range(NUM_LAYERS)]
        _move(self.server.path, moved)
        _wait_for(lambda: self.server.layout()["attached"] == 0)
        return [row for _, row in moved]

    def _config(self, **overrides):
        fields = dict(
            device="cpu",
            tp_size=2,
            ep_size=2,
            nnodes=1,
            enable_eplb=True,
            ep_num_redundant_experts=NUM_PHYSICAL - NUM_LOGICAL,
            moe_a2a_backend="none",
            ep_dispatch_algorithm="dynamic",
            weight_cache_mode="client",
            weight_cache_socket=self.server.path,
        )
        fields.update(overrides)
        return get_context().override_server_args(**fields)

    def _adopt(self, moe_ep_rank=0):
        with get_parallel().override(tp_rank=moe_ep_rank):
            initial = compute_initial_expert_location_metadata(
                model_config=None, moe_ep_rank=moe_ep_rank
            )
            adopted = adopt_weight_cache_expert_layout(
                initial, model_config=None, moe_ep_rank=moe_ep_rank, gpu_id=0
            )
        session = get_expert_layout_session()
        if session is not None:
            self.addCleanup(session.client.close)
        return initial, adopted, session

    def test_the_daemon_layout_equals_init_expert_location(self):
        moved_rows = self._move_all_layers()
        for a2a, dispatch in (
            ("none", "dynamic"),
            ("deepep", "dynamic"),
            ("deepep", "static"),
        ):
            for moe_ep_rank in (0, 1):
                with self.subTest(a2a=a2a, dispatch=dispatch, moe_ep_rank=moe_ep_rank):
                    with self._config(
                        moe_a2a_backend=a2a, ep_dispatch_algorithm=dispatch
                    ):
                        _, adopted, session = self._adopt(moe_ep_rank)
                    with self._config(
                        moe_a2a_backend=a2a,
                        ep_dispatch_algorithm=dispatch,
                        weight_cache_mode="off",
                        init_expert_location=json.dumps(
                            {"physical_to_logical_map": moved_rows}
                        ),
                    ):
                        direct = compute_initial_expert_location_metadata(
                            model_config=None, moe_ep_rank=moe_ep_rank
                        )
                    self.assertEqual(
                        adopted.physical_to_logical_map.tolist(), moved_rows
                    )
                    for field in (
                        "physical_to_logical_map",
                        "physical_to_logical_map_cpu",
                        "logical_to_all_physical_map",
                        "logical_to_all_physical_map_cpu",
                        "logical_to_all_physical_map_num_valid",
                        "logical_to_rank_dispatch_physical_map",
                    ):
                        expected = getattr(direct, field)
                        if expected is None:
                            self.assertIsNone(getattr(adopted, field), field)
                        else:
                            self.assertTrue(
                                torch.equal(getattr(adopted, field), expected), field
                            )
                    self.assertEqual(adopted.ep_size, direct.ep_size)
                    self.assertEqual(
                        (session.kind, session.version), (TRACKED, NUM_LAYERS)
                    )
                    self.assertEqual(session.digest, layout_digest(moved_rows))

    def test_an_unmoved_layout_keeps_the_engine_metadata(self):
        with self._config():
            initial, adopted, session = self._adopt()
        self.assertIs(adopted, initial)
        self.assertEqual(session.digest, layout_digest(_initial_rows()))
        self.assertEqual(self.server.layout()["attached"], 1)

    def test_an_engine_without_eplb_cannot_follow_a_moved_layout(self):
        self._move_all_layers()
        with self._config(enable_eplb=False, ep_dispatch_algorithm=None):
            with self.assertRaisesRegex(RuntimeError, "--enable-eplb"):
                self._adopt()
        # The engine that failed to start is not left attached.
        _wait_for(lambda: self.server.layout()["attached"] == 0)

    def test_the_layout_shapes_must_match(self):
        self._move_all_layers()
        # 12 physical experts here, 10 in the daemon.
        with self._config(ep_num_redundant_experts=4):
            with self.assertRaisesRegex(RuntimeError, "--ep-num-redundant-experts"):
                self._adopt()
        _wait_for(lambda: self.server.layout()["attached"] == 0)

    def test_without_the_weight_cache_nothing_changes(self):
        with self._config(weight_cache_mode="off"):
            initial, adopted, session = self._adopt()
        self.assertIs(adopted, initial)
        self.assertIsNone(session)
        self.assertEqual(self.server.layout()["attached"], 0)


class TestIpcLoaderSendsItsLayout(CustomTestCase):
    def test_fetch_state_carries_the_layout_and_reports_a_refusal(self):
        server = _DaemonServer(_initial_rows())
        self.addCleanup(server.close)
        session, _ = start_expert_layout_session(
            socket_path=server.path, label="engine", all_gather=lambda obj: [obj]
        )
        self.addCleanup(session.client.close)
        loader = ipc_loader.IpcModelLoader(
            load_config=LoadConfig(load_format=LoadFormat.IPC_CACHE),
            socket_path=server.path,
            weight_cache_mode="client",
        )
        parallel = SimpleNamespace(
            tp_size=1,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            moe_ep_size=1,
            moe_dp_size=1,
            moe_dp_rank=0,
            moe_ep_rank=0,
            dp_size=1,
            attn_dp_size=1,
            enable_dp_lm_head=False,
            attn_cp_size=1,
            moe_dense_tp_size=None,
        )
        model_config = SimpleNamespace(
            model_path="/models/demo",
            hf_config=SimpleNamespace(
                architectures=["Qwen3MoeForCausalLM"], quantization_config=None
            ),
            quantization=None,
            revision=None,
            dtype="torch.bfloat16",
        )
        with (
            mock.patch.object(ipc_loader, "get_parallel", return_value=parallel),
            mock.patch.object(
                ipc_loader,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(moe_a2a_backend="none")
                ),
            ),
            get_resources().override(expert_layout_session=session),
        ):
            session.digest = "stale"
            with self.assertRaisesRegex(
                RuntimeError,
                "refused to serve its weights: the expert layout the engine routes "
                "with .* is not the one its weights are in",
            ):
                loader._fetch_from_cache(model_config, SimpleNamespace(gpu_id=0))
            session.digest = layout_digest(_initial_rows())
            result = loader._fetch_from_cache(model_config, SimpleNamespace(gpu_id=0))
            self.assertEqual(result["status"], "ok")

            mark_expert_weights_shared(True)
            self.assertTrue(session.weights_shared)
            mark_expert_weights_shared(False)
            self.assertFalse(session.weights_shared)


def _old_daemon_handle_connection(conn):
    """How a daemon from before the expert layout record answers: fetch_state
    reads only the config, and other request types get an error."""
    req = recv_msg(conn)
    if req.get("type") == "fetch_state":
        send_msg(
            conn, {"status": "ok", "entries": {}, "transport_backend": "torch_ipc"}
        )
    else:
        send_msg(
            conn,
            {"status": "error", "message": f"Unknown request type: {req.get('type')}"},
        )


class TestCompatibility(CustomTestCase):
    def test_a_new_engine_with_an_old_daemon_keeps_its_layout(self):
        server = _DaemonServer(_initial_rows())
        self.addCleanup(server.close)
        server.daemon._handle_connection = _old_daemon_handle_connection
        session, rows = start_expert_layout_session(
            socket_path=server.path, label="engine", all_gather=lambda obj: [obj]
        )
        self.assertEqual(session.kind, LEGACY)
        self.assertIsNone(rows)
        self.assertFalse(session.client.attached)
        # Its fetch_state carries a field the old daemon does not read.
        reply = server.request(_fetch_state(**session.fetch_fields()))
        self.assertEqual(reply["status"], "ok")
        # It does not rebalance the shared weights, as before.
        session.weights_shared = True
        mode, reason = session.rebalance_mode()
        self.assertEqual(mode, "off")
        self.assertIn("older version", reason)

    def test_an_old_engine_with_a_new_daemon(self):
        server = _DaemonServer(_initial_rows())
        self.addCleanup(server.close)
        # Served as before while every expert is where it started ...
        reply = server.request(_fetch_state())
        self.assertEqual(reply["status"], "ok")
        # ... after which no engine may move experts, since the old one would
        # not follow them.
        client = ExpertLayoutClient(server.path, "engine")
        self.addCleanup(client.close)
        client.attach()
        self.assertEqual(
            client.request(BEGIN, base_version=0, layer_ids=[0])["status"], REFUSED
        )

    def test_an_old_engine_is_refused_moved_experts(self):
        server = _DaemonServer(_initial_rows())
        self.addCleanup(server.close)
        _move(server.path, [(1, _moved_row(1, 1))])
        reply = server.request(_fetch_state())
        self.assertEqual(reply["status"], LAYOUT_REFUSED)
        self.assertIn("Upgrade the engine", reply["message"])


if __name__ == "__main__":
    unittest.main()
