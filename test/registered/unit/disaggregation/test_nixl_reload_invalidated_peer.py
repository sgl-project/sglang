import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.disaggregation.nixl import conn as nixl_conn
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeAgent:
    def __init__(self, known):
        self.known = set(known)
        self.loaded = []

    def check_remote_metadata(self, agent_name):
        return agent_name in self.known

    def add_remote_agent(self, metadata):
        name = metadata.decode("ascii")
        self.loaded.append(name)
        self.known.add(name)
        return name


def _make_manager(known, peers):
    mgr = NixlKVManager.__new__(NixlKVManager)
    mgr.agent = _FakeAgent(known)
    mgr.disaggregation_mode = DisaggregationMode.PREFILL
    mgr.decode_kv_args_table = {
        name: SimpleNamespace(
            agent_name=name,
            agent_metadata=name.encode("ascii"),
            kv_xfer_segments=["stale"],
        )
        for name in peers
    }
    mgr.prep_handles = {"": "src", **{name: f"old-{name}" for name in peers}}
    mgr.prep_handles_slice_dst = {name: ("old", 0, 0) for name in peers}
    mgr._peer_reload_lock = threading.Lock()
    mgr._peer_reload_times = {}

    def prepare(peer_info):
        mgr.prep_handles[peer_info.agent_name] = f"new-{peer_info.agent_name}"

    mgr._prepare_payload_xfer = mock.Mock(side_effect=prepare)
    return mgr


class TestNixlReloadInvalidatedPeer(unittest.TestCase):
    def test_reloads_invalidated_peer_and_rebuilds_prep_handles(self):
        mgr = _make_manager(known={"healthy"}, peers=["healthy", "dropped"])

        mgr._reload_invalidated_peers({"healthy": object(), "dropped": object()})

        self.assertEqual(mgr.agent.loaded, ["dropped"])
        mgr._prepare_payload_xfer.assert_called_once_with(
            mgr.decode_kv_args_table["dropped"]
        )
        self.assertEqual(mgr.prep_handles["dropped"], "new-dropped")
        self.assertEqual(mgr.prep_handles["healthy"], "old-healthy")
        self.assertEqual(mgr.prep_handles[""], "src")
        self.assertNotIn("dropped", mgr.prep_handles_slice_dst)
        self.assertIsNone(mgr.decode_kv_args_table["dropped"].kv_xfer_segments)
        self.assertEqual(
            mgr.decode_kv_args_table["healthy"].kv_xfer_segments, ["stale"]
        )

    def test_ignores_unregistered_agent(self):
        mgr = _make_manager(known=set(), peers=[])

        mgr._reload_invalidated_peers({"unknown": object()})

        self.assertEqual(mgr.agent.loaded, [])
        mgr._prepare_payload_xfer.assert_not_called()

    def test_throttles_repeated_reloads(self):
        mgr = _make_manager(known=set(), peers=["dropped"])

        with mock.patch.object(nixl_conn.time, "monotonic", return_value=100.0):
            mgr._reload_invalidated_peers({"dropped": object()})
        mgr.agent.known.clear()
        with mock.patch.object(nixl_conn.time, "monotonic", return_value=100.5):
            mgr._reload_invalidated_peers({"dropped": object()})
        self.assertEqual(mgr.agent.loaded, ["dropped"])

        with mock.patch.object(nixl_conn.time, "monotonic", return_value=102.0):
            mgr._reload_invalidated_peers({"dropped": object()})
        self.assertEqual(mgr.agent.loaded, ["dropped", "dropped"])

    def test_reload_failure_does_not_raise(self):
        mgr = _make_manager(known=set(), peers=["dropped"])
        mgr.agent.add_remote_agent = mock.Mock(side_effect=RuntimeError("boom"))

        mgr._reload_invalidated_peers({"dropped": object()})

        mgr._prepare_payload_xfer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
