"""CPU-only unit tests for the weight cache `status` request and CLI."""

import os
import shutil
import socket
import tempfile
import textwrap
import threading
import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.weight_cache import status as status_cli
from sglang.srt.weight_cache.daemon import WeightCacheDaemon
from sglang.srt.weight_cache.protocol import (
    CacheConfig,
    iter_daemon_device_uuids,
    recv_msg,
    send_msg,
)
from sglang.srt.weight_cache.transport import TorchIpcTransportBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _cache_config(**overrides) -> CacheConfig:
    base = dict(
        model_path="/models/demo",
        model_arch="LlamaForCausalLM",
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        ep_size=1,
        moe_dp_size=1,
        moe_dp_rank=0,
        moe_ep_rank=0,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        attn_cp_size=1,
        moe_dense_tp_size=None,
        moe_a2a_backend="none",
        quant_method="",
        quant_config_hash="",
        dtype="torch.float16",
        revision="",
        device_capability="8.0",
        torch_version="2.5.1",
    )
    base.update(overrides)
    return CacheConfig(**base)


def _make_daemon(**overrides) -> WeightCacheDaemon:
    """A WeightCacheDaemon with just the attributes the socket handlers touch,
    built without __init__ (which needs a resolved ServerArgs + CUDA)."""
    daemon = object.__new__(WeightCacheDaemon)
    for key, value in {
        "gpu_id": 0,
        "tp_rank": 0,
        "socket_path": "/tmp/sglang_weight_cache_GPU-test.sock",
        "ready_path": "/tmp/sglang_weight_cache_GPU-test.ready",
        "config": _cache_config(),
        "state_entries": {},
        "transport_backend": None,
        "preloaded_weights_bytes": 0,
        "_started_at": 1_000.0,
        "_loaded_at": None,
        "_load_seconds": None,
        "_serve_count": 0,
        "_mismatch_count": 0,
        "_last_served_at": None,
        "_served_client_pids": set(),
        **overrides,
    }.items():
        setattr(daemon, key, value)
    return daemon


def _exchange(daemon, request) -> dict:
    """Drive one _handle_connection call over a socket pair and return the reply."""
    server_sock, client_sock = socket.socketpair(socket.AF_UNIX)
    try:
        send_msg(client_sock, request)
        daemon._handle_connection(server_sock)
        return recv_msg(client_sock)
    finally:
        server_sock.close()
        client_sock.close()


class TestStatusSnapshot(CustomTestCase):
    def test_snapshot_renders_through_cli(self):
        """Snapshot field names are the CLI's contract: a rename on either side
        must fail here, not print zeros in production."""
        now = 1_700_000_000.0
        daemon = _make_daemon(
            state_entries={"w": {}},
            preloaded_weights_bytes=4096,
            _started_at=now - 7200,
            _loaded_at=now - 600,
            _load_seconds=9.5,
            _serve_count=3,
            _mismatch_count=1,
            _last_served_at=now - 30,
        )
        with mock.patch("time.time", return_value=now):
            resp = _exchange(daemon, {"type": "status"})
            row = {
                "label": "GPU-x",
                "socket_path": daemon.socket_path,
                "reachable": True,
            }
            row.update(resp)
            text = status_cli.render_human([row])

        self.assertEqual(
            (resp["status"], resp["pid"], resp["uptime_seconds"]),
            ("ok", os.getpid(), 7200),
        )
        self.assertMultiLineEqual(
            text,
            textwrap.dedent(
                f"""\
                GPU-x  pid {os.getpid()}  up 2h00m
                    model      /models/demo (arch=LlamaForCausalLM)
                    parallel   tp 2/0  pp 1/0  dp 1  ep 1
                    quant      none  dtype torch.float16
                    transport  n/a  tensors 1
                    preloaded  4.00 KiB
                    load       9s  (loaded 10m00s ago)
                    serves     hit 3  mismatch 1  (last 30s ago)
                    clients    live 0 []
                    socket     /tmp/sglang_weight_cache_GPU-test.sock
                """
            ),
        )

        # A live PID stays in the live set; a dead one is pruned from the
        # daemon's own set so it cannot be revived by PID reuse later.
        dead = _make_daemon(_served_client_pids={os.getpid(), 999_999_999})
        live = _exchange(dead, {"type": "status"})
        self.assertEqual(live["live_client_pids"], [os.getpid()])
        self.assertEqual(live["live_client_count"], 1)
        self.assertEqual(dead._served_client_pids, {os.getpid()})


class TestServeCounters(CustomTestCase):
    def test_fetch_state_counts_hits_and_mismatches(self):
        daemon = _make_daemon(config=_cache_config(tp_rank=0))
        mismatch = _exchange(
            daemon,
            {"type": "fetch_state", "config": _cache_config(tp_rank=1).to_dict()},
        )
        self.assertEqual(mismatch["status"], "mismatch")
        self.assertEqual((daemon._serve_count, daemon._mismatch_count), (0, 1))

        backend = TorchIpcTransportBackend()
        entries = backend.prepare_export({"x": (torch.arange(4), True)})
        daemon = _make_daemon(
            config=_cache_config(), transport_backend=backend, state_entries=entries
        )
        req = {"type": "fetch_state", "config": daemon.config.to_dict()}
        self.assertEqual(_exchange(daemon, {**req, "client_pid": 4242})["status"], "ok")
        # A client that does not report its PID (older engine) is still a hit.
        self.assertEqual(_exchange(daemon, req)["status"], "ok")
        self.assertEqual((daemon._serve_count, daemon._mismatch_count), (2, 0))
        self.assertEqual(daemon._served_client_pids, {4242})
        self.assertIsNotNone(daemon._last_served_at)


class TestDaemonDiscovery(CustomTestCase):
    def test_iter_daemon_device_uuids_from_ready_glob(self):
        tmp = tempfile.mkdtemp(prefix="wc_status_test_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with envs.SGLANG_WEIGHT_CACHE_READY_TEMPLATE.override(
            os.path.join(tmp, "wc_{device_uuid}.ready")
        ):
            for uuid in ("GPU-aaaa", "GPU-bbbb"):
                with open(os.path.join(tmp, f"wc_{uuid}.ready"), "w") as f:
                    f.write("pid=123\n")
            # An unrelated file must not be picked up.
            with open(os.path.join(tmp, "unrelated.txt"), "w") as f:
                f.write("x")
            self.assertEqual(iter_daemon_device_uuids(), ["GPU-aaaa", "GPU-bbbb"])


class TestStatusCli(CustomTestCase):
    def test_collect_falls_back_to_ready_file_when_unreachable(self):
        """A wedged daemon (has a .ready file) is marked but still reports its
        PID; a gone one (no trace) is marked with nothing -- neither aborts."""
        tmp = tempfile.mkdtemp(prefix="wc_status_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        ready_path = os.path.join(tmp, "wedged.ready")
        with open(ready_path, "w") as f:
            f.write("pid=4321\n")
            f.write("config={'model_path': '/m'}\n")

        rows = status_cli.collect(
            [
                {
                    "label": "GPU-wedged",
                    "socket_path": os.path.join(tmp, "w.sock"),
                    "ready_path": ready_path,
                },
                {
                    "label": "GPU-gone",
                    "socket_path": os.path.join(tmp, "g.sock"),
                    "ready_path": None,
                },
            ],
            timeout=1.0,
        )

        wedged, gone = rows
        self.assertEqual([r["reachable"] for r in rows], [False, False])
        self.assertEqual(wedged["ready_pid"], 4321)
        self.assertEqual(wedged["ready_config"], {"model_path": "/m"})
        self.assertNotIn("ready_pid", gone)

        text = status_cli.render_human(rows)
        self.assertIn(
            "GPU-wedged  UNREACHABLE  (ready pid 4321, socket not answering", text
        )
        self.assertIn("GPU-gone  UNREACHABLE  (socket not answering", text)

    def test_old_daemon_is_reachable_but_unsupported(self):
        """A daemon that predates the status request answers with an error
        reply; that is not the same as a dead socket and must not exit 4."""
        tmp = tempfile.mkdtemp(prefix="wc_status_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        sock_path = os.path.join(tmp, "old.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(server.close)
        server.bind(sock_path)
        server.listen(1)

        def old_daemon():
            conn, _ = server.accept()
            req = recv_msg(conn)
            send_msg(
                conn,
                {"status": "error", "message": f"Unknown request type: {req['type']}"},
            )
            conn.close()

        threading.Thread(target=old_daemon, daemon=True).start()

        rows = status_cli.collect(
            [{"label": "GPU-old", "socket_path": sock_path, "ready_path": None}],
            timeout=5.0,
        )
        self.assertTrue(rows[0]["reachable"])
        self.assertEqual(rows[0]["error"], "Unknown request type: status")
        self.assertIn("GPU-old  UNSUPPORTED  (", status_cli.render_human(rows))
        with mock.patch.object(status_cli, "collect", return_value=rows):
            self.assertEqual(
                status_cli.main(["--socket", sock_path]), status_cli.EXIT_OK
            )


if __name__ == "__main__":
    unittest.main()
