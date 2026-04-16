import multiprocessing as mp
import os
import pickle
import shutil
import tempfile
import threading
import time
import unittest

import msgspec
import torch
import zmq

from sglang.srt.utils.common import safe_pickle_load
from sglang.srt.utils.gen_zmq_keys import generate_certificates
from sglang.srt.utils.network import (
    CurveConfig,
    apply_curve_client,
    apply_curve_server,
    config_socket,
    connect_with_curve,
    get_curve_config,
    get_server_public_key,
    get_zmq_socket,
    get_zmq_socket_on_host,
    propagate_curve_keys_to_env,
    set_curve_config,
    set_server_public_key,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

CURVE_AVAILABLE = zmq.has("curve")
GPU_AVAILABLE = torch.cuda.is_available()

try:
    from sglang.multimodal_gen.runtime.server_args import (  # noqa: F401
        ServerArgs as _MMServerArgs,
    )

    MM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MM_AVAILABLE = False

try:
    import mori  # noqa: F401

    MORI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MORI_AVAILABLE = False


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestGenZmqKeys(CustomTestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_generates_key_files(self):
        generate_certificates(self.tmp_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp_dir, "cluster.key")))
        self.assertTrue(
            os.path.isfile(os.path.join(self.tmp_dir, "cluster.key_secret"))
        )

    def test_key_files_are_loadable(self):
        generate_certificates(self.tmp_dir)
        pub, sec = zmq.auth.load_certificate(
            os.path.join(self.tmp_dir, "cluster.key_secret")
        )
        self.assertIsNotNone(pub)
        self.assertIsNotNone(sec)
        self.assertGreater(len(pub), 0)
        self.assertGreater(len(sec), 0)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestCurveConfig(CustomTestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        generate_certificates(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_from_keys_dir(self):
        cfg = CurveConfig.from_keys_dir(self.tmp_dir)
        self.assertIsNotNone(cfg.public_key)
        self.assertIsNotNone(cfg.secret_key)
        self.assertGreater(len(cfg.public_key), 0)
        self.assertGreater(len(cfg.secret_key), 0)

    def test_from_keys_dir_missing_file(self):
        empty_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(Exception):
                CurveConfig.from_keys_dir(empty_dir)
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)


_CURVE_ENV_VARS = (
    "SGLANG_NO_ZMQ_CURVE",
    "SGLANG_ZMQ_CURVE_KEYS_DIR",
    "SGLANG_ZMQ_CURVE_PUBLIC_KEY",
    "SGLANG_ZMQ_CURVE_SECRET_KEY",
    "SGLANG_ZMQ_SERVER_PUBLIC_KEY",
)


def _reset_curve_state():
    """Reset global CurveConfig and server-public-key caches + all CURVE env vars."""
    import sglang.srt.utils.network as net_mod

    saved = {
        "cache": net_mod._curve_config_cache,
        "loaded": net_mod._curve_config_loaded,
        "spk": net_mod._server_public_key,
        "spk_loaded": net_mod._server_public_key_loaded,
        "env": {k: os.environ.get(k) for k in _CURVE_ENV_VARS},
    }
    net_mod._curve_config_cache = None
    net_mod._curve_config_loaded = False
    net_mod._server_public_key = None
    net_mod._server_public_key_loaded = False
    for k in _CURVE_ENV_VARS:
        os.environ.pop(k, None)
    return saved


def _restore_curve_state(saved):
    import sglang.srt.utils.network as net_mod

    net_mod._curve_config_cache = saved["cache"]
    net_mod._curve_config_loaded = saved["loaded"]
    net_mod._server_public_key = saved["spk"]
    net_mod._server_public_key_loaded = saved["spk_loaded"]
    for k, v in saved["env"].items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def _propagate_curve_keys_standalone():
    """Call the production helper used before mp.Process fan-out."""
    return propagate_curve_keys_to_env()


# ---------------------------------------------------------------------------
# Child-process worker functions (must be top-level for mp.Process(spawn))
# ---------------------------------------------------------------------------


def _router_server_worker(endpoint, result_pipe):
    """Bind a ROUTER socket with auto-generated CURVE, wait for a message."""
    try:
        ctx = zmq.Context(io_threads=1)
        server = get_zmq_socket(ctx, zmq.ROUTER, endpoint, bind=True)
        if server.poll(timeout=3000):
            parts = server.recv_multipart(zmq.NOBLOCK)
            identity, payload = parts[0], parts[-1]
            server.send_multipart([identity, b"", pickle.dumps("pong")])
            result_pipe.send({"ok": True, "received": pickle.loads(payload)})
        else:
            result_pipe.send({"ok": False, "error": "timeout"})
        server.close()
        ctx.term()
    except Exception as e:
        result_pipe.send({"ok": False, "error": str(e)})


def _req_client_worker(endpoint, result_pipe):
    """Connect a REQ socket with auto-generated CURVE, send a message."""
    try:
        ctx = zmq.Context(io_threads=1)
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 3000)
        sock.setsockopt(zmq.SNDTIMEO, 3000)
        config_socket(sock, zmq.REQ)
        connect_with_curve(sock, endpoint)
        sock.send(pickle.dumps("ping"))
        reply = pickle.loads(sock.recv())
        result_pipe.send({"ok": True, "reply": reply})
        sock.close()
        ctx.term()
    except zmq.Again:
        result_pipe.send({"ok": False, "error": "timeout"})
    except Exception as e:
        result_pipe.send({"ok": False, "error": str(e)})


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestGetCurveConfigDisabled(CustomTestCase):
    def test_returns_none_when_no_zmq_curve_set(self):
        saved = _reset_curve_state()
        try:
            os.environ["SGLANG_NO_ZMQ_CURVE"] = "1"
            try:
                result = get_curve_config()
                self.assertIsNone(result)
            finally:
                os.environ.pop("SGLANG_NO_ZMQ_CURVE", None)
        finally:
            _restore_curve_state(saved)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestPropagateCurveKeys(CustomTestCase):
    """Verify that exporting the keypair to os.environ before spawning
    ensures all child processes share the same CurveZMQ identity."""

    def setUp(self):
        self.saved = _reset_curve_state()

    def tearDown(self):
        _restore_curve_state(self.saved)

    def test_propagate_sets_env_vars(self):
        """After propagation, the keypair is in os.environ."""
        self.assertIsNone(os.environ.get("SGLANG_ZMQ_CURVE_PUBLIC_KEY"))
        self.assertIsNone(os.environ.get("SGLANG_ZMQ_CURVE_SECRET_KEY"))

        _propagate_curve_keys_standalone()

        pub = os.environ.get("SGLANG_ZMQ_CURVE_PUBLIC_KEY")
        sec = os.environ.get("SGLANG_ZMQ_CURVE_SECRET_KEY")
        self.assertIsNotNone(pub, "Public key should be in os.environ")
        self.assertIsNotNone(sec, "Secret key should be in os.environ")
        self.assertEqual(len(pub), 40, "Z85 public key should be 40 chars")
        self.assertEqual(len(sec), 40, "Z85 secret key should be 40 chars")

    def test_propagate_preserves_user_keys(self):
        """If the user already set CURVE keys via env, propagation doesn't
        overwrite them."""
        user_cfg = CurveConfig.generate()
        user_pub = user_cfg.public_key.decode("ascii")
        user_sec = user_cfg.secret_key.decode("ascii")
        os.environ["SGLANG_ZMQ_CURVE_PUBLIC_KEY"] = user_pub
        os.environ["SGLANG_ZMQ_CURVE_SECRET_KEY"] = user_sec

        _propagate_curve_keys_standalone()

        self.assertEqual(os.environ["SGLANG_ZMQ_CURVE_PUBLIC_KEY"], user_pub)
        self.assertEqual(os.environ["SGLANG_ZMQ_CURVE_SECRET_KEY"], user_sec)

    def test_propagate_noop_when_curve_disabled(self):
        """When SGLANG_NO_ZMQ_CURVE=1, propagation is a no-op."""
        os.environ["SGLANG_NO_ZMQ_CURVE"] = "1"
        _propagate_curve_keys_standalone()

        self.assertIsNone(os.environ.get("SGLANG_ZMQ_CURVE_PUBLIC_KEY"))
        self.assertIsNone(os.environ.get("SGLANG_ZMQ_CURVE_SECRET_KEY"))

    def test_cross_process_with_propagation(self):
        """After propagation, a ROUTER in one spawned process and a REQ
        client in another can communicate — they inherit the same keypair
        from os.environ."""
        from sglang.srt.utils.network import get_open_port

        _propagate_curve_keys_standalone()

        port = get_open_port()
        endpoint = f"tcp://127.0.0.1:{port}"

        server_r, server_w = mp.Pipe(duplex=False)
        client_r, client_w = mp.Pipe(duplex=False)

        ctx = mp.get_context("spawn")
        server_proc = ctx.Process(
            target=_router_server_worker,
            args=(endpoint, server_w),
        )
        client_proc = ctx.Process(
            target=_req_client_worker,
            args=(endpoint, client_w),
        )

        server_proc.start()
        server_w.close()

        time.sleep(0.5)

        client_proc.start()
        client_w.close()

        try:
            server_result = server_r.recv()
            client_result = client_r.recv()
        finally:
            server_proc.join(timeout=10)
            client_proc.join(timeout=10)
            if server_proc.is_alive():
                server_proc.kill()
            if client_proc.is_alive():
                client_proc.kill()
            server_r.close()
            client_r.close()

        self.assertTrue(
            server_result["ok"],
            f"Server should receive the message: {server_result.get('error')}",
        )
        self.assertEqual(server_result["received"], "ping")
        self.assertTrue(
            client_result["ok"],
            f"Client should receive the reply: {client_result.get('error')}",
        )
        self.assertEqual(client_result["reply"], "pong")


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestCurveZMQConnection(CustomTestCase):
    """Verify that CURVE-authenticated PUSH/PULL sockets can communicate,
    and that an unauthenticated client is rejected."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        generate_certificates(self.tmp_dir)
        self.curve = CurveConfig.from_keys_dir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_authenticated_push_pull(self):
        ctx = zmq.Context()
        try:
            server = ctx.socket(zmq.PULL)
            apply_curve_server(server, self.curve)
            port = server.bind_to_random_port("tcp://127.0.0.1")

            client = ctx.socket(zmq.PUSH)
            apply_curve_client(client, self.curve)
            client.connect(f"tcp://127.0.0.1:{port}")

            client.send(b"hello-curve")
            self.assertTrue(server.poll(timeout=3000))
            msg = server.recv()
            self.assertEqual(msg, b"hello-curve")

            client.close()
            server.close()
        finally:
            ctx.term()

    def test_unauthenticated_client_rejected(self):
        ctx = zmq.Context()
        try:
            server = ctx.socket(zmq.PULL)
            apply_curve_server(server, self.curve)
            port = server.bind_to_random_port("tcp://127.0.0.1")

            # Plain (non-CURVE) client trying to connect
            plain_client = ctx.socket(zmq.PUSH)
            plain_client.setsockopt(zmq.LINGER, 0)
            plain_client.setsockopt(zmq.SNDTIMEO, 500)
            plain_client.connect(f"tcp://127.0.0.1:{port}")

            try:
                plain_client.send(b"should-fail")
            except zmq.Again:
                pass

            # The server should NOT receive the message
            self.assertFalse(server.poll(timeout=1000))

            plain_client.close()
            server.close()
        finally:
            ctx.term()

    def test_wrong_key_client_rejected(self):
        """A client with a different keypair cannot talk to the server."""
        wrong_dir = tempfile.mkdtemp()
        try:
            generate_certificates(wrong_dir)
            wrong_curve = CurveConfig.from_keys_dir(wrong_dir)

            ctx = zmq.Context()
            try:
                server = ctx.socket(zmq.PULL)
                apply_curve_server(server, self.curve)
                port = server.bind_to_random_port("tcp://127.0.0.1")

                client = ctx.socket(zmq.PUSH)
                client.setsockopt(zmq.LINGER, 0)
                client.setsockopt(zmq.SNDTIMEO, 500)
                apply_curve_client(client, wrong_curve)
                client.connect(f"tcp://127.0.0.1:{port}")

                try:
                    client.send(b"wrong-key-msg")
                except zmq.Again:
                    pass

                self.assertFalse(server.poll(timeout=1000))

                client.close()
                server.close()
            finally:
                ctx.term()
        finally:
            shutil.rmtree(wrong_dir, ignore_errors=True)


class TestLocalhostBinding(CustomTestCase):
    """Verify that get_zmq_socket_on_host and get_zmq_socket bind to
    localhost by default, not to all interfaces (CVE-2026-3059/3060)."""

    def test_get_zmq_socket_on_host_defaults_to_localhost(self):
        ctx = zmq.Context()
        try:
            port, sock = get_zmq_socket_on_host(ctx, zmq.PULL)
            endpoint = sock.getsockopt_string(zmq.LAST_ENDPOINT)
            self.assertIn("127.0.0.1", endpoint)
            self.assertNotIn("0.0.0.0", endpoint)
            sock.close()
        finally:
            ctx.term()

    def test_get_zmq_socket_on_host_explicit_host(self):
        ctx = zmq.Context()
        try:
            port, sock = get_zmq_socket_on_host(ctx, zmq.PULL, host="127.0.0.1")
            endpoint = sock.getsockopt_string(zmq.LAST_ENDPOINT)
            self.assertIn("127.0.0.1", endpoint)
            sock.close()
        finally:
            ctx.term()

    def test_get_zmq_socket_random_port_binds_all_interfaces(self):
        ctx = zmq.Context()
        try:
            port, sock = get_zmq_socket(ctx, zmq.PULL)
            endpoint = sock.getsockopt_string(zmq.LAST_ENDPOINT)
            self.assertIn("0.0.0.0", endpoint)
            sock.close()
        finally:
            ctx.term()

    def test_localhost_socket_reachable_locally(self):
        ctx = zmq.Context()
        try:
            port, server = get_zmq_socket_on_host(ctx, zmq.PULL)
            client = ctx.socket(zmq.PUSH)
            connect_with_curve(client, f"tcp://127.0.0.1:{port}")
            client.send(b"test-localhost")
            self.assertTrue(server.poll(timeout=3000))
            msg = server.recv()
            self.assertEqual(msg, b"test-localhost")
            client.close()
            server.close()
        finally:
            ctx.term()


def _import_scheduler_client():
    """Import scheduler_client directly from its file path, bypassing the
    heavy sglang.multimodal_gen package __init__ which pulls in imageio."""
    import importlib.util
    import sys

    mod_name = "sglang.multimodal_gen.runtime.scheduler_client"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    from types import ModuleType

    package_stubs = [
        "sglang.multimodal_gen",
        "sglang.multimodal_gen.runtime",
        "sglang.multimodal_gen.runtime.utils",
    ]
    for stub_name in package_stubs:
        if stub_name not in sys.modules:
            stub = ModuleType(stub_name)
            stub.__path__ = []
            stub.__package__ = stub_name
            sys.modules[stub_name] = stub

    logging_mod_name = "sglang.multimodal_gen.runtime.utils.logging_utils"
    if logging_mod_name not in sys.modules:
        log_stub = ModuleType(logging_mod_name)
        import logging as _logging

        log_stub.init_logger = lambda name: _logging.getLogger(name)
        sys.modules[logging_mod_name] = log_stub

    server_args_mod_name = "sglang.multimodal_gen.runtime.server_args"
    if server_args_mod_name not in sys.modules:
        sa_stub = ModuleType(server_args_mod_name)
        sa_stub.ServerArgs = type("ServerArgs", (), {})
        sys.modules[server_args_mod_name] = sa_stub

    sc_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "python",
        "sglang",
        "multimodal_gen",
        "runtime",
        "scheduler_client.py",
    )
    sc_file = os.path.abspath(sc_file)
    spec = importlib.util.spec_from_file_location(mod_name, sc_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestSchedulerClientSecurity(CustomTestCase):
    """Verify that run_zeromq_broker binds to localhost and applies CURVE,
    and that SchedulerClient applies CURVE on its socket (CVE-2026-3059)."""

    @classmethod
    def setUpClass(cls):
        cls.sc_mod = _import_scheduler_client()

    def test_broker_binds_to_localhost(self):
        """run_zeromq_broker must bind to 127.0.0.1, not tcp://*."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_server_args = MagicMock()
        mock_server_args.broker_port = 0

        mock_socket = MagicMock()
        mock_socket.recv = AsyncMock(side_effect=asyncio.CancelledError)
        mock_socket.bind = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.socket.return_value = mock_socket

        with patch("zmq.asyncio.Context", return_value=mock_ctx), patch.object(
            self.sc_mod, "get_curve_config", return_value=None
        ):
            try:
                asyncio.run(
                    asyncio.wait_for(
                        self.sc_mod.run_zeromq_broker(mock_server_args), 0.1
                    )
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        bound_endpoint = mock_socket.bind.call_args[0][0]
        self.assertIn("127.0.0.1", bound_endpoint, "Broker must bind to localhost")
        self.assertNotIn("*", bound_endpoint, "Broker must not bind to tcp://*")

    @unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
    def test_broker_applies_curve_when_enabled(self):
        """run_zeromq_broker must call apply_curve_server when CURVE is configured."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        tmp_dir = tempfile.mkdtemp()
        try:
            generate_certificates(tmp_dir)
            curve = CurveConfig.from_keys_dir(tmp_dir)

            mock_socket = MagicMock()
            mock_socket.recv = AsyncMock(side_effect=asyncio.CancelledError)

            mock_ctx = MagicMock()
            mock_ctx.socket.return_value = mock_socket

            mock_server_args = MagicMock()
            mock_server_args.broker_port = 0

            with patch("zmq.asyncio.Context", return_value=mock_ctx), patch.object(
                self.sc_mod, "get_curve_config", return_value=curve
            ):
                try:
                    asyncio.run(
                        asyncio.wait_for(
                            self.sc_mod.run_zeromq_broker(mock_server_args), 0.1
                        )
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

            self.assertTrue(
                mock_socket.curve_server,
                "Broker socket must have curve_server=True",
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
    def test_broker_rejects_unauthenticated_client(self):
        """A real broker REP socket with CURVE should reject plain clients."""
        tmp_dir = tempfile.mkdtemp()
        try:
            generate_certificates(tmp_dir)
            curve = CurveConfig.from_keys_dir(tmp_dir)

            ctx = zmq.Context()
            server = ctx.socket(zmq.REP)
            apply_curve_server(server, curve)
            port = server.bind_to_random_port("tcp://127.0.0.1")

            plain_client = ctx.socket(zmq.REQ)
            plain_client.setsockopt(zmq.LINGER, 0)
            plain_client.setsockopt(zmq.SNDTIMEO, 500)
            plain_client.setsockopt(zmq.RCVTIMEO, 500)
            plain_client.connect(f"tcp://127.0.0.1:{port}")

            try:
                plain_client.send(b"attack-payload")
            except zmq.Again:
                pass

            self.assertFalse(
                server.poll(timeout=1000),
                "CURVE-protected server must reject plain client",
            )

            plain_client.close()
            server.close()
            ctx.term()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
    def test_scheduler_client_uses_connect_with_curve(self):
        """SchedulerClient.initialize must use connect_with_curve (not manual boilerplate)."""
        from unittest.mock import MagicMock, patch

        mock_server_args = MagicMock()
        mock_server_args.scheduler_endpoint = "tcp://127.0.0.1:9999"

        client = self.sc_mod.SchedulerClient()
        with patch.object(self.sc_mod, "connect_with_curve") as mock_connect:
            client.initialize(mock_server_args)

        mock_connect.assert_called_once()
        call_args = mock_connect.call_args[0]
        self.assertEqual(
            call_args[1],
            "tcp://127.0.0.1:9999",
            "connect_with_curve must be called with the scheduler endpoint",
        )
        client.close()


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestConnectWithCurve(CustomTestCase):
    """Verify connect_with_curve() applies CURVE for TCP and skips for IPC."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        generate_certificates(self.tmp_dir)
        self.curve = CurveConfig.from_keys_dir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_tcp_applies_curve_and_connects(self):
        """connect_with_curve must apply CURVE client opts for TCP endpoints."""
        from unittest.mock import patch

        ctx = zmq.Context()
        try:
            server = ctx.socket(zmq.PULL)
            apply_curve_server(server, self.curve)
            port = server.bind_to_random_port("tcp://127.0.0.1")

            client = ctx.socket(zmq.PUSH)
            client.setsockopt(zmq.LINGER, 0)
            with patch(
                "sglang.srt.utils.network.get_curve_config", return_value=self.curve
            ):
                connect_with_curve(client, f"tcp://127.0.0.1:{port}")

            client.send(b"via-helper")
            self.assertTrue(server.poll(timeout=3000))
            msg = server.recv()
            self.assertEqual(msg, b"via-helper")

            client.close()
            server.close()
        finally:
            ctx.term()

    def test_ipc_skips_curve(self):
        """connect_with_curve must skip CURVE for IPC endpoints."""
        from unittest.mock import MagicMock, patch

        mock_socket = MagicMock()
        with patch(
            "sglang.srt.utils.network.get_curve_config", return_value=self.curve
        ), patch("sglang.srt.utils.network.apply_curve_client") as mock_apply:
            connect_with_curve(mock_socket, "ipc:///tmp/test.sock")

        mock_apply.assert_not_called()
        mock_socket.connect.assert_called_once_with("ipc:///tmp/test.sock")

    def test_explicit_curve_config(self):
        """connect_with_curve must use an explicitly provided CurveConfig."""

        ctx = zmq.Context()
        try:
            server = ctx.socket(zmq.PULL)
            apply_curve_server(server, self.curve)
            port = server.bind_to_random_port("tcp://127.0.0.1")

            client = ctx.socket(zmq.PUSH)
            client.setsockopt(zmq.LINGER, 0)
            connect_with_curve(client, f"tcp://127.0.0.1:{port}", curve=self.curve)

            client.send(b"explicit-curve")
            self.assertTrue(server.poll(timeout=3000))
            msg = server.recv()
            self.assertEqual(msg, b"explicit-curve")

            client.close()
            server.close()
        finally:
            ctx.term()


class TestMMSchedulerUsesSrtGetZmqSocket(CustomTestCase):
    """Verify the multimodal scheduler imports get_zmq_socket from
    srt/utils/network.py (the hardened version with CURVE support)."""

    def test_mm_scheduler_imports_srt_get_zmq_socket(self):
        """The multimodal scheduler must not use its own unhardened get_zmq_socket."""

        mm_sched_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "python",
            "sglang",
            "multimodal_gen",
            "runtime",
            "managers",
            "scheduler.py",
        )
        mm_sched_path = os.path.abspath(mm_sched_path)
        with open(mm_sched_path) as f:
            source = f.read()

        self.assertIn(
            "from sglang.srt.utils.network import get_zmq_socket",
            source,
            "multimodal scheduler must import get_zmq_socket from srt.utils.network",
        )
        self.assertNotIn(
            "from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket",
            source,
            "multimodal scheduler must NOT import get_zmq_socket from multimodal_gen",
        )


class TestSafePickleLoad(CustomTestCase):
    """Verify SafeUnpickler blocks RCE gadgets (CVE-2026-3989)."""

    def test_safe_load_normal_data(self):
        data = {"requests": [{"text": "hello", "id": 1}], "count": 42}
        buf = pickle.dumps(data)
        import io

        result = safe_pickle_load(io.BytesIO(buf))
        self.assertEqual(result, data)

    def test_safe_load_blocks_os_system(self):
        class Exploit:
            def __reduce__(self):
                import os

                return (os.system, ("echo pwned",))

        buf = pickle.dumps(Exploit())
        import io

        with self.assertRaises(RuntimeError):
            safe_pickle_load(io.BytesIO(buf))

    def test_safe_load_blocks_subprocess(self):
        class Exploit:
            def __reduce__(self):
                import subprocess

                return (subprocess.Popen, (["echo", "pwned"],))

        buf = pickle.dumps(Exploit())
        import io

        with self.assertRaises(RuntimeError):
            safe_pickle_load(io.BytesIO(buf))

    def test_safe_load_blocks_eval(self):
        class Exploit:
            def __reduce__(self):
                return (eval, ("__import__('os').system('echo pwned')",))

        buf = pickle.dumps(Exploit())
        import io

        with self.assertRaises(RuntimeError):
            safe_pickle_load(io.BytesIO(buf))


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestCurveConfigGenerate(CustomTestCase):
    """Verify CurveConfig.generate() creates a valid in-memory keypair."""

    def test_generate_returns_valid_keypair(self):
        cfg = CurveConfig.generate()
        self.assertIsNotNone(cfg.public_key)
        self.assertIsNotNone(cfg.secret_key)
        self.assertEqual(len(cfg.public_key), 40)
        self.assertEqual(len(cfg.secret_key), 40)

    def test_generate_produces_unique_keys(self):
        a = CurveConfig.generate()
        b = CurveConfig.generate()
        self.assertNotEqual(a.public_key, b.public_key)
        self.assertNotEqual(a.secret_key, b.secret_key)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestCurveConfigFromRawEnv(CustomTestCase):
    """Verify CurveConfig.from_raw_env() reads Z85 keys from env vars."""

    def test_from_raw_env(self):
        cfg = CurveConfig.generate()
        os.environ["SGLANG_ZMQ_CURVE_PUBLIC_KEY"] = cfg.public_key.decode("ascii")
        os.environ["SGLANG_ZMQ_CURVE_SECRET_KEY"] = cfg.secret_key.decode("ascii")
        try:
            loaded = CurveConfig.from_raw_env()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.public_key, cfg.public_key)
            self.assertEqual(loaded.secret_key, cfg.secret_key)
        finally:
            os.environ.pop("SGLANG_ZMQ_CURVE_PUBLIC_KEY", None)
            os.environ.pop("SGLANG_ZMQ_CURVE_SECRET_KEY", None)

    def test_from_raw_env_returns_none_when_unset(self):
        os.environ.pop("SGLANG_ZMQ_CURVE_PUBLIC_KEY", None)
        os.environ.pop("SGLANG_ZMQ_CURVE_SECRET_KEY", None)
        self.assertIsNone(CurveConfig.from_raw_env())


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestAutoGenViaCurveConfig(CustomTestCase):
    """Verify get_curve_config() auto-generates when no env/flags are set."""

    def test_auto_generates_keypair(self):
        saved = _reset_curve_state()
        os.environ.pop("SGLANG_ZMQ_CURVE_KEYS_DIR", None)
        os.environ.pop("SGLANG_ZMQ_CURVE_PUBLIC_KEY", None)
        os.environ.pop("SGLANG_ZMQ_CURVE_SECRET_KEY", None)
        os.environ.pop("SGLANG_NO_ZMQ_CURVE", None)
        try:
            cfg = get_curve_config()
            self.assertIsNotNone(cfg)
            self.assertEqual(len(cfg.public_key), 40)
            self.assertEqual(len(cfg.secret_key), 40)
        finally:
            _restore_curve_state(saved)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestSetCurveConfig(CustomTestCase):
    """Verify set_curve_config() injects a config that get_curve_config() returns."""

    def test_set_and_get(self):
        saved = _reset_curve_state()
        try:
            injected = CurveConfig.generate()
            set_curve_config(injected)
            result = get_curve_config()
            self.assertIs(result, injected)
        finally:
            _restore_curve_state(saved)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestConnectWithCurveServerKey(CustomTestCase):
    """Verify connect_with_curve passes server_public_key correctly."""

    def test_explicit_server_public_key(self):
        from unittest.mock import MagicMock, patch

        server_cfg = CurveConfig.generate()
        client_cfg = CurveConfig.generate()

        mock_socket = MagicMock()
        with patch(
            "sglang.srt.utils.network.get_curve_config", return_value=client_cfg
        ):
            connect_with_curve(
                mock_socket,
                "tcp://127.0.0.1:9999",
                server_public_key=server_cfg.public_key,
            )

        self.assertEqual(mock_socket.curve_serverkey, server_cfg.public_key)
        self.assertEqual(mock_socket.curve_publickey, client_cfg.public_key)
        mock_socket.connect.assert_called_once_with("tcp://127.0.0.1:9999")

    def test_shared_key_fallback(self):
        from unittest.mock import MagicMock, patch

        cfg = CurveConfig.generate()
        mock_socket = MagicMock()
        with patch("sglang.srt.utils.network.get_curve_config", return_value=cfg):
            connect_with_curve(mock_socket, "tcp://127.0.0.1:9999")

        self.assertEqual(mock_socket.curve_serverkey, cfg.public_key)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestGetZmqSocketOnHostAppliesCurveOnLoopback(CustomTestCase):
    """Verify get_zmq_socket_on_host applies CURVE even on loopback addresses."""

    def test_loopback_gets_curve(self):
        saved = _reset_curve_state()
        os.environ.pop("SGLANG_NO_ZMQ_CURVE", None)
        os.environ.pop("SGLANG_ZMQ_CURVE_KEYS_DIR", None)
        os.environ.pop("SGLANG_ZMQ_CURVE_PUBLIC_KEY", None)
        os.environ.pop("SGLANG_ZMQ_CURVE_SECRET_KEY", None)
        try:
            ctx = zmq.Context()
            try:
                port, sock = get_zmq_socket_on_host(ctx, zmq.PULL, host="127.0.0.1")
                self.assertTrue(
                    sock.curve_server,
                    "CURVE server should be applied even on loopback",
                )
                sock.close()
            finally:
                ctx.term()
        finally:
            _restore_curve_state(saved)


@unittest.skipUnless(
    GPU_AVAILABLE, "GPU required for disaggregation CURVE runtime tests"
)
class TestDisaggregationMetadataRoundTrip(CustomTestCase):
    """Verify curve_public_key round-trips through PrefillRankInfo."""

    def test_prefill_rank_info_with_curve_key(self):
        import dataclasses

        from sglang.srt.disaggregation.common.conn import PrefillRankInfo

        info = PrefillRankInfo(
            rank_ip="192.168.1.1",
            rank_port=5555,
            curve_public_key="A" * 40,
        )
        as_dict = dataclasses.asdict(info)
        self.assertEqual(as_dict["curve_public_key"], "A" * 40)

        restored = PrefillRankInfo(**as_dict)
        self.assertEqual(restored.curve_public_key, "A" * 40)

    def test_prefill_rank_info_without_curve_key(self):
        from sglang.srt.disaggregation.common.conn import PrefillRankInfo

        info = PrefillRankInfo(rank_ip="10.0.0.1", rank_port=6666)
        self.assertIsNone(info.curve_public_key)


@unittest.skipUnless(
    GPU_AVAILABLE, "GPU required for disaggregation CURVE runtime tests"
)
class TestDPBootstrapPayload(CustomTestCase):
    def test_round_trip_with_curve_public_key(self):
        from sglang.srt.managers.data_parallel_controller import (
            BootstrapPayload,
            _bootstrap_decoder,
            _bootstrap_encoder,
        )

        worker_ports = [10000, 10001, 10002]
        payload = BootstrapPayload(worker_ports=worker_ports, curve_public_key="A" * 40)
        encoded = _bootstrap_encoder.encode(payload)
        decoded = _bootstrap_decoder.decode(encoded)

        self.assertEqual(decoded.worker_ports, worker_ports)
        self.assertEqual(decoded.curve_public_key, "A" * 40)

    def test_round_trip_without_curve_public_key(self):
        from sglang.srt.managers.data_parallel_controller import (
            BootstrapPayload,
            _bootstrap_decoder,
            _bootstrap_encoder,
        )

        payload = BootstrapPayload(worker_ports=[4321])
        encoded = _bootstrap_encoder.encode(payload)
        decoded = _bootstrap_decoder.decode(encoded)

        self.assertEqual(decoded.worker_ports, [4321])
        self.assertIsNone(decoded.curve_public_key)

    def test_invalid_json_raises(self):
        from sglang.srt.managers.data_parallel_controller import _bootstrap_decoder

        with self.assertRaises(msgspec.ValidationError):
            _bootstrap_decoder.decode(b"not-json")

    def test_missing_field_raises(self):
        from sglang.srt.managers.data_parallel_controller import _bootstrap_decoder

        with self.assertRaises(msgspec.ValidationError):
            _bootstrap_decoder.decode(b'{"curve_public_key": "x"}')


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
@unittest.skipUnless(
    GPU_AVAILABLE, "GPU required for disaggregation CURVE runtime tests"
)
class TestSinglePhaseBootstrap(CustomTestCase):
    """Integration: bootstrap framing distributes public key only."""

    def test_public_key_exchange(self):
        """Server sends public key via msgspec JSON; client decodes it."""
        from sglang.srt.managers.data_parallel_controller import (
            BootstrapPayload,
            _bootstrap_decoder,
            _bootstrap_encoder,
        )

        server_curve = CurveConfig.generate()
        worker_ports = [10000, 10001, 10002]

        saved = _reset_curve_state()
        set_curve_config(server_curve)

        ctx = zmq.Context()
        results = {}
        error_box = [None]

        server_socket = ctx.socket(zmq.REP)
        config_socket(server_socket, zmq.REP)
        port = server_socket.bind_to_random_port("tcp://127.0.0.1")

        def client_fn():
            try:
                cctx = zmq.Context()
                req = cctx.socket(zmq.REQ)
                config_socket(req, zmq.REQ)
                req.connect(f"tcp://127.0.0.1:{port}")
                req.send(b"1")
                msg = _bootstrap_decoder.decode(req.recv())
                req.close()
                results["worker_ports"] = msg.worker_ports
                results["curve_public_key"] = msg.curve_public_key
                cctx.term()
            except Exception as e:
                error_box[0] = e

        t = threading.Thread(target=client_fn)
        t.start()

        server_socket.recv()
        payload = BootstrapPayload(
            worker_ports=worker_ports,
            curve_public_key=server_curve.public_key.decode("ascii"),
        )
        server_socket.send(_bootstrap_encoder.encode(payload))
        server_socket.close()

        t.join(timeout=10)
        ctx.term()
        _restore_curve_state(saved)

        self.assertIsNone(error_box[0], f"Client thread failed: {error_box[0]}")
        self.assertEqual(
            results["curve_public_key"], server_curve.public_key.decode("ascii")
        )
        self.assertEqual(results["worker_ports"], worker_ports)

    def test_server_public_key_env_propagation(self):
        """set_server_public_key stores to env; get reads from env on miss."""
        saved = _reset_curve_state()
        try:
            key = CurveConfig.generate().public_key
            set_server_public_key(key)
            self.assertEqual(
                os.environ.get("SGLANG_ZMQ_SERVER_PUBLIC_KEY"),
                key.decode("ascii"),
            )
            self.assertEqual(get_server_public_key(), key)

            import sglang.srt.utils.network as net_mod

            net_mod._server_public_key = None
            net_mod._server_public_key_loaded = False

            self.assertEqual(get_server_public_key(), key)
        finally:
            _restore_curve_state(saved)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestRequireServerKey(CustomTestCase):
    """Verify require_server_key fail-fast for cross-node connections."""

    def test_raises_when_no_server_key(self):
        """require_server_key=True raises if no server key is available."""
        saved = _reset_curve_state()
        try:
            cfg = CurveConfig.generate()
            with self.assertRaises(ValueError):
                apply_curve_client(
                    zmq.Context().socket(zmq.PUSH),
                    cfg,
                    require_server_key=True,
                )
        finally:
            _restore_curve_state(saved)

    def test_passes_with_global_server_key(self):
        """require_server_key=True succeeds after set_server_public_key."""
        saved = _reset_curve_state()
        try:
            server_cfg = CurveConfig.generate()
            client_cfg = CurveConfig.generate()
            set_server_public_key(server_cfg.public_key)

            ctx = zmq.Context()
            sock = ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.LINGER, 0)
            apply_curve_client(sock, client_cfg, require_server_key=True)
            sock.close()
            ctx.term()
        finally:
            _restore_curve_state(saved)

    def test_passes_with_explicit_server_key(self):
        """require_server_key=True succeeds with explicit server_public_key."""
        saved = _reset_curve_state()
        try:
            server_cfg = CurveConfig.generate()
            client_cfg = CurveConfig.generate()

            ctx = zmq.Context()
            sock = ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.LINGER, 0)
            apply_curve_client(
                sock,
                client_cfg,
                server_public_key=server_cfg.public_key,
                require_server_key=True,
            )
            sock.close()
            ctx.term()
        finally:
            _restore_curve_state(saved)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
@unittest.skipUnless(
    GPU_AVAILABLE, "GPU required for disaggregation CURVE runtime tests"
)
class TestShmBroadcastCurveKey(CustomTestCase):
    def test_writer_exports_remote_curve_public_key(self):
        from unittest.mock import MagicMock, patch

        from sglang.srt.distributed.device_communicators.shm_broadcast import (
            MessageQueue,
        )

        curve = CurveConfig.generate()
        fake_context = MagicMock()
        fake_socket = MagicMock()
        fake_context.socket.return_value = fake_socket

        with patch(
            "sglang.srt.distributed.device_communicators.shm_broadcast.Context",
            return_value=fake_context,
        ), patch(
            "sglang.srt.distributed.device_communicators.shm_broadcast.get_open_port",
            return_value=54321,
        ), patch(
            "sglang.srt.distributed.device_communicators.shm_broadcast.get_curve_config",
            return_value=curve,
        ), patch(
            "sglang.srt.distributed.device_communicators.shm_broadcast.apply_curve_server"
        ) as mock_apply_curve_server:
            queue = MessageQueue(n_reader=1, n_local_reader=0, connect_ip="127.0.0.1")

        self.assertEqual(
            queue.handle.remote_curve_public_key,
            curve.public_key.decode("ascii"),
        )
        mock_apply_curve_server.assert_called_once_with(fake_socket, curve)

    def test_remote_reader_uses_exported_server_public_key(self):
        from unittest.mock import MagicMock, patch

        from sglang.srt.distributed.device_communicators.shm_broadcast import (
            Handle,
            MessageQueue,
        )

        fake_context = MagicMock()
        fake_socket = MagicMock()
        fake_context.socket.return_value = fake_socket
        handle = Handle(
            connect_ip="127.0.0.1",
            local_reader_ranks=[],
            remote_subscribe_port=65432,
            remote_curve_public_key="A" * 40,
        )

        with patch(
            "sglang.srt.distributed.device_communicators.shm_broadcast.Context",
            return_value=fake_context,
        ), patch(
            "sglang.srt.distributed.device_communicators.shm_broadcast.connect_with_curve"
        ) as mock_connect:
            queue = MessageQueue.create_from_handle(handle, rank=7)

        self.assertTrue(queue._is_remote_reader)
        mock_connect.assert_called_once()
        self.assertEqual(
            mock_connect.call_args.kwargs["server_public_key"],
            b"A" * 40,
        )


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
@unittest.skipUnless(
    GPU_AVAILABLE, "GPU required for disaggregation CURVE runtime tests"
)
class TestDisaggBootstrapCurveKeyRoundTrip(CustomTestCase):
    """Integration: spin up a real CommonKVBootstrapServer, register a prefill
    worker with curve_public_key, and verify the decode-side GET returns it."""

    def test_bootstrap_put_get_curve_key(self):
        import requests as http_requests

        from sglang.srt.utils.network import get_open_port

        port = get_open_port()
        from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer

        server = CommonKVBootstrapServer(host="127.0.0.1", port=port)
        time.sleep(0.5)  # let aiohttp start

        try:
            base_url = f"http://127.0.0.1:{port}"

            test_curve_key = "A" * 40
            payload = {
                "attn_tp_size": 1,
                "attn_tp_rank": 0,
                "attn_cp_size": 1,
                "attn_cp_rank": 0,
                "attn_dp_size": 1,
                "attn_dp_rank": 0,
                "pp_size": 1,
                "pp_rank": 0,
                "system_dp_size": 1,
                "system_dp_rank": 0,
                "rank_ip": "192.168.1.100",
                "rank_port": 5555,
                "page_size": 16,
                "kv_cache_dtype": "auto",
                "load_balance_method": "round_robin",
                "curve_public_key": test_curve_key,
            }
            resp = http_requests.put(f"{base_url}/route", json=payload, timeout=5)
            self.assertEqual(resp.status_code, 200)

            get_resp = http_requests.get(
                f"{base_url}/route",
                params={
                    "prefill_dp_rank": "0",
                    "prefill_cp_rank": "0",
                    "target_tp_rank": "0",
                    "target_pp_rank": "0",
                },
                timeout=5,
            )
            self.assertEqual(get_resp.status_code, 200)
            data = get_resp.json()
            self.assertEqual(data["rank_ip"], "192.168.1.100")
            self.assertEqual(data["rank_port"], 5555)
            self.assertEqual(
                data["curve_public_key"],
                test_curve_key,
                "curve_public_key must survive bootstrap PUT → GET round-trip",
            )
        finally:
            server.close()

    def test_bootstrap_put_get_without_curve_key(self):
        """Backward compat: registrations without curve_public_key still work."""
        import requests as http_requests

        from sglang.srt.utils.network import get_open_port

        port = get_open_port()
        from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer

        server = CommonKVBootstrapServer(host="127.0.0.1", port=port)
        time.sleep(0.5)

        try:
            base_url = f"http://127.0.0.1:{port}"
            payload = {
                "attn_tp_size": 1,
                "attn_tp_rank": 0,
                "attn_cp_size": 1,
                "attn_cp_rank": 0,
                "attn_dp_size": 1,
                "attn_dp_rank": 0,
                "pp_size": 1,
                "pp_rank": 0,
                "system_dp_size": 1,
                "system_dp_rank": 0,
                "rank_ip": "10.0.0.1",
                "rank_port": 6666,
                "page_size": 16,
                "kv_cache_dtype": "auto",
            }
            resp = http_requests.put(f"{base_url}/route", json=payload, timeout=5)
            self.assertEqual(resp.status_code, 200)

            get_resp = http_requests.get(
                f"{base_url}/route",
                params={
                    "prefill_dp_rank": "0",
                    "prefill_cp_rank": "0",
                    "target_tp_rank": "0",
                    "target_pp_rank": "0",
                },
                timeout=5,
            )
            self.assertEqual(get_resp.status_code, 200)
            data = get_resp.json()
            self.assertIsNone(data.get("curve_public_key"))
        finally:
            server.close()


@unittest.skipUnless(
    GPU_AVAILABLE, "GPU required for disaggregation CURVE runtime tests"
)
class TestCommonKVReceiverBootstrapCacheRefresh(CustomTestCase):
    def test_refreshes_cached_curve_key_when_registration_changes(self):
        """After explicit cache invalidation, _setup_bootstrap_infos re-fetches
        and picks up a changed curve_public_key from the bootstrap server."""
        from sglang.srt.disaggregation.common.conn import CommonKVReceiver

        class DummyReceiver(CommonKVReceiver):
            def poll(self):
                return None

        class DummyKVManager:
            def __init__(self):
                self.connection_pool = {}
                self.is_mla_backend = False
                self.failures = []
                self.status_updates = []

            def record_failure(self, room, message):
                self.failures.append((room, message))

            def update_status(self, room, status):
                self.status_updates.append((room, status))

        kv_mgr = DummyKVManager()
        receiver = DummyReceiver.__new__(DummyReceiver)
        receiver.kv_mgr = kv_mgr
        receiver.bootstrap_addr = "127.0.0.1:8000"
        receiver.bootstrap_room = 11
        receiver.prefill_dp_rank = 0
        receiver.target_cp_ranks = [0]
        receiver.target_tp_ranks = [0]
        receiver.target_pp_ranks = [0]
        receiver.target_tp_rank = 0
        receiver.conclude_state = None

        response_box = [
            {
                "rank_ip": "127.0.0.1",
                "rank_port": 9000,
                "curve_public_key": "A" * 40,
            }
        ]
        register_calls = []

        def fake_get_bootstrap_info(*_args):
            return response_box[0].copy()

        def fake_register_kv_args():
            register_calls.append([info.copy() for info in receiver.bootstrap_infos])

        receiver._get_bootstrap_info_from_server = fake_get_bootstrap_info
        receiver._register_kv_args = fake_register_kv_args

        CommonKVReceiver._setup_bootstrap_infos(receiver)
        self.assertEqual(len(register_calls), 1)
        self.assertEqual(register_calls[0][0]["curve_public_key"], "A" * 40)

        CommonKVReceiver._setup_bootstrap_infos(receiver)
        self.assertEqual(len(register_calls), 1, "Cache hit must not re-register")

        response_box[0] = {
            "rank_ip": "127.0.0.1",
            "rank_port": 9000,
            "curve_public_key": "B" * 40,
        }
        kv_mgr.connection_pool.clear()
        CommonKVReceiver._setup_bootstrap_infos(receiver)

        bootstrap_key = "127.0.0.1:8000_0_0_0"
        self.assertEqual(len(register_calls), 2)
        self.assertEqual(register_calls[1][0]["curve_public_key"], "B" * 40)
        self.assertEqual(
            kv_mgr.connection_pool[bootstrap_key][0]["curve_public_key"],
            "B" * 40,
        )
        self.assertEqual(receiver.bootstrap_infos[0]["curve_public_key"], "B" * 40)
        self.assertEqual(kv_mgr.failures, [])


@unittest.skipUnless(
    GPU_AVAILABLE, "GPU required for disaggregation CURVE runtime tests"
)
class TestTransferInfoFromZmq(CustomTestCase):
    """Verify TransferInfo.from_zmq parses the curve_public_key frame."""

    def test_with_curve_public_key(self):
        import numpy as np

        from sglang.srt.disaggregation.mooncake.conn import TransferInfo

        kv_indices = np.array([1, 2, 3], dtype=np.int32)
        curve_key = "B" * 40
        msg = [
            b"42",  # room
            b"192.168.1.1",  # endpoint
            b"5555",  # dst_port
            b"session-abc",  # mooncake_session_id
            kv_indices.tobytes(),  # dst_kv_indices
            b"7",  # dst_aux_index
            b"",  # dst_state_indices (empty)
            b"2",  # required_dst_info_num
            curve_key.encode("ascii"),  # curve_public_key
        ]
        info = TransferInfo.from_zmq(msg)
        self.assertEqual(info.room, 42)
        self.assertEqual(info.endpoint, "192.168.1.1")
        self.assertEqual(info.dst_port, 5555)
        self.assertEqual(info.curve_public_key, curve_key)
        self.assertFalse(info.is_dummy)

    def test_without_curve_public_key(self):
        import numpy as np

        from sglang.srt.disaggregation.mooncake.conn import TransferInfo

        kv_indices = np.array([10], dtype=np.int32)
        msg = [
            b"1",
            b"10.0.0.2",
            b"6000",
            b"session-xyz",
            kv_indices.tobytes(),
            b"0",
            b"",
            b"1",
        ]
        info = TransferInfo.from_zmq(msg)
        self.assertIsNone(info.curve_public_key)

    def test_dummy_transfer_info(self):
        from sglang.srt.disaggregation.mooncake.conn import TransferInfo

        msg = [
            b"99",
            b"10.0.0.3",
            b"7000",
            b"session-dummy",
            b"",  # empty kv_indices → dummy
            b"",  # empty aux_index → dummy
            b"",
            b"1",
            b"C" * 40,  # curve key still present
        ]
        info = TransferInfo.from_zmq(msg)
        self.assertTrue(info.is_dummy)
        self.assertEqual(info.curve_public_key, "C" * 40)


@unittest.skipUnless(
    GPU_AVAILABLE, "GPU required for disaggregation CURVE runtime tests"
)
class TestKVArgsRegisterInfoFromZmq(CustomTestCase):
    """Verify KVArgsRegisterInfo.from_zmq parses the curve_public_key frame."""

    def test_with_curve_public_key(self):
        import struct

        from sglang.srt.disaggregation.mooncake.conn import KVArgsRegisterInfo

        curve_key = "D" * 40
        msg = [
            b"None",  # room
            b"10.0.0.5",  # endpoint
            b"8000",  # dst_port
            b"session-reg",  # mooncake_session_id
            struct.pack("Q", 0xDEAD),  # dst_kv_ptrs
            struct.pack("Q", 0xBEEF),  # dst_aux_ptrs
            struct.pack("Q", 0xCAFE),  # dst_state_data_ptrs
            b"0",  # dst_tp_rank
            b"1",  # dst_attn_tp_size
            b"128",  # dst_kv_item_len
            b"",  # dst_state_item_lens
            b"",  # dst_state_dim_per_tensor
            b"0",  # enable_hisparse
            curve_key.encode("ascii"),  # curve_public_key
            b"",  # staging_base_ptr (none)
            b"",  # staging_total_size (none)
        ]
        info = KVArgsRegisterInfo.from_zmq(msg)
        self.assertEqual(info.room, "None")
        self.assertEqual(info.endpoint, "10.0.0.5")
        self.assertEqual(info.curve_public_key, curve_key)

    def test_without_curve_public_key(self):
        import struct

        from sglang.srt.disaggregation.mooncake.conn import KVArgsRegisterInfo

        msg = [
            b"None",
            b"10.0.0.6",
            b"9000",
            b"session-noreg",
            struct.pack("Q", 0x1234),
            struct.pack("Q", 0x5678),
            struct.pack("Q", 0x9ABC),
            b"0",
            b"2",
            b"256",
            b"",
            b"",
        ]
        info = KVArgsRegisterInfo.from_zmq(msg)
        self.assertIsNone(info.curve_public_key)


@unittest.skipUnless(MORI_AVAILABLE, "mori dependencies not installed")
@unittest.skipUnless(
    GPU_AVAILABLE, "GPU required for disaggregation CURVE runtime tests"
)
class TestMoriTransferInfoFromZmq(CustomTestCase):
    def test_with_curve_public_key(self):
        import numpy as np

        from sglang.srt.disaggregation.mori.conn import TransferInfo

        kv_indices = np.array([4, 5], dtype=np.int32)
        msg = [
            b"8",
            b"10.0.0.1",
            b"7001",
            b"engine-xyz",
            kv_indices.tobytes(),
            b"3",
            b"",
            b"2",
            b"E" * 40,
        ]
        info = TransferInfo.from_zmq(msg)

        self.assertEqual(info.room, 8)
        self.assertEqual(info.endpoint, "10.0.0.1")
        self.assertEqual(info.dst_port, 7001)
        self.assertEqual(info.engine_key, "engine-xyz")
        self.assertEqual(info.curve_public_key, "E" * 40)

    def test_without_curve_public_key(self):
        import numpy as np

        from sglang.srt.disaggregation.mori.conn import TransferInfo

        kv_indices = np.array([9], dtype=np.int32)
        msg = [
            b"1",
            b"10.0.0.2",
            b"7002",
            b"engine-no-key",
            kv_indices.tobytes(),
            b"0",
            b"",
            b"1",
        ]
        info = TransferInfo.from_zmq(msg)

        self.assertIsNone(info.curve_public_key)


@unittest.skipUnless(MORI_AVAILABLE, "mori dependencies not installed")
@unittest.skipUnless(
    GPU_AVAILABLE, "GPU required for disaggregation CURVE runtime tests"
)
class TestMoriKVArgsRegisterInfoFromZmq(CustomTestCase):
    def test_with_curve_public_key(self):
        from unittest.mock import MagicMock, patch

        from sglang.srt.disaggregation.mori.conn import KVArgsRegisterInfo

        msg = [
            b"unused-room",
            b"10.0.0.3",
            b"7003",
            b"packed-engine-desc",
            b"packed-kv-mem-descs",
            b"packed-aux-mem-descs",
            b"packed-state-mem-descs",
            b"0",
            b"2",
            b"1",
            b"128",
            b"F" * 40,
        ]

        fake_engine_desc = MagicMock(key="engine-key")
        with patch(
            "sglang.srt.disaggregation.mori.conn.EngineDesc.unpack",
            return_value=fake_engine_desc,
        ), patch(
            "sglang.srt.disaggregation.mori.conn._unpack_mem_desc_list",
            return_value=[],
        ):
            info = KVArgsRegisterInfo.from_zmq(msg)

        self.assertEqual(info.endpoint, "10.0.0.3")
        self.assertEqual(info.dst_port, 7003)
        self.assertEqual(info.curve_public_key, "F" * 40)

    def test_without_curve_public_key(self):
        from unittest.mock import MagicMock, patch

        from sglang.srt.disaggregation.mori.conn import KVArgsRegisterInfo

        msg = [
            b"unused-room",
            b"10.0.0.4",
            b"7004",
            b"packed-engine-desc",
            b"packed-kv-mem-descs",
            b"packed-aux-mem-descs",
            b"packed-state-mem-descs",
            b"1",
            b"4",
            b"3",
            b"256",
        ]

        fake_engine_desc = MagicMock(key="engine-key")
        with patch(
            "sglang.srt.disaggregation.mori.conn.EngineDesc.unpack",
            return_value=fake_engine_desc,
        ), patch(
            "sglang.srt.disaggregation.mori.conn._unpack_mem_desc_list",
            return_value=[],
        ):
            info = KVArgsRegisterInfo.from_zmq(msg)

        self.assertIsNone(info.curve_public_key)


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestDisaggZmqCurveHandshake(CustomTestCase):
    """Integration: simulate prefill PULL + decode PUSH with different CURVE
    keypairs using server_public_key routing (asymmetric CURVE)."""

    def test_prefill_decode_zmq_with_curve(self):
        prefill_cfg = CurveConfig.generate()
        decode_cfg = CurveConfig.generate()

        ctx = zmq.Context()
        try:
            prefill_pull = ctx.socket(zmq.PULL)
            apply_curve_server(prefill_pull, prefill_cfg)
            port = prefill_pull.bind_to_random_port("tcp://127.0.0.1")

            decode_push = ctx.socket(zmq.PUSH)
            decode_push.setsockopt(zmq.LINGER, 0)
            apply_curve_client(
                decode_push, decode_cfg, server_public_key=prefill_cfg.public_key
            )
            decode_push.connect(f"tcp://127.0.0.1:{port}")

            decode_push.send_multipart([b"42", b"test-kv-data"])
            self.assertTrue(prefill_pull.poll(timeout=3000))
            msg = prefill_pull.recv_multipart()
            self.assertEqual(msg, [b"42", b"test-kv-data"])

            decode_push.close()
            prefill_pull.close()
        finally:
            ctx.term()

    def test_connect_with_curve_server_public_key(self):
        """connect_with_curve with explicit server_public_key works across
        different instance keypairs."""
        prefill_cfg = CurveConfig.generate()
        decode_cfg = CurveConfig.generate()

        saved = _reset_curve_state()
        try:
            set_curve_config(decode_cfg)

            ctx = zmq.Context()
            try:
                server = ctx.socket(zmq.PULL)
                apply_curve_server(server, prefill_cfg)
                port = server.bind_to_random_port("tcp://127.0.0.1")

                client = ctx.socket(zmq.PUSH)
                client.setsockopt(zmq.LINGER, 0)
                connect_with_curve(
                    client,
                    f"tcp://127.0.0.1:{port}",
                    server_public_key=prefill_cfg.public_key,
                )

                client.send(b"cross-key-msg")
                self.assertTrue(server.poll(timeout=3000))
                self.assertEqual(server.recv(), b"cross-key-msg")

                client.close()
                server.close()
            finally:
                ctx.term()
        finally:
            _restore_curve_state(saved)


@unittest.skipUnless(
    GPU_AVAILABLE, "GPU required for disaggregation CURVE runtime tests"
)
class TestBootstrapCacheFastPath(CustomTestCase):
    """Verify that _setup_bootstrap_infos skips HTTP on warm cache hits."""

    def test_cache_hit_skips_http_and_register(self):
        from sglang.srt.disaggregation.common.conn import CommonKVReceiver

        class DummyReceiver(CommonKVReceiver):
            def poll(self):
                return None

        class DummyKVManager:
            def __init__(self):
                self.connection_pool = {}
                self.is_mla_backend = False
                self.failures = []
                self.status_updates = []

            def record_failure(self, room, message):
                self.failures.append((room, message))

            def update_status(self, room, status):
                self.status_updates.append((room, status))

        kv_mgr = DummyKVManager()
        receiver = DummyReceiver.__new__(DummyReceiver)
        receiver.kv_mgr = kv_mgr
        receiver.bootstrap_addr = "127.0.0.1:8000"
        receiver.bootstrap_room = 11
        receiver.prefill_dp_rank = 0
        receiver.target_cp_ranks = [0]
        receiver.target_tp_ranks = [0]
        receiver.target_pp_ranks = [0]
        receiver.target_tp_rank = 0
        receiver.conclude_state = None

        fetch_count = [0]
        register_calls = []

        def fake_get_bootstrap_info(*_args):
            fetch_count[0] += 1
            return {
                "rank_ip": "127.0.0.1",
                "rank_port": 9000,
                "curve_public_key": "A" * 40,
            }

        def fake_register_kv_args():
            register_calls.append(True)

        receiver._get_bootstrap_info_from_server = fake_get_bootstrap_info
        receiver._register_kv_args = fake_register_kv_args

        CommonKVReceiver._setup_bootstrap_infos(receiver)
        self.assertEqual(fetch_count[0], 1, "First call must fetch from HTTP")
        self.assertEqual(len(register_calls), 1, "First call must register")

        fetch_count[0] = 0
        register_calls.clear()
        CommonKVReceiver._setup_bootstrap_infos(receiver)
        self.assertEqual(fetch_count[0], 0, "Second call must skip HTTP (cache hit)")
        self.assertEqual(len(register_calls), 0, "Second call must skip registration")

        bootstrap_key = "127.0.0.1:8000_0_0_0"
        self.assertIn(bootstrap_key, kv_mgr.connection_pool)
        self.assertEqual(
            kv_mgr.connection_pool[bootstrap_key][0]["curve_public_key"],
            "A" * 40,
        )

    def test_cache_invalidation_forces_refetch(self):
        """After clearing connection_pool, the next call must fetch again."""
        from sglang.srt.disaggregation.common.conn import CommonKVReceiver

        class DummyReceiver(CommonKVReceiver):
            def poll(self):
                return None

        class DummyKVManager:
            def __init__(self):
                self.connection_pool = {}
                self.is_mla_backend = False

            def record_failure(self, room, message):
                pass

            def update_status(self, room, status):
                pass

        kv_mgr = DummyKVManager()
        receiver = DummyReceiver.__new__(DummyReceiver)
        receiver.kv_mgr = kv_mgr
        receiver.bootstrap_addr = "127.0.0.1:8000"
        receiver.bootstrap_room = 11
        receiver.prefill_dp_rank = 0
        receiver.target_cp_ranks = [0]
        receiver.target_tp_ranks = [0]
        receiver.target_pp_ranks = [0]
        receiver.target_tp_rank = 0
        receiver.conclude_state = None

        fetch_count = [0]

        def fake_get_bootstrap_info(*_args):
            fetch_count[0] += 1
            return {"rank_ip": "127.0.0.1", "rank_port": 9000}

        receiver._get_bootstrap_info_from_server = fake_get_bootstrap_info
        receiver._register_kv_args = lambda: None

        CommonKVReceiver._setup_bootstrap_infos(receiver)
        self.assertEqual(fetch_count[0], 1)

        kv_mgr.connection_pool.clear()
        fetch_count[0] = 0
        CommonKVReceiver._setup_bootstrap_infos(receiver)
        self.assertEqual(
            fetch_count[0], 1, "After invalidation, must fetch from HTTP again"
        )


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
@unittest.skipUnless(
    GPU_AVAILABLE, "GPU required – expert_backup_client import chain needs sgl_kernel"
)
class TestExpertBackupClientPeerKeys(CustomTestCase):
    """Verify ExpertBackupClient gathers (ip, curve_public_key) tuples
    and passes peer server_public_key into connect_with_curve."""

    def test_gathers_ip_and_curve_key_tuples(self):
        from unittest.mock import MagicMock, patch

        server_curve = CurveConfig.generate()
        peer_curve = CurveConfig.generate()
        local_ip = "10.0.0.1"
        peer_ip = "10.0.0.2"
        local_pub = server_curve.public_key.decode("ascii")
        peer_pub = peer_curve.public_key.decode("ascii")

        world_size = 2

        def fake_all_gather(output_list, local_val, group=None):
            output_list[0] = (local_ip, local_pub)
            output_list[1] = (peer_ip, peer_pub)

        mock_world_group = MagicMock()

        # ExpertBackupClient.__init__ starts _receive_loop on a thread; mocked
        # recv_pyobj() returns MagicMock and breaks max(int, response.buffer_size).
        # This test only checks connect_with_curve args, so skip starting threads.
        def _thread_start_noop(self, *args, **kwargs):
            return None

        with patch(
            "sglang.srt.elastic_ep.expert_backup_client.get_local_ip_auto",
            return_value=local_ip,
        ), patch(
            "sglang.srt.elastic_ep.expert_backup_client.get_curve_config",
            return_value=server_curve,
        ), patch(
            "sglang.srt.elastic_ep.expert_backup_client.get_world_size",
            return_value=world_size,
        ), patch(
            "sglang.srt.elastic_ep.expert_backup_client.get_world_group",
            return_value=mock_world_group,
        ), patch(
            "torch.distributed.all_gather_object", side_effect=fake_all_gather
        ), patch(
            "sglang.srt.elastic_ep.expert_backup_client.connect_with_curve"
        ) as mock_connect, patch(
            "zmq.Context"
        ) as mock_ctx_cls, patch.object(
            threading.Thread, "start", _thread_start_noop
        ):
            mock_ctx = MagicMock()
            mock_ctx_cls.return_value = mock_ctx
            mock_socket = MagicMock()
            mock_ctx.socket.return_value = mock_socket

            mock_server_args = MagicMock()
            mock_server_args.nnodes = 2
            mock_server_args.node_rank = 0
            mock_model_runner = MagicMock()
            mock_model_runner.moe_ep_size = 2
            mock_model_runner.model_config = MagicMock()
            mock_model_runner.moe_ep_rank = 0

            from sglang.srt.elastic_ep.expert_backup_client import ExpertBackupClient

            _orig_init = ExpertBackupClient.__init__

            def patched_init(self_inner, sa, mr):
                self_inner.server_args = sa
                self_inner.engine_num = sa.nnodes
                self_inner.engine_rank = sa.node_rank
                self_inner.recv_list = [None] * self_inner.engine_num
                self_inner.ready_sockets = [None] * self_inner.engine_num
                self_inner.model_runner = mr
                self_inner.moe_ep_size = mr.moe_ep_size
                self_inner.model_config = mr.model_config
                self_inner.moe_ep_rank = mr.moe_ep_rank
                self_inner.dram_map_list = [None] * self_inner.engine_num
                self_inner.session_id_list = [None] * self_inner.engine_num
                self_inner.transfer_engine = None
                self_inner.gpu_buffer = None
                self_inner.buffer_size = 0
                self_inner.use_backup = False
                _orig_init(self_inner, sa, mr)

            with patch.object(ExpertBackupClient, "__init__", patched_init):
                client = ExpertBackupClient(mock_server_args, mock_model_runner)

        connect_calls = mock_connect.call_args_list
        self.assertTrue(
            len(connect_calls) >= 4,
            f"Expected >= 4 connect_with_curve calls, got {len(connect_calls)}",
        )

        for c in connect_calls:
            self.assertIn(
                "server_public_key",
                c.kwargs,
                "connect_with_curve must receive server_public_key kwarg",
            )


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestSchedulerClientPassesServerKey(CustomTestCase):
    """Verify SchedulerClient.initialize passes server_public_key."""

    @classmethod
    def setUpClass(cls):
        cls.sc_mod = _import_scheduler_client()

    def test_initialize_passes_server_public_key(self):
        from unittest.mock import MagicMock, patch

        server_key = CurveConfig.generate().public_key

        mock_server_args = MagicMock()
        mock_server_args.scheduler_endpoint = "tcp://10.0.0.5:9999"

        client = self.sc_mod.SchedulerClient()

        with patch.object(
            self.sc_mod, "connect_with_curve"
        ) as mock_connect, patch.object(
            self.sc_mod, "get_server_public_key", return_value=server_key
        ):
            client.initialize(mock_server_args)

        mock_connect.assert_called_once()
        _, kwargs = mock_connect.call_args
        self.assertEqual(
            kwargs.get("server_public_key"),
            server_key,
            "SchedulerClient must pass server_public_key to connect_with_curve",
        )
        client.close()

    def test_initialize_passes_none_when_no_server_key(self):
        from unittest.mock import MagicMock, patch

        mock_server_args = MagicMock()
        mock_server_args.scheduler_endpoint = "tcp://127.0.0.1:9999"

        client = self.sc_mod.SchedulerClient()

        with patch.object(
            self.sc_mod, "connect_with_curve"
        ) as mock_connect, patch.object(
            self.sc_mod, "get_server_public_key", return_value=None
        ):
            client.initialize(mock_server_args)

        mock_connect.assert_called_once()
        _, kwargs = mock_connect.call_args
        self.assertIsNone(
            kwargs.get("server_public_key"),
            "When no server key is configured, server_public_key should be None",
        )
        client.close()


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
class TestKvEventsPublisherServerKey(CustomTestCase):
    """Verify ZmqEventPublisher passes server_public_key in connect mode."""

    def test_connect_mode_passes_server_public_key(self):
        from unittest.mock import MagicMock, patch

        server_key = CurveConfig.generate().public_key

        with patch(
            "sglang.srt.disaggregation.kv_events.zmq.Context"
        ) as mock_ctx_cls, patch(
            "sglang.srt.disaggregation.kv_events.ZmqEventPublisher.offset_endpoint_port",
            side_effect=lambda ep, _rank: ep,
        ), patch(
            "sglang.srt.utils.network.get_curve_config",
            return_value=CurveConfig.generate(),
        ), patch(
            "sglang.srt.utils.network.connect_with_curve"
        ) as mock_connect:
            mock_ctx = MagicMock()
            mock_ctx_cls.instance.return_value = mock_ctx
            mock_socket = MagicMock()
            mock_ctx.socket.return_value = mock_socket

            from sglang.srt.disaggregation.kv_events import ZmqEventPublisher

            pub = ZmqEventPublisher(
                attn_dp_rank=0,
                endpoint="tcp://10.0.0.1:5557",
                server_public_key=server_key,
            )
            pub.shutdown()

        mock_connect.assert_called_once()
        _, kwargs = mock_connect.call_args
        self.assertEqual(
            kwargs.get("server_public_key"),
            server_key,
            "ZmqEventPublisher in connect mode must pass server_public_key",
        )

    def test_bind_mode_does_not_use_server_key(self):
        from unittest.mock import MagicMock, patch

        curve = CurveConfig.generate()

        with patch(
            "sglang.srt.disaggregation.kv_events.zmq.Context"
        ) as mock_ctx_cls, patch(
            "sglang.srt.disaggregation.kv_events.ZmqEventPublisher.offset_endpoint_port",
            side_effect=lambda ep, _rank: ep,
        ), patch(
            "sglang.srt.utils.network.get_curve_config",
            return_value=curve,
        ), patch(
            "sglang.srt.utils.network.apply_curve_server"
        ) as mock_apply_server, patch(
            "sglang.srt.utils.network.connect_with_curve"
        ) as mock_connect:
            mock_ctx = MagicMock()
            mock_ctx_cls.instance.return_value = mock_ctx
            mock_socket = MagicMock()
            mock_ctx.socket.return_value = mock_socket

            from sglang.srt.disaggregation.kv_events import ZmqEventPublisher

            pub = ZmqEventPublisher(
                attn_dp_rank=0,
                endpoint="tcp://*:5557",
                server_public_key=CurveConfig.generate().public_key,
            )
            pub.shutdown()

        mock_connect.assert_not_called()
        mock_apply_server.assert_called_once()


@unittest.skipUnless(CURVE_AVAILABLE, "libzmq built without CURVE support")
@unittest.skipUnless(MM_AVAILABLE, "multimodal_gen dependencies not installed")
class TestMultimodalGenServerArgsCurveFlags(CustomTestCase):
    """multimodal_gen ServerArgs supports --no-zmq-curve and
    --zmq-curve-keys-dir for parity with srt ServerArgs."""

    @classmethod
    def setUpClass(cls):
        import sys

        to_remove = [k for k in sys.modules if k.startswith("sglang.multimodal_gen")]
        for k in to_remove:
            del sys.modules[k]

    def setUp(self):
        self.saved = _reset_curve_state()

    def tearDown(self):
        _restore_curve_state(self.saved)

    def test_no_zmq_curve_flag_sets_env(self):
        """--no-zmq-curve sets SGLANG_NO_ZMQ_CURVE in os.environ."""
        from unittest.mock import patch

        from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
            QwenImagePipelineConfig,
        )
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        with patch.object(
            PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
        ):
            ServerArgs.from_dict({"model_path": "/fake/model", "no_zmq_curve": True})

        self.assertEqual(
            os.environ.get("SGLANG_NO_ZMQ_CURVE"),
            "1",
            "--no-zmq-curve should set SGLANG_NO_ZMQ_CURVE=1",
        )

    def test_zmq_curve_keys_dir_flag_sets_env(self):
        """--zmq-curve-keys-dir sets SGLANG_ZMQ_CURVE_KEYS_DIR in os.environ."""
        from unittest.mock import patch

        from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
            QwenImagePipelineConfig,
        )
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        keys_dir = tempfile.mkdtemp()
        try:
            generate_certificates(keys_dir)

            with patch.object(
                PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
            ):
                ServerArgs.from_dict(
                    {"model_path": "/fake/model", "zmq_curve_keys_dir": keys_dir}
                )

            self.assertEqual(
                os.environ.get("SGLANG_ZMQ_CURVE_KEYS_DIR"),
                keys_dir,
                "--zmq-curve-keys-dir should propagate to env var",
            )
        finally:
            shutil.rmtree(keys_dir, ignore_errors=True)

    def test_zmq_curve_keys_dir_missing_file_raises(self):
        """--zmq-curve-keys-dir with a directory that has no cluster.key_secret
        should raise ValueError."""
        from unittest.mock import patch

        from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
            QwenImagePipelineConfig,
        )
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        empty_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(ValueError):
                with patch.object(
                    PipelineConfig,
                    "from_kwargs",
                    return_value=QwenImagePipelineConfig(),
                ):
                    ServerArgs.from_dict(
                        {
                            "model_path": "/fake/model",
                            "zmq_curve_keys_dir": empty_dir,
                        }
                    )
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)

    def test_cli_parser_accepts_curve_flags(self):
        """The argparse parser should accept --no-zmq-curve and
        --zmq-curve-keys-dir without errors."""
        from sglang.multimodal_gen.runtime.server_args import ServerArgs
        from sglang.multimodal_gen.utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser()
        ServerArgs.add_cli_args(parser)

        args = parser.parse_args(["--model-path", "/fake", "--no-zmq-curve"])
        self.assertTrue(args.no_zmq_curve)

        args2 = parser.parse_args(
            ["--model-path", "/fake", "--zmq-curve-keys-dir", "/some/dir"]
        )
        self.assertEqual(args2.zmq_curve_keys_dir, "/some/dir")


if __name__ == "__main__":
    unittest.main()
