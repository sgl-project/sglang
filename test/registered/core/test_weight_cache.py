"""Unit tests for the weight cache daemon and IPC loader.

Tests protocol layer (CacheConfig, socket messaging), module imports,
LoadFormat dispatch, cross-process CUDA IPC, and config mismatch detection.

Requires GPU for CUDA IPC tests.
"""

import multiprocessing as mp
import os
import socket
import time
import unittest

import torch

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.loader import get_model_loader
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_cache.daemon import WeightCacheDaemon
from sglang.srt.weight_cache.ipc_loader import IpcModelLoader
from sglang.srt.weight_cache.protocol import (
    CacheConfig,
    get_ready_path,
    get_socket_path,
    hash_quant_config,
    recv_msg,
    send_msg,
)

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-a", runner_config="1-gpu-small")


class TestCacheConfig(unittest.TestCase):
    """Test CacheConfig matching logic."""

    def test_matching_configs(self):
        c1 = CacheConfig(
            model_path="/a", model_arch="Llama", tp_size=4, tp_rank=0,
            dp_size=1, quant_method="fp8", quant_config_hash="abc",
            dtype="torch.float16",
        )
        c2 = CacheConfig(
            model_path="/a", model_arch="Llama", tp_size=4, tp_rank=0,
            dp_size=1, quant_method="fp8", quant_config_hash="abc",
            dtype="torch.float16",
        )
        self.assertTrue(c1.matches(c2))

    def test_mismatch_tp_size(self):
        c1 = CacheConfig(
            model_path="/a", model_arch="Llama", tp_size=4, tp_rank=0,
            dp_size=1, quant_method="fp8", quant_config_hash="abc",
            dtype="torch.float16",
        )
        c2 = CacheConfig(
            model_path="/a", model_arch="Llama", tp_size=2, tp_rank=0,
            dp_size=1, quant_method="fp8", quant_config_hash="abc",
            dtype="torch.float16",
        )
        self.assertFalse(c1.matches(c2))

    def test_mismatch_model_path(self):
        c1 = CacheConfig(
            model_path="/a", model_arch="Llama", tp_size=4, tp_rank=0,
            dp_size=1, quant_method="fp8", quant_config_hash="abc",
            dtype="torch.float16",
        )
        c2 = CacheConfig(
            model_path="/b", model_arch="Llama", tp_size=4, tp_rank=0,
            dp_size=1, quant_method="fp8", quant_config_hash="abc",
            dtype="torch.float16",
        )
        self.assertFalse(c1.matches(c2))

    def test_mismatch_quant(self):
        c1 = CacheConfig(
            model_path="/a", model_arch="Llama", tp_size=4, tp_rank=0,
            dp_size=1, quant_method="fp8", quant_config_hash="abc",
            dtype="torch.float16",
        )
        c2 = CacheConfig(
            model_path="/a", model_arch="Llama", tp_size=4, tp_rank=0,
            dp_size=1, quant_method="gptq", quant_config_hash="abc",
            dtype="torch.float16",
        )
        self.assertFalse(c1.matches(c2))

    def test_to_dict_roundtrip(self):
        c = CacheConfig(
            model_path="/a", model_arch="Llama", tp_size=4, tp_rank=0,
            dp_size=1, quant_method="fp8", quant_config_hash="abc",
            dtype="torch.float16",
        )
        c2 = CacheConfig.from_dict(c.to_dict())
        self.assertTrue(c.matches(c2))


class TestSocketProtocol(unittest.TestCase):
    """Test send_msg/recv_msg over Unix sockets."""

    def test_socketpair_roundtrip(self):
        s1, s2 = socket.socketpair()
        try:
            send_msg(s1, {"type": "test", "data": 42})
            result = recv_msg(s2)
            self.assertEqual(result, {"type": "test", "data": 42})
        finally:
            s1.close()
            s2.close()

    def test_large_payload(self):
        s1, s2 = socket.socketpair()
        try:
            large_data = {"entries": {f"key_{i}": "x" * 1000 for i in range(100)}}
            send_msg(s1, large_data)
            result = recv_msg(s2)
            self.assertEqual(len(result["entries"]), 100)
        finally:
            s1.close()
            s2.close()


class TestHashQuantConfig(unittest.TestCase):
    """Test quantization config hashing."""

    def test_none_config(self):
        self.assertEqual(hash_quant_config(None), "")

    def test_dict_config(self):
        h1 = hash_quant_config({"method": "fp8", "block_size": 128})
        h2 = hash_quant_config({"method": "fp8", "block_size": 128})
        self.assertEqual(h1, h2)

    def test_different_config(self):
        h1 = hash_quant_config({"method": "fp8"})
        h2 = hash_quant_config({"method": "gptq"})
        self.assertNotEqual(h1, h2)


class TestLoadFormatAndDispatch(unittest.TestCase):
    """Test LoadFormat.IPC_CACHE and get_model_loader dispatch."""

    def test_ipc_cache_enum(self):
        self.assertEqual(LoadFormat.IPC_CACHE.value, "ipc_cache")

    def test_get_model_loader_returns_ipc_loader(self):
        config = LoadConfig(
            load_format=LoadFormat.IPC_CACHE,
            weight_cache_mode="copy",
            weight_cache_socket="/tmp/test.sock",
        )
        loader = get_model_loader(config)
        self.assertIsInstance(loader, IpcModelLoader)
        self.assertTrue(loader.copy_mode)
        self.assertEqual(loader.socket_path, "/tmp/test.sock")

    def test_ipc_loader_zero_copy_mode(self):
        config = LoadConfig(
            load_format=LoadFormat.IPC_CACHE,
            weight_cache_mode="client",
            weight_cache_socket="/tmp/test2.sock",
        )
        loader = get_model_loader(config)
        self.assertIsInstance(loader, IpcModelLoader)
        self.assertFalse(loader.copy_mode)


class TestCrossProcessCUDAIPC(unittest.TestCase):
    """Test CUDA IPC tensor sharing between processes via daemon protocol."""

    @classmethod
    def setUpClass(cls):
        cls.socket_path = "/tmp/test_wc_unit_ipc.sock"
        cls.ready_path = "/tmp/test_wc_unit_ipc.ready"
        cls.done_path = "/tmp/test_wc_unit_ipc.done"
        for p in [cls.socket_path, cls.ready_path, cls.done_path]:
            if os.path.exists(p):
                os.unlink(p)

        cls.proc = mp.Process(
            target=_run_ipc_daemon,
            args=(cls.socket_path, cls.ready_path, cls.done_path),
        )
        cls.proc.start()

        for _ in range(60):
            if os.path.exists(cls.ready_path):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("IPC daemon did not become ready")

    @classmethod
    def tearDownClass(cls):
        with open(cls.done_path, "w") as f:
            f.write("done\n")
        cls.proc.join(timeout=10)
        if cls.proc.is_alive():
            cls.proc.terminate()
        for p in [cls.socket_path, cls.ready_path, cls.done_path]:
            if os.path.exists(p):
                os.unlink(p)

    def test_fetch_state_ok(self):
        config = CacheConfig(
            model_path="/test/model", model_arch="Llama", tp_size=1, tp_rank=0,
            dp_size=1, quant_method="", quant_config_hash="", dtype="torch.float16",
        )
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect(self.socket_path)
        send_msg(s, {"type": "fetch_state", "config": config.to_dict()})
        result = recv_msg(s)
        s.close()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["entries"]), 3)

    def test_ipc_import_tensors(self):
        config = CacheConfig(
            model_path="/test/model", model_arch="Llama", tp_size=1, tp_rank=0,
            dp_size=1, quant_method="", quant_config_hash="", dtype="torch.float16",
        )
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect(self.socket_path)
        send_msg(s, {"type": "fetch_state", "config": config.to_dict()})
        result = recv_msg(s)
        s.close()

        for name, entry in result["entries"].items():
            imported = MultiprocessingSerializer.deserialize(entry["handle"])
            self.assertEqual(imported.shape, torch.Size(entry["shape"]))
            cloned = imported.clone()
            del imported
            self.assertEqual(cloned.shape, torch.Size(entry["shape"]))

    def test_config_mismatch_rejected(self):
        wrong_config = CacheConfig(
            model_path="/test/model", model_arch="Llama", tp_size=2, tp_rank=0,
            dp_size=1, quant_method="", quant_config_hash="", dtype="torch.float16",
        )
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect(self.socket_path)
        send_msg(s, {"type": "fetch_state", "config": wrong_config.to_dict()})
        result = recv_msg(s)
        s.close()
        self.assertEqual(result["status"], "mismatch")

    def test_query_config(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect(self.socket_path)
        send_msg(s, {"type": "query_config"})
        result = recv_msg(s)
        s.close()
        self.assertEqual(result["status"], "ok")
        self.assertIn("config", result)


def _run_ipc_daemon(socket_path, ready_path, done_path):
    """Simple daemon process for unit testing."""
    import torch as _torch
    from sglang.srt.utils import MultiprocessingSerializer as _MS
    from sglang.srt.weight_cache.protocol import send_msg as _send, recv_msg as _recv, CacheConfig as _CC
    import socket as _sock

    _torch.cuda.set_device(0)
    tensors = {
        "layer.0.weight": _torch.randn(128, 128, device="cuda:0"),
        "layer.0.bias": _torch.randn(128, device="cuda:0"),
        "layer.1.weight": _torch.randn(512, 256, device="cuda:0"),
    }
    entries = {}
    for name, tensor in tensors.items():
        entries[name] = {
            "handle": _MS.serialize(tensor, output_str=True),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "is_param": True,
        }

    config = _CC(
        model_path="/test/model", model_arch="Llama", tp_size=1, tp_rank=0,
        dp_size=1, quant_method="", quant_config_hash="", dtype="torch.float16",
    )

    if os.path.exists(socket_path):
        os.unlink(socket_path)
    s = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
    s.bind(socket_path)
    s.listen(5)
    s.settimeout(60)
    with open(ready_path, "w") as f:
        f.write("ready\n")

    try:
        while not os.path.exists(done_path):
            try:
                conn, _ = s.accept()
                try:
                    req = _recv(conn)
                    if req.get("type") == "query_config":
                        _send(conn, {"status": "ok", "config": config.to_dict()})
                    elif req.get("type") == "fetch_state":
                        ec = _CC.from_dict(req["config"])
                        if config.matches(ec):
                            _send(conn, {"status": "ok", "config": config.to_dict(), "entries": entries})
                        else:
                            _send(conn, {"status": "mismatch", "daemon_config": config.to_dict()})
                    elif req.get("type") == "ping":
                        _send(conn, {"status": "ok"})
                except Exception:
                    pass
                finally:
                    conn.close()
            except _sock.timeout:
                continue
    finally:
        s.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)


if __name__ == "__main__":
    unittest.main()