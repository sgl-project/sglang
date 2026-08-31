"""
CPU-only unit tests for the weight cache protocol layer.

These cover the pure-Python logic that the GPU end-to-end test
(test_weight_cache_daemon.py) cannot exercise cheaply:

  - length-prefixed socket framing (send_msg/recv_msg) over socketpair()
  - CacheConfig fingerprint matching / (de)serialization
  - quant-config hashing and method-name extraction
  - daemon spawn configuration and socket/ready path derivation
  - the IPC quantization allowlist (the gate that keeps silently-wrong
    quant methods off the zero-copy path)
  - stale-vs-live daemon file cleanup

They intentionally require no CUDA, no model download, and no daemon
process, so they run in the fast CPU suite and would catch a regression
in any of these branches before it reaches the expensive GPU path.
"""

import os
import socket
import struct
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.weight_cache.protocol import (
    IPC_QUANT_ALLOWLIST,
    CacheConfig,
    UnsupportedQuantForIPCError,
    check_ipc_quant_support,
    cleanup_stale_daemon_files,
    compute_global_rank,
    compute_local_gpu_id,
    get_quant_method_name,
    get_ready_path,
    get_socket_path,
    hash_quant_config,
    is_ipc_quant_supported,
    recv_msg,
    send_msg,
)
from sglang.srt.weight_cache.transport import (
    TORCH_IPC_BACKEND,
    TorchIpcTransportBackend,
    get_client_transport_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_cache_config(**overrides) -> CacheConfig:
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


class TestProtocolFraming(CustomTestCase):
    """Length-prefixed pickle framing over a real socket pair."""

    def test_round_trip(self):
        a, b = socket.socketpair()
        try:
            payload = {"handles": [1, 2, 3], "meta": ("x", 4.5), "flag": True}
            send_msg(a, payload)
            self.assertEqual(recv_msg(b), payload)
        finally:
            a.close()
            b.close()

    def test_multiple_messages_are_framed_independently(self):
        a, b = socket.socketpair()
        try:
            send_msg(a, {"n": 1})
            send_msg(a, {"n": 2})
            self.assertEqual(recv_msg(b), {"n": 1})
            self.assertEqual(recv_msg(b), {"n": 2})
        finally:
            a.close()
            b.close()

    def test_connection_closed_mid_header_raises(self):
        a, b = socket.socketpair()
        try:
            # Peer sends only a partial header, then hangs up.
            a.sendall(struct.pack("!I", 128)[:2])
            a.close()
            with self.assertRaises(ConnectionError):
                recv_msg(b)
        finally:
            b.close()

    def test_connection_closed_mid_body_raises(self):
        a, b = socket.socketpair()
        try:
            # Full header promising 128 bytes, but no body follows.
            a.sendall(struct.pack("!I", 128))
            a.close()
            with self.assertRaises(ConnectionError):
                recv_msg(b)
        finally:
            b.close()


class TestTransportBackend(CustomTestCase):
    def test_default_backend_is_torch_ipc(self):
        backend = get_client_transport_backend(None)
        self.assertEqual(backend.name, TORCH_IPC_BACKEND)

    def test_unknown_backend_raises(self):
        with self.assertRaises(RuntimeError):
            get_client_transport_backend("does_not_exist")

    def test_torch_ipc_backend_round_trip(self):
        backend = TorchIpcTransportBackend()
        state_tensors = {"x": (torch.arange(8, dtype=torch.float32), True)}
        entries = backend.prepare_export(state_tensors)

        a, b = socket.socketpair()
        try:
            backend.send_fetch_state_response(
                a,
                config={"k": "v"},
                entries=entries,
                pid=123,
                preloaded_weights_bytes=65536,
            )
            resp = recv_msg(b)
            resp = backend.recv_fetch_state_response(b, resp)
            imported = backend.import_tensor(resp["entries"]["x"])
            self.assertTrue(torch.equal(imported.cpu(), state_tensors["x"][0]))
            self.assertEqual(resp["transport_backend"], TORCH_IPC_BACKEND)
            self.assertEqual(resp["preloaded_weights_bytes"], 65536)
        finally:
            a.close()
            b.close()


class TestCacheConfig(CustomTestCase):
    def test_identical_configs_match(self):
        self.assertTrue(_make_cache_config().matches(_make_cache_config()))

    def test_any_field_difference_breaks_match(self):
        base = _make_cache_config()
        for field, value in (
            ("tp_rank", 1),
            ("moe_dp_rank", 1),
            ("moe_ep_rank", 1),
            ("enable_dp_attention", True),
            ("moe_dense_tp_size", 1),
            ("moe_a2a_backend", "mooncake"),
            ("dtype", "torch.bfloat16"),
            ("quant_method", "fp8"),
            ("model_path", "/models/other"),
            ("revision", "v2"),
            ("device_capability", "9.0"),
            ("torch_version", "2.4.0"),
        ):
            self.assertFalse(
                base.matches(_make_cache_config(**{field: value})),
                msg=f"{field} difference should break match",
            )

    def test_dict_round_trip(self):
        cfg = _make_cache_config(quant_method="fp8", quant_config_hash="abc123")
        restored = CacheConfig.from_dict(cfg.to_dict())
        self.assertTrue(cfg.matches(restored))
        self.assertEqual(cfg.to_dict(), restored.to_dict())


class TestQuantConfigHashing(CustomTestCase):
    def test_none_hashes_to_empty(self):
        self.assertEqual(hash_quant_config(None), "")

    def test_dict_hash_is_deterministic_and_order_insensitive(self):
        h1 = hash_quant_config({"bits": 8, "group_size": 128})
        h2 = hash_quant_config({"group_size": 128, "bits": 8})
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, hash_quant_config({"bits": 4, "group_size": 128}))

    def test_hash_is_not_truncated(self):
        # A correctness gate must use the full SHA-256 digest, not a 16-char prefix.
        self.assertEqual(len(hash_quant_config({"bits": 8})), 64)

    def test_hash_does_not_embed_object_address(self):
        # Two distinct instances with identical public attrs must hash equal,
        # otherwise configs would mismatch across processes (the bug the
        # docstring warns about).
        class _Q:
            def __init__(self):
                self.bits = 8
                self.method = "fp8"

        self.assertEqual(hash_quant_config(_Q()), hash_quant_config(_Q()))

    def test_get_quant_method_name_variants(self):
        self.assertEqual(get_quant_method_name(None), "")
        self.assertEqual(get_quant_method_name("fp8"), "fp8")

        class _WithGetName:
            def get_name(self):
                return "gptq_marlin"

        class _WithName:
            name = "awq"

        self.assertEqual(get_quant_method_name(_WithGetName()), "gptq_marlin")
        self.assertEqual(get_quant_method_name(_WithName()), "awq")


class TestGlobalRankAndPaths(CustomTestCase):
    def test_compute_global_rank_formula(self):
        self.assertEqual(compute_global_rank(tp_size=4, pp_rank=0, tp_rank=3), 3)
        self.assertEqual(compute_global_rank(tp_size=4, pp_rank=1, tp_rank=0), 4)
        self.assertEqual(compute_global_rank(tp_size=4, pp_rank=2, tp_rank=1), 9)

    def test_socket_and_ready_paths_are_unique_per_rank(self):
        self.assertNotEqual(get_socket_path(0), get_socket_path(1))
        self.assertTrue(get_socket_path(3).endswith("rank3.sock"))
        self.assertTrue(get_ready_path(3).endswith("rank3.ready"))

    def test_compute_local_gpu_id_honors_base_and_step(self):
        # Single-node TP=4: identity mapping rank -> gpu.
        self.assertEqual(
            compute_local_gpu_id(0, 2, pp_size_per_node=1, tp_size_per_node=4),
            2,
        )
        # base_gpu_id offsets every rank; gpu_id_step strides between them.
        self.assertEqual(
            compute_local_gpu_id(
                0, 2, pp_size_per_node=1, tp_size_per_node=4, base_gpu_id=4
            ),
            6,
        )
        self.assertEqual(
            compute_local_gpu_id(
                0, 2, pp_size_per_node=1, tp_size_per_node=4, gpu_id_step=2
            ),
            4,
        )


class TestDaemonLaunchConfiguration(CustomTestCase):
    def test_spawn_forwards_complete_server_args_without_projection(self):
        from sglang.srt.weight_cache import daemon

        # The spawn helper receives Engine's already-resolved ServerArgs. A
        # minimal namespace keeps this forwarding test CPU-only and
        # model-independent while covering the static DP/EP and EPLB fields
        # consumed by the daemon.
        server_args = SimpleNamespace(
            model_path="/models/demo",
            tp_size=8,
            pp_size=1,
            dp_size=8,
            ep_size=8,
            moe_dp_size=2,
            enable_dp_attention=True,
            enable_dp_lm_head=True,
            attn_cp_size=2,
            moe_dense_tp_size=1,
            moe_a2a_backend="mooncake",
            deepep_mode="low_latency",
            enable_eplb=True,
            ep_num_redundant_experts=72,
            load_format="safetensors",
            dtype="bfloat16",
            quantization="fp8",
            model_loader_extra_config='{"key": "value"}',
            trust_remote_code=True,
            revision="test-revision",
        )

        class FakeProcess:
            pid = 1234

            def start(self):
                pass

        class FakeContext:
            def Process(self, **kwargs):
                self.kwargs = kwargs
                return FakeProcess()

        fake_context = FakeContext()
        from unittest import mock

        with mock.patch.object(
            daemon.multiprocessing, "get_context", return_value=fake_context
        ) as get_context:
            result = daemon.spawn_weight_cache_daemon(
                server_args,
                gpu_id=3,
                tp_rank=3,
                pp_rank=0,
                dist_init_method="tcp://127.0.0.1:29500",
            )

        get_context.assert_called_once_with("spawn")
        self.assertIsInstance(result, FakeProcess)
        self.assertIs(fake_context.kwargs["target"], daemon.run_weight_cache_daemon)
        self.assertEqual(
            fake_context.kwargs["args"],
            (server_args, 3, 3, 0, "tcp://127.0.0.1:29500"),
        )


class TestIpcQuantAllowlist(CustomTestCase):
    def test_unquantized_is_supported(self):
        self.assertTrue(is_ipc_quant_supported("", None))

    def test_block_fp8_supported_but_per_tensor_fp8_rejected(self):
        self.assertTrue(
            is_ipc_quant_supported("fp8", {"weight_block_size": [128, 128]})
        )
        # Per-tensor FP8 (no weight_block_size) transposes the weight during
        # post-processing -> not reproducible by the meta-init client.
        self.assertFalse(is_ipc_quant_supported("fp8", {}))
        self.assertFalse(is_ipc_quant_supported("fp8", None))

    def test_unknown_method_rejected(self):
        self.assertFalse(is_ipc_quant_supported("gptq_marlin", None))
        self.assertFalse(is_ipc_quant_supported("awq", None))

    def test_check_raises_on_unsupported(self):
        with self.assertRaises(UnsupportedQuantForIPCError):
            check_ipc_quant_support("awq", None, where="client")
        # Per-tensor FP8 must also raise even though "fp8" is a known key.
        with self.assertRaises(UnsupportedQuantForIPCError):
            check_ipc_quant_support("fp8", {}, where="daemon")

    def test_check_passes_on_supported(self):
        # Should not raise.
        check_ipc_quant_support("", None, where="daemon")
        check_ipc_quant_support(
            "fp8", {"weight_block_size": [128, 128]}, where="daemon"
        )

    def test_allowlist_registry_shape(self):
        # Guard against accidentally widening the allowlist without review.
        self.assertEqual(set(IPC_QUANT_ALLOWLIST), {"", "fp8"})


class TestCleanupStaleDaemonFiles(CustomTestCase):
    # Use a rank far outside any realistic tp*pp layout so we never collide
    # with a daemon that might actually be running on the test host.
    RANK = 987654

    def _paths(self):
        return get_ready_path(self.RANK), get_socket_path(self.RANK)

    def tearDown(self):
        for path in self._paths():
            if os.path.exists(path):
                os.unlink(path)

    def test_no_files_is_noop(self):
        # Neither file present: must return quietly, not raise.
        cleanup_stale_daemon_files(self.RANK)

    def test_stale_files_without_live_pid_are_removed(self):
        ready_path, socket_path = self._paths()
        # A .ready file whose PID is unreadable is treated as a crashed-daemon
        # leftover and cleaned up.
        with open(ready_path, "w") as f:
            f.write("stale contents, no pid line\n")
        open(socket_path, "w").close()

        cleanup_stale_daemon_files(self.RANK)

        self.assertFalse(os.path.exists(ready_path))
        self.assertFalse(os.path.exists(socket_path))

    def test_live_daemon_pid_blocks_cleanup(self):
        ready_path, socket_path = self._paths()
        # Our own PID is alive -> cleanup must refuse and leave files intact.
        with open(ready_path, "w") as f:
            f.write(f"pid={os.getpid()}\n")
        open(socket_path, "w").close()

        with self.assertRaises(RuntimeError):
            cleanup_stale_daemon_files(self.RANK)

        self.assertTrue(os.path.exists(ready_path))
        self.assertTrue(os.path.exists(socket_path))

    def test_force_takes_over_from_live_pid(self):
        ready_path, socket_path = self._paths()
        # Spawn a real child we are allowed to kill, point the ready file at it,
        # then force-takeover: the child must be killed and the files removed.
        import subprocess
        import sys

        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            with open(ready_path, "w") as f:
                f.write(f"pid={child.pid}\n")
            open(socket_path, "w").close()

            cleanup_stale_daemon_files(self.RANK, force=True)

            self.assertFalse(os.path.exists(ready_path))
            self.assertFalse(os.path.exists(socket_path))
            # The daemon holding the rank must have been killed.
            self.assertEqual(child.wait(timeout=5), -9)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


class TestDaemonModeRefusesDiskLoad(CustomTestCase):
    """In daemon mode the engine and daemon share a GPU, so a missing daemon
    must be a hard error — NOT a silent disk-load that would OOM the shared
    device. This exercises that contract without a GPU or a live daemon by
    pointing the loader at a socket path that does not exist.
    """

    RANK = 987655

    def _model_config(self):
        from types import SimpleNamespace

        # Minimal stand-in: the loader only reads hf_config.quantization_config,
        # quantization, and (unreached here) hf_config.architectures.
        hf_config = SimpleNamespace(
            architectures=["LlamaForCausalLM"], quantization_config=None
        )
        return SimpleNamespace(
            model_path="/models/demo",
            hf_config=hf_config,
            quantization=None,
            revision=None,
            dtype="torch.float16",
        )

    def test_daemon_mode_missing_daemon_raises_instead_of_disk_load(self):
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        from sglang.srt.weight_cache.ipc_loader import IpcModelLoader

        missing_socket = get_socket_path(self.RANK)
        if os.path.exists(missing_socket):
            os.unlink(missing_socket)

        loader = IpcModelLoader(
            load_config=LoadConfig(load_format=LoadFormat.IPC_CACHE),
            socket_path=missing_socket,
            weight_cache_mode="daemon",
            fallback_load_format="auto",
        )

        with self.assertRaises(RuntimeError) as ctx:
            loader.load_model(model_config=self._model_config(), device_config=None)
        # The error must be about the missing daemon, proving we did not quietly
        # fall through to a disk load.
        self.assertIn("daemon", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
