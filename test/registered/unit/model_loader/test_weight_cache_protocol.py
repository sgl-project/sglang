"""
CPU-only unit tests for the weight cache protocol layer.

These cover the pure-Python logic that the GPU end-to-end test
(test_weight_cache_daemon.py) cannot exercise cheaply:

  - length-prefixed socket framing (send_msg/recv_msg) over socketpair()
  - CacheConfig fingerprint matching / (de)serialization
  - quant-config hashing and method-name extraction
  - the daemon rank formula and socket/ready path derivation
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

from sglang.srt.weight_cache.protocol import (
    IPC_QUANT_ALLOWLIST,
    CacheConfig,
    UnsupportedQuantForIPCError,
    check_ipc_quant_support,
    cleanup_stale_daemon_files,
    compute_global_rank,
    get_quant_method_name,
    get_ready_path,
    get_socket_path,
    hash_quant_config,
    is_ipc_quant_supported,
    recv_msg,
    send_msg,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
        quant_method="",
        quant_config_hash="",
        dtype="torch.float16",
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


class TestCacheConfig(CustomTestCase):
    def test_identical_configs_match(self):
        self.assertTrue(_make_cache_config().matches(_make_cache_config()))

    def test_any_field_difference_breaks_match(self):
        base = _make_cache_config()
        for field, value in (
            ("tp_rank", 1),
            ("dtype", "torch.bfloat16"),
            ("quant_method", "fp8"),
            ("model_path", "/models/other"),
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
        check_ipc_quant_support("fp8", {"weight_block_size": [128, 128]}, where="daemon")

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


if __name__ == "__main__":
    unittest.main()
