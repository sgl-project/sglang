"""CPU-only tests for weight-cache identity and file-backed discovery."""

import multiprocessing
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from sglang.srt.weight_cache.protocol import (
    CacheConfig,
    hash_loader_extra_config,
)
from sglang.srt.weight_cache.registry import (
    DaemonClaim,
    DaemonRegistration,
    FileWeightCacheRegistry,
    default_runtime_dir,
    normalize_namespace,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _make_cache_config(**overrides) -> CacheConfig:
    values = dict(
        model_path="/models/demo",
        model_arch="LlamaForCausalLM",
        tp_size=4,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        ep_size=1,
        quant_method="",
        quant_config_hash="",
        dtype="torch.float16",
        revision="",
        resolved_revision="0123456789abcdef",
        device_capability="8.0",
        torch_version="2.13.0",
        load_format="auto",
        model_loader_extra_config_hash=hash_loader_extra_config({}),
        trust_remote_code=False,
    )
    values.update(overrides)
    return CacheConfig(**values)


def _claim_in_process(
    runtime_dir, namespace, config_dict, device_uuid, daemon_id, start, finish, results
):
    registry = FileWeightCacheRegistry(runtime_dir, namespace=namespace)
    identity = registry.identity_for(CacheConfig.from_dict(config_dict), device_uuid)
    start.wait(timeout=30)
    try:
        registry.claim(identity, pid=os.getpid(), daemon_id=daemon_id)
    except RuntimeError:
        results.put("rejected")
        return
    results.put("won")
    finish.wait(timeout=30)
    registry.release(identity, daemon_id=daemon_id)


class TestCacheIdentity(CustomTestCase):
    def test_default_runtime_dir_is_stable_and_per_user(self):
        with mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": "/run/user/test"}):
            self.assertEqual(
                default_runtime_dir(),
                f"/tmp/sglang-weight-cache-{os.getuid()}",
            )
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                default_runtime_dir(), f"/tmp/sglang-weight-cache-{os.getuid()}"
            )

    def test_physical_gpu_and_namespace_change_identity(self):
        with tempfile.TemporaryDirectory() as runtime_dir:
            blue = FileWeightCacheRegistry(runtime_dir, namespace="blue")
            green = FileWeightCacheRegistry(runtime_dir, namespace="green")
            config = _make_cache_config()

            blue_gpu_0 = blue.identity_for(config, "GPU-0000")
            blue_gpu_1 = blue.identity_for(config, "GPU-1111")
            green_gpu_0 = green.identity_for(config, "GPU-0000")

            self.assertNotEqual(blue_gpu_0.key, blue_gpu_1.key)
            self.assertNotEqual(blue_gpu_0.key, green_gpu_0.key)
            self.assertNotEqual(
                blue.socket_path(blue_gpu_0), blue.socket_path(blue_gpu_1)
            )

    def test_namespace_rejects_path_traversal(self):
        self.assertEqual(normalize_namespace(None), "default")
        self.assertEqual(normalize_namespace("team-a_1"), "team-a_1")
        for invalid in ("", "../other", "a/b", ".", "white space"):
            with self.subTest(namespace=invalid):
                with self.assertRaises(ValueError):
                    normalize_namespace(invalid)

    def test_runtime_directory_must_be_private(self):
        with tempfile.TemporaryDirectory() as parent:
            runtime_dir = os.path.join(parent, "shared")
            os.mkdir(runtime_dir, mode=0o755)
            with self.assertRaisesRegex(RuntimeError, "must be private"):
                FileWeightCacheRegistry(runtime_dir, namespace="test")


class TestFileWeightCacheRegistry(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.registry = FileWeightCacheRegistry(
            self._tmp.name, namespace="integration-test"
        )
        self.config = _make_cache_config()
        self.identity = self.registry.identity_for(self.config, "GPU-0000")
        self.daemon_id = "daemon-a"
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket_path = self.registry.socket_path(self.identity)

    def tearDown(self):
        self.sock.close()
        self.registry.release(self.identity, daemon_id=self.daemon_id)
        self._tmp.cleanup()

    def _publish_live_registration(self):
        self.registry.claim(self.identity, pid=os.getpid(), daemon_id=self.daemon_id)
        self.sock.bind(self.socket_path)
        self.sock.listen(1)
        self.registry.publish(
            self.identity,
            config=self.config,
            socket_path=self.socket_path,
            pid=os.getpid(),
            daemon_id=self.daemon_id,
        )

    def test_publish_discover_and_release(self):
        self._publish_live_registration()

        registration = self.registry.discover(self.config, device_uuid="GPU-0000")
        self.assertIsNotNone(registration)
        self.assertEqual(registration.socket_path, self.socket_path)
        self.assertEqual(registration.daemon_id, self.daemon_id)
        self.assertIsNotNone(
            self.registry.find_registration(daemon_id=self.daemon_id, pid=os.getpid())
        )

        self.registry.release(self.identity, daemon_id=self.daemon_id)
        self.assertIsNone(self.registry.discover(self.config, device_uuid="GPU-0000"))

    def test_discovery_requires_exact_config_device_and_namespace(self):
        self._publish_live_registration()

        with self.assertRaisesRegex(RuntimeError, "refusing disk fallback"):
            self.registry.discover(
                _make_cache_config(tp_rank=1), device_uuid="GPU-0000"
            )
        self.assertIsNone(self.registry.discover(self.config, device_uuid="GPU-1111"))
        other_namespace = FileWeightCacheRegistry(self._tmp.name, namespace="other")
        self.assertIsNone(other_namespace.discover(self.config, device_uuid="GPU-0000"))

    def test_incompatible_live_cache_on_same_gpu_blocks_disk_fallback(self):
        other_config = _make_cache_config(tp_rank=1)
        other_identity = self.registry.identity_for(other_config, "GPU-0000")
        other_socket_path = self.registry.socket_path(other_identity)
        other_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.registry.claim(
                other_identity, pid=os.getpid(), daemon_id="other-daemon"
            )
            other_socket.bind(other_socket_path)
            other_socket.listen(1)
            self.registry.publish(
                other_identity,
                config=other_config,
                socket_path=other_socket_path,
                pid=os.getpid(),
                daemon_id="other-daemon",
            )

            with self.assertRaisesRegex(RuntimeError, "refusing disk fallback"):
                self.registry.discover(self.config, device_uuid="GPU-0000")
        finally:
            other_socket.close()
            self.registry.release(other_identity, daemon_id="other-daemon")

    def test_dead_exact_registration_does_not_hide_live_gpu_owner(self):
        self._publish_live_registration()
        claim = self.registry._read(
            self.registry.claim_path(self.identity), DaemonClaim
        )
        registration = self.registry._read(
            self.registry.registration_path(self.identity), DaemonRegistration
        )
        claim.process_start_time = 0.0
        registration.process_start_time = 0.0
        self.registry._atomic_write(self.registry.claim_path(self.identity), claim)
        self.registry._atomic_write(
            self.registry.registration_path(self.identity), registration
        )

        other_config = _make_cache_config(tp_rank=1)
        other_identity = self.registry.identity_for(other_config, "GPU-0000")
        other_socket_path = self.registry.socket_path(other_identity)
        other_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.registry.claim(
                other_identity, pid=os.getpid(), daemon_id="other-daemon"
            )
            other_socket.bind(other_socket_path)
            other_socket.listen(1)
            self.registry.publish(
                other_identity,
                config=other_config,
                socket_path=other_socket_path,
                pid=os.getpid(),
                daemon_id="other-daemon",
            )

            with self.assertRaisesRegex(RuntimeError, "refusing disk fallback"):
                self.registry.discover(self.config, device_uuid="GPU-0000")
        finally:
            other_socket.close()
            self.registry.release(other_identity, daemon_id="other-daemon")

    def test_live_duplicate_claim_is_rejected(self):
        self.registry.claim(self.identity, pid=os.getpid(), daemon_id=self.daemon_id)
        with self.assertRaises(RuntimeError):
            self.registry.claim(self.identity, pid=os.getpid(), daemon_id="daemon-b")

    def test_force_replaces_live_exact_owner(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            self.registry.claim(self.identity, pid=child.pid, daemon_id="old-daemon")
            self.registry.claim(
                self.identity,
                pid=os.getpid(),
                daemon_id=self.daemon_id,
                force=True,
            )
            child.wait(timeout=5)
            claim = self.registry._read(
                self.registry.claim_path(self.identity), DaemonClaim
            )
            self.assertEqual(claim.daemon_id, self.daemon_id)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)

    def test_incompatible_live_claim_is_rejected_before_loading(self):
        self.registry.claim(self.identity, pid=os.getpid(), daemon_id=self.daemon_id)
        other_identity = self.registry.identity_for(
            _make_cache_config(tp_rank=1), "GPU-0000"
        )
        with self.assertRaisesRegex(RuntimeError, "already occupies physical GPU"):
            self.registry.claim(
                other_identity, pid=os.getpid(), daemon_id="other-daemon"
            )

    def test_exact_live_claim_blocks_fallback_while_daemon_loads(self):
        self.registry.claim(self.identity, pid=os.getpid(), daemon_id=self.daemon_id)

        with self.assertRaisesRegex(RuntimeError, "still loading"):
            self.registry.discover(self.config, device_uuid="GPU-0000")

    def test_incompatible_live_claim_on_same_gpu_blocks_fallback(self):
        other_config = _make_cache_config(tp_rank=1)
        other_identity = self.registry.identity_for(other_config, "GPU-0000")
        self.registry.claim(other_identity, pid=os.getpid(), daemon_id="other-daemon")
        try:
            with self.assertRaisesRegex(RuntimeError, "refusing disk fallback"):
                self.registry.discover(self.config, device_uuid="GPU-0000")
        finally:
            self.registry.release(other_identity, daemon_id="other-daemon")

    def test_live_unregistered_socket_is_not_unlinked(self):
        self.sock.bind(self.socket_path)
        self.sock.listen(1)

        with self.assertRaisesRegex(RuntimeError, "live unregistered service"):
            self.registry.claim(
                self.identity, pid=os.getpid(), daemon_id=self.daemon_id
            )
        self.assertTrue(os.path.exists(self.socket_path))

        self.registry.release(self.identity, daemon_id=self.daemon_id)
        self.assertTrue(os.path.exists(self.socket_path))

    def test_unpublish_stops_discovery_but_retains_live_claim(self):
        self._publish_live_registration()

        self.registry.unpublish(self.identity, daemon_id=self.daemon_id)

        self.assertFalse(os.path.exists(self.socket_path))
        self.assertFalse(os.path.exists(self.registry.registration_path(self.identity)))
        self.assertTrue(os.path.exists(self.registry.claim_path(self.identity)))
        with self.assertRaisesRegex(RuntimeError, "still loading"):
            self.registry.discover(self.config, device_uuid="GPU-0000")

    def test_zombie_process_is_not_alive(self):
        process = mock.Mock()
        process.create_time.return_value = 123.0
        process.status.return_value = "zombie"
        with mock.patch(
            "sglang.srt.weight_cache.registry.psutil.Process", return_value=process
        ):
            from sglang.srt.weight_cache.registry import process_identity_is_alive

            self.assertFalse(process_identity_is_alive(42, 123.0))

    def test_refused_stale_socket_is_unlinked_before_claim(self):
        self.sock.bind(self.socket_path)
        self.sock.close()

        self.registry.claim(self.identity, pid=os.getpid(), daemon_id=self.daemon_id)
        self.assertFalse(os.path.exists(self.socket_path))

    def test_cross_process_claims_have_one_winner(self):
        # macOS spawn re-imports CustomTestCase's CUDA-heavy dependencies before
        # the test platform stubs are installed. This worker only exercises
        # flock/filesystem state, so fork is safe here; Linux CI keeps spawn.
        start_method = "fork" if sys.platform == "darwin" else "spawn"
        context = multiprocessing.get_context(start_method)
        start = context.Event()
        finish = context.Event()
        results = context.Queue()
        processes = [
            context.Process(
                target=_claim_in_process,
                args=(
                    self._tmp.name,
                    self.registry.namespace,
                    self.config.to_dict(),
                    "GPU-0000",
                    daemon_id,
                    start,
                    finish,
                    results,
                ),
            )
            for daemon_id in ("daemon-a", "daemon-b")
        ]
        for process in processes:
            process.start()
        start.set()
        try:
            outcomes = [results.get(timeout=30) for _ in processes]
            self.assertCountEqual(outcomes, ["won", "rejected"])
        finally:
            finish.set()
            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=30)

    def test_live_registration_without_valid_claim_cannot_be_stolen(self):
        self._publish_live_registration()
        for payload in (None, b"not-json"):
            claim_path = self.registry.claim_path(self.identity)
            if payload is None:
                os.unlink(claim_path)
            else:
                with open(claim_path, "wb") as file:
                    file.write(payload)
            with self.subTest(payload=payload):
                with self.assertRaises(RuntimeError):
                    self.registry.claim(
                        self.identity, pid=os.getpid(), daemon_id="daemon-b"
                    )

    def test_wrong_owner_cannot_remove_registration(self):
        self._publish_live_registration()
        self.registry.release(self.identity, daemon_id="daemon-b")
        self.assertIsNotNone(
            self.registry.discover(self.config, device_uuid="GPU-0000")
        )

    def test_dead_claim_is_reclaimed(self):
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait(timeout=5)

        self.registry.claim(
            self.identity,
            pid=child.pid,
            daemon_id="dead-daemon",
            process_start_time=0.0,
        )
        self.registry.claim(self.identity, pid=os.getpid(), daemon_id=self.daemon_id)

    def test_dead_exact_claim_is_cleaned_before_cache_miss(self):
        self.registry.claim(
            self.identity,
            pid=os.getpid(),
            daemon_id="dead-daemon",
            process_start_time=0.0,
        )

        self.assertIsNone(self.registry.discover(self.config, device_uuid="GPU-0000"))
        self.assertFalse(os.path.exists(self.registry.claim_path(self.identity)))

    def test_reused_pid_with_different_start_time_is_stale(self):
        self.registry.claim(
            self.identity,
            pid=os.getpid(),
            daemon_id="old-daemon",
            process_start_time=0.0,
        )
        self.registry.claim(self.identity, pid=os.getpid(), daemon_id=self.daemon_id)

    def test_unknown_process_state_is_not_reclaimed(self):
        self.registry.claim(self.identity, pid=os.getpid(), daemon_id=self.daemon_id)
        with mock.patch(
            "sglang.srt.weight_cache.registry._probe_process_identity",
            return_value=None,
        ):
            with self.assertRaises(RuntimeError):
                self.registry.claim(
                    self.identity, pid=os.getpid(), daemon_id="daemon-b"
                )

    def test_malformed_claim_blocks_automatic_takeover(self):
        with open(self.registry.claim_path(self.identity), "wb") as file:
            file.write(b"not-json")

        with self.assertRaisesRegex(RuntimeError, "refusing automatic takeover"):
            self.registry.claim(
                self.identity, pid=os.getpid(), daemon_id=self.daemon_id
            )

    def test_malformed_exact_claim_blocks_discovery_fallback(self):
        with open(self.registry.claim_path(self.identity), "wb") as file:
            file.write(b"not-json")

        with self.assertRaisesRegex(RuntimeError, "malformed.*exact claim"):
            self.registry.discover(self.config, device_uuid="GPU-0000")

    def test_claim_record_identity_must_match_its_full_key_path(self):
        other_identity = self.registry.identity_for(self.config, "GPU-1111")
        self.registry._atomic_write(
            self.registry.claim_path(self.identity),
            DaemonClaim(
                version=1,
                identity=other_identity,
                daemon_id="other-daemon",
                pid=os.getpid(),
                process_start_time=0.0,
                created_at=1.0,
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "full-key path"):
            self.registry.claim(
                self.identity, pid=os.getpid(), daemon_id=self.daemon_id
            )

    def test_publish_rejects_non_identity_socket(self):
        self.registry.claim(self.identity, pid=os.getpid(), daemon_id=self.daemon_id)
        other_path = os.path.join(self.registry.sockets_dir, "other.sock")
        other_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            other_sock.bind(other_path)
            with self.assertRaises(RuntimeError):
                self.registry.publish(
                    self.identity,
                    config=self.config,
                    socket_path=other_path,
                    pid=os.getpid(),
                    daemon_id=self.daemon_id,
                )
        finally:
            other_sock.close()
            os.unlink(other_path)

    def test_malformed_exact_registration_blocks_fallback(self):
        registration_path = self.registry.registration_path(self.identity)
        with open(registration_path, "wb") as file:
            file.write(b"not-json")

        with self.assertRaisesRegex(RuntimeError, "malformed.*exact registration"):
            self.registry.discover(self.config, device_uuid="GPU-0000")

    def test_non_private_exact_registration_blocks_fallback(self):
        self._publish_live_registration()
        registration_path = self.registry.registration_path(self.identity)
        os.chmod(registration_path, 0o644)

        with self.assertRaisesRegex(RuntimeError, "non-private exact registration"):
            self.registry.discover(self.config, device_uuid="GPU-0000")

    def test_socket_path_length_is_checked(self):
        long_runtime_dir = os.path.join(self._tmp.name, "x" * 100)
        registry = FileWeightCacheRegistry(
            long_runtime_dir, namespace="integration-test"
        )
        identity = registry.identity_for(self.config, "GPU-0000")
        with self.assertRaises(ValueError):
            registry.socket_path(identity)


if __name__ == "__main__":
    unittest.main()
