"""
CPU-only unit tests for the weight cache protocol layer.

These cover the pure-Python logic that the GPU end-to-end test
(test_weight_cache_daemon.py) cannot exercise cheaply:

  - length-prefixed socket framing (send_msg/recv_msg) over socketpair()
  - CacheConfig fingerprint matching / (de)serialization
  - quant-config hashing and method-name extraction
  - the daemon rank and local-GPU formulas
  - the IPC quantization allowlist (the gate that keeps silently-wrong
    quant methods off the zero-copy path)

They intentionally require no CUDA, no model download, and no daemon
process, so they run in the fast CPU suite and would catch a regression
in any of these branches before it reaches the expensive GPU path.
"""

import os
import socket
import stat
import struct
import unittest
from unittest import mock

from sglang.srt.weight_cache.protocol import (
    IPC_QUANT_ALLOWLIST,
    CacheConfig,
    UnsupportedQuantForIPCError,
    check_ipc_parallelism,
    check_ipc_quant_support,
    compute_global_rank,
    compute_local_gpu_id,
    get_quant_method_name,
    get_resolved_model_revision,
    hash_loader_extra_config,
    hash_quant_config,
    is_ipc_quant_supported,
    normalize_model_path_for_cache,
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
        revision="",
        resolved_revision="0123456789abcdef",
        device_capability="8.0",
        torch_version="2.5.1",
        load_format="auto",
        model_loader_extra_config_hash=hash_loader_extra_config({}),
        trust_remote_code=False,
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
    def test_local_model_path_is_canonicalized_but_model_id_is_preserved(self):
        import tempfile

        with tempfile.TemporaryDirectory() as model_dir:
            self.assertEqual(
                normalize_model_path_for_cache(model_dir + "/"),
                os.path.realpath(model_dir),
            )
        self.assertEqual(
            normalize_model_path_for_cache("org/model-id"), "org/model-id"
        )

    def test_identical_configs_match(self):
        self.assertTrue(_make_cache_config().matches(_make_cache_config()))

    def test_any_field_difference_breaks_match(self):
        base = _make_cache_config()
        for field, value in (
            ("tp_rank", 1),
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

    def test_every_field_affects_fingerprint(self):
        base = _make_cache_config()
        for field in base.__struct_fields__:
            value = getattr(base, field)
            if isinstance(value, bool):
                replacement = not value
            elif isinstance(value, int):
                replacement = value + 1
            else:
                replacement = f"{value}-different"
            self.assertNotEqual(
                base.fingerprint(),
                _make_cache_config(**{field: replacement}).fingerprint(),
                msg=field,
            )

    def test_resolved_model_revision_uses_huggingface_commit_hash(self):
        from types import SimpleNamespace

        model_config = SimpleNamespace(hf_config=SimpleNamespace(_commit_hash="abc123"))
        self.assertEqual(get_resolved_model_revision(model_config), "abc123")
        self.assertEqual(get_resolved_model_revision(SimpleNamespace()), "")


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

    def test_unserializable_quant_config_fails_closed(self):
        class _Q:
            value = object()

            def __init__(self):
                self.value = object()

        with self.assertRaises(ValueError):
            hash_quant_config(_Q())

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


class TestRankAndGpuMapping(CustomTestCase):
    def test_compute_global_rank_formula(self):
        self.assertEqual(compute_global_rank(tp_size=4, pp_rank=0, tp_rank=3), 3)
        self.assertEqual(compute_global_rank(tp_size=4, pp_rank=1, tp_rank=0), 4)
        self.assertEqual(compute_global_rank(tp_size=4, pp_rank=2, tp_rank=1), 9)

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


class TestParallelismGate(CustomTestCase):
    def test_single_dp_ep_layout_is_supported(self):
        check_ipc_parallelism(1, 1, where="test")

    def test_dp_and_ep_layouts_fail_closed(self):
        for dp_size, ep_size in ((2, 1), (1, 2), (2, 2)):
            with self.subTest(dp_size=dp_size, ep_size=ep_size):
                with self.assertRaisesRegex(ValueError, "not supported yet"):
                    check_ipc_parallelism(dp_size, ep_size, where="test")

    def test_server_args_rejects_dp_and_ep_before_model_inspection(self):
        from sglang.srt.server_args import ServerArgs

        for kwargs in ({"dp_size": 2}, {"ep_size": 2}):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, "not supported yet"):
                    ServerArgs(
                        model_path="dummy",
                        weight_cache_mode="client",
                        **kwargs,
                    )

    def test_standalone_daemon_rejects_dp_and_ep_before_runtime_setup(self):
        from sglang.srt.weight_cache.daemon import WeightCacheDaemon

        for kwargs in ({"dp_size": 2}, {"ep_size": 2}):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, "not supported yet"):
                    WeightCacheDaemon(model_path="/models/demo", gpu_id=0, **kwargs)

    def test_client_config_builder_rejects_runtime_dp_and_ep(self):
        from types import SimpleNamespace

        from sglang.srt.weight_cache.ipc_loader import IpcModelLoader

        loader = object.__new__(IpcModelLoader)
        model_config = SimpleNamespace()
        for topology in (
            {"dp_size": 2, "moe_ep_size": 1},
            {"dp_size": 1, "moe_ep_size": 2},
        ):
            parallel = SimpleNamespace(**topology)
            with (
                mock.patch(
                    "sglang.srt.runtime_context.get_parallel",
                    return_value=parallel,
                ),
                self.subTest(**topology),
            ):
                with self.assertRaisesRegex(ValueError, "not supported yet"):
                    loader._build_engine_config(model_config, device_id=0)


class TestDaemonLifecycle(CustomTestCase):
    def test_shutdown_keeps_exported_tensors_alive_until_process_exit(self):
        from sglang.srt.weight_cache.daemon import WeightCacheDaemon

        daemon = object.__new__(WeightCacheDaemon)
        daemon._running = True
        model = object()
        daemon.model = model
        daemon.state_entries = {"weight": {}}

        with (
            mock.patch(
                "sglang.srt.weight_cache.daemon.dist.is_initialized",
                return_value=True,
            ),
            mock.patch(
                "sglang.srt.weight_cache.daemon.dist.destroy_process_group",
            ) as destroy_process_group,
        ):
            daemon.shutdown()

        destroy_process_group.assert_called_once()
        self.assertFalse(daemon._running)
        self.assertIs(daemon.model, model)
        self.assertEqual(daemon.state_entries, {"weight": {}})


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


class TestExplicitSocketBypass(CustomTestCase):

    def _model_config(self):
        from types import SimpleNamespace

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

    def test_client_explicit_socket_bypasses_registry(self):
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        from sglang.srt.weight_cache.ipc_loader import IpcModelLoader

        missing_socket = f"/tmp/sglang-weight-cache-missing-{os.getpid()}.sock"
        if os.path.exists(missing_socket):
            os.unlink(missing_socket)

        loader = IpcModelLoader(
            load_config=LoadConfig(load_format=LoadFormat.IPC_CACHE),
            socket_path=missing_socket,
            weight_cache_mode="client",
            fallback_load_format="auto",
        )
        fallback_model = object()

        with (
            mock.patch(
                "sglang.srt.weight_cache.ipc_loader.FileWeightCacheRegistry"
            ) as registry_cls,
            mock.patch.object(
                loader, "_fallback_load", return_value=fallback_model
            ) as fallback,
        ):
            result = loader.load_model(
                model_config=self._model_config(), device_config=None
            )
        registry_cls.assert_not_called()
        fallback.assert_called_once()
        self.assertIs(result, fallback_model)


class TestClientResponseIdentity(CustomTestCase):
    def _registration(self, config):
        from sglang.srt.weight_cache.registry import (
            CacheIdentity,
            DaemonRegistration,
        )

        identity = CacheIdentity(
            namespace="test",
            device_uuid="GPU-0000",
            config_fingerprint=config.fingerprint(),
        )
        return DaemonRegistration(
            version=1,
            identity=identity,
            daemon_id="daemon-a",
            pid=os.getpid(),
            process_start_time=123.5,
            hostname="host",
            socket_path="/tmp/test.sock",
            config=config.to_dict(),
            created_at=1.0,
        )

    def _result(self, config, registration):
        return {
            "status": "ok",
            "config": config.to_dict(),
            "daemon": {
                "daemon_id": registration.daemon_id,
                "device_uuid": registration.identity.device_uuid,
                "config_fingerprint": registration.identity.config_fingerprint,
                "pid": registration.pid,
                "process_start_time": registration.process_start_time,
            },
        }

    def test_response_binds_config_gpu_and_registration(self):
        from sglang.srt.weight_cache.ipc_loader import IpcModelLoader

        config = _make_cache_config()
        registration = self._registration(config)
        IpcModelLoader._verify_response_identity(
            self._result(config, registration),
            engine_config=config,
            device_uuid="GPU-0000",
            registration=registration,
        )

        mutations = {
            "wrong_config": lambda result: result.update(
                config=_make_cache_config(tp_rank=1).to_dict()
            ),
            "wrong_gpu": lambda result: result["daemon"].update(device_uuid="GPU-1111"),
            "wrong_daemon": lambda result: result["daemon"].update(
                daemon_id="daemon-b"
            ),
            "wrong_fingerprint": lambda result: result["daemon"].update(
                config_fingerprint="0" * 64
            ),
            "wrong_start_time": lambda result: result["daemon"].update(
                process_start_time=999.0
            ),
        }
        for name, mutate in mutations.items():
            result = self._result(config, registration)
            mutate(result)
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError):
                    IpcModelLoader._verify_response_identity(
                        result,
                        engine_config=config,
                        device_uuid="GPU-0000",
                        registration=registration,
                    )

    def test_watchdog_rechecks_liveness_before_tensor_import(self):
        from sglang.srt.weight_cache.ipc_loader import IpcModelLoader

        loader = object.__new__(IpcModelLoader)
        with mock.patch(
            "sglang.srt.weight_cache.ipc_loader.process_identity_is_alive",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "before tensor import"):
                loader._start_daemon_liveness_watchdog(os.getpid(), 123.5)

    def test_discovered_live_registration_cannot_become_disk_fallback(self):
        from types import SimpleNamespace

        from sglang.srt.weight_cache.ipc_loader import IpcModelLoader

        config = _make_cache_config()
        registration = self._registration(config)
        registry = mock.Mock()
        registry.discover.return_value = registration
        loader = object.__new__(IpcModelLoader)
        loader.socket_path = None
        loader._runtime_dir = "/tmp/test-runtime"
        loader._namespace = "test"

        with (
            mock.patch.object(loader, "_build_engine_config", return_value=config),
            mock.patch(
                "sglang.srt.weight_cache.ipc_loader.FileWeightCacheRegistry",
                return_value=registry,
            ),
            mock.patch(
                "sglang.srt.platforms.current_platform.get_device_uuid",
                return_value="GPU-0000",
            ),
            mock.patch(
                "sglang.srt.weight_cache.ipc_loader.os.lstat",
                side_effect=FileNotFoundError,
            ),
            mock.patch(
                "sglang.srt.weight_cache.ipc_loader.process_identity_is_alive",
                return_value=True,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "refusing disk fallback"):
                loader._fetch_from_cache(
                    model_config=SimpleNamespace(),
                    device_config=SimpleNamespace(gpu_id=0),
                )

    def test_refused_connection_falls_back_only_for_provably_dead_registration(self):
        from types import SimpleNamespace

        from sglang.srt.weight_cache.ipc_loader import IpcModelLoader

        config = _make_cache_config()
        registration = self._registration(config)
        registry = mock.Mock()
        registry.discover.return_value = registration
        loader = object.__new__(IpcModelLoader)
        loader.socket_path = None
        loader._runtime_dir = "/tmp/test-runtime"
        loader._namespace = "test"
        fake_socket = mock.Mock()
        fake_socket.connect.side_effect = ConnectionRefusedError

        common_patches = (
            mock.patch.object(loader, "_build_engine_config", return_value=config),
            mock.patch(
                "sglang.srt.weight_cache.ipc_loader.FileWeightCacheRegistry",
                return_value=registry,
            ),
            mock.patch(
                "sglang.srt.platforms.current_platform.get_device_uuid",
                return_value="GPU-0000",
            ),
            mock.patch(
                "sglang.srt.weight_cache.ipc_loader.os.lstat",
                return_value=SimpleNamespace(
                    st_mode=stat.S_IFSOCK, st_uid=os.getuid()
                ),
            ),
            mock.patch("socket.socket", return_value=fake_socket),
        )
        with (
            common_patches[0],
            common_patches[1],
            common_patches[2],
            common_patches[3],
            common_patches[4],
        ):
            with mock.patch(
                "sglang.srt.weight_cache.ipc_loader.process_identity_is_alive",
                return_value=False,
            ):
                self.assertIsNone(
                    loader._fetch_from_cache(
                        model_config=SimpleNamespace(),
                        device_config=SimpleNamespace(gpu_id=0),
                    )
                )

            with mock.patch(
                "sglang.srt.weight_cache.ipc_loader.process_identity_is_alive",
                return_value=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "refused the connection"):
                    loader._fetch_from_cache(
                        model_config=SimpleNamespace(),
                        device_config=SimpleNamespace(gpu_id=0),
                    )


if __name__ == "__main__":
    unittest.main()
