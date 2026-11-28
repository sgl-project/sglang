"""
CPU-only unit tests for the weight cache protocol layer.

These cover the pure-Python logic that the GPU end-to-end test
(test_weight_cache_daemon.py) cannot exercise cheaply:

  - length-prefixed socket framing (send_msg/recv_msg) over socketpair()
  - CacheConfig compatibility matching / (de)serialization
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
from unittest.mock import patch

import torch

import sglang.srt.platforms as platforms_mod
from sglang.srt.environ import envs
from sglang.srt.platforms.device_mixin import DeviceMixin
from sglang.srt.platforms.xpu import XpuDeviceMixin
from sglang.srt.weight_cache.protocol import (
    CUDA_IPC_QUANT_SUPPORT,
    XPU_VMM_QUANT_SUPPORT,
    CacheConfig,
    QuantIpcSupport,
    UnsupportedQuantForIPCError,
    check_ipc_quant_support,
    cleanup_stale_daemon_files,
    compute_env_stamp,
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

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

BLOCK_FP8 = {"weight_block_size": [128, 128]}


def _quant_platform(device_type, support):
    """Stand in for the active platform with one quantization-support table.

    The gate reads the table off ``current_platform``, so a runner's own device
    must not decide what these tests assert.
    """
    return patch.object(
        platforms_mod,
        "current_platform",
        SimpleNamespace(
            device_type=device_type,
            weight_cache_quant_support=lambda: support,
        ),
        create=True,
    )


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
        attn_dp_size=1,
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
        platform_env="",
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
            ("attn_dp_size", 2),
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

    def test_socket_and_ready_paths_are_unique_per_device_uuid(self):
        self.assertNotEqual(get_socket_path("gpu-aaa"), get_socket_path("gpu-bbb"))
        self.assertTrue(get_socket_path("gpu-aaa").endswith("gpu-aaa.sock"))
        self.assertTrue(get_ready_path("gpu-aaa").endswith("gpu-aaa.ready"))

    def test_custom_template_with_device_uuid_placeholder_is_honored(self):
        with envs.SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE.override(
            "/custom/dir/{device_uuid}.custom-sock"
        ):
            self.assertEqual(
                get_socket_path("gpu-aaa"), "/custom/dir/gpu-aaa.custom-sock"
            )

    def test_template_missing_device_uuid_placeholder_raises(self):
        with envs.SGLANG_WEIGHT_CACHE_SOCKET_TEMPLATE.override(
            "/tmp/sglang_weight_cache_rank{global_rank}.sock"
        ):
            with self.assertRaises(ValueError):
                get_socket_path("gpu-aaa")

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
            attn_dp_size=8,
            ep_size=8,
            moe_dp_size=2,
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


class TestIpcQuantAllowlistCudaAlike(CustomTestCase):
    """The CUDA-alike decisions, which must not change: every platform that does
    not override the hook reaches this table through the DeviceMixin default."""

    def setUp(self):
        self.platform = _quant_platform("cuda", CUDA_IPC_QUANT_SUPPORT)
        self.platform.start()
        self.addCleanup(self.platform.stop)

    def test_unquantized_is_supported_dense_and_moe(self):
        self.assertTrue(is_ipc_quant_supported("", None, is_moe=False))
        self.assertTrue(is_ipc_quant_supported("", None, is_moe=True))

    def test_block_fp8_supported_but_per_tensor_fp8_rejected(self):
        self.assertTrue(is_ipc_quant_supported("fp8", BLOCK_FP8, is_moe=False))
        self.assertTrue(is_ipc_quant_supported("fp8", BLOCK_FP8, is_moe=True))
        # Per-tensor FP8 (no weight_block_size) transposes the weight during
        # post-processing -> not reproducible by the meta-init client.
        self.assertFalse(is_ipc_quant_supported("fp8", {}, is_moe=False))
        self.assertFalse(is_ipc_quant_supported("fp8", None, is_moe=False))

    def test_unknown_method_rejected(self):
        self.assertFalse(is_ipc_quant_supported("gptq_marlin", None, is_moe=False))
        self.assertFalse(is_ipc_quant_supported("awq", None, is_moe=False))

    def test_check_raises_on_unsupported(self):
        with self.assertRaises(UnsupportedQuantForIPCError):
            check_ipc_quant_support("awq", None, where="client", is_moe=False)
        # Per-tensor FP8 must also raise even though "fp8" is a known key.
        with self.assertRaises(UnsupportedQuantForIPCError):
            check_ipc_quant_support("fp8", {}, where="daemon", is_moe=False)

    def test_check_passes_on_supported(self):
        # Should not raise.
        check_ipc_quant_support("", None, where="daemon", is_moe=False)
        check_ipc_quant_support("", None, where="daemon", is_moe=True)
        check_ipc_quant_support("fp8", BLOCK_FP8, where="daemon", is_moe=True)


class TestIpcQuantAllowlistXpu(CustomTestCase):
    """XPU allows only unquantized dense weights."""

    def setUp(self):
        self.platform = _quant_platform("xpu", XPU_VMM_QUANT_SUPPORT)
        self.platform.start()
        self.addCleanup(self.platform.stop)

    def test_unquantized_dense_is_supported(self):
        self.assertTrue(is_ipc_quant_supported("", None, is_moe=False))
        check_ipc_quant_support("", None, where="daemon", is_moe=False)

    def test_unquantized_moe_fails_closed(self):
        # Dense parity says nothing about MoE expert weights, which take their
        # own create_weights branch; no real MoE checkpoint has been compared.
        self.assertFalse(is_ipc_quant_supported("", None, is_moe=True))
        for where in ("daemon", "client"):
            with self.assertRaises(UnsupportedQuantForIPCError) as caught:
                check_ipc_quant_support("", None, where=where, is_moe=True)
            self.assertIn("not for a MoE model", str(caught.exception))

    def test_every_quantized_method_fails_closed(self):
        # Block FP8 included, which CUDA has verified: a method is verified per
        # platform, and nothing establishes that the XPU vmm_fd manifest carries
        # what an XPU FP8 post-process leaves behind.
        for method, config in (
            ("fp8", BLOCK_FP8),
            ("fp8", {}),
            ("mxfp4", None),
            ("w8a8_int8", None),
            ("awq", None),
            ("gptq_marlin", None),
            ("compressed-tensors", None),
        ):
            for is_moe in (False, True):
                with self.subTest(method=method, is_moe=is_moe):
                    self.assertFalse(
                        is_ipc_quant_supported(method, config, is_moe=is_moe)
                    )
                    with self.assertRaises(UnsupportedQuantForIPCError):
                        check_ipc_quant_support(
                            method, config, where="daemon", is_moe=is_moe
                        )

    def test_error_names_the_platform_and_the_model_shape(self):
        with self.assertRaises(UnsupportedQuantForIPCError) as caught:
            check_ipc_quant_support("awq", None, where="client", is_moe=True)
        message = str(caught.exception)
        self.assertIn("'xpu'", message)
        self.assertIn("MoE", message)
        self.assertIn("'awq'", message)


class TestIpcQuantAllowlistPlatformIsolation(CustomTestCase):
    """A method verified on one device must not become enabled on another."""

    def test_cuda_verification_does_not_enable_xpu(self):
        with _quant_platform("cuda", CUDA_IPC_QUANT_SUPPORT):
            self.assertTrue(is_ipc_quant_supported("fp8", BLOCK_FP8, is_moe=False))
        with _quant_platform("xpu", XPU_VMM_QUANT_SUPPORT):
            self.assertFalse(is_ipc_quant_supported("fp8", BLOCK_FP8, is_moe=False))

    def test_xpu_entries_cannot_widen_what_a_cuda_alike_device_serves(self):
        # Every XPU entry is also a CUDA-alike entry, and no narrower there, so
        # adding to the XPU table can never enable something on another device.
        for method, support in XPU_VMM_QUANT_SUPPORT.items():
            with self.subTest(method=method):
                self.assertIn(method, CUDA_IPC_QUANT_SUPPORT)
                cuda = CUDA_IPC_QUANT_SUPPORT[method]
                self.assertTrue(cuda.dense or not support.dense)
                self.assertTrue(cuda.moe or not support.moe)

    def test_real_platform_classes_hand_out_the_expected_tables(self):
        # Called off the class: neither override reads self, and instantiating a
        # platform would need its device present on the runner.
        self.assertIs(
            DeviceMixin.weight_cache_quant_support(None), CUDA_IPC_QUANT_SUPPORT
        )
        self.assertIs(
            XpuDeviceMixin.weight_cache_quant_support(None), XPU_VMM_QUANT_SUPPORT
        )

    def test_xpu_allows_nothing_quantized_at_all(self):
        self.assertEqual(set(XPU_VMM_QUANT_SUPPORT), {""})


class TestQuantIpcSupportShapeGating(CustomTestCase):
    """Dense and MoE gate independently, and an absent entry fails closed."""

    def test_dense_only_entry_refuses_moe(self):
        table = {"": QuantIpcSupport(dense=True, moe=False)}
        with _quant_platform("xpu", table):
            self.assertTrue(is_ipc_quant_supported("", None, is_moe=False))
            self.assertFalse(is_ipc_quant_supported("", None, is_moe=True))
            with self.assertRaises(UnsupportedQuantForIPCError) as caught:
                check_ipc_quant_support("", None, where="daemon", is_moe=True)
            self.assertIn("not for a MoE model", str(caught.exception))

    def test_moe_only_entry_refuses_dense(self):
        table = {"": QuantIpcSupport(dense=False, moe=True)}
        with _quant_platform("xpu", table):
            self.assertFalse(is_ipc_quant_supported("", None, is_moe=False))
            self.assertTrue(is_ipc_quant_supported("", None, is_moe=True))

    def test_empty_table_refuses_everything(self):
        with _quant_platform("npu", {}):
            self.assertFalse(is_ipc_quant_supported("", None, is_moe=False))
            with self.assertRaises(UnsupportedQuantForIPCError):
                check_ipc_quant_support("", None, where="daemon", is_moe=False)

    def test_requires_predicate_runs_after_the_shape_gate(self):
        calls = []

        def requires(config):
            calls.append(config)
            return True

        table = {"fp8": QuantIpcSupport(dense=True, moe=False, requires=requires)}
        with _quant_platform("cuda", table):
            self.assertFalse(is_ipc_quant_supported("fp8", BLOCK_FP8, is_moe=True))
            self.assertEqual(calls, [])
            self.assertTrue(is_ipc_quant_supported("fp8", BLOCK_FP8, is_moe=False))
            self.assertEqual(calls, [BLOCK_FP8])


class TestCleanupStaleDaemonFiles(CustomTestCase):
    # A key no real daemon would ever compute, so this never collides with
    # one that might actually be running on the test host.
    KEY = "test-cleanup-stale-daemon-files"

    def _paths(self):
        return get_ready_path(self.KEY), get_socket_path(self.KEY)

    def tearDown(self):
        for path in self._paths():
            if os.path.exists(path):
                os.unlink(path)

    def test_no_files_is_noop(self):
        # Neither file present: must return quietly, not raise.
        cleanup_stale_daemon_files(self.KEY)

    def test_stale_files_without_live_pid_are_removed(self):
        ready_path, socket_path = self._paths()
        # A .ready file whose PID is unreadable is treated as a crashed-daemon
        # leftover and cleaned up.
        with open(ready_path, "w") as f:
            f.write("stale contents, no pid line\n")
        open(socket_path, "w").close()

        cleanup_stale_daemon_files(self.KEY)

        self.assertFalse(os.path.exists(ready_path))
        self.assertFalse(os.path.exists(socket_path))

    def test_live_daemon_pid_blocks_cleanup(self):
        ready_path, socket_path = self._paths()
        # Our own PID is alive -> cleanup must refuse and leave files intact.
        with open(ready_path, "w") as f:
            f.write(f"pid={os.getpid()}\n")
        open(socket_path, "w").close()

        with self.assertRaises(RuntimeError):
            cleanup_stale_daemon_files(self.KEY)

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

            cleanup_stale_daemon_files(self.KEY, force=True)

            self.assertFalse(os.path.exists(ready_path))
            self.assertFalse(os.path.exists(socket_path))
            # The daemon holding the identity must have been killed.
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

    KEY = "test-daemon-mode-refuses-disk-load"

    def _model_config(self):
        from types import SimpleNamespace

        # Minimal stand-in: the loader reads hf_config.quantization_config,
        # quantization, and hf_config.architectures plus hf_text_config, which
        # the quantization gate walks to tell a dense checkpoint from an MoE one.
        hf_config = SimpleNamespace(
            architectures=["LlamaForCausalLM"], quantization_config=None
        )
        return SimpleNamespace(
            model_path="/models/demo",
            hf_config=hf_config,
            hf_text_config=hf_config,
            quantization=None,
            revision=None,
            dtype="torch.float16",
        )

    def test_daemon_mode_missing_daemon_raises_instead_of_disk_load(self):
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        from sglang.srt.weight_cache.ipc_loader import WeightCacheModelLoader

        missing_socket = get_socket_path(self.KEY)
        if os.path.exists(missing_socket):
            os.unlink(missing_socket)

        loader = WeightCacheModelLoader(
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

    def test_default_discovery_queries_device_uuid_for_the_caller_gpu(self):
        """Without an explicit --weight-cache-socket (the production default),
        discovery must derive the socket from the caller's own gpu_id via
        get_device_uuid -- not silently substitute GPU 0."""
        from unittest import mock

        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        from sglang.srt.weight_cache.ipc_loader import WeightCacheModelLoader

        loader = WeightCacheModelLoader(
            load_config=LoadConfig(load_format=LoadFormat.IPC_CACHE),
            weight_cache_mode="client",
            fallback_load_format="auto",
        )
        with mock.patch(
            "sglang.srt.platforms.current_platform.get_device_uuid",
            return_value="gpu-under-test",
        ) as get_uuid:
            result = loader._fetch_from_cache(
                self._model_config(), SimpleNamespace(gpu_id=5)
            )
        get_uuid.assert_called_once_with(5)
        self.assertIsNone(result)  # no real daemon at that socket -> absent


class TestEnvStampPlatformContribution(CustomTestCase):
    @staticmethod
    def _platform(stamp):
        return patch.object(
            platforms_mod,
            "current_platform",
            SimpleNamespace(
                get_device_capability=lambda: None,
                weight_cache_env_stamp=lambda: stamp,
            ),
            create=True,
        )

    def test_contribution_is_sorted_so_dict_order_cannot_change_it(self):
        with self._platform({"b": "2", "a": "1"}):
            first = compute_env_stamp()["platform_env"]
        with self._platform({"a": "1", "b": "2"}):
            second = compute_env_stamp()["platform_env"]
        self.assertEqual(first, "a=1,b=2")
        self.assertEqual(first, second)

    def test_platform_with_no_extra_identity_leaves_the_field_empty(self):
        with self._platform({}):
            stamp = compute_env_stamp()
        self.assertEqual(stamp["platform_env"], "")
        self.assertEqual(
            set(stamp), {"device_capability", "torch_version", "platform_env"}
        )

    def test_identity_failure_is_not_swallowed(self):
        def boom():
            raise RuntimeError("incomplete XPU environment stamp")

        with patch.object(
            platforms_mod,
            "current_platform",
            SimpleNamespace(
                get_device_capability=lambda: None, weight_cache_env_stamp=boom
            ),
            create=True,
        ):
            with self.assertRaises(RuntimeError):
                compute_env_stamp()

    def test_differing_platform_identity_breaks_compatibility(self):
        base = (
            "sgl_kernel_xpu_version=0.5.0,xpu_architecture=21479031808,"
            "xpu_driver_version=1.15.39122+11"
        )
        daemon = _make_cache_config(platform_env=base)
        for label, other in (
            ("architecture", base.replace("21479031808", "11476656128")),
            ("driver", base.replace("1.15.39122+11", "1.14.38000+5")),
            ("kernel", base.replace("0.5.0", "0.6.0")),
            ("absent", ""),
        ):
            with self.subTest(differs=label):
                self.assertFalse(daemon.matches(_make_cache_config(platform_env=other)))
        self.assertTrue(daemon.matches(_make_cache_config(platform_env=base)))

    def test_peer_predating_the_field_mismatches_rather_than_crashing(self):
        config = _make_cache_config(platform_env="xpu_architecture=21479031808")
        legacy = config.to_dict()
        legacy.pop("platform_env")
        decoded = CacheConfig.from_dict(legacy)
        self.assertEqual(decoded.platform_env, "")
        self.assertFalse(config.matches(decoded))


if __name__ == "__main__":
    unittest.main()
