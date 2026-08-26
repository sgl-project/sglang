"""
CPU-only unit tests for the weight cache protocol layer.

These cover the pure-Python logic that the GPU end-to-end test
(test_weight_cache_daemon.py) cannot exercise cheaply:

  - length-prefixed socket framing (send_msg/recv_msg) over socketpair()
  - CacheConfig fingerprint matching / (de)serialization
  - quant-config hashing and method-name extraction
  - daemon spawn configuration and socket/ready path derivation, including
    the physical-GPU keying that lets two same-shape replicas share a node
  - the IPC quantization allowlist (the gate that keeps silently-wrong
    quant methods off the zero-copy path)
  - stale-vs-live daemon file cleanup
  - seed (daemon -> daemon mirroring) manifest metadata and the mirror's
    fingerprint subset verification
  - the ServerArgs guards on the weight-cache socket/seed options

They intentionally require no CUDA, no model download, and no daemon
process, so they run in the fast CPU suite and would catch a regression
in any of these branches before it reaches the expensive GPU path.
"""

import os
import socket
import struct
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.weight_cache.protocol import (
    IPC_QUANT_ALLOWLIST,
    CacheConfig,
    UnsupportedQuantForIPCError,
    check_ipc_quant_support,
    cleanup_stale_daemon_files,
    compute_daemon_key,
    compute_global_rank,
    compute_local_gpu_id,
    get_quant_method_name,
    get_ready_path,
    get_socket_path,
    hash_quant_config,
    is_ipc_quant_supported,
    recv_msg,
    send_msg,
    visible_daemon_keys,
)
from sglang.srt.weight_cache.transport import (
    TORCH_IPC_BACKEND,
    TorchIpcTransportBackend,
    get_client_transport_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _no_visible_device_restriction():
    """Pin the daemon key to the logical index for tests that need a synthetic id.

    compute_daemon_key resolves through CUDA_VISIBLE_DEVICES and rejects an id
    beyond the visible list, so tests picking an out-of-range id to avoid
    colliding with a real daemon must state that no restriction is in effect.
    """
    return mock.patch.dict(
        os.environ, {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
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
            )
            resp = recv_msg(b)
            resp = backend.recv_fetch_state_response(b, resp)
            imported = backend.import_tensor(resp["entries"]["x"])
            self.assertTrue(torch.equal(imported.cpu(), state_tensors["x"][0]))
            self.assertEqual(resp["transport_backend"], TORCH_IPC_BACKEND)
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

    def test_socket_and_ready_paths_are_unique_per_gpu(self):
        with _no_visible_device_restriction():
            self.assertNotEqual(get_socket_path(0), get_socket_path(1))
            self.assertTrue(get_socket_path(3).endswith("gpu3.sock"))
            self.assertTrue(get_ready_path(3).endswith("gpu3.ready"))

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


class TestDaemonKeyAddressing(CustomTestCase):
    """Socket/ready files are keyed by the *physical* GPU, not the global rank.

    This is what makes two same-shape replicas on one node possible at all:
    under the old rank keying, replica B's tp_rank 0 derived byte-for-byte the
    same /tmp path as replica A's tp_rank 0, and cleanup_stale_daemon_files saw
    A's live PID and refused to start.
    """

    def test_unrestricted_visibility_keys_on_logical_id(self):
        with _no_visible_device_restriction():
            self.assertEqual(compute_daemon_key(0), "0")
            self.assertEqual(compute_daemon_key(5), "5")
            self.assertIsNone(visible_daemon_keys())

    def test_narrowed_visibility_keys_on_physical_id(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}):
            # Logical 0 inside this replica is physically device 4.
            self.assertEqual(compute_daemon_key(0), "4")
            self.assertEqual(compute_daemon_key(3), "7")
            self.assertEqual(visible_daemon_keys(), ["4", "5", "6", "7"])

    def test_two_replicas_on_one_node_do_not_collide(self):
        # Both replicas run tp_rank 0 on their own logical device 0; only the
        # physical keying keeps them off the same socket.
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}):
            replica_a = [get_socket_path(i) for i in range(4)]
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}):
            replica_b = [get_socket_path(i) for i in range(4)]

        self.assertEqual(len(set(replica_a) | set(replica_b)), 8)
        self.assertTrue(replica_b[0].endswith("gpu4.sock"))

    def test_reordered_visibility_still_keys_on_physical_id(self):
        # A permuted list must follow the physical device, not the position.
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,2,1,0"}):
            self.assertEqual(compute_daemon_key(0), "3")
            self.assertEqual(compute_daemon_key(3), "0")

    def test_uuid_entries_are_filename_safe(self):
        with mock.patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-1a2b/3c4d,GPU-5e6f"}
        ):
            key = compute_daemon_key(0)
            self.assertNotIn("/", key)
            self.assertIn(key, get_socket_path(0))

    def test_gpu_id_beyond_visible_list_raises(self):
        # Claiming a device this process cannot see is a configuration error, not
        # something to paper over with a fabricated key.
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5"}):
            with self.assertRaises(ValueError):
                compute_daemon_key(2)


class TestDaemonLaunchConfiguration(CustomTestCase):
    def test_spawn_forwards_complete_server_args_without_projection(self):
        from sglang.srt.weight_cache import daemon

        # The spawn helper receives Engine's already-resolved ServerArgs. A
        # minimal namespace keeps this projection test CPU-only and
        # model-independent; importantly, no EPLB configuration is involved.
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
    # Use a gpu id far outside any realistic node so we never collide with a
    # daemon that might actually be running on the test host. The visibility
    # restriction is cleared so the id maps straight through to the key.
    GPU_ID = 987654

    def setUp(self):
        self._env = _no_visible_device_restriction()
        self._env.start()
        self.addCleanup(self._env.stop)

    def _paths(self):
        return get_ready_path(self.GPU_ID), get_socket_path(self.GPU_ID)

    def tearDown(self):
        for path in self._paths():
            if os.path.exists(path):
                os.unlink(path)

    def test_no_files_is_noop(self):
        # Neither file present: must return quietly, not raise.
        cleanup_stale_daemon_files(self.GPU_ID)

    def test_stale_files_without_live_pid_are_removed(self):
        ready_path, socket_path = self._paths()
        # A .ready file whose PID is unreadable is treated as a crashed-daemon
        # leftover and cleaned up.
        with open(ready_path, "w") as f:
            f.write("stale contents, no pid line\n")
        open(socket_path, "w").close()

        cleanup_stale_daemon_files(self.GPU_ID)

        self.assertFalse(os.path.exists(ready_path))
        self.assertFalse(os.path.exists(socket_path))

    def test_live_daemon_pid_blocks_cleanup(self):
        ready_path, socket_path = self._paths()
        # Our own PID is alive -> cleanup must refuse and leave files intact.
        with open(ready_path, "w") as f:
            f.write(f"pid={os.getpid()}\n")
        open(socket_path, "w").close()

        with self.assertRaises(RuntimeError):
            cleanup_stale_daemon_files(self.GPU_ID)

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

            cleanup_stale_daemon_files(self.GPU_ID, force=True)

            self.assertFalse(os.path.exists(ready_path))
            self.assertFalse(os.path.exists(socket_path))
            # The daemon holding the GPU must have been killed.
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

    GPU_ID = 987655

    def setUp(self):
        self._env = _no_visible_device_restriction()
        self._env.start()
        self.addCleanup(self._env.stop)

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

        missing_socket = get_socket_path(self.GPU_ID)
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


class TestSeedManifest(CustomTestCase):
    """The manifest is the mirror's only description of what to allocate.

    A mirror builds no nn.Module, so anything the manifest omits is gone for
    good -- hence the insistence that it cover the whole export set, not just
    named_parameters().
    """

    def test_manifest_describes_every_entry(self):
        from sglang.srt.weight_cache.seed import build_manifest

        state_tensors = {
            "w": (torch.zeros(2, 3, dtype=torch.float16), True),
            "scale": (torch.zeros(2, dtype=torch.float32), True),
            "rotary.cos_sin_cache": (torch.zeros(4, dtype=torch.bfloat16), False),
        }
        manifest = build_manifest(state_tensors)

        self.assertEqual(set(manifest), set(state_tensors))
        self.assertEqual(manifest["w"]["shape"], [2, 3])
        self.assertEqual(manifest["w"]["dtype"], "float16")
        self.assertTrue(manifest["w"]["is_param"])
        # A non-persistent buffer must survive as a buffer, not be promoted.
        self.assertFalse(manifest["rotary.cos_sin_cache"]["is_param"])

    def test_manifest_dtype_round_trip(self):
        from sglang.srt.weight_cache.seed import build_manifest, manifest_dtype

        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float8_e4m3fn,
            torch.int8,
            torch.uint8,
        ):
            manifest = build_manifest({"t": (torch.zeros(2, dtype=dtype), True)})
            self.assertEqual(manifest_dtype(manifest["t"]["dtype"]), dtype)

    def test_manifest_survives_the_wire(self):
        # The manifest travels through the same framing as everything else, so a
        # non-picklable field would only show up here.
        from sglang.srt.weight_cache.seed import build_manifest

        manifest = build_manifest({"w": (torch.zeros(2, 3, dtype=torch.float16), True)})
        a, b = socket.socketpair()
        try:
            send_msg(a, {"status": "ok", "manifest": manifest})
            self.assertEqual(recv_msg(b)["manifest"], manifest)
        finally:
            a.close()
            b.close()

    def test_unknown_dtype_name_raises(self):
        # A dtype this torch build lacks means the source runs a different torch
        # version; that must surface as a version mismatch, not an AttributeError.
        from sglang.srt.weight_cache.seed import manifest_dtype

        with self.assertRaises(RuntimeError):
            manifest_dtype("not_a_dtype")
        # "nn" resolves on torch but is not a dtype -- the isinstance check, not
        # mere attribute presence, is what makes this safe.
        with self.assertRaises(RuntimeError):
            manifest_dtype("nn")

    def test_manifest_nbytes_sums_the_payload(self):
        from sglang.srt.weight_cache.seed import build_manifest, manifest_nbytes

        manifest = build_manifest(
            {
                "a": (torch.zeros(2, 3, dtype=torch.float16), True),
                "b": (torch.zeros(4, dtype=torch.float32), True),
            }
        )
        self.assertEqual(manifest_nbytes(manifest), 2 * 3 * 2 + 4 * 4)


class TestPeerIpcSourceReachability(CustomTestCase):
    """CUDA IPC handles carry the *source process's logical* device index.

    Visibility alone is not enough: if the same physical card sits at a
    different logical index here, the handle opens against the wrong device.
    Both failures must be caught with actionable text before
    cudaIpcOpenMemHandle fails opaquely.
    """

    def _check(self, seed_meta, device_index=1):
        from sglang.srt.weight_cache.seed import PeerIpcSeedSource

        PeerIpcSeedSource._assert_source_reachable(
            seed_meta, torch.device("cuda", device_index)
        )

    def test_invisible_source_card_names_the_fix(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            with self.assertRaises(RuntimeError) as ctx:
                self._check({"source_local_index": 0, "source_daemon_key": "6"})
        message = str(ctx.exception)
        self.assertIn("CUDA_VISIBLE_DEVICES", message)
        self.assertIn("6", message)

    def test_reindexed_source_card_names_the_fix(self):
        # Physical device 4 is logical 0 in the source but logical 1 here.
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,4"}):
            with self.assertRaises(RuntimeError) as ctx:
                self._check({"source_local_index": 0, "source_daemon_key": "4"})
        message = str(ctx.exception)
        self.assertIn("CUDA_VISIBLE_DEVICES", message)
        self.assertIn("index", message)

    def test_matching_indices_pass(self):
        # Same visibility on both sides: nothing to complain about. The peer
        # access probe below is advisory and tolerates a CPU-only host.
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}):
            self._check({"source_local_index": 0, "source_daemon_key": "4"})


class TestMirrorFingerprintVerification(CustomTestCase):
    """A mirror adopts the source's CacheConfig, but only after proving the
    subset it can compute on its own is identical.

    That subset is what makes the two shards byte-identical; moe_dp_rank and
    moe_ep_rank are excluded because they are read off process groups a mirror
    never creates, and they are deterministic functions of the verified size
    fields plus tp_rank/pp_rank.
    """

    def _daemon(self, **overrides):
        from sglang.srt.weight_cache.daemon import WeightCacheDaemon

        # Bypass __init__: it touches ServerArgs and address resolution, none of
        # which the fingerprint logic under test reads.
        d = object.__new__(WeightCacheDaemon)
        attrs = dict(
            gpu_id=0,
            model_path="/models/demo",
            tp_size=4,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            dp_size=1,
            ep_size=1,
            moe_dp_size=1,
            enable_dp_attention=False,
            enable_dp_lm_head=False,
            attn_cp_size=1,
            moe_dense_tp_size=None,
            moe_a2a_backend="none",
            revision=None,
            seed_addr="/tmp/sglang_weight_cache_gpu4.sock",
        )
        attrs.update(overrides)
        for key, value in attrs.items():
            setattr(d, key, value)
        return d

    def _model_config(self):
        return SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            dtype=torch.float16,
        )

    def _source_config(self, daemon, **overrides):
        """What a matching source daemon would have published."""
        fields = daemon._fingerprint_fields(self._model_config(), "", None)
        # The two fields a mirror cannot derive; the source always has them.
        fields.update(moe_dp_rank=0, moe_ep_rank=0)
        fields.update(overrides)
        return fields

    def test_identical_configuration_verifies(self):
        d = self._daemon()
        # Must not raise.
        d._verify_seed_config(self._source_config(d), self._model_config(), "", None)

    def test_source_config_covers_every_cache_config_field(self):
        # Guards the test's own premise: the verified subset plus the two MoE
        # ranks must be the whole fingerprint, or the mirror would adopt fields
        # nobody checked.
        d = self._daemon()
        self.assertEqual(
            set(self._source_config(d)),
            set(CacheConfig.__struct_fields__),
        )

    def test_differing_shard_defining_field_is_rejected(self):
        d = self._daemon()
        for field, value in (
            ("model_path", "/models/other"),
            ("model_arch", "Qwen2ForCausalLM"),
            ("tp_size", 8),
            ("tp_rank", 1),
            ("pp_size", 2),
            ("dp_size", 2),
            ("ep_size", 2),
            ("moe_dp_size", 2),
            ("enable_dp_attention", True),
            ("moe_dense_tp_size", 1),
            ("moe_a2a_backend", "deepep"),
            ("quant_method", "fp8"),
            ("quant_config_hash", "deadbeef"),
            ("dtype", "torch.bfloat16"),
            ("revision", "v2"),
            ("torch_version", "0.0.0-not-this-build"),
        ):
            source = self._source_config(d, **{field: value})
            with self.assertRaises(
                RuntimeError, msg=f"{field} must be rejected"
            ) as ctx:
                d._verify_seed_config(source, self._model_config(), "", None)
            # The message must name the offending field, or a mismatch on a
            # 20-field fingerprint is undebuggable.
            self.assertIn(field, str(ctx.exception))

    def test_moe_ranks_are_adopted_not_verified(self):
        # A mirror has no process groups, so these two cannot be recomputed. They
        # are deliberately excluded from verification; differing values must NOT
        # fail the check (they are a function of the fields that were checked).
        d = self._daemon()
        source = self._source_config(d, moe_dp_rank=3, moe_ep_rank=5)
        d._verify_seed_config(source, self._model_config(), "", None)
        adopted = CacheConfig.from_dict(source)
        self.assertEqual(adopted.moe_dp_rank, 3)
        self.assertEqual(adopted.moe_ep_rank, 5)

    def test_source_config_missing_fields_is_version_skew(self):
        d = self._daemon()
        source = self._source_config(d)
        del source["moe_ep_rank"]
        with self.assertRaises(RuntimeError) as ctx:
            d._verify_seed_config(source, self._model_config(), "", None)
        self.assertIn("moe_ep_rank", str(ctx.exception))


class TestWeightCacheServerArgsGuards(CustomTestCase):
    """The addressing preconditions ServerArgs refuses to start without.

    _validate_weight_cache_options is exercised directly: the surrounding
    __post_init__ probes the checkpoint on disk, which these purely
    configuration-level rules do not need.
    """

    def _args(self, **overrides):
        from sglang.srt.server_args import ServerArgs

        args = object.__new__(ServerArgs)
        attrs = dict(
            weight_cache_mode="daemon",
            speculative_algorithm=None,
            enable_eplb=False,
            weight_cache_socket=None,
            weight_cache_seed=None,
            weight_cache_listen_addr=None,
            weight_cache_seed_token=None,
            tp_size=1,
            pp_size=1,
        )
        attrs.update(overrides)
        for key, value in attrs.items():
            setattr(args, key, value)
        return args

    def test_baseline_configuration_passes(self):
        self._args()._validate_weight_cache_options()

    def test_explicit_socket_rejected_beyond_one_rank(self):
        # --weight-cache-socket is a scalar: honoring it for tp>1 would point
        # every rank at one daemon and map another rank's shard.
        with self.assertRaises(ValueError) as ctx:
            self._args(
                weight_cache_socket="/tmp/x.sock", tp_size=2
            )._validate_weight_cache_options()
        self.assertIn("single rank", str(ctx.exception))

        # pp is part of the same world size, so it must trip the guard too.
        with self.assertRaises(ValueError):
            self._args(
                weight_cache_socket="/tmp/x.sock", pp_size=2
            )._validate_weight_cache_options()

    def test_explicit_socket_allowed_for_single_rank(self):
        self._args(weight_cache_socket="/tmp/x.sock")._validate_weight_cache_options()

    def test_seed_list_must_cover_every_rank(self):
        # Rank i can only mirror the source's rank i, so a short list would
        # silently leave later ranks disk-loading.
        with self.assertRaises(ValueError) as ctx:
            self._args(
                weight_cache_seed="/tmp/a.sock", tp_size=2
            )._validate_weight_cache_options()
        self.assertIn("per rank", str(ctx.exception))

        self._args(
            weight_cache_seed="/tmp/a.sock,/tmp/b.sock", tp_size=2
        )._validate_weight_cache_options()

    def test_tcp_control_plane_requires_a_token(self):
        # The cross-node plane has no equivalent of the peer-uid check that
        # protects the node-local Unix socket.
        with self.assertRaises(ValueError) as ctx:
            self._args(
                weight_cache_listen_addr="0.0.0.0:29700"
            )._validate_weight_cache_options()
        self.assertIn("token", str(ctx.exception))

        self._args(
            weight_cache_listen_addr="0.0.0.0:29700",
            weight_cache_seed_token="s3cret",
        )._validate_weight_cache_options()


if __name__ == "__main__":
    unittest.main()
