"""
CPU-only unit tests for the weight cache protocol layer.

These cover the pure-Python logic that the GPU end-to-end test
(test_weight_cache_daemon.py) cannot exercise cheaply:

  - length-prefixed socket framing (send_msg/recv_msg) over socketpair()
  - the split between WeightCacheConfig (the model) and LoadSpec (the rest),
    their builders, digests and schema validation
  - the config message a daemon and its engines compare
  - quant-config hashing and method-name extraction
  - daemon spawn configuration and socket/ready path derivation
  - the IPC quantization allowlist (the gate that keeps silently-wrong
    quant methods off the zero-copy path)
  - stale-vs-live daemon file cleanup

They intentionally require no CUDA, no model download, and no daemon
process, so they run in the fast CPU suite and would catch a regression
in any of these branches before it reaches the expensive GPU path.
"""

import contextlib
import os
import socket
import struct
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import msgspec
import torch
from transformers import PretrainedConfig

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.weight_cache.config import (
    build_load_spec,
    build_weight_cache_config,
    path_checkpoint_id,
)
from sglang.srt.weight_cache.daemon import WeightCacheDaemon
from sglang.srt.weight_cache.ipc_loader import WeightCacheModelLoader
from sglang.srt.weight_cache.protocol import (
    UnsupportedQuantForIPCError,
    cache_config_message,
    cache_config_mismatches,
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
from sglang.srt.weight_cache.schema import LoadSpec, WeightCacheConfig, snapshot_tag
from sglang.srt.weight_cache.transport import (
    TORCH_IPC_BACKEND,
    TorchIpcTransportBackend,
    get_client_transport_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _model_config(
    *,
    hf_config=None,
    model_path="m",
    revision=None,
    quantization=None,
    dtype=torch.float32,
):
    return SimpleNamespace(
        hf_config=PretrainedConfig() if hf_config is None else hf_config,
        model_path=model_path,
        revision=revision,
        quantization=quantization,
        dtype=dtype,
    )


def _make_weight_cache_config(*, checkpoint_id="rev1", draft_model_idx=None, **kw):
    return build_weight_cache_config(
        _model_config(**kw),
        checkpoint_id=checkpoint_id,
        draft_model_idx=draft_model_idx,
    )


@contextlib.contextmanager
def _runtime_context(*, tp_size=2, tp_rank=0, **server_args):
    with (
        get_context().override_server_args(tp_size=tp_size, **server_args),
        get_parallel().override(
            tp_rank=tp_rank,
            pp_rank=0,
            moe_dp_rank=0,
            moe_ep_rank=0,
            moe_dp_group=SimpleNamespace(world_size=1),
        ),
    ):
        yield


def _make_load_spec(**context):
    with _runtime_context(**context):
        return build_load_spec(device=torch.device("cpu"))


def _make_message(**spec_overrides):
    spec = msgspec.structs.replace(_make_load_spec(), **spec_overrides)
    return cache_config_message(_make_weight_cache_config(), spec)


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


class TestWeightCacheConfigAndLoadSpec(CustomTestCase):
    def test_the_model_and_the_rest_have_no_field_in_common(self):
        model_fields = set(WeightCacheConfig.__struct_fields__)
        self.assertEqual(
            model_fields,
            {
                "checkpoint_id",
                "model_arch",
                "hf_config_hash",
                "dtype",
                "quant_method",
                "quant_config_hash",
                "draft_model_idx",
            },
        )
        self.assertEqual(
            set(LoadSpec.__struct_fields__),
            {
                *("tp_size", "pp_size", "dp_size", "ep_size", "moe_dp_size"),
                *("attn_dp_size", "attn_cp_size", "moe_cp_size", "moe_dense_tp_size"),
                *("enable_dp_lm_head", "tp_rank", "pp_rank", "moe_dp_rank"),
                *("moe_ep_rank", "ep_num_redundant_experts", "enable_eplb"),
                *("init_expert_location", "moe_a2a_backend", "moe_runner_backend"),
                *("fp4_gemm_runner_backend", "fp8_gemm_runner_backend"),
                *("bf16_gemm_backend", "flashinfer_cutedsl_swap_ab"),
                *("enable_fp32_lm_head", "device_capability", "torch_version"),
            },
        )
        self.assertTrue(model_fields.isdisjoint(LoadSpec.__struct_fields__))

    def test_wire_schema_rejects_a_field_of_the_other_struct(self):
        model, spec = _make_weight_cache_config(), _make_load_spec()
        for value, struct, foreign in (
            (model, WeightCacheConfig, "tp_size"),
            (spec, LoadSpec, "checkpoint_id"),
        ):
            with self.subTest(struct=struct.__name__):
                wire = msgspec.json.encode(value)
                self.assertEqual(msgspec.json.decode(wire, type=struct), value)
                invalid = {**msgspec.to_builtins(value), foreign: 1}
                with self.assertRaisesRegex(msgspec.ValidationError, foreign):
                    msgspec.convert(invalid, struct)
                with self.assertRaises(AttributeError):
                    setattr(value, struct.__struct_fields__[0], "frozen")

    def test_every_model_field_changes_the_model_digest(self):
        base = _make_weight_cache_config()
        quant_config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
        other = _make_weight_cache_config(
            checkpoint_id="rev2",
            hf_config=PretrainedConfig(
                architectures=["OtherModel"], quantization_config=quant_config
            ),
            dtype=torch.bfloat16,
            quantization="fp8",
            draft_model_idx=1,
        )
        self.assertEqual(other.model_arch, "OtherModel")
        self.assertEqual(other.dtype, "torch.bfloat16")
        self.assertEqual(other.quant_method, "fp8")
        self.assertEqual(other.quant_config_hash, hash_quant_config(quant_config))
        self.assertEqual(other.draft_model_idx, 1)
        # Change each field in isolation. Several inputs change the HF hash too.
        for field, value in msgspec.to_builtins(other).items():
            with self.subTest(field=field):
                self.assertNotEqual(getattr(base, field), value)
                changed = msgspec.structs.replace(base, **{field: value})
                self.assertNotEqual(base.digest(), changed.digest())

    def test_staged_checkpoint_keeps_the_model(self):
        base = _make_weight_cache_config()
        staged = _make_weight_cache_config(
            hf_config=PretrainedConfig(name_or_path="/tmp/staging"),
            model_path="/tmp/staging",
            revision="release-alias",
        )
        self.assertEqual(base, staged)
        overridden = _make_weight_cache_config(
            hf_config=PretrainedConfig(hidden_size=128)
        )
        self.assertNotEqual(base.hf_config_hash, overridden.hf_config_hash)

    def test_checkpoint_id_is_required(self):
        with self.assertRaisesRegex(ValueError, "checkpoint ID"):
            _make_weight_cache_config(checkpoint_id="")

    def test_daemon_checkpoint_id_names_the_requested_path_and_revision(self):
        for revision, expected in ((None, "/m"), ("v2", "/m@v2")):
            with get_context().override_server_args(model_path="/m", revision=revision):
                self.assertEqual(path_checkpoint_id(), expected)

    def test_load_spec_reads_the_runtime_context(self):
        spec = _make_load_spec(tp_size=4, tp_rank=3)
        self.assertEqual((spec.tp_size, spec.tp_rank), (4, 3))
        self.assertEqual((spec.pp_size, spec.pp_rank), (1, 0))

    def test_moe_backends_come_from_the_published_config(self):
        # A dense model's engine never initializes the MoE flags; the daemon does.
        spec = _make_load_spec(moe_runner_backend="triton")
        self.assertEqual(spec.moe_runner_backend, "triton")
        self.assertEqual(get_moe_runner_backend().value, "auto")

    def test_rank_changes_only_the_shard_digest(self):
        spec = _make_load_spec()
        for field in ("tp_rank", "pp_rank", "moe_dp_rank", "moe_ep_rank"):
            with self.subTest(field=field):
                peer = msgspec.structs.replace(spec, **{field: 1})
                self.assertNotEqual(spec.digest(), peer.digest())
                self.assertEqual(
                    spec.digest(include_rank=False), peer.digest(include_rank=False)
                )

    def test_shared_load_settings_change_both_digests(self):
        spec = _make_load_spec()
        for other in (
            _make_load_spec(tp_size=4),
            msgspec.structs.replace(spec, flashinfer_cutedsl_swap_ab=True),
            msgspec.structs.replace(spec, fp4_gemm_runner_backend="flashinfer_trtllm"),
        ):
            self.assertNotEqual(spec.digest(), other.digest())
            self.assertNotEqual(
                spec.digest(include_rank=False), other.digest(include_rank=False)
            )


class TestCacheConfigMessage(CustomTestCase):
    def test_identical_messages_match(self):
        self.assertEqual(cache_config_mismatches(_make_message(), _make_message()), {})

    def test_tag_uses_the_snapshot_identity(self):
        with _runtime_context(tp_size=8, tp_rank=3):
            spec = build_load_spec(device=torch.device("cpu"))
        model = _make_weight_cache_config()
        message = cache_config_message(model, spec)
        self.assertEqual(message["tag"], snapshot_tag(model, spec))
        self.assertNotEqual(
            message["tag"], snapshot_tag(model, spec, include_rank=False)
        )
        wrong = {**message, "tag": "tp=8"}
        self.assertEqual(
            cache_config_mismatches(message, wrong),
            {"tag": (message["tag"], wrong["tag"])},
        )

    def test_any_field_difference_is_reported(self):
        base = _make_message()
        for part, field, value in (
            ("model", "checkpoint_id", "/models/other"),
            ("model", "hf_config_hash", "changed"),
            ("model", "dtype", "torch.bfloat16"),
            ("model", "quant_method", "fp8"),
            ("model", "quant_config_hash", "changed"),
            ("spec", "tp_rank", 1),
            ("spec", "moe_ep_rank", 1),
            ("spec", "attn_dp_size", 2),
            ("spec", "moe_dense_tp_size", 1),
            ("spec", "moe_a2a_backend", "mooncake"),
            ("spec", "fp8_gemm_runner_backend", "cutlass"),
            ("spec", "device_capability", "0.0"),
            ("spec", "torch_version", "0.0.0"),
        ):
            with self.subTest(field=f"{part}.{field}"):
                other = {**base, part: {**base[part], field: value}}
                self.assertEqual(
                    cache_config_mismatches(base, other),
                    {f"{part}.{field}": (base[part][field], value)},
                )

    def test_missing_extra_and_legacy_fields_are_reported(self):
        base = _make_message()
        missing = {**base, "spec": dict(base["spec"])}
        del missing["spec"]["moe_dense_tp_size"]  # None on both sides by .get()
        self.assertEqual(
            cache_config_mismatches(base, missing),
            {"spec.moe_dense_tp_size": (None, None)},
        )
        extra = {**base, "model": {**base["model"], "new_field": 1}}
        self.assertEqual(
            cache_config_mismatches(base, extra), {"model.new_field": (None, 1)}
        )
        # The flat CacheConfig an older engine sends matches no field.
        legacy = {**base["model"], **base["spec"]}
        self.assertEqual(
            len(cache_config_mismatches(base, legacy)),
            len(base["model"]) + len(base["spec"]),
        )

    def test_daemon_serves_only_a_matching_engine(self):
        daemon = WeightCacheDaemon.__new__(WeightCacheDaemon)
        daemon.gpu_id = 0
        daemon.config = _make_message()
        daemon.state_entries = {}
        daemon.preloaded_weights_bytes = 0
        daemon.transport_backend = mock.Mock(name="torch_ipc")
        for engine_config, served in (
            (_make_message(tp_rank=1), False),
            (_make_message(), True),
        ):
            with self.subTest(served=served):
                a, b = socket.socketpair()
                try:
                    send_msg(b, {"type": "fetch_state", "config": engine_config})
                    daemon._handle_connection(a)
                    send = daemon.transport_backend.send_fetch_state_response
                    self.assertEqual(send.called, served)
                    if not served:
                        response = recv_msg(b)
                        self.assertEqual(response["status"], "mismatch")
                        self.assertEqual(response["daemon_config"], daemon.config)
                finally:
                    a.close()
                    b.close()


class TestEngineFetchesFromTheDaemon(CustomTestCase):
    """The engine's request and its handling of the reply, over a real socket."""

    def _fetch(self, daemon_config, *, handle_connection=None):
        daemon = WeightCacheDaemon.__new__(WeightCacheDaemon)
        daemon.gpu_id = 0
        daemon.config = daemon_config
        daemon.state_entries = {}
        daemon.preloaded_weights_bytes = 0
        daemon.transport_backend = TorchIpcTransportBackend()
        loader = WeightCacheModelLoader(LoadConfig(load_format=LoadFormat.IPC_CACHE))
        with tempfile.TemporaryDirectory() as tmp:
            loader.socket_path = os.path.join(tmp, "daemon.sock")
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(loader.socket_path)
            server.listen(1)
            server.settimeout(10)  # a fetch that never connects ends the thread

            def serve():
                conn, _ = server.accept()
                with conn:
                    (handle_connection or daemon._handle_connection)(conn)

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            try:
                with _runtime_context():
                    return loader._fetch_from_cache(
                        _model_config(), SimpleNamespace(device=torch.device("cpu"))
                    )
            finally:
                thread.join(10)
                server.close()

    def _daemon_config(self, **spec_overrides):
        with _runtime_context():
            model = build_weight_cache_config(
                _model_config(), checkpoint_id=path_checkpoint_id()
            )
        spec = msgspec.structs.replace(_make_load_spec(), **spec_overrides)
        return cache_config_message(model, spec)

    def test_a_matching_daemon_serves_the_engine(self):
        result = self._fetch(self._daemon_config())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["config"], self._daemon_config())

    def test_a_mismatch_raises_with_daemon_and_engine_values(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._fetch(self._daemon_config(tp_rank=1))
        self.assertIn("Daemon config mismatch", str(ctx.exception))
        self.assertIn("'spec.tp_rank': (1, 0)", str(ctx.exception))

    def test_rejects_a_wrong_rank_reply_before_receiving_storage(self):
        wrong_rank = self._daemon_config(tp_rank=1)

        def serve_wrong_rank(conn):
            recv_msg(conn)
            TorchIpcTransportBackend().send_fetch_state_response(
                conn, config=wrong_rank, entries={}, pid=1
            )

        with mock.patch.object(
            TorchIpcTransportBackend, "recv_fetch_state_response"
        ) as receive:
            with self.assertRaisesRegex(RuntimeError, "spec.tp_rank"):
                self._fetch(wrong_rank, handle_connection=serve_wrong_rank)
        receive.assert_not_called()


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


if __name__ == "__main__":
    unittest.main()
