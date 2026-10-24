# SPDX-License-Identifier: Apache-2.0
"""Daemon configuration, distributed topology checks, IPC rebinding and cleanup."""

import argparse
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.weight_cache.daemon import (
    WeightCacheDaemon,
    WeightCacheDaemonArgs,
    _prepare_weight_heterogeneous_transfer,
    launch_weight_cache_daemons,
)
from sglang.srt.weight_cache.ipc_loader import IpcModelLoader
from sglang.srt.weight_cache.protocol import get_ready_path, get_socket_path
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _daemon_server_args(**overrides):
    values = {
        "model_path": "/models/demo",
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "attn_dp_size": 1,
        "ep_size": 1,
        "moe_dp_size": 1,
        "enable_dp_attention": False,
        "enable_dp_lm_head": False,
        "attn_cp_size": 1,
        "moe_dense_tp_size": 1,
        "moe_a2a_backend": "none",
        "deepep_mode": "auto",
        "load_format": "auto",
        "dtype": "auto",
        "quantization": None,
        "model_loader_extra_config": {},
        "trust_remote_code": False,
        "revision": None,
        "nnodes": 1,
        "node_rank": 0,
        "base_gpu_id": 0,
        "gpu_id_step": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestWeightHeterogeneousTransfer(unittest.TestCase):
    def test_attention_dp_uses_resolved_size_without_legacy_flag(self):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(model_path="test", tp_size=2, attn_dp_size=2)
        daemon_args = WeightCacheDaemonArgs(
            weight_heterogeneous_transfer_mode="target",
            weight_heterogeneous_transfer_registry_url="tcp://source:1234",
            weight_heterogeneous_transfer_host="source",
        )
        module = "sglang.srt.weight_cache.weight_heterogeneous_transfer"
        from sglang.srt.weight_cache.weight_heterogeneous_transfer import (
            validate_weight_heterogeneous_transfer_configuration,
        )

        with patch(
            f"{module}.validate_weight_heterogeneous_transfer_configuration",
            wraps=validate_weight_heterogeneous_transfer_configuration,
        ) as validate:
            _prepare_weight_heterogeneous_transfer(
                args, daemon_args, expected_rank_count=2
            )
        self.assertEqual(validate.call_args.kwargs["attn_dp_size"], 2)

    def test_attention_dp_runtime_topology_matches_resolved_size(self):
        from contextlib import ExitStack

        from sglang.srt.weight_cache.weight_heterogeneous_transfer import (
            WeightHeterogeneousTransferError,
            _build_weight_runtime_manifest,
        )

        module = "sglang.srt.weight_cache.weight_heterogeneous_transfer"
        with ExitStack() as stack:
            for axis, size in (
                ("tensor", 2),
                ("moe_tensor", 2),
                ("moe_expert", 1),
                ("pipeline", 1),
            ):
                for suffix, value in (("rank", 0), ("world_size", size)):
                    stack.enter_context(
                        patch(
                            f"sglang.srt.distributed.parallel_state.get_{axis}_model_parallel_{suffix}"
                            if axis in ("tensor", "pipeline")
                            else f"sglang.srt.distributed.parallel_state.get_{axis}_parallel_{suffix}",
                            return_value=value,
                        )
                    )
            stack.enter_context(
                patch(
                    "sglang.srt.runtime_context.get_parallel",
                    return_value=SimpleNamespace(
                        attn_dp_size=2,
                        moe_dp_size=1,
                        attn_dp_rank=0,
                    ),
                )
            )
            builder = stack.enter_context(
                patch(f"{module}.ImmutableWeightRuntimeManifestBuilder")
            )
            kwargs = dict(
                model=object(),
                tp_size=2,
                tp_rank=0,
                pp_size=1,
                pp_rank=0,
                ep_size=1,
                attn_dp_size=2,
                gpu_id=0,
                transfer_endpoint="test",
                model_id="model",
                revision="test",
                layout_plan=object(),
            )
            _build_weight_runtime_manifest(**kwargs)
            builder.reset_mock()
            with self.assertRaisesRegex(
                WeightHeterogeneousTransferError,
                "distributed parallel topology differs",
            ):
                _build_weight_runtime_manifest(**dict(kwargs, attn_dp_size=1))
            builder.assert_not_called()

    def test_ipc_mapping_rebinds_parameter_references_in_runtime_configs(self):
        from sglang.srt.layers.moe.topk import TopKConfig

        @dataclass(frozen=True)
        class TensorRef:
            value: torch.Tensor

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.empty(2, device="meta"))
                self.register_buffer("scale", torch.empty(2, device="meta"))
                self.topk = torch.nn.Module()
                self.topk.config = TopKConfig(top_k=1, correction_bias=self.bias)
                self.topk.nested = TensorRef(self.scale)
                self.scale_alias = self.scale
                self.unrelated = torch.empty(2, device="meta")

        model = Model()
        unrelated = model.unrelated
        entries = {
            "bias": {"tensor": torch.tensor([1.0, 2.0]), "is_param": True},
            "scale": {"tensor": torch.tensor([3.0, 4.0]), "is_param": False},
            # An explicitly exported derived value takes precedence over its
            # constructor-time alias relationship.
            "scale_alias": {
                "tensor": torch.tensor([5.0, 6.0]),
                "is_plain_tensor": True,
            },
        }
        loader = IpcModelLoader.__new__(IpcModelLoader)
        loader.load_config = None
        loader._transport_backend = SimpleNamespace(import_tensor=lambda e: e["tensor"])
        with patch(
            "sglang.srt.weight_cache.ipc_loader._initialize_model", return_value=model
        ):
            result = loader._load_zero_copy_mode(
                SimpleNamespace(dtype=torch.float32), None, entries, None
            )
        self.assertIs(result.topk.config.correction_bias, result.bias)
        self.assertEqual(result.topk.config.correction_bias.device.type, "cpu")
        self.assertIs(result.topk.nested.value, result.scale)
        self.assertTrue(
            torch.equal(result.scale_alias, entries["scale_alias"]["tensor"])
        )
        self.assertIs(result.unrelated, unrelated)
        self.assertEqual(result.unrelated.device.type, "meta")

    def test_daemon_exports_exact_plain_derived_tensor_for_ipc_client(self):
        class DerivedLayer(torch.nn.Module):
            def __init__(self, derived):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2))
                self.w_kc = derived
                self.w_scale = 1.0
                self.use_deep_gemm_bmm = derived is not None

        class Model(torch.nn.Module):
            def __init__(self, derived):
                super().__init__()
                self.layer = DerivedLayer(derived)

        derived = torch.arange(24).reshape(2, 3, 4).transpose(1, 2)
        daemon = object.__new__(WeightCacheDaemon)
        daemon.model = Model(derived)
        daemon.gpu_id = 0
        daemon.state_entries = {}

        captured = {}
        backend = MagicMock()
        backend.name = "test"

        def prepare_export(state_tensors):
            captured.update(state_tensors)
            return {
                name: {"handle": b"", "is_param": is_param}
                for name, (_, is_param) in state_tensors.items()
            }

        backend.prepare_export.side_effect = prepare_export
        with patch(
            "sglang.srt.weight_cache.daemon.choose_daemon_transport_backend",
            return_value=backend,
        ):
            daemon._export_state()

        exported, is_param = captured["layer.w_kc"]
        self.assertFalse(is_param)
        self.assertEqual(exported.data_ptr(), derived.data_ptr())
        self.assertEqual(exported.stride(), derived.stride())
        self.assertTrue(daemon.state_entries["layer.w_kc"]["is_plain_tensor"])
        self.assertNotIn("layer.w_scale", captured)
        self.assertEqual(daemon.state_entries["layer.w_scale"]["value"], 1.0)
        self.assertTrue(daemon.state_entries["layer.use_deep_gemm_bmm"]["value"])

        client = Model(None)
        imported = torch.full(derived.shape, 7, dtype=derived.dtype)
        IpcModelLoader._set_plain_module_tensor(client, "layer.w_kc", imported)
        self.assertIs(client.layer.w_kc, imported)
        self.assertNotIn("w_kc", client.layer._parameters)
        self.assertNotIn("w_kc", client.layer._buffers)
        imported_scale = torch.tensor(3.0)
        IpcModelLoader._set_plain_module_tensor(client, "layer.w_scale", imported_scale)
        self.assertIs(client.layer.w_scale, imported_scale)
        IpcModelLoader._set_plain_module_value(client, "layer.use_deep_gemm_bmm", True)
        self.assertTrue(client.layer.use_deep_gemm_bmm)

    def test_state_compare_checks_plain_derived_values(self):
        from sglang.test import weight_cache_state_compare as comparison

        tensor = torch.ones(2)
        entries = {
            "weight": {
                "shape": (2,),
                "dtype": "float32",
                "is_param": True,
                "tensor": tensor,
            },
            "use_deep_gemm_bmm": {"is_plain_value": True, "value": False},
            "w_scale": {"is_plain_value": True, "value": 1.0},
        }
        backend = MagicMock()
        backend.import_tensor.side_effect = lambda entry: entry["tensor"]
        with patch.object(comparison, "fetch_state", return_value=(entries, backend)):
            self.assertEqual(
                comparison.compare_rank(0, 1, 0),
                (len(entries), tensor.numel() * tensor.element_size()),
            )
        self.assertEqual(backend.import_tensor.call_count, 2)
        for changed in (True, 0):
            right = {
                **entries,
                "use_deep_gemm_bmm": {"is_plain_value": True, "value": changed},
            }
            with (
                self.subTest(changed=changed),
                patch.object(
                    comparison,
                    "fetch_state",
                    side_effect=((entries, backend), (right, backend)),
                ),
                self.assertRaisesRegex(AssertionError, "derived values differ"),
            ):
                comparison.compare_rank(0, 1, 0)

    @patch(
        "sglang.srt.weight_cache.daemon.current_platform.get_device_uuid",
        return_value="GPU-test-4",
    )
    def test_socket_paths_use_physical_device_uuid(self, get_device_uuid):
        daemon = WeightCacheDaemon(
            server_args=_daemon_server_args(tp_size=2),
            gpu_id=4,
            tp_rank=0,
            pp_rank=0,
        )
        self.assertEqual(daemon.socket_path, get_socket_path("GPU-test-4"))
        self.assertEqual(daemon.ready_path, get_ready_path("GPU-test-4"))
        get_device_uuid.assert_called_once_with(4)

    def test_weight_heterogeneous_transfer_mode_contract(self):
        invalid_cases = (
            (
                {
                    "weight_heterogeneous_transfer_mode": "invalid",
                    "weight_heterogeneous_transfer_host": "host",
                },
                "mode must be",
            ),
            (
                {
                    "weight_heterogeneous_transfer_host": "host",
                },
                "require an active mode",
            ),
            (
                {
                    "weight_heterogeneous_transfer_mode": "source",
                },
                "requires a host",
            ),
            (
                {
                    "weight_heterogeneous_transfer_mode": "target",
                    "weight_heterogeneous_transfer_host": "",
                },
                "requires a host",
            ),
            (
                {
                    "weight_heterogeneous_transfer_mode": "target",
                    "weight_heterogeneous_transfer_host": "source",
                    "weight_heterogeneous_transfer_port": 0,
                },
                "port must be positive",
            ),
        )
        for kwargs, error in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, error):
                    WeightCacheDaemonArgs(**kwargs)

    def test_weight_heterogeneous_transfer_cli_args(self):
        parser = argparse.ArgumentParser()
        WeightCacheDaemonArgs.add_cli_args(parser)

        disabled = WeightCacheDaemonArgs.from_cli_args(parser.parse_args([]))
        source = WeightCacheDaemonArgs.from_cli_args(
            parser.parse_args(
                [
                    "--weight-heterogeneous-transfer-mode",
                    "source",
                    "--weight-heterogeneous-transfer-host",
                    "0.0.0.0",
                ]
            )
        )
        target = WeightCacheDaemonArgs.from_cli_args(
            parser.parse_args(
                [
                    "--weight-heterogeneous-transfer-mode",
                    "target",
                    "--weight-heterogeneous-transfer-host",
                    "source",
                    "--weight-heterogeneous-transfer-port",
                    "43210",
                ]
            )
        )
        self.assertIsNone(disabled.weight_heterogeneous_transfer_mode)
        self.assertEqual(source.weight_heterogeneous_transfer_mode, "source")
        self.assertEqual(target.weight_heterogeneous_transfer_mode, "target")
        self.assertEqual(source.weight_heterogeneous_transfer_host, "0.0.0.0")
        self.assertEqual(target.weight_heterogeneous_transfer_host, "source")
        self.assertEqual(target.weight_heterogeneous_transfer_port, 43210)

    def test_source_registry_url_uses_actual_bind_host_and_port(self):
        cases = (
            ("0.0.0.0", "tcp://127.0.0.1:43210"),
            ("::", "tcp://[::1]:43210"),
            ("10.0.0.8", "tcp://10.0.0.8:43210"),
            ("::1", "tcp://[::1]:43210"),
        )
        for bind_host, expected_url in cases:
            with self.subTest(bind_host=bind_host):
                manifest_server = MagicMock()
                manifest_server.port = 43210
                with patch(
                    "sglang.srt.weight_cache.weight_manifest_server.WeightManifestServer",
                    return_value=manifest_server,
                ):
                    actual_server, registry_url = (
                        _prepare_weight_heterogeneous_transfer(
                            _daemon_server_args(),
                            WeightCacheDaemonArgs(
                                weight_heterogeneous_transfer_mode="source",
                                weight_heterogeneous_transfer_host=bind_host,
                                weight_heterogeneous_transfer_port=31999,
                            ),
                            expected_rank_count=1,
                        )
                    )
                self.assertIs(actual_server, manifest_server)
                self.assertEqual(registry_url, expected_url)

    def test_resolved_registry_url_takes_precedence_over_host_and_port(self):
        for mode in ("source", "target"):
            with self.subTest(mode=mode):
                daemon_args = WeightCacheDaemonArgs(
                    weight_heterogeneous_transfer_mode=mode,
                    weight_heterogeneous_transfer_host="host",
                    weight_heterogeneous_transfer_registry_url=("tcp://resolved:31999"),
                )
                manifest_server, registry_url = _prepare_weight_heterogeneous_transfer(
                    _daemon_server_args(),
                    daemon_args,
                    expected_rank_count=1,
                )
                self.assertIsNone(manifest_server)
                self.assertEqual(registry_url, "tcp://resolved:31999")

    def test_launcher_cleans_up_after_partial_spawn_failure(self):
        manifest_server = MagicMock()
        process = MagicMock(pid=123)
        process.is_alive.side_effect = (True, True)

        with (
            patch(
                "sglang.srt.weight_cache.daemon._prepare_weight_heterogeneous_transfer",
                return_value=(manifest_server, "tcp://source:31999"),
            ),
            patch("sglang.srt.weight_cache.daemon.cleanup_stale_daemon_files"),
            patch(
                "sglang.srt.weight_cache.daemon.spawn_weight_cache_daemon",
                side_effect=(process, RuntimeError("second spawn failed")),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "second spawn failed"):
                launch_weight_cache_daemons(
                    _daemon_server_args(tp_size=2),
                    dist_init_method="tcp://127.0.0.1:12345",
                    daemon_args=WeightCacheDaemonArgs(
                        weight_heterogeneous_transfer_mode="source",
                        weight_heterogeneous_transfer_host="0.0.0.0",
                    ),
                )

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        process.join.assert_called()
        manifest_server.close.assert_called_once_with()

    def test_target_launcher_passes_manifest_registry_url(self):
        process = MagicMock(pid=123, exitcode=0)
        process.is_alive.return_value = False

        with (
            patch(
                "sglang.srt.weight_cache.daemon.cleanup_stale_daemon_files"
            ) as cleanup,
            patch(
                "sglang.srt.weight_cache.daemon.current_platform.get_device_uuid",
                return_value="GPU-test-4",
            ),
            patch("sglang.srt.weight_cache.daemon.os.path.exists", return_value=True),
            patch(
                "sglang.srt.weight_cache.daemon.spawn_weight_cache_daemon",
                return_value=process,
            ) as spawn,
        ):
            with self.assertRaisesRegex(RuntimeError, "exited with code 0"):
                launch_weight_cache_daemons(
                    _daemon_server_args(base_gpu_id=4),
                    timeout=1,
                    daemon_args=WeightCacheDaemonArgs(
                        weight_heterogeneous_transfer_mode="target",
                        weight_heterogeneous_transfer_host="source",
                        weight_heterogeneous_transfer_port=31999,
                    ),
                )

        cleanup.assert_called_once_with("GPU-test-4", force=False)
        spawn.assert_called_once()
        call = spawn.call_args
        self.assertEqual(call.kwargs["gpu_id"], 4)
        child_args = call.kwargs["daemon_args"]
        self.assertEqual(
            child_args.weight_heterogeneous_transfer_mode,
            "target",
        )
        self.assertEqual(
            child_args.weight_heterogeneous_transfer_host,
            "source",
        )
        self.assertEqual(child_args.weight_heterogeneous_transfer_port, 31999)
        self.assertEqual(
            child_args.weight_heterogeneous_transfer_registry_url,
            "tcp://source:31999",
        )


if __name__ == "__main__":
    unittest.main()
