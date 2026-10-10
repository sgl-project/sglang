# SPDX-License-Identifier: Apache-2.0
"""Unit and contract tests for weight-cache heterogeneous transfer.

Cover recorder placement, manifests, Mooncake planning, the TCP registry, and
daemon/IPC coordination using small Torch modules and mocked transfer calls.
"""

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

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
        import importlib.util
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[2]
            / "manual/test_weight_cache_state_compare.py"
        )
        spec = importlib.util.spec_from_file_location(
            "weight_cache_state_compare", path
        )
        comparison = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparison)
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
            self.assertEqual(comparison.compare_rank(0, 1, 0), (3, 8))
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
        for mode in ("source", "target", None):
            with self.subTest(mode=mode):
                daemon_args = WeightCacheDaemonArgs()
                if mode is not None:
                    daemon_args = WeightCacheDaemonArgs(
                        weight_heterogeneous_transfer_mode=mode,
                        weight_heterogeneous_transfer_host="host",
                        weight_heterogeneous_transfer_registry_url=("tcp://host:31999"),
                    )
                daemon = WeightCacheDaemon(
                    server_args=_daemon_server_args(tp_size=2),
                    gpu_id=4,
                    tp_rank=0,
                    pp_rank=0,
                    daemon_args=daemon_args,
                )
                self.assertEqual(daemon.socket_path, get_socket_path("GPU-test-4"))
                self.assertEqual(daemon.ready_path, get_ready_path("GPU-test-4"))

        self.assertEqual(get_device_uuid.call_count, 3)
        get_device_uuid.assert_any_call(4)

    def test_weight_heterogeneous_transfer_mode_contract(self):
        disabled = WeightCacheDaemonArgs()
        source = WeightCacheDaemonArgs(
            weight_heterogeneous_transfer_mode="source",
            weight_heterogeneous_transfer_host="0.0.0.0",
        )
        target = WeightCacheDaemonArgs(
            weight_heterogeneous_transfer_mode="target",
            weight_heterogeneous_transfer_host="source",
        )
        self.assertIsNone(disabled.weight_heterogeneous_transfer_mode)
        self.assertEqual(source.weight_heterogeneous_transfer_mode, "source")
        self.assertEqual(target.weight_heterogeneous_transfer_mode, "target")

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
                    "31999",
                ]
            )
        )
        self.assertIsNone(disabled.weight_heterogeneous_transfer_mode)
        self.assertEqual(source.weight_heterogeneous_transfer_mode, "source")
        self.assertEqual(target.weight_heterogeneous_transfer_mode, "target")

    def test_daemon_socket_does_not_serve_weight_manifests(self):
        daemon = object.__new__(WeightCacheDaemon)
        responses = []
        with (
            patch(
                "sglang.srt.weight_cache.daemon.recv_msg",
                return_value={"type": "query_weight_manifest"},
            ),
            patch(
                "sglang.srt.weight_cache.daemon.send_msg",
                side_effect=lambda _conn, response: responses.append(response),
            ),
        ):
            daemon._handle_connection(MagicMock())

        self.assertEqual(
            responses,
            [
                {
                    "status": "error",
                    "message": "Unknown request type: query_weight_manifest",
                }
            ],
        )

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

    def test_resolved_registry_url_does_not_change_mode(self):
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
                self.assertEqual(
                    daemon_args.weight_heterogeneous_transfer_mode,
                    mode,
                )

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
        self.assertEqual(
            process.join.call_args_list,
            [call(timeout=5), call()],
        )
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
