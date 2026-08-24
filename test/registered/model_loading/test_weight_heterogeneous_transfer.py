# SPDX-License-Identifier: Apache-2.0

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    maybe_enable_ipc_weight_cache,
)
from sglang.srt.weight_cache.mooncake_weight_adapter import (
    build_mooncake_weight_manifests,
)
from sglang.srt.weight_cache.weight_runtime_manifest import (
    IMMUTABLE_WEIGHT_GENERATION,
    ImmutableWeightRuntimeManifestBuilder,
    LogicalTensorView,
    WeightParallelTopology,
    model_identity_from_config,
)
from sglang.srt.weight_cache.daemon import (
    WeightCacheDaemon,
    launch_weight_cache_daemons,
)
from sglang.srt.weight_cache.weight_heterogeneous_transfer import (
    _initialize_weight_transfer_engine,
    _resolve_active_moe_runner_backend,
    fetch_source_weights_manifest,
    transfer_weights_from_source_daemons,
    WeightHeterogeneousTransferError,
    WeightParallelLayout,
    SourceWeightsManifest,
    validate_weight_heterogeneous_transfer_configuration,
)
from sglang.srt.weight_cache.weight_manifest_server import WeightManifestServer
from sglang.srt.weight_cache.protocol import get_ready_path, get_socket_path
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestWeightHeterogeneousTransfer(unittest.TestCase):
    def test_model_identity_ignores_local_path_and_runtime_metadata(self):
        class Config:
            def __init__(self, path, hidden_size=128):
                self.model_type = "demo"
                self.path = path
                self.hidden_size = hidden_size

            def to_dict(self):
                return {
                    "model_type": self.model_type,
                    "hidden_size": self.hidden_size,
                    "_name_or_path": self.path,
                    "_commit_hash": "local-commit",
                    "transformers_version": "local-version",
                    "text_config": {
                        "hidden_size": self.hidden_size,
                        "layer_types": {0: "attention"},
                    },
                }

        source = model_identity_from_config(Config("/source/model"))
        target = model_identity_from_config(Config("/target/model"))
        incompatible = model_identity_from_config(Config("/target/model", 256))

        self.assertEqual(source, target)
        self.assertNotEqual(source, incompatible)

    def test_transfer_engine_uses_normal_mooncake_endpoint_discovery(self):
        engine = MagicMock()
        engine.initialize.return_value = 0
        engine.get_rpc_port.return_value = 12345

        with (
            patch("mooncake.engine.TransferEngine", return_value=engine),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.get_local_ip_auto",
                return_value="10.0.0.8",
            ),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.envs.MOONCAKE_PROTOCOL.get",
                return_value="rdma",
            ),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.envs.MOONCAKE_DEVICE.get",
                return_value="mlx5_0",
            ),
        ):
            actual_engine, endpoint = _initialize_weight_transfer_engine()

        self.assertIs(actual_engine, engine)
        self.assertEqual(endpoint, "10.0.0.8:12345")
        engine.initialize.assert_called_once_with(
            "10.0.0.8", "P2PHANDSHAKE", "rdma", "mlx5_0"
        )

    def test_daemon_manifest_adapts_to_immutable_runtime_binding(self):
        class FullTensorAdapter:
            @staticmethod
            def describe_parameter(*, names, parameter, topology):
                del topology
                shape = tuple(parameter.shape)
                return (
                    LogicalTensorView(
                        tensor_id=names[0],
                        global_shape=shape,
                        global_offset=(0,) * len(shape),
                        local_shape=shape,
                        partition_dim=None,
                        byte_offset=0,
                        layer_id=None,
                        expert_id=None,
                        layout_fingerprint="contiguous",
                    ),
                )

        model = torch.nn.Linear(2, 2, bias=False)
        manifest = ImmutableWeightRuntimeManifestBuilder(
            model=model,
            adapter=FullTensorAdapter(),
            topology=WeightParallelTopology(),
            allowed_devices=("cpu",),
        ).build(
            model_id="model",
            revision="revision",
            instance_id="worker",
            worker_id="worker",
            endpoint="127.0.0.1:1",
        )

        self.assertEqual(manifest.generation, IMMUTABLE_WEIGHT_GENERATION)
        self.assertFalse(hasattr(manifest, "lease_id"))
        placement, bindings = build_mooncake_weight_manifests(
            (manifest,),
            placement_set_id="test",
            tp_size=1,
            pp_size=1,
            ep_size=1,
        )
        self.assertEqual(bindings[0].generation, IMMUTABLE_WEIGHT_GENERATION)
        self.assertEqual(bindings[0].placement_id, placement.placement_id)
        self.assertEqual(bindings[0].lease_id, "immutable:worker")

    def test_active_moe_runner_backend_overrides_auto_config(self):
        backend = SimpleNamespace(value="triton")
        quant_method = SimpleNamespace(
            runner=SimpleNamespace(runner_backend=backend),
            _aiter_runner=None,
        )
        model = SimpleNamespace(
            modules=lambda: (SimpleNamespace(quant_method=quant_method),)
        )
        self.assertEqual(_resolve_active_moe_runner_backend(model, "auto"), "triton")

    def test_weight_parallel_layout_supports_pp_and_ep_but_not_dp(self):
        validate_weight_heterogeneous_transfer_configuration(
            tp_size=8,
            dp_size=1,
            ep_size=4,
            pp_size=2,
            quantization=None,
        )
        parallel_layout = WeightParallelLayout(tp_size=4, pp_size=2, ep_size=2)
        self.assertEqual(parallel_layout.world_size, 8)
        self.assertEqual(
            tuple(parallel_layout.iter_ranks()),
            tuple((pp, tp) for pp in range(2) for tp in range(4)),
        )
        with self.assertRaises(WeightHeterogeneousTransferError):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=2,
                dp_size=2,
                ep_size=1,
                pp_size=1,
                quantization=None,
            )
        with self.assertRaises(WeightHeterogeneousTransferError):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=2,
                dp_size=1,
                ep_size=1,
                pp_size=1,
                quantization="fp8",
            )

    def test_manifest_server_aggregates_source_ranks(self):
        server = object.__new__(WeightManifestServer)
        server.expected_rank_count = 2
        server._node_id = None
        server._parallel_layout = None
        server._model_identity = None
        server._rank_manifests = {}
        server._lock = threading.Lock()

        for global_rank, gpu_id in ((0, 2), (1, 3)):
            server.register_weight_manifest(
                {
                    "node_id": "source-node",
                    "global_rank": global_rank,
                    "gpu_id": gpu_id,
                    "parallel_layout": {
                        "tp_size": 2,
                        "pp_size": 1,
                        "ep_size": 1,
                    },
                    "runtime_inventory": {
                        "model_id": "model",
                        "revision": "revision",
                        "rank": global_rank,
                    },
                    "content_checksums": [{"rank": global_rank}],
                }
            )

        source = SourceWeightsManifest.from_wire(server.get_source_weights_manifest())
        self.assertEqual(source.node_id, "source-node")
        self.assertEqual(source.gpu_ids, (2, 3))
        self.assertEqual(
            tuple(item["rank"] for item in source.runtime_inventories), (0, 1)
        )

    def test_target_reads_complete_source_weights_manifest(self):
        expected = SourceWeightsManifest(
            node_id="source-node",
            parallel_layout=WeightParallelLayout(tp_size=2, pp_size=1, ep_size=1),
            gpu_ids=(0, 1),
            runtime_inventories=({"rank": 0}, {"rank": 1}),
            content_checksum_groups=(({"rank": 0},), ({"rank": 1},)),
        )
        response = MagicMock(status_code=200)
        response.json.return_value = {"source_weights_manifest": expected.to_wire()}
        with patch("requests.get", return_value=response) as get:
            actual = fetch_source_weights_manifest("http://source:31999")

        get.assert_called_once_with(
            "http://source:31999/get_source_weights_manifest", timeout=5
        )
        self.assertEqual(actual, expected)

    def test_target_rejects_overlap_after_fetching_source_manifest(self):
        source = SourceWeightsManifest(
            node_id="shared-node",
            parallel_layout=WeightParallelLayout(1),
            gpu_ids=(0,),
            runtime_inventories=({"rank": 0},),
            content_checksum_groups=(({"rank": 0},),),
        )

        def all_gather(output, _local):
            output[:] = [0]

        with (
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.fetch_source_weights_manifest",
                return_value=source,
            ) as fetch,
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.broadcast_object_list"),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.all_gather_object", side_effect=all_gather),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.get_local_ip_auto",
                return_value="shared-node",
            ),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.pull_weights_from_source"
            ) as pull,
        ):
            with self.assertRaisesRegex(
                WeightHeterogeneousTransferError, "overlapping ranks"
            ):
                transfer_weights_from_source_daemons(
                    target_manifest_state=MagicMock(),
                    model=MagicMock(),
                    gpu_id=0,
                    registry_url="http://source:31999",
                )

        fetch.assert_called_once_with("http://source:31999")
        pull.assert_not_called()

    def test_target_allows_same_gpu_id_on_another_node(self):
        source = SourceWeightsManifest(
            node_id="source-node",
            parallel_layout=WeightParallelLayout(1),
            gpu_ids=(0,),
            runtime_inventories=({"rank": 0},),
            content_checksum_groups=(({"rank": 0},),),
        )

        def all_gather(output, _local):
            output[:] = [0]

        with (
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.fetch_source_weights_manifest",
                return_value=source,
            ),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.broadcast_object_list"),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.all_gather_object", side_effect=all_gather),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.get_local_ip_auto",
                return_value="target-node",
            ),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.pull_weights_from_source",
                return_value={},
            ) as pull,
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.rebuild_transferred_weight_state"
            ),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.validate_transferred_weights",
                return_value={},
            ),
        ):
            transfer_weights_from_source_daemons(
                target_manifest_state=MagicMock(),
                model=MagicMock(),
                gpu_id=0,
                registry_url="http://source:31999",
            )

        pull.assert_called_once()

    def test_heterogeneous_socket_paths_use_physical_gpu_rank(self):
        daemon = WeightCacheDaemon(
            model_path="/models/demo",
            gpu_id=4,
            tp_size=2,
            tp_rank=0,
            enable_weight_heterogeneous_copy=True,
            weight_heterogeneous_transfer_registry_url="http://source:31999",
        )
        self.assertEqual(daemon.socket_path, get_socket_path(4))
        self.assertEqual(daemon.ready_path, get_ready_path(4))

    def test_client_uses_physical_socket_without_heterogeneous_server_args(self):
        client_load_config = SimpleNamespace(
            load_format="auto", weight_cache_socket=None
        )
        maybe_enable_ipc_weight_cache(
            load_config=client_load_config,
            server_args=SimpleNamespace(weight_cache_mode="client"),
            tp_size=2,
            pp_rank=0,
            tp_rank=0,
            gpu_id=4,
        )
        self.assertEqual(client_load_config.weight_cache_socket, get_socket_path(4))

        daemon_load_config = SimpleNamespace(
            load_format="auto", weight_cache_socket=None
        )
        maybe_enable_ipc_weight_cache(
            load_config=daemon_load_config,
            server_args=SimpleNamespace(weight_cache_mode="daemon"),
            tp_size=2,
            pp_rank=0,
            tp_rank=0,
            gpu_id=4,
        )
        self.assertEqual(daemon_load_config.weight_cache_socket, get_socket_path(0))

    def test_source_and_target_gates_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            WeightCacheDaemon(
                model_path="/models/demo",
                gpu_id=0,
                enable_weight_heterogeneous_copy=True,
                weight_heterogeneous_copy=True,
                weight_heterogeneous_transfer_registry_url="http://source:31999",
            )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            launch_weight_cache_daemons(
                model_path="/models/demo",
                enable_weight_heterogeneous_copy=True,
                weight_heterogeneous_copy=True,
            )

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

    def test_target_launcher_passes_manifest_registry_url(self):
        process = MagicMock(pid=123, returncode=0)
        process.poll.return_value = 0
        commands = []

        def spawn(command):
            commands.append(command)
            return process

        with (
            patch(
                "sglang.srt.weight_cache.daemon.cleanup_stale_daemon_files"
            ) as cleanup,
            patch("sglang.srt.weight_cache.daemon.os.path.exists", return_value=True),
            patch("subprocess.Popen", side_effect=spawn),
        ):
            with self.assertRaisesRegex(RuntimeError, "exited with code 0"):
                launch_weight_cache_daemons(
                    model_path="/models/demo",
                    tp_size=1,
                    base_gpu_id=4,
                    weight_heterogeneous_copy=True,
                    weight_heterogeneous_transfer_source_ip="source",
                    weight_heterogeneous_transfer_source_port=31999,
                    timeout=1,
                )

        cleanup.assert_called_once_with(4, force=False)
        self.assertIn("--weight-cache-socket-rank", commands[0])
        self.assertIn("4", commands[0])
        self.assertIn("--weight-heterogeneous-copy", commands[0])
        self.assertIn("--weight-heterogeneous-transfer-registry-url", commands[0])
        self.assertIn("http://source:31999", commands[0])
        self.assertNotIn("--cache-id", commands[0])


if __name__ == "__main__":
    unittest.main()
