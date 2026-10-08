# SPDX-License-Identifier: Apache-2.0
"""Unit and contract tests for weight-cache heterogeneous transfer.

Cover recorder placement, manifests, Mooncake planning, the TCP registry, and
daemon/IPC coordination using small Torch modules and mocked transfer calls.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.weight_cache.mooncake_weight_adapter import (
    _parallel_axes,
    build_mooncake_placement_and_bindings,
    build_mooncake_placement_groups,
    global_tp_tensor_ids,
)
from sglang.srt.weight_cache.weight_heterogeneous_transfer import (
    SourceDaemonGroup,
    WeightHeterogeneousTransferError,
    WeightParallelLayout,
    _all_gather_rank_local_phase,
    _initialize_weight_transfer_engine,
    fetch_weight_manifest,
    parse_source_daemon_group,
    pull_weights_from_source,
    register_weight_manifest,
    transfer_weights_from_source_daemons,
    validate_weight_heterogeneous_transfer_configuration,
)
from sglang.srt.weight_cache.weight_load_recorder import (
    WeightLoadRecorder,
    record_target_weight_load_plan,
)
from sglang.srt.weight_cache.weight_manifest_server import WeightManifestServer
from sglang.srt.weight_cache.weight_runtime_manifest import (
    IMMUTABLE_WEIGHT_GENERATION,
    ImmutableWeightRuntimeManifestBuilder,
    WeightParallelTopology,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestMooncakeWeightAdapter(unittest.TestCase):
    def test_scalar_checkpoint_values_survive_manifest_replay(self):
        import msgspec

        from sglang.srt.weight_cache.weight_load_recorder import (
            logical_weight_metadata_from_runtime_manifests,
        )

        class ScalarModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.zeros(1))
                self.weight = torch.nn.Parameter(torch.zeros(2, 2))
                self.transpose = False

            def load_weights(self, weights):
                for name, loaded in weights:
                    if name == "scale":
                        self.transpose = loaded.item() > 0
                        self.scale.data.fill_(loaded.item())
                    else:
                        self.weight.data.copy_(loaded.t() if self.transpose else loaded)

        for value in (-2.5, 0.0, 3.0):
            with self.subTest(value=value):
                source = ScalarModel()
                recorder = WeightLoadRecorder()
                recorder.record_model_load(
                    source,
                    (
                        ("scale", torch.tensor(value)),
                        ("weight", torch.arange(4).reshape(2, 2).float()),
                    ),
                    execute_writes=True,
                )
                source_plan = recorder.build_plan()
                manifest = ImmutableWeightRuntimeManifestBuilder(
                    model=source,
                    load_plan=source_plan,
                    topology=WeightParallelTopology(),
                    allowed_devices=("cpu",),
                ).build(
                    model_id="scalar-model",
                    revision="test",
                    instance_id="source",
                    worker_id="source",
                    endpoint="127.0.0.1:1",
                )
                placement, _ = build_mooncake_placement_and_bindings(
                    (manifest,),
                    placement_set_id="scalar-source",
                    tp_size=1,
                    pp_size=1,
                    ep_size=1,
                )
                scalar_descriptor = next(
                    tensor
                    for tensor in placement.parts[0].tensors
                    if tensor.tensor_id == "scale"
                )
                self.assertEqual(scalar_descriptor.global_shape, (1,))
                metadata = logical_weight_metadata_from_runtime_manifests(
                    (msgspec.to_builtins(manifest),)
                )
                target = ScalarModel()
                plan = record_target_weight_load_plan(target, metadata)
                self.assertEqual(target.transpose, source.transpose)
                self.assertEqual(source.scale.item(), value)
                self.assertTrue(torch.equal(target.scale, torch.zeros(1)))
                self.assertTrue(torch.equal(target.weight, torch.zeros(2, 2)))
                self.assertEqual(
                    [
                        (
                            v.tensor_id,
                            v.global_offset,
                            v.local_shape,
                            v.layout_fingerprint,
                        )
                        for v in plan.views
                    ],
                    [
                        (
                            v.tensor_id,
                            v.global_offset,
                            v.local_shape,
                            v.layout_fingerprint,
                        )
                        for v in source_plan.views
                    ],
                )

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
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(2, 2))

            def load_weights(self, weights):
                for _, loaded_weight in weights:
                    self.weight.data.copy_(loaded_weight)

        model = Model()
        recorder = WeightLoadRecorder()
        recorder.record_model_load(
            model,
            (("weight", torch.arange(4).reshape(2, 2).float()),),
            execute_writes=True,
        )
        manifest = ImmutableWeightRuntimeManifestBuilder(
            model=model,
            load_plan=recorder.build_plan(),
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
        placement, bindings = build_mooncake_placement_and_bindings(
            (manifest,),
            placement_set_id="test",
            tp_size=1,
            pp_size=1,
            ep_size=1,
        )
        self.assertEqual(bindings[0].generation, IMMUTABLE_WEIGHT_GENERATION)
        self.assertEqual(bindings[0].placement_id, placement.placement_id)
        self.assertEqual(bindings[0].lease_id, "immutable:worker")

    def test_weight_parallel_layout_supports_ep_and_rejects_unsupported_dp(self):
        validate_weight_heterogeneous_transfer_configuration(
            tp_size=2,
            dp_size=2,
            ep_size=1,
            pp_size=2,
            enable_dp_attention=True,
            quantization=None,
        )
        parallel_layout = WeightParallelLayout(tp_size=2, dp_size=2, pp_size=2)
        self.assertEqual(parallel_layout.world_size, 4)
        self.assertEqual(parallel_layout.dp_size, 2)
        with self.assertRaisesRegex(
            WeightHeterogeneousTransferError, "attention-DP partial replication"
        ):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=8,
                dp_size=2,
                ep_size=1,
                pp_size=1,
                enable_dp_attention=True,
                quantization=None,
            )
        validate_weight_heterogeneous_transfer_configuration(
            tp_size=4,
            dp_size=1,
            ep_size=2,
            pp_size=1,
            enable_dp_attention=False,
            quantization=None,
        )
        with self.assertRaisesRegex(WeightHeterogeneousTransferError, "moe_dp_size=2"):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=4,
                dp_size=1,
                ep_size=2,
                pp_size=1,
                enable_dp_attention=False,
                moe_dp_size=2,
                quantization=None,
            )
        with self.assertRaisesRegex(WeightHeterogeneousTransferError, "attn_cp_size=2"):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=4,
                dp_size=1,
                ep_size=2,
                pp_size=1,
                enable_dp_attention=False,
                attn_cp_size=2,
                quantization=None,
            )
        with self.assertRaisesRegex(
            WeightHeterogeneousTransferError, "without enable_dp_attention"
        ):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=2,
                dp_size=2,
                ep_size=1,
                pp_size=1,
                enable_dp_attention=False,
                quantization=None,
            )
        with self.assertRaisesRegex(
            WeightHeterogeneousTransferError, "must be divisible by dp_size"
        ):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=3,
                dp_size=2,
                ep_size=1,
                pp_size=1,
                enable_dp_attention=True,
                quantization=None,
            )
        with self.assertRaises(WeightHeterogeneousTransferError):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=2,
                dp_size=1,
                ep_size=1,
                pp_size=1,
                enable_dp_attention=False,
                quantization="fp8",
            )

    def test_parallel_axes_use_inferred_process_semantics(self):
        from mooncake.reshard.weight import ReplicatedAxis, SplitAxis

        tensor = {"tensor_id": "tensor", "expert_id": None}
        self.assertEqual(
            _parallel_axes(
                tensor,
                tp_size=2,
                pp_size=1,
                ep_size=1,
                tp_split_dims=(2,),
            ),
            (SplitAxis("tp", dim=2),),
        )
        self.assertEqual(
            _parallel_axes(
                tensor,
                tp_size=2,
                pp_size=1,
                ep_size=1,
                tp_replicated=True,
            ),
            (ReplicatedAxis("tp"),),
        )

    def test_internal_fragments_replicated_by_dp_attention_are_not_tp_shards(self):
        from mooncake.reshard.weight import ReplicatedAxis

        manifests = []
        for tp_rank in range(2):
            tensors = []
            storage_address = 100_000 + tp_rank * 1_000
            for part in range(3):
                tensors.append(
                    {
                        "fragment_id": f"rank{tp_rank}:part{part}",
                        "tensor_id": "model.layers.0.linear_attn.conv1d.weight",
                        "runtime_name": "runtime.conv1d.weight",
                        "global_shape": (6, 1, 4),
                        "global_offset": (part * 2, 0, 0),
                        "local_shape": (2, 1, 4),
                        "dtype": "bfloat16",
                        "itemsize": 2,
                        "shard_dims": (0,),
                        "layer_id": 0,
                        "expert_id": None,
                        "layout_fingerprint": "recorded-slices",
                        "address": storage_address + part * 16,
                        "nbytes": 16,
                        "byte_offset": part * 16,
                        "stride": (4, 4, 1),
                        "storage_offset": part * 8,
                        "device": f"cuda:{tp_rank}",
                        "worker_id": f"worker-{tp_rank}",
                        "endpoint": f"127.0.0.1:{1000 + tp_rank}",
                        "rank": {"dp": 0, "tp": tp_rank, "pp": 0, "ep": 0},
                    }
                )
            manifests.append(
                {
                    "model_id": "model",
                    "revision": "revision",
                    "instance_id": f"worker-{tp_rank}",
                    "generation": IMMUTABLE_WEIGHT_GENERATION,
                    "tensors": tensors,
                }
            )

        placement, _ = build_mooncake_placement_and_bindings(
            manifests,
            placement_set_id="target",
            tp_size=2,
            pp_size=1,
            ep_size=1,
        )

        descriptors = {
            tensor.tensor_id: tensor
            for part in placement.parts
            for tensor in part.tensors
        }
        descriptor = descriptors["model.layers.0.linear_attn.conv1d.weight"]
        self.assertEqual(descriptor.parallel_axes, (ReplicatedAxis("tp"),))
        self.assertEqual(descriptor.shard_dims, ())

    def test_ep_moe_tp_manifest_reshards_to_different_ep_layout(self):
        import ctypes
        from math import prod

        from mooncake.reshard.weight import (
            OwnershipAxis,
            ReplicatedAxis,
            SplitAxis,
            plan_placement_transfer,
        )

        def contiguous_stride(shape):
            stride = 1
            result = []
            for extent in reversed(shape):
                result.append(stride)
                stride *= extent
            return tuple(reversed(result))

        buffers = {}
        expected = {}

        def make_manifests(*, ep_size, moe_tp_size, prefix):
            manifests = []
            num_experts = 4
            experts_per_rank = num_experts // ep_size
            for ep_rank in range(ep_size):
                for tp_rank in range(moe_tp_size):
                    tensors = []

                    def add_tensor(
                        tensor_id,
                        global_shape,
                        global_offset,
                        local_shape,
                        shard_dims,
                        *,
                        expert_id=None,
                    ):
                        tensor_index = len(tensors)
                        fragment_id = (
                            f"{prefix}:e{ep_rank}:t{tp_rank}:{tensor_index}:{tensor_id}"
                        )
                        full = torch.arange(prod(global_shape)).reshape(global_shape)
                        slices = tuple(
                            slice(offset, offset + extent)
                            for offset, extent in zip(global_offset, local_shape)
                        )
                        values = full[slices].to(torch.bfloat16).contiguous()
                        buffer = (
                            values.clone()
                            if prefix == "source"
                            else torch.full_like(values, -1)
                        )
                        buffers[fragment_id] = buffer
                        expected[fragment_id] = values
                        tensors.append(
                            {
                                "fragment_id": fragment_id,
                                "tensor_id": tensor_id,
                                "runtime_name": tensor_id,
                                "global_shape": global_shape,
                                "global_offset": global_offset,
                                "local_shape": local_shape,
                                "dtype": "bfloat16",
                                "itemsize": 2,
                                "shard_dims": shard_dims,
                                "layer_id": 0,
                                "expert_id": expert_id,
                                "layout_fingerprint": "ep-moe-tp-test",
                                "address": buffer.data_ptr(),
                                "nbytes": prod(local_shape) * 2,
                                "byte_offset": 0,
                                "stride": contiguous_stride(local_shape),
                                "storage_offset": 0,
                                "device": "cpu",
                                "worker_id": f"{prefix}-e{ep_rank}-t{tp_rank}",
                                "endpoint": f"127.0.0.1:{2000 + ep_rank * moe_tp_size + tp_rank}",
                                "rank": {
                                    "dp": 0,
                                    "tp": tp_rank,
                                    "pp": 0,
                                    "ep": ep_rank,
                                },
                            }
                        )

                    global_tp_rank = ep_rank * moe_tp_size + tp_rank
                    global_tp_size = ep_size * moe_tp_size
                    for internal_part in range(2):
                        add_tensor(
                            "model.layers.0.dense.weight",
                            (4, 8),
                            (
                                internal_part * 2,
                                global_tp_rank * (8 // global_tp_size),
                            ),
                            (2, 8 // global_tp_size),
                            (0, 1),
                        )
                    add_tensor(
                        "model.layers.0.experts.weight",
                        (num_experts, 8),
                        (ep_rank * experts_per_rank, tp_rank * (8 // moe_tp_size)),
                        (experts_per_rank, 8 // moe_tp_size),
                        tuple(
                            dim
                            for dim, size in enumerate((ep_size, moe_tp_size))
                            if size > 1
                        ),
                    )
                    first_expert = ep_rank * experts_per_rank
                    for expert_id in range(
                        first_expert, first_expert + experts_per_rank
                    ):
                        add_tensor(
                            f"model.layers.0.experts.{expert_id}.weight",
                            (2, 8),
                            (0, tp_rank * (8 // moe_tp_size)),
                            (2, 8 // moe_tp_size),
                            (1,) if moe_tp_size > 1 else (),
                            expert_id=expert_id,
                        )
                        for internal_part in range(2):
                            add_tensor(
                                f"model.layers.0.experts.{expert_id}.replicated",
                                (4, 4),
                                (internal_part * 2, 0),
                                (2, 4),
                                (0,),
                                expert_id=expert_id,
                            )

                    manifests.append(
                        {
                            "model_id": "model",
                            "revision": "revision",
                            "instance_id": f"{prefix}-e{ep_rank}-t{tp_rank}",
                            "generation": IMMUTABLE_WEIGHT_GENERATION,
                            "tensors": tensors,
                        }
                    )
            return tuple(manifests)

        source_manifests = make_manifests(
            ep_size=2,
            moe_tp_size=2,
            prefix="source",
        )
        target_manifests = make_manifests(
            ep_size=4,
            moe_tp_size=1,
            prefix="target",
        )
        global_tp_ids = global_tp_tensor_ids(
            source_manifests, tp_size=4, ep_size=2
        ) | global_tp_tensor_ids(target_manifests, tp_size=4, ep_size=4)
        source_groups = build_mooncake_placement_groups(
            source_manifests,
            placement_set_id="source",
            tp_size=4,
            pp_size=1,
            ep_size=2,
            global_tp_ids=global_tp_ids,
        )
        target_groups = build_mooncake_placement_groups(
            target_manifests,
            placement_set_id="target",
            tp_size=4,
            pp_size=1,
            ep_size=4,
            global_tp_ids=global_tp_ids,
        )

        descriptors = {
            tensor.tensor_id: tensor
            for source, _ in source_groups.values()
            for part in source.parts
            for tensor in part.tensors
        }
        self.assertEqual(
            descriptors["model.layers.0.dense.weight"].parallel_axes,
            (SplitAxis("tp", dim=1),),
        )
        self.assertEqual(
            descriptors["model.layers.0.experts.weight"].parallel_axes,
            (SplitAxis("ep", dim=0), SplitAxis("tp", dim=1)),
        )
        self.assertEqual(
            descriptors["model.layers.0.experts.0.weight"].parallel_axes,
            (OwnershipAxis("ep"), SplitAxis("tp", dim=1)),
        )
        self.assertEqual(
            descriptors["model.layers.0.experts.0.replicated"].parallel_axes,
            (OwnershipAxis("ep"), ReplicatedAxis("tp")),
        )
        self.assertEqual(source_groups.keys(), target_groups.keys())
        self.assertEqual(source_groups["global_tp"][0].topology.tp_size, 4)
        self.assertEqual(source_groups["ep_moe_tp"][0].topology.tp_size, 2)
        for name, (source, _) in source_groups.items():
            target, _ = target_groups[name]
            logical_plan = plan_placement_transfer(source, target)
            self.assertTrue(logical_plan.operations)
            reverse_plan = plan_placement_transfer(target, source)
            self.assertTrue(reverse_plan.operations)

        class HostCopyEngine:
            def batch_transfer_sync_read(self, endpoint, targets, sources, sizes):
                for target, source, size in zip(targets, sources, sizes):
                    ctypes.memmove(target, source, size)
                return 0

        for manifest in target_manifests:
            stats = pull_weights_from_source(
                target_session=SimpleNamespace(
                    transfer_engine=HostCopyEngine(),
                    runtime_manifest=SimpleNamespace(
                        instance_id=manifest["instance_id"]
                    ),
                    parallel_layout=WeightParallelLayout(tp_size=4, ep_size=4),
                ),
                source_runtime_manifests=source_manifests,
                source_parallel_layout=WeightParallelLayout(tp_size=4, ep_size=2),
                target_runtime_manifests=target_manifests,
            )
            self.assertEqual(stats["logical_bytes"], stats["wire_bytes"])
            self.assertGreater(stats["wire_operations"], 0)
            for tensor in manifest["tensors"]:
                fragment_id = tensor["fragment_id"]
                torch.testing.assert_close(
                    buffers[fragment_id], expected[fragment_id], rtol=0, atol=0
                )

    def test_manifest_server_aggregates_source_ranks(self):
        server = object.__new__(WeightManifestServer)
        server.expected_rank_count = 2
        server._node_id = None
        server._parallel_layout = None
        server._model_identity = None
        server._rank_manifests = {}
        server._lock = threading.Lock()

        for global_rank, device_uuid in ((0, "GPU-2"), (1, "GPU-3")):
            server.register_weight_manifest(
                {
                    "node_id": "source-node",
                    "global_rank": global_rank,
                    "device_uuid": device_uuid,
                    "parallel_layout": {
                        "tp_size": 2,
                        "dp_size": 2,
                        "pp_size": 1,
                        "ep_size": 1,
                    },
                    "runtime_manifest": {
                        "model_id": "model",
                        "revision": "revision",
                        "rank": global_rank,
                    },
                }
            )

        source = parse_source_daemon_group(server.get_weight_manifest())
        self.assertEqual(source.device_uuids, ("GPU-2", "GPU-3"))
        self.assertEqual(source.parallel_layout.dp_size, 2)
        self.assertEqual(
            tuple(item["rank"] for item in source.runtime_manifests), (0, 1)
        )

    def test_tcp_manifest_registry_round_trip(self):
        server = WeightManifestServer(host="127.0.0.1", port=0, expected_rank_count=1)
        registry_url = f"tcp://127.0.0.1:{server.port}"
        session = MagicMock()
        session.parallel_layout = WeightParallelLayout(tp_size=1)
        session.runtime_manifest_for_wire.return_value = {
            "model_id": "model",
            "revision": "revision",
            "rank": 0,
        }
        try:
            register_weight_manifest(
                registry_url,
                global_rank=0,
                device_uuid="GPU-2",
                session=session,
                timeout=2,
            )
            actual = fetch_weight_manifest(registry_url, timeout=2)
        finally:
            server.close()

        self.assertEqual(actual.device_uuids, ("GPU-2",))
        self.assertEqual(actual.parallel_layout, WeightParallelLayout(tp_size=1))
        self.assertEqual(
            actual.runtime_manifests,
            ({"model_id": "model", "revision": "revision", "rank": 0},),
        )

    def test_target_rejects_overlapping_device_uuid(self):
        source = SourceDaemonGroup(
            parallel_layout=WeightParallelLayout(1),
            device_uuids=("GPU-shared",),
            runtime_manifests=({"rank": 0},),
        )

        def all_gather(output, local):
            output[:] = [local]

        with (
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.fetch_weight_manifest",
                return_value=source,
            ) as fetch,
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.broadcast_object_list"),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.all_gather_object", side_effect=all_gather),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.pull_weights_from_source"
            ) as pull,
        ):
            with self.assertRaisesRegex(
                WeightHeterogeneousTransferError,
                "overlapping device UUIDs",
            ):
                transfer_weights_from_source_daemons(
                    target_session=MagicMock(),
                    model=MagicMock(),
                    device_uuid="GPU-shared",
                    registry_url="tcp://source:31999",
                )

        fetch.assert_called_once_with("tcp://source:31999")
        pull.assert_not_called()

    def test_target_allows_distinct_device_uuid(self):
        source = SourceDaemonGroup(
            parallel_layout=WeightParallelLayout(1),
            device_uuids=("GPU-source",),
            runtime_manifests=({"rank": 0},),
        )

        def all_gather(output, local):
            output[:] = [local]

        with (
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.fetch_weight_manifest",
                return_value=source,
            ),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.broadcast_object_list"),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.all_gather_object", side_effect=all_gather),
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.pull_weights_from_source",
                return_value={},
            ) as pull,
            patch(
                "sglang.srt.weight_cache.weight_heterogeneous_transfer.rebuild_transferred_weight_state"
            ),
        ):
            transfer_weights_from_source_daemons(
                target_session=MagicMock(),
                model=MagicMock(),
                device_uuid="GPU-target",
                registry_url="tcp://source:31999",
            )

        pull.assert_called_once()

    def test_rank_local_phase_propagates_failure_to_every_rank(self):
        def all_gather(output, local):
            output[:] = [local]

        def fail():
            raise RuntimeError("rank-local boom")

        with (
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.distributed.all_gather_object", side_effect=all_gather),
        ):
            with self.assertRaisesRegex(
                WeightHeterogeneousTransferError,
                "pull failed on target rank.*rank 0.*rank-local boom",
            ):
                _all_gather_rank_local_phase("pull", fail)


if __name__ == "__main__":
    unittest.main()
