# SPDX-License-Identifier: Apache-2.0
"""Declared layouts, Mooncake bindings, CPU reshard copies and manifest exchange."""

import ctypes
import importlib.util
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.weight_cache.model_adapters.layers import LayerLayoutBuilder
from sglang.srt.weight_cache.mooncake_weight_adapter import (
    build_mooncake_transfer_groups,
)
from sglang.srt.weight_cache.weight_heterogeneous_transfer import (
    SourceDaemonGroup,
    WeightHeterogeneousTransferError,
    WeightParallelLayout,
    WeightTransferSession,
    _all_gather_rank_local_phase,
    _initialize_weight_transfer_engine,
    fetch_weight_manifest,
    pull_weights_from_source,
    register_weight_manifest,
    transfer_weights_from_source_daemons,
    validate_weight_heterogeneous_transfer_configuration,
)
from sglang.srt.weight_cache.weight_manifest_server import WeightManifestServer
from sglang.srt.weight_cache.weight_runtime_manifest import (
    IMMUTABLE_WEIGHT_GENERATION,
    ImmutableWeightRuntimeManifestBuilder,
    WeightManifestError,
    WeightParallelTopology,
    WeightPartition,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

# Check each parent before looking up its child. Do not swallow import errors
# from an installed module: broken dependencies and incompatible APIs must fail.
HAS_MOONCAKE_RESHARD = all(
    importlib.util.find_spec(name) is not None
    for name in ("mooncake", "mooncake.reshard", "mooncake.reshard.weight")
)


def _manifest(model, plan, *, worker="source", topology=None, endpoint="127.0.0.1:1"):
    return ImmutableWeightRuntimeManifestBuilder(
        model=model,
        layout_plan=plan,
        topology=topology or WeightParallelTopology(),
        allowed_devices=(next(model.parameters()).device.type,),
    ).build(
        model_id="model",
        revision="test",
        instance_id=worker,
        worker_id=worker,
        endpoint=endpoint,
    )


class HostCopyEngine:
    def batch_transfer_sync_read(self, endpoint, targets, sources, sizes):
        for target, source, size in zip(targets, sources, sizes):
            ctypes.memmove(target, source, size)
        return 0


def _single_placement(manifests, *, tp_size, pp_size, ep_size):
    layout = WeightParallelLayout(tp_size, pp_size=pp_size, ep_size=ep_size)
    groups = build_mooncake_transfer_groups(
        manifests,
        manifests,
        source_parallel_layout=layout,
        target_parallel_layout=layout,
    )
    assert len(groups) == 1
    return next(iter(groups.values()))[:2]


@contextmanager
def _target_transfer(source_uuid):
    source = SourceDaemonGroup(
        parallel_layout=WeightParallelLayout(1),
        device_uuids=(source_uuid,),
        runtime_manifests=({"rank": 0},),
    )

    def all_gather(output, local):
        output[:] = [local]

    module = "sglang.srt.weight_cache.weight_heterogeneous_transfer"
    with (
        patch(f"{module}.fetch_weight_manifest", return_value=source) as fetch,
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.broadcast_object_list"),
        patch("torch.distributed.get_world_size", return_value=1),
        patch("torch.distributed.all_gather_object", side_effect=all_gather),
        patch(f"{module}.pull_weights_from_source", return_value={}) as pull,
    ):
        yield fetch, pull


class TestMooncakeWeightAdapter(unittest.TestCase):
    @unittest.skipUnless(HAS_MOONCAKE_RESHARD, "requires mooncake.reshard.weight")
    def test_native_gqa_replica_groups_and_vocab_padding_reshard(self):
        import msgspec
        from sglang.srt.layers.linear import QKVParallelLinear
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
        from sglang.srt.weight_cache.weight_heterogeneous_transfer import (
            WeightTransferSession,
        )

        class Model(torch.nn.Module):
            def __init__(self, kind, rank, size):
                super().__init__()
                self.kind = kind
                group = SimpleNamespace(rank_in_group=rank, world_size=size)
                with (
                    patch(
                        "sglang.srt.layers.linear._resolve_linear_group",
                        return_value=group,
                    ),
                    patch(
                        "sglang.srt.layers.vocab_parallel_embedding._resolve_linear_group",
                        return_value=group,
                    ),
                    patch(
                        "sglang.srt.layers.vocab_parallel_embedding.get_parallel",
                        return_value=SimpleNamespace(tp_rank=rank, tp_size=size),
                    ),
                ):
                    self.layer = (
                        QKVParallelLinear(
                            4, 2, 8, 2, bias=False, params_dtype=torch.float32
                        )
                        if kind == "gqa"
                        else VocabParallelEmbedding(
                            65, 4, padding_size=128, params_dtype=torch.float32
                        )
                    )
                self.weight.data.fill_(-99)

            @property
            def weight(self):
                return self.layer.weight

            def load_weights(self, weights):
                for name, loaded in weights:
                    if self.kind == "gqa":
                        self.weight.weight_loader(self.weight, loaded, name)
                    else:
                        self.weight.weight_loader(self.weight, loaded)

            def layout(self):
                builder = LayerLayoutBuilder(self)
                if self.kind == "gqa":
                    builder.qkv_parallel(self.layer, ("q", "k", "v"))
                else:
                    builder.vocab_parallel(self.layer, "embedding")
                return builder.build()

        for kind, pairs in (
            ("gqa", ((1, 4), (2, 4), (4, 2), (4, 1), (4, 8), (8, 4))),
            ("padding", ((1, 2), (2, 1))),
        ):
            checkpoint = (
                tuple(
                    (name, torch.arange(rows * 4).reshape(rows, 4).float() + 1000 * i)
                    for i, (name, rows) in enumerate((("q", 16), ("k", 4), ("v", 4)))
                )
                if kind == "gqa"
                else (("embedding", torch.arange(260).reshape(65, 4).float()),)
            )
            for source_size, target_size in pairs:
                with self.subTest(
                    kind=kind, source_tp=source_size, target_tp=target_size
                ):
                    # Manifests hold raw addresses, so keep source storage alive.
                    sources, source_manifests = [], []
                    for rank in range(source_size):
                        model = Model(kind, rank, source_size)
                        model.load_weights(checkpoint)
                        plan = model.layout()
                        manifest = _manifest(
                            model,
                            plan,
                            worker=f"source-{rank}",
                            topology=WeightParallelTopology(
                                tp_rank=rank, tp_size=source_size
                            ),
                        )
                        sources.append(model)
                        source_manifests.append(msgspec.to_builtins(manifest))
                        # Source padding must never become checkpoint payload.
                        for region in getattr(plan, "initializations", ()):
                            model.weight.data.view(-1)[region.byte_offset // 4 :].fill_(
                                12345
                            )
                    targets, sessions = [], []
                    for rank in range(target_size):
                        model = Model(kind, rank, target_size)
                        session = WeightTransferSession(
                            transfer_engine=HostCopyEngine(),
                            runtime_manifest=None,
                            parallel_layout=WeightParallelLayout(target_size),
                            registered_memory_ranges=(),
                            manifest_context={},
                            layout_plan=model.layout(),
                        )
                        module = "sglang.srt.weight_cache.weight_heterogeneous_transfer"

                        def build_manifest(*, model, layout_plan):
                            return _manifest(
                                model,
                                layout_plan,
                                worker=f"target-{rank}",
                                topology=WeightParallelTopology(
                                    tp_rank=rank, tp_size=target_size
                                ),
                            )

                        with (
                            patch(
                                f"{module}._build_weight_runtime_manifest",
                                side_effect=build_manifest,
                            ),
                            patch(
                                f"{module}._register_weight_fragment_allocations",
                                return_value=(),
                            ),
                        ):
                            session.prepare_target(model=model)
                        targets.append(model)
                        sessions.append(session)
                    target_manifests = tuple(
                        session.runtime_manifest_for_wire() for session in sessions
                    )
                    for rank, (model, session) in enumerate(zip(targets, sessions)):
                        pull_weights_from_source(
                            target_session=session,
                            source_runtime_manifests=source_manifests,
                            source_parallel_layout=WeightParallelLayout(source_size),
                            target_runtime_manifests=target_manifests,
                        )
                        reference = Model(kind, rank, target_size)
                        reference.load_weights(checkpoint)
                        torch.testing.assert_close(
                            model.weight, reference.weight, rtol=0, atol=0
                        )
                    if kind == "padding":
                        self.assertEqual(
                            sum(
                                t["nbytes"]
                                for m in target_manifests
                                for t in m["fragments"]
                            ),
                            65 * 4 * torch.float32.itemsize,  # Unpadded vocabulary.
                        )

    @unittest.skipUnless(
        all(
            importlib.util.find_spec(name) is not None
            for name in ("mooncake", "mooncake.engine")
        ),
        "requires mooncake.engine",
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

    @unittest.skipUnless(HAS_MOONCAKE_RESHARD, "requires mooncake.reshard.weight")
    def test_daemon_manifest_adapts_to_immutable_runtime_binding(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty(2, 2))

            def load_weights(self, weights):
                for _, loaded_weight in weights:
                    self.weight.data.copy_(loaded_weight)

        model = Model()
        model.load_weights((("weight", torch.arange(4).reshape(2, 2).float()),))
        builder = LayerLayoutBuilder(model)
        builder.replicated(model.weight, "weight")
        manifest = _manifest(model, builder.build(), worker="worker")

        self.assertEqual(manifest.generation, IMMUTABLE_WEIGHT_GENERATION)
        placement, bindings = _single_placement(
            (manifest,),
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
            attn_dp_size=2,
            ep_size=1,
            pp_size=2,
            quantization=None,
        )
        parallel_layout = WeightParallelLayout(tp_size=2, attn_dp_size=2, pp_size=2)
        self.assertEqual(parallel_layout.world_size, 4)
        self.assertEqual(parallel_layout.attn_dp_size, 2)
        with self.assertRaisesRegex(
            WeightHeterogeneousTransferError, "attention-DP partial replication"
        ):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=8,
                attn_dp_size=2,
                ep_size=1,
                pp_size=1,
                quantization=None,
            )
        validate_weight_heterogeneous_transfer_configuration(
            tp_size=4,
            attn_dp_size=1,
            ep_size=2,
            pp_size=1,
            quantization=None,
        )
        with self.assertRaisesRegex(WeightHeterogeneousTransferError, "moe_dp_size=2"):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=4,
                attn_dp_size=1,
                ep_size=2,
                pp_size=1,
                moe_dp_size=2,
                quantization=None,
            )
        with self.assertRaisesRegex(WeightHeterogeneousTransferError, "attn_cp_size=2"):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=4,
                attn_dp_size=1,
                ep_size=2,
                pp_size=1,
                attn_cp_size=2,
                quantization=None,
            )
        with self.assertRaisesRegex(
            WeightHeterogeneousTransferError, "engine dp_size=2"
        ):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=2,
                attn_dp_size=1,
                dp_size=2,
                ep_size=1,
                pp_size=1,
            )
        with self.assertRaisesRegex(
            WeightHeterogeneousTransferError, "must be divisible by attn_dp_size"
        ):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=3,
                attn_dp_size=2,
                ep_size=1,
                pp_size=1,
                quantization=None,
            )
        with self.assertRaisesRegex(
            WeightHeterogeneousTransferError, "quantization='fp8'"
        ):
            validate_weight_heterogeneous_transfer_configuration(
                tp_size=2,
                attn_dp_size=1,
                ep_size=1,
                pp_size=1,
                quantization="fp8",
            )

    @unittest.skipUnless(HAS_MOONCAKE_RESHARD, "requires mooncake.reshard.weight")
    def test_parallel_axes_use_declared_process_semantics(self):
        from mooncake.reshard.weight import ReplicatedAxis, SplitAxis

        for split in (False, True):
            manifests = []
            for rank in range(2):
                model = torch.nn.Module()
                model.weight = torch.nn.Parameter(torch.empty(2, 3, 2))
                builder = LayerLayoutBuilder(model)
                builder.region(
                    model.weight,
                    "weight",
                    global_shape=(2, 3, 4 if split else 2),
                    global_offset=(0, 0, rank * 2 if split else 0),
                    partition=WeightPartition(rank=rank, size=2, dim=2)
                    if split
                    else WeightPartition(),
                )
                manifests.append(
                    _manifest(
                        model,
                        builder.build(),
                        worker=f"rank{rank}",
                        topology=WeightParallelTopology(tp_rank=rank, tp_size=2),
                    )
                )
            placement, _ = _single_placement(manifests, tp_size=2, pp_size=1, ep_size=1)
            self.assertEqual(
                placement.tensors[0].parallel_axes,
                (SplitAxis("tp", dim=2),) if split else (ReplicatedAxis("tp"),),
            )

    @unittest.skipUnless(HAS_MOONCAKE_RESHARD, "requires mooncake.reshard.weight")
    def test_replicated_internal_fragments_are_not_tp_shards(self):
        from mooncake.reshard.weight import ReplicatedAxis

        manifests = []
        for rank in range(2):
            model = torch.nn.Module()
            model.weight = torch.nn.Parameter(torch.empty(6, 1, 4))
            builder = LayerLayoutBuilder(model)
            for part in range(3):
                builder.region(
                    model.weight,
                    "conv",
                    global_shape=(6, 1, 4),
                    global_offset=(part * 2, 0, 0),
                    local_shape=(2, 1, 4),
                    byte_offset=part * 32,
                )
            manifests.append(
                _manifest(
                    model,
                    builder.build(),
                    worker=f"rank{rank}",
                    topology=WeightParallelTopology(tp_rank=rank, tp_size=2),
                )
            )
        placement, _ = _single_placement(manifests, tp_size=2, pp_size=1, ep_size=1)
        self.assertEqual(placement.tensors[0].parallel_axes, (ReplicatedAxis("tp"),))
        self.assertEqual(placement.tensors[0].shard_dims, ())

    @unittest.skipUnless(HAS_MOONCAKE_RESHARD, "requires mooncake.reshard.weight")
    def test_ep_moe_tp_manifest_reshards_to_different_ep_layout(self):
        from math import prod

        from mooncake.reshard.weight import (
            OwnershipAxis,
            ReplicatedAxis,
            SplitAxis,
        )

        models, expected = {}, {}

        def make_manifests(ep_size, moe_tp_size, prefix):
            manifests = []
            for ep in range(ep_size):
                for tp in range(moe_tp_size):
                    model = torch.nn.Module()
                    declarations = []
                    worker = f"{prefix}-e{ep}-t{tp}"

                    def add(name, shape, offset, local, partition, expert_id=None):
                        values = (
                            torch.arange(prod(shape))
                            .reshape(shape)[
                                tuple(slice(o, o + n) for o, n in zip(offset, local))
                            ]
                            .to(torch.bfloat16)
                            .contiguous()
                        )
                        param_name = f"p{len(declarations)}"
                        parameter = torch.nn.Parameter(
                            values.clone()
                            if prefix == "source"
                            else torch.full_like(values, -1),
                            requires_grad=False,
                        )
                        model.register_parameter(param_name, parameter)
                        expected[worker, param_name] = values
                        declarations.append(
                            (
                                parameter,
                                name,
                                shape,
                                offset,
                                local,
                                partition,
                                expert_id,
                            )
                        )

                    global_tp = ep * moe_tp_size + tp
                    for part in range(2):
                        add(
                            "dense",
                            (4, 8),
                            (part * 2, global_tp * 2),
                            (2, 2),
                            WeightPartition(rank=global_tp, size=4, dim=1),
                        )
                    add(
                        "stacked_experts",
                        (4, 8),
                        (ep * (4 // ep_size), tp * (8 // moe_tp_size)),
                        (4 // ep_size, 8 // moe_tp_size),
                        WeightPartition(
                            kind="expert", rank=tp, size=moe_tp_size, dim=1
                        ),
                    )
                    for expert in range(ep * (4 // ep_size), (ep + 1) * (4 // ep_size)):
                        add(
                            f"experts.{expert}.weight",
                            (2, 8),
                            (0, tp * (8 // moe_tp_size)),
                            (2, 8 // moe_tp_size),
                            WeightPartition(
                                kind="expert", rank=tp, size=moe_tp_size, dim=1
                            ),
                            expert,
                        )
                        for part in range(2):
                            add(
                                f"experts.{expert}.replicated",
                                (4, 4),
                                (part * 2, 0),
                                (2, 4),
                                WeightPartition(
                                    kind="expert", rank=tp, size=moe_tp_size
                                ),
                                expert,
                            )
                    builder = LayerLayoutBuilder(model)
                    for (
                        parameter,
                        name,
                        shape,
                        offset,
                        local,
                        partition,
                        expert,
                    ) in declarations:
                        builder.region(
                            parameter,
                            name,
                            global_shape=shape,
                            global_offset=offset,
                            local_shape=local,
                            partition=partition,
                            expert_id=expert,
                            layout_fingerprint="ep-moe-tp-test",
                        )
                    manifests.append(
                        _manifest(
                            model,
                            builder.build(),
                            worker=worker,
                            topology=WeightParallelTopology(
                                tp_rank=tp,
                                tp_size=moe_tp_size,
                                ep_rank=ep,
                                ep_size=ep_size,
                            ),
                        )
                    )
                    models[worker] = model
            return manifests

        sources = make_manifests(2, 2, "source")
        targets = make_manifests(4, 1, "target")
        source_layout, target_layout = (
            WeightParallelLayout(4, ep_size=2),
            WeightParallelLayout(4, ep_size=4),
        )
        groups = build_mooncake_transfer_groups(
            sources,
            targets,
            source_parallel_layout=source_layout,
            target_parallel_layout=target_layout,
        )
        self.assertEqual(len(groups), 2)
        descriptors = {
            t.tensor_id: t
            for source, _, _, _ in groups.values()
            for t in source.tensors
        }
        self.assertEqual(descriptors["dense"].parallel_axes, (SplitAxis("tp", dim=1),))
        self.assertEqual(
            descriptors["stacked_experts"].parallel_axes,
            (SplitAxis("ep", dim=0), SplitAxis("tp", dim=1)),
        )
        self.assertEqual(
            descriptors["experts.0.weight"].parallel_axes,
            (OwnershipAxis("ep"), SplitAxis("tp", dim=1)),
        )
        self.assertEqual(
            descriptors["experts.0.replicated"].parallel_axes,
            (OwnershipAxis("ep"), ReplicatedAxis("tp")),
        )
        for manifest in targets:
            stats = pull_weights_from_source(
                target_session=SimpleNamespace(
                    transfer_engine=HostCopyEngine(),
                    runtime_manifest=manifest,
                    parallel_layout=target_layout,
                    layout_plan=None,
                ),
                source_runtime_manifests=sources,
                source_parallel_layout=source_layout,
                target_runtime_manifests=targets,
            )
            self.assertGreater(stats["wire_operations"], 0)
            for name, parameter in models[manifest.instance_id].named_parameters():
                torch.testing.assert_close(
                    parameter, expected[manifest.instance_id, name], rtol=0, atol=0
                )

    @unittest.skipUnless(HAS_MOONCAKE_RESHARD, "requires mooncake.reshard.weight")
    def test_gqa_groups_share_both_sides_mapping_and_preserve_bindings(self):
        from sglang.srt.layers.linear import QKVParallelLinear

        checkpoint = {
            (layer, part): torch.arange(rows * 4).reshape(rows, 4).float()
            + 1000 * (3 * layer + i)
            for layer in range(2)
            for i, (part, rows) in enumerate((("q", 16), ("k", 4), ("v", 4)))
        }

        def make_model(rank, size, permuted, load, pp_rank=0, pp_size=1):
            model = torch.nn.Module()
            model.layers = torch.nn.ModuleDict()
            for layer in range(2) if pp_size == 1 else (pp_rank,):

                def group(name):
                    ordinal = (
                        [0, 2, 1, 3][rank]
                        if permuted and layer == 1 and size == 4 and name == "attn_tp"
                        else rank
                    )
                    return SimpleNamespace(rank_in_group=ordinal, world_size=size)

                with patch(
                    "sglang.srt.layers.linear._resolve_linear_group", side_effect=group
                ):
                    module = QKVParallelLinear(
                        4,
                        2,
                        8,
                        2,
                        bias=False,
                        params_dtype=torch.float32,
                        parallel_group="tp",
                        kv_parallel_group="attn_tp",
                    )
                module.weight.data.fill_(-99)
                if load:
                    for part in ("q", "k", "v"):
                        module.weight.weight_loader(
                            module.weight, checkpoint[layer, part], part
                        )
                model.layers[str(layer)] = module
            builder = LayerLayoutBuilder(model)
            for layer, module in model.layers.items():
                builder.qkv_parallel(
                    module, tuple(f"layer{layer}.{part}" for part in ("q", "k", "v"))
                )
            return model, builder.build()

        for permuted, expected_groups, target_pp in (
            (False, 2, 1),
            (True, 3, 1),
            (False, 2, 2),
            (True, 3, 2),
        ):
            with self.subTest(permuted=permuted, pp=target_pp):
                sources, targets, live, sessions = [], [], [], []
                for source, size in ((True, 2), (False, 4)):
                    pp_size = 1 if source else target_pp
                    for ordinal in range(size * pp_size):
                        pp_rank, rank = divmod(ordinal, size)
                        model, plan = make_model(
                            rank, size, permuted, source, pp_rank, pp_size
                        )
                        manifest = _manifest(
                            model,
                            plan,
                            worker=f"{source}-pp{pp_rank}-tp{rank}",
                            topology=WeightParallelTopology(
                                tp_rank=rank,
                                tp_size=size,
                                pp_rank=pp_rank,
                                pp_size=pp_size,
                            ),
                        )
                        (sources if source else targets).append(manifest)
                        live.append(model)
                        if not source:
                            sessions.append(
                                WeightTransferSession(
                                    HostCopyEngine(),
                                    manifest,
                                    WeightParallelLayout(size, pp_size=pp_size),
                                    (),
                                    {},
                                    layout_plan=plan,
                                )
                            )
                groups = build_mooncake_transfer_groups(
                    sources,
                    targets,
                    source_parallel_layout=WeightParallelLayout(2),
                    target_parallel_layout=WeightParallelLayout(4, pp_size=target_pp),
                )
                self.assertEqual(len(groups), expected_groups)
                if not permuted:
                    kv = [g for g in groups.values() if g[2].topology.dp_size == 2]
                    self.assertEqual(len(kv), 1)
                    self.assertEqual(len(kv[0][2].tensors), 4)
                for side, manifests in ((0, sources), (2, targets)):
                    actual = {
                        (
                            b.instance_id,
                            f.address,
                            f.nbytes,
                            f.storage_address,
                            f.storage_nbytes,
                            f.storage_offset_bytes,
                        )
                        for g in groups.values()
                        for b in g[side + 1]
                        for f in b.fragments
                    }
                    expected_bindings = {
                        (
                            m.instance_id,
                            f.address,
                            f.nbytes,
                            f.storage_address,
                            f.storage_nbytes,
                            f.storage_offset_bytes,
                        )
                        for m in manifests
                        for f in m.fragments
                    }
                    self.assertEqual(actual, expected_bindings)
                for rank, session in enumerate(sessions):
                    pull_weights_from_source(
                        target_session=session,
                        source_runtime_manifests=sources,
                        source_parallel_layout=WeightParallelLayout(2),
                        target_runtime_manifests=targets,
                    )
                    reference, _ = make_model(
                        rank % 4, 4, permuted, True, rank // 4, target_pp
                    )
                    for actual, want in zip(
                        live[2 + rank].parameters(), reference.parameters()
                    ):
                        torch.testing.assert_close(actual, want, rtol=0, atol=0)

    @unittest.skipUnless(HAS_MOONCAKE_RESHARD, "requires mooncake.reshard.weight")
    def test_invalid_partition_declarations_fail_before_transfer(self):
        import msgspec

        models, manifests = [], []
        for rank in range(2):
            model = torch.nn.Linear(2, 2, bias=False)
            builder = LayerLayoutBuilder(model)
            builder.region(
                model.weight,
                "w",
                global_shape=(4, 2),
                global_offset=(rank * 2, 0),
                partition=WeightPartition(rank=rank, size=2, dim=0),
            )
            manifests.append(
                _manifest(
                    model,
                    builder.build(),
                    worker=f"r{rank}",
                    topology=WeightParallelTopology(tp_rank=rank, tp_size=2),
                )
            )
            models.append(model)
        bad = msgspec.structs.replace(
            manifests[1].tensors[0], partition=WeightPartition(rank=0, size=2, dim=0)
        )
        invalid = [manifests[0], msgspec.structs.replace(manifests[1], tensors=(bad,))]
        with self.assertRaisesRegex(WeightManifestError, "uneven partition replicas"):
            build_mooncake_transfer_groups(
                invalid,
                manifests,
                source_parallel_layout=WeightParallelLayout(2),
                target_parallel_layout=WeightParallelLayout(2),
            )

    @unittest.skipUnless(HAS_MOONCAKE_RESHARD, "requires mooncake.reshard.weight")
    def test_padding_only_rank_preserves_declaration_and_fails_before_io(self):
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

        sources, targets, live = [], [], []
        for source, size in ((True, 1), (False, 4)):
            for rank in range(size):
                group = SimpleNamespace(rank_in_group=rank, world_size=size)
                with (
                    patch(
                        "sglang.srt.layers.vocab_parallel_embedding._resolve_linear_group",
                        return_value=group,
                    ),
                    patch(
                        "sglang.srt.layers.vocab_parallel_embedding.get_parallel",
                        return_value=SimpleNamespace(tp_rank=rank, tp_size=size),
                    ),
                ):
                    model = VocabParallelEmbedding(
                        65, 4, padding_size=128, params_dtype=torch.float32
                    )
                model.weight.data.fill_(-99)
                builder = LayerLayoutBuilder(model)
                builder.vocab_parallel(model, "embedding")
                manifest = _manifest(
                    model,
                    builder.build(),
                    worker=f"{source}-{rank}",
                    topology=WeightParallelTopology(tp_rank=rank, tp_size=size),
                )
                (sources if source else targets).append(manifest)
                live.append(model)
        self.assertEqual(len(targets[-1].tensors), 1)
        self.assertFalse(targets[-1].fragments)
        # The unchanged Mooncake contract requires every split-axis participant
        # to have a fragment. Do not silently reclassify an empty shard as a replica.
        engine = MagicMock()
        session = WeightTransferSession(
            engine, targets[0], WeightParallelLayout(4), (), {}
        )
        with self.assertRaisesRegex(
            ValueError, "missing or unexpected split-axis participant"
        ):
            pull_weights_from_source(
                target_session=session,
                source_runtime_manifests=sources,
                source_parallel_layout=WeightParallelLayout(1),
                target_runtime_manifests=targets,
            )
        engine.batch_transfer_sync_read.assert_not_called()
        for model in live:
            torch.testing.assert_close(
                model.weight, torch.full_like(model.weight, -99), rtol=0, atol=0
            )

    def test_tcp_manifest_registry_aggregates_all_source_ranks(self):
        server = WeightManifestServer(host="127.0.0.1", port=0, expected_rank_count=2)
        registry_url = f"tcp://127.0.0.1:{server.port}"
        session = MagicMock()
        session.parallel_layout = WeightParallelLayout(tp_size=2, attn_dp_size=2)
        try:
            # Arrival order must not change the rank ordering in the response.
            for rank in (1, 0):
                session.runtime_manifest_for_wire.return_value = {
                    "model_id": "model",
                    "revision": "revision",
                    "rank": rank,
                }
                register_weight_manifest(
                    registry_url,
                    global_rank=rank,
                    device_uuid=f"GPU-{rank + 2}",
                    session=session,
                    timeout=2,
                )
                if rank == 1:
                    with self.assertRaisesRegex(
                        RuntimeError, "registered=1, expected=2"
                    ):
                        server.get_weight_manifest()
            actual = fetch_weight_manifest(registry_url, timeout=2)
        finally:
            server.close()

        self.assertEqual(actual.device_uuids, ("GPU-2", "GPU-3"))
        self.assertEqual(actual.parallel_layout, session.parallel_layout)
        self.assertEqual(
            actual.runtime_manifests,
            tuple(
                {
                    "model_id": "model",
                    "revision": "revision",
                    "rank": rank,
                }
                for rank in (0, 1)
            ),
        )

    def test_target_rejects_overlapping_device_uuid(self):
        with _target_transfer("GPU-shared") as (fetch, pull):
            with self.assertRaisesRegex(
                WeightHeterogeneousTransferError, "overlapping device UUIDs"
            ):
                transfer_weights_from_source_daemons(
                    target_session=MagicMock(),
                    model=MagicMock(),
                    device_uuid="GPU-shared",
                    registry_url="tcp://source:31999",
                    timeout=1800,
                )
        fetch.assert_called_once_with("tcp://source:31999", timeout=1800)
        pull.assert_not_called()

    def test_target_allows_distinct_device_uuid(self):
        with _target_transfer("GPU-source") as (_, pull):
            transfer_weights_from_source_daemons(
                target_session=MagicMock(),
                model=MagicMock(),
                device_uuid="GPU-target",
                registry_url="tcp://source:31999",
                timeout=1800,
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
