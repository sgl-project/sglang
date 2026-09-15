from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.weight_sync.nccl_m2n import NcclM2NReceiver

_QUANTIZATION = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    "weight_dtype": "float8_e4m3fn",
    "scale_dtype": "float32",
    "scale_format": "canonical",
}
_TOPOLOGY = {
    "tp_rank": 0,
    "tp_size": 2,
    "moe_ep_rank": 0,
    "moe_ep_size": 2,
    "moe_tp_rank": 0,
    "moe_tp_size": 1,
    "dp_rank": 0,
    "dp_size": 1,
    "pp_rank": 0,
    "pp_size": 1,
}
_MOE_TP_TOPOLOGY = {
    **_TOPOLOGY,
    "moe_ep_size": 1,
    "moe_tp_size": 2,
}


def _parameter(shape, dtype):
    return torch.nn.Parameter(
        torch.zeros(shape, dtype=dtype),
        requires_grad=False,
    )


def _model():
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    layer = root.model.layers[0]
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = torch.nn.Module()
    experts = layer.mlp.experts

    experts.w13_weight = _parameter(
        (1, 512, 256),
        torch.float8_e4m3fn,
    )
    experts.w2_weight = _parameter(
        (1, 256, 256),
        torch.float8_e4m3fn,
    )
    experts.w13_weight_scale_inv = _parameter(
        (1, 4, 2),
        torch.float32,
    )
    experts.w2_weight_scale_inv = _parameter(
        (1, 2, 2),
        torch.float32,
    )
    experts.quant_method = SimpleNamespace(
        block_quant=True,
        use_mxfp8=False,
        is_fp4_expert=False,
        load_up_proj_weight_first=False,
        quant_config=SimpleNamespace(
            use_mxfp8=False,
            is_fp4_experts=False,
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        ),
    )
    experts.moe_ep_rank = 0
    experts.moe_ep_size = 2
    experts._num_local_routed = 1
    experts._num_global_routed = 2
    return root


def _moe_tp_model(moe_tp_rank=0):
    root = _model()
    experts = root.model.layers[0].mlp.experts
    experts.w13_weight = _parameter(
        (2, 256, 256),
        torch.float8_e4m3fn,
    )
    experts.w2_weight = _parameter(
        (2, 256, 128),
        torch.float8_e4m3fn,
    )
    experts.w13_weight_scale_inv = _parameter(
        (2, 2, 2),
        torch.float32,
    )
    experts.w2_weight_scale_inv = _parameter(
        (2, 2, 1),
        torch.float32,
    )
    experts.moe_ep_size = 1
    experts.moe_tp_rank = moe_tp_rank
    experts.moe_tp_size = 2
    experts._num_local_routed = 2
    return root


def _bf16_moe_tp_model():
    root = _moe_tp_model()
    experts = root.model.layers[0].mlp.experts
    experts.w13_weight = _parameter(
        (2, 256, 256),
        torch.bfloat16,
    )
    experts.w2_weight = _parameter(
        (2, 256, 128),
        torch.bfloat16,
    )
    del experts.w13_weight_scale_inv
    del experts.w2_weight_scale_inv
    experts.quant_method = SimpleNamespace(load_up_proj_weight_first=False)
    return root


def _layout(local_shape, ranks):
    return {
        "mesh": [ranks],
        "placements": [
            {"type": "replicate"},
            {"type": "shard", "dim": 0},
        ],
        "local_shape": list(local_shape),
    }


def _manifest():
    entries = []
    for component, source_recipe, parameter in (
        ("gate", "expert_fc1_0", "w13_weight"),
        ("up", "expert_fc1_1", "w13_weight"),
        ("down", "expert_fc2", "w2_weight"),
    ):
        pair_id = f"model.layers.0.mlp.experts.{component}_proj.weight"
        weight_shape = [2, 256, 256]
        scale_shape = [2, 2, 2]
        source_names = {
            "0": [f"trainer.{source_recipe}.weight0"],
            "1": [f"trainer.{source_recipe}.weight1"],
        }
        destination_prefix = "model.layers.0.mlp.experts."
        weight_parameter = destination_prefix + parameter
        weight = {
            "name": pair_id,
            "family": "routed_expert",
            "pp_rank": 0,
            "dtype": "float8_e4m3fn",
            "global_shape": weight_shape,
            "pair_id": pair_id,
            "tensor_role": "weight",
            "source": {
                **_layout((1, 256, 256), [0, 1]),
                "names_by_rank": source_names,
                "recipe": source_recipe,
            },
            "destination": {
                **_layout((1, 256, 256), [2, 3]),
                "parameter": weight_parameter,
                "recipe": f"expert_{component}",
            },
        }
        scale = {
            "name": (
                f"model.layers.0.mlp.experts." f"{component}_proj.weight_scale_inv"
            ),
            "family": "routed_expert",
            "pp_rank": 0,
            "dtype": "float32",
            "global_shape": scale_shape,
            "pair_id": pair_id,
            "tensor_role": "scale",
            "source": {
                **_layout((1, 2, 2), [0, 1]),
                "names_by_rank": source_names,
                "recipe": f"{source_recipe}_scale",
            },
            "destination": {
                **_layout((1, 2, 2), [2, 3]),
                "parameter": f"{weight_parameter}_scale_inv",
                "recipe": f"expert_{component}_scale",
            },
        }
        entries.extend((weight, scale))
    entries.sort(key=lambda entry: entry["name"])
    return {
        "schema_version": 1,
        "communicator_world_size": 4,
        "quantization": deepcopy(_QUANTIZATION),
        "entries": entries,
    }


def _moe_tp_manifest():
    manifest = _manifest()
    for entry in manifest["entries"]:
        recipe = entry["destination"]["recipe"].removesuffix("_scale")
        is_scale = entry["tensor_role"] == "scale"
        if recipe in ("expert_gate", "expert_up"):
            shard_dim = 1
            local_shape = (2, 1, 2) if is_scale else (2, 128, 256)
        else:
            shard_dim = 2
            local_shape = (2, 2, 1) if is_scale else (2, 256, 128)
        entry["destination"]["placements"][1]["dim"] = shard_dim
        entry["destination"]["local_shape"] = list(local_shape)
    return manifest


def _bf16_moe_tp_manifest():
    manifest = _moe_tp_manifest()
    manifest.pop("quantization")
    manifest["entries"] = [
        entry for entry in manifest["entries"] if entry["tensor_role"] == "weight"
    ]
    for entry in manifest["entries"]:
        entry["dtype"] = "bfloat16"
        entry.pop("pair_id")
        entry.pop("tensor_role")
    return manifest


def _receiver(manifest=None, *, model=None, topology=None, comm_rank=2):
    receiver = object.__new__(NcclM2NReceiver)
    receiver.manifest = _manifest() if manifest is None else manifest
    receiver.model = _model() if model is None else model
    receiver.device = torch.device("cpu")
    receiver._world_size = 4
    receiver._topology = dict(_TOPOLOGY if topology is None else topology)
    receiver._comm_rank = comm_rank
    receiver._params = dict(receiver.model.named_parameters())
    return receiver


def _entry(manifest, name):
    return next(entry for entry in manifest["entries"] if entry["name"] == name)


def test_receiver_uses_custom_process_group_rank_with_nonzero_offset():
    pg = Mock()
    pg.rank.return_value = 2

    with (
        patch("sglang.srt.weight_sync.nccl_m2n._nccl_rl"),
        patch(
            "sglang.srt.weight_sync.nccl_m2n._warm_and_borrow_nccl_comm",
            return_value=123,
        ),
        patch("sglang.srt.weight_sync.nccl_m2n.dist.get_world_size", return_value=4),
        patch.object(NcclM2NReceiver, "_validate_manifest", return_value=[]),
        patch("sglang.srt.weight_sync.nccl_m2n.torch.cuda.Stream"),
    ):
        receiver = NcclM2NReceiver(
            pg=pg,
            manifest=_manifest(),
            model=_model(),
            device=torch.device("cpu"),
            topology=_TOPOLOGY,
            static_expert_placement=True,
        )

    assert receiver._comm_rank == 2
    pg.rank.assert_called_once_with()


def test_pp_receiver_rejects_entries_owned_by_another_stage():
    manifest = _manifest()
    manifest["pp_rank"] = 1
    receiver = _receiver(manifest)
    with pytest.raises(ValueError, match="another PP stage"):
        receiver._validate_manifest(4)


def test_receiver_orders_source_handoffs_on_its_stage_process_group():
    manifest = _manifest()
    # First weight/scale pair is sourced by rank 0 alone; later pairs use 0,1.
    for entry in manifest["entries"][:2]:
        source = entry["source"]
        source["mesh"] = [[0]]
        source["local_shape"][0] *= 2
        source["names_by_rank"] = {
            "0": [name for names in source["names_by_rank"].values() for name in names]
        }
    receiver = _receiver(manifest)
    receiver._pg = object()
    receiver.comm_ptr = 123
    receiver.stream = Mock()
    receiver._entries = receiver._validate_manifest(4)
    events = []
    m2n = Mock()
    m2n.reshard.side_effect = lambda *args, **kwargs: events.append(kwargs["src_mesh"])
    with (
        patch("sglang.srt.weight_sync.nccl_m2n._nccl_rl", return_value=m2n),
        patch("torch.cuda.current_stream"),
        patch("torch.cuda.stream", side_effect=lambda stream: nullcontext()),
        patch(
            "torch.distributed.barrier",
            side_effect=lambda group: events.append("handoff"),
        ) as barrier,
    ):
        receiver.receive()
    assert events == [[[0]], [[0]], "handoff", [[0, 1]], [[0, 1]], [[0, 1]], [[0, 1]]]
    barrier.assert_called_once_with(group=receiver._pg)


def test_pp_receivers_keep_separate_communicators_and_destination_storage_across_updates():
    model = _model()
    model.model.layers.append(deepcopy(model.model.layers[0]))
    manifests = [_manifest(), _manifest()]
    for pp_rank, manifest in enumerate(manifests):
        manifest["pp_rank"] = pp_rank
        for entry in manifest["entries"]:
            entry["pp_rank"] = pp_rank
            for field in ("name", "pair_id"):
                entry[field] = entry[field].replace("layers.0.", f"layers.{pp_rank}.")
            entry["destination"]["parameter"] = entry["destination"][
                "parameter"
            ].replace("layers.0.", f"layers.{pp_rank}.")
    m2n = Mock()
    update = 0

    def transfer(source, destination, comm_ptr, stream, **kwargs):
        assert source is None
        assert kwargs["src_mesh"] == [[0, 1]]
        assert kwargs["dst_mesh"] == [[2, 3]]
        destination.fill_((comm_ptr - 100) + update)

    m2n.reshard.side_effect = transfer
    groups = [Mock(), Mock()]
    for group in groups:
        group.rank.return_value = 2
    with (
        patch("sglang.srt.weight_sync.nccl_m2n._nccl_rl", return_value=m2n),
        patch(
            "sglang.srt.weight_sync.nccl_m2n._warm_and_borrow_nccl_comm",
            side_effect=[101, 102],
        ),
        patch("sglang.srt.weight_sync.nccl_m2n.dist.get_world_size", return_value=4),
        patch("torch.cuda.Stream"),
        patch("torch.cuda.current_stream"),
        patch("torch.cuda.stream", side_effect=lambda stream: nullcontext()),
    ):
        receivers = [
            NcclM2NReceiver(
                pg=group,
                manifest=manifest,
                model=model,
                device=torch.device("cpu"),
                topology=_TOPOLOGY,
                static_expert_placement=True,
            )
            for group, manifest in zip(groups, manifests)
        ]
        for update in (0, 2):
            receivers[0].receive()
            first_stage = model.model.layers[0].mlp.experts.w2_weight.detach().clone()
            receivers[1].receive()
            assert torch.equal(
                model.model.layers[0].mlp.experts.w2_weight.float(), first_stage.float()
            )
            for pp_rank in (0, 1):
                experts = model.model.layers[pp_rank].mlp.experts
                for parameter in experts.parameters():
                    assert torch.all(parameter.float() == pp_rank + 1 + update)
        assert [invocation.args[2] for invocation in m2n.reshard.call_args_list] == [
            101
        ] * 6 + [102] * 6 + [101] * 6 + [102] * 6
        receivers[0].destroy()
        receivers[0].destroy()
        assert receivers[0].comm_ptr is None
        m2n.finalize.assert_called_once()


@pytest.mark.parametrize("rank", [0, 1])
def test_receiver_accepts_ep1_with_moe_tensor_parallel_expert_shards(rank):
    topology = {
        **_MOE_TP_TOPOLOGY,
        "tp_rank": rank,
        "moe_tp_rank": rank,
    }
    NcclM2NReceiver._validate_topology(topology, static_expert_placement=True)
    receiver = _receiver(
        _moe_tp_manifest(),
        model=_moe_tp_model(rank),
        topology=topology,
        comm_rank=2 + rank,
    )

    records = receiver._validate_manifest(4)

    assert len(records) == 6
    for entry, _source, destination in records:
        recipe = entry["destination"]["recipe"].removesuffix("_scale")
        expected_dim = 1 if recipe in ("expert_gate", "expert_up") else 2
        assert destination.placements[1] == ("shard", expected_dim)


def test_receiver_rejects_hybrid_ep_and_moe_tensor_parallel_layout():
    topology = {
        **_TOPOLOGY,
        "tp_size": 4,
        "moe_ep_size": 2,
        "moe_tp_size": 2,
    }

    with pytest.raises(ValueError, match="EP=TP.*EP=1"):
        NcclM2NReceiver._validate_topology(topology, static_expert_placement=True)


def test_receiver_initialization_accepts_equal_numel_packed_fp8_weight_storage():
    model = _moe_tp_model()
    experts = model.model.layers[0].mlp.experts
    experts.w2_weight.data = experts.w2_weight.data.view(2, 2, 256, 64)
    receiver = _receiver(
        _moe_tp_manifest(),
        model=model,
        topology=_MOE_TP_TOPOLOGY,
    )

    receiver._entries = receiver._validate_manifest(
        4,
        allow_packed_expert_weights=True,
    )
    with pytest.raises(ValueError, match="incompatible with expert_down"):
        receiver._validate_manifest(4)

    receiver._prepare_fp8_destinations()

    assert experts.w2_weight.shape == (2, 256, 128)
    receiver._validate_manifest(4)


def test_receiver_restores_blocked_bf16_expert_weights_before_transfer():
    model = _bf16_moe_tp_model()
    experts = model.model.layers[0].mlp.experts
    experts.w13_weight.data = experts.w13_weight.data.view(2, 4, 256, 64)
    experts.w2_weight.data = experts.w2_weight.data.view(2, 2, 256, 64)
    w13_ptr = experts.w13_weight.data_ptr()
    w2_ptr = experts.w2_weight.data_ptr()
    receiver = _receiver(
        _bf16_moe_tp_manifest(),
        model=model,
        topology=_MOE_TP_TOPOLOGY,
    )

    receiver._entries = receiver._validate_manifest(
        4,
        allow_packed_expert_weights=True,
    )
    with pytest.raises(ValueError, match="incompatible with expert_down"):
        receiver._validate_manifest(4)

    receiver._prepare_unquantized_expert_destinations()

    assert experts.w13_weight.dtype == torch.bfloat16
    assert experts.w13_weight.shape == (2, 256, 256)
    assert experts.w13_weight.data_ptr() == w13_ptr
    assert experts.w2_weight.dtype == torch.bfloat16
    assert experts.w2_weight.shape == (2, 256, 128)
    assert experts.w2_weight.data_ptr() == w2_ptr
    receiver._validate_manifest(4)


def test_fp8_manifest_requires_exact_metadata_and_complete_atomic_pairs():
    receiver = _receiver()

    records = receiver._validate_manifest(4)

    assert len(records) == 6
    assert {
        (entry["pair_id"], entry["tensor_role"])
        for entry, _source, _destination in records
    } == {
        (
            f"model.layers.0.mlp.experts.{component}_proj.weight",
            role,
        )
        for component in ("gate", "up", "down")
        for role in ("weight", "scale")
    }

    bad_metadata = _manifest()
    bad_metadata["quantization"]["scale_format"] = "ue8m0"
    with pytest.raises(ValueError, match=r"(?i)(canonical|quantization)"):
        _receiver(bad_metadata)._validate_manifest(4)

    missing_scale = _manifest()
    missing_scale["entries"] = [
        entry
        for entry in missing_scale["entries"]
        if entry["name"] != "model.layers.0.mlp.experts.gate_proj.weight_scale_inv"
    ]
    with pytest.raises(ValueError, match=r"(?i)(pair|weight|scale)"):
        _receiver(missing_scale)._validate_manifest(4)

    missing_projection = _manifest()
    missing_projection["entries"] = [
        entry
        for entry in missing_projection["entries"]
        if entry["pair_id"] != "model.layers.0.mlp.experts.down_proj.weight"
    ]
    with pytest.raises(ValueError, match=r"(?i)(atomically|gate|up|down)"):
        _receiver(missing_projection)._validate_manifest(4)


@pytest.mark.parametrize(
    ("method_up_first", "trtllm_swaps_w13", "expected_up_first"),
    [
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, False),
    ],
)
def test_fp8_gate_up_weight_and_scale_slices_share_the_effective_order(
    method_up_first,
    trtllm_swaps_w13,
    expected_up_first,
):
    receiver = _receiver()
    experts = receiver.model.model.layers[0].mlp.experts
    experts.quant_method.load_up_proj_weight_first = method_up_first
    experts.use_flashinfer_trtllm_moe = trtllm_swaps_w13

    for component, weight_value, scale_value in (
        ("gate", 1, 3),
        ("up", 2, 4),
    ):
        weight_entry = _entry(
            receiver.manifest,
            f"model.layers.0.mlp.experts.{component}_proj.weight",
        )
        weight, post_weight = receiver._destination(
            weight_entry,
            (1, 256, 256),
        )
        weight.fill_(weight_value)
        assert post_weight is not None
        post_weight()

        scale_entry = _entry(
            receiver.manifest,
            (f"model.layers.0.mlp.experts." f"{component}_proj.weight_scale_inv"),
        )
        scale, post_scale = receiver._destination(
            scale_entry,
            (1, 2, 2),
        )
        scale.fill_(scale_value)
        assert post_scale is not None
        post_scale()

    gate_start, up_start = (256, 0) if expected_up_first else (0, 256)
    scale_gate_start, scale_up_start = (2, 0) if expected_up_first else (0, 2)
    w13 = experts.w13_weight
    w13_scale = experts.w13_weight_scale_inv
    assert torch.all(w13[:, gate_start : gate_start + 256] == 1)
    assert torch.all(w13[:, up_start : up_start + 256] == 2)
    assert torch.all(w13_scale[:, scale_gate_start : scale_gate_start + 2] == 3)
    assert torch.all(w13_scale[:, scale_up_start : scale_up_start + 2] == 4)


def test_bf16_gate_up_slices_use_flashinfer_trtllm_w31_order():
    receiver = _receiver(
        _bf16_moe_tp_manifest(),
        model=_bf16_moe_tp_model(),
        topology=_MOE_TP_TOPOLOGY,
    )
    experts = receiver.model.model.layers[0].mlp.experts
    experts.quant_method.load_up_proj_weight_first = False
    experts.use_flashinfer_trtllm_moe = True

    for component, value in (("gate", 1), ("up", 2)):
        entry = _entry(
            receiver.manifest,
            f"model.layers.0.mlp.experts.{component}_proj.weight",
        )
        destination, post_copy = receiver._destination(entry, (2, 128, 256))
        destination.fill_(value)
        assert post_copy is not None
        post_copy()

    assert torch.all(experts.w13_weight[:, :128] == 2)
    assert torch.all(experts.w13_weight[:, 128:] == 1)


def test_prepare_fp8_destinations_repeatedly_restores_canonical_scale_storage():
    receiver = _receiver()
    receiver._entries = receiver._validate_manifest(4)
    receiver._entries.sort(key=lambda record: record[0]["tensor_role"] != "scale")
    experts = receiver.model.model.layers[0].mlp.experts

    receiver._prepare_fp8_destinations()
    first_weight_ptr = experts.w13_weight.data_ptr()
    first_scale_storage = experts.w13_weight_scale_inv.data
    first_scale_ptr = experts.w13_weight_scale_inv.data_ptr()

    experts.w13_weight_scale_inv.data = torch.empty(
        (16,),
        dtype=torch.uint8,
    )
    experts.w13_weight_scale_inv.format_ue8m0 = True
    receiver._prepare_fp8_destinations()

    second_weight_ptr = experts.w13_weight.data_ptr()
    second_scale_ptr = experts.w13_weight_scale_inv.data_ptr()
    assert second_weight_ptr != first_weight_ptr
    assert second_scale_ptr != first_scale_ptr
    assert first_scale_storage.data_ptr() == first_scale_ptr
    assert experts.w13_weight.dtype == torch.float8_e4m3fn
    assert experts.w13_weight.shape == (1, 512, 256)
    assert experts.w13_weight_scale_inv.dtype == torch.float32
    assert experts.w13_weight_scale_inv.shape == (1, 4, 2)
    assert experts.w13_weight_scale_inv.is_contiguous()
    assert experts.w13_weight_scale_inv.format_ue8m0 is False

    # Model post-load may mutate the rebound canonical buffers in place.
    # A second begin/receive cycle must allocate fresh canonical storage.
    experts.w13_weight_scale_inv.format_ue8m0 = True
    receiver._prepare_fp8_destinations()

    assert experts.w13_weight.data_ptr() != second_weight_ptr
    assert experts.w13_weight_scale_inv.data_ptr() != second_scale_ptr
    assert experts.w13_weight_scale_inv.dtype == torch.float32
    assert experts.w13_weight_scale_inv.shape == (1, 4, 2)
    assert experts.w13_weight_scale_inv.format_ue8m0 is False
