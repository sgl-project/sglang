import weakref
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.weight_sync.nccl_m2n import M2NFP8Storage, NcclM2NReceiver

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


def _concurrent_pp_receivers(*, fp8=True, layers_per_stage=(1, 1)):
    model = _model() if fp8 else _bf16_moe_tp_model()
    template = model.model.layers[0]
    model.model.layers = torch.nn.ModuleList(
        [deepcopy(template) for _ in range(sum(layers_per_stage))]
    )
    base_manifest = _manifest() if fp8 else _bf16_moe_tp_manifest()
    receivers = []
    layer = 0
    for stage, count in enumerate(layers_per_stage):
        manifest = deepcopy(base_manifest)
        manifest["pp_rank"] = stage
        manifest["entries"] = []
        for _ in range(count):
            entries = deepcopy(base_manifest["entries"])
            for entry in entries:
                entry["pp_rank"] = stage
                for field in ("name", "pair_id"):
                    if field in entry:
                        entry[field] = entry[field].replace(
                            "layers.0.", f"layers.{layer}."
                        )
                entry["destination"]["parameter"] = entry["destination"][
                    "parameter"
                ].replace("layers.0.", f"layers.{layer}.")
            manifest["entries"].extend(entries)
            layer += 1
        receiver = _receiver(
            manifest, model=model, topology=_TOPOLOGY if fp8 else _MOE_TP_TOPOLOGY
        )
        receiver.comm_ptr = 101 + stage
        receiver._pg = object()
        receiver.stream = Mock()
        receiver._entries = receiver._validate_manifest(4)
        receivers.append(receiver)
    return model, receivers


@pytest.mark.parametrize("fp8", [False, True])
def test_concurrent_pp_receives_interleave_streams_and_keep_buffers_alive(fp8):
    model, receivers = _concurrent_pp_receivers(fp8=fp8, layers_per_stage=(1, 2))
    events = []
    pending = {receiver.comm_ptr: [] for receiver in receivers}
    active_stream = None
    update = 0

    @contextmanager
    def stream_context(stream):
        nonlocal active_stream
        # A suspended stage must exit its CUDA stream context before yielding.
        assert active_stream is None
        active_stream = stream
        try:
            yield
        finally:
            active_stream = None

    def drain(receiver):
        assert active_stream is None
        events.append(("sync", receiver.comm_ptr))
        for callback in pending[receiver.comm_ptr]:
            callback()
        pending[receiver.comm_ptr].clear()

    def transfer(source, destination, comm_ptr, stream, **kwargs):
        assert active_stream is stream
        events.append(("enqueue", comm_ptr))
        reference = weakref.ref(destination)

        def complete():
            tensor = reference()
            assert (
                tensor is not None
            ), "receive buffer released before its stream completed"
            tensor.fill_(comm_ptr - 100 + update)

        pending[comm_ptr].append(complete)

    for receiver in receivers:
        receiver.stream.synchronize.side_effect = lambda receiver=receiver: drain(
            receiver
        )
        destination_impl = receiver._destination

        def destination(entry, shape, receiver=receiver, impl=destination_impl):
            tensor, copy_back = impl(entry, shape)
            if copy_back is None:
                return tensor, None

            def enqueue_copy():
                assert active_stream is receiver.stream
                pending[receiver.comm_ptr].append(copy_back)

            return tensor, enqueue_copy

        receiver._destination = destination

    m2n = Mock()
    # Do not use Mock call recording here: it would itself retain destinations
    # and hide a premature release by the receiver.
    m2n.reshard = transfer
    with (
        patch("sglang.srt.weight_sync.nccl_m2n._nccl_rl", return_value=m2n),
        patch("torch.cuda.current_stream"),
        patch("torch.cuda.stream", side_effect=stream_context),
    ):
        for update in (0, 2):
            events.clear()
            NcclM2NReceiver.receive_many(receivers)
            assert events[:3] == [("enqueue", 101), ("enqueue", 102), ("sync", 101)]
            per_layer = 6 if fp8 else 3
            assert [comm for kind, comm in events if kind == "enqueue"] == (
                [101, 102] * per_layer + [102] * per_layer
            )
            assert all(not work for work in pending.values())
            for layer, stage in enumerate((0, 1, 1)):
                for parameter in model.model.layers[layer].mlp.experts.parameters():
                    assert torch.all(parameter.float() == stage + 1 + update)


def test_concurrent_pp_handoffs_use_only_their_own_stage_group():
    _, receivers = _concurrent_pp_receivers()
    events = []
    for index, receiver in enumerate(receivers):
        # Stagger the handoffs: PP0 changes source mesh after its first pair,
        # PP1 after its second. Neither handoff may use the other stage's group.
        for entry in receiver.manifest["entries"][: 2 * (index + 1)]:
            source = entry["source"]
            source["mesh"] = [[0]]
            source["local_shape"][0] *= 2
            source["names_by_rank"] = {
                "0": [
                    name for names in source["names_by_rank"].values() for name in names
                ]
            }
        receiver._entries = receiver._validate_manifest(4)
    m2n = Mock()
    m2n.reshard.side_effect = lambda src, dst, comm, stream, **kw: events.append(
        ("receive", comm, kw["src_mesh"])
    )
    with (
        patch("sglang.srt.weight_sync.nccl_m2n._nccl_rl", return_value=m2n),
        patch("torch.cuda.current_stream"),
        patch("torch.cuda.stream", side_effect=lambda stream: nullcontext()),
        patch(
            "torch.distributed.barrier",
            side_effect=lambda group: events.append(("handoff", group)),
        ),
    ):
        NcclM2NReceiver.receive_many(receivers)
    expected = []
    for entry in range(6):
        for stage, receiver in enumerate(receivers):
            handoff = 2 * (stage + 1)
            if entry == handoff:
                expected.append(("handoff", receiver._pg))
            expected.append(
                ("receive", receiver.comm_ptr, [[0]] if entry < handoff else [[0, 1]])
            )
    assert events == expected


def test_concurrent_receive_failure_drains_all_streams_before_releasing_buffers():
    _, receivers = _concurrent_pp_receivers()
    references = []
    drains = []

    def transfer(source, destination, *args, **kwargs):
        references.append(weakref.ref(destination))

    def drain(index):
        assert len(references) == 2
        assert all(reference() is not None for reference in references)
        drains.append(index)
        if index == 0:
            raise RuntimeError("secondary drain failure")

    for index, receiver in enumerate(receivers):
        receiver.stream.synchronize.side_effect = lambda index=index: drain(index)
    destination_impl = receivers[1]._destination

    def failing_destination(entry, shape):
        tensor, _ = destination_impl(entry, shape)

        def fail():
            raise RuntimeError("injected copy failure")

        return tensor, fail

    receivers[1]._destination = failing_destination
    m2n = Mock()
    m2n.reshard = transfer
    with (
        patch("sglang.srt.weight_sync.nccl_m2n._nccl_rl", return_value=m2n),
        patch("torch.cuda.current_stream"),
        patch("torch.cuda.stream", side_effect=lambda stream: nullcontext()),
        pytest.raises(RuntimeError, match="injected copy failure"),
    ):
        NcclM2NReceiver.receive_many(receivers)
    assert drains == [0, 1]
    assert references[0]() is not None
    with pytest.raises(RuntimeError, match="destroyed before retrying"):
        receivers[0]._prepare_receive()
    # The failed receiver retains its buffer through a failed teardown too.
    with patch("sglang.srt.weight_sync.nccl_m2n._nccl_rl", return_value=m2n):
        receivers[0].stream.synchronize.side_effect = RuntimeError("drain failed again")
        with pytest.raises(RuntimeError, match="drain failed again"):
            receivers[0].destroy()
        assert references[0]() is not None
        receivers[0].stream.synchronize.side_effect = None
        receivers[0].destroy()
    assert receivers[0]._failed_receive_buffers is None
    assert receivers[0].comm_ptr is None


@pytest.mark.parametrize("conflict", ["communicator", "stage", "parameter", "model"])
def test_concurrent_receive_rejects_unsafe_batches_before_launch(conflict):
    _, receivers = _concurrent_pp_receivers()
    if conflict == "communicator":
        receivers[1].comm_ptr = receivers[0].comm_ptr
    elif conflict == "stage":
        receivers[1].manifest["pp_rank"] = 0
    elif conflict == "parameter":
        receivers[1].manifest["entries"] = receivers[0].manifest["entries"]
    else:
        receivers[1].model = _model()
    for receiver in receivers:
        receiver._prepare_receive = Mock()
    with pytest.raises(ValueError):
        NcclM2NReceiver.receive_many(receivers)
    for receiver in receivers:
        receiver._prepare_receive.assert_not_called()


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
    experts.use_flashinfer_trtllm_moe = True
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


@pytest.mark.parametrize("layout_hint", ["triton_kernel", "parameter", "canonical"])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("hidden,intermediate", [(6, 3), (4, 4), (5, 3)])
def test_bf16_refits_preserve_expert_layout_and_values(
    layout_hint, contiguous, hidden, intermediate
):
    # Include square W13 and square W2: shape inference cannot distinguish
    # their transposed layouts. Nonuniform values catch incorrect axis copies.
    model = _bf16_moe_tp_model()
    experts = model.model.layers[0].mlp.experts
    transposed = layout_hint != "canonical"
    experts.use_triton_kernels = layout_hint == "triton_kernel"
    for name, shape in (
        ("w13_weight", (2, 2 * intermediate, hidden)),
        ("w2_weight", (2, hidden, intermediate)),
    ):
        value = torch.zeros(shape, dtype=torch.bfloat16)
        if transposed:
            value = value.transpose(1, 2)
        elif not contiguous:
            value = value.transpose(1, 2).contiguous().transpose(1, 2)
        if contiguous:
            value = value.contiguous()
        param = torch.nn.Parameter(value, requires_grad=False)
        param.is_transposed = layout_hint == "parameter"
        setattr(experts, name, param)

    manifest = _bf16_moe_tp_manifest()
    for entry in manifest["entries"]:
        is_down = entry["destination"]["recipe"] == "expert_down"
        entry["global_shape"] = (
            [2, hidden, 2 * intermediate] if is_down else [2, 2 * intermediate, hidden]
        )
        entry["source"]["local_shape"] = [1, *entry["global_shape"][1:]]
        entry["destination"]["local_shape"] = (
            [2, hidden, intermediate] if is_down else [2, intermediate, hidden]
        )
    receiver = _receiver(manifest, model=model, topology=_MOE_TP_TOPOLOGY)
    receiver._entries = receiver._validate_manifest(4, allow_packed_expert_weights=True)
    receiver._pg = object()
    receiver.comm_ptr = 123
    receiver.stream = Mock()
    graph_weights = [experts.w13_weight.detach(), experts.w2_weight.detach()]
    pointers = [value.data_ptr() for value in graph_weights]
    strides = [value.stride() for value in graph_weights]
    m2n = Mock()

    for update in (1, 2):
        payloads = {}
        for index, entry in enumerate(manifest["entries"]):
            shape = entry["destination"]["local_shape"]
            value = torch.arange(shape[0] * shape[1] * shape[2]).reshape(shape)
            payloads[entry["destination"]["recipe"]] = (
                (value % 17 + index * 19 + update) / 32
            ).to(torch.bfloat16)
        incoming = iter(payloads.values())
        m2n.reshard.side_effect = lambda source, destination, *args, **kwargs: (
            destination.copy_(next(incoming))
        )
        with (
            patch("sglang.srt.weight_sync.nccl_m2n._nccl_rl", return_value=m2n),
            patch("torch.cuda.current_stream"),
            patch("torch.cuda.stream", side_effect=lambda stream: nullcontext()),
        ):
            receiver.receive()

        canonical = [
            torch.cat((payloads["expert_gate"], payloads["expert_up"]), dim=1),
            payloads["expert_down"],
        ]
        for index, param in enumerate((experts.w13_weight, experts.w2_weight)):
            expected = (
                canonical[index].transpose(1, 2) if transposed else canonical[index]
            )
            assert param.shape == expected.shape
            assert param.data_ptr() == pointers[index]
            assert param.stride() == strides[index]
            torch.testing.assert_close(param, expected)
            torch.testing.assert_close(graph_weights[index], expected)

        # Exercise both GEMMs with the stored inference orientation.
        x = torch.arange(2 * hidden, dtype=torch.float32).reshape(1, 2, hidden)
        x = x.expand(2, -1, -1) / 8
        w13, w2 = [value.float() for value in graph_weights]
        if not transposed:
            w13, w2 = w13.transpose(1, 2), w2.transpose(1, 2)
        gate, up = torch.bmm(x, w13).chunk(2, dim=-1)
        actual = torch.bmm(torch.nn.functional.silu(gate) * up, w2)
        gate, up = torch.bmm(x, canonical[0].float().transpose(1, 2)).chunk(2, dim=-1)
        expected = torch.bmm(
            torch.nn.functional.silu(gate) * up, canonical[1].float().transpose(1, 2)
        )
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("triton_kernel", [False, True])
def test_receiver_rejects_unknown_bf16_blocked_layouts(triton_kernel):
    model = _bf16_moe_tp_model()
    experts = model.model.layers[0].mlp.experts
    experts.use_triton_kernels = triton_kernel
    experts.w2_weight.data = experts.w2_weight.data.view(2, 2, 256, 64)
    receiver = _receiver(
        _bf16_moe_tp_manifest(), model=model, topology=_MOE_TP_TOPOLOGY
    )
    with pytest.raises(ValueError, match="incompatible with expert_down"):
        receiver._validate_manifest(4, allow_packed_expert_weights=True)


def test_receiver_rejects_canonical_shape_for_transposed_bf16_backend():
    model = _bf16_moe_tp_model()
    model.model.layers[0].mlp.experts.use_triton_kernels = True
    receiver = _receiver(
        _bf16_moe_tp_manifest(), model=model, topology=_MOE_TP_TOPOLOGY
    )
    with pytest.raises(ValueError, match="incompatible with expert_down"):
        receiver._validate_manifest(4, allow_packed_expert_weights=True)


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


def _mock_fp8_postprocess(model, *, packed, replace_parameters=False):
    """Simulate rebinding by quant hooks; this does not test FP8 numerics."""
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            is_scale = name.endswith("_scale_inv")
            value = param.detach().clone()
            if packed and is_scale:
                weight = getattr(module, name.removesuffix("_scale_inv"))
                rows = weight.shape[-2]
                words = (weight.shape[-1] // 128 + 3) // 4
                # Packed DeepGEMM scales use int32 with a padded, transposed
                # layout. Retain nontrivial strides to catch contiguous rebinds.
                value = torch.empty(
                    (weight.shape[0], words, rows + 4),
                    dtype=torch.int32,
                    device=param.device,
                )[:, :, :rows].transpose(1, 2)
                value.fill_(int(param.flatten()[0].item()))
            if replace_parameters:
                replacement = torch.nn.Parameter(value, requires_grad=False)
                replacement.__dict__.update(param.__dict__)
                setattr(module, name, replacement)
                param = replacement
            else:
                param.data = value
            if is_scale:
                param.format_ue8m0 = packed


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("replace_parameters", [False, True])
def test_fp8_refits_restore_graph_buffers_after_postprocess(packed, replace_parameters):
    receiver = _receiver(
        _moe_tp_manifest(), model=_moe_tp_model(), topology=_MOE_TP_TOPOLOGY
    )
    _mock_fp8_postprocess(receiver.model, packed=packed)
    receiver._params = dict(receiver.model.named_parameters())
    receiver._entries = receiver._validate_manifest(4)
    receiver._pg = object()
    receiver.comm_ptr = 123
    receiver.stream = Mock()
    # These references model the pointers/strides captured at model startup.
    graph_buffers = {
        name: param.detach() for name, param in receiver.model.named_parameters()
    }
    pointers = {name: value.data_ptr() for name, value in graph_buffers.items()}
    strides = {name: value.stride() for name, value in graph_buffers.items()}
    if packed:
        scales = receiver.model.model.layers[0].mlp.experts.w13_weight_scale_inv
        assert not scales.is_contiguous()
    m2n = Mock()

    for update in (1, 2):
        storage = M2NFP8Storage(receiver.model, [receiver.manifest])
        m2n.reshard.side_effect = lambda source, destination, *args, **kwargs: (
            destination.fill_(update)
        )
        with (
            patch("sglang.srt.weight_sync.nccl_m2n._nccl_rl", return_value=m2n),
            patch("torch.cuda.current_stream"),
            patch("torch.cuda.stream", side_effect=lambda stream: nullcontext()),
        ):
            receiver.receive()
        _mock_fp8_postprocess(
            receiver.model, packed=packed, replace_parameters=replace_parameters
        )
        storage.restore(receiver.model)

        for name, param in receiver.model.named_parameters():
            assert param.data_ptr() == pointers[name]
            assert param.stride() == strides[name]
            assert torch.all(graph_buffers[name].float() == update)
            if name.endswith("_scale_inv"):
                assert param.format_ue8m0 is packed
                assert param.dtype == (torch.int32 if packed else torch.float32)


def test_fp8_storage_covers_all_pp_groups_but_not_residual_parameters():
    receiver = _receiver()
    model = receiver.model
    model.model.layers.append(deepcopy(model.model.layers[0]))
    second_manifest = deepcopy(receiver.manifest)
    for entry in second_manifest["entries"]:
        entry["destination"]["parameter"] = entry["destination"]["parameter"].replace(
            "layers.0.", "layers.1."
        )
    _mock_fp8_postprocess(model, packed=True)
    model.residual = _parameter((2,), torch.bfloat16)
    old = {name: param.detach() for name, param in model.named_parameters()}
    storage = M2NFP8Storage(model, [receiver.manifest, second_manifest])
    for param in model.parameters():
        param.data = torch.full_like(param, 2)
    residual_ptr = model.residual.data_ptr()

    storage.restore(model)

    for name, param in model.named_parameters():
        if name == "residual":
            assert param.data_ptr() == residual_ptr != old[name].data_ptr()
        else:
            assert param.data_ptr() == old[name].data_ptr()
            assert torch.all(old[name].float() == 2)


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "missing"])
def test_fp8_storage_rejects_incompatible_postprocess_and_remains_retryable(mismatch):
    receiver = _receiver()
    model = receiver.model
    storage = M2NFP8Storage(model, [receiver.manifest])
    old = {name: param.detach() for name, param in model.named_parameters()}
    for param in model.parameters():
        param.data = torch.full_like(param, 2)
    experts = model.model.layers[0].mlp.experts
    scale = experts.w2_weight_scale_inv
    valid = scale.data
    if mismatch == "missing":
        del experts.w2_weight_scale_inv
    elif mismatch == "shape":
        scale.data = scale.data.flatten()
    else:
        scale.data = scale.data.to(torch.int32)

    with pytest.raises(RuntimeError, match="Cannot resume generation"):
        storage.restore(model)
    # Validation must finish before any of the graph buffers is overwritten.
    assert all(torch.all(value.float() == 0) for value in old.values())

    experts.w2_weight_scale_inv = scale
    scale.data = valid
    storage.restore(model)
    assert all(torch.all(value.float() == 2) for value in old.values())


def test_fp8_storage_handles_a_postprocess_view_aliasing_the_original_buffer():
    receiver = _receiver()
    storage = M2NFP8Storage(receiver.model, [receiver.manifest])
    param = receiver.model.model.layers[0].mlp.experts.w2_weight_scale_inv
    original = param.detach()
    param.data.copy_(torch.arange(4, dtype=param.dtype).reshape_as(param))
    param.data = param.data.transpose(1, 2)
    expected = param.detach().clone()

    storage.restore(receiver.model)

    assert param.data_ptr() == original.data_ptr()
    assert param.stride() == original.stride()
    torch.testing.assert_close(original, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
def test_fp8_refit_updates_previously_captured_cuda_graph():
    receiver = _receiver(
        _moe_tp_manifest(), model=_moe_tp_model().cuda(), topology=_MOE_TP_TOPOLOGY
    )
    model = receiver.model
    _mock_fp8_postprocess(model, packed=True)
    experts = model.model.layers[0].mlp.experts
    output = torch.empty((), device="cuda")

    def read_weights():
        output.copy_(
            experts.w13_weight.float().sum()
            + experts.w13_weight_scale_inv.float().sum()
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            read_weights()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        read_weights()

    for update in (1, 2):
        storage = M2NFP8Storage(model, [receiver.manifest])
        receiver._params = dict(model.named_parameters())
        receiver._entries = receiver._validate_manifest(4)
        receiver._prepare_fp8_destinations()
        for param in model.parameters():
            param.data.fill_(update)
        _mock_fp8_postprocess(model, packed=True, replace_parameters=True)
        storage.restore(model)
        graph.replay()
        expected = update * (
            experts.w13_weight.numel() + experts.w13_weight_scale_inv.numel()
        )
        assert output.item() == expected
