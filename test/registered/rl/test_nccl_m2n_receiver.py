"""Critical M2N receive/layout regressions; only graph replay requires CUDA."""

import weakref
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.weight_sync.nccl_m2n import M2NFP8Storage, NcclM2NReceiver

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
_MOE_TP_TOPOLOGY = {**_TOPOLOGY, "moe_ep_size": 1, "moe_tp_size": 2}
_QUANTIZATION = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    "weight_dtype": "float8_e4m3fn",
    "scale_dtype": "float32",
    "scale_format": "canonical",
}


def _model(*, fp8=True, moe_tp=False):
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    layer = root.model.layers[0]
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = experts = torch.nn.Module()
    shapes = (
        ((2, 256, 256), (2, 256, 128)) if moe_tp else ((1, 512, 256), (1, 256, 256))
    )
    for name, shape in zip(("w13_weight", "w2_weight"), shapes):
        setattr(
            experts,
            name,
            torch.nn.Parameter(
                torch.zeros(
                    shape, dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16
                ),
                requires_grad=False,
            ),
        )
        if fp8:
            scale_shape = (shape[0], shape[1] // 128, shape[2] // 128)
            setattr(
                experts,
                name + "_scale_inv",
                torch.nn.Parameter(torch.zeros(scale_shape), requires_grad=False),
            )
    experts.quant_method = SimpleNamespace(
        block_quant=True,
        use_mxfp8=False,
        is_fp4_expert=False,
        load_up_proj_weight_first=False,
        quant_config=SimpleNamespace(
            use_mxfp8=False,
            is_fp4_experts=False,
            is_checkpoint_fp8_serialized=fp8,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        ),
    )
    experts.moe_ep_rank = experts.moe_tp_rank = 0
    experts.moe_ep_size, experts.moe_tp_size = (1, 2) if moe_tp else (2, 1)
    experts._num_local_routed = 2 if moe_tp else 1
    experts._num_global_routed = 2
    return root


def _manifest(*, fp8=True, moe_tp=False):
    manifest = {"schema_version": 1, "communicator_world_size": 4, "entries": []}
    if fp8:
        manifest["quantization"] = deepcopy(_QUANTIZATION)
    for component, recipe, parameter in (
        ("gate", "expert_fc1_0", "w13_weight"),
        ("up", "expert_fc1_1", "w13_weight"),
        ("down", "expert_fc2", "w2_weight"),
    ):
        pair = f"model.layers.0.mlp.experts.{component}_proj.weight"
        for scale in ((False, True) if fp8 else (False,)):
            size = 2 if scale else 256
            suffix = "_scale" if scale else ""
            dim = (2 if component == "down" else 1) if moe_tp else 0
            shape = [2, size, size]
            local = shape.copy()
            local[dim] //= 2
            entry = {
                "name": pair + ("_scale_inv" if scale else ""),
                "family": "routed_expert",
                "pp_rank": 0,
                "dtype": (
                    "float32" if scale else ("float8_e4m3fn" if fp8 else "bfloat16")
                ),
                "global_shape": shape,
                "source": {
                    "mesh": [[0, 1]],
                    "placements": [{"type": "replicate"}, {"type": "shard", "dim": 0}],
                    "local_shape": [1, size, size],
                    "recipe": recipe + suffix,
                    "names_by_rank": {
                        str(rank): [f"trainer.{recipe}.weight{rank}"]
                        for rank in range(2)
                    },
                },
                "destination": {
                    "mesh": [[2, 3]],
                    "placements": [
                        {"type": "replicate"},
                        {"type": "shard", "dim": dim},
                    ],
                    "local_shape": local,
                    "recipe": f"expert_{component}" + suffix,
                    "parameter": f"model.layers.0.mlp.experts.{parameter}"
                    + ("_scale_inv" if scale else ""),
                },
            }
            if fp8:
                entry.update(pair_id=pair, tensor_role="scale" if scale else "weight")
            manifest["entries"].append(entry)
    manifest["entries"].sort(key=lambda entry: entry["name"])
    return manifest


def _receiver(manifest=None, *, model=None, topology=None):
    pg = Mock()
    pg.rank.return_value = 2
    with (
        patch("sglang.srt.weight_sync.nccl_m2n._nccl_m2n"),
        patch(
            "sglang.srt.weight_sync.nccl_m2n._warm_and_borrow_nccl_comm",
            return_value=123,
        ),
        patch("torch.distributed.get_world_size", return_value=4),
        patch("torch.cuda.Stream"),
    ):
        receiver = NcclM2NReceiver(
            pg=pg,
            manifest=_manifest() if manifest is None else manifest,
            model=_model() if model is None else model,
            device=torch.device("cpu"),
            topology=_TOPOLOGY if topology is None else topology,
            static_expert_placement=True,
        )
    # Rollout TP rank 0 is rank 2 in the trainer+rollout communicator.
    assert receiver._comm_rank == 2
    pg.rank.assert_called_once_with()
    return receiver


def _concurrent_pp_receivers(*, fp8=True, layers_per_stage=(1, 1)):
    model = _model() if fp8 else _model(fp8=False, moe_tp=True)
    template = model.model.layers[0]
    model.model.layers = torch.nn.ModuleList(
        [deepcopy(template) for _ in range(sum(layers_per_stage))]
    )
    base_manifest = _manifest() if fp8 else _manifest(fp8=False, moe_tp=True)
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
    handoffs = []
    for stage, receiver in enumerate(receivers):
        pair_size = 2 if fp8 else 1
        for entry in receiver.manifest["entries"][: pair_size * (stage + 1)]:
            source = entry["source"]
            source["mesh"] = [[0]]
            source["local_shape"][0] *= 2
            source["names_by_rank"] = {
                "0": [n for names in source["names_by_rank"].values() for n in names]
            }
        receiver._entries = receiver._validate_manifest(4)
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
        patch("sglang.srt.weight_sync.nccl_m2n._nccl_m2n", return_value=m2n),
        patch(
            "torch.distributed.barrier",
            side_effect=lambda group: handoffs.append(group),
        ),
        patch("torch.cuda.current_stream"),
        patch("torch.cuda.stream", side_effect=stream_context),
    ):
        for update in (0, 2):
            events.clear()
            handoffs.clear()
            NcclM2NReceiver.receive_many(receivers)
            assert events[:3] == [("enqueue", 101), ("enqueue", 102), ("sync", 101)]
            per_layer = 6 if fp8 else 3
            assert [comm for kind, comm in events if kind == "enqueue"] == (
                [101, 102] * per_layer + [102] * per_layer
            )
            assert all(not work for work in pending.values())
            assert handoffs == [receiver._pg for receiver in receivers]
            for layer, stage in enumerate((0, 1, 1)):
                for parameter in model.model.layers[layer].mlp.experts.parameters():
                    assert torch.all(parameter.float() == stage + 1 + update)


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
        patch("sglang.srt.weight_sync.nccl_m2n._nccl_m2n", return_value=m2n),
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
    with patch("sglang.srt.weight_sync.nccl_m2n._nccl_m2n", return_value=m2n):
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


@pytest.mark.parametrize(
    "layout_hint,contiguous,hidden,intermediate",
    [
        ("canonical", False, 6, 3),
        ("triton_kernel", False, 6, 3),
        ("parameter", True, 4, 4),
    ],
)
def test_bf16_refits_preserve_expert_layout_and_values(
    layout_hint, contiguous, hidden, intermediate
):
    # Include square W13 and square W2: shape inference cannot distinguish
    # their transposed layouts. Nonuniform values catch incorrect axis copies.
    model = _model(fp8=False, moe_tp=True)
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

    manifest = _manifest(fp8=False, moe_tp=True)
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
        m2n.reshard.side_effect = (
            lambda source, destination, *args, incoming=incoming, **kwargs: (
                destination.copy_(next(incoming))
            )
        )
        with (
            patch("sglang.srt.weight_sync.nccl_m2n._nccl_m2n", return_value=m2n),
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


@pytest.mark.parametrize(
    "method_first,trtllm", [(False, False), (True, False), (False, True), (True, True)]
)
def test_gate_up_weights_and_scales_use_the_same_effective_order(method_first, trtllm):
    receiver = _receiver()
    experts = receiver.model.model.layers[0].mlp.experts
    experts.quant_method.load_up_proj_weight_first = method_first
    experts.use_flashinfer_trtllm_moe = trtllm
    for entry in receiver.manifest["entries"]:
        component = entry["destination"]["recipe"].removesuffix("_scale")
        value = {"expert_gate": 1, "expert_up": 2, "expert_down": 3}[component]
        if entry["tensor_role"] == "scale":
            value += 4
        tensor, copy_back = receiver._destination(
            entry, tuple(entry["destination"]["local_shape"])
        )
        tensor.fill_(value)
        if copy_back is not None:
            copy_back()
    order = [2, 1] if method_first != trtllm else [1, 2]
    for weight, value in zip(experts.w13_weight.chunk(2, dim=1), order):
        assert torch.all(weight.float() == value)
    for scale, value in zip(experts.w13_weight_scale_inv.chunk(2, dim=1), order):
        assert torch.all(scale == value + 4)


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


def _pack_scales_reference(scales):
    """Mock DeepGEMM packing, including row expansion and partial packed words."""
    exponents = scales.repeat_interleave(128, dim=1).view(torch.int32) >> 23
    result = torch.zeros(
        (*exponents.shape[:-1], (exponents.shape[-1] + 3) // 4),
        dtype=torch.int32,
        device=scales.device,
    )
    for byte in range(4):
        part = exponents[..., byte::4]
        result[..., : part.shape[-1]] |= part << (8 * byte)
    return result


@pytest.mark.parametrize(
    "device,moe_tp,up_first",
    [
        ("cpu", False, False),
        ("cpu", True, True),
        pytest.param(
            "cuda",
            True,
            False,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA graphs"
            ),
        ),
    ],
)
def test_ue8m0_refits_only_pack_scales_and_never_replace_inference_storage(
    device, moe_tp, up_first
):
    model = _model(moe_tp=moe_tp).to(device)
    experts = model.model.layers[0].mlp.experts
    experts.quant_method.is_deepgemm_moe_runner_backend_enabled = Mock(
        return_value=True
    )
    experts.quant_method.load_up_proj_weight_first = up_first
    _mock_fp8_postprocess(model, packed=True)
    manifest = _manifest(moe_tp=moe_tp)
    manifest["quantization"]["scale_format"] = "ue8m0_unpacked"
    receiver = _receiver(
        manifest, model=model, topology=_MOE_TP_TOPOLOGY if moe_tp else _TOPOLOGY
    )
    receiver.device = torch.device(device)
    buffers = {name: param.detach() for name, param in model.named_parameters()}
    graph = None
    if device == "cuda":
        output = torch.empty((), device=device)

        def read_weights():
            output.copy_(sum(value.float().sum() for value in buffers.values()))

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
        storage = M2NFP8Storage(model, [manifest])
        assert not storage._buffers  # No copy-back safety net for this path.
        payloads = {}
        for index, entry in enumerate(manifest["entries"]):
            shape = entry["destination"]["local_shape"]
            values = torch.arange(
                shape[0] * shape[1] * shape[2], device=device
            ).reshape(shape)
            payloads[entry["destination"]["recipe"]] = (
                (2.0 ** (values % 3 - 4 + index + update)).float()
                if entry["tensor_role"] == "scale"
                else (values % 7 + index + update).to(torch.float8_e4m3fn)
            )
        incoming = iter(manifest["entries"])

        def transfer(source, destination, *args, **kwargs):
            for name, parameter in model.named_parameters():
                assert parameter.data_ptr() == buffers[name].data_ptr()
                assert parameter.stride() == buffers[name].stride()
            entry = next(incoming)
            recipe = entry["destination"]["recipe"]
            if recipe == "expert_down":
                assert destination.data_ptr() == experts.w2_weight.data_ptr()
            destination.copy_(payloads[recipe])

        m2n = Mock()
        m2n.reshard.side_effect = transfer
        with (
            patch("sglang.srt.weight_sync.nccl_m2n._nccl_m2n", return_value=m2n),
            patch(
                "sglang.srt.weight_sync.nccl_m2n._pack_fp8_scales",
                side_effect=_pack_scales_reference,
            ) as pack,
            patch("torch.cuda.current_stream"),
            patch("torch.cuda.stream", side_effect=lambda stream: nullcontext()),
        ):
            receiver.receive()
        assert pack.call_count == 3
        order = (
            ("expert_up", "expert_gate") if up_first else ("expert_gate", "expert_up")
        )
        for parameter, components in (
            ("w13_weight", order),
            ("w2_weight", ("expert_down",)),
        ):
            for scale in (False, True):
                suffix = "_scale" if scale else ""
                values = [
                    payloads[component + suffix].float() for component in components
                ]
                expected = torch.cat(values, dim=1) if len(values) == 2 else values[0]
                if scale:
                    expected = _pack_scales_reference(expected)
                name = f"model.layers.0.mlp.experts.{parameter}" + (
                    "_scale_inv" if scale else ""
                )
                actual = model.get_parameter(name)
                assert actual.data_ptr() == buffers[name].data_ptr()
                assert actual.stride() == buffers[name].stride()
                torch.testing.assert_close(
                    actual.float(), expected.float(), rtol=0, atol=0
                )
                if scale:
                    # The finalization hook must skip requantization.
                    assert actual.format_ue8m0
        if graph is not None:
            graph.replay()
            assert (
                output.item()
                == sum(value.float().sum() for value in buffers.values()).item()
            )


@pytest.mark.parametrize(
    "scale_format,deepgemm,packed",
    [
        ("canonical", True, True),
        ("ue8m0_unpacked", False, True),
        ("ue8m0_unpacked", True, False),
    ],
)
def test_packing_only_requires_explicit_wire_format_and_compatible_backend(
    scale_format, deepgemm, packed
):
    model = _model()
    experts = model.model.layers[0].mlp.experts
    experts.quant_method.is_deepgemm_moe_runner_backend_enabled = Mock(
        return_value=deepgemm
    )
    _mock_fp8_postprocess(model, packed=packed)
    manifest = _manifest()
    manifest["quantization"]["scale_format"] = scale_format
    receiver = _receiver(manifest, model=model)
    storage = M2NFP8Storage(model, [manifest])
    assert set(storage._buffers) == set(dict(model.named_parameters()))
    receiver._prepare_fp8_destinations()
    assert not receiver._packing_only_fp8_parameters
    assert not experts.w13_weight_scale_inv.format_ue8m0


@pytest.mark.parametrize(
    "device,packed,replace_parameters",
    [
        ("cpu", False, False),
        ("cpu", True, True),
        pytest.param(
            "cuda",
            True,
            True,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA graphs"
            ),
        ),
    ],
)
def test_fp8_refits_preserve_storage_and_captured_graph(
    device, packed, replace_parameters
):
    receiver = _receiver(
        _manifest(moe_tp=True),
        model=_model(moe_tp=True).to(device),
        topology=_MOE_TP_TOPOLOGY,
    )
    receiver.device = torch.device(device)
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
    graph = None
    if device == "cuda":
        output = torch.empty((), device=device)

        def read_weights():
            output.copy_(sum(value.float().sum() for value in graph_buffers.values()))

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                read_weights()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            read_weights()
    m2n = Mock()

    for update in (1, 2):
        storage = M2NFP8Storage(receiver.model, [receiver.manifest])
        m2n.reshard.side_effect = (
            lambda source, destination, *args, update=update, **kwargs: (
                destination.fill_(update)
            )
        )
        with (
            patch("sglang.srt.weight_sync.nccl_m2n._nccl_m2n", return_value=m2n),
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
        if graph is not None:
            graph.replay()
            assert output.item() == update * sum(
                value.numel() for value in graph_buffers.values()
            )
