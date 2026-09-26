# SPDX-License-Identifier: Apache-2.0
"""Checkpoint slicing must match the ordinary loader, before any device copy."""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.runtime.layers import linear
from sglang.multimodal_gen.runtime.loader import fsdp_load
from sglang.multimodal_gen.runtime.loader import rank_local_checkpoint as local
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3Attention


@contextmanager
def tp_context(size, rank):
    with (
        patch.object(linear, "get_tp_group", return_value=object()),
        patch.object(linear, "get_group_size", return_value=size),
        patch.object(linear, "get_group_rank", return_value=rank),
        patch.object(local, "get_tp_world_size", return_value=size),
        patch.object(local, "get_tp_rank", return_value=rank),
        patch.object(torch.distributed, "get_rank", return_value=rank),
    ):
        yield


class TinyModel(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.col = linear.ColumnParallelLinear(8, 16, bias=True)
        self.row = linear.RowParallelLinear(16, 8, bias=True)
        self.packed = linear.MergedColumnParallelLinear(8, [16, 8], bias=True)
        self.rep = linear.ReplicatedLinear(8, 4, bias=False)
        self.register_buffer("scalar", torch.empty((), dtype=torch.float32))
        self.register_buffer("fp32_buffer", torch.empty(4, dtype=torch.float32))
        self.grouped = MiniMaxH3Attention.__new__(MiniMaxH3Attention)
        nn.Module.__init__(self.grouped)
        self.grouped.tp_size = size
        arch = MiniMaxH3DiTArchConfig(
            hidden_size=8, num_attention_heads=8, attention_head_dim=2
        )
        self.grouped.qkv_proj = linear.MergedColumnParallelLinear(
            8, [16, 16, 16], bias=False, params_dtype=torch.bfloat16
        )
        self.grouped._install_qkv_weight_loader(arch)


def checkpoint():
    shapes = {
        "col.weight": (16, 8),
        "col.bias": (16,),
        "row.weight": (8, 16),
        "row.bias": (8,),
        "packed.weight": (24, 8),
        "packed.bias": (24,),
        "rep.weight": (4, 8),
        "scalar": (),
        "fp32_buffer": (4,),
        "grouped.qkv_proj.weight": (48, 8),
    }
    return {
        name: (
            torch.arange(torch.Size(shape).numel(), dtype=torch.float32) % 127
        ).reshape(shape)
        for name, shape in shapes.items()
    }


def materialize(model, weights, converted=None):
    fsdp_load.load_model_from_full_model_state_dict(
        model,
        iter(weights.items()),
        checkpoint_load_device=torch.device("cpu"),
        param_dtype=torch.bfloat16,
        strict=True,
        param_names_mapping=lambda name: (name, None, None),
        preconverted_state_dict=converted,
    )


@pytest.mark.parametrize("size,rank", [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)])
def test_rank_local_matches_ordinary_loader(tmp_path, size, rank):
    weights = checkpoint()
    # Multiple files must not change the logical packing order.
    files = []
    for index, items in enumerate(
        (list(weights.items())[::2], list(weights.items())[1::2])
    ):
        path = tmp_path / f"shard-{index}.safetensors"
        save_file(dict(items), path)
        files.append(str(path))
    with tp_context(size, rank), torch.device("meta"):
        reference, model = TinyModel(size), TinyModel(size)
    with tp_context(size, rank):
        materialize(reference, weights)
        converted = local.try_load_rank_local_tp_state_dict(
            model, files, lambda name: (name, None, None)
        )
        assert converted is not None
        materialize(model, {}, converted)
    assert model.state_dict().keys() == reference.state_dict().keys()
    for name, tensor in reference.state_dict().items():
        actual = model.state_dict()[name]
        assert actual.dtype == tensor.dtype
        assert torch.equal(actual, tensor), name
    assert model.fp32_buffer.dtype == torch.float32
    assert model.grouped.qkv_proj.weight.dtype == torch.bfloat16


@pytest.mark.parametrize("rank", [0, 1])
def test_fused_and_separate_merged_sources_agree(tmp_path, rank):
    first = torch.arange(128, dtype=torch.float32).reshape(16, 8)
    second = torch.arange(128, 192, dtype=torch.float32).reshape(8, 8)
    path = tmp_path / "split.safetensors"
    save_file({"up": first, "gate": second}, path)
    with tp_context(2, rank), torch.device("meta"):
        model = nn.Module()
        model.proj = linear.MergedColumnParallelLinear(8, [16, 8], bias=False)
    with tp_context(2, rank):
        converted = local.try_load_rank_local_tp_state_dict(
            model,
            [str(path)],
            lambda name: ("proj.weight", 0 if name == "up" else 1, 2),
        )
    assert converted is not None
    expected = torch.cat(
        (first[rank * 8 : (rank + 1) * 8], second[rank * 4 : (rank + 1) * 4])
    )
    assert torch.equal(converted[0]["proj.weight"].tensor, expected)


def test_grouped_qkv_reads_only_local_rows(tmp_path):
    path = tmp_path / "qkv.safetensors"
    weight = checkpoint()["grouped.qkv_proj.weight"]
    save_file({"weight": weight}, path)
    reads = []
    real_open = local.safe_open

    class Slice:
        def __init__(self, value):
            self.value = value

        def get_shape(self):
            return self.value.get_shape()

        def get_dtype(self):
            return self.value.get_dtype()

        def __getitem__(self, indices):
            result = self.value[indices]
            reads.append((indices, result.numel()))
            return result

    class Handle:
        def __init__(self, *args, **kwargs):
            self.context = real_open(*args, **kwargs)

        def __enter__(self):
            self.handle = self.context.__enter__()
            return self

        def __exit__(self, *args):
            return self.context.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_slice(self, name):
            return Slice(self.handle.get_slice(name))

        def get_tensor(self, name):
            raise AssertionError("Must not read a full QKV tensor")

    with tp_context(2, 1), torch.device("meta"):
        parent = TinyModel(2)
        model = nn.Module()
        model.register_parameter("weight", parent.grouped.qkv_proj.weight)
    with tp_context(2, 1), patch.object(local, "safe_open", Handle):
        converted = local.try_load_rank_local_tp_state_dict(
            model, [str(path)], lambda name: (name, None, None)
        )
    assert converted is not None
    assert reads == [((slice(24, 48), slice(None)), weight.numel() // 2)]


@pytest.mark.parametrize(
    "failure",
    ["custom", "shape", "quantized", "subclass", "metadata", "packed_sizes", "method"],
)
def test_unsupported_layout_falls_back_before_payload_read(tmp_path, failure):
    weights = checkpoint()
    with tp_context(2, 0), torch.device("meta"):
        model = TinyModel(2)
    if failure == "custom":
        model.col.weight.weight_loader = lambda *args: None
    elif failure == "shape":
        weights["col.weight"] = torch.ones(18, 8)
    elif failure == "quantized":
        weights["col.weight"] = weights["col.weight"].to(torch.float8_e4m3fn)
    elif failure == "subclass":

        class CustomParameter(nn.Parameter):
            pass

        model.col.weight = CustomParameter(model.col.weight)
    elif failure == "metadata":
        model.col.weight.is_metadata = True
    elif failure == "method":
        model.col.quant_method = object()
    else:
        model.packed.output_sizes = [15, 9]
    path = tmp_path / "model.safetensors"
    save_file(weights, path)
    with tp_context(2, 0), patch.object(local, "read_tp_local_tensor") as read:
        assert (
            local.try_load_rank_local_tp_state_dict(
                model, [str(path)], lambda name: (name, None, None)
            )
            is None
        )
        read.assert_not_called()


def test_merged_scalar_source_falls_back_before_payload_read(tmp_path):
    path = tmp_path / "scalar.safetensors"
    save_file({"up": torch.ones(16, 8), "gate": torch.tensor(1.0)}, path)
    with tp_context(2, 0), torch.device("meta"):
        model = nn.Module()
        model.proj = linear.MergedColumnParallelLinear(8, [16, 8], bias=False)
    with tp_context(2, 0), patch.object(local, "read_tp_local_tensor") as read:
        assert (
            local.try_load_rank_local_tp_state_dict(
                model,
                [str(path)],
                lambda name: ("proj.weight", 0 if name == "up" else 1, 2),
            )
            is None
        )
        read.assert_not_called()


@pytest.mark.parametrize("failure", ["shape", "dtype"])
def test_invalid_local_transform_is_rejected(tmp_path, failure):
    path = tmp_path / "model.safetensors"
    save_file(checkpoint(), path)
    with tp_context(2, 0), torch.device("meta"):
        model = TinyModel(2)
    model.grouped.qkv_proj.weight.rank_local_tp_weight_transform = (
        (lambda tensor: tensor[:1])
        if failure == "shape"
        else (lambda tensor: tensor.double())
    )
    with tp_context(2, 0), pytest.raises(RuntimeError, match="preserve shape, dtype"):
        local.try_load_rank_local_tp_state_dict(
            model, [str(path)], lambda name: (name, None, None)
        )


def test_tp1_keeps_ordinary_path(tmp_path):
    with tp_context(1, 0), torch.device("meta"):
        model = TinyModel(1)
    with (
        tp_context(1, 0),
        patch.object(local, "_collect_safetensors_sources") as collect,
    ):
        assert (
            local.try_load_rank_local_tp_state_dict(
                model, [], lambda name: (name, None, None)
            )
            is None
        )
        collect.assert_not_called()
