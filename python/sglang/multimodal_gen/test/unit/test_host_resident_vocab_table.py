"""A declared vocab table stays in host memory; the gather runs there."""

from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.quantization.comfy_int8 import (
    ComfyInt8EmbeddingMethod,
)
from sglang.multimodal_gen.runtime.managers.memory_managers import layerwise_offload
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    _host_resident_tables,
    detach_host_resident_tables,
    restore_host_resident_tables,
)

THRESHOLD_PATH = f"{layerwise_offload.__name__}.HOST_RESIDENT_TABLE_MIN_BYTES"


class _Declared(torch.nn.Module):
    host_resident_table_names = ["embed"]

    def __init__(self, num_embeddings: int = 4096, dim: int = 64):
        super().__init__()
        self.embed = torch.nn.Embedding(num_embeddings, dim)
        self.proj = torch.nn.Linear(dim, dim)


class _Undeclared(torch.nn.Module):
    def __init__(self, num_embeddings: int = 4096, dim: int = 64):
        super().__init__()
        self.embed = torch.nn.Embedding(num_embeddings, dim)


class _Nested(torch.nn.Module):
    host_resident_table_names = ["language_model.embed"]

    def __init__(self):
        super().__init__()
        self.language_model = _Undeclared()


class _Int8Embedding(torch.nn.Module):
    def __init__(self, tensorwise: bool):
        super().__init__()
        self.quant_method = ComfyInt8EmbeddingMethod(tensorwise=tensorwise)
        self.quant_method.create_weights(self, 4, [2], 4, 2, torch.bfloat16)
        self.weight.copy_(torch.arange(8, dtype=torch.int8).reshape(2, 4))
        self.weight_scale.fill_(0.5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.quant_method.embedding(self, input_ids)


class TestSelection:
    def test_a_declared_table_is_selected(self):
        model = _Declared()
        with patch(THRESHOLD_PATH, 1024):
            assert _host_resident_tables(model) == [model.embed]

    def test_an_undeclared_table_is_left_alone(self):
        # the regression guard: a third-party backbone may read the weight
        # outside forward, and a forward hook would not cover that
        with patch(THRESHOLD_PATH, 1024):
            assert _host_resident_tables(_Undeclared()) == []

    def test_a_dotted_path_resolves(self):
        model = _Nested()
        with patch(THRESHOLD_PATH, 1024):
            assert _host_resident_tables(model) == [model.language_model.embed]

    def test_a_missing_declared_path_is_skipped(self):
        model = _Declared()
        model.host_resident_table_names = ["not_there"]
        with patch(THRESHOLD_PATH, 1024):
            assert _host_resident_tables(model) == []

    def test_a_small_table_is_left_alone(self):
        with patch(THRESHOLD_PATH, 1 << 40):
            assert _host_resident_tables(_Declared()) == []

    def test_a_sharded_table_is_left_alone(self):
        model = _Declared()
        model.embed.tp_size = 2
        with patch(THRESHOLD_PATH, 1024):
            assert _host_resident_tables(model) == []


class TestDetachAndRestore:
    def test_the_weight_survives_a_move_that_skips_it(self):
        model = _Declared()
        original = model.embed.weight.data.clone()
        with patch(THRESHOLD_PATH, 1024):
            detached = detach_host_resident_tables(model)
            assert model.embed.weight.numel() == 0
            model.to("cpu")
            restore_host_resident_tables(detached, "cpu")
        assert torch.equal(model.embed.weight.data, original)

    def test_the_gather_matches_a_plain_lookup(self):
        model = _Declared()
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        expected = torch.nn.functional.embedding(ids, model.embed.weight.data.clone())
        with patch(THRESHOLD_PATH, 1024):
            restore_host_resident_tables(detach_host_resident_tables(model), "cpu")
        assert torch.equal(model.embed(ids), expected)

    def test_nothing_is_hooked_when_nothing_qualifies(self):
        model = _Undeclared()
        with patch(THRESHOLD_PATH, 1024):
            detached = detach_host_resident_tables(model)
            restore_host_resident_tables(detached, "cpu")
        assert detached == []
        assert not model.embed._forward_pre_hooks

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_a_device_resident_table_is_parked_on_the_host(self):
        model = _Declared().to("cuda")
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda")
        with torch.no_grad():
            expected = model.embed(ids)

        with patch(THRESHOLD_PATH, 1024):
            detached = detach_host_resident_tables(model)
            assert model.embed.weight.numel() == 0
            model.to("cuda")
            restore_host_resident_tables(detached, "cuda")

        with torch.no_grad():
            actual = model.embed(ids)
        assert model.embed.weight.device.type == "cpu"
        assert actual.device.type == "cuda"
        assert torch.equal(actual, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    @pytest.mark.parametrize(
        "tensorwise", [False, True], ids=["rowwise-int8", "scalar-int8"]
    )
    def test_quantized_table_scales_follow_the_host_lookup(self, tensorwise):
        model = _Declared(num_embeddings=2, dim=4)
        model.embed = _Int8Embedding(tensorwise)
        ids = torch.tensor([[0, 1], [1, 0]])
        with torch.no_grad():
            expected = model.embed(ids)

        model.to("cuda")
        ids = ids.to("cuda")
        with patch(THRESHOLD_PATH, 1):
            detached = detach_host_resident_tables(model)
            assert model.embed.weight.numel() == 0
            model.to("cuda")
            restore_host_resident_tables(detached, "cuda")

        with torch.no_grad():
            actual = model.embed(ids)
        assert model.embed.weight.device.type == "cpu"
        assert model.embed.weight_scale.device.type == "cpu"
        assert model.proj.weight.device.type == "cuda"
        assert actual.device.type == "cuda"
        assert torch.equal(actual.cpu(), expected)
