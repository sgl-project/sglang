"""A declared vocab table stays in host memory; the gather runs there."""

from unittest.mock import patch

import torch

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
