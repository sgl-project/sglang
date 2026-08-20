"""A large vocab table stays in host memory; the gather runs there."""

from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.managers.memory_managers import layerwise_offload
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    _host_resident_tables,
    detach_host_resident_tables,
    restore_host_resident_tables,
)

THRESHOLD_PATH = f"{layerwise_offload.__name__}.HOST_RESIDENT_TABLE_MIN_BYTES"


def _model(num_embeddings: int = 4096, dim: int = 64) -> torch.nn.Module:
    model = torch.nn.Module()
    model.embed = torch.nn.Embedding(num_embeddings, dim)
    model.proj = torch.nn.Linear(dim, dim)
    return model


class TestHostResidentTableSelection:
    def test_a_large_table_is_selected(self):
        model = _model()
        with patch(THRESHOLD_PATH, 1024):
            assert _host_resident_tables(model) == [model.embed]

    def test_a_small_table_is_left_alone(self):
        model = _model()
        with patch(THRESHOLD_PATH, 1 << 40):
            assert _host_resident_tables(model) == []

    def test_a_plain_weight_matrix_is_left_alone(self):
        model = torch.nn.Module()
        model.proj = torch.nn.Linear(512, 512)
        with patch(THRESHOLD_PATH, 1024):
            assert _host_resident_tables(model) == []

    def test_a_sharded_table_is_left_alone(self):
        model = _model()
        model.embed.tp_size = 2
        with patch(THRESHOLD_PATH, 1024):
            assert _host_resident_tables(model) == []


class TestDetachAndRestore:
    def test_the_weight_survives_a_move_that_skips_it(self):
        model = _model()
        original = model.embed.weight.data.clone()
        with patch(THRESHOLD_PATH, 1024):
            detached = detach_host_resident_tables(model)
            assert model.embed.weight.numel() == 0
            model.to("cpu")
            restore_host_resident_tables(detached, "cpu")
        assert torch.equal(model.embed.weight.data, original)

    def test_the_gather_matches_a_plain_lookup(self):
        model = _model()
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        expected = torch.nn.functional.embedding(ids, model.embed.weight.data.clone())
        with patch(THRESHOLD_PATH, 1024):
            restore_host_resident_tables(detach_host_resident_tables(model), "cpu")
        assert torch.equal(model.embed(ids), expected)

    def test_nothing_is_hooked_when_no_table_qualifies(self):
        model = _model()
        with patch(THRESHOLD_PATH, 1 << 40):
            detached = detach_host_resident_tables(model)
            restore_host_resident_tables(detached, "cpu")
        assert detached == []
        assert not model.embed._forward_pre_hooks
