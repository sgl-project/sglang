from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

dp_attention = pytest.importorskip("sglang.srt.layers.dp_attention")


def test_partial_dp_gather_does_not_alias_collective_input(monkeypatch):
    tp_size = 8
    attn_tp_size = 4
    attn_tp_rank = 1
    local_tokens = torch.arange(32, dtype=torch.float32).view(8, 4)
    original = local_tokens.clone()
    global_tokens = torch.empty(16, 4)

    tp_group = Mock()

    def all_gather(full_output, local_input):
        for chunk in full_output.tensor_split(tp_size):
            chunk.copy_(local_input)

    tp_group.all_gather_into_tensor.side_effect = all_gather
    attn_tp_group = Mock()

    def reduce_scatter(local_output, full_input):
        assert local_output.untyped_storage() is not full_input.untyped_storage()
        torch.testing.assert_close(full_input, original)
        local_output.copy_(original.tensor_split(attn_tp_size)[attn_tp_rank])

    attn_tp_group.reduce_scatter_tensor.side_effect = reduce_scatter
    parallel = SimpleNamespace(
        attn_tp_group=attn_tp_group,
        attn_tp_rank=attn_tp_rank,
        tp_group=tp_group,
    )

    monkeypatch.setattr(dp_attention, "world_dp_gather_enabled", lambda: False)
    monkeypatch.setattr(
        dp_attention,
        "get_attn_tensor_model_parallel_world_size",
        lambda: attn_tp_size,
    )
    monkeypatch.setattr(dp_attention, "get_parallel", lambda: parallel)

    dp_attention._dp_gather_via_all_gather(
        global_tokens, local_tokens, None, is_partial=True
    )

    expected_chunk = original.tensor_split(attn_tp_size)[attn_tp_rank]
    torch.testing.assert_close(global_tokens, expected_chunk.repeat(tp_size, 1))
    torch.testing.assert_close(local_tokens, original)


def test_partial_dp_reduce_scatter_does_not_alias_collective_input(monkeypatch):
    tp_size = 8
    attn_tp_size = 4
    tp_rank = 5
    rows_per_dp_rank = 12
    hidden_size = 3
    collective_input = torch.arange(
        2 * rows_per_dp_rank * hidden_size, dtype=torch.float32
    ).view(2 * rows_per_dp_rank, hidden_size)
    original = collective_input.clone()
    output = torch.empty(rows_per_dp_rank, hidden_size)

    tp_group = Mock()

    def reduce_scatter(local_output, full_input):
        assert local_output.untyped_storage() is not full_input.untyped_storage()
        torch.testing.assert_close(full_input, original)
        local_output.copy_(
            original.tensor_split(tp_size)[tp_rank]
        )

    tp_group.reduce_scatter_tensor.side_effect = reduce_scatter
    attn_tp_group = Mock()

    def all_gather(full_output, local_input):
        assert full_output.untyped_storage() is not local_input.untyped_storage()
        for chunk in full_output.tensor_split(attn_tp_size):
            chunk.copy_(local_input)

    attn_tp_group.all_gather_into_tensor.side_effect = all_gather
    parallel = SimpleNamespace(
        attn_dp_size=2,
        attn_tp_group=attn_tp_group,
        tp_group=tp_group,
        tp_rank=tp_rank,
    )

    monkeypatch.setattr(dp_attention, "_note_dp_gather_in_prefill_graph", lambda: None)
    monkeypatch.setattr(dp_attention, "is_dp_gatherv_active", lambda: False)
    monkeypatch.setattr(
        dp_attention, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(dp_attention, "get_parallel", lambda: parallel)

    dp_attention.dp_reduce_scatter_tensor(output, collective_input)

    expected_chunk = original.tensor_split(tp_size)[tp_rank]
    torch.testing.assert_close(
        output,
        torch.cat([expected_chunk] * attn_tp_size),
    )
    torch.testing.assert_close(collective_input, original)
