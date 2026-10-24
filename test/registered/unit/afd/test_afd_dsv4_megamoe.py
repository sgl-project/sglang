"""DSV4 local MegaMoE row ownership; the CUDA kernel is a CPU row oracle."""

from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.afd.model_adapters.deepseek_v4 import DeepseekV4AFDAdapter
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_padded_hash_rows_empty_rank_and_ready_weights(monkeypatch):
    from sglang.srt.layers import moe
    from sglang.srt.layers.moe import mega_moe

    calls = []
    ready = [True]
    monkeypatch.setattr(moe, "get_moe_a2a_backend", lambda: NS(is_megamoe=lambda: True))
    monkeypatch.setattr(mega_moe, "is_mega_moe_experts_ready", lambda _: ready[0])

    def compute(model, hidden, batch, input_ids_global=None):
        calls.append((hidden.clone(), input_ids_global.clone()))
        return hidden * 8 + 7  # Already includes routed and shared contributions.

    monkeypatch.setattr(mega_moe, "forward_mega_moe", compute)
    adapter = DeepseekV4AFDAdapter.__new__(DeepseekV4AFDAdapter)
    layer = NS(mlp=NS(is_hash=True, experts=object()))
    adapter.inner = NS(layers=[layer])
    adapter.validate_merge_reduce_scatter()
    for empty in (False, True):
        inputs = () if empty else (torch.ones(2, 3), torch.full((1, 3), 2.0))
        widths = () if empty else (4, 4)
        stage = NS(
            group_widths=widths,
            merge_sizes=(sum(widths), sum(widths)),
            merged_input_ids=None,
            input_ids=()
            if empty
            else (torch.tensor([[11], [12]]), torch.tensor([[21]])),
            collective_input=torch.empty(0, 3),
            collective_input_ids=torch.empty(0, 1, dtype=torch.int64),
            forward_batch=None,
            graph_output=None,
        )
        result = adapter._group_ffn_output(
            decoder_layer=layer, stage=stage, lane_hidden_states=inputs
        )
        hidden, ids = calls[-1]
        assert ids.tolist() == ([0] if empty else [11, 12, 0, 0, 21, 0, 0, 0])
        assert hidden.shape == (1 if empty else 8, 3)
        for actual, source in zip(result, inputs):
            torch.testing.assert_close(actual, source * 8 + 7)
        if empty:
            assert result == ()
            torch.testing.assert_close(stage.graph_output, torch.full((1, 3), 7.0))
        else:
            assert not hidden[2:4].count_nonzero()
    assert len(calls) == 2
    ready[0] = False
    with pytest.raises(RuntimeError, match="WEIGHTS_NOT_READY"):
        adapter.validate_merge_reduce_scatter()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
