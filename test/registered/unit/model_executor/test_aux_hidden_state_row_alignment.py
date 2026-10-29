"""Aux and last-layer capture must retain the same original token rows."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.encoder_swa_replay import FoldedExtend, drop_folded_rows
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "mode,real_rows",
    [(ForwardMode.DECODE, 2), (ForwardMode.IDLE, 0), (ForwardMode.EXTEND, 3)],
    ids=["decode", "idle", "extend"],
)
@pytest.mark.parametrize(
    "with_last_hidden", [False, True], ids=["stock", "aux-capture"]
)
def test_mlp_sync_unpads_both_hidden_outputs(mode, real_rows, with_last_hidden):
    padded_rows = 5
    batch = ForwardBatch(
        forward_mode=mode,
        batch_size=padded_rows,
        input_ids=torch.arange(padded_rows),
        req_pool_indices=torch.arange(padded_rows),
        seq_lens=torch.ones(padded_rows, dtype=torch.int64),
        out_cache_loc=torch.arange(padded_rows),
        positions=torch.arange(padded_rows),
        seq_lens_sum=real_rows,
        _original_batch_size=real_rows,
        _original_num_tokens=real_rows,
    )
    hidden = torch.arange(padded_rows * 6).reshape(padded_rows, 6)
    last_hidden = torch.arange(padded_rows * 2).reshape(padded_rows, 2)
    output = LogitsProcessorOutput(
        next_token_logits=torch.zeros(padded_rows, 7),
        hidden_states=hidden,
        last_hidden_states=last_hidden if with_last_hidden else None,
    )

    batch.post_forward_mlp_sync_batch(output)

    torch.testing.assert_close(output.hidden_states, hidden[:real_rows])
    if with_last_hidden:
        torch.testing.assert_close(output.last_hidden_states, last_hidden[:real_rows])
    else:
        assert output.last_hidden_states is None


@pytest.mark.parametrize("with_tail", [False, True], ids=["full", "decoder-tail"])
@pytest.mark.parametrize(
    "with_last_hidden", [False, True], ids=["stock", "aux-capture"]
)
def test_folded_replay_removes_the_same_rows_from_both_outputs(
    with_tail, with_last_hidden
):
    # Two requests: one replay row before two new rows, then two replay rows
    # before one new row. The decoder tail drops the first replay row already.
    keep_rows = torch.tensor([1, 2, 5])
    folded = FoldedExtend(
        batch=SimpleNamespace(extend_lens=[3, 3]),
        keep_rows=keep_rows,
        scheduled_row=torch.tensor([-1, 0, 1, -1, -1, 2]),
        row_floor=torch.zeros(6, dtype=torch.int64),
        replay_lens=[1, 2],
        num_rows=6,
    )
    hidden = torch.arange(8 * 6).reshape(8, 6)
    last_hidden = torch.arange(8 * 2).reshape(8, 2)
    token_indices = torch.tensor([1, 2, 3, 4, 5]) if with_tail else None
    output = LogitsProcessorOutput(
        next_token_logits=None,
        hidden_states=hidden[token_indices] if with_tail else hidden,
        last_hidden_states=(
            (last_hidden[token_indices] if with_tail else last_hidden)
            if with_last_hidden
            else None
        ),
        hidden_states_token_indices=token_indices,
    )

    drop_folded_rows(
        logits_output=output,
        folded=folded,
        tail_lens=[2, 3] if with_tail else None,
    )

    torch.testing.assert_close(output.hidden_states, hidden[keep_rows])
    if with_last_hidden:
        torch.testing.assert_close(output.last_hidden_states, last_hidden[keep_rows])
    else:
        assert output.last_hidden_states is None
    if with_tail:
        torch.testing.assert_close(output.hidden_states_token_indices, torch.arange(3))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
