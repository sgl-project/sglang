"""DSPARK draft tokens must stay within the target's true vocabulary.

The draft head may be padded wider than the target's true vocab; padded rows
are input-only and must never be sampled. A sampled pad-range id propagates
through verify into token gathers and kills the engine with device-side
scatter/gather asserts. Verification must also normalize the proposal
distribution q over the same capped vocabulary that was sampled.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.dspark_draft import (
    sample_draft_block,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TP_SYNC = SimpleNamespace(sync=lambda _site, value: value)
# Padded draft head: 5 rows, true target vocab 3. The pad rows carry the
# largest logits, so an uncapped path deterministically samples them.
_PADDED_LOGITS = torch.tensor([[1.0, 2.0, 3.0, 100.0, 200.0]])
_TARGET_VOCAB = 3


def _sample_block(
    _base_logits: torch.Tensor,
    *,
    sampler: Callable[[torch.Tensor, int], torch.Tensor],
    **_kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = sampler(_PADDED_LOGITS, 0)
    return tokens.reshape(1, 1), _PADDED_LOGITS[:, None, :]


@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_eager_draft_block_caps_proposals_and_q(greedy: bool) -> None:
    sampling_info = (
        None
        if greedy
        else SimpleNamespace(
            is_all_greedy=False,
            top_ks=torch.tensor([2]),
            temperatures=torch.tensor([[1.0]]),
        )
    )
    with envs.SGLANG_DSPARK_FAST_SAMPLING.override(False):
        result = sample_draft_block(
            base_logits=torch.zeros(1, 5),
            anchor_tokens=torch.zeros(1, dtype=torch.int64),
            draft_hidden=torch.zeros(1, 1),
            sampling_info=sampling_info,
            markov_head=SimpleNamespace(sample_block=_sample_block),
            device=torch.device("cpu"),
            tp_sync=_TP_SYNC,
            sample_vocab_size=_TARGET_VOCAB,
        )

    assert int(result.draft_tokens.max().item()) < _TARGET_VOCAB
    # q (corrected_logits) must be cropped to the same capped vocabulary.
    assert result.corrected_logits.shape[-1] == _TARGET_VOCAB


def test_chain_verify_buffers_start_zeroed() -> None:
    from sglang.srt.speculative.dflash_utils import (
        _DFLASH_CHAIN_VERIFY_BUFFERS,
        _get_or_create_chain_verify_buffers,
    )

    _DFLASH_CHAIN_VERIFY_BUFFERS.clear()
    try:
        _ri, _rnt, _rns, predicts, accept_index, accept_token_num = (
            _get_or_create_chain_verify_buffers(
                bs=2, draft_token_num=4, device=torch.device("cpu")
            )
        )
        assert torch.count_nonzero(predicts).item() == 0
        assert torch.count_nonzero(accept_index).item() == 0
        assert torch.count_nonzero(accept_token_num).item() == 0
    finally:
        _DFLASH_CHAIN_VERIFY_BUFFERS.clear()
