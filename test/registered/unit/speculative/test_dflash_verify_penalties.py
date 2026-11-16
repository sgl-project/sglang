"""Sampling penalties must reach the DFlash/DSpark verify logits.

``apply_dflash_verify_logits_adjustments`` is the only place the speculative
verify path touches penalties, and it silently dropped them two ways:

A. Under overlap scheduling the worker holds a ``copy_for_forward()`` sampling
   info: ``penalizer_orchestrator`` is None and the state lives in
   ``acc_additive_penalties`` / ``acc_scaling_penalties``. The function only
   read ``acc_linear_penalties``, a field nothing in the tree writes, so both
   buffers were ignored and ``repetition_penalty`` / ``frequency_penalty`` /
   ``presence_penalty`` / ``min_tokens`` were dropped without a warning.
B. The non-overlap "dense fallback" applied the multiplicative (repetition)
   penalty to a zeros buffer. ``where(x < 0, x * p, x / p)`` on zeros is the
   identity, so the penalty was lost on that path too.

The fast paths in ``dflash_worker_v2`` / ``dspark_verify`` that read the same
state are covered at the bottom: they decide whether logits are gathered and
whether the folded in-graph accept may run, and neither applies penalties.

CPU only.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

VOCAB = 8


class _SamplingInfo(SimpleNamespace):
    """``apply_dflash_verify_logits_adjustments`` sizes the verify block with
    ``len(sampling_info)``, which reads ``temperatures`` on the real class."""

    def __len__(self):
        return len(self.temperatures)


def _si(bs=1, **overrides):
    fields = dict(
        temperatures=torch.ones(bs, 1),
        penalizer_orchestrator=None,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        grammar_mask=None,
        logit_bias=None,
        has_custom_logit_processor=False,
    )
    fields.update(overrides)
    return _SamplingInfo(**fields)


def _run(logits, sampling_info, draft_token_num):
    apply_dflash_verify_logits_adjustments(
        next_token_logits=logits,
        sampling_info=sampling_info,
        draft_token_num=draft_token_num,
    )
    return logits


def _scaled(value, penalty):
    """Reference for the non-speculative path: sign decides multiply or divide."""
    return value * penalty if value < 0 else value / penalty


# ---------------------------------------------------------------------------
# Overlap scheduling: the two acc_* buffers are the only penalty state
# ---------------------------------------------------------------------------


def test_scaling_penalties_scale_the_real_logits():
    """Regression for B: a token in the repetition table must be scaled, not
    passed through the identity a zeros buffer yields."""
    row = torch.tensor([1.0, -2.0, 3.0, -4.0, 0.5, -1.5, 2.5, 0.0])
    logits = row.repeat(2, 1)  # one request, two draft positions
    scaling = torch.ones(1, VOCAB)
    scaling[0, 1] = 2.0  # negative logit -> multiply
    scaling[0, 2] = 2.0  # positive logit -> divide

    _run(logits, _si(acc_scaling_penalties=scaling), draft_token_num=2)

    expected = torch.tensor([[_scaled(v, p) for v, p in zip(row, scaling[0].tolist())]])
    assert torch.equal(logits, expected.repeat(2, 1)), logits


def test_additive_penalties_broadcast_over_the_verify_block():
    """Frequency / presence state is per request, so its row is broadcast to
    every draft position of that request."""
    base = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0, 0.5, -1.5, 2.5, 0.0],
            [-1.0, 4.0, -3.0, 0.0, 1.5, -0.5, 2.0, -2.5],
        ]
    )
    draft_token_num = 3
    logits = base.repeat_interleave(draft_token_num, dim=0)
    additive = torch.zeros(2, VOCAB)
    additive[0, 0] = -0.25
    additive[1, 1] = 0.5

    _run(
        logits,
        _si(bs=2, acc_additive_penalties=additive),
        draft_token_num=draft_token_num,
    )

    expected = (base + additive).repeat_interleave(draft_token_num, dim=0)
    assert torch.equal(logits, expected), logits


@pytest.mark.parametrize("bs,draft_token_num", [(1, 1), (2, 3), (3, 8)])
def test_rows_are_major_over_the_draft_block(bs, draft_token_num):
    """Row ``b * draft + j`` belongs to request ``b``; a mis-reshaped layout
    would hand request 0's penalty to another request."""
    rows = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0, 0.5, -1.5, 2.5, -0.5],
            [-1.0, 4.0, -3.0, 0.0, 1.5, -0.5, 2.0, -2.5],
            [2.0, -1.0, 0.5, -0.5, 3.0, -4.0, 1.5, -3.0],
        ]
    )[:bs]
    logits = rows.repeat_interleave(draft_token_num, dim=0)
    # Powers of two keep the reference divisions exact in float32.
    scaling = torch.ones(bs, VOCAB)
    for b in range(bs):
        scaling[b] = 2.0**b

    _run(
        logits,
        _si(bs=bs, acc_scaling_penalties=scaling),
        draft_token_num=draft_token_num,
    )

    for b in range(bs):
        expected = torch.tensor(
            [_scaled(v, scaling[b, i].item()) for i, v in enumerate(rows[b])]
        )
        for j in range(draft_token_num):
            got = logits[b * draft_token_num + j]
            assert torch.equal(got, expected), (b, j, got, expected)


def test_additive_terms_land_before_scaling():
    """The non-speculative path adds the frequency/presence buffer first and
    scales afterwards, which matters when the add flips the sign. +2 with
    -5 then a 2.0 penalty is -6; the other order gives -4."""
    logits = torch.full((1, VOCAB), 2.0)
    additive = torch.zeros(1, VOCAB)
    additive[0, 0] = -5.0
    scaling = torch.ones(1, VOCAB)
    scaling[0, 0] = 2.0

    _run(
        logits,
        _si(acc_additive_penalties=additive, acc_scaling_penalties=scaling),
        draft_token_num=1,
    )

    assert logits[0, 0].item() == -6.0, logits
    assert logits[0, 1:].eq(2.0).all(), logits


def test_no_penalty_state_is_identity():
    logits = torch.randn(4, VOCAB)
    original = logits.clone()
    _run(logits, _si(), draft_token_num=4)
    assert torch.equal(logits, original)


# ---------------------------------------------------------------------------
# Non-overlap scheduling: the live orchestrator must feed the same path
# ---------------------------------------------------------------------------


class _Batch:
    """``BatchedPenalizerOrchestrator`` holds the batch through a weakref and
    reads ``device`` from it; SimpleNamespace cannot be weak-referenced."""

    def __init__(self, reqs):
        self.reqs = reqs
        self.device = torch.device("cpu")


def _orchestrator(bs, repetition_penalty=1.5, frequency_penalty=0.0):
    from sglang.srt.sampling.penaltylib.frequency_penalty import (
        BatchedFrequencyPenalizer,
    )
    from sglang.srt.sampling.penaltylib.orchestrator import (
        BatchedPenalizerOrchestrator,
    )
    from sglang.srt.sampling.penaltylib.presence_penalty import (
        BatchedPresencePenalizer,
    )
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        BatchedRepetitionPenalizer,
    )

    reqs = [
        SimpleNamespace(
            sampling_params=SimpleNamespace(
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=0.0,
            )
        )
        for _ in range(bs)
    ]
    return BatchedPenalizerOrchestrator(
        VOCAB,
        _Batch(reqs),
        {
            BatchedRepetitionPenalizer,
            BatchedFrequencyPenalizer,
            BatchedPresencePenalizer,
        },
    )


@pytest.mark.parametrize(
    "repetition_penalty,frequency_penalty", [(1.5, 0.0), (1.0, 0.5)]
)
def test_live_orchestrator_materialises_the_same_state(
    repetition_penalty, frequency_penalty
):
    """Without overlap the acc_* buffers were never built; the function has to
    accumulate both views from the orchestrator. Both modes then share one
    path, and the additive view keeps landing before the scaling one."""
    bs, draft_token_num = 2, 2
    torch.manual_seed(0)
    base = torch.randn(bs, VOCAB)
    logits = base.repeat_interleave(draft_token_num, dim=0)

    orchestrator = _orchestrator(
        bs,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
    )
    orchestrator.cumulate_output_tokens(torch.tensor([3, 5]))
    _run(
        logits,
        _si(bs=bs, penalizer_orchestrator=orchestrator),
        draft_token_num=draft_token_num,
    )

    for b, repeated_token in enumerate((3, 5)):
        penalty = torch.ones(VOCAB)
        penalty[repeated_token] = repetition_penalty
        value = base[b].clone()
        value[repeated_token] -= frequency_penalty
        expected = torch.where(value < 0, value * penalty, value / penalty)
        for j in range(draft_token_num):
            got = logits[b * draft_token_num + j]
            torch.testing.assert_close(got, expected)


def test_inactive_orchestrator_is_identity():
    """Default sampling params leave the orchestrator unprepared; nothing may
    be applied on its behalf."""
    logits = torch.randn(2, VOCAB)
    original = logits.clone()
    orchestrator = _orchestrator(1, repetition_penalty=1.0)
    assert not orchestrator.is_required

    _run(
        logits,
        _si(penalizer_orchestrator=orchestrator),
        draft_token_num=2,
    )

    assert torch.equal(logits, original)


# ---------------------------------------------------------------------------
# Grammar / logit_bias: both used to be applied in one apply_logits_bias call
# ---------------------------------------------------------------------------


class _GrammarMask:
    def __init__(self, blocked_column):
        self.blocked_column = blocked_column

    def apply(self, logits):
        logits[:, self.blocked_column] = float("-inf")


def test_logit_bias_is_applied_next_to_a_grammar_mask():
    """The old code applied all three terms through one ``apply_logits_bias``
    call on a zeros row; the rewrite applies them one by one and must keep
    logit_bias and the mask together."""
    draft_token_num = 2
    logits = torch.zeros(draft_token_num, VOCAB)
    logit_bias = torch.zeros(1, VOCAB)
    logit_bias[0, 1] = 0.5

    _run(
        logits,
        _si(grammar_mask=_GrammarMask(blocked_column=2), logit_bias=logit_bias),
        draft_token_num=draft_token_num,
    )

    expected = torch.zeros(VOCAB)
    expected[1] = 0.5
    expected[2] = float("-inf")
    assert torch.equal(logits, expected.repeat(draft_token_num, 1)), logits


def test_grammar_mask_still_constrains_the_block():
    logits = torch.zeros(2, VOCAB)
    _run(logits, _si(grammar_mask=_GrammarMask(blocked_column=7)), draft_token_num=2)
    assert torch.isinf(logits[:, 7]).all() and (logits[:, 7] < 0).all()
    assert torch.equal(logits[:, :7], torch.zeros(2, 7))


# ---------------------------------------------------------------------------
# Fast paths must notice the accumulated buffers
# ---------------------------------------------------------------------------


def _noop_sampling_info(**overrides):
    fields = dict(
        has_custom_logit_processor=False,
        penalizer_orchestrator=None,
        acc_linear_penalties=None,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        grammar_mask=None,
        logit_bias=None,
        sampling_mask_batch_indices=None,
        is_all_greedy=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_dflash_verify_does_not_skip_the_gather_with_accumulated_penalties():
    """The fast path drops the full-vocab gather, and with it the penalty
    application. Under overlap the only trace of a penalty is the acc_* buffer,
    so the check has to look at those too."""
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    worker = SimpleNamespace(_selector_sample=None, _sharded_greedy_logits=True)
    batch = SimpleNamespace(has_grammar=False, return_logprob=False)

    assert DFlashWorkerV2._can_skip_verify_logits(worker, batch, _noop_sampling_info())
    for field in ("acc_additive_penalties", "acc_scaling_penalties"):
        sampling_info = _noop_sampling_info(**{field: torch.ones(1, VOCAB)})
        assert (
            DFlashWorkerV2._can_skip_verify_logits(worker, batch, sampling_info)
            is False
        ), field


def test_dspark_noop_check_sees_accumulated_penalties():
    """The same buffers decide whether the folded, in-graph accept may run,
    which never applies penalties."""
    from sglang.srt.speculative.dspark_components.dspark_verify import (
        verify_logits_adjustments_are_noop,
    )

    assert verify_logits_adjustments_are_noop(_noop_sampling_info())
    for field in ("acc_additive_penalties", "acc_scaling_penalties"):
        sampling_info = _noop_sampling_info(**{field: torch.ones(1, VOCAB)})
        assert verify_logits_adjustments_are_noop(sampling_info) is False, field


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
