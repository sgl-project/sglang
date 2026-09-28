import dataclasses
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.sampling.bad_words_device import (
    BadWordsRequestSpec,
    BadWordsSamplingContext,
    DeviceBatch,
    apply_device_bad_words,
    take_bad_words_context,
    commit_accepted_bad_words,
)
from test_bad_words_device import params, suffix, check_mask

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def info(entries, mask=None):
    return SimpleNamespace(
        custom_params=entries, bad_words_commit_mask=mask, bad_words_context=None
    )


def test_exactly_once_and_unconsumed_guard():
    e = params([[1, 2, 3]], [1])
    i = info([e])
    apply_device_bad_words(torch.zeros(1, 32, device="cuda"), i)
    with pytest.raises(RuntimeError, match="not consumed"):
        apply_device_bad_words(torch.zeros(1, 32, device="cuda"), i)
    context = take_bad_words_context(i)
    assert take_bad_words_context(i) is None
    commit_accepted_bad_words(context, torch.tensor([2], device="cuda"))
    with pytest.raises(RuntimeError, match="already committed"):
        commit_accepted_bad_words(context, torch.tensor([9], device="cuda"))
    assert suffix(e["__req__"].bad_words_state) == [1, 2]


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_acceptance_rejection_bonus_and_stale_cpu_history(count):
    e = params([[1, 2, 3], [2, 4], [7, 8, 9]], [1])
    i = info([e])
    draft = torch.tensor([[1, 2, 3, 8]], device="cuda")
    apply_device_bad_words(torch.zeros(4, 32, device="cuda"), i, 4, draft)
    context = take_bad_words_context(i)
    # Tokens come from acceptance, including replacement/bonus, never raw draft.
    output = [2, 7, 8, 9]
    commit_accepted_bad_words(
        context,
        torch.tensor([output], device="cuda"),
        torch.tensor([count], device="cuda"),
    )
    assert e["__req__"].output_ids == [1]  # Deliberately lagging CPU result handling.
    view = DeviceBatch.build([e], "cuda")
    check_mask(view, [e], [[1] + output[:count]], None, 1)
    assert suffix(view.states[0]) == ([1] + output[:count])[-view.states[0].capacity :]


def test_retraction_generation_and_late_old_commit():
    e = params([[1, 2, 3]], [1])
    old = BadWordsSamplingContext(DeviceBatch.build([e], "cuda"), 1)
    e["__req__"].retraction_count += 1
    e["__req__"].output_ids = [7, 8]
    new = DeviceBatch.build([e], "cuda")
    old.commit_accepted(torch.tensor([2], device="cuda"))
    assert suffix(old.view.states[0]) == [1, 2]
    assert suffix(new.states[0]) == [7, 8]
    assert old.view.states[0] is not new.states[0]


def test_snapshot_reorder_mixed_requests_and_chunk_mask():
    a, b = params([[1, 2, 3]], [1]), params([[4, 5, 6]], [4])
    entries, mask = [a, None, b], [False, True, True]
    c = BadWordsSamplingContext(DeviceBatch.build(entries, "cuda", mask), 1)
    entries.reverse()
    mask[:] = [True, True, False]
    c.commit_accepted(torch.tensor([2, 9, 5], device="cuda"))
    assert suffix(a["__req__"].bad_words_state) == [1]
    assert suffix(b["__req__"].bad_words_state) == [4, 5]
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.view.batch_size = 42


def test_invalid_launch_shapes_rejected_before_kernel():
    e = params([[1, 2]], [1])
    view = DeviceBatch.build([e], "cuda")
    with pytest.raises(ValueError):
        view.apply(
            torch.zeros(4, 32, device="cuda"),
            4,
            torch.zeros(1, 3, device="cuda", dtype=torch.int64),
        )
    c = BadWordsSamplingContext(view, 1)
    with pytest.raises(ValueError):
        c.commit_accepted(torch.tensor([1, 2], device="cuda"))
    assert not c.committed
    c.commit_accepted(torch.tensor([3], device="cuda"))


@pytest.mark.parametrize("words", [[[]], [[-1]], [[True]], [[1.0]]])
def test_invalid_spec(words):
    with pytest.raises(ValueError):
        BadWordsRequestSpec.build(words)


def test_cancel_release_does_not_keep_state_in_global_registry():
    import gc, weakref

    e = params([[1, 2]], [1])
    view = DeviceBatch.build([e], "cuda")
    state_ref = weakref.ref(view.states[0])
    del e
    assert state_ref() is not None  # In-flight view owns old request state.
    del view
    gc.collect()
    assert state_ref() is None


def test_forward_snapshot_does_not_alias_pending_context():
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    source = SamplingBatchInfo(
        temperatures=torch.ones(1, 1), top_ps=torch.ones(1),
        top_ks=torch.ones(1, dtype=torch.int32), min_ps=torch.zeros(1),
        is_all_greedy=True, is_any_greedy=True, need_top_p_sampling=False,
        need_top_k_sampling=False, need_min_p_sampling=False, vocab_size=32,
        penalizer_orchestrator=SimpleNamespace(is_required=False),
    )
    marker = object()
    source.bad_words_context = marker
    snapshot = source.copy_for_forward()
    assert snapshot.bad_words_context is None
    assert source.bad_words_context is marker
