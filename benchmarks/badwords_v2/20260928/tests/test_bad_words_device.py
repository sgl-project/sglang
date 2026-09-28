"""Reference comparisons for device masking and accepted-history commits."""

import random
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.sampling.bad_words_device import DeviceBatch, DeviceState

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def params(words, history=()):
    return {
        "bad_words_token_ids": words,
        "__req__": SimpleNamespace(output_ids=list(history), retraction_count=0),
    }


def suffix(state):
    values = state.state.cpu().tolist()
    valid, head = values[:2]
    return [values[2 + (head - valid + i) % state.capacity] for i in range(valid)]


def check_mask(batch, entries, histories, drafts, width):
    logits = torch.zeros(len(entries) * width, 32, device="cuda")
    batch.apply(logits, width, drafts)
    expected = torch.zeros_like(logits, device="cpu", dtype=torch.bool)
    candidates = drafts.cpu().tolist() if drafts is not None else None
    for row, param in enumerate(entries):
        if not param:
            continue
        for position in range(width):
            effective = histories[row] + (
                candidates[row][1 : position + 1] if candidates else []
            )
            for word in param["bad_words_token_ids"]:
                prefix = word[:-1]
                if not prefix or (
                    len(effective) >= len(prefix)
                    and effective[-len(prefix) :] == prefix
                ):
                    expected[row * width + position, word[-1]] = True
    assert torch.equal(torch.isneginf(logits).cpu(), expected)


def test_random_acceptance_and_ring_wrap():
    rng = random.Random(193)
    for case in range(100):
        bs, width = rng.randint(1, 6), rng.choice([1, 2, 4, 16])
        entries, histories = [], []
        for row in range(bs):
            history = [rng.randrange(8) for _ in range(rng.randrange(12))]
            words = [
                [rng.randrange(8) for _ in range(rng.randint(1, 12))]
                for _ in range(rng.randint(1, 20))
            ]
            words += words[:2]  # duplicate terminals / complete sequences
            entries.append(params(words, history) if row % 3 != 1 else None)
            histories.append(history)
        for step in range(5):
            order = list(range(bs))
            rng.shuffle(order)
            entries = [entries[i] for i in order]
            histories = [histories[i] for i in order]
            batch = DeviceBatch.build(entries, "cuda")
            draft = torch.tensor(
                [[rng.randrange(8) for _ in range(width)] for _ in range(bs)],
                device="cuda",
            )
            check_mask(batch, entries, histories, draft if width > 1 else None, width)
            counts = [rng.randint(1, width) for _ in range(bs)]
            output = [[rng.randrange(8) for _ in range(width)] for _ in range(bs)]
            batch.commit(
                torch.tensor(output, device="cuda"),
                torch.tensor(counts, device="cuda", dtype=torch.int32),
            )
            for row, param in enumerate(entries):
                histories[row].extend(output[row][: counts[row]])
                if param:
                    state = param["__req__"].bad_words_state
                    assert suffix(state) == histories[row][-state.capacity :]


def test_overlap_prefixes_and_prompt_exclusion():
    entry = params([[1, 2], [1, 2, 1], [2, 1], [1], [2, 1]])
    batch = DeviceBatch.build([entry], "cuda")
    check_mask(batch, [entry], [[]], torch.tensor([[9, 1, 2, 1]], device="cuda"), 4)
    batch.commit(
        torch.tensor([[1, 2, 3, 4]], device="cuda"), torch.tensor([2], device="cuda")
    )
    check_mask(batch, [entry], [[1, 2]], None, 1)


def test_chunk_skip_resume_and_stream_order():
    entry = params([[1, 2, 3, 4]], [1])
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(first):
        batch = DeviceBatch.build([entry], "cuda", [False])
        batch.apply(torch.zeros(1, 32, device="cuda"))
        batch.commit(torch.tensor([9], device="cuda"))
    with torch.cuda.stream(second):
        batch = DeviceBatch.build([entry], "cuda")
        batch.commit(torch.tensor([[2, 3]], device="cuda"))
        logits = torch.zeros(1, 32, device="cuda")
        batch.apply(logits)
    second.synchronize()
    assert torch.isneginf(logits[0, 4]).item()
    assert suffix(entry["__req__"].bad_words_state) == [1, 2, 3]
    old = entry["__req__"].bad_words_state
    entry["__req__"].retraction_count += 1
    entry["__req__"].output_ids = [7, 8]
    batch = DeviceBatch.build([entry], "cuda")
    assert batch.states[0] is not old
    assert suffix(batch.states[0]) == [7, 8]


def test_limits_and_long_prefix():
    with pytest.raises(ValueError):
        DeviceState([[1] * 258], [], "cuda")
    history = list(range(8)) * 32
    entry = params([history + [9]], history)
    batch = DeviceBatch.build([entry], "cuda")
    check_mask(batch, [entry], [history], None, 1)


def test_hot_path_has_no_device_to_host_reads(monkeypatch):
    import sglang.srt.sampling.bad_words_device as device_module
    from sglang.srt.layers.sampler import apply_custom_logit_processor
    from sglang.srt.sampling.custom_logit_processor import BadWordsLogitsProcessor

    class Info:
        custom_params = [params([[1, 2, 3], [1]])]
        bad_words_commit_mask = [True]
        custom_logit_processor = {
            0: (
                BadWordsLogitsProcessor(),
                torch.ones(1, device="cuda", dtype=torch.bool),
            )
        }

        def __len__(self):
            return 1

    info = Info()
    monkeypatch.setattr(device_module, "_BAD_WORDS_BACKEND", "cuda")
    logits = torch.zeros(4, 32, device="cuda")
    draft = torch.tensor([[9, 1, 2, 4]], device="cuda")
    output = torch.tensor([[1, 2, 4, 0]], device="cuda")
    lengths = torch.tensor([3], device="cuda")
    # Warm up Triton compilation before intercepting runtime host reads.
    apply_custom_logit_processor(logits, info, 4, draft)
    device_module.commit_accepted_bad_words(device_module.take_bad_words_context(info), output, lengths)
    for name in ("tolist", "item", "cpu", "numpy"):
        original = getattr(torch.Tensor, name)

        def checked(tensor, *args, _original=original, **kwargs):
            assert not tensor.is_cuda, "hot path attempted a GPU-to-CPU read"
            return _original(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, checked)
    apply_custom_logit_processor(logits, info, 4, draft)
    device_module.commit_accepted_bad_words(device_module.take_bad_words_context(info), output, lengths)


def test_pointer_lifetime_on_new_request_and_cross_stream():
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    entry = params([[1, 2]], [1])
    with torch.cuda.stream(first):
        batch = DeviceBatch.build([entry], "cuda")
    # Consume the mapping on another stream without a CPU synchronize.
    with torch.cuda.stream(second):
        logits = torch.zeros(1, 32, device="cuda")
        batch.apply(logits)
    replacement = params([[3, 4]], [3])
    entry["__req__"].bad_words_state = None
    del entry, batch
    with torch.cuda.stream(first):
        other = DeviceBatch.build([replacement], "cuda")
        other_logits = torch.zeros(1, 32, device="cuda")
        other.apply(other_logits)
    torch.cuda.synchronize()
    assert torch.isneginf(logits[0, 2]).item()
    assert not torch.isneginf(logits[0, 4]).item()
    assert torch.isneginf(other_logits[0, 4]).item()
