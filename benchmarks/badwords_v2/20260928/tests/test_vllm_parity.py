import random
import torch
import pytest
from sglang.srt.sampling.bad_words_device import DeviceBatch
from test_bad_words_device import params, check_mask
from vllm_reference_kernel import apply_bad_words


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_mask_parity_with_pinned_vllm_v2_kernel():
    rng = random.Random(7429)
    for trial in range(100):
        bs, width = rng.randint(1, 8), rng.choice([1, 2, 4, 16])
        words, histories, prompts = [], [], []
        for r in range(bs):
            h = [rng.randrange(8) for _ in range(rng.randrange(20))]
            w = (
                [
                    [rng.randrange(8) for _ in range(rng.randint(1, 15))]
                    for _ in range(rng.randint(1, 20))
                ]
                if r % 3 != 1
                else []
            )
            # Guarantee some matches against histories as well as random misses.
            if h and w:
                w += [h[-min(3, len(h)) :] + [9]]
            words.append(w)
            histories.append(h)
            prompts.append([rng.randrange(8) for _ in range(rng.randint(1, 12))])
        if not any(words):
            continue
        order = list(range(bs))
        rng.shuffle(order)
        slots = torch.tensor(order, dtype=torch.int32, device="cuda").repeat_interleave(
            width
        )
        all_ids = torch.zeros(bs, 64, dtype=torch.int32, device="cuda")
        flat = torch.zeros(bs, 512, dtype=torch.int32, device="cuda")
        offsets = torch.zeros(bs, 32, dtype=torch.int32, device="cuda")
        counts = torch.tensor([len(w) for w in words], dtype=torch.int32, device="cuda")
        prompt_len = torch.tensor(
            [len(p) for p in prompts], dtype=torch.int32, device="cuda"
        )
        total_len = torch.tensor(
            [len(p) + len(h) for p, h in zip(prompts, histories)],
            dtype=torch.int32,
            device="cuda",
        )
        for r in range(bs):
            ids = prompts[r] + histories[r]
            all_ids[r, : len(ids)] = torch.tensor(ids, device="cuda")
            tokens = [t for word in words[r] for t in word]
            ends = [0]
            for word in words[r]:
                ends.append(ends[-1] + len(word))
            if tokens:
                flat[r, : len(tokens)] = torch.tensor(tokens, device="cuda")
            offsets[r, : len(ends)] = torch.tensor(ends, device="cuda")
        draft = torch.tensor(
            [[rng.randrange(8) for _ in range(width)] for _ in range(bs)],
            dtype=torch.int32,
            device="cuda",
        )
        ref = torch.zeros(bs * width, 32, device="cuda")
        apply_bad_words(
            ref,
            slots,
            flat,
            offsets,
            counts,
            all_ids,
            prompt_len,
            total_len,
            draft.flatten(),
            torch.arange(width, device="cuda").repeat(bs),
            max(map(len, words)),
        )
        entries = [params(words[r], histories[r]) if words[r] else None for r in order]
        view = DeviceBatch.build(entries, "cuda")
        actual = torch.zeros_like(ref)
        view.apply(actual, width, draft if width > 1 else None)
        assert torch.equal(torch.isneginf(actual), torch.isneginf(ref)), (
            trial,
            width,
            order,
        )
        check_mask(
            view,
            entries,
            [histories[r] for r in order],
            draft if width > 1 else None,
            width,
        )
