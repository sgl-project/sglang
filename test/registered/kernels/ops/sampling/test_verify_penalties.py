import random
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.sampling import penalties
from sglang.srt.sampling.penalties import PenaltyStatusPool
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.sampling.sampling_pipeline import _adjust_before_watermark
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.sampling_test_utils import build_step_penalties

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _assert_bitwise(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # Bit patterns, so -0.0 and +0.0 count as different.
    bits = {
        torch.float32: torch.int32,
        torch.bfloat16: torch.int16,
        torch.float16: torch.int16,
    }
    actual, expected = actual.cpu(), expected.cpu()
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.view(bits[actual.dtype]), expected.view(bits[expected.dtype])
    )


# Several kernel chunks, with draft tokens in each and a partial last one.
VOCAB_SIZE = 5000
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _req(output_ids, **sampling_params):
    return SimpleNamespace(
        penalty_index=None,
        output_ids=list(output_ids),
        sampling_params=SamplingParams(**sampling_params),
        eos_token_ids=None,
        tokenizer=SimpleNamespace(eos_token_id=None, additional_stop_token_ids=None),
    )


def _requests():
    """Every penalty kind, a request with none, out-of-vocab stop ids, stop sets of different sizes."""
    return [
        _req([7, 7, 2100], frequency_penalty=0.7),
        _req([], temperature=0.5),
        # Many scaled tokens, so a division that rounds differently shows.
        _req(
            [4999, 11] + list(range(100, VOCAB_SIZE, 50)),
            presence_penalty=0.4,
            repetition_penalty=1.3,
        ),
        _req(
            [3, 3],
            min_new_tokens=4,
            stop_token_ids=[11, 4100, VOCAB_SIZE + 5],
            repetition_penalty=0.8,
        ),
        _req(
            [2100, 5, 5, 5],
            frequency_penalty=0.3,
            presence_penalty=0.2,
            repetition_penalty=1.1,
            min_new_tokens=6,
            stop_token_ids=[5, 2100, 4999],
        ),
    ]


def _random_parents(width, rng):
    """parent[a] < a, so node 0 is the root; one node may sit off the tree."""
    parents = [-1] + [rng.randrange(a) for a in range(1, width)]
    if width > 2 and rng.random() < 0.5:
        parents[rng.randrange(1, width)] = -1
    return parents


def _tree_links(parents):
    width = len(parents)
    next_child, next_sibling = [-1] * width, [-1] * width
    last_child = {}
    for a in range(1, width):
        p = parents[a]
        if p < 0:
            continue
        if p in last_child:
            next_sibling[last_child[p]] = a
        else:
            next_child[p] = a
        last_child[p] = a
    return next_child, next_sibling


def _random_tokens(width, rng):
    # Few distinct ids, so paths repeat tokens; some ids fall outside the vocab.
    choices = [7, 11, 5, 2100, 4100, 4999, -1, VOCAB_SIZE + 3]
    return [rng.randrange(VOCAB_SIZE)] + [rng.choice(choices) for _ in range(width - 1)]


def _decode_one_token_at_a_time(logits_row, req, path, device):
    """`logits_row` as non-spec decoding penalizes it after `req`'s output and `path`,
    one committed token per step."""
    decoded = SimpleNamespace(**{**vars(req), "penalty_index": None, "output_ids": []})
    pool = PenaltyStatusPool(VOCAB_SIZE, device)
    pool.alloc([decoded])
    pool.reset_history([decoded])
    for token in req.output_ids + path:
        step = build_step_penalties([decoded], pool)
        if step is not None:
            step.commit(torch.tensor([token], device=device))
    info = SamplingBatchInfo(
        temperatures=torch.ones(1, 1, device=device),
        top_ps=torch.ones(1, device=device),
        top_ks=torch.ones(1, dtype=torch.int32, device=device),
        min_ps=torch.zeros(1, device=device),
        is_all_greedy=True,
        is_any_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=VOCAB_SIZE,
        device=device,
    )
    info.step_penalties = build_step_penalties([decoded], pool)
    row = logits_row.to(device).clone()[None]
    _adjust_before_watermark(row, info, observer=None)
    return row[0]


def _expected(logits, reqs, tokens, parents, device):
    """Each verify row as `_decode_one_token_at_a_time` gives it, over its path
    from the root (exclusive) to its node."""
    expected = logits.to(device).clone()
    width = len(tokens[0])
    for b, req in enumerate(reqs):
        for j in range(width):
            path = []
            a = j
            while parents[b][a] >= 0:
                path.insert(0, tokens[b][a])
                a = parents[b][a]
            row = b * width + j
            expected[row] = _decode_one_token_at_a_time(logits[row], req, path, device)
    return expected.cpu()


def _setup(device, dtype, width, tree, seed, reqs=None):
    rng = random.Random(seed)
    reqs = reqs or _requests()
    pool = PenaltyStatusPool(VOCAB_SIZE, device)
    pool.alloc(reqs)
    pool.reset_history(reqs)
    step = build_step_penalties(reqs, pool)
    bs = len(reqs)
    parents = [
        _random_parents(width, rng) if tree else list(range(-1, width - 1))
        for _ in range(bs)
    ]
    tokens = [_random_tokens(width, rng) for _ in range(bs)]
    links = [_tree_links(p) for p in parents]
    gen = torch.Generator().manual_seed(seed)
    logits = (torch.randn(bs * width, VOCAB_SIZE, generator=gen) * 4).to(dtype)
    kwargs = {}
    if tree:
        kwargs = dict(
            next_child=torch.tensor([l[0] for l in links], device=device),
            next_sibling=torch.tensor([l[1] for l in links], device=device),
        )
    return SimpleNamespace(
        reqs=reqs,
        step=step,
        parents=parents,
        tokens=tokens,
        draft_tokens=torch.tensor(tokens, device=device),
        logits=logits,
        kwargs=kwargs,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tree", [True, False])
@pytest.mark.parametrize("width", [1, 4, 8])
def test_kernel_matches_per_row_reference(dtype, tree, width) -> None:
    case = _setup("cuda", dtype, width, tree, seed=width)
    expected = _expected(case.logits, case.reqs, case.tokens, case.parents, "cuda")
    logits = case.logits.cuda()
    case.step.apply_verify(logits, case.draft_tokens, **case.kwargs)
    _assert_bitwise(logits, expected)


# A row stride of VOCAB_SIZE + 3 leaves rows unaligned for vector loads.
@pytest.mark.parametrize("padding", [64, 3])
def test_kernel_reads_int32_tokens_and_strided_rows(padding) -> None:
    case = _setup("cuda", torch.bfloat16, 6, True, seed=3)
    expected = _expected(case.logits, case.reqs, case.tokens, case.parents, "cuda")
    padded = torch.zeros(
        case.logits.shape[0], VOCAB_SIZE + padding, dtype=torch.bfloat16, device="cuda"
    )
    logits = padded[:, :VOCAB_SIZE]
    logits.copy_(case.logits)
    case.step.apply_verify(logits, case.draft_tokens.int(), **case.kwargs)
    _assert_bitwise(logits, expected)
    assert not padded[:, VOCAB_SIZE:].any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kernel_matches_torch_path_on_a_large_batch(dtype, monkeypatch) -> None:
    """Many rows per CTA, against the torch path."""
    rng = random.Random(6)
    reqs = [
        SimpleNamespace(
            **{
                **vars(req),
                "output_ids": [rng.choice([7, 11, 2100, 4999]) for _ in range(i % 9)],
            }
        )
        for i, req in enumerate(_requests() * 128)
    ]
    case = _setup("cuda", dtype, 12, True, seed=7, reqs=reqs)
    expected = case.logits.cuda()
    with monkeypatch.context() as patch:
        patch.setattr(penalties, "_has_verify_kernel", False)
        case.step.apply_verify(expected, case.draft_tokens, **case.kwargs)
    logits = case.logits.cuda()
    case.step.apply_verify(logits, case.draft_tokens, **case.kwargs)
    _assert_bitwise(logits, expected)


@pytest.mark.parametrize("token_dtype", [torch.int32, torch.int64])
def test_commit_kernel_matches_torch_fallback(token_dtype) -> None:
    """Repeated, out-of-vocab, and masked tokens; strided token columns."""
    rng = random.Random(5)
    pools = {}
    for device in ("cpu", "cuda"):
        reqs = _requests()
        pool = PenaltyStatusPool(VOCAB_SIZE, device)
        pool.alloc(reqs)
        pool.reset_history(reqs)
        pools[device] = (pool, reqs)
    bs = len(_requests())
    for k in (1, 3, 8):
        choices = [7, 7, 11, 4999, -1, VOCAB_SIZE, VOCAB_SIZE + 9]
        tokens = torch.tensor(
            [[rng.choice(choices) for _ in range(2 * k)] for _ in range(bs)],
            dtype=token_dtype,
        )
        accept_mask = (
            torch.rand(bs, k, generator=torch.Generator().manual_seed(k)) < 0.7
        )
        for device, (pool, reqs) in pools.items():
            step = build_step_penalties(reqs, pool)
            step.commit(
                tokens.to(device)[:, ::2], accept_mask.to(device) if k > 1 else None
            )
    cpu_tables = pools["cpu"][0].live_tables()
    cuda_tables = pools["cuda"][0].live_tables()
    assert torch.equal(cuda_tables.counts.cpu(), cpu_tables.counts)
    assert torch.equal(cuda_tables.lens.cpu(), cpu_tables.lens)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tree", [True, False])
def test_torch_fallback_matches_per_row_reference(dtype, tree) -> None:
    case = _setup("cpu", dtype, 8, tree, seed=4)
    expected = _expected(case.logits, case.reqs, case.tokens, case.parents, "cpu")
    logits = case.logits.clone()
    case.step.apply_verify(logits, case.draft_tokens, **case.kwargs)
    _assert_bitwise(logits, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
