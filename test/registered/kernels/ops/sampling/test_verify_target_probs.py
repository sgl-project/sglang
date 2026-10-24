"""Speculative verify target probabilities: selecting the top-k on the raw logits
and dividing only the selected values by the temperature must give the same
distribution as dividing the whole vocabulary first and running ``torch.topk``.

When several logits tie at a row's top-k boundary, which of them keep the mass
is chosen deterministically but may differ from ``torch.topk``; the kept count
and the probability values are the same."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import sglang.kernels.ops.sampling.topk as topk_ops
from sglang.kernels.ops.sampling.topk import radix_topk
from sglang.srt.speculative.dflash_utils import (
    _dflash_top_p_renorm_prob,
    build_speculative_verify_target_probs,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_DRAFT_TOKEN_NUM = 8


def _skip_unsupported_top_p(device, need_top_p):
    # The top-p renormalization kernel is resolved for the accelerator only.
    if need_top_p and device == "cpu":
        pytest.skip("top-p renormalization needs the accelerator kernel")


def _case(device, logits, top_ks, *, need_top_p, seed, temperature=None):
    """Sampling params for ``len(top_ks)`` requests and the call kwargs."""
    bs = len(top_ks)
    g = torch.Generator(device=device).manual_seed(seed)
    if temperature is None:
        temperatures = 0.2 + 1.6 * torch.rand((bs,), generator=g, device=device)
    else:
        temperatures = torch.full((bs,), temperature, device=device)
    info = SimpleNamespace(
        temperatures=temperatures.view(-1, 1),
        top_ks=torch.tensor(top_ks, dtype=torch.int32, device=device),
        top_ps=0.5 + 0.5 * torch.rand((bs,), generator=g, device=device),
        need_top_k_sampling=True,
        need_top_p_sampling=need_top_p,
    )
    uniform = top_ks[0] if len(set(top_ks)) == 1 else None
    return dict(
        next_token_logits=logits,
        sampling_info=info,
        draft_token_num=logits.shape[0] // bs,
        bs=bs,
        max_top_k=max(top_ks),
        uniform_top_k_value=uniform,
    )


def _reference(next_token_logits, sampling_info, draft_token_num, bs, max_top_k, **_):
    """The previous implementation: scale the whole vocabulary, then top-k."""
    info = sampling_info
    temperature = torch.repeat_interleave(info.temperatures, draft_token_num, dim=0)
    scaled = next_token_logits / temperature
    top_ks = torch.repeat_interleave(info.top_ks, draft_token_num, dim=0).long()
    top_vals, top_ids = torch.topk(scaled, k=max_top_k, dim=-1)
    ranks = torch.arange(max_top_k, device=scaled.device)[None, :]
    top_vals = top_vals.masked_fill(ranks >= top_ks[:, None], float("-inf"))
    top_probs = F.softmax(top_vals, dim=-1)
    if info.need_top_p_sampling:
        top_probs = _dflash_top_p_renorm_prob(
            top_probs, torch.repeat_interleave(info.top_ps, draft_token_num, dim=0)
        )
    probs = torch.zeros_like(scaled).scatter_(1, top_ids, top_probs)
    return probs.view(bs, draft_token_num, -1)


def _random_logits(device, rows, vocab, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    return 4.0 * torch.randn((rows, vocab), generator=g, device=device)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("need_top_p", [False, True])
@pytest.mark.parametrize(
    "vocab,top_ks",
    [
        (1000, [20, 20, 20, 20]),  # uniform top-k, no per-row mask
        (1000, [1, 20, 7, 64]),  # mixed, including a greedy row
        (151936, [1, 20, 7, 64]),
        (151936, [2048] * 4),
        (151936, [2049] * 4),
        (151936, [4096] * 4),
        (151936, [1, 4096, 2049, 20]),
    ],
)
def test_matches_full_vocab_temperature(device, need_top_p, vocab, top_ks):
    _skip_unsupported_top_p(device, need_top_p)
    rows = len(top_ks) * _DRAFT_TOKEN_NUM
    logits = _random_logits(device, rows, vocab, seed=vocab + max(top_ks))
    kwargs = _case(device, logits, top_ks, need_top_p=need_top_p, seed=max(top_ks))
    got = build_speculative_verify_target_probs(**kwargs)
    want = _reference(**kwargs)
    # Continuous random logits have no ties, so the selected sets agree and the
    # probabilities are computed from identical values.
    assert got.shape == want.shape and got.dtype == want.dtype
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    for i in (i for i, k in enumerate(top_ks) if k == 1):
        assert torch.equal(
            got[i].argmax(-1), logits.view(len(top_ks), -1, vocab)[i].argmax(-1)
        )
        assert torch.all(got[i].max(-1).values == 1)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("need_top_p", [False, True])
@pytest.mark.parametrize("top_ks", [[20] * 4, [1, 20, 2049, 5]])
def test_vocab_masked_rows(device, need_top_p, top_ks):
    """Grammar masks leave rows with fewer finite logits than k."""
    _skip_unsupported_top_p(device, need_top_p)
    rows, vocab = len(top_ks) * _DRAFT_TOKEN_NUM, 32000
    logits = _random_logits(device, rows, vocab, seed=3)
    g = torch.Generator(device=device).manual_seed(4)
    keep = torch.rand((rows, vocab), generator=g, device=device).argsort(-1)
    finite = torch.zeros_like(logits, dtype=torch.bool)
    for r in range(rows):
        finite[r, keep[r, : r % 15 + 1]] = True
    logits = logits.masked_fill(~finite, float("-inf"))
    kwargs = _case(device, logits, top_ks, need_top_p=need_top_p, seed=5)
    got = build_speculative_verify_target_probs(**kwargs)
    torch.testing.assert_close(got, _reference(**kwargs), rtol=0, atol=0)
    assert torch.all(got.view(rows, vocab)[~finite] == 0)


def _check_tied(got, want, logits, top_ks):
    """``got`` is an exact top-k distribution: the right count of kept ids per
    row, none of them below a dropped logit, and the reference probabilities."""
    rows, vocab = logits.shape
    got, want = got.view(rows, vocab), want.view(rows, vocab)
    row_k = torch.tensor(top_ks, device=logits.device).repeat_interleave(
        rows // len(top_ks)
    )
    kept = got > 0
    assert torch.equal(kept.sum(-1), row_k)
    min_kept = logits.masked_fill(~kept, float("inf")).min(-1).values
    max_dropped = logits.masked_fill(kept, float("-inf")).max(-1).values
    assert torch.all(min_kept >= max_dropped)
    torch.testing.assert_close(
        got.sort(-1, descending=True).values,
        want.sort(-1, descending=True).values,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("top_ks", [[20, 20], [3, 20], [1, 4096]])
def test_ties_at_the_k_boundary(device, top_ks):
    """Integer logits tie at every row's own boundary, also in a mixed batch
    where a row's k is below the selected max_top_k."""
    rows, vocab = len(top_ks) * _DRAFT_TOKEN_NUM, 50000
    g = torch.Generator(device=device).manual_seed(6)
    logits = torch.randint(0, 6, (rows, vocab), generator=g, device=device).float()
    kwargs = _case(device, logits, top_ks, need_top_p=False, seed=7)
    got = build_speculative_verify_target_probs(**kwargs)
    assert torch.equal(got, build_speculative_verify_target_probs(**kwargs))
    _check_tied(got, _reference(**kwargs), logits, top_ks)


@pytest.mark.parametrize("device", _DEVICES)
def test_ties_created_by_the_temperature(device):
    """Two logits one ulp apart can round to one value after the division.
    Selecting on the raw logits keeps the larger one, a valid top-k of the
    scaled logits, with the reference probabilities."""
    vocab, k, temperature = 4096, 8, 3.7
    # A tensor, not a Python scalar: CUDA multiplies by the reciprocal of a
    # scalar divisor, which rounds differently from the verify path.
    t = torch.full((1,), temperature, device=device)
    x = torch.tensor(1.9999, device=device)
    for _ in range(4096):
        y = torch.nextafter(x, x + 1)
        if x / t == y / t:
            break
        x = y
    assert x / t == y / t
    logits = _random_logits(device, 2, vocab, seed=8).clamp(max=1.0)
    logits[:, : k - 1] = torch.arange(10.0, 10.0 + k - 1, device=device)
    logits[:, 100], logits[:, 101] = y, x  # compete for the k-th place
    kwargs = _case(
        device, logits, [k], need_top_p=False, seed=9, temperature=temperature
    )
    got = build_speculative_verify_target_probs(**kwargs)
    want = _reference(**kwargs)
    _check_tied(got, want, logits / t, [k])
    assert torch.all((got.view(2, vocab)[:, 100:102] > 0).sum(-1) == 1)


@pytest.mark.parametrize("device", _DEVICES)
def test_torch_fallback(device, monkeypatch):
    monkeypatch.setattr(topk_ops, "_flashinfer_top_k", None)
    need_top_p = device != "cpu"
    top_ks = [5, 2049, 1]
    logits = _random_logits(device, 3 * _DRAFT_TOKEN_NUM, 32000, seed=11)
    kwargs = _case(device, logits, top_ks, need_top_p=need_top_p, seed=12)
    got = build_speculative_verify_target_probs(**kwargs)
    torch.testing.assert_close(got, _reference(**kwargs), rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k", [1, 20, 2048, 2049, 4096])
@pytest.mark.parametrize("strided", [False, True])
def test_radix_topk_matches_torch(device, dtype, k, strided):
    g = torch.Generator(device=device).manual_seed(k)
    scores = torch.randn((37, 2 * 151936), generator=g, device=device).to(dtype)
    # A column slice is not contiguous; the op must still read it correctly.
    scores = scores[:, ::2] if strided else scores[:, :151936].contiguous()
    values, ids = radix_topk(scores, k)
    ref_values, _ = torch.topk(scores, k, dim=-1)
    assert ids.dtype == torch.int64
    torch.testing.assert_close(values, ref_values, rtol=0, atol=0)
    torch.testing.assert_close(scores.gather(-1, ids), ref_values, rtol=0, atol=0)
    assert torch.all(ids.sort(-1).values.diff(dim=-1) > 0)
