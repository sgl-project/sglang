"""Fused draft softmax + sample + greedy argmax (`softmax_jit.softmax_sample`)."""

import math
import sys

import pytest
import torch

from sglang.kernels.arch import current_arch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_arch().is_sm(10),
    reason="the JIT softmax runs on the SM100 family only",
)

SEED = 0x5EED1234ABCD

# fp32 unit roundoff; the PTX ISA's bounds for ex2.approx (2 ulp, relative) and lg2.approx (absolute
# in (0.5, 2), relative elsewhere)
U = 2.0**-24
EX2_ERR = 2.0**-22
LG2_ERR = 2.0**-22


def _sm():
    from sglang.kernels.ops.sampling import softmax_jit

    return softmax_jit


def _generator():
    torch.cuda.init()
    return torch.cuda.default_generators[torch.cuda.current_device()]


def _run(logits, temperatures, top_ks, offset=0, seed=SEED, out=None):
    """One call with the default generator at (seed, offset)."""
    gen = _generator()
    gen.manual_seed(seed)
    gen.set_offset(offset)
    return _sm().softmax_sample(logits, temperatures, top_ks=top_ks, out=out)


def _inputs(rows, vocab, seed=0, greedy_frac=0.3):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(rows, vocab, device="cuda", generator=g) * 3.0
    temps = torch.rand(rows, device="cuda", generator=g) * 1.5 + 0.1
    top_ks = torch.full((rows,), 20, dtype=torch.int32, device="cuda")
    top_ks[torch.rand(rows, device="cuda", generator=g) < greedy_frac] = 1
    return logits, temps, top_ks


def _probs_tolerance(logits, temps):
    """Per-row bound on |p - p_ref| / p between this kernel's probs and the softmax kernel's.

    Both compute p_i = 2^((x_i - f) * s) / S, s = log2(e) / T, from frames f (running, warp and row
    maxima, |f| <= A = the row's largest finite |x|) and an fp32 sum S of such terms. Per kernel:
    - exponent (log2 units): x - f and * s round once each (2u|a| <= 4uAs), moving a partial to the
      row's frame rounds twice more (4uAs), this kernel's FMA with -lg2(S) once more
      (u (2As + 18), S < 2^18) plus lg2's error (18 * LG2_ERR): d <= 10uAs + 18 (u + LG2_ERR)
      (the softmax kernel's part is the smaller one);
      relative error ln2 * d;
    - ex2: EX2_ERR for the element, EX2_ERR for its frame factor;
    - S: its terms carry the same relative errors, plus the sum's own rounding, at most u per
      addition on a term's path: <= 88 (thread) + 8 (frame rescales) + 5 (warp) + 8 (pool) + 5 = 114;
    - the final scale / divide: 2u.
    So per kernel e <= 2 (ln2 d + 2 EX2_ERR) + 116u, and the two kernels differ by at most 2e.
    """
    a = torch.where(logits.isfinite(), logits.abs(), torch.zeros_like(logits))
    a = a.amax(-1).double()
    t = torch.ones_like(a) if temps is None else temps.view(-1).double()
    a_s = a * math.log2(math.e) / t
    d = 10 * U * a_s + 18 * (U + LG2_ERR)
    e = 2 * (math.log(2) * d + 2 * EX2_ERR) + 116 * U
    return (2 * e).view(-1, 1)


def _assert_probs_close(probs, ref, logits, temps):
    """`probs` within `_probs_tolerance` of `ref`; ex2's flush to zero below 2^-126 as slack.
    NaN rows (a NaN logit makes the whole row NaN) must match."""
    tol = _probs_tolerance(logits, temps)
    p, r = probs.double(), ref.double()
    nan = p.isnan()
    assert torch.equal(nan, r.isnan())
    p, r = p.masked_fill(nan, 0.0), r.masked_fill(nan, 0.0)
    err = (p - r).abs() - tol * torch.maximum(p, r)
    assert (err <= 2.0**-125).all(), (err.max().item(), tol.max().item())


def _check_consistency(logits, temps, top_ks, probs, topk_p, topk_index):
    _assert_probs_close(probs, _sm().softmax(logits, temps), logits, temps)
    idx = topk_index.view(-1)
    assert topk_index.shape == (logits.shape[0], 1) and topk_index.dtype == torch.int64
    assert topk_p.shape == (logits.shape[0], 1) and topk_p.dtype == torch.float32
    gathered = probs.gather(1, topk_index)
    assert torch.equal(topk_p.view(torch.int32), gathered.view(torch.int32))
    greedy = top_ks <= 1
    if greedy.any():
        assert torch.equal(idx[greedy], torch.argmax(logits[greedy], dim=-1))
    sample = ~greedy
    if sample.any():
        assert (topk_p.view(-1)[sample] > 0).all()


def _philox_words(seed, offset, groups):
    """curand's Philox4x32-10 at counter (offset / 4, group) under key seed: [*, 4] words."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import _philox

    ctr = offset // 4
    lo = torch.full_like(groups, ctr & 0xFFFFFFFF)
    hi = torch.full_like(groups, ctr >> 32)
    return torch.stack(_philox(seed, lo, hi, groups & 0xFFFFFFFF, groups >> 32), -1)


def _exponential(words):
    """torch's exponential_ of curand_uniform's u (u in fp32, the log in fp64)."""
    u = words.float() * 2.0**-32 + 2.0**-33
    return torch.where(u >= 1 - 2.0**-24, 2.0**-24, -torch.log(u.double()))


def _reference_race(logits, temps, seed, offset):
    """fp64 exponential race: (log p - log E, E) per element; element i of row r takes word
    i % 4 of group r * vocab / 4 + i / 4."""
    rows, vocab = logits.shape
    groups = torch.arange(rows * vocab // 4, device=logits.device, dtype=torch.int64)
    e = _exponential(_philox_words(seed, offset, groups)).view(rows, vocab)
    log_p = torch.log_softmax(logits.double() / temps.double().view(-1, 1), dim=-1)
    return log_p - torch.log(e), e


def test_reference_rng_matches_torch_exponential():
    # Self-check of the reference: with numel <= 1024 torch's exponential_ gives element i
    # word x of counter (offset / 4, i), i.e. the curand layout and transform the kernel uses.
    gen = _generator()
    n = 1000
    gen.manual_seed(SEED)
    gen.set_offset(40)
    torch_e = torch.empty(n, device="cuda").exponential_().double()
    groups = torch.arange(n, device="cuda")
    ref = _exponential(_philox_words(SEED, 40, groups))[:, 0]
    # __logf: about 4e-7 absolute error near u = 1
    assert torch.allclose(torch_e, ref, rtol=1e-5, atol=1e-6)
    shifted = _exponential(_philox_words(SEED, 44, groups))[:, 0]
    assert not torch.allclose(torch_e, shifted, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "rows,vocab",
    [
        (1, 248320),
        (3, 248320),
        (64, 151936),
        (256, 248320),
        (7, 129280),
        (33, 163840),
        (16, 151552),
        (128, 4096),
        (2, 1024),
    ],
)
def test_matches_softmax_and_argmax(rows, vocab):
    logits, temps, top_ks = _inputs(rows, vocab, seed=rows * 7 + vocab)
    out = _run(logits, temps, top_ks, offset=12)
    _check_consistency(logits, temps, top_ks, *out)


def test_probs_tolerance_detects_error():
    # Self-check of the comparison: an error 4x the bound is caught
    logits, temps, _ = _inputs(4, 151936)
    ref = _sm().softmax(logits, temps)
    off = ref * (1 + 4 * _probs_tolerance(logits, temps))
    _assert_probs_close(ref, ref, logits, temps)
    with pytest.raises(AssertionError):
        _assert_probs_close(off, ref, logits, temps)


def test_no_temperature():
    logits, _, top_ks = _inputs(8, 151936)
    probs, topk_p, topk_index = _run(logits, None, top_ks)
    _check_consistency(logits, None, top_ks, probs, topk_p, topk_index)


@pytest.mark.parametrize("rows,vocab,scale", [(256, 4096, 0.5), (64, 248320, 1.0)])
def test_matches_reference_draw(rows, vocab, scale):
    # Same Philox words and transform as the fp64 race. A row may pick another element only when
    # the two keys (natural log units) are within the kernel's error, summed over both elements:
    # - E = -lg2.approx(u) is off by at most LG2_ERR absolute (u in (0.5, 1)) or relative, i.e.
    #   LG2_ERR ln2 / e relative in e (e = E ln2), doubled for margin;
    # - key = (x - f) s - lg2(E) in fp32 from frames |f| <= A: 10uAs + u|lg2 E| (roundings, as in
    #   `_probs_tolerance`) plus lg2(E)'s own error LG2_ERR max(1, |lg2 E|), times ln2.
    logits, temps, _ = _inputs(rows, vocab, seed=3)
    logits *= scale
    top_ks = torch.full((rows,), 20, dtype=torch.int32, device="cuda")
    offset = (7 << 34) + 20  # both counter words of the offset nonzero
    idx = _run(logits, temps, top_ks, offset=offset)[2].view(-1)
    key, e = _reference_race(logits, temps, SEED, offset)
    ref = key.argmax(dim=1)
    a_s = logits.abs().amax(-1).double() * math.log2(math.e) / temps.double()
    for r in (ref != idx).nonzero().view(-1).tolist():
        a, b = ref[r].item(), idx[r].item()
        gap = (key[r, a] - key[r, b]).item()
        bound = 0.0
        for i in (a, b):
            ei = e[r, i].item()
            lg2_e = abs(math.log2(ei / math.log(2)))
            d = 10 * U * a_s[r].item() + U * lg2_e + LG2_ERR * max(1.0, lg2_e)
            bound += 2 * LG2_ERR * math.log(2) / ei + math.log(2) * d
        assert gap <= bound, (r, a, b, gap, bound)


def test_generator_state():
    # Same (seed, offset) draws the same; each call advances the offset by 4
    logits = _inputs(64, 151936)[0] / 3.0  # flat enough that draws vary
    temps = torch.ones(64, device="cuda")
    top_ks = torch.full((64,), 20, dtype=torch.int32, device="cuda")
    sm = _sm()
    gen = _generator()
    gen.manual_seed(SEED)
    a = sm.softmax_sample(logits, temps, top_ks=top_ks)[2]
    assert gen.get_offset() == 4
    b = sm.softmax_sample(logits, temps, top_ks=top_ks)[2]
    assert gen.get_offset() == 8
    torch.manual_seed(SEED)
    assert torch.equal(sm.softmax_sample(logits, temps, top_ks=top_ks)[2], a)
    assert torch.equal(_run(logits, temps, top_ks, offset=4)[2], b)
    c = _run(logits, temps, top_ks, seed=SEED + 1)[2]
    # An independent draw repeats a row's token with probability sum p^2 < 0.01;
    # more than 6 of 64 rows then has probability < 1e-6
    probs = torch.softmax(logits.double() / temps.double().view(-1, 1), -1)
    assert (probs**2).sum(-1).max().item() < 0.01
    assert (a == b).sum().item() <= 6
    assert (a == c).sum().item() <= 6


def test_draw_independent_of_batch_size():
    # The counter holds (row, element), so a row's draw does not depend on the launch config
    logits, temps, _ = _inputs(256, 248320)
    top_ks = torch.full((256,), 20, dtype=torch.int32, device="cuda")
    full = _run(logits, temps, top_ks, offset=8)[2].view(-1)
    for rows in (1, 4, 64):
        part = _run(logits[:rows], temps[:rows], top_ks[:rows], offset=8)[2].view(-1)
        assert torch.equal(part, full[:rows])


def test_rows_independent():
    # Identical rows draw independently: each of 2048 pairs agrees with probability q = sum p^2
    vocab, rows = 4096, 4096
    base = torch.randn(1, vocab, device="cuda") * 0.5
    logits = base.repeat(rows, 1)
    top_ks = torch.full((rows,), 20, dtype=torch.int32, device="cuda")
    probs, _, idx = _run(logits, None, top_ks, offset=4)
    idx = idx.view(-1)
    q = (probs[0].double() ** 2).sum().item()
    same = (idx[0::2] == idx[1::2]).sum().item()
    mean = rows // 2 * q
    # mean + 6 sd + 3 of a binomial: false-failure rate < 1e-6
    assert same <= mean + 6 * math.sqrt(mean) + 3, (same, mean)


def _chi2_pvalue(observed, expected):
    stat = float((((observed - expected) ** 2) / expected).sum())
    dof = len(observed) - 1
    try:
        from scipy.stats import chi2

        return float(chi2.sf(stat, dof)), stat, dof
    except ImportError:  # Wilson-Hilferty
        z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
        return 0.5 * math.erfc(z / math.sqrt(2)), stat, dof


@pytest.mark.parametrize("temperature", [0.3, 1.0, 1.6])
def test_chi_square_small_vocab(temperature):
    # Under the null hypothesis each case fails with probability 1e-4 (the p-value threshold)
    vocab, rows, calls = 4096, 2048, 40
    g = torch.Generator(device="cuda").manual_seed(17)
    row = torch.randn(vocab, device="cuda", generator=g)
    row[:6] += torch.tensor([6.0, 5.5, 5.0, 4.0, 3.0, 2.0], device="cuda")
    logits = row.repeat(rows, 1)
    temps = torch.full((rows,), temperature, device="cuda")
    top_ks = torch.full((rows,), 20, dtype=torch.int32, device="cuda")
    counts = torch.zeros(vocab, dtype=torch.float64, device="cuda")
    probs = None
    _generator().manual_seed(1000 + int(temperature * 10))
    for _ in range(calls):
        probs, _, idx = _sm().softmax_sample(logits, temps, top_ks=top_ks)
        counts += torch.bincount(idx.view(-1), minlength=vocab).double()
    n = rows * calls
    expected = probs[0].double() * n
    # Bins with expected >= 5 kept, the rest pooled into one bin
    big = expected >= 5
    obs = torch.cat([counts[big], counts[~big].sum().view(1)])
    exp = torch.cat([expected[big], expected[~big].sum().view(1)])
    if exp[-1] < 5:
        obs, exp = obs[:-1], exp[:-1]
        obs[-1] += counts[~big].sum()
        exp[-1] += expected[~big].sum()
    pvalue, stat, dof = _chi2_pvalue(obs.cpu(), exp.cpu())
    assert pvalue > 1e-4, (pvalue, stat, dof)


def test_two_level_mass_like_torch():
    # Half the vocab at twice the weight: the heavy half must win 2/3 of the draws. In a 248K row
    # the winner's E sits where lg2 / __logf are coarsest, so this checks that error against the
    # true mass, for the fused draw and torch's stock draw alike. Each check is two-sided at
    # 5 standard deviations: false-failure rate < 1e-6.
    from sglang.srt.speculative.spec_utils import sample_draft_proposal

    vocab, rows, calls = 248320, 256, 200
    logits = torch.zeros(rows, vocab, device="cuda")
    logits[:, vocab // 2 :] = math.log(2.0)
    temps = torch.ones(rows, 1, device="cuda")
    top_ks = torch.full((rows,), 20, dtype=torch.int32, device="cuda")
    n = rows * calls
    sd = math.sqrt(2 / 9 / n)
    _generator().manual_seed(31)
    for fused in (True, False):
        heavy = 0
        for _ in range(calls):
            idx = sample_draft_proposal(logits, temps, top_ks, fused=fused)[2]
            heavy += (idx >= vocab // 2).sum().item()
        assert abs(heavy / n - 2 / 3) < 5 * sd, (fused, heavy / n)


def test_top_token_frequencies_large_vocab():
    # 11 two-sided checks at 5 standard deviations: false-failure rate < 1e-5
    vocab, rows, calls = 248320, 256, 40
    g = torch.Generator(device="cuda").manual_seed(23)
    row = torch.randn(vocab, device="cuda", generator=g) * 1.5
    row[[5, 777, 100000, 248319]] += torch.tensor([9.0, 8.5, 8.0, 7.5], device="cuda")
    logits = row.repeat(rows, 1)
    temps = torch.full((rows,), 0.8, device="cuda")
    top_ks = torch.full((rows,), 20, dtype=torch.int32, device="cuda")
    counts = torch.zeros(vocab, dtype=torch.float64, device="cuda")
    _generator().manual_seed(23)
    for _ in range(calls):
        probs, _, idx = _sm().softmax_sample(logits, temps, top_ks=top_ks)
        counts += torch.bincount(idx.view(-1), minlength=vocab).double()
    n = rows * calls
    p = probs[0].double()
    top = torch.topk(p, 10).indices
    for t in top.tolist():
        e, o = p[t].item() * n, counts[t].item()
        assert abs(o - e) < 5 * math.sqrt(e * (1 - p[t].item())) + 1, (t, o, e)
    rest_e = (1 - p[top].sum().item()) * n
    rest_o = n - counts[top].sum().item()
    assert abs(rest_o - rest_e) < 5 * math.sqrt(rest_e) + 1


def test_cuda_graph_replay():
    # Each replay draws at the generator's next offsets (+4 per call) and equals eager calls there
    sm = _sm()
    logits, temps, _ = _inputs(16, 151936)
    logits /= 3.0
    top_ks = torch.full((16,), 20, dtype=torch.int32, device="cuda")
    out = torch.empty_like(logits)
    holder = {}

    def step():
        holder["a"] = sm.softmax_sample(logits, temps, top_ks=top_ks, out=out)[2]
        holder["b"] = sm.softmax_sample(logits, temps, top_ks=top_ks, out=out)[2]

    gen = _generator()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    gen.manual_seed(SEED)
    gen.set_offset(100)
    draws = []
    for _ in range(3):
        graph.replay()
        draws.append((holder["a"].clone(), holder["b"].clone()))
    torch.cuda.synchronize()
    assert gen.get_offset() == 100 + 3 * 8
    for i, (a, b) in enumerate(draws):
        assert torch.equal(a, _run(logits, temps, top_ks, offset=100 + 8 * i)[2])
        assert torch.equal(b, _run(logits, temps, top_ks, offset=104 + 8 * i)[2])
    assert (draws[0][0] != draws[1][0]).any()
    assert (draws[0][0] != draws[0][1]).any()


def test_edge_rows():
    vocab = 151936
    rows = 8
    logits = torch.randn(rows, vocab, device="cuda")
    # 0: grammar-style mask, only 5 tokens finite
    logits[0] = float("-inf")
    allowed = torch.tensor([3, 70000, 70001, 150000, 151935], device="cuda")
    logits[0, allowed] = torch.randn(5, device="cuda")
    # 1: very peaked
    logits[1, 4242] = 80.0
    # 2: sanitizer sentinels, two tied +1e30 tokens
    logits[2, 9] = 1e30
    logits[2, 90000] = 1e30
    logits[2, 100:200] = -1e30
    # 3: NaNs sprinkled, a huge NaN-free max elsewhere
    logits[3, ::1000] = float("nan")
    logits[3, 12345] = 40.0
    # 4: all -inf
    logits[4] = float("-inf")
    # 5 / 6: tiny and large temperature; 7: masked like 0 but greedy
    logits[7] = logits[0]
    temps = torch.tensor([1.0, 1.0, 0.7, 1.0, 1.0, 0.1, 1.6, 1.0], device="cuda")
    top_ks = torch.tensor(
        [20, 20, 20, 20, 20, 20, 20, 1], dtype=torch.int32, device="cuda"
    )
    # Row 2's +1e30 makes its probs bound vacuous; the other rows are compared
    checked = [0, 1, 3, 4, 5, 6, 7]
    ref = _sm().softmax(logits, temps)
    seen2 = set()
    # Row 2 draws each tied token with probability 1/2: missing one in 64 calls has P = 2^-63
    for call in range(64):
        probs, topk_p, idx = _run(logits, temps, top_ks, offset=4 * call)
        idx = idx.view(-1)
        if call == 0:
            _assert_probs_close(
                probs[checked], ref[checked], logits[checked], temps[checked]
            )
        assert idx[0].item() in allowed.tolist()
        assert idx[1].item() == 4242
        assert idx[2].item() in (9, 90000)
        seen2.add(idx[2].item())
        assert idx[3].item() == 12345
        assert idx[4].item() == 0
        assert idx[7].item() == torch.argmax(logits[7]).item()
        for r in (0, 1, 2, 5, 6, 7):  # row 3: NaN logits make the whole softmax row NaN
            assert topk_p[r].item() > 0
            assert (
                topk_p[r].view(torch.int32).item()
                == probs[r, idx[r]].view(torch.int32).item()
            )
    assert seen2 == {9, 90000}, "tied +1e30 tokens must both be drawn"


def test_greedy_ties_lowest_index_and_nan():
    vocab = 248320
    logits = torch.randn(4, vocab, device="cuda")
    logits[0, [100, 200000, 5]] = 9.0  # tie: lowest index 5
    logits[1, :] = 0.0  # all equal: 0
    logits[2, 0] = float("nan")
    logits[2, 77] = 9.0
    logits[3, 248319] = 9.0
    top_ks = torch.ones(4, dtype=torch.int32, device="cuda")
    _, _, idx = _run(logits, torch.ones(4, device="cuda"), top_ks)
    assert idx.view(-1).tolist() == [5, 0, 77, 248319]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
