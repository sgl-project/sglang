# SPDX-License-Identifier: Apache-2.0
"""Fused int8 DSpark markov walk (ops.speculative.dspark.markov_walk), synthetic W.

References replay the int8 planes of quantize_markov (the walker's own quantization)
in float64, emulating each kernel's fp32 op order and its bf16 rounding of the logits;
ties go to the smallest row, as in torch.argmax.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dspark import markov_walk as mw
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="the DSpark markov-walk kernels are sm_90 only",
)

DEV = "cuda"
SENTINEL = -777.0  # exact in bf16
# case -> (wgmma tiles per CTA, gamma, batch sizes): the standard layout routes bs
# 1 / 2..4 / 5..64 to single / small_batch / wgmma, the big-vocabulary one every bs to wgmma.
CASES = {"std": (18, 16, (1, 3, 4, 5, 33, 64)), "big": (30, 15, (1, 5, 33, 64))}


def _vocab_for(tiles: int) -> int:
    """Qwen3 / Qwen3.6 vocabularies on 132 SMs, else one with the same tile count."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms == 132:
        return {18: 151936, 30: 248320}[tiles]
    return sms * mw.WGMMA_TILE_ROWS * tiles - 5120


class Reference:
    """Exact replays of the kernels' arithmetic on the walker's int8 planes."""

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor):
        planes = mw.quantize_markov(w1, w2)
        self.vocab = w2.shape[0]
        self.q_hi, self.q_lo = planes["q_hi"], planes["q_lo"]
        self.s_hi, self.s_lo = planes["s_hi"].double(), planes["s_lo"].double()
        self.row_scale = planes["row_scale"].double()
        self.qw2_t = planes["q_w2"].float().t().contiguous()  # [256, V], integer-valued

    def logits(
        self, kind: str, base_k: torch.Tensor, prev: torch.Tensor
    ) -> torch.Tensor:
        """fp32 logits [n, V] in the kernel's op order, before its bf16 rounding
        (the int8 dots are exact in fp32)."""

        def f32(x):
            return x.float().double()

        d_hi = (self.q_hi[prev].float() @ self.qw2_t).double()
        d_lo = (self.q_lo[prev].float() @ self.qw2_t).double()
        s_hi, s_lo = self.s_hi[prev][:, None], self.s_lo[prev][:, None]
        base = base_k.double()
        if kind in (
            "single",
            "small_batch",
        ):  # fmul(sr, fma(s_hi, d_hi, fmul(s_lo, d_lo))) + base
            bias = f32(self.row_scale * f32(s_hi * d_hi + f32(s_lo * d_lo)))
            return (base + bias).float()
        bias = f32(
            s_lo * f32(256.0 * d_hi + d_lo)
        )  # wgmma: fma(sr, fmul(s_lo, 256 d_hi + d_lo), base)
        return (self.row_scale * bias + base).float()

    def greedy(
        self, kind: str, base: torch.Tensor, anchor: torch.Tensor
    ) -> torch.Tensor:
        prev, toks = anchor.clone(), []
        for k in range(base.shape[1]):
            prev = _first_argmax(self.logits(kind, base[:, k], prev).bfloat16().float())
            toks.append(prev)
        return torch.stack(toks, 1)

    def path_logits(
        self, kind: str, base: torch.Tensor, anchor: torch.Tensor, toks: torch.Tensor
    ):
        """bf16 logits [n, K, V] along a given token path."""
        out = torch.empty(base.shape, dtype=torch.bfloat16, device=base.device)
        prev = anchor.clone()
        for k in range(base.shape[1]):
            out[:, k] = self.logits(kind, base[:, k], prev).bfloat16()
            prev = toks[:, k]
        return out


def _first_argmax(x: torch.Tensor) -> torch.Tensor:
    m = x.max(-1, keepdim=True).values
    idx = torch.arange(x.shape[-1], device=x.device)
    return torch.where(x == m, idx, x.shape[-1]).min(-1).values


def _synthetic_weights(vocab: int):
    gen = torch.Generator(device=DEV).manual_seed(vocab)
    row_norm = torch.exp(0.3 * torch.randn(vocab, 1, device=DEV, generator=gen))
    w2 = 0.05 * row_norm * torch.randn(vocab, mw.MARKOV_RANK, device=DEV, generator=gen)
    w1 = torch.randn(vocab, mw.MARKOV_RANK, device=DEV, generator=gen)
    return w1.bfloat16(), w2.bfloat16()


def _batch(walker: mw.MarkovWalker, bs: int, seed: int):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    base = 2 * torch.randn(bs, walker.gamma, walker.vocab, device=DEV, generator=gen)
    anchor = torch.randint(0, walker.vocab, (bs,), device=DEV, generator=gen)
    anchor[-1] = walker.vocab - 1  # the last W1 row: the last CTA's tail tile
    return base.bfloat16(), anchor


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request):
    tiles, gamma, batch_sizes = CASES[request.param]
    w1, w2 = _synthetic_weights(_vocab_for(tiles))
    walker = mw.MarkovWalker(w1, w2, gamma=gamma, device=torch.device(DEV), seed=1234)
    assert walker.weights.tiles == tiles
    assert walker.weights.stream_mask == (0x4A96 if tiles == 18 else 0x1BCB78F6)
    walker.warmup()
    yield walker, Reference(w1, w2), w1, w2, batch_sizes
    del walker
    torch.cuda.empty_cache()


def test_greedy_matches_int8_replay(case):
    """Greedy tokens equal an exact replay of the kernel's int8 arithmetic and an
    all-greedy round leaves corrected_out untouched: a wrong fragment / tile / W1
    byte order, fp32 op order or tie-break moves tokens. The anchor is a strided
    column view, as the DSpark draft hands it over."""
    walker, ref, _, _, batch_sizes = case
    for bs in batch_sizes:
        base, anchor = _batch(walker, bs, seed=bs)
        ids = torch.zeros(bs, walker.gamma + 1, dtype=torch.int64, device=DEV)
        ids[:, 0] = anchor
        tokens = torch.empty(bs * walker.gamma, dtype=torch.int64, device=DEV)
        corrected = torch.full(
            (bs * walker.gamma, walker.vocab),
            SENTINEL,
            dtype=torch.bfloat16,
            device=DEV,
        )
        out = walker.walk(base, ids[:, 0], None, tokens, corrected)
        kind = walker.kernel_for(bs)
        assert torch.equal(out, ref.greedy(kind, base, anchor)), (kind, bs)
        assert bool((corrected == SENTINEL).all()), (kind, bs)


def test_greedy_near_fp64_dequant_reference(case):
    """On the kernel's own path, a greedy token may differ from the fp64 argmax of
    base + W2 W1[prev] on the dequantized weights only at a near-tie (bf16 rounding
    of the logits); a layout or stride bug shows as gaps of whole logits. The
    dequantized weights stay within 2 % of the bf16 originals."""
    walker, ref, w1, w2, batch_sizes = case
    w2_deq = (ref.qw2_t.double() * ref.row_scale[None]).t()
    w1_deq = (
        ref.s_hi[:, None] * ref.q_hi.double() + ref.s_lo[:, None] * ref.q_lo.double()
    )
    for w, w_deq in ((w1, w1_deq), (w2, w2_deq)):
        assert float((w_deq - w.double()).norm() / w.double().norm()) < 0.02
    max_gap, n_diff = 0.0, 0
    for bs in (batch_sizes[0], batch_sizes[1], batch_sizes[-1]):
        base, anchor = _batch(walker, bs, seed=7 + bs)
        tokens = torch.empty(bs * walker.gamma, dtype=torch.int64, device=DEV)
        out = walker.walk(base, anchor, None, tokens)
        prev = anchor
        for k in range(walker.gamma):
            lg = base[:, k].double() + w1_deq[prev] @ w2_deq.t()
            best = lg.argmax(-1)
            gap = lg.gather(1, best[:, None]) - lg.gather(1, out[:, k : k + 1])
            n_diff += int((best != out[:, k]).sum())
            max_gap = max(max_gap, float(gap.max()))
            prev = out[:, k]
    assert max_gap < 0.25, (n_diff, max_gap)


def test_sampling_writes_exact_corrected_logits(case):
    """In a mixed batch, T > 0 requests' corrected_out rows are bit-equal to the
    bf16 logits along the kernel's own path (the verifier rebuilds q from them);
    T <= 0 / NaN requests walk greedily and leave their rows untouched (stale rows
    reach the mixed-batch verifier's softmax). +inf, 1e30 and a denormal clamp."""
    walker, ref, _, _, batch_sizes = case
    temps_cycle = [1.0, 0.0, 0.6, float("nan"), float("inf"), -1.0, 1e30, 1e-40]
    for bs in batch_sizes[:3]:
        base, anchor = _batch(walker, bs, seed=100 + bs)
        sel = [temps_cycle[(i + bs) % len(temps_cycle)] for i in range(bs)]
        temps = torch.tensor(sel, dtype=torch.float32, device=DEV)
        tokens = torch.empty(bs * walker.gamma, dtype=torch.int64, device=DEV)
        corrected = torch.full(
            (bs * walker.gamma, walker.vocab),
            SENTINEL,
            dtype=torch.bfloat16,
            device=DEV,
        )
        out = walker.walk(base, anchor, temps, tokens, corrected)
        kind = walker.kernel_for(bs)
        cv = corrected.view(bs, walker.gamma, walker.vocab)
        assert bool(((out >= 0) & (out < walker.vocab)).all())
        for i, t in enumerate(sel):
            if t > 0:
                expect = ref.path_logits(
                    kind, base[i : i + 1], anchor[i : i + 1], out[i : i + 1]
                )
                assert torch.equal(
                    cv[i : i + 1].view(torch.int16), expect.view(torch.int16)
                ), (kind, bs, t)
            else:
                assert torch.equal(
                    out[i : i + 1], ref.greedy(kind, base[i : i + 1], anchor[i : i + 1])
                ), (kind, t)
                assert bool((cv[i] == SENTINEL).all()), (kind, bs, t)


def _chi2_pvalue(samples: torch.Tensor, prob: torch.Tensor) -> float:
    """Pearson chi-square p-value of samples against prob: head tokens with an
    expected count >= 20 one category each (<= 64), the tail in <= 16 bins."""
    from scipy.stats import chi2

    n = samples.numel()
    order = prob.argsort(descending=True)
    ps = prob[order]
    head = int(min(64, int((ps * n >= 20).sum())))
    tail_mass = float(ps[head:].sum())
    n_tail = max(1, min(16, int(tail_mass * n / 20)))
    cat = torch.empty(prob.numel(), dtype=torch.int64, device=prob.device)
    cat[order[:head]] = torch.arange(head, device=prob.device)
    before = torch.cumsum(ps[head:], 0) - ps[head:]
    cat[order[head:]] = head + (
        before / max(tail_mass, 1e-300) * n_tail
    ).floor().long().clamp(0, n_tail - 1)
    expected = (
        torch.zeros(head + n_tail, dtype=torch.float64, device=prob.device).index_add_(
            0, cat, prob
        )
        * n
    )
    counts = torch.bincount(cat[samples], minlength=head + n_tail).double()
    keep = expected >= 5
    stat = float(((counts[keep] - expected[keep]) ** 2 / expected[keep]).sum())
    return float(chi2.sf(stat, int(keep.sum()) - 1))


def _launch_one_step(walker, kind, base, anchor, temps, tokens):
    w = walker.weights
    common = dict(
        row_scale=w.row_scale,
        base_logits=base,
        anchor=anchor,
        tokens_out=tokens,
        corrected_out=None,
        state=walker.states[kind],
        temps=temps,
        num_steps=1,
        valid_rows=walker.vocab,
        seed=walker.seeds[kind],
    )
    if kind == "wgmma":
        mw.markov_walk_wgmma(
            w2_res=w.w2_res,
            w2_str=w.w2_str,
            w1f=w.w1f,
            stream_mask=w.stream_mask,
            **common,
        )
    elif kind == "small_batch":
        mw.markov_walk_small_batch(frag=w.frag, w1q=w.w1q, **common)
    else:
        mw.markov_walk_single(frag=w.frag, w1q=w.w1q, **common)


@pytest.mark.parametrize("kind", ["single", "small_batch", "wgmma"])
def test_sampling_distribution(case, kind):
    """Single-step samples follow softmax(bf16 logits / T) (chi-square p > 1e-4)
    for a flat and a peaked row at T in {0.6, 1, 2}: guards the Gumbel and
    inverse-CDF transforms and the per-request, per-launch noise counters."""
    walker, ref, _, _, _ = case
    if walker.weights.big_vocab and kind != "wgmma":
        pytest.skip("the big-vocabulary layout has no single / small_batch weights")
    n_samples = 8000
    cand_base, cand_anchor = _batch(walker, 16, seed=211)
    top1 = torch.softmax(
        ref.logits(kind, cand_base[:, 0], cand_anchor).bfloat16().float(), -1
    ).amax(-1)
    flat, peaked = int(top1.argmin()), int((top1 - 0.6).abs().argmin())
    cases = [(row, t) for row in (flat, peaked) for t in (0.6, 1.0, 2.0)]
    rows = torch.tensor([c[0] for c in cases], device=DEV)
    base6 = cand_base[rows, :1].contiguous()
    anchor6 = cand_anchor[rows].contiguous()
    temps6 = torch.tensor([c[1] for c in cases], dtype=torch.float32, device=DEV)
    # per launch: one case for single, three for small_batch, 6 cases x 8 for wgmma (two 32-request blocks)
    groups = {
        "single": [[i] for i in range(6)],
        "small_batch": [[0, 1, 2], [3, 4, 5]],
        "wgmma": [list(range(6)) * 8],
    }[kind]
    samples = [[] for _ in cases]
    for grp in groups:
        idx = torch.tensor(grp, device=DEV)
        n_launch = -(-n_samples // (len(grp) // len(set(grp))))
        out = torch.empty(n_launch, len(grp), dtype=torch.int64, device=DEV)
        base, anchor, temps = (
            base6[idx].contiguous(),
            anchor6[idx].contiguous(),
            temps6[idx].contiguous(),
        )
        for i in range(n_launch):
            _launch_one_step(walker, kind, base, anchor, temps, out[i])
        for j, c in enumerate(grp):
            samples[c].append(out[:, j])
    for c, (row, t) in enumerate(cases):
        lg = (
            ref.logits(kind, base6[c : c + 1, 0], anchor6[c : c + 1])
            .bfloat16()
            .double()[0]
        )
        p_value = _chi2_pvalue(torch.cat(samples[c]), torch.softmax(lg / t, 0))
        assert p_value > 1e-4, (kind, "flat" if row == flat else "peaked", t, p_value)


def test_nan_and_inf_rows_stay_in_range(case):
    """A padded graph row of all NaN / -inf (or half NaN) logits still yields
    tokens in [0, V) on every step, greedy and sampled: the next step gathers
    W1[token], and an out-of-range token is an illegal address. wgmma answers 0, as
    torch.argmax would; the finite rows are unaffected."""
    walker, ref, _, _, _ = case
    for bs in (1, 2, 5, 33):
        base, anchor = _batch(walker, bs, seed=31 + bs)
        base[0] = float("nan")
        if bs > 1:
            base[1] = float("-inf")
        if bs > 2:
            base[2, :, ::2] = float("nan")
        kind = walker.kernel_for(bs)
        tokens = torch.empty(bs * walker.gamma, dtype=torch.int64, device=DEV)
        corrected = torch.zeros(
            bs * walker.gamma, walker.vocab, dtype=torch.bfloat16, device=DEV
        )
        for temps in (None, torch.ones(bs, dtype=torch.float32, device=DEV)):
            out = walker.walk(base, anchor, temps, tokens, corrected)
            assert bool(((out >= 0) & (out < walker.vocab)).all()), (kind, bs)
            if kind == "wgmma":
                assert bool((out[: min(bs, 2)] == 0).all()), (kind, bs)
            if temps is None and bs > 3:
                assert torch.equal(out[3:], ref.greedy(kind, base[3:], anchor[3:])), (
                    kind,
                    bs,
                )


@pytest.mark.parametrize("bs", [1, 5])
def test_graph_replay_draws_fresh_noise(case, bs):
    """Each replay of one captured graph advances the in-kernel round counter, so
    sampled proposals change every round while greedy replays stay exact; temps
    are staged in-graph from (greedy_mask, temperatures), as the draft does."""
    walker, ref, _, _, _ = case
    kind = walker.kernel_for(bs)
    base, anchor = _batch(walker, bs, seed=5)
    greedy_mask = torch.zeros(bs, dtype=torch.bool, device=DEV)
    temperatures = torch.full((bs,), 2.0, dtype=torch.float32, device=DEV)
    zero = torch.zeros((), dtype=torch.float32, device=DEV)
    tokens = torch.empty(bs * walker.gamma, dtype=torch.int64, device=DEV)
    corrected = torch.zeros(
        bs * walker.gamma, walker.vocab, dtype=torch.bfloat16, device=DEV
    )

    def body():
        temps = walker.temps_buf[:bs]
        torch.where(greedy_mask, zero, temperatures, out=temps)
        walker.walk(base, anchor, temps, tokens, corrected)

    body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    state = walker.states[kind]
    seen = []
    for _ in range(3):
        round_before = int(state[0])
        graph.replay()
        torch.cuda.synchronize()
        assert int(state[0]) == round_before + 1, kind
        seen.append(tokens.clone())
    assert not torch.equal(seen[0], seen[1]) and not torch.equal(seen[1], seen[2]), kind
    greedy_mask.fill_(True)
    graph.replay()
    assert torch.equal(tokens.view(bs, -1), ref.greedy(kind, base, anchor)), kind


def test_host_rejects_bad_arguments(case):
    """Bad shapes, dtypes, strides, step counts and batch sizes fail on the host
    with an exception, never as a device fault that poisons the CUDA context (a
    valid walk after them still matches the exact replay)."""
    walker, ref, _, _, _ = case
    w = walker.weights
    bs = 5
    base, anchor = _batch(walker, bs, seed=9)
    temps = torch.zeros(bs, dtype=torch.float32, device=DEV)
    tokens = torch.empty(bs * walker.gamma, dtype=torch.int64, device=DEV)

    def wgmma(**over):
        args = dict(
            w2_res=w.w2_res,
            w2_str=w.w2_str,
            row_scale=w.row_scale,
            w1f=w.w1f,
            base_logits=base,
            anchor=anchor,
            tokens_out=tokens,
            corrected_out=None,
            state=walker.states["wgmma"],
            temps=temps,
            num_steps=walker.gamma,
            valid_rows=walker.vocab,
            seed=1,
            stream_mask=w.stream_mask,
        )
        args.update(over)
        mw.markov_walk_wgmma(**args)

    big = torch.zeros(
        mw.MAX_BS + 1, walker.gamma, walker.vocab, dtype=torch.bfloat16, device=DEV
    )
    bad_calls = [
        dict(base_logits=base.half()),
        dict(base_logits=base.transpose(1, 2).contiguous().transpose(1, 2)),
        dict(base_logits=base.cpu()),
        dict(num_steps=mw.MAX_STEPS + 1),
        dict(num_steps=walker.gamma + 1),
        dict(valid_rows=walker.vocab - 4),
        dict(valid_rows=walker.vocab + 8),
        dict(tokens_out=tokens[:-1]),
        dict(temps=temps[:-1]),
        dict(state=walker.states["wgmma"][1:]),
        dict(w2_res=w.w2_res[:-1]),
        dict(
            corrected_out=torch.zeros(
                bs, walker.gamma, walker.vocab - 8, dtype=torch.bfloat16, device=DEV
            )
        ),
        dict(
            base_logits=big,
            anchor=torch.zeros(mw.MAX_BS + 1, dtype=torch.int64, device=DEV),
            temps=torch.ones(mw.MAX_BS + 1, device=DEV),
            tokens_out=torch.empty(
                (mw.MAX_BS + 1) * walker.gamma, dtype=torch.int64, device=DEV
            ),
        ),
    ]
    for over in bad_calls:
        with pytest.raises(RuntimeError):
            wgmma(**over)
    with pytest.raises(ValueError):
        walker.walk(base[:, :-1], anchor, temps, tokens[: bs * (walker.gamma - 1)])
    wgmma()
    assert torch.equal(tokens.view(bs, -1), ref.greedy("wgmma", base, anchor))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
