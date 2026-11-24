"""Inkling2's ROCm kernels against float64 / torch references, and their run-to-run
determinism (two ROCm Triton miscompilations showed up only as nondeterminism)."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

# backend-specific: the ROCm kernels and dispatch Inkling2 runs on MI35x
register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="Requires a ROCm GPU",
)


@pytest.fixture(autouse=True)
def _no_grad():
    with torch.no_grad():
        yield


def _rel(a, b):
    return (
        (a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)
    ).item()


def _attn_res_reference(sources, addend, queries, normalize):
    values = torch.stack(sources).double()
    if addend is not None:
        values[-1] += addend.double()
    inv = torch.rsqrt(values.square().mean(-1) + 1e-6)
    p = torch.softmax(torch.einsum("ntd,qd,nt->qtn", values, queries.double(), inv), -1)
    if normalize:
        values = values * inv.unsqueeze(-1)
    return torch.einsum("qtn,ntd->qtd", p, values)


@pytest.mark.parametrize("split", [False, 512, 1024])
@pytest.mark.parametrize("tokens", [1, 64, 300])
@pytest.mark.parametrize("n,nq", [(1, 1), (5, 1), (12, 3)])
def test_block_attention_residual(tokens, n, nq, split):
    """One-CTA kernel and the split kernel at both chunk widths (300 rows take the
    one-CTA kernel either way)."""
    from functools import partial

    from sglang.kernels.ops.layernorm import inkling2_norm_hip as norm_hip

    attn_res = partial(norm_hip.attn_res, split=bool(split), split_chunk=split or None)

    torch.manual_seed(n * 7 + nq)
    srcs = [
        torch.randn(tokens, 4608, device="cuda", dtype=torch.bfloat16) * (i + 1)
        for i in range(n)
    ]
    addend = (
        torch.randn(tokens, 4608, device="cuda", dtype=torch.bfloat16)
        if n % 2
        else None
    )
    q = torch.randn(nq, 4608, device="cuda") * 4608**-0.5 * 3
    ref = _attn_res_reference(srcs, addend, q, True)
    first = [y.clone() for y in attn_res(srcs, addend, q, normalize_values=True)]
    for y, r in zip(first, ref):
        assert y.dtype == torch.float32 and _rel(y, r) < 1e-5
    norms = [
        (
            torch.randn(4608, device="cuda", dtype=torch.bfloat16),
            torch.rand(4608, device="cuda", dtype=torch.bfloat16) + 0.5,
            1e-6,
            torch.bfloat16,
        )
        for _ in range(nq)
    ]
    normed = attn_res(srcs, addend, q, normalize_values=True, norms=norms)
    for y, r, (w, pw, eps, _) in zip(normed, ref, norms):
        u = r * pw.double()
        want = u * torch.rsqrt(u.square().mean(-1, keepdim=True) + eps) * w.double()
        assert y.dtype == torch.bfloat16 and _rel(y, want) < 4e-3
    for _ in range(10):
        assert all(
            torch.equal(a, b)
            for a, b in zip(attn_res(srcs, addend, q, normalize_values=True), first)
        )


def test_split_attention_residual_graph_replay_is_deterministic():
    """The split kernel's row counters persist across launches and graph replays. A
    graph mixing rows, sources, chunk widths and one or two barriers per launch
    (without and with the fused norm) replays bitwise equal to its eager run."""
    from sglang.kernels.ops.layernorm import inkling2_norm_hip as norm_hip

    torch.manual_seed(0)
    calls = []
    for i in range(24):
        tokens, n, nq = (1, 2, 8, 16, 64)[i % 5], 1 + i % 10, 3 if i % 4 == 0 else 1
        srcs = [
            torch.randn(tokens, 4608, device="cuda", dtype=torch.bfloat16)
            for _ in range(n)
        ]
        addend = torch.randn(tokens, 4608, device="cuda", dtype=torch.bfloat16)
        q = torch.randn(nq, 4608, device="cuda") * 4608**-0.5
        norms = [
            (
                torch.randn(4608, device="cuda", dtype=torch.bfloat16),
                None,
                1e-6,
                torch.bfloat16,
            )
        ] * nq
        calls.append(
            (srcs, addend, q, norms if i % 2 else None, (512, 1024)[i % 3 == 0])
        )

    def run():
        return [
            torch.stack(
                norm_hip.attn_res(
                    s, a, q, normalize_values=True, norms=nm, split=True, split_chunk=c
                )
            )
            for s, a, q, nm, c in calls
        ]

    eager = [o.clone() for o in run()]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = run()
    for _ in range(20):
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(outs, eager))


@pytest.mark.parametrize("seqs", [1, 32])
@pytest.mark.parametrize("mode", ["decode", "verify"])
@pytest.mark.parametrize("with_partial", [False, True])
def test_conv_epilogue_takes_fp32_expert_sums(seqs, mode, with_partial):
    """ROCm decode and small verify pass the FP32 expert sums to the conv epilogue
    as Hopper does (identity src2dst, top_k 1, rounded to BF16 on load): equal to
    rounding them to BF16 values first and convolving those FP32 rows."""
    from sglang.kernels.ops.mamba.inkling2_sconv import routed_sconv_epilogue

    torch.manual_seed(seqs)
    hidden, width, rows_per_seq = 4608, 4, 1 if mode == "decode" else 8
    slots, tokens = 2 * seqs + 5, seqs * rows_per_seq
    idx = (torch.randperm(slots - 1, device="cuda") + 1)[:seqs]
    meta = {
        "safe_idx": idx,
        "cache_mask": torch.ones(seqs, 1, 1, device="cuda", dtype=torch.bool),
        "cu": torch.arange(0, tokens + 1, rows_per_seq, device="cuda"),
        "si": torch.arange(seqs, device="cuda", dtype=torch.int32).repeat_interleave(
            rows_per_seq
        ),
    }
    sums = torch.randn(tokens, hidden, device="cuda") * 3
    sums[::5] = -0.0
    identity = torch.arange(tokens, device="cuda", dtype=torch.int32)
    weight = torch.randn(hidden, width, device="cuda")
    attention = torch.randn(tokens, hidden, device="cuda").bfloat16()
    partial = torch.randn(tokens, hidden, device="cuda").bfloat16()
    cache = torch.randn(slots, width - 1, hidden, device="cuda").bfloat16()
    window = torch.randn(seqs, 8, width - 1, hidden, device="cuda").bfloat16()
    decode = mode == "decode"

    def run(rows, src2dst, top_k, state):
        return routed_sconv_epilogue(
            rows,
            src2dst,
            top_k,
            weight,
            state[0],
            meta,
            attention,
            partial if with_partial else None,
            window=None if decode else state[1],
            round_bf16=top_k > 0,
        )

    routed, rounded = [cache.clone(), window.clone()], [cache, window]
    actual = run(sums, identity, 1, routed)
    expected = run(sums.bfloat16().float(), None, 0, rounded)
    # Values, not bits: the top-k gather sums from +0, so -0 rows come out +0.
    assert torch.equal(actual, expected)
    for a, b in zip(routed, rounded):
        assert torch.equal(a, b)


def test_rms_norm_and_l2_clip():
    """norm_hip's RMSNorm with a pre-weight and its scaled L2 clip match float64."""
    from sglang.kernels.ops.layernorm import inkling2_norm_hip as norm_hip

    torch.manual_seed(0)
    x = torch.randn(37, 4608, device="cuda", dtype=torch.bfloat16) * 3
    w, pw = (
        torch.randn(4608, device="cuda", dtype=torch.bfloat16),
        torch.rand(4608, device="cuda", dtype=torch.bfloat16) + 0.5,
    )
    u = x.double() * pw.double()
    want = u * torch.rsqrt(u.square().mean(-1, keepdim=True) + 1e-6) * w.double()
    assert (
        _rel(
            norm_hip.rms_norm(x, w, 1e-6, pre_weight=pw, output_dtype=torch.float32),
            want,
        )
        < 1e-6
    )
    xs = (torch.randn(37, 3 * 6 * 128, device="cuda", dtype=torch.bfloat16) * 4)[
        :, : 6 * 128
    ].view(37, 6, 128)
    scale = torch.rand(6, 128, device="cuda") + 0.5
    v = xs.double() * scale.double()
    clip = v * (
        22.627 / (torch.linalg.vector_norm(v, dim=-1, keepdim=True) + 1e-6)
    ).clamp(max=1.0)
    assert _rel(norm_hip.l2_clip(xs, 22.627, 1e-6, scale=scale), clip) < 4e-3


@pytest.mark.parametrize("reqs", [1, 16])
def test_mix_rmsnorm_last_matches_gather_and_copy(reqs):
    """mix_rmsnorm_last equals gathering each request's last committed row,
    mix_rmsnorm and index_copy_ into the f_rnn buffer (Inkling2 MTP's
    store_f_rnn after verify), bitwise."""
    from sglang.kernels.ops.speculative.inkling2_mtp import mix_rmsnorm
    from sglang.kernels.ops.speculative.inkling2_mtp_hip import mix_rmsnorm_last

    torch.manual_seed(reqs)
    width, taps, hidden, slots = 8, 8, 4608, 64
    targets = torch.randn(reqs * width, taps, hidden, device="cuda").bfloat16()
    commit_lens = torch.randint(1, width + 1, (reqs,), device="cuda").int()
    req_slots = torch.randperm(slots, device="cuda")[:reqs]
    alpha = torch.softmax(torch.randn(taps, device="cuda"), 0).bfloat16()
    weight = (torch.rand(hidden, device="cuda") + 0.5).bfloat16()
    buf = torch.randn(slots, hidden, device="cuda").bfloat16()
    expected = buf.clone()
    rows = torch.arange(reqs, device="cuda")
    last = targets.view(reqs, width, taps, hidden)[rows, commit_lens.long() - 1]
    expected.index_copy_(0, req_slots, mix_rmsnorm(last, alpha, weight, 1e-6))
    mix_rmsnorm_last(targets, width, commit_lens, req_slots, alpha, weight, 1e-6, buf)
    assert torch.equal(buf.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("rows", [1, 4])
def test_rnn_step_matches_gemm_path(rows):
    """rnn_step (Inkling2 MTP's Markov RNN step for a few rows, in two launches) equals
    rnn_inputs, the rnn_proj GEMM and rnn_gates up to one BF16 step where the split-K
    sum rounds differently, and repeats bitwise."""
    import torch.nn.functional as F

    from sglang.kernels.ops.speculative.inkling2_mtp import rnn_gates, rnn_inputs
    from sglang.kernels.ops.speculative.inkling2_mtp_hip import rnn_step, rnn_step_fits

    torch.manual_seed(rows)
    rank, hidden, vocab = 256, 4608, 4096
    embedding = (torch.randn(vocab, rank, device="cuda") * 0.05).bfloat16()
    weight = (
        torch.randn(3 * rank, 2 * rank + 2 * hidden, device="cuda") * 0.01
    ).bfloat16()
    bias = (torch.randn(3 * rank, device="cuda") * 0.1).bfloat16()
    state = (torch.randn(rows, rank, device="cuda") * 0.5).bfloat16()
    tokens = torch.randint(0, vocab, (rows,), device="cuda")
    hidden_states = torch.randn(rows, 2 * hidden, device="cuda").bfloat16()[:, :hidden]
    f_rnn = torch.randn(rows, hidden, device="cuda").bfloat16()
    assert rnn_step_fits(state, hidden_states, weight, bias)
    inputs = rnn_inputs(state, tokens, embedding, hidden_states, f_rnn)
    ref_state, ref_latent = rnn_gates(F.linear(inputs, weight, bias), state)
    new_state, latent = rnn_step(
        state, tokens, embedding, hidden_states, f_rnn, weight, bias
    )
    for out, ref in ((new_state, ref_state), (latent, ref_latent)):
        assert out.dtype == ref.dtype and out.shape == ref.shape
        # A projection one BF16 step off moves the gates by a few BF16 steps.
        assert (out.float() - ref.float()).abs().max() <= 0.02
        assert (out != ref).float().mean() < 0.01
    again = rnn_step(state, tokens, embedding, hidden_states, f_rnn, weight, bias)
    assert torch.equal(again[0], new_state) and torch.equal(again[1], latent)


@pytest.mark.parametrize("rows", [1, 4])
def test_markov_step_fold_matches_addmm_bias(rows):
    """A sharded DSpark Markov step with the folded bias computed in its sampling kernel
    (fold) picks the tokens of the step given torch.addmm's bias, and keeps the same
    step logits up to rare one-step BF16 roundings of the bias; it repeats bitwise."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep

    torch.manual_seed(rows)
    padded, rank = 25128, 256
    base = torch.randn(rows, padded, device="cuda").bfloat16() * 4
    latents = torch.tanh(torch.randn(rows, rank, device="cuda")).bfloat16()
    weight = (torch.randn(padded, rank, device="cuda") * 0.1).bfloat16()
    fold_bias = torch.randn(padded, device="cuda").bfloat16()
    temperatures = torch.tensor([0.0, 0.7, 1.3, 0.0][:rows], device="cuda")
    common = dict(
        base_logits=base, temperatures=temperatures, key=99,
        offset=torch.tensor([3], device="cuda", dtype=torch.int64), step=1,
        step_dtype=torch.bfloat16, vocab_start=2 * padded, gather=lambda p: p,
        softcap=30.0, vocab_limit=3 * padded - 7,
    )  # fmt: skip
    expected_logits = torch.empty_like(base)
    expected = MarkovSampleStep.triton(
        bias=torch.addmm(fold_bias, latents, weight.T), logits_out=expected_logits,
        **common,
    )  # fmt: skip
    outs = []
    for _ in range(2):
        logits = torch.empty_like(base)
        tokens = MarkovSampleStep.triton(
            bias=latents, fold=(weight, fold_bias), logits_out=logits, **common
        )
        outs.append((tokens, logits))
    tokens, logits = outs[0]
    assert torch.equal(tokens, expected)
    assert torch.equal(outs[1][0], tokens) and torch.equal(outs[1][1], logits)
    finite = expected_logits.isfinite()
    assert torch.equal(finite, logits.isfinite())
    assert (logits != expected_logits).float().mean() < 0.01
    diff = (logits.float() - expected_logits.float())[finite].abs().max()
    assert diff <= 0.25  # one BF16 step at |logit| < 30


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
