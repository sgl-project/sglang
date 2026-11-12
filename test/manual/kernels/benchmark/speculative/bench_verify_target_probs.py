"""Manual benchmark: unfused verify target-prob chain vs fused ``verify_target_probs``.

The baseline is the sequence ``eagle_sample`` runs today (in-place penalties on
the logits, then ``sampling_softmax``), without the oracle's defensive clone.
Times both inside a CUDA graph (``do_bench_cudagraph``) so launch gaps are not
hidden by eager overhead, plus an eager run that includes CPU launch cost.

    python test/manual/kernels/benchmark/speculative/bench_verify_target_probs.py
"""

import itertools

import torch
import triton

from sglang.kernels.ops.sampling import softmax as sampling_softmax
from sglang.kernels.ops.speculative.verify_target_probs import verify_target_probs
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties

VOCAB = 128256
CASES = {
    "temp_only": dict(),
    "freq_pres": dict(additive=True),
    "all": dict(additive=True, scaling=True, bias=True),
}


def _inputs(bs, k, additive=False, scaling=False, bias=False):
    dev = "cuda"
    kw = dict(rows_per_request=k, temperatures=torch.full((bs, 1), 0.7, device=dev))
    if additive:
        kw["additive_penalties"] = -torch.rand(bs, VOCAB, device=dev)
    if scaling:
        kw["scaling_penalties"] = 1 + torch.rand(bs, VOCAB, device=dev)
    if bias:
        kw["logit_bias"] = torch.randn(bs, VOCAB, device=dev)
    return torch.randn(bs * k, VOCAB, device=dev) * 4, kw


def _unfused(logits, *, rows_per_request, temperatures, additive_penalties=None,
             scaling_penalties=None, logit_bias=None):  # fmt: skip
    k = rows_per_request
    # Values drift across repeats because this is in place, as in production;
    # only the timing matters here.
    if additive_penalties is not None:
        logits.add_(torch.repeat_interleave(additive_penalties, k, dim=0))
    if scaling_penalties is not None:
        apply_scaling_penalties(
            logits, torch.repeat_interleave(scaling_penalties, k, dim=0)
        )
    if logit_bias is not None:
        logits.add_(torch.repeat_interleave(logit_bias, k, dim=0))
    return sampling_softmax(logits, temperatures=temperatures, rows_per_request=k)


def _time(fn):
    graph_us = triton.testing.do_bench_cudagraph(fn) * 1e3
    eager_us = triton.testing.do_bench(fn) * 1e3
    return graph_us, eager_us


def main():
    print(
        f"{'case':<10} {'bs':>3} {'K':>2} | {'base graph':>10} {'base eager':>10} "
        f"| {'fused graph':>11} {'fused eager':>11} | {'speedup':>7}"
    )
    for (name, feats), bs, k in itertools.product(CASES.items(), [1, 4, 16, 64], [4]):
        logits, kw = _inputs(bs, k, **feats)
        base = _time(lambda: _unfused(logits, **kw))
        logits, kw = _inputs(bs, k, **feats)
        fused = _time(lambda: verify_target_probs(logits, **kw))
        print(
            f"{name:<10} {bs:>3} {k:>2} | {base[0]:>10.1f} {base[1]:>10.1f} "
            f"| {fused[0]:>11.1f} {fused[1]:>11.1f} | {base[0] / fused[0]:>6.1f}x"
        )


if __name__ == "__main__":
    main()
