# SPDX-License-Identifier: Apache-2.0
"""One DSpark draft walk (gamma markov steps over [bs, gamma, V] base logits) per
call, three implementations:

* int8_walk: the fused int8 walk -- anchor copy and one cooperative launch;
* default: draft_markov_block keeping the step logits (MarkovSampleStep per step),
  the default draft;
* greedy_draft: draft_markov_block with SGLANG_DSPARK_USE_GREEDY_DRAFT
  (MarkovGreedyStep per step), greedy only.

Every weight, logit and output tensor is an input cloned per CUDA-graph
iteration, so W1 / W2 (77.8 MB each in bf16 at V = 151936, 38.9 MB of int8 W2)
are as cold in L2 as after the target forward. V = 248320 streams more of W2
from L2.
"""

import msgspec
import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.speculative.dspark import markov_walk as mw
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

IMPLS = ["int8_walk", "default", "greedy_draft"]


class _NoTpSync:
    def sync(self, site, values):
        return values


# The weights the kernel reads, cloned per iteration with its state.
_KERNEL_WEIGHTS = ("row_scale", "w2_res", "w2_str", "w1f")


@cache_once
def _setup(vocab: int, num_steps: int):
    from sglang.srt.models.dspark import VanillaMarkov

    gen = torch.Generator(device="cuda").manual_seed(0)
    w2 = 0.05 * torch.randn(vocab, mw.MARKOV_RANK, device="cuda", generator=gen)
    w1 = torch.randn(vocab, mw.MARKOV_RANK, device="cuda", generator=gen)
    head = VanillaMarkov(vocab_size=vocab, markov_rank=mw.MARKOV_RANK)
    head = head.to(device="cuda", dtype=torch.bfloat16).eval()
    with torch.no_grad():
        head.markov_w1.weight.copy_(w1)
        head.markov_w2.weight.copy_(w2)
    walker = mw.MarkovWalker(w1.bfloat16(), w2.bfloat16(), gamma=num_steps)
    walker.warmup()
    return walker, _DefaultWalk(head)


class _DefaultWalk(torch.nn.Module):
    """The default walk as a module: torch.func.functional_call swaps in each
    iteration's cloned W1 / W2."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, base_logits, anchor, temps, noise, logits_out):
        from sglang.srt.speculative.dspark_components.dspark_draft import (
            draft_markov_block,
        )

        return draft_markov_block(
            markov_head=self.head,
            base_logits=base_logits,
            anchor_tokens=anchor,
            draft_hidden=None,
            step_temperatures=temps,
            noise=noise,
            tp_sync=_NoTpSync(),
            logits_out=logits_out,
            greedy_draft=logits_out is None,
        )[0]


def _default_walk(
    default_walk, *, base_logits, anchor, temps, offset, tokens, corrected, w1, w2
):
    from sglang.srt.speculative.dspark_components.dspark_draft import DraftNoise

    params = {"head.markov_w1.weight": w1, "head.markov_w2.weight": w2}
    noise = DraftNoise(key=1, offset=offset)
    toks = torch.func.functional_call(
        default_walk, params, (base_logits, anchor, temps, noise, corrected)
    )
    tokens.copy_(toks.reshape(-1))


def _int8_walk(
    walker,
    *,
    base_logits,
    anchor,
    temps,
    tokens,
    corrected,
    state,
    **weights,
):
    anchors = walker.anchor_buf[: anchor.shape[0]]
    anchors.copy_(anchor)
    # the per-iteration clones of the weights and of the state replace the walker's
    mw.markov_walk(
        weights=msgspec.structs.replace(walker.weights, **weights),
        state=msgspec.structs.replace(walker.state, buffer=state),
        base_logits=base_logits,
        anchor=anchors,
        temps=temps,
        tokens_out=tokens,
        corrected_out=corrected,
        seed=1,
    )


@marker.parametrize(
    "vocab,num_steps,bs",
    [
        (151936, 7, 1),
        (151936, 7, 4),
        (151936, 7, 16),
        (151936, 7, 64),
        (151936, 16, 1),
        (151936, 16, 16),
        (248320, 15, 1),
        (248320, 15, 16),
    ],
    [(151936, 7, 1), (151936, 7, 16)],
)
@marker.parametrize("mode", ["greedy", "t1"])
@marker.benchmark("impl", IMPLS)
def benchmark(vocab: int, num_steps: int, bs: int, mode: str, impl: str):
    if torch.cuda.get_device_capability() != (9, 0):
        marker.skip("the fused markov walk is sm_90 only")
    if mode == "t1" and impl == "greedy_draft":
        marker.skip("greedy-only implementation")
    walker, default_walk = _setup(vocab, num_steps)
    gen = torch.Generator(device="cuda").manual_seed(1)
    inputs = dict(
        base_logits=(
            2 * torch.randn(bs, num_steps, vocab, device="cuda", generator=gen)
        ).bfloat16(),
        anchor=torch.randint(0, vocab, (bs,), device="cuda", generator=gen),
        # The staged step temperatures: 0 on an argmax row.
        temps=torch.full((bs,), float(mode == "t1"), device="cuda"),
        tokens=torch.empty(bs * num_steps, dtype=torch.int64, device="cuda"),
        corrected=(
            None
            if impl == "greedy_draft"
            else torch.zeros(bs, num_steps, vocab, dtype=torch.bfloat16, device="cuda")
        ),
    )
    if impl == "int8_walk":
        weights = msgspec.structs.asdict(walker.weights)
        inputs.update(
            state=walker.state.buffer.clone(),
            **{name: weights[name] for name in _KERNEL_WEIGHTS},
        )
        fn = lambda **kw: _int8_walk(walker, **kw)  # noqa: E731
    else:
        head = default_walk.head
        inputs.update(
            offset=torch.zeros(1, dtype=torch.int64, device="cuda"),
            w1=head.markov_w1.weight.detach(),
            w2=head.markov_w2.weight.detach(),
        )
        fn = lambda **kw: _default_walk(default_walk, **kw)  # noqa: E731
    with torch.no_grad():
        return marker.do_bench(fn, input_kwargs=inputs, disable_log_bandwidth=True)


if __name__ == "__main__":
    benchmark.run()
