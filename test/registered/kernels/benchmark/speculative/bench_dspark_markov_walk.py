# SPDX-License-Identifier: Apache-2.0
"""One DSpark draft walk (gamma markov steps over [bs, gamma, V] base logits) per
call, four implementations:

* int8_walk: the fused int8 walk -- in-graph temps staging, anchor copy and one
  cooperative launch, as MarkovWalker.walk does;
* default_folded: VanillaMarkov.sample_block with the folded draft sampler
  (exponential noise + SampleStepTokens);
* default_greedy_only: VanillaMarkov.sample_block with the greedy argmax sampler;
* markov_greedy_step: VanillaMarkov.sample_block_greedy_fused (MarkovGreedyStep
  per step), greedy only.

Every weight, logit and output tensor is an input cloned per CUDA-graph
iteration, so W1 / W2 (77.8 MB each in bf16 at V = 151936, 38.9 MB of int8 W2)
are as cold in L2 as after the target forward.
"""

import msgspec
import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.speculative.dspark import markov_walk as mw
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import SampleStepTokens
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

IMPLS = ["int8_walk", "default_folded", "default_greedy_only", "markov_greedy_step"]
GREEDY_ONLY = ("default_greedy_only", "markov_greedy_step")


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

    def forward(self, base_logits, anchor, sampler, fused_greedy):
        if fused_greedy:
            return self.head.sample_block_greedy_fused(
                base_logits, first_prev_tokens=anchor
            )
        return self.head.sample_block(
            base_logits,
            first_prev_tokens=anchor,
            hidden_states=None,
            sampler=sampler,
            collect_corrected=sampler is not _greedy_sampler,
        )


def _greedy_sampler(step_logits, step_idx):
    return torch.argmax(step_logits, dim=-1)


def _default_walk(
    impl,
    default_walk,
    *,
    base_logits,
    anchor,
    greedy_mask,
    temperatures,
    exp_noise,
    tokens,
    corrected,
    w1,
    w2,
):
    def folded_sampler(step_logits, step_idx):
        return SampleStepTokens.execute(
            step_logits=step_logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise.exponential_(),
        )

    sampler = folded_sampler if impl == "default_folded" else _greedy_sampler
    params = {"head.markov_w1.weight": w1, "head.markov_w2.weight": w2}
    out = torch.func.functional_call(
        default_walk,
        params,
        (base_logits, anchor, sampler, impl == "markov_greedy_step"),
    )
    toks, corr = out if isinstance(out, tuple) else (out, None)
    tokens.copy_(toks.reshape(-1))
    if corr is not None:
        corrected.copy_(corr.reshape(corrected.shape))


def _int8_walk(
    walker,
    *,
    base_logits,
    anchor,
    greedy_mask,
    temperatures,
    zero,
    tokens,
    corrected,
    state,
    **weights,
):
    bs, steps, vocab = base_logits.shape
    temps = walker.temps_buf[:bs]
    torch.where(greedy_mask, zero, temperatures, out=temps)
    walker.anchor_buf[:bs].copy_(anchor)
    # the per-iteration clones of the weights and of this bs's state replace the walker's
    mw.markov_walk(
        weights=msgspec.structs.replace(walker.weights, **weights),
        states=msgspec.structs.replace(walker.states, **{walker.kernel_for(bs): state}),
        base_logits=base_logits,
        anchor=walker.anchor_buf[:bs],
        temps=temps,
        tokens_out=tokens,
        corrected_out=corrected.view(bs, steps, vocab),
        seed=1,
    )


def _int8_walk_weights(walker, bs):
    w = walker.weights
    if walker.kernel_for(bs) == "wgmma":
        return dict(row_scale=w.row_scale, w2_res=w.w2_res, w2_str=w.w2_str, w1f=w.w1f)
    return dict(row_scale=w.row_scale, frag=w.frag, w1q=w.w1q)


@marker.parametrize(
    "vocab,num_steps", [(151936, 7), (151936, 16), (248320, 15)], [(151936, 7)]
)
@marker.parametrize("bs", [1, 2, 4, 8, 16, 32, 64], [1, 64])
@marker.parametrize("mode", ["greedy", "t1"])
@marker.benchmark("impl", IMPLS)
def benchmark(vocab: int, num_steps: int, bs: int, mode: str, impl: str):
    if torch.cuda.get_device_capability() != (9, 0):
        marker.skip("the fused markov walk is sm_90 only")
    if mode == "t1" and impl in GREEDY_ONLY:
        marker.skip("greedy-only implementation")
    walker, default_walk = _setup(vocab, num_steps)
    gen = torch.Generator(device="cuda").manual_seed(bs)
    inputs = dict(
        base_logits=(
            2 * torch.randn(bs, num_steps, vocab, device="cuda", generator=gen)
        ).bfloat16(),
        anchor=torch.randint(0, vocab, (bs,), device="cuda", generator=gen),
        greedy_mask=torch.full((bs,), mode == "greedy", device="cuda"),
        temperatures=torch.ones(bs, device="cuda"),
        tokens=torch.empty(bs * num_steps, dtype=torch.int64, device="cuda"),
        corrected=torch.zeros(
            bs * num_steps, vocab, dtype=torch.bfloat16, device="cuda"
        ),
    )
    if impl == "int8_walk":
        inputs.update(
            zero=torch.zeros((), device="cuda"),
            state=walker.states.of(walker.kernel_for(bs)).clone(),
            **_int8_walk_weights(walker, bs),
        )
        fn = lambda **kw: _int8_walk(walker, **kw)  # noqa: E731
    else:
        head = default_walk.head
        inputs.update(
            exp_noise=torch.empty(bs, vocab, device="cuda"),
            w1=head.markov_w1.weight.detach(),
            w2=head.markov_w2.weight.detach(),
        )
        fn = lambda **kw: _default_walk(impl, default_walk, **kw)  # noqa: E731
    with torch.no_grad():
        return marker.do_bench(fn, input_kwargs=inputs, disable_log_bandwidth=True)


if __name__ == "__main__":
    benchmark.run()
