import sys

import pytest
import torch

from sglang.kernels.ops.speculative.domino_corrector import (
    corrector_context_attention,
    corrector_draw,
    corrector_history_attention,
)
from sglang.srt.speculative.domino_corrector import (
    CorrectorSampling,
    _context_attention,
    _draw,
    _history_attention,
    _rope_tables,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

HEADS, HEAD_DIM = 4, 64
WIDTH = HEADS * HEAD_DIM


def _assert_matches(out, ref):
    # The kernels accumulate in FP32 and round once; the reference gets the same
    # values widened, so BF16 differs by the final rounding only.
    rtol = 4e-3 if out.dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(out.float().cpu(), ref, rtol=rtol, atol=1e-5)


def _randn(shape, dtype, gen):
    return torch.randn(shape, device="cuda", dtype=dtype, generator=gen)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bs,slots", [(1, 7), (3, 7), (33, 4), (0, 7)])
def test_history_attention_matches_reference(dtype, bs, slots):
    gen = torch.Generator(device="cuda").manual_seed(bs * 10 + slots)
    # Unwritten slots must never be read.
    history = torch.full(
        (bs, HEADS, slots, 2 * HEAD_DIM), float("nan"), device="cuda", dtype=dtype
    )
    ref_history = history.float().cpu()
    for slot in range(slots):
        qkv = _randn((bs, 2, 3 * WIDTH), dtype, gen)
        qkv[..., :WIDTH] *= HEAD_DIM**-0.5
        out = corrector_history_attention(qkv, history, slot, heads=HEADS)
        ref = _history_attention(qkv.float().cpu(), ref_history, slot, HEADS)
        _assert_matches(out, ref)
        torch.testing.assert_close(
            history[:, :, : slot + 1].float().cpu(),
            ref_history[:, :, : slot + 1],
            rtol=0,
            atol=0,
        )
        assert history[:, :, slot + 1 :].isnan().all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "bs,window", [(1, 256), (4, 256), (4, 40), (2, 600), (33, 5), (0, 16)]
)
def test_context_attention_matches_reference(dtype, bs, window):
    layers = 2
    gen = torch.Generator(device="cuda").manual_seed(bs * 1000 + window)
    context = _randn((bs, window, layers, 2 * WIDTH), dtype, gen)
    offsets = torch.arange(-window, 0, device="cuda")
    cos, sin = _rope_tables(offsets, HEAD_DIM, torch.float32)
    cos = cos[:, : HEAD_DIM // 2].contiguous()
    sin = sin[:, : HEAD_DIM // 2].contiguous()
    # A request has the newest rows of its window: all, some, or none of them.
    held = torch.arange(bs, device="cuda") * 7 % (window + 1)
    if bs > 1:
        held[0], held[1] = window, 0
    valid = (torch.arange(window, device="cuda") >= window - held[:, None]).to(
        torch.uint8
    )
    for layer in range(layers):
        query = _randn((bs, 2, WIDTH), dtype, gen) * HEAD_DIM**-0.5
        out = corrector_context_attention(
            query, context, valid, cos, sin, layer=layer, heads=HEADS
        )
        ref = _context_attention(
            query.float().cpu(),
            context.float().cpu(),
            valid.cpu(),
            cos.cpu(),
            sin.cpu(),
            layer=layer,
            heads=HEADS,
        )
        _assert_matches(out, ref)
        if bs > 1:
            assert out[1].count_nonzero() == 0


@pytest.mark.parametrize("bs,pool", [(1, 2048), (5, 2048), (33, 100), (4, 1), (0, 64)])
def test_draw_matches_reference(bs, pool):
    gen = torch.Generator(device="cuda").manual_seed(bs * 10000 + pool)
    scores = 3 * _randn((bs, pool), torch.float32, gen)
    candidate_ids = torch.randint(0, 163840, (bs, pool), device="cuda", generator=gen)
    temperatures = 0.3 + torch.rand(bs, device="cuda", generator=gen)
    greedy_mask = torch.arange(bs, device="cuda") % 3 == 1
    # A greedy row may carry a temperature of 0, clamped just above it.
    temperatures[greedy_mask] = 1e-5
    torch.manual_seed(bs + pool)
    sampling = CorrectorSampling(
        temperatures=temperatures.cpu(),
        greedy_mask=greedy_mask.cpu(),
        noise=torch.empty(bs, pool),
    )
    ref_tokens, ref_q = _draw(scores.cpu(), candidate_ids.cpu(), sampling)
    # The reference drew the noise into its scratch; the kernel takes the same draws.
    tokens, q = corrector_draw(
        scores,
        sampling.noise.cuda(),
        temperatures,
        greedy_mask.view(torch.uint8),
        candidate_ids,
    )
    assert torch.equal(tokens.cpu(), ref_tokens)
    torch.testing.assert_close(q.cpu(), ref_q, rtol=1e-4, atol=1e-7)
    greedy_q = q[greedy_mask]
    assert torch.equal(greedy_q.sum(dim=-1), torch.ones_like(greedy_q[:, 0]))
    assert torch.equal(
        candidate_ids[greedy_mask].gather(1, greedy_q.argmax(dim=-1, keepdim=True)),
        tokens[greedy_mask, None],
    )


def test_draw_survives_a_zero_noise_draw():
    scores = torch.zeros((1, 8), device="cuda")
    noise = torch.ones((1, 8), device="cuda")
    noise[0, 5] = 0.0
    tokens, q = corrector_draw(
        scores,
        noise,
        torch.ones(1, device="cuda"),
        torch.zeros(1, dtype=torch.uint8, device="cuda"),
        torch.arange(8, device="cuda")[None] + 100,
    )
    assert tokens.item() == 105
    torch.testing.assert_close(q, torch.full_like(q, 1 / 8))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
