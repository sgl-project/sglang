import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.sample_step_tokens import (
    SampleStepTokens,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)


@requires_cuda
@pytest.mark.parametrize("bs", [1, 3])
@pytest.mark.parametrize("vocab", [5003, 130000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_matches_torch_with_injected_noise(bs, vocab, dtype):
    torch.manual_seed(0)
    device = torch.device("cuda")
    step_logits = (torch.randn(bs, vocab, device=device) * 4.0).to(dtype)
    temperatures = torch.rand(bs, device=device) + 0.5
    greedy_mask = (torch.arange(bs, device=device) % 2) == 0
    exp_noise = torch.empty(bs, vocab, dtype=torch.float32, device=device).exponential_(
        1
    )
    ref = SampleStepTokens.torch(
        step_logits=step_logits,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
        exp_noise=exp_noise,
    )
    got = SampleStepTokens.triton(
        step_logits=step_logits,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
        exp_noise=exp_noise,
    )
    assert torch.equal(got, ref)


def test_dropping_softmax_normalization_is_argmax_invariant():
    torch.manual_seed(0)
    bs, vocab = 3, 512
    step_logits = torch.randn(bs, vocab) * 5.0
    temperatures = torch.tensor([1.0, 0.7, 1.3])
    s = step_logits / temperatures[:, None]
    exp_noise = torch.empty(bs, vocab).exponential_(1)
    softmax_argmax = (torch.softmax(s, dim=-1) / exp_noise).argmax(dim=-1)
    row_max = s.max(dim=-1, keepdim=True).values
    ratio_argmax = (torch.exp(s - row_max) / exp_noise).argmax(dim=-1)
    assert torch.equal(softmax_argmax, ratio_argmax)


def test_underflow_token_not_selected_in_ratio_space():
    vocab = 8
    step_logits = torch.zeros(1, vocab)
    step_logits[0, 3] = -300.0
    exp_noise = torch.ones(1, vocab)
    exp_noise[0, 3] = 1e-30
    s = step_logits
    row_max = s.max(dim=-1, keepdim=True).values
    ratio_argmax = (torch.exp(s - row_max) / exp_noise).argmax(dim=-1)
    assert ratio_argmax.item() != 3
    assert torch.softmax(s, dim=-1)[0, 3].item() == 0.0


def test_greedy_rows_pick_argmax_logits_regardless_of_noise():
    torch.manual_seed(1)
    bs, vocab = 4, 256
    step_logits = torch.randn(bs, vocab)
    temperatures = torch.rand(bs) + 0.5
    greedy_mask = torch.tensor([True, False, True, False])
    exp_noise = torch.empty(bs, vocab).exponential_(1)
    tokens = SampleStepTokens.torch(
        step_logits=step_logits,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
        exp_noise=exp_noise,
    )
    expected = torch.argmax(step_logits, dim=-1)
    assert torch.equal(tokens[greedy_mask], expected[greedy_mask])


def test_tie_break_picks_smallest_index_on_equal_greedy_logits():
    vocab = 16
    step_logits = torch.zeros(1, vocab)
    step_logits[0, 3] = 5.0
    step_logits[0, 9] = 5.0
    temperatures = torch.tensor([1.0])
    greedy_mask = torch.tensor([True])
    exp_noise = torch.empty(1, vocab).exponential_(1)
    tokens = SampleStepTokens.torch(
        step_logits=step_logits,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
        exp_noise=exp_noise,
    )
    assert tokens.item() == 3


@requires_cuda
def test_triton_tie_break_straddles_block_boundary():
    device = torch.device("cuda")
    vocab = 2050
    step_logits = torch.zeros(1, vocab, device=device)
    step_logits[0, 1000] = 5.0
    step_logits[0, 1100] = 5.0
    temperatures = torch.tensor([1.0], device=device)
    greedy_mask = torch.tensor([True], device=device)
    exp_noise = torch.ones(1, vocab, device=device)
    tokens = SampleStepTokens.triton(
        step_logits=step_logits,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
        exp_noise=exp_noise,
    )
    assert tokens.item() == 1000


@requires_cuda
@pytest.mark.parametrize("padded", [130048, 129536])
def test_triton_reads_strided_cropped_view_without_contiguous(padded):
    torch.manual_seed(3)
    device = torch.device("cuda")
    bs, vocab = 2, 129280
    full = torch.randn(bs, padded, device=device) * 4.0
    view = full[:, :vocab]
    assert view.stride(0) == padded and not view.is_contiguous()
    temperatures = torch.rand(bs, device=device) + 0.5
    greedy_mask = torch.tensor([True, False], device=device)
    exp_noise = torch.empty(bs, vocab, dtype=torch.float32, device=device).exponential_(
        1
    )
    got_view = SampleStepTokens.triton(
        step_logits=view,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
        exp_noise=exp_noise,
    )
    got_contig = SampleStepTokens.triton(
        step_logits=view.contiguous(),
        temperatures=temperatures,
        greedy_mask=greedy_mask,
        exp_noise=exp_noise,
    )
    assert torch.equal(got_view, got_contig)


def test_fresh_noise_drawn_each_call():
    torch.manual_seed(7)
    shape = (2, 128)
    first = torch.empty(shape, dtype=torch.float32).exponential_(1)
    second = torch.empty(shape, dtype=torch.float32).exponential_(1)
    assert not torch.equal(first, second)
