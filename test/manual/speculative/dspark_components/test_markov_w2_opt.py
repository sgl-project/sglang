import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    pad_vocab_size,
    vocab_range_from_global_vocab_size,
)


def _lm_head_partition(vocab: int, tp_size: int) -> tuple[int, int]:
    padding_size = DEFAULT_VOCAB_PADDING_SIZE
    if pad_vocab_size(vocab, padding_size) % tp_size != 0:
        padding_size *= tp_size
    num_padded = pad_vocab_size(vocab, padding_size)
    return num_padded // tp_size, num_padded


@pytest.mark.parametrize("bs", [1, 4])
@pytest.mark.parametrize("vocab", [5003, 4096])
def test_bf16_project_bias_stays_close_to_fp32_and_is_fp32(bs: int, vocab: int) -> None:
    torch.manual_seed(0)
    rank = 512
    latent = torch.randn(bs, rank, dtype=torch.bfloat16)
    weight_bf16 = torch.randn(vocab, rank, dtype=torch.bfloat16)
    weight_fp32 = weight_bf16.float()

    fp32_bias = F.linear(latent.float(), weight_fp32)
    bf16_bias = F.linear(latent.to(weight_bf16.dtype), weight_bf16).float()

    assert bf16_bias.dtype == torch.float32
    assert torch.allclose(bf16_bias, fp32_bias, rtol=3e-2, atol=5e-1)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("vocab", [5003, 4096])
def test_sharded_corrected_logits_equal_full_vocab(tp_size: int, vocab: int) -> None:
    torch.manual_seed(0)
    bs, rank = 3, 128
    per_partition, num_padded = _lm_head_partition(vocab, tp_size)

    latent = torch.randn(bs, rank)
    weight_full = torch.randn(vocab, rank)
    base_full_padded = torch.randn(bs, num_padded)

    full_bias = F.linear(latent, weight_full)
    reference = base_full_padded[:, :vocab] + full_bias

    per_rank_local = []
    for rank_id in range(tp_size):
        padded_start, padded_end = vocab_range_from_global_vocab_size(
            num_padded, rank_id, tp_size
        )
        org_start = min(padded_start, vocab)
        org_end = min(padded_end, vocab)
        base_local = base_full_padded[:, padded_start:padded_end]
        weight_local = weight_full[org_start:org_end]
        bias_local = F.linear(latent, weight_local)
        pad = per_partition - bias_local.shape[-1]
        if pad > 0:
            bias_local = F.pad(bias_local, (0, pad))
        per_rank_local.append(base_local + bias_local)

    gathered = torch.cat(per_rank_local, dim=-1)
    got = gathered[:, :vocab]

    assert got.shape == reference.shape
    assert torch.allclose(got, reference, rtol=0, atol=1e-6)


@pytest.mark.parametrize("tp_size", [2, 4])
def test_sharded_argmax_matches_full_vocab_argmax(tp_size: int) -> None:
    torch.manual_seed(1)
    bs, rank, vocab = 5, 64, 4096
    per_partition, num_padded = _lm_head_partition(vocab, tp_size)

    latent = torch.randn(bs, rank)
    weight_full = torch.randn(vocab, rank)
    base_full_padded = torch.randn(bs, num_padded)
    reference = base_full_padded[:, :vocab] + F.linear(latent, weight_full)

    per_rank_local = []
    for rank_id in range(tp_size):
        padded_start, padded_end = vocab_range_from_global_vocab_size(
            num_padded, rank_id, tp_size
        )
        org_end = min(padded_end, vocab)
        org_start = min(padded_start, vocab)
        base_local = base_full_padded[:, padded_start:padded_end]
        bias_local = F.linear(latent, weight_full[org_start:org_end])
        pad = per_partition - bias_local.shape[-1]
        if pad > 0:
            bias_local = F.pad(bias_local, (0, pad))
        per_rank_local.append(base_local + bias_local)
    got = torch.cat(per_rank_local, dim=-1)[:, :vocab]

    assert torch.equal(got.argmax(dim=-1), reference.argmax(dim=-1))
