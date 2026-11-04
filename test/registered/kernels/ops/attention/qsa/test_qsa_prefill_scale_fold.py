"""The relative prefill scorer with its 1/sqrt(128) folded into the store is
bit-identical to the scorer followed by torch's divide.

Each case poisons the allocator block the logits reuse with NaN, compares each
row's own section bit for bit, records which kernel ran, and runs a seeded
positive control (one perturbed key) that the comparison must catch.
"""

import math

import pytest
import torch

from sglang.srt.layers.attention.qsa import mqa
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
        reason="needs an SM100+ GPU",
    ),
]
needs_tilelang = pytest.mark.skipif(not mqa.HAS_TILELANG, reason="needs TileLang")


def test_cuda_scalar_divide_is_the_folded_multiply_for_every_fp32():
    """Exhaustive over all 2^32 FP32 inputs: x.div_(sqrt(128)) on CUDA equals the
    IEEE product with the store's constant (NaN inputs only need to stay NaN)."""
    inv = torch.tensor([mqa._QSA_INV_SQRT128_BITS], dtype=torch.int32).view(torch.float32)
    inv = inv.cuda()
    chunk = 1 << 28
    for start in range(-(1 << 31), 1 << 31, chunk):
        bits = torch.arange(start, start + chunk, dtype=torch.int32, device="cuda")
        x = bits.view(torch.float32)
        divided = x.clone().div_(math.sqrt(128))
        folded = x * inv
        same = divided.view(torch.int32) == folded.view(torch.int32)
        same |= torch.isnan(divided) & torch.isnan(folded)
        assert bool(same.all()), f"mismatch in bit patterns [{start}, {start + chunk})"
    # Positive control: true division by sqrt(128) differs from the fold somewhere.
    x = torch.randn(1 << 20, device="cuda")
    exact = (x.double() / math.sqrt(128)).float()
    assert not torch.equal(exact.view(torch.int32), (x * inv).view(torch.int32))


def _prefill_batch(seed, requests=6, max_keys=6000, heads=4, head_dim=128):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    lengths = torch.randint(1, max_keys, (requests,), generator=gen)
    starts = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), lengths[:-1]]), 0)
    # Each request contributes a few query rows that see a prefix of its keys.
    rows_per_request = torch.randint(1, 40, (requests,), generator=gen)
    row_starts, row_ends = [], []
    for start, length, rows in zip(starts, lengths, rows_per_request):
        ends = torch.randint(1, int(length) + 1, (int(rows),), generator=gen).sort().values
        row_starts += [int(start)] * int(rows)
        row_ends += (int(start) + ends).tolist()
    q = torch.randn(len(row_starts), heads, head_dim, generator=gen).to(torch.bfloat16)
    k = torch.randn(int(lengths.sum()), 1, head_dim, generator=gen).to(torch.bfloat16)
    row_starts = torch.tensor(row_starts, dtype=torch.int32)
    row_ends = torch.tensor(row_ends, dtype=torch.int32)
    width = int((row_ends - row_starts).max())
    return q.cuda(), k.cuda(), row_starts.cuda(), row_ends.cuda(), width


def _poison_next(rows, width):
    # The scorer allocates with torch.empty; leave NaN in the block it will reuse.
    padded = rows + (-rows) % 32
    poison = torch.full((padded, width), float("nan"), device="cuda")
    del poison


def _score(inputs, fold, monkeypatch):
    q, k, row_starts, row_ends, width = inputs
    monkeypatch.setattr(mqa, "_QSA_PREFILL_SCALE_FOLD", fold)
    _poison_next(q.shape[0], width)
    out = mqa.tilelang_qsa_mqa_prefill_relative(q, k, row_starts, row_ends, width)
    torch.cuda.synchronize()
    return out


def _sections(logits, row_starts, row_ends):
    lengths = (row_ends - row_starts).long()
    mask = torch.arange(logits.shape[1], device=logits.device)[None, :] < lengths[:, None]
    return logits[mask]


@needs_tilelang
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_folded_store_matches_divided_logits(seed, monkeypatch, record_property):
    inputs = _prefill_batch(seed)
    built = []
    for name in (
        "_tilelang_qsa_mqa_prefill_relative_kernel",
        "_tilelang_qsa_mqa_prefill_relative_scaled_kernel",
    ):
        factory = getattr(mqa, name)
        monkeypatch.setattr(
            mqa, name, lambda *a, _n=name, _f=factory, **kw: built.append(_n) or _f(*a, **kw)
        )
    divided = _sections(_score(inputs, False, monkeypatch), *inputs[2:4])
    folded = _sections(_score(inputs, True, monkeypatch), *inputs[2:4])

    record_property("kernels", built)
    assert built == [
        "_tilelang_qsa_mqa_prefill_relative_kernel",
        "_tilelang_qsa_mqa_prefill_relative_scaled_kernel",
    ]
    assert not torch.isnan(folded).any(), "a scored logit was never written"
    assert torch.equal(folded.view(torch.int32), divided.view(torch.int32))

    q, k, row_starts, row_ends, width = inputs
    perturbed = k.clone()
    perturbed[int(row_starts[0])] = q[0, 0]
    control = _sections(
        _score((q, perturbed, row_starts, row_ends, width), True, monkeypatch),
        row_starts,
        row_ends,
    )
    assert not torch.equal(control.view(torch.int32), folded.view(torch.int32))
