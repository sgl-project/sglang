"""Check ratio-2 pooling within two bf16 ulps and the pair state bitwise."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.c2 import (
    c2_decode_norm,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
    DeepseekV41Compressor,
    RMSNorm,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or (is_hip() and not is_gfx95_supported()),
    reason="requires a GPU; on ROCm the byte-for-byte e4m3fn cache oracle needs gfx950"
    " (gfx942's hardware fp8 pack encodes E4M3FNUZ)",
)

EPS = 1e-6
# The 584-byte FlashMLA layout fixes head_dim at 512:
# 448 fp8 nope values plus 64 bf16 RoPE values.
HEAD_DIM = 512
BATCHES = (1, 64)
# `CompressStatePool.ring_size`; two values because a wrong modular wrap shows on only
# one of them (served: `next_power_of_2(draft + 1)`)
RING_SIZES = (2, 8)

# Two bf16 ulp: the intermediate bf16 cast in `finish` makes one ulp reachable and two
# the ceiling. The floor only engages on elements the pair pooling cancelled to near zero.
POOL_RTOL = 2**-6
POOL_ATOL = 2**-20


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Explicit torch RMSNorm: `RMSNorm.forward` may switch kernels with the batch size."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (weight * x).to(dtype)


def _norm(dim: int, seed: int) -> RMSNorm:
    """`DeepseekV41Compressor.norm` as the model holds it: bf16 weight, because the
    parameter is created inside `set_default_torch_dtype(model dtype)`."""
    with set_default_torch_dtype(torch.bfloat16):
        norm = RMSNorm(dim, EPS).cuda()
    assert norm.weight.dtype == torch.bfloat16
    # Not `ones`: a constant weight cannot catch a wrong per-element index.
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(dim, generator=g, device="cuda"))
    return norm


def _inputs(
    n,
    dim,
    seed,
    *,
    positions=None,
    req=None,
    raw_out_loc=None,
    num_state_rows=None,
    ring_size=RING_SIZES[-1],
):
    """Distinct partner rows, alternating parity, no padding; `test_raw_out_loc_int64`
    covers the int64 locations."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    num_state_rows = num_state_rows if num_state_rows is not None else n + 2
    kv_input = torch.randn(n, 2 * dim, generator=g, device="cuda", dtype=torch.float32)
    kv_state = torch.randn(
        num_state_rows * ring_size,
        2 * dim,
        generator=g,
        device="cuda",
        dtype=torch.float32,
    )
    if positions is None:
        positions = torch.arange(1, n + 1, device="cuda", dtype=torch.int32)
    if req is None:
        req = torch.arange(1, n + 1, device="cuda", dtype=torch.int64)
    if raw_out_loc is None:
        # Odd and never 0, so no row is taken for padding and `>> 1` gives each
        # row its own compressed slot.
        raw_out_loc = torch.arange(n, device="cuda", dtype=torch.int32) * 2 + 3
    return kv_input, kv_state, positions, req, raw_out_loc


def _state_rows(req, positions, ring_size):
    """`CompressStatePool.translate_from_req_position_to_state_loc`, for the slot a row
    reads (`pos - 1`) and the one it writes (`pos`)."""
    base = req.to(torch.int64) * ring_size
    pos = positions.to(torch.int64)
    return base + (pos - 1) % ring_size, base + pos % ring_size


def _torch_reference(kv_input, kv_state, norm, positions, req, raw_out_loc, ring_size):
    """The ratio-2 decode branch and `finish`, transcribed."""
    dim = kv_input.shape[1] // 2
    kv, score = kv_input[:, :dim], kv_input[:, dim:]
    state_kv, state_score = kv_state[:, :dim], kv_state[:, dim:]
    odd = (positions.to(torch.int64) % 2) == 1
    read, write = _state_rows(req, positions, ring_size)
    # a completing row reads what `pos - 1` left and a pending one writes its own
    # slot, so there is no read-modify-write between them
    partner_kv, partner_score = state_kv[read], state_score[read]
    # only an even live row writes: padded rows alias a live `req_pool_idx`, and a
    # duplicated scatter index picks a winner instead of skipping
    parks = (~odd) & (raw_out_loc != 0)
    w = write[parks]
    assert w.unique().numel() == w.numel(), "a decode step parks once per request"
    state_kv[w] = kv[parks]
    state_score[w] = score[parks]
    pooled = DeepseekV41Compressor.pool_pairs(
        torch.stack([partner_kv, kv], dim=1),
        torch.stack([partner_score, score], dim=1),
    )
    return _torch_rmsnorm(pooled.to(torch.bfloat16), norm.weight.data, EPS), odd


def _compare(got, expected, rows, ctx):
    got, expected = got.float()[rows], expected.float()[rows]
    if got.numel() == 0:
        return
    diff = (got - expected).abs()
    worst = (diff / (POOL_ATOL + POOL_RTOL * expected.abs())).max().item()
    assert worst <= 1.0, (
        f"{ctx}: out of tolerance at {worst:.3f}x the bound "
        f"(rtol={POOL_RTOL}, atol={POOL_ATOL}); "
        f"max|diff| {diff.max().item():.3e} on |ref| up to "
        f"{expected.abs().max().item():.3e}"
    )


def _run(n, dim, seed, *, ring_size=RING_SIZES[-1], **kw):
    """One `c2_decode_norm` call against the reference, returning the two pair states so
    callers can add their own assertions."""
    kv_input, kv_state, positions, req, raw_out_loc = _inputs(
        n, dim, seed, ring_size=ring_size, **kw
    )
    norm = _norm(dim, seed + 1)
    ref_state, got_state = kv_state.clone(), kv_state.clone()

    expected, odd = _torch_reference(
        kv_input, ref_state, norm, positions, req, raw_out_loc, ring_size
    )
    got = c2_decode_norm(
        kv_input,
        got_state,
        norm.weight.data,
        positions,
        req,
        raw_out_loc,
        EPS,
        ring_size=ring_size,
    )
    # A padded row still computes and still publishes its latent -- the caller
    # discards that row -- so the tolerance gate covers the live ones.
    live = odd & (raw_out_loc != 0)
    _compare(got, expected, live, f"{n=} {dim=} {seed=}")
    return got, expected, odd, got_state, ref_state


# ---------------------------------------------------------------- pool + norm


@pytest.mark.parametrize("ring_size", RING_SIZES)
@pytest.mark.parametrize("n", BATCHES)
def test_mixed_parity(n, ring_size):
    """Odd rows complete a group against the state while even rows park in it; both ring
    sizes, since a wrong modular wrap shows on only one."""
    *_, got_state, ref_state = _run(n, HEAD_DIM, seed=1000 + n, ring_size=ring_size)
    # Pure copy on the even rows, untouched on the odd ones -- no arithmetic, so
    # nothing here is allowed to differ by even one bit.
    assert torch.equal(got_state, ref_state), "pair state diverged"


def test_pair_state_carried_across_two_steps():
    """A group spans two decode steps: an all-even step only parks, and the all-odd step
    after it must pool against exactly what it left."""
    n = 8
    g = torch.Generator(device="cuda").manual_seed(3000)
    first = torch.randn(
        n, 2 * HEAD_DIM, generator=g, device="cuda", dtype=torch.float32
    )
    second = torch.randn(
        n, 2 * HEAD_DIM, generator=g, device="cuda", dtype=torch.float32
    )
    kv_state = torch.randn(
        n * RING_SIZES[-1],
        2 * HEAD_DIM,
        generator=g,
        device="cuda",
        dtype=torch.float32,
    )
    # Rotated by one so the carry cannot be confused with `req == row`.
    req = ((torch.arange(n, device="cuda") + 1) % n).to(torch.int64)
    raw_out_loc = torch.arange(n, device="cuda", dtype=torch.int32) * 2 + 3
    even = torch.full((n,), 4, device="cuda", dtype=torch.int32)
    odd = even + 1
    norm = _norm(HEAD_DIM, 3001)

    ref_state, got_state = kv_state.clone(), kv_state.clone()
    ring = RING_SIZES[-1]
    _torch_reference(first, ref_state, norm, even, req, raw_out_loc, ring)
    expected, mask = _torch_reference(
        second, ref_state, norm, odd, req, raw_out_loc, ring
    )

    args = (norm.weight.data,)
    kw = {"ring_size": ring}
    c2_decode_norm(first, got_state, *args, even, req, raw_out_loc, EPS, **kw)
    got = c2_decode_norm(second, got_state, *args, odd, req, raw_out_loc, EPS, **kw)

    assert mask.all(), "step two must be all-odd"
    _compare(got, expected, mask, "two-step")
    assert torch.equal(got_state, ref_state), "pair state diverged across steps"
    # A step-two row must actually depend on step one: pooling against the
    # original state instead would give a different answer.
    stale = c2_decode_norm(
        second, kv_state.clone(), *args, odd, req, raw_out_loc, EPS, **kw
    )
    assert not torch.equal(got, stale), "step two ignored what step one parked"


def test_padded_rows_publish_nothing():
    """Padding rows with `raw_out_loc == 0` write neither cache nor state, even when
    their `req_pool_idx` aliases a live request."""
    n = 8
    num_state_rows = 5
    req = torch.tensor([0, 1, 2, 3, 4, 0, 0, 0], device="cuda", dtype=torch.int64)
    # Both parities among the padded rows: an even one would park in the state,
    # an odd one would consume it, and neither may happen.
    positions = torch.tensor([2, 3, 4, 5, 6, 0, 1, 0], device="cuda", dtype=torch.int32)
    raw_out_loc = torch.tensor(
        [7, 9, 11, 13, 15, 0, 0, 0], device="cuda", dtype=torch.int32
    )
    got, expected, odd, got_state, ref_state = _run(
        n,
        HEAD_DIM,
        seed=5000,
        positions=positions,
        req=req,
        raw_out_loc=raw_out_loc,
        num_state_rows=num_state_rows,
    )
    assert torch.equal(got_state, ref_state), "a padded row wrote the pair state"
    live = odd & (raw_out_loc != 0)
    assert live.any(), "test needs a live completing row"
    _compare(got, expected, live, "padded")


def test_saturated_and_tied_scores():
    """The closed form is `exp(-|s0 - s1|)`: a tie must give exactly 0.5/0.5, and a
    large gap must underflow to a clean one-sided pick, not a NaN."""
    n = 6
    kv_input, kv_state, positions, req, raw_out_loc = _inputs(n, HEAD_DIM, seed=6000)
    positions = torch.arange(1, 2 * n + 1, 2, device="cuda", dtype=torch.int32)
    ring = RING_SIZES[-1]
    # row 0 ties its partner; odd rows sit 200 above theirs and rows 2 and 4 200
    # below, so `exp(-|delta|)` underflows in fp32 from either side
    read, _ = _state_rows(req, positions, ring)
    kv_input[:, HEAD_DIM:] = 0.0
    kv_state[:, HEAD_DIM:] = 0.0
    kv_input[1::2, HEAD_DIM:] = 200.0
    kv_state[read[2::2], HEAD_DIM:] = 200.0
    norm = _norm(HEAD_DIM, 6001)
    ref_state, got_state = kv_state.clone(), kv_state.clone()

    expected, odd = _torch_reference(
        kv_input, ref_state, norm, positions, req, raw_out_loc, ring
    )
    got = c2_decode_norm(
        kv_input,
        got_state,
        norm.weight.data,
        positions,
        req,
        raw_out_loc,
        EPS,
        ring_size=ring,
    )
    assert odd.all() and torch.isfinite(got.float()).all()
    _compare(got, expected, odd, "saturated")
    # The saturated rows are one-sided picks: the odd rows keep their own kv,
    # rows 2 and 4 take their partner's.
    own = _torch_rmsnorm(
        kv_input[:, :HEAD_DIM].to(torch.bfloat16), norm.weight.data, EPS
    )
    partner = _torch_rmsnorm(
        kv_state[read, :HEAD_DIM].to(torch.bfloat16), norm.weight.data, EPS
    )
    rows = torch.zeros(n, dtype=torch.bool, device="cuda")
    rows[1::2] = True
    _compare(got, own, rows, "saturated own")
    rows = torch.zeros(n, dtype=torch.bool, device="cuda")
    rows[2::2] = True
    _compare(got, partner, rows, "saturated partner")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
