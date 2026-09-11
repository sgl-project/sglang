"""Verify compressor: exact decode replay, padding, wrap and rejected prefixes."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.c2 import (
    c2_decode_norm_rope_store,
    c2_verify_norm_rope_store,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import RMSNorm
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or torch.cuda.get_device_capability()[0] < 10,
    reason="the compressor packs FP4 with SM100 instructions",
)

EPS = 1e-6
HEAD_DIMS = (512,)
ROPE_DIM = 64
RATIO = 2
PAGE_SIZE = 128
PAGE_BYTES = -(-584 * PAGE_SIZE // 576) * 576
DRAFT_LENS = (2, 5, 6, 9)
VERIFY_BATCHES = (1, 3, 8)


def _norm(dim: int, seed: int) -> RMSNorm:
    """`DeepseekV41Compressor.norm` as the model holds it: bf16 weight, because
    the parameter is created inside `set_default_torch_dtype(model dtype)`."""
    with set_default_torch_dtype(torch.bfloat16):
        norm = RMSNorm(dim, EPS).cuda()
    assert norm.weight.dtype == torch.bfloat16
    # Not `ones`: a constant weight cannot catch a wrong per-element index.
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(dim, generator=g, device="cuda"))
    return norm


def _freqs(max_pos, seed):
    """`layer.freqs_cis` and the fp32 real/imag-interleaved view the kernel
    indexes itself, at `positions - 1`."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ang = torch.randn(max_pos, ROPE_DIM // 2, generator=g, device="cuda")
    freqs = torch.polar(torch.ones_like(ang), ang)
    return freqs, torch.view_as_real(freqs).flatten(-2).contiguous().float()


def _cache(max_slot):
    """A compressed-pool buffer wide enough for `max_slot`, zeroed so an
    untouched slot is recognizable."""
    return torch.zeros(
        max_slot // PAGE_SIZE + 2, PAGE_BYTES, dtype=torch.uint8, device="cuda"
    )


def _spec_ring_size(draft_len):
    """`get_compress_state_ring_size(2, is_speculative=True, draft_len)`."""
    return 1 << (draft_len + 1).bit_length()


def _verify_inputs(bs, draft_len, dim, seed, *, starts=None, pad_reqs=0):
    """A target-verify batch: `draft_len` consecutive positions per request,
    request-major. Block heads alternate parity by default, so some blocks open
    by consuming the ring and some by parking into it."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = bs * draft_len
    ring_size = _spec_ring_size(draft_len)
    kv_input = torch.randn(n, 2 * dim, generator=g, device="cuda", dtype=torch.float32)
    kv_state = torch.randn(
        (bs + 1) * ring_size, 2 * dim, generator=g, device="cuda", dtype=torch.float32
    )
    if starts is None:
        starts = torch.arange(bs, device="cuda") + 4
    offsets = torch.arange(draft_len, device="cuda")
    positions = (starts[:, None] + offsets[None, :]).flatten().to(torch.int32)
    req = torch.arange(bs, device="cuda", dtype=torch.int64).repeat_interleave(
        draft_len
    )
    raw_out_loc = torch.arange(n, device="cuda", dtype=torch.int32) * 2 + 3
    if pad_reqs:
        # Graph padding pads whole request slots, aliasing a live
        # `req_pool_idx`, and leaves their position buffer at zero.
        pad = req >= bs - pad_reqs
        raw_out_loc[pad] = 0
        positions[pad] = 0
        req[pad] = 0
    return kv_input, kv_state, positions, req, raw_out_loc, ring_size


def _decode_replay(
    kv_input, kv_state, norm, positions, req, raw_out_loc, freqs_cis, cache, **kw
):
    """The same block, one position per launch -- what one fused verify launch
    has to reproduce. Step `j` is a decode step over row `j` of every request,
    carrying the pair ring between steps exactly as the served decode path does.
    """
    draft_len = kw["draft_len"]
    n, dim = positions.shape[0], kv_input.shape[1] // 2
    rows = torch.arange(n, device="cuda").view(-1, draft_len)
    out = torch.zeros(n, dim, device="cuda", dtype=torch.bfloat16)
    for j in range(draft_len):
        idx = rows[:, j]
        out[idx] = c2_decode_norm_rope_store(
            kv_input[idx].contiguous(),
            kv_state,
            norm.weight.data,
            positions[idx].contiguous(),
            req[idx].contiguous(),
            raw_out_loc[idx].contiguous(),
            EPS,
            freqs_cis,
            cache,
            page_size=PAGE_SIZE,
            ring_size=kw["ring_size"],
            out=torch.zeros(idx.numel(), dim, device="cuda", dtype=torch.bfloat16),
        )
    return out


def _run_verify(kv_input, kv_state, norm, positions, req, raw_out_loc, **kw):
    """One `c2_verify_norm_rope_store` call; returns `(latent, cache)`. `out` is
    zeroed so rows the kernel skips compare equal to the replay's."""
    n, dim = positions.shape[0], kv_input.shape[1] // 2
    freqs_cis, cache = kw["freqs_cis"], kw["cache"]
    got = c2_verify_norm_rope_store(
        kv_input,
        kv_state,
        norm.weight.data,
        positions,
        req,
        raw_out_loc,
        EPS,
        freqs_cis,
        cache,
        page_size=PAGE_SIZE,
        ring_size=kw["ring_size"],
        draft_len=kw["draft_len"],
        out=torch.zeros(n, dim, device="cuda", dtype=torch.bfloat16),
    )
    return got, cache


@pytest.mark.parametrize("pad_reqs", (0, 1))
@pytest.mark.parametrize("draft_len", DRAFT_LENS)
@pytest.mark.parametrize("bs", VERIFY_BATCHES)
def test_verify_matches_decode_replay(bs, draft_len, pad_reqs):
    """The load-bearing property: one verify launch over a block equals
    `draft_len` decode launches over the same rows -- same latents, same pair
    ring, same cache bytes, bitwise. Verify takes the in-block partner from
    `kv_input` where decode takes it from the ring, and that substitution has to
    be invisible."""
    if pad_reqs >= bs:
        pytest.skip("an all-padded batch has no live block to compare")
    dim = HEAD_DIMS[0]
    kv_input, kv_state, positions, req, raw_out_loc, ring_size = _verify_inputs(
        bs, draft_len, dim, seed=20000 + bs * 97 + draft_len, pad_reqs=pad_reqs
    )
    norm = _norm(dim, 20001 + bs + draft_len)
    _, freqs_cis = _freqs(int(positions.max().item()) + 2, 20002 + draft_len)
    slots_max = int((raw_out_loc // RATIO).max().item())
    cache_v, cache_d = _cache(slots_max), _cache(slots_max)
    state_v, state_d = kv_state.clone(), kv_state.clone()
    kw = dict(ring_size=ring_size, draft_len=draft_len)

    got, _ = _run_verify(
        kv_input,
        state_v,
        norm,
        positions,
        req,
        raw_out_loc,
        freqs_cis=freqs_cis,
        cache=cache_v,
        **kw,
    )
    expected = _decode_replay(
        kv_input, state_d, norm, positions, req, raw_out_loc, freqs_cis, cache_d, **kw
    )

    assert cache_d.any(), "the replay stored nothing, so the comparison is empty"
    assert torch.equal(got, expected), "latent differs from the decode replay"
    assert torch.equal(state_v, state_d), "pair ring differs from the decode replay"
    assert torch.equal(cache_v, cache_d), "cache bytes differ from the decode replay"


@pytest.mark.parametrize("draft_len", DRAFT_LENS)
def test_verify_reads_the_ring_only_on_the_first_row(draft_len):
    """A block's first row is the only one allowed to consume the ring. Move the
    slot it reads and its latent must move with it; every later row pairs inside
    the block and must not notice."""
    bs, dim = 4, HEAD_DIMS[0]
    # Odd heads, so every block opens by completing a group against the ring.
    starts = 2 * torch.arange(bs, device="cuda") + 5
    kv_input, kv_state, positions, req, raw_out_loc, ring_size = _verify_inputs(
        bs, draft_len, dim, seed=21000 + draft_len, starts=starts
    )
    norm = _norm(dim, 21001 + draft_len)
    _, freqs_cis = _freqs(int(positions.max().item()) + 2, 21002 + draft_len)
    slots_max = int((raw_out_loc // RATIO).max().item())
    kw = dict(ring_size=ring_size, draft_len=draft_len, freqs_cis=freqs_cis)

    base, _ = _run_verify(
        kv_input,
        kv_state.clone(),
        norm,
        positions,
        req,
        raw_out_loc,
        cache=_cache(slots_max),
        **kw,
    )
    heads = torch.arange(0, bs * draft_len, draft_len, device="cuda")
    read = req[heads] * ring_size + (positions[heads].to(torch.int64) - 1) % ring_size
    moved = kv_state.clone()
    moved[read] += 1.0
    got, _ = _run_verify(
        kv_input,
        moved,
        norm,
        positions,
        req,
        raw_out_loc,
        cache=_cache(slots_max),
        **kw,
    )

    rest = torch.ones(bs * draft_len, dtype=torch.bool, device="cuda")
    rest[heads] = False
    assert not torch.equal(got[heads], base[heads]), "a block head ignored the ring"
    assert torch.equal(got[rest], base[rest]), "a later row went through the ring"


def test_verify_rejects_a_ring_narrower_than_the_block():
    """`ring_size > draft_len` is the whole reason a block's own publishes stay
    off the slot its first row reads, so the kernel refuses a narrower ring
    rather than racing quietly."""
    bs, draft_len, dim = 2, 4, HEAD_DIMS[0]
    kv_input, kv_state, positions, req, raw_out_loc, _ = _verify_inputs(
        bs, draft_len, dim, seed=22000
    )
    norm = _norm(dim, 22001)
    _, freqs_cis = _freqs(int(positions.max().item()) + 2, 22002)
    with pytest.raises(Exception, match="must be wider than the draft length"):
        _run_verify(
            kv_input,
            kv_state,
            norm,
            positions,
            req,
            raw_out_loc,
            cache=_cache(int((raw_out_loc // RATIO).max().item())),
            freqs_cis=freqs_cis,
            ring_size=draft_len,
            draft_len=draft_len,
        )


@pytest.mark.parametrize("start", (31, 32))
@pytest.mark.parametrize("dtype", (torch.int32, torch.int64))
def test_rejected_prefix_then_next_verify(start, dtype):
    # Compare with a decode history that never saw the rejected suffix. The
    # next verify starts at the committed position, including across ring wrap.
    bs, draft_len, dim = 3, 6, 512
    inputs, initial, pos, req, loc, ring = _verify_inputs(
        bs,
        draft_len,
        dim,
        23000,
        starts=torch.full((bs,), start, device="cuda"),
    )
    pos, loc = pos.to(dtype), loc.to(dtype)
    norm = _norm(dim, 23001)
    _, freqs = _freqs(start + 2 * draft_len + 2, 23002)
    rows = torch.arange(bs * draft_len, device="cuda").view(bs, draft_len)
    for accepted in range(1, draft_len + 1):
        state = initial.clone()
        reference = initial.clone()
        cache, ref_cache = _cache(128), _cache(128)
        kw = dict(draft_len=draft_len, ring_size=ring, freqs_cis=freqs)
        _run_verify(inputs, state, norm, pos, req, loc, cache=cache, **kw)
        for j in range(accepted):
            idx = rows[:, j]
            c2_decode_norm_rope_store(
                inputs[idx],
                reference,
                norm.weight.data,
                pos[idx],
                req[idx],
                loc[idx],
                EPS,
                freqs,
                ref_cache,
                page_size=PAGE_SIZE,
                ring_size=ring,
            )
        # Do not count speculative cache bytes that have no committed reader.
        cache.zero_()
        ref_cache.zero_()
        next_inputs = inputs.flip(0).contiguous()
        got, _ = _run_verify(
            next_inputs,
            state,
            norm,
            pos + accepted,
            req,
            loc,
            cache=cache,
            **kw,
        )
        expected = _decode_replay(
            next_inputs,
            reference,
            norm,
            pos + accepted,
            req,
            loc,
            freqs,
            ref_cache,
            ring_size=ring,
            draft_len=draft_len,
        )
        assert torch.equal(got, expected), f"{start=} {accepted=}: latent differs"
        assert torch.equal(cache, ref_cache), f"{start=} {accepted=}: cache differs"


def test_verify_cuda_graph_replay():
    bs, draft_len, dim = 3, 6, 512
    inputs, initial, pos, req, loc, ring = _verify_inputs(bs, draft_len, dim, 24000)
    norm = _norm(dim, 24001)
    _, freqs = _freqs(32, 24002)
    state, cache = initial.clone(), _cache(128)
    kw = dict(ring_size=ring, draft_len=draft_len, freqs_cis=freqs, cache=cache)
    _run_verify(inputs, state, norm, pos, req, loc, **kw)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got, _ = _run_verify(inputs, state, norm, pos, req, loc, **kw)
    state.copy_(initial)
    cache.zero_()
    graph.replay()
    ref_cache = _cache(128)
    expected = _decode_replay(
        inputs,
        initial,
        norm,
        pos,
        req,
        loc,
        freqs,
        ref_cache,
        ring_size=ring,
        draft_len=draft_len,
    )
    assert torch.equal(got, expected)
    assert torch.equal(state, initial)
    assert torch.equal(cache, ref_cache)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
