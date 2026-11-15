"""DFlash2 draft selection on a real TP group: every rank's K1 publishes its top-K words into the group's push plane
by multicast, every rank's K2 consumes the round (merge, lattice, walk) and commits it.

Usage::

    python test/registered/kernels/ops/speculative/test_dflash_select.py              # world sizes 2, 4, 8
    python test/registered/kernels/ops/speculative/test_dflash_select.py --num-gpu 8  # one world size
    # DFLASH_REAL_LOGITS=<bf16 [rows, 20480] logits shards> adds the real-row K1 cases.
"""

from __future__ import annotations

import atexit
import itertools
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.speculative.dflash_select import (
    dflash_local_topk,
    dflash_select_reference,
    dflash_select_walk,
    get_dflash_select_workspace_request,
    reference_candidates,
)
from sglang.srt.distributed.device_communicators.symm_workspace import (
    request_symm_workspace,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

R = 256
# The push slot holds the largest round a test publishes: 2 x SMs + 8 rows (the dispatch test), K 32.
MAX_ROWS, MAX_K = 512, 32


@cache_once
def _lease():
    """Per worker: the group of every rank and its symmetric workspace (None when it cannot host one)."""
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = ps.init_world_group(
        ranks=list(range(world)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=ps._WORLD)
    atexit.register(dist.destroy_process_group)
    request = get_dflash_select_workspace_request(MAX_ROWS, MAX_K)
    return request_symm_workspace(
        ps._WORLD.cpu_group, torch.device("cuda", local_rank), request
    )


@pytest.fixture(scope="module")
def ws():
    lease = _lease()
    if lease is None or not lease.has_multicast:
        pytest.skip("the group has no symmetric workspace with a multicast address")
    return lease


def _plane(lease):
    return lease.get_communicator().push


def _settle():
    """Every rank's kernels have completed, so every multicast store has landed (and nothing new is in flight)."""
    torch.cuda.synchronize()
    dist.barrier()


def _local_slots(lease):
    return lease.data.push_slots[lease.rank]  # [2 * world, slot_bytes] uint8


def _phase(lease):
    return (int(lease.control.epoch.item()) >> 32) & 1


def _shard_width(world):
    return 20480 if world <= 8 else 10240  # TP8 / TP16 production shards


def _make_inputs(world, gamma, bs, k, seed, ties=False, greedy_frac=0.25):
    """The same inputs on every rank (one seed), the vocab split into `world` shards of the production width."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n = bs * gamma
    w = _shard_width(world)
    V = w * world
    if ties:
        # Ties at the top, inside and across shards: per row 48 ids at 4.0 over normal noise.
        logits = torch.randn(n, V, device="cuda", generator=gen).to(torch.bfloat16)
        ids = torch.rand(n, V, device="cuda", generator=gen).argsort(dim=-1)[:, :48]
        logits.scatter_(1, ids, 4.0)
    else:
        logits = torch.randn(n, V, device="cuda", generator=gen).to(torch.bfloat16)
    a = (torch.randn(V, R, device="cuda", generator=gen) * 0.5).bfloat16()
    b = (torch.randn(V, R, device="cuda", generator=gen) * 0.5).bfloat16()
    hp_values = torch.randn(n, R, device="cuda", generator=gen).bfloat16()
    # The production layout: each rank's projection GEMM writes [logits shard | hp] rows of W + R columns.
    fused = [
        torch.cat([logits[:, r * w : (r + 1) * w], hp_values], dim=1)
        for r in range(world)
    ]
    shards = [f[:, :w] for f in fused]
    hp = fused[0][:, w:]
    # A column of [bs, gamma + 1] block ids, strided as the serving path passes it.
    anchor = torch.randint(0, V, (bs, gamma + 1), device="cuda", generator=gen)[:, 0]
    temps = torch.rand(bs, device="cuda", generator=gen) * 1.5 + 0.1
    greedy = torch.rand(bs, device="cuda", generator=gen) < greedy_frac
    unif = torch.rand(bs, gamma, device="cuda", generator=gen)
    return shards, a, b, hp, anchor, temps, greedy, unif


def _run_kernels(lease, shards, a, b, hp, anchor, temps, greedy, unif, mult, k):
    """One round: this rank publishes its shard, then consumes every rank's words and commits."""
    plane = _plane(lease)
    dflash_local_topk(shards[lease.rank], plane, k)
    return dflash_select_walk(plane, k, a, b, hp, anchor, temps, greedy, unif, mult)


def _round_state(lease, epoch_before, rounds):
    """After `rounds` rounds, once every rank is done: this rank's slots are back to the empty marker and the phase
    word advanced by one commit per round. Read before the next round may start; asserted by the caller."""
    _settle()
    clean = bool((_local_slots(lease) == 0).all())
    committed = int(lease.control.epoch.item()) == epoch_before + (rounds << 32)
    _settle()
    return clean, committed


def ref_vals_unary(cand, shards, mult, bs, gamma, k):
    full = torch.cat(shards, dim=-1).float()
    return (full.gather(-1, cand.view(bs * gamma, k)) * mult).view(bs, gamma, k)


def _check(lease, gamma, bs, k, seed, ties=False):
    mult = 0.75
    inputs = _make_inputs(lease.world_size, gamma, bs, k, seed, ties=ties)
    shards, a, b, hp, anchor, temps, greedy, unif = inputs
    epoch = int(lease.control.epoch.item())
    tokens, cand, q = _run_kernels(lease, *inputs, mult, k)
    clean, committed = _round_state(lease, epoch, 1)

    ref_tokens, ref_cand, ref_q, scores, ref_path = dflash_select_reference(
        shards, a, b, hp, anchor, temps, greedy, unif, mult, k
    )
    # Exact global top-K, (value desc, id asc).
    torch.testing.assert_close(cand, ref_cand, rtol=0, atol=0)
    # The slots are back to the empty marker and the round committed (the phase flipped once).
    assert clean and committed

    # q along the kernel's own path, against the reference lattice.
    path = (cand == tokens[:, :, None]).float().argmax(-1)
    assert bool((cand.gather(-1, path[:, :, None])[:, :, 0] == tokens).all())
    prev = torch.cat([torch.zeros_like(path[:, :1]), path[:, :-1]], dim=1)
    rows = scores.gather(2, prev[:, :, None, None].expand(-1, -1, 1, k))[:, :, 0]
    probs = torch.softmax(rows / temps[:, None, None], dim=-1)
    exp_q = torch.where(
        greedy[:, None, None], torch.nn.functional.one_hot(path, k).float(), probs
    )
    # Lattice entries are bf16(fp32 dot) in any accumulation order, so an entry may sit one bf16 ulp off the
    # reference; a row shift of up to d moves its softmax by at most q * 2d / T. Scaling by 1/T and 1/sum instead
    # of dividing adds up to an ulp of each scaled logit (2^-23 |row / T|) and of each probability.
    unary = ref_vals_unary(ref_cand, shards, mult, bs, gamma, k)
    dots = (rows - unary).abs().clamp_min(1e-30)
    ulp = torch.exp2(torch.floor(torch.log2(dots)) - 7).amax(-1, keepdim=True)
    scaled = (rows / temps[:, None, None]).abs().amax(-1, keepdim=True)
    slack = 2 * ulp / temps[:, None, None] + 4 * scaled * 2.0**-23 + 2.0**-22
    tol = exp_q * slack + 2e-6
    assert bool(((q - exp_q).abs() <= tol).all()), float(
        ((q - exp_q).abs() - tol).max()
    )

    # Tokens: every choice is the reference choice on the kernel's own row, except at a CDF boundary.
    cdf = probs.cumsum(-1)
    exp_idx = (unif[:, :, None] >= cdf).sum(-1).clamp_max(k - 1)
    exp_idx = torch.where(greedy[:, None], rows.argmax(-1), exp_idx)
    bad = exp_idx != path
    if bool(bad.any()):
        lo = cdf.gather(-1, torch.minimum(path, exp_idx)[:, :, None])[:, :, 0]
        near = (unif - lo).abs() < 1e-5 + slack[:, :, 0]
        assert bool((~bad | (near & ~greedy[:, None])).all()), (path[bad], exp_idx[bad])
    match = (tokens == ref_tokens).float().mean().item()
    return match


@pytest.mark.parametrize(
    "gamma,bs,k",
    list(itertools.product([4, 7], [1, 8, 64], [16])) + [(4, 8, 32), (7, 2, 32)],
)
def test_select_matches_reference(ws, gamma, bs, k):
    match = _check(ws, gamma, bs, k, seed=gamma * 100 + bs + k)
    assert match > 0.98


@pytest.mark.parametrize("k", [16, 32])
def test_select_ties(ws, k):
    _check(ws, 4, 8, k, seed=7, ties=True)


def test_matches_main_candidate_set(ws):
    from sglang.kernels.ops.sampling.topk import radix_topk

    gamma, bs, k = 4, 8, 16
    shards = _make_inputs(ws.world_size, gamma, bs, k, seed=3)[0]
    w = shards[0].shape[1]
    vals, ids = [], []
    for r, s in enumerate(shards):
        v, i = radix_topk(s.contiguous(), k)
        vals.append(v.float())
        ids.append(i.long() + r * w)
    top_v, sel = torch.topk(torch.cat(vals, -1), k, dim=-1)
    main_ids = torch.gather(torch.cat(ids, -1), -1, sel)
    ref_ids, ref_vals = reference_candidates(shards, k + 1)
    torch.testing.assert_close(top_v, ref_vals[:, :k], rtol=0, atol=0)
    # Rows with a tie at the K-th value may pick other tied ids; compare the rest as sets.
    clean = ref_vals[:, k - 1] != ref_vals[:, k]
    same = (main_ids.sort(-1).values == ref_ids[:, :k].sort(-1).values).all(-1)
    assert bool(same[clean].all()) and int(clean.sum()) > 0


def test_deterministic_and_reusable(ws):
    """Back-to-back rounds with no host synchronization in between: the plane's two halves alternate."""
    gamma, bs, k = 7, 8, 16
    inputs = _make_inputs(ws.world_size, gamma, bs, k, seed=11)
    epoch = int(ws.control.epoch.item())
    outs = [[x.clone() for x in _run_kernels(ws, *inputs, 1.0, k)] for _ in range(3)]
    clean, committed = _round_state(ws, epoch, 3)
    for o in outs[1:]:
        for x, y in zip(outs[0], o):
            assert torch.equal(x, y)
    assert clean and committed


@pytest.mark.parametrize("gamma,bs", [(4, 64), (7, 64)])
def test_against_main_chain(ws, gamma, bs):
    """Tokens and q against main's lattice (`_score_edges`, bf16 einsum) and Triton walk on the same candidates."""
    from sglang.kernels.ops.speculative.dflash import selector_walk_triton
    from sglang.srt.models.dflash import _score_edges

    k, mult = 16, 0.75
    inputs = _make_inputs(ws.world_size, gamma, bs, k, seed=bs + gamma, greedy_frac=0.0)
    shards, a, b, hp, anchor, temps, greedy, unif = inputs
    epoch = int(ws.control.epoch.item())
    tokens, cand, q = _run_kernels(ws, *inputs, mult, k)
    clean, committed = _round_state(ws, epoch, 1)
    assert clean and committed
    unary = ref_vals_unary(cand, shards, mult, bs, gamma, k)
    scores = _score_edges(
        predecessor_table=a,
        successor_table=b,
        candidate_ids=cand,
        unary_logits=unary,
        hidden=hp.view(bs, gamma, -1),
        anchor_token_ids=anchor,
        top_k=k,
    )
    main_tokens, main_q = selector_walk_triton(
        candidate_ids=cand,
        scores=scores,
        uniforms=unif,
        temperatures=temps,
        greedy_mask=greedy,
    )
    # Compare up to the first slot where the paths split (a CDF-boundary choice), then stop that request.
    same_prefix = (tokens == main_tokens).int().cumprod(1).bool()
    rows = same_prefix.sum().item()
    dq = (q - main_q).abs()[same_prefix]
    mismatch = 1.0 - same_prefix[:, -1].float().mean().item()
    print(
        f"tp {ws.world_size} gamma {gamma} bs {bs}: requests with a different path {mismatch:.4f}, slots compared {rows}, max |dq| {dq.max().item():.3e}"
    )
    assert mismatch < 0.05
    assert dq.max().item() < 1e-2


# K1 shard widths (W = warps x 1024): the dispatched TP8 20 warps and TP16 10, and 8 warps (with 10, 4 redux rounds for
# the K 32 warp bound, the most the kernel allows).
_K1_WIDTHS = [20480, 10240, 8192]
_K1_MIN_BLOCKS = [1, 2]


def _k1_rows(kind, n, w, gen):
    """Rows of width w in the production layout (a view of [n, w + 256]); w / 1024 warps x 32 lanes x 32 elements."""
    nw = w // 1024
    buf = torch.randn(n, w + 256, device="cuda", generator=gen).bfloat16()
    x = buf[:, :w]
    # 40 maxima at +0.0 / -0.0, one per lane over all warps (they tie), noise near -10
    if kind == "signed_zero":
        x.sub_(10.0)
        lanes = [(wp, ln) for ln in range(32) for wp in range(nw)][:40]
        for j, (wp, ln) in enumerate(lanes):
            x[:, wp * 1024 + ln * 16] = -0.0 if j % 2 else 0.0
    # equal spikes in tiny noise: ties at the K-th value across lanes and warps
    elif kind == "spikes":
        x.mul_(0.01)
        x[:, ::997] = 4.0
    # 70 equal maxima, one per lane: T is their value and n = 70 > 64 candidates
    elif kind == "spikes70":
        x.mul_(0.01).sub_(100.0)
        lanes = [(wp, ln) for ln in range(32) for wp in range(nw)][:70]
        for wp, ln in lanes:
            x[:, wp * 1024 + ln * 16] = 10.0
    # 20 chunks of 8 equal maxima over noise near -100: 20 chunks but 160 candidates
    elif kind == "dense160":
        x.mul_(0.01).sub_(100.0)
        lanes = [(wp, ln) for ln in range(32) for wp in range(nw)][:20]
        for wp, ln in lanes:
            x[:, wp * 1024 + ln * 16 : wp * 1024 + ln * 16 + 8] = 10.0
    # few distinct values, -0.0 mixed with +0.0: every chunk reaches T
    elif kind == "ties":
        x.copy_(
            (torch.randint(0, 6, (n, w), device="cuda", generator=gen) - 3).bfloat16()
        )
        x[x == 0] = -0.0
    # all-equal rows: +0.0 and -0.0, then a constant; the order is by position alone
    elif kind == "equal":
        x.zero_()
        x[: n // 2, ::3] = -0.0
        x[n // 2 :] = 1.5
    elif kind == "real":
        import os

        path = os.environ.get("DFLASH_REAL_LOGITS")
        if path is None:
            pytest.skip(
                "DFLASH_REAL_LOGITS (bf16 [rows, 20480] TP8 logits shards) not set"
            )
        real = torch.load(path).cuda()
        rows = real if w == real.shape[1] else real.view(-1, w)
        x.copy_(
            rows[torch.randint(0, rows.shape[0], (n,), device="cuda", generator=gen)]
        )
    return x


def _k1_round(lease, shards, k, min_blocks=None):
    """Rank r publishes shards[r] (a K1-only round, nothing consumes it); returns every rank's words as this rank's
    workspace holds them, then refills the empty marker on every rank."""
    n = shards[0].shape[0]
    dflash_local_topk(shards[lease.rank], _plane(lease), k, _min_blocks=min_blocks)
    _settle()
    slots = _local_slots(lease)
    phase = _phase(lease)
    words = [
        slots[phase * lease.world_size + src, : n * k * 4]
        .view(torch.int32)
        .view(n, k)
        .clone()
        for src in range(lease.world_size)
    ]
    _settle()
    slots.zero_()
    _settle()
    return words


def _k1_ref_words(x, k):
    bits = x.view(torch.int16).to(torch.int32) & 0xFFFF
    bits = torch.where(bits == 0x8000, torch.zeros_like(bits), bits)
    key = torch.where(bits & 0x8000 != 0, bits ^ 0xFFFF, bits ^ 0x8000).long()
    pos = torch.arange(x.shape[1], device=x.device)
    top = (key * 65536 + (65535 - pos)).topk(k, dim=-1).values
    return ((top >> 16 << 16) | (65536 - (top & 65535))).to(torch.int32)  # position + 1


@pytest.mark.parametrize("w", _K1_WIDTHS)
@pytest.mark.parametrize("k", [32, 16])
@pytest.mark.parametrize("min_blocks", _K1_MIN_BLOCKS)
@pytest.mark.parametrize(
    "kind",
    [
        "random",
        "signed_zero",
        "spikes",
        "spikes70",
        "dense160",
        "ties",
        "equal",
        "real",
    ],
)
def test_local_topk_tilings(ws, w, k, min_blocks, kind):
    """Every K1 tiling publishes the reference words to every rank. spikes70 ranks 64 < n <= kCountCap candidates
    (128 slots); dense160 overflows the candidate list, ties and equal the chunk list, so they take the exact
    fallback."""
    n = 48
    shards = [
        _k1_rows(
            kind, n, w, torch.Generator(device="cuda").manual_seed(w + k + 1000 * src)
        )
        for src in range(ws.world_size)
    ]
    words = _k1_round(ws, shards, k, min_blocks)
    for src, x in enumerate(shards):
        assert torch.equal(words[src], _k1_ref_words(x, k)), f"rank {src}'s words"


@pytest.mark.parametrize("w", [20480, 10240])
@pytest.mark.parametrize("sms_times", [0, 1, 2])
def test_local_topk_dispatch(ws, w, sms_times):
    """The dispatched tiling for row counts that need 1, 2 and (a second wave) 3 CTAs per SM."""
    from sglang.kernels.arch import current_arch

    k = 32
    n = sms_times * current_arch().num_sms + 8
    shards = [
        _k1_rows(
            "random", n, w, torch.Generator(device="cuda").manual_seed(n + 1000 * src)
        )
        for src in range(ws.world_size)
    ]
    words = _k1_round(ws, shards, k)
    for src, x in enumerate(shards):
        assert torch.equal(words[src], _k1_ref_words(x, k)), f"rank {src}'s words"


def _precompile(world_sizes):
    from sglang.kernels.jit.utils.compile.parallel import parallel_compile
    from sglang.kernels.ops.speculative.dflash_select import (
        _jit_local_topk_module,
        _jit_select_walk_module,
    )

    builders = [
        lambda k=k, w=w, mb=mb: _jit_local_topk_module(k, w, mb)
        for w in _K1_WIDTHS
        for k in (32, 16)
        for mb in _K1_MIN_BLOCKS
    ]
    builders += [
        lambda k=k, g=g: _jit_select_walk_module(k, g) for k in (32, 16) for g in (4, 7)
    ]
    parallel_compile(builders)


if __name__ == "__main__":
    multigpu_pytest_main(
        __name__, __file__, num_gpus=(2, 4, 8), pre_launch_fn=_precompile
    )
