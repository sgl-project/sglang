# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_FUSED_SELECTOR: the fused DFlash2 draft head is bitwise equal to the unfused one.

dflash_selector_select must return torch.topk's values, indices and tie order exactly, plus the lattice operands that
the compiled _score_edges builds; the whole head (_SelectorDraftSampler with the flag) must write the same tokens,
candidates, q rows and draws as without it, eagerly and under CUDA-graph replay; and an FP64 walk written from the
model code checks the walk and the lattice independently of both paths.
"""

import types
import zlib

import pytest
import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-large")

DEV = "cuda"
# The TP16 K3.1 draft head: 16 ranks x top-16 gathered, selector rank 256, block 8 (7 draft slots).
TP, K, RANK, SLOTS, BLOCK = 16, 16, 256, 7, 8
WIDTH = TP * K
VOCAB = 163840
SHARD = VOCAB // TP
HIDDEN = 7168
# This process plays rank 9: its global ids start at 92160, above 2**16.
THIS_RANK = 9


def _bits(x: torch.Tensor) -> torch.Tensor:
    """A NaN-safe bitwise view for torch.equal."""
    return {torch.float32: x.view(torch.int32), torch.bfloat16: x.view(torch.int16)}.get(x.dtype, x)


def _assert_bitwise(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
    assert got.shape == want.shape and got.dtype == want.dtype, (name, got.shape, want.shape, got.dtype, want.dtype)
    eq = _bits(got) == _bits(want)
    if not bool(eq.all()):
        bad = (~eq).nonzero()[:5].tolist()
        raise AssertionError(f"{name}: {int((~eq).sum())} of {eq.numel()} differ, first at {bad}")


# --------------------------------------------------------------------------- T1: dflash_selector_select alone


def _gathered_rows(case: str, rows: int, gen: torch.Generator) -> torch.Tensor:
    shape = (rows, WIDTH)
    if case == "bf16_normal":
        return (torch.randn(shape, generator=gen, device=DEV) * 3).bfloat16().float()
    if case == "sorted_runs":
        # Like the real gather: 16 ranks' descending bf16 top-16 runs side by side, ties across ranks.
        v = (torch.randn(shape, generator=gen, device=DEV) * 2).bfloat16().float().view(rows, TP, K)
        return v.sort(dim=-1, descending=True).values.reshape(shape)
    if case == "one_octave":
        # Every key shares sign and exponent: the radix starts below bit 23 (an odd number of bits above bit 16).
        return torch.rand(shape, generator=gen, device=DEV) + 1.0
    if case == "one_octave_bf16":
        return (torch.rand(shape, generator=gen, device=DEV) + 1.0).bfloat16().float()
    if case == "radix_boundary":
        # Exactly 16 keys have the mantissa MSB (bit 22) set: the first radix round finds count == rank, an odd span.
        # Four of them tie at the smallest selected value, so the order in which the selection is handed to the sort
        # (keys above the k-th, then the k-th's ties) shows in the tie order.
        out = torch.rand(shape, generator=gen, device=DEV) * 0.49 + 1.0
        for r in range(rows):
            perm = torch.randperm(WIDTH, generator=gen, device=DEV)[:K]
            top = torch.rand((K,), generator=gen, device=DEV) * 0.49 + 1.51
            top[:4] = 1.5
            out[r, perm] = top
        return out
    if case == "few_values":
        return torch.randint(0, 4, shape, generator=gen, device=DEV).float() * 0.5 - 1.0
    if case == "all_equal":
        return torch.full(shape, 1.25, device=DEV)
    if case == "tie_at_k":
        # 10 values above, 30 copies of the 16th value, the rest below, in random positions.
        out = torch.rand(shape, generator=gen, device=DEV) - 2.0
        for r in range(rows):
            perm = torch.randperm(WIDTH, generator=gen, device=DEV)
            out[r, perm[:10]] = torch.arange(10, device=DEV, dtype=torch.float32) + 5.0
            out[r, perm[10:40]] = 2.0
        return out
    if case == "signed_zero":
        sign = torch.randint(0, 2, shape, generator=gen, device=DEV).float() * 2 - 1
        out = sign * 0.0
        out[:, ::50] = 1.0
        return out
    if case == "neg_inf":
        out = torch.full(shape, float("-inf"), device=DEV)
        out[:, ::43] = torch.randn((rows, len(range(0, WIDTH, 43))), generator=gen, device=DEV)
        return out
    if case == "nan_inf":
        out = torch.randn(shape, generator=gen, device=DEV)
        out[:, 3] = float("nan")
        # A sign-bit NaN (0xffc00000): torch's radix key still ranks it above every number.
        out[:, 77] = torch.tensor(-4194304, dtype=torch.int32, device=DEV).view(torch.float32)
        out[:, 100] = float("inf")
        out[:, 200] = float("-inf")
        out[:, 201] = float("inf")
        return out
    raise ValueError(case)


def _gathered_ids(rows: int, dtype: torch.dtype, gen: torch.Generator) -> torch.Tensor:
    ids = torch.stack(
        [torch.randperm(VOCAB, generator=gen, device=DEV)[:WIDTH] for _ in range(rows)]
    )
    ids[0, :4] = torch.tensor([VOCAB - 1, 65535, 65536, 0], device=DEV)
    return ids.to(dtype)


def _reference_select(vals, ids, proj, anchors, pred_t, succ_t):
    """torch.topk + torch.gather + the model's indexing and bf16 product (dflash.py _score_edges)."""
    bs = anchors.shape[0]
    top_vals, sel = torch.topk(vals, K, dim=-1)
    cand = torch.gather(ids, -1, sel).long()
    cand3 = cand.view(bs, SLOTS, K)
    pred_ids = torch.cat([anchors[:, None, None].expand(-1, 1, K), cand3[:, :-1]], dim=1)
    pred_h = pred_t[pred_ids] * proj[:, :, None]
    keys = succ_t[cand3]
    return cand, top_vals, pred_h.reshape(bs * SLOTS, K, RANK), keys.reshape(bs * SLOTS, K, RANK)


@pytest.fixture(scope="module")
def tables():
    gen = torch.Generator(device=DEV).manual_seed(7)
    pred_t = (torch.randn((VOCAB, RANK), generator=gen, device=DEV) * 0.5).bfloat16()
    succ_t = (torch.randn((VOCAB, RANK), generator=gen, device=DEV) * 0.5).bfloat16()
    return pred_t, succ_t


@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("id_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize(
    "case",
    [
        "bf16_normal",
        "sorted_runs",
        "one_octave",
        "one_octave_bf16",
        "radix_boundary",
        "few_values",
        "all_equal",
        "tie_at_k",
        "signed_zero",
        "neg_inf",
        "nan_inf",
    ],
)
def test_select_matches_torch_topk_and_model_indexing(tables, bs, id_dtype, case):
    """Catches any change of torch.topk's selection, value order or tie order (the verify consumes candidate order
    through q), of the predecessor slot shift and anchor row, and of the bf16 product rounding."""
    from sglang.kernels.ops.speculative.dflash_selector import dflash_selector_select

    pred_t, succ_t = tables
    gen = torch.Generator(device=DEV).manual_seed(zlib.crc32(f"{bs}/{id_dtype}/{case}".encode()))
    rows = bs * SLOTS
    vals = _gathered_rows(case, rows, gen)
    ids = _gathered_ids(rows, id_dtype, gen)
    proj = torch.randn((bs, SLOTS, RANK), generator=gen, device=DEV).bfloat16()
    input_ids = torch.randint(0, VOCAB, (bs * BLOCK,), generator=gen, device=DEV)
    input_ids[0] = VOCAB - 1
    anchors = input_ids.view(bs, BLOCK)[:, 0]  # strided, as the sampler passes it
    want = _reference_select(vals, ids, proj, anchors, pred_t, succ_t)

    cand = torch.full((rows, K), -7, dtype=torch.int64, device=DEV)
    unary = torch.full((rows, K), float("nan"), device=DEV)
    pred_h = torch.full((rows, K, RANK), float("nan"), dtype=torch.bfloat16, device=DEV)
    keys = torch.full_like(pred_h, float("nan"))
    dflash_selector_select(
        gathered_vals=vals,
        gathered_ids=ids,
        hidden=proj,
        anchors=anchors,
        pred_table=pred_t,
        succ_table=succ_t,
        candidates=cand,
        unary=unary,
        pred_h=pred_h,
        keys=keys,
    )
    for name, got, ref in zip(("candidates", "unary", "pred_h", "keys"), (cand, unary, pred_h, keys), want):
        _assert_bitwise(name, got, ref)


# --------------------------------------------------------------------------- T4: the local top-k alone


def _local_rows(case: str, rows: int, gen: torch.Generator) -> torch.Tensor:
    shape = (rows, SHARD)
    if case == "normal":
        return (torch.randn(shape, generator=gen, device=DEV) * 2).bfloat16()
    if case == "integers":
        # Integer logits in a narrow range: hundreds of BF16 ties at the 16th place.
        return torch.randint(-40, 41, shape, generator=gen, device=DEV).bfloat16()
    if case == "top_ties":
        # 3000 copies of the maximum: more candidates than the kernel keeps, so the exact radix path runs.
        out = (torch.randn(shape, generator=gen, device=DEV) - 5).bfloat16()
        for r in range(rows):
            out[r, torch.randperm(SHARD, generator=gen, device=DEV)[:3000]] = 3.0
        return out
    if case == "all_equal":
        return torch.full(shape, 0.5, dtype=torch.bfloat16, device=DEV)
    if case == "all_negative":
        return (-torch.rand(shape, generator=gen, device=DEV) * 8 - 1).bfloat16()
    if case == "special":
        out = (torch.randn(shape, generator=gen, device=DEV)).bfloat16()
        bits = out.view(torch.int16)
        bits[:, 5] = 0x7FC0  # NaN
        bits[:, 9] = -64  # 0xFFC0: sign-bit NaN, below -inf in flashinfer's order
        bits[:, 1000] = 0x7F80  # +inf
        bits[:, 3000] = -128  # 0xFF80: -inf
        bits[:, 77:1100:7] = 0  # +0.0
        bits[:, 78:1100:7] = -32768  # -0.0
        return out
    if case == "neg_inf_tail":
        out = torch.full(shape, float("-inf"), dtype=torch.bfloat16, device=DEV)
        out[:, ::700] = torch.randn((rows, len(range(0, SHARD, 700))), generator=gen, device=DEV).bfloat16()
        return out
    raise ValueError(case)


@pytest.mark.parametrize("rows", [7, 14])
@pytest.mark.parametrize(
    "case", ["normal", "integers", "top_ties", "all_equal", "all_negative", "special", "neg_inf_tail"]
)
def test_local_topk_matches_flashinfer(rows, case):
    """Catches any departure from flashinfer.top_k(sorted=True, deterministic=True): which pivot ties are kept and their
    order (thread-strided over 1024 threads, then a stable value sort), NaN / -0 / -inf ordering, the exact fallback
    when too many keys tie at the top, and the global-id offset and FP32 cast the gather needs."""
    from flashinfer import top_k

    from sglang.kernels.ops.speculative.dflash_selector import dflash_selector_local_topk

    gen = torch.Generator(device=DEV).manual_seed(zlib.crc32(f"local/{rows}/{case}".encode()))
    logits = _local_rows(case, rows, gen)
    offset = THIS_RANK * SHARD
    want_vals, want_idx = top_k(logits, K, sorted=True, deterministic=True)
    got_ids, got_vals = dflash_selector_local_topk(logits=logits, k=K, offset=offset)
    _assert_bitwise("ids", got_ids, want_idx.long() + offset)
    _assert_bitwise("vals", got_vals, want_vals.float())
    # A row view with a stride, as a sliced logits tensor would be.
    wide = torch.empty((rows, SHARD + 64), dtype=torch.bfloat16, device=DEV)
    wide[:, :SHARD] = logits
    got_ids2, got_vals2 = dflash_selector_local_topk(logits=wide[:, :SHARD], k=K, offset=offset)
    _assert_bitwise("ids (strided)", got_ids2, got_ids)
    _assert_bitwise("vals (strided)", got_vals2, got_vals)


# --------------------------------------------------------------------------- T2/T3: the whole draft head


class _FakeGroup:
    """TP16 all_gather along the last dim: this rank's tensor in slot THIS_RANK, fixed other-rank candidates around it."""

    world_size = TP

    def __init__(self, others_vals: torch.Tensor, others_ids: torch.Tensor):
        self._vals, self._ids = others_vals, others_ids  # [TP, max_rows, K]

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == -1
        src = self._vals if x.is_floating_point() else self._ids
        parts = [x if r == THIS_RANK else src[r, : x.shape[0]].to(x.dtype) for r in range(TP)]
        return torch.cat(parts, dim=-1)


def _make_head(gen: torch.Generator, ties: bool, packed: bool, max_bs: int):
    from sglang.srt.models.dflash import CandidateSelector, DFlash2DraftModel

    selector = CandidateSelector(hidden_size=HIDDEN, vocab_size=VOCAB, state_rank=RANK, top_k=K).to(DEV)
    with torch.no_grad():
        selector.predecessor_codebook.copy_(torch.randn((VOCAB, RANK), generator=gen, device=DEV) * 0.5)
        selector.successor_codebook.copy_(torch.randn((VOCAB, RANK), generator=gen, device=DEV) * 0.5)
        selector.hidden_projection.weight.copy_(torch.randn((RANK, HIDDEN), generator=gen, device=DEV) * 0.02)
    selector = selector.bfloat16()
    if ties:
        # Small-integer weights and hiddens: integer logits, so local and global top-k cut through ties.
        weight = torch.randint(-2, 3, (SHARD, HIDDEN), generator=gen, device=DEV).bfloat16()
    else:
        weight = (torch.randn((SHARD, HIDDEN), generator=gen, device=DEV) * 0.02).bfloat16()
    lm_head = types.SimpleNamespace(
        weight=weight,
        quant_method=None,
        use_attn_tp_group=False,
        shard_indices=types.SimpleNamespace(
            num_org_elements=SHARD, org_vocab_start_index=THIS_RANK * SHARD
        ),
    )
    rows = max_bs * SLOTS
    # Other ranks' top-16 straddle this rank's (top-16 of 10240 logits sits near 3.5 sigma), so the global top-k
    # mixes ranks and, with integer logits, cuts through cross-rank ties.
    sigma = (HIDDEN * 2.0) ** 0.5 if ties else 0.02 * HIDDEN**0.5
    others_vals = torch.randn((TP, rows, K), generator=gen, device=DEV) * (0.2 * sigma) + 3.5 * sigma
    others_vals = (others_vals.round() if ties else others_vals.bfloat16().float()).sort(-1, descending=True).values
    others_ids = torch.stack(
        [
            torch.stack([torch.randperm(SHARD, generator=gen, device=DEV)[:K] for _ in range(rows)]) + r * SHARD
            for r in range(TP)
        ]
    )
    group = _FakeGroup(others_vals, others_ids)
    model = types.SimpleNamespace(
        lm_head=lm_head,
        candidate_selector=selector,
        topk_gather=(lambda g, v, i: (g.all_gather(v, dim=-1), g.all_gather(i, dim=-1))) if packed else None,
        draft_config=types.SimpleNamespace(output_multiplier=1.0, final_logit_softcapping=None),
        block_size=BLOCK,
    )
    model._transform_unary_logits = lambda logits: DFlash2DraftModel._transform_unary_logits(model, logits)
    model.compute_candidates = lambda hidden: DFlash2DraftModel.compute_candidates(model, hidden)
    return model, group


def _sampler(model, fused: bool, max_bs: int):
    from sglang.srt.speculative.dflash_worker_v2 import _SelectorDraftSampler

    with envs.SGLANG_OPT_DFLASH_FUSED_SELECTOR.override(fused):
        sampler = _SelectorDraftSampler(
            draft_model=model, block_size=BLOCK, max_bs=max_bs, device=DEV, sampling_enabled=True
        )
    assert (sampler._fused_head is not None) == fused
    return sampler


def _outputs(sampler, bs: int):
    return (
        sampler.out[: bs * SLOTS].clone(),
        sampler.candidate_out[:bs].clone(),
        sampler.q_out[:bs].clone(),
        sampler.uniforms[:bs].clone(),
    )


def _poison(sampler) -> None:
    sampler.out.fill_(-1)
    sampler.candidate_out.fill_(-1)
    sampler.q_out.fill_(float("nan"))
    sampler.uniforms.fill_(float("nan"))


def _set_sampling(sampler, temps, greedy) -> None:
    bs = len(temps)
    sampler.temperatures[:bs].copy_(torch.tensor(temps, dtype=torch.float32))
    sampler.greedy_mask[:bs].copy_(torch.tensor(greedy, dtype=torch.bool))


_SAMPLING = {1: [((0.7,), (False,)), ((1.0,), (True,))], 2: [((1.3, 0.6), (False, False)), ((1.0, 0.9), (True, False))]}


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("bs", [1, 2])
def test_fused_head_matches_unfused_head(monkeypatch, bs, ties, packed, graph):
    """Catches a fused head that reorders candidates, drops the unary logit, walks the wrong lattice row, mixes up
    the predecessor slot, loses or bakes the in-graph draws, or misses a write (outputs are poisoned first)."""
    gen = torch.Generator(device=DEV).manual_seed(11 + bs + 2 * ties + 4 * packed)
    max_bs = 2
    model, group = _make_head(gen, ties, packed, max_bs)
    monkeypatch.setattr(
        "sglang.srt.models.dflash.get_parallel",
        lambda: types.SimpleNamespace(tp_group=group, attn_tp_group=group),
    )
    samplers = {fused: _sampler(model, fused, max_bs) for fused in (False, True)}
    hidden = torch.randn((bs * BLOCK, HIDDEN), generator=gen, device=DEV)
    hidden = (hidden.round() if ties else hidden).bfloat16()
    input_ids = torch.randint(0, VOCAB, (bs * BLOCK,), generator=gen, device=DEV)

    graphs = {}
    for fused, sampler in samplers.items():
        sampler(hidden, input_ids)  # compile / JIT outside capture
        torch.cuda.synchronize()
        if graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                sampler(hidden, input_ids)
            graphs[fused] = g

    seen_uniforms = []
    for temps, greedy in _SAMPLING[bs]:
        for seed in (101, 202):
            got = {}
            for fused, sampler in samplers.items():
                _set_sampling(sampler, temps, greedy)
                _poison(sampler)
                torch.manual_seed(seed)
                if graph:
                    graphs[fused].replay()
                else:
                    sampler(hidden, input_ids)
                torch.cuda.synchronize()
                got[fused] = _outputs(sampler, bs)
            for name, a, b in zip(("tokens", "candidates", "q_rows", "uniforms"), got[False], got[True]):
                _assert_bitwise(f"{name} (temps {temps}, greedy {greedy}, seed {seed})", b, a)
            assert not torch.isnan(got[True][2]).any() and bool((got[True][1] >= 0).all())
            seen_uniforms.append(got[True][3])
    # The draws are live in the replayed graph, not values baked at capture.
    assert not torch.equal(seen_uniforms[0], seen_uniforms[1])


def _fp64_walk(scores: torch.Tensor, uniforms, temps, greedy):
    """The selector walk of the model code (CandidateSelector.sample_path) in float64, one request at a time."""
    bs = scores.shape[0]
    path, q = [], []
    for b in range(bs):
        prev, toks, qs = 0, [], []
        for s in range(SLOTS):
            row = scores[b, s, prev]
            if greedy[b]:
                idx = int(torch.argmax(row))
                p = torch.zeros_like(row)
                p[idx] = 1.0
            else:
                p = torch.softmax(row / temps[b], dim=-1)
                idx = min(int((uniforms[b, s].double() >= p.cumsum(-1)).sum()), K - 1)
            toks.append(idx)
            qs.append(p)
            prev = idx
        path.append(toks)
        q.append(torch.stack(qs))
    return torch.tensor(path, device=scores.device), torch.stack(q)


def _near_tie(scores_row: torch.Tensor, prob_row: torch.Tensor, uniform: float, greedy: bool) -> bool:
    """Whether float32 rounding may legitimately pick another index than float64 at this slot."""
    if greedy:
        top2 = scores_row.topk(2).values
        return float(top2[0] - top2[1]) <= 1e-5 * max(1.0, abs(float(top2[0])))
    return bool(((uniform - prob_row.cumsum(-1)).abs() <= 1e-5).any())


@pytest.mark.parametrize("bs", [1, 2])
def test_fused_head_matches_fp64_model_walk(monkeypatch, bs):
    """Independent of both heads' selection and walk code: candidates from compute_candidates (torch.topk), the lattice
    operands from the model's indexing (bf16 where the model rounds), the lattice GEMM itself, then an FP64 walk.
    Tolerances fixed before data: q within 1e-5 absolute up to the first slot where the walks part; they may part only
    where float32 rounding can decide (a draw within 1e-5 of a CDF step, or a greedy top-2 gap within 1e-5 relative),
    in at most 1% of slots."""
    from sglang.srt.models.dflash import DFlash2DraftModel

    gen = torch.Generator(device=DEV).manual_seed(29 + bs)
    model, group = _make_head(gen, ties=False, packed=True, max_bs=2)
    monkeypatch.setattr(
        "sglang.srt.models.dflash.get_parallel",
        lambda: types.SimpleNamespace(tp_group=group, attn_tp_group=group),
    )
    sampler = _sampler(model, True, 2)
    hidden = torch.randn((bs * BLOCK, HIDDEN), generator=gen, device=DEV).bfloat16()
    input_ids = torch.randint(0, VOCAB, (bs * BLOCK,), generator=gen, device=DEV)
    temps = (0.8, 1.2)[:bs]
    greedy = (False, True)[:bs]

    selector = model.candidate_selector
    hs = hidden.view(bs, BLOCK, -1)[:, 1:, :]
    ref_ids, unary = DFlash2DraftModel.compute_candidates(model, hs.reshape(-1, HIDDEN))
    ref_ids = ref_ids.view(bs, SLOTS, K)
    h = selector.hidden_projection(hs)
    anchors = input_ids.view(bs, BLOCK)[:, 0]
    pred_ids = torch.cat([anchors[:, None, None].expand(-1, 1, K), ref_ids[:, :-1]], dim=1)
    pred_h = selector.predecessor_codebook[pred_ids] * h[:, :, None]
    keys = selector.successor_codebook[ref_ids]
    lattice = torch.einsum("blpr,blcr->blpc", pred_h, keys)
    scores = unary.view(bs, SLOTS, K).double()[:, :, None] + lattice.double()

    parted = slots_seen = 0
    for seed in range(20):
        _set_sampling(sampler, temps, greedy)
        _poison(sampler)
        torch.manual_seed(seed)
        sampler(hidden, input_ids)
        torch.cuda.synchronize()
        tokens, cand, q, uniforms = _outputs(sampler, bs)
        _assert_bitwise("candidates vs compute_candidates", cand, ref_ids)
        want_idx, want_q = _fp64_walk(scores, uniforms, temps, greedy)
        got_idx = (cand == tokens.view(bs, SLOTS, 1)).float().argmax(-1)
        for b in range(bs):
            prev = 0
            for s in range(SLOTS):
                slots_seen += 1
                err = float((q[b, s].double() - want_q[b, s]).abs().max())
                assert err <= 1e-5, f"seed {seed} request {b} slot {s}: q differs by {err}"
                if int(got_idx[b, s]) != int(want_idx[b, s]):
                    row = scores[b, s, prev]
                    assert _near_tie(row, want_q[b, s], float(uniforms[b, s]), greedy[b]), (
                        f"seed {seed} request {b} slot {s}: walk picks {int(got_idx[b, s])}, FP64 picks "
                        f"{int(want_idx[b, s])}, away from any rounding tie"
                    )
                    parted += 1
                    break
                prev = int(want_idx[b, s])
    assert parted <= 0.01 * slots_seen
