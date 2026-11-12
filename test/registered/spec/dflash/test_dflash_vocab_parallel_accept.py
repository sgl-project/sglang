# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_K31_VP_ACCEPT / _VP_SIM_ARGMAX: vocab-parallel sampled accept and target argmax vs the served path on
the gathered logits (one GPU; the TP shards are emulated by slicing the vocabulary and stacking the payloads, which is
the all-gather's [world, rows, P] layout).

References are independent of the code under test: torch.argmax over gathered_softcap_logits' output (the served
simulator input), the served selector_logit_accept on verify_row_stats (chain walk), and the exact joint law of
(num_correct_drafts, bonus) computed in FP64 from the logits (distribution).

Tolerances, stated before data:
- target argmax: bitwise, every row, ties included (lowest global index).
- chain walk: num_correct equal to the served kernel's on the same uniforms, except rows whose coin * q is within
  1e-5 relative of p (fp32 order of the softmax denominator); such boundary rows are counted and must be < 0.1 %.
- distribution (rewritten after the first case proved degenerate): three chi-square tests vs the exact
  FP64 law on a case with every num_correct >= 2 %: joint (num_correct, bonus) bins with expected count >= 10, the 8-bin
  num_correct marginal, and per-n candidate ids vs the rest; PASS if each p > 1e-4 and no impossible outcome. The served
  sampler on the same setup must pass too (control on the reference).
- cross-rank guard: with world 4 and ONE rank's input perturbed for ONE request (one ulp of the
  seed, q, coin or temperature; one candidate id; one draft), the mismatch counter must read exactly (1, 0) for the seed
  and (0, 1) for every other field; (0, 0) with no perturbation. Expected values follow from the construction.
- sync fold (SGLANG_OPT_DFLASH_SELECTOR_SYNC_FOLD on the VP path): the fp32 all-reduce input equals the accept result
  exactly (root) or zeros (other ranks), padding zeros; commit_from_fp32_sync on it equals the served
  commit_selector_block on the int64 result, every output, bit for bit.
K31_VP_MUT=<name> loads a mutated copy of the module (pattern replaced once) so the runner can show each test fails.
K31_VP_TEST_SPLIT (1 | auto | n) and K31_VP_TEST_LAYOUT (world | row) run every test on another payload form: column
chunks per shard (SGLANG_OPT_K31_VP_SPLIT) and the gathered layout (world-major = NCCL all_gather_into_tensor,
row-major with a padded row stride = the multimem all-gather's view). The tolerances above hold for every form.
"""

import importlib
import importlib.util
import math
import os
import tempfile
import types

import pytest
import torch

from sglang.kernels.ops.activation.softcap import gathered_softcap_logits
from sglang.kernels.ops.speculative.selector_sampled_accept import selector_logit_accept
from sglang.kernels.ops.speculative.verify_row_stats import verify_row_stats

_MOD = "sglang.kernels.ops.speculative.vocab_parallel_accept"
_MUTANTS = {
    # combine picks the HIGHEST id among equal maxima
    "tie": (
        "best_i = tl.min(tl.where(m2 & (av == best_v[None, :]), ai, _BIG), axis=0)",
        "best_i = tl.max(tl.where(m2 & (av == best_v[None, :]), ai, -1), axis=0)",
    ),
    # local argmax id not shifted by the shard offset
    "offset": (
        "tl.store(out_row + 1, (a_idx + col_offset).to(tl.float32))",
        "tl.store(out_row + 1, (a_idx).to(tl.float32))",
    ),
    # local pass skips the softcap
    "softcap": (
        "x = _softcap(tl.load(lg_row + cols, mask=mask, other=0.0).to(tl.float32), softcap, APPLY_SOFTCAP)",
        "x = tl.load(lg_row + cols, mask=mask, other=0.0).to(tl.float32)",
    ),
    # residual ignores q at the candidate ids
    "noq": ("    r = p_c - q_res\n", "    r = p_c + 0.0 * q_res\n"),
    # denominator sums the shards' s without rescaling to the global max
    "zscale": (
        "z = tl.sum(tl.where(mv, ss * tl.exp(mm - big_m[None, :]), 0.0), axis=0)",
        "z = tl.sum(tl.where(mv, ss, 0.0), axis=0)",
    ),
    # reads past the valid width into the padded columns
    "valid": ("        mask = cols < hi\n", "        mask = cols < hi + 8\n"),
    # local Gumbel pass lets candidate columns compete as non-candidates (weight p instead of relu(p - q))
    "noexcl": (
        "gs = tl.where(mask & ~is_cand, t + _gumbel(seed, j * VOCAB + gid), -float(\"inf\"))",
        "gs = tl.where(mask, t + _gumbel(seed, j * VOCAB + gid), -float(\"inf\"))",
    ),
    # non-candidate score without -log Z (weights p * Z vs the candidates' relu(p - q))
    "nolz": ("    nb = nb_raw - big_m - log_z  #", "    nb = nb_raw - big_m  #"),
    # residual drawn from the wrong row (row num_accept - 1)
    "rowoff": ("    sel = s == (num_accept + 1)\n", "    sel = s == num_accept\n"),
    # all-accepted requests take the residual branch instead of row gamma's full law
    "allres": (
        "token = tl.where(num_accept == NUM_SLOTS - 1, tok_all, tok_rej)",
        "token = tl.where(num_accept == NUM_SLOTS - 1, tok_rej, tok_rej)",
    ),
    # guard: seed field not written from the rank's own seed
    "noseed": ("    tl.store(out_row + 6, tl.load(ufin_ptr + req))\n", "    tl.store(out_row + 6, 0.0)\n"),
    # guard: the hash leaves out the draft
    "nodraft": ("    h = _fmix(h ^ _fmix(draft + 0x3B9ACA07))\n", "    h = _fmix(h ^ _fmix(0 + 0x3B9ACA07))\n"),
    # fold: the sync input carries the bonus token in the num_correct slot
    "foldswap": (
        "            tl.store(sync_ptr + pid * 2, num_accept.to(tl.float32))\n",
        "            tl.store(sync_ptr + pid * 2, token.to(tl.float32))\n",
    ),
    # fold: padding left unwritten
    "foldpad": (
        "            tl.store(sync_ptr + pad, tl.zeros((4,), dtype=tl.float32), mask=pad < sync_padded)\n",
        "            pass\n",
    ),
    # split: the combine reads every candidate logit from column chunk 0 of the owning rank
    "ownsub": (
        "    own_c = (cid - own_r * shard_width) // chunk\n",
        "    own_c = (cid - own_r * shard_width) // chunk * 0\n",
    ),
    # split: every chunk program runs to the end of the shard (chunks overlap: softmax mass counted twice)
    "subhi": ("    hi = tl.minimum(lo + chunk, valid_width)\n", "    hi = valid_width + 0 * lo\n"),
    # layout: row-major gathers read with the packed row stride instead of the view's (padded) stride
    "rowstride": (
        "        stride_rank, stride_row = splits * p, gathered.stride(0)\n",
        "        stride_rank, stride_row = splits * p, world * splits * p\n",
    ),
    # guard: the combine compares each rank with itself instead of rank 0
    "selfcmp": (
        "hash_bad = tl.max(tl.max(tl.where(m2 & (hs != hs0[None, :]), 1, 0), axis=1), axis=0)",
        "hash_bad = tl.max(tl.max(tl.where(m2 & (hs != hs), 1, 0), axis=1), axis=0)",
    ),
}


def _impl():
    name = os.environ.get("K31_VP_MUT", "")
    real = importlib.import_module(_MOD)
    if not name:
        return real
    old, new = _MUTANTS[name]
    src = open(real.__file__).read()
    assert src.count(old) == 1, (name, src.count(old))
    d = tempfile.mkdtemp(prefix=f"k31vpmut_{name}_")
    path = os.path.join(d, f"k31vpmut_{name}.py")
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    spec = importlib.util.spec_from_file_location(f"k31vpmut_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"K31_MUTATION {name} applied", flush=True)
    return mod


VP = _impl()
DEV = "cuda"
_SPLIT = os.environ.get("K31_VP_TEST_SPLIT", "1")
_LAYOUT = os.environ.get("K31_VP_TEST_LAYOUT", "world")
print(f"VP_TEST_FORM split={_SPLIT} layout={_LAYOUT}", flush=True)
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _case(*, bs, block, top_k, vocab, world, softcap, scale, seed, temp=1.0, q_scale=None):
    """Raw shard logits (padded tail = +1e4 so a valid-width bug shows), selector candidates / q, drafts (on the GPU)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    shard = -(-vocab // world)
    raw = torch.randn((bs * block, world * shard), generator=g, device=DEV) * scale
    raw[:, vocab:] = 1.0e4
    gamma = block - 1
    keys = torch.rand((bs * gamma, vocab), generator=g, device=DEV)
    cand_ids = keys.topk(top_k, dim=-1).indices.view(bs, gamma, top_k)  # distinct random ids per row
    del keys
    q = torch.rand((bs, gamma, top_k), generator=g, device=DEV)
    q = q / q.sum(-1, keepdim=True) * 0.95
    pick = torch.randint(0, top_k, (bs, gamma), generator=g, device=DEV)
    if q_scale is not None:
        q_scale(raw, cand_ids, q, pick, vocab, softcap, temp)
    drafts = cand_ids.gather(-1, pick[..., None]).squeeze(-1)
    candidates = torch.cat([torch.randint(0, vocab, (bs, 1), generator=g, device=DEV), drafts], 1)
    temps = torch.full((bs, 1), temp, device=DEV)
    return dict(
        raw=raw, cand_ids=cand_ids.contiguous(), q=q.contiguous(),
        candidates=candidates.contiguous(), temps=temps, shard=shard,
    )


def _served_logits(c, world, vocab, softcap):
    rows = c["raw"].shape[0]
    shards = c["raw"].view(rows, world, c["shard"]).permute(1, 0, 2).contiguous()
    out = torch.empty((rows, vocab), dtype=torch.float32, device=DEV)
    # the served launcher puts rows on grid Y (CUDA limit 65535): serving passes <= 512 rows, the distribution case
    # 160000, so the control calls the same kernel on row chunks (rows are independent).
    step = 32768
    for r0 in range(0, rows, step):
        r1 = min(rows, r0 + step)
        gathered_softcap_logits(shards[:, r0:r1].contiguous(), softcap, out[r0:r1])
    return out


def _vp(c, world, vocab, softcap, u, uf, rank_inputs=None, counter=None, sync_out=None, sync_values=True):
    """rank_inputs(w) -> dict overriding that rank's local inputs (keys of ``c`` plus "u" / "uf"); the combine
    always uses rank 0's (unperturbed) inputs, as the root of a real group would use its own."""
    bs, block = c["candidates"].shape
    rows = c["raw"].shape[0]
    if _SPLIT == "1":
        splits, chunk = 1, c["shard"]
    else:
        splits, chunk = VP.vp_split(rows, c["shard"], 0 if _SPLIT == "auto" else int(_SPLIT))
    pays = []
    for w in range(world):
        sl = c["raw"][:, w * c["shard"] : (w + 1) * c["shard"]].contiguous()
        off = w * c["shard"]
        x = dict(c, u=u, uf=uf)
        if rank_inputs is not None:
            x.update(rank_inputs(w))
        pays.append(
            VP.vp_local_payload(
                logits_shard=sl, temperatures=x["temps"], uniform_samples_final=x["uf"],
                candidate_ids=x["cand_ids"], rows_per_request=block,
                valid_width=max(0, min(c["shard"], vocab - off)), col_offset=off,
                vocab_size=vocab, softcap=softcap, candidates=x["candidates"], q_rows=x["q"],
                uniform_samples=x["u"], splits=splits, chunk=chunk,
            )
        )  # fmt: skip
    if _LAYOUT == "row":
        width = pays[0].shape[1]
        buf = torch.full((rows, world * width + 40), float("nan"), device=DEV)  # padded row stride, as the buffer
        for w in range(world):
            buf[:, w * width : (w + 1) * width] = pays[w]
        gathered, gw, gs = buf[:, : world * width], world, splits
    else:
        gathered, gw, gs = torch.stack(pays).contiguous(), None, None
    return VP.vp_combine(
        gathered=gathered, candidates=c["candidates"], candidate_ids=c["cand_ids"],
        q_rows=c["q"], uniform_samples=u, uniform_samples_final=uf, temperatures=c["temps"], vocab_size=vocab,
        shard_width=c["shard"], chunk=chunk, world=gw, splits=gs,
        mismatch_counter=counter, sync_out=sync_out, sync_values=sync_values,
    )  # fmt: skip


def _plant_ties(raw, vocab, shard, softcap):
    """Row 0: equal max at a low id (shard 0) and a high id (last shard). Row 1: equal max twice in one shard.
    Row 2: two raw values that saturate to the same softcapped value, the larger raw one at the HIGHER id."""
    raw[0, 100] = raw[0, vocab - 5] = 40.0
    base = shard if shard + 9 < vocab else 0  # shard 1 when there is one
    raw[1, base + 7] = raw[1, base + 9] = 41.0
    if softcap:
        raw[2, vocab - 3] = 30.0 * softcap
        raw[2, 11] = 20.0 * softcap


@needs_gpu
@pytest.mark.parametrize("world", [1, 2, 4, 16])
@pytest.mark.parametrize("softcap", [30.0, None])
@pytest.mark.parametrize("vocab", [163840, 163840 - 37])
def test_target_argmax_bitwise(world, softcap, vocab):
    bs, block = 4, 8
    c = _case(bs=bs, block=block, top_k=16, vocab=vocab, world=world, softcap=softcap, scale=3.0, seed=world)
    _plant_ties(c["raw"], vocab, c["shard"], softcap)
    full = _served_logits(c, world, vocab, softcap)
    ref = torch.argmax(full, dim=-1).view(bs, block)
    if softcap:
        assert full[2, vocab - 3] == full[2, 11], "planted saturation tie did not tie"
    u = torch.rand((bs, block - 1), device=DEV)
    uf = torch.rand((bs,), device=DEV)
    _, tp = _vp(c, world, vocab, softcap, u, uf)
    bad = (tp != ref).sum().item()
    print(f"VP_ARGMAX world={world} softcap={softcap} vocab={vocab} rows={ref.numel()} mismatches={bad}")
    assert bad == 0


def _p_scale(raw, cand_ids, q, pick, vocab, softcap, temp):
    # Each draft gets real target mass (raw logit = row max + U[-2, 2)); a random id at scale 4 over 32K ids has p far
    # below the 1e-6 q floor, which made acceptance degenerate in an earlier run. Then q of each draft = its
    # exact target p times a factor in [0.5, 2): per-slot accept E[min(1, 1/f)] ~ 0.80, boundaries hit often.
    bs, gamma, top_k = cand_ids.shape
    rows = (torch.arange(bs, device=DEV)[:, None] * (gamma + 1) + torch.arange(gamma, device=DEV)[None, :]).reshape(-1)
    d = cand_ids.gather(-1, pick[..., None]).reshape(-1)
    g = torch.Generator(device=DEV).manual_seed(5)
    raw[rows, d] = raw[rows, :vocab].max(-1).values + 4.0 * torch.rand(rows.shape, generator=g, device=DEV) - 2.0
    x = raw[:, :vocab].double()
    if softcap:
        x = softcap * torch.tanh(x / softcap)
    p = torch.softmax(x / temp, -1)
    pd = p[rows, d].view(bs, gamma)
    f = 0.5 + 1.5 * torch.rand((bs, gamma), dtype=torch.float64, device=DEV)
    q.scatter_(-1, pick[..., None], (pd * f).clamp(1e-6, 1.0).float()[..., None])


@needs_gpu
@pytest.mark.parametrize("world,temp", [(16, 1.0), (4, 0.7)])
def test_chain_walk_matches_served(world, temp):
    bs, block, vocab, softcap = 1024, 8, 32768, 30.0
    c = _case(bs=bs, block=block, top_k=16, vocab=vocab, world=world, softcap=softcap, scale=4.0, seed=7,
              temp=temp, q_scale=_p_scale)  # fmt: skip
    full = _served_logits(c, world, vocab, softcap)
    lg, stats = verify_row_stats(logits=full, temperatures=c["temps"], rows_per_request=block)
    u = torch.rand((bs, block - 1), device=DEV)
    uf = torch.rand((bs,), device=DEV)
    served = selector_logit_accept(
        candidates=c["candidates"], target_logits=lg, logit_stats=stats, candidate_ids=c["cand_ids"],
        q_rows=c["q"], uniform_samples=u, uniform_samples_final=uf,
    )  # fmt: skip
    res, _ = _vp(c, world, vocab, softcap, u, uf)
    n_s, n_v = served[:, 0], res[:, 0]
    diff = (n_s != n_v).nonzero().flatten()
    # boundary test: at the first differing slot, |coin * q - p| <= 1e-5 * p (p from FP64)
    x = softcap * torch.tanh(c["raw"][:, :vocab].double() / softcap)
    p_all = torch.softmax(x / temp, -1)
    boundary = 0
    for b in diff.tolist():
        s = int(min(n_s[b], n_v[b])) + 1
        d = int(c["candidates"][b, s])
        pd = float(p_all[b * block + s - 1, d])
        k = (c["cand_ids"][b, s - 1] == d).nonzero().flatten()
        qd = float(c["q"][b, s - 1, k[0]]) if len(k) else 0.0
        if abs(float(u[b, s - 1]) * qd - pd) <= 1e-5 * pd:
            boundary += 1
    hard = len(diff) - boundary
    acc_mean = n_s.float().mean().item()
    print(f"VP_WALK world={world} temp={temp} reqs={bs} mean_accept={acc_mean:.3f} diff={len(diff)} boundary={boundary} hard={hard}")
    assert 0.5 < acc_mean < block - 1.5, "acceptance degenerate: the case does not exercise the walk"
    assert hard == 0 and boundary <= bs // 1000


def _exact_joint(c, vocab, softcap, temp, block):
    """FP64 law of (num_correct, bonus) for request 0 (all requests in the distribution case are copies)."""
    x = c["raw"][:block, :vocab].double()
    if softcap:
        x = softcap * torch.tanh(x / softcap)
    p = torch.softmax(x / temp, -1)
    gamma = block - 1
    cid, q, cand = c["cand_ids"][0].cpu(), c["q"][0].double().cpu(), c["candidates"][0].cpu()
    p = p.cpu()
    acc = []
    for s in range(1, block):
        d = int(cand[s])
        m = (cid[s - 1] == d).nonzero().flatten()
        qd = float(q[s - 1, m[0]]) if len(m) else 0.0
        acc.append(min(1.0, float(p[s - 1, d]) / qd) if 0.0 < qd <= 1.0 else 0.0)
    law = {}
    reach = 1.0
    for n in range(block):
        if n < gamma:
            pn = reach * (1.0 - acc[n])
            r = p[n].clone()
            qd = torch.zeros(vocab, dtype=torch.float64)
            qq = q[n].clone()
            qq[(qq < 0) | (qq > 1)] = 0.0
            qd[cid[n]] = qq
            r = (r - qd).clamp_min(0.0)
            reach *= acc[n]
        else:
            pn, r = reach, p[n].clone()
        if pn > 0:
            law[n] = (pn, r / r.sum())
    return law


def _chi2_pvalue(chi2, dof):
    # Wilson-Hilferty normal approximation of the chi-square upper tail.
    z = ((chi2 / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / math.sqrt(2.0 / (9.0 * dof))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _joint_test(law, n_acc, tok, total):
    obs = {}
    for n, t in zip(n_acc.tolist(), tok.tolist()):
        obs[(n, t)] = obs.get((n, t), 0) + 1
    chi2, bins = 0.0, 0
    for n, (pn, r) in law.items():
        e = pn * r * total
        keep = (e >= 10).nonzero().flatten().tolist()
        pooled_e = float(e.sum()) - float(e[keep].sum()) if keep else float(e.sum())
        pooled_o = sum(v for (nn, t), v in obs.items() if nn == n) - sum(obs.get((n, t), 0) for t in keep)
        for t in keep:
            o = obs.get((n, t), 0)
            chi2 += (o - float(e[t])) ** 2 / float(e[t])
            bins += 1
        if pooled_e >= 10:
            chi2 += (pooled_o - pooled_e) ** 2 / pooled_e
            bins += 1
    impossible = sum(v for (n, t), v in obs.items() if n not in law or float(law[n][1][t]) == 0.0)
    return _chi2_pvalue(chi2, max(1, bins - 1)), chi2, bins, impossible


def _chi2_sf(chi2, dof):
    """Upper tail of chi-square: exact series for dof <= 200, Wilson-Hilferty above."""
    if dof > 200:
        return _chi2_pvalue(chi2, dof)
    m = chi2 / 2.0
    if dof % 2 == 0:
        term, s = 1.0, 1.0
        for i in range(1, dof // 2):
            term *= m / i
            s += term
        return math.exp(-m) * s
    s = math.erfc(math.sqrt(m))
    for i in range(1, (dof - 1) // 2 + 1):
        s += math.exp(-m + (i - 0.5) * math.log(m) - math.lgamma(i + 0.5)) if m > 0 else 0.0
    return s


# Distribution case: every n in 0..7 has mass >= 2 %, P(n = 7) >= 5 %; slot s's draft has target p _P_D[s]
# and q = p / _A_D[s] (accept probability _A_D[s]); the slot's next candidate p 0.10 with q 0.05 (residual > 0), the next
# q = p + 0.05 (residual 0), the last q 0.02. Z ~ 1 / p_max ~ 3 so a missing -log Z moves the non-candidate weights.
_P_D = [0.30, 0.25, 0.35, 0.30, 0.20, 0.40, 0.30]
_A_D = [0.65, 0.70, 0.60, 0.70, 0.65, 0.60, 0.70]


def _dist_case(softcap, scale):
    def shape(raw, cand_ids, q, pick, vocab, sc, temp):
        gamma = cand_ids.shape[1]
        block = gamma + 1
        raw[:] = raw[:block].repeat(raw.shape[0] // block, 1)
        cand_ids[:] = cand_ids[:1]
        pick[:] = pick[:1]

        def xs(v):
            return sc * torch.tanh(v / sc) if sc else v

        for s in range(gamma):
            ids = cand_ids[0, s].tolist()
            k0 = int(pick[0, s])
            order = [ids[(k0 + j) % len(ids)] for j in range(len(ids))]  # draft first
            tgt = {order[0]: _P_D[s], order[1]: 0.10}
            x = xs(raw[s, :vocab].double())
            rest = torch.ones(vocab, dtype=torch.bool, device=raw.device)
            rest[list(tgt)] = False
            s_rest = float(torch.exp(x[rest]).sum())
            tot = sum(tgt.values())
            for t_id, t_p in tgt.items():
                xv = math.log(t_p * s_rest / (1.0 - tot))
                raw[s, t_id] = (sc * math.atanh(xv / sc)) if sc else xv
        raw[:] = raw[:block].repeat(raw.shape[0] // block, 1)
        x = xs(raw[:block, :vocab].double())
        p = torch.softmax(x / temp, -1)
        for s in range(gamma):
            ids = cand_ids[0, s].tolist()
            k0 = int(pick[0, s])
            for j in range(len(ids)):
                k = (k0 + j) % len(ids)
                pc = float(p[s, ids[k]])
                qv = [pc / _A_D[s], 0.05, min(1.0, pc + 0.05), 0.02][j] if j < 4 else 0.01
                q[0, s, k] = qv
        q[:] = q[:1]

    return shape


def _marginal_test(law, n_acc, total, block):
    obs = torch.bincount(n_acc, minlength=block).tolist()
    chi2, bins, impossible = 0.0, 0, 0
    for n in range(block):
        e = law[n][0] * total if n in law else 0.0
        if e >= 10:
            chi2 += (obs[n] - e) ** 2 / e
            bins += 1
        elif e == 0.0:
            impossible += obs[n]
    return _chi2_sf(chi2, max(1, bins - 1)), chi2, bins, impossible


def _split_test(law, n_acc, tok, cand0, total, block):
    """Per n: the residual's candidate ids (top-4 of the full row for n = gamma) vs the rest."""
    chi2, bins = 0.0, 0
    for n, (pn, r) in law.items():
        ids = cand0[n].tolist() if n < block - 1 else r.topk(4).indices.tolist()
        sel = n_acc == n
        tn = tok[sel]
        rest_e, rest_o = pn * total, int(sel.sum())
        for i in ids:
            e = pn * float(r[i]) * total
            o = int((tn == i).sum())
            rest_e -= e
            rest_o -= o
            if e >= 10:
                chi2 += (o - e) ** 2 / e
                bins += 1
            else:
                rest_e += e
                rest_o += o
        if rest_e >= 10:
            chi2 += (rest_o - rest_e) ** 2 / rest_e
            bins += 1
    return _chi2_sf(chi2, max(1, bins - 1)), chi2, bins


@needs_gpu
@pytest.mark.parametrize("sampler,vocab", [("vp", 1024), ("vp", 1024 - 37), ("served", 1024)])
def test_distribution_matches_exact(sampler, vocab):
    # served control only at vocab % 4 == 0 (verify_row_stats); rows 64000 per call (< 65535).
    block, top_k, world, softcap, temp = 8, 4, 4, 30.0, 1.0
    reqs, calls = 8000, 50
    c = _case(bs=reqs, block=block, top_k=top_k, vocab=vocab, world=world, softcap=softcap, scale=1.0, seed=3,
              temp=temp, q_scale=_dist_case(softcap, 1.0))  # fmt: skip
    c["candidates"][:] = c["candidates"][:1]
    law = _exact_joint(c, vocab, softcap, temp, block)
    assert min(law[n][0] for n in range(block)) >= 0.02 and law[block - 1][0] >= 0.05, "case lost its power"
    full = _served_logits(c, world, vocab, softcap) if sampler == "served" else None
    if full is not None:
        lg, stats = verify_row_stats(logits=full, temperatures=c["temps"], rows_per_request=block)
    gen = torch.Generator(device=DEV).manual_seed(11)
    ns, ts = [], []
    for _ in range(calls):
        u = torch.rand((reqs, block - 1), device=DEV, generator=gen)
        uf = torch.rand((reqs,), device=DEV, generator=gen)
        if sampler == "vp":
            res, _ = _vp(c, world, vocab, softcap, u, uf)
        else:
            res = selector_logit_accept(
                candidates=c["candidates"], target_logits=lg, logit_stats=stats, candidate_ids=c["cand_ids"],
                q_rows=c["q"], uniform_samples=u, uniform_samples_final=uf,
            )  # fmt: skip
        ns.append(res[:, 0].cpu())
        ts.append(res[:, 1].cpu())
    n_acc, tok = torch.cat(ns), torch.cat(ts)
    total = reqs * calls
    pj, chi2, bins, impossible = _joint_test(law, n_acc, tok, total)
    pm, chi2m, binsm, imp_m = _marginal_test(law, n_acc, total, block)
    ps, chi2s, binss = _split_test(law, n_acc, tok, c["cand_ids"][0].cpu(), total, block)
    accept_law = " ".join(f"{n}:{law[n][0]:.4f}" for n in sorted(law))
    emp = " ".join(f"{n}:{(n_acc == n).float().mean().item():.4f}" for n in sorted(law))
    print(f"VP_DIST sampler={sampler} vocab={vocab} N={total} joint bins={bins} chi2={chi2:.1f} p={pj:.3g} impossible={impossible}")
    print(f"VP_DIST_MARGINAL sampler={sampler} vocab={vocab} bins={binsm} chi2={chi2m:.1f} p={pm:.3g} impossible={imp_m}")
    print(f"VP_DIST_SPLIT sampler={sampler} vocab={vocab} bins={binss} chi2={chi2s:.1f} p={ps:.3g}")
    print(f"  accept exact {accept_law}\n  accept emp   {emp}")
    reasons = [r for r, bad in (("p_joint", pj <= 1e-4), ("p_marginal", pm <= 1e-4), ("p_split", ps <= 1e-4),
                                ("impossible", impossible + imp_m > 0)) if bad]  # fmt: skip
    print(f"VP_DIST_VERDICT sampler={sampler} vocab={vocab} reasons={','.join(reasons) or 'none'}")
    assert not reasons, reasons


_GATE_MUTANTS = {
    # gate forgets the logit bias (indentation as inspect.getsource + dedent gives it)
    "nobias": ('        and getattr(sampling_info, "logit_bias", None) is None\n', ""),
    # gate forgets a grammar batch
    "nogrammar": ("    if batch.has_grammar:\n        return False\n", ""),
}


class _SI(types.SimpleNamespace):
    def __len__(self):
        return self.bs

    def apply_logits_bias(self, t):
        t.add_(self.dense)


def test_sim_argmax_gate_covers_every_logit_edit():
    """The sim-argmax shortcut reads the raw logits, so it must be off whenever
    apply_dflash_verify_logits_adjustments (the independent oracle here) or the grammar mask edits them."""
    from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    gate = DFlashWorkerV2._vp_sim_argmax_ok
    mut = os.environ.get("K31_GATE_MUT", "")
    if mut:
        import inspect
        import sys
        import textwrap

        g_old, g_new = _GATE_MUTANTS[mut]
        src = textwrap.dedent(inspect.getsource(DFlashWorkerV2.__dict__["_vp_sim_argmax_ok"].__func__))
        assert src.count(g_old) == 1, (mut, src.count(g_old))
        src = src.replace(g_old, g_new).replace("@staticmethod\n", "")
        scope = dict(vars(sys.modules[DFlashWorkerV2.__module__]))
        exec(src, scope)
        gate = scope["_vp_sim_argmax_ok"]
        print(f"K31_MUTATION gate_{mut} applied", flush=True)
    bs, block, vocab = 2, 8, 64
    bump = torch.zeros((bs, vocab))
    bump[:, 5] = 1e3

    def si(**kw):
        base = dict(bs=bs, has_custom_logit_processor=False, acc_linear_penalties=None, penalizer_orchestrator=None,
                    grammar_mask=None, logit_bias=None, sampling_mask_batch_indices=None, dense=bump)  # fmt: skip
        base.update(kw)
        return _SI(**base)

    req = types.SimpleNamespace(is_required=True)
    edit_cases = {
        "acc_linear_penalties": si(acc_linear_penalties=bump.clone()),
        "logit_bias": si(logit_bias=bump.clone()),
        "penalizer_required": si(penalizer_orchestrator=req),
        "grammar_mask": si(grammar_mask=torch.ones((bs, vocab), dtype=torch.bool)),
    }
    plain_cases = {"none": si(), "penalizer_not_required": si(penalizer_orchestrator=types.SimpleNamespace(is_required=False))}
    nb = types.SimpleNamespace(has_grammar=False)
    for name, s in list(edit_cases.items()) + list(plain_cases.items()):
        lg = torch.randn((bs * block, vocab))
        lg[:, 0] = 10.0
        ref = lg.clone()
        apply_dflash_verify_logits_adjustments(next_token_logits=lg, sampling_info=s, draft_token_num=block)
        moved = bool((lg.argmax(-1) != ref.argmax(-1)).any())
        got = gate(nb, s)
        print(f"VP_GATE case={name} argmax_moved={moved} gate={got}")
        assert moved == (name in edit_cases), f"oracle setup: {name}"
        assert got == (not moved), name
    # edits the oracle cannot run here: custom processors, sampling masks, a grammar batch
    for name, s, b in (("custom_processor", si(has_custom_logit_processor=True), nb),
                       ("sampling_mask", si(sampling_mask_batch_indices=torch.tensor([0])), nb),
                       ("batch_grammar", si(), types.SimpleNamespace(has_grammar=True))):  # fmt: skip
        got = gate(b, s)
        print(f"VP_GATE case={name} gate={got}")
        assert got is False, name


def _ulp_up(t):
    return torch.nextafter(t, torch.full_like(t, float("inf")))


@needs_gpu
@pytest.mark.parametrize("field", ["none", "seed", "cid", "q", "draft", "coin", "temp"])
def test_cross_rank_drift_is_counted(field):
    """Cross-rank guard: rank 2 of 4 alone perturbs request 3 (bs 8, block 8, top_k 16). Want (1, 0) for the seed, (0, 1) for the
    other fields, (0, 0) for none; and the accept result with no perturbation is unchanged by the counter."""
    world, vocab, softcap, bs, block, r, bad_rank = 4, 4096, 30.0, 8, 8, 3, 2
    c = _case(bs=bs, block=block, top_k=16, vocab=vocab, world=world, softcap=softcap, scale=3.0, seed=11)
    g = torch.Generator(device=DEV).manual_seed(12)
    u = torch.rand((bs, block - 1), generator=g, device=DEV)
    uf = torch.rand((bs,), generator=g, device=DEV)

    def perturbed(w):
        if w != bad_rank or field == "none":
            return {}
        if field == "seed":
            x = uf.clone(); x[r] = _ulp_up(x[r]); return {"uf": x}
        if field == "coin":
            x = u.clone(); x[r, 4] = _ulp_up(x[r, 4]); return {"u": x}
        if field == "temp":
            x = c["temps"].clone(); x[r, 0] = _ulp_up(x[r, 0]); return {"temps": x}
        if field == "q":
            x = c["q"].clone(); x[r, 2, 5] = _ulp_up(x[r, 2, 5]); return {"q": x}
        if field == "cid":
            x = c["cand_ids"].clone(); x[r, 2, 5] = (x[r, 2, 5] + 1) % vocab; return {"cand_ids": x.contiguous()}
        x = c["candidates"].clone(); x[r, 3] = (x[r, 3] + 1) % vocab; return {"candidates": x.contiguous()}

    cnt = torch.zeros((2,), dtype=torch.int32, device=DEV)
    res, tp = _vp(c, world, vocab, softcap, u, uf, rank_inputs=perturbed, counter=cnt)
    want = {"none": [0, 0], "seed": [1, 0]}.get(field, [0, 1])
    got = cnt.tolist()
    print(f"VP_GUARD field={field} counter={got} want={want}")
    assert got == want
    if field == "none":
        res0, tp0 = _vp(c, world, vocab, softcap, u, uf)
        assert torch.equal(res, res0) and torch.equal(tp, tp0)


@needs_gpu
@pytest.mark.parametrize("values", [True, False])
def test_sync_fold_matches_served_commit(values):
    """Fold: bs 5 (10 values padded to 12). Root: sync input == result exactly, pad 0, and the fp32 commit == the served
    commit_selector_block on the int64 result. Non-root: all zeros (the all-reduce sum is then the root's values)."""
    from sglang.kernels.ops.speculative.selector_sampled_accept import commit_selector_block

    world, vocab, softcap, bs, block = 4, 4096, 30.0, 5, 8
    c = _case(bs=bs, block=block, top_k=16, vocab=vocab, world=world, softcap=softcap, scale=3.0, seed=21)
    g = torch.Generator(device=DEV).manual_seed(22)
    u = torch.rand((bs, block - 1), generator=g, device=DEV)
    uf = torch.rand((bs,), generator=g, device=DEV)
    sync = torch.full((12,), float("nan"), device=DEV)
    res, _ = _vp(c, world, vocab, softcap, u, uf, sync_out=sync, sync_values=values)
    pad_ok = bool(torch.equal(sync[10:], torch.zeros(2, device=DEV)))
    if values:
        vals_ok = bool(torch.equal(sync[:10], res.reshape(-1).float()))
        prefix = torch.arange(100, 100 + bs, device=DEV, dtype=torch.int64)
        got = VP.commit_from_fp32_sync(candidates=c["candidates"], summed=sync, prefix_lens=prefix)
        ref = commit_selector_block(candidates=c["candidates"], accept_result=res.contiguous(), prefix_lens=prefix)
        commit_ok = all(torch.equal(a, b) for a, b in zip(got, ref))
    else:
        vals_ok = bool(torch.equal(sync[:10], torch.zeros(10, device=DEV)))
        commit_ok = True
    print(f"VP_FOLD values={values} vals_ok={vals_ok} pad_ok={pad_ok} commit_ok={commit_ok} res={res.tolist()}")
    assert vals_ok and pad_ok and commit_ok
