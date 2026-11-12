# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Vocab-parallel shard deferral for 16-bit logits without softcap (1 GPU, TP16 emulated).

The served K3.1 LM head emits BF16 logits and the model has no final softcap, so the fused gather + softcap path (the
only one that used to defer the target-verify gather) never applies and SGLANG_OPT_K31_VP_ACCEPT / _VP_SIM_ARGMAX never
engaged in situ. With ``defer_verify_gather_any`` (set by the DFLASH worker when a VP knob is on) the logits processor
returns the raw BF16 vocab shard on target verify. Checks, every one against an independent reference
R = concat over the 16 ranks of (hidden @ W_r^T in BF16), widened to FP32 (the served gather + cast):

  c1  deferral ON  : target verify returns this rank's BF16 shard; finish_deferred_gather(shard) == R bit for bit.
  c2  deferral OFF : the served path (fused deferral only) returns full FP32 logits == R bit for bit (control).
  c3  deferred_argmax(shard) == torch.argmax(R) on planted cross-rank and in-shard ties at realistic logit scale.
  c4  vp_local_payload(BF16 shard) vs vp_local_payload(shard widened to FP32) on all 16 shards: every payload
      field bit for bit except field 3 (s, an FP32 reduction whose order follows the per-dtype load layout), and s
      of both within S_RTOL (relative) of an FP64 reference (the FP32 VP path is checked against the served accept
      in test_dflash_vocab_parallel_accept.py). An earlier all-fields-bitwise form of c4 failed (field not named).
  c4x control for c4's exception: on a shard whose sum is exact in any order, the whole payload is bitwise.
  c5  no deferral on decode, or with defer_verify_gather False (the knob-off case is c2).

Mutants (K31_VPD_MUT, applied by patching the code under test in this process; each must make the run fail):
  dtype   - _can_defer_verify_shard accepts FP32 only (the pre-fix behaviour)          -> c1 fails
  anymode - the deferral ignores the forward mode                                      -> c5 fails
  capone  - softcap_row_argmax always applies a softcap of 1.0 (APPLY_SOFTCAP ignored) -> c3 fails (tanh saturates)
  sens    - c4 compares against a shard with one element moved by 1 BF16 ulp           -> c4 fails (sensitivity control)
  sref    - c4's FP64 reference for s drops the row max's term                         -> c4 fails (tolerance control)
Prints 'VPD_CHECK <name> ok=0|1' per check and 'VPD_DONE fails=N'; exit 1 when any check fails.
"""

import contextlib
import os
import sys
import types

import torch

import sglang.srt.layers.logits_processor as lp_mod
from sglang.kernels.ops.sampling import softcap_row_argmax as sra_mod
from sglang.kernels.ops.speculative.vocab_parallel_accept import vp_local_payload

MUT = os.environ.get("K31_VPD_MUT", "")
WORLD, VOCAB, HIDDEN = 16, 163840, 512  # served K3.1 text_config: vocab 163840, no softcap / LM-head FP32 keys
SHARD = VOCAB // WORLD
# s = sum of <= 10240 positive FP32 terms (tree sum, 5 online-rescaled blocks; ex2.approx per term): error bound
# ~ (11 + 5 + 4) * 2^-24 ~ 1.2e-6 relative; 1e-5 leaves ~8x margin. A wrong term (sref) moves s by >= 1 / s ~ 0.1.
S_RTOL = 1e-5
DEV = torch.device("cuda")
FAILS = 0


def check(name, ok, extra=""):
    global FAILS
    FAILS += 0 if ok else 1
    print(f"VPD_CHECK {name} ok={int(bool(ok))} {extra}", flush=True)


class FakeGroup:
    """A TP group of WORLD ranks emulated on one GPU: rank ``rank`` contributes ``inp``; the other ranks contribute
    ``provider(k, inp)`` (their own value of the same quantity)."""

    def __init__(self, rank):
        self.world_size, self.rank_in_group, self.provider, self.cpu_group = WORLD, rank, None, None

    def all_gather_into_tensor(self, out, inp):
        parts = [inp if k == self.rank_in_group else self.provider(k, inp) for k in range(WORLD)]
        out.copy_(torch.stack(parts).view_as(out))


class FakeGatherer:
    """The served non-fused gather: every rank's shard concatenated along the vocabulary."""

    def __init__(self, *a, **k):
        self.group = None

    def __call__(self, logits):
        return torch.cat([logits if k == self.group.rank_in_group else self.group.provider(k, logits)
                          for k in range(WORLD)], dim=-1)  # fmt: skip


def install_stubs(group):
    par = types.SimpleNamespace(
        enable_dp_lm_head=False, enable_tp_lm_head_all_to_all=False, tp_size=WORLD, attn_dp_size=1,
        attn_tp_size=WORLD, tp_group=group, attn_tp_group=group, attn_dp_rank=0, tp_rank=group.rank_in_group,
    )  # fmt: skip
    feats = types.SimpleNamespace(enable_fp32_lm_head=False, enable_mis=False, aux_hidden_state_capture=None,
                                  return_aux_hidden_state_layer_ids=None)  # fmt: skip
    ex = types.SimpleNamespace(features=feats, deterministic=types.SimpleNamespace(rl_on_policy_target=None))
    fwd = types.SimpleNamespace(scoped=lambda **k: contextlib.nullcontext())
    lp_mod.get_parallel = lambda: par
    lp_mod.get_exec = lambda: ex
    lp_mod.get_forward = lambda: fwd
    lp_mod.triton_symm_mem_ag = types.SimpleNamespace(MultimemAllGatherer=FakeGatherer,
                                                     recommended_max_tokens=lambda **k: 128)  # fmt: skip
    lp_mod.InputLogprobProcessor = lambda *a, **k: None


def make_processor(group, defer, defer_any):
    install_stubs(group)
    cfg = types.SimpleNamespace(vocab_size=VOCAB)  # no final_logit_softcapping, no enable_lm_head(_output)_fp32
    proc = lp_mod.LogitsProcessor(cfg, logit_scale=1.0)  # kimi_k3.py: logit_scale = getattr(config, 'logit_scale', 1.0)
    proc._logits_gatherer.group = group
    proc.defer_verify_gather, proc.defer_verify_gather_any = defer, defer_any
    return proc


def metadata(verify):
    mode = types.SimpleNamespace(is_target_verify=lambda: verify)
    return types.SimpleNamespace(forward_mode=mode, next_token_logits_buffer=None,
                                 global_num_tokens_for_logprob_cpu=None, global_num_tokens_for_logprob_gpu=None)  # fmt: skip


def main():
    torch.manual_seed(1234)
    if MUT == "dtype":
        orig = lp_mod.LogitsProcessor._can_defer_verify_shard
        lp_mod.LogitsProcessor._can_defer_verify_shard = lambda self, lg, *a: lg.dtype == torch.float32 and orig(self, lg, *a)
    if MUT == "anymode":
        orig_get = lp_mod.LogitsProcessor._get_logits

        def patched(self, hidden, lm_head, md, *a, **k):
            return orig_get(self, hidden, lm_head, metadata(True), *a, **k)

        lp_mod.LogitsProcessor._get_logits = patched
    if MUT == "capone":
        orig_sra = sra_mod.softcap_row_argmax
        sra_mod.softcap_row_argmax = lambda lg, cap, vw, off: orig_sra(lg, cap if cap else 1.0, vw, off)
        lp_mod.softcap_row_argmax = sra_mod.softcap_row_argmax
    print(f"K31_MUTATION vpd {MUT or 'none'} applied", flush=True)

    # Logit scale: bulk std ~4 (sqrt(512) * 0.18), planted tops at 38, so a softcap of 1.0 (mutant capone) would
    # saturate hundreds of entries per row to tanh = 1.0 and move the first argmax.
    weights = [(torch.randn(SHARD, HIDDEN, device=DEV) * 0.18).to(torch.bfloat16) for _ in range(WORLD)]
    for rows in (8, 64):  # Breq 1 and 8 at Q8
        hidden = torch.randn(rows, HIDDEN, device=DEV).to(torch.bfloat16)
        base = [hidden @ w.T for w in weights]  # BF16, as the served LM head emits them (lm_head = torch.matmul)
        shards = [b.clone() for b in base]
        # Plant ties: row 0 has the same top value at global ids 3 * SHARD + 5 and 9 * SHARD + 7 (cross-rank);
        # row 1 at local columns 11 and 4000 of rank 6 (in-shard). The lowest global id must win.
        top = torch.tensor(38.0, dtype=torch.bfloat16, device=DEV)
        shards[3][0, 5] = top
        shards[9][0, 7] = top
        shards[6][1, 11] = top
        shards[6][1, 4000] = top
        ref = torch.cat(shards, dim=-1).float()  # served gather (concat) + exact cast
        ref_argmax = torch.argmax(ref, dim=-1)
        for rank in (0, 6, 15):
            group = FakeGroup(rank)
            lm_head = types.SimpleNamespace(weight=weights[rank])
            md = metadata(True)

            # Every rank's shard is the same matmul the LM head runs (bitwise: same kernel, same inputs).
            def shard_of(k, inp):
                return shards[k] if inp.dim() == 2 and inp.shape[-1] == SHARD and inp.dtype == torch.bfloat16 else None

            group.provider = shard_of
            # c1: deferral on (VP knob) -> BF16 shard; finished gather == R.
            proc = make_processor(group, True, True)
            out = proc._get_logits(hidden, lm_head, md, use_logits_buffer=False)
            is_shard = out.shape == (rows, SHARD) and out.dtype == torch.bfloat16 and proc.is_deferred_shard(out)
            same_mm = is_shard and torch.equal(out, base[rank])  # the returned shard is the LM head's own output
            if same_mm:
                out.copy_(shards[rank])  # add the planted ties to this rank's shard (the providers carry them)
                full = proc.finish_deferred_gather(out)
                c1 = full.dtype == torch.float32 and torch.equal(full, ref)
            else:
                c1 = False
            check(f"c1_shard_and_finish rows={rows} rank={rank}", c1, f"out={tuple(out.shape)} {out.dtype} same_mm={int(bool(same_mm))}")

            # c3: deferred_argmax on the shard (ties included) == torch.argmax(R).
            if same_mm:
                def local_argmax(k, inp):
                    off = k * SHARD
                    return lp_mod.softcap_row_argmax(shards[k], None, min(SHARD, VOCAB - off), off)

                group.provider = local_argmax
                got = proc.deferred_argmax(out)
                group.provider = shard_of
                ok3 = torch.equal(got.cpu(), ref_argmax.cpu())
                check(f"c3_deferred_argmax rows={rows} rank={rank}", ok3,
                      f"row0={int(got[0])}/{int(ref_argmax[0])} row1={int(got[1])}/{int(ref_argmax[1])}")  # fmt: skip
            else:
                check(f"c3_deferred_argmax rows={rows} rank={rank}", False, "no shard")

            # c2: served path (fused deferral only, knob off) -> full FP32 == R.
            proc2 = make_processor(group, True, False)
            out2 = proc2._get_logits(hidden, lm_head, md, use_logits_buffer=False)
            # the served gather reads this rank's matmul output; the planted entries come from the providers
            ok2 = out2.shape == (rows, VOCAB) and out2.dtype == torch.float32
            if ok2:
                own_cols = slice(rank * SHARD, (rank + 1) * SHARD)
                ref2 = ref.clone()
                ref2[:, own_cols] = base[rank].float()  # the served gather reads this rank's unplanted matmul output
                ok2 = torch.equal(out2, ref2)
            check(f"c2_served_control rows={rows} rank={rank}", ok2, f"out={tuple(out2.shape)} {out2.dtype}")

            # c5: no deferral on decode or without the worker's defer flag (knob off = c2).
            dec = make_processor(group, True, True)._get_logits(hidden, lm_head, metadata(False), use_logits_buffer=False)
            nodefer = make_processor(group, False, True)._get_logits(hidden, lm_head, md, use_logits_buffer=False)
            ok5 = dec.shape == (rows, VOCAB) and nodefer.shape == (rows, VOCAB)
            check(f"c5_no_defer rows={rows} rank={rank}", ok5,
                  f"decode={tuple(dec.shape)} nodefer={tuple(nodefer.shape)}")  # fmt: skip

        # c4: VP local payload on BF16 shards vs on their FP32 widening, every rank. An earlier run found the whole
        # payload is NOT bitwise across input dtypes; c4 (pre-registered before its data) checks every field
        # bitwise EXCEPT field 3 (s = sum exp(t - m), an FP32 tree reduction whose order follows the load layout,
        # which Triton picks per pointer dtype), and checks s of BOTH dtypes against an FP64 reference within
        # S_RTOL. c4x is the control for that mechanism: on a shard whose sum is exact in any order (values 8 or
        # -10000 only: every term is exp(0) = 1 or 0), the whole payload must be bitwise, field 3 included.
        bs, q, topk = rows // 8, 8, 16
        cand = torch.randint(0, VOCAB, (bs, q), device=DEV)
        cids = torch.randint(0, VOCAB, (bs, q - 1, topk), device=DEV, dtype=torch.int64).contiguous()
        qr = torch.rand((bs, q - 1, topk), device=DEV).contiguous()
        uni = torch.rand((bs, q - 1), device=DEV)
        ufin = torch.rand((bs,), device=DEV)
        temps = torch.ones((bs,), device=DEV)
        ok4, ok4x, nfield, s_rel_b, s_rel_f, s_rel_bf = True, True, [0] * 9, 0.0, 0.0, 0.0
        exact = torch.full((rows, SHARD), -10000.0, device=DEV)
        exact[:, ::7] = 8.0
        exact = exact.to(torch.bfloat16)
        for k in range(WORLD):
            off = k * SHARD
            kw = dict(temperatures=temps, uniform_samples_final=ufin, candidate_ids=cids, rows_per_request=q,
                      valid_width=min(SHARD, VOCAB - off), col_offset=off, vocab_size=VOCAB, softcap=None,
                      candidates=cand, q_rows=qr, uniform_samples=uni)  # fmt: skip
            sb = shards[k].contiguous()
            sf = sb.float()
            if MUT == "sens" and k == 0:
                sf = sf.clone()
                j = int(sf[0].argmax())
                sf[0, j] = sf[0, j] * (1 + 2**-7)  # one BF16 ulp (or more) up at this shard's row-0 max
            pb = vp_local_payload(logits_shard=sb, **kw)
            pf = vp_local_payload(logits_shard=sf, **kw)
            for f in range(9):  # fields 0..7, then 8 = the candidate-logit block 8..8+top_k
                cols = slice(f, f + 1) if f < 8 else slice(8, None)
                nfield[f] += int((pb[:, cols] != pf[:, cols]).sum())
            keep = [f for f in range(pb.shape[1]) if f != 3]
            ok4 = ok4 and torch.equal(pb[:, keep], pf[:, keep])
            # FP64 reference for s (inv_temp = 1): sum over the shard of exp(x - max x).
            x64 = sb.double()
            mx = x64.max(dim=1, keepdim=True).values
            e64 = torch.exp(x64 - mx)
            if MUT == "sref":  # sensitivity control: drop the row max's term (exp(0) = 1) from the reference
                e64 = e64.masked_fill(x64 == mx, 0.0)
            s_ref = e64.sum(dim=1)
            rb = ((pb[:, 3].double() - s_ref).abs() / s_ref).max().item()
            rf = ((pf[:, 3].double() - s_ref).abs() / s_ref).max().item()
            s_rel_b, s_rel_f = max(s_rel_b, rb), max(s_rel_f, rf)
            s_rel_bf = max(s_rel_bf, ((pb[:, 3].double() - pf[:, 3].double()).abs() / s_ref).max().item())
            # c4x: exact-sum shard, whole payload bitwise across dtypes.
            xb = exact  # ties at every 8.0: tie_break_left + the strict cross-block '>' keep the first column
            qb_ = vp_local_payload(logits_shard=xb, **kw)
            qf_ = vp_local_payload(logits_shard=xb.float(), **kw)
            ok4x = ok4x and torch.equal(qb_, qf_)
        ok4 = ok4 and s_rel_b <= S_RTOL and s_rel_f <= S_RTOL
        diag = (f"fields_differ={nfield} s_relerr_fp64 bf16={s_rel_b:.3e} fp32={s_rel_f:.3e} "
                f"bf16_vs_fp32={s_rel_bf:.3e} tol={S_RTOL:.0e}")  # fmt: skip
        check(f"c4_vp_payload_bf16_eq_fp32 rows={rows}", ok4, diag)
        check(f"c4x_exact_sum_bitwise rows={rows}", ok4x)

    print(f"VPD_DONE fails={FAILS}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
