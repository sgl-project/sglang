# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Kernel-contract test for the B2 SINGLE-CALL CP-causal verify
(``SGLANG_DCP_SINGLE_CALL_VERIFY``).

The single-call design folds each rank's ENTIRE local interleaved slice
(committed prefix + the draft tokens this rank owns) in one
``tokenspeed_mla_decode`` call with ``causal_mask=True`` and per-request
GLOBAL ``causal_seqs`` (= prefix + T); the kernel must resolve each q_tok's
causal bound in GLOBAL coordinates BEFORE the cp divide. The cross-rank
base-2 LSE merge of the D partials must then reproduce the unsharded result.

Two configurations, both asserted against an fp32 reference computed here:
  - CLEAN: random fp8 latents. PASS bar: cp8-merged rel-fro within 3x of the
    cp1 (unsharded single-call) calibration floor, i.e. fp8 requant noise.
  - POISON: every draft row's latent set to +200 (fp8-representable, dominates
    softmax). The reference includes the same poison, so outputs match ONLY if
    the kernel's per-(request, q_tok) bound agrees with the reference on every
    poisoned row: any +-1 bound error explodes to O(1) rel error. This detects
    off-by-one masking DETERMINISTICALLY, not statistically.

Run manually:
    python -m pytest test_dcp_single_call_verify.py -v
"""

import math
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-large")

KV_LORA, ROPE = 512, 64
DIM = KV_LORA + ROPE
PAGE = 64
SCALE = 1.0 / math.sqrt(DIM)
CP_WORLD = 8
T = 8  # DFlash ndt


def _tokenspeed():
    import tokenspeed_mla

    return tokenspeed_mla


def _build_scene(prefixes, heads, poison, dev, seed=1234):
    torch.manual_seed(seed)
    bs = len(prefixes)
    totals = [p + T for p in prefixes]
    q = torch.randn(bs, T, heads, DIM, device=dev).to(torch.float8_e4m3fn)
    kv_rows = []
    for i, tot in enumerate(totals):
        rows = torch.randn(tot, DIM, device=dev)
        if poison:
            rows[prefixes[i] : tot] = 200.0
        kv_rows.append(rows.to(torch.float8_e4m3fn))
    return q, kv_rows, totals


def _fp32_reference(q, kv_rows, prefixes):
    outs = []
    for i, rows in enumerate(kv_rows):
        qf = q[i].float()
        kf = rows.float()
        vf = kf[:, :KV_LORA]
        toks = []
        for t in range(T):
            bound = prefixes[i] + t + 1  # exclusive end, self-inclusive rule
            logits = torch.einsum("hd,nd->hn", qf[t], kf[:bound]) * SCALE
            w = torch.softmax(logits, dim=-1)
            toks.append(torch.einsum("hn,nc->hc", w, vf[:bound]))
        outs.append(torch.stack(toks))
    return torch.stack(outs)  # [bs, T, H, KV_LORA]


def _paged_local(kv_rows, totals, cp_world, cp_rank, dev):
    seqs, pools = [], []
    for rows, tot in zip(kv_rows, totals):
        owned = torch.arange(cp_rank, tot, cp_world, device=dev)
        local = rows[owned]
        n = local.shape[0]
        pages = max(1, math.ceil(n / PAGE))
        pool = torch.zeros(pages, PAGE, DIM, device=dev, dtype=torch.float8_e4m3fn)
        pool.view(-1, DIM)[:n] = local
        pools.append(pool)
        seqs.append(n)
    max_pages = max(p.shape[0] for p in pools)
    kv = torch.zeros(
        1 + sum(p.shape[0] for p in pools),
        PAGE,
        DIM,
        device=dev,
        dtype=torch.float8_e4m3fn,
    )
    bt = torch.zeros(len(pools), max_pages, dtype=torch.int32, device=dev)
    nxt = 1
    for i, pool in enumerate(pools):
        np_ = pool.shape[0]
        kv[nxt : nxt + np_] = pool
        bt[i, :np_] = torch.arange(nxt, nxt + np_, dtype=torch.int32, device=dev)
        nxt += np_
    return kv, bt, torch.tensor(seqs, dtype=torch.int32, device=dev)


def _call(ts, q, kv, bt, seq, max_seq, heads, cp_world, cp_rank, causal_seqs, dev):
    ws_bytes = ts.get_num_sm(dev) * heads * max(T, 8) * (KV_LORA + 1) * 4
    ws = torch.empty(ws_bytes, dtype=torch.int8, device=dev)
    return ts.tokenspeed_mla_decode(
        query=q,
        kv_cache=kv,
        workspace_buffer=ws,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        block_tables=bt,
        seq_lens=seq,
        max_seq_len=max(int(max_seq), 1),
        softmax_scale=SCALE,
        output_scale=1.0,
        enable_pdl=True,
        return_lse=True,
        cp_world=cp_world,
        cp_rank=cp_rank,
        causal_seqs=causal_seqs,
        causal_mask=True,
    )


def _lse_merge(outs, lses):
    m = torch.stack(lses).max(0).values
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    num = torch.zeros_like(outs[0])
    den = torch.zeros_like(lses[0])
    for o, l in zip(outs, lses):
        w = torch.exp2(l - m)
        w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
        num += o * w.unsqueeze(-1)
        den += w
    return num / den.clamp_min(1e-30).unsqueeze(-1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDcpSingleCallVerify(unittest.TestCase):
    HEADS = 16
    PREFIXES = [997, 1531]  # distinct residues mod CP_WORLD, multi-page

    def _run(self, poison):
        ts = _tokenspeed()
        dev = torch.device("cuda")
        q, kv_rows, totals = _build_scene(self.PREFIXES, self.HEADS, poison, dev)
        ref = _fp32_reference(q, kv_rows, self.PREFIXES)

        # calibration: unsharded single call, same causal semantics
        kv1, bt1, seq1 = _paged_local(kv_rows, totals, 1, 0, dev)
        o1, _ = _call(
            ts, q, kv1, bt1, seq1, seq1.max().item(), self.HEADS, 1, 0, None, dev
        )
        floor = ((o1.reshape(ref.shape).float() - ref).norm() / ref.norm()).item()

        causal_seqs = torch.tensor(totals, dtype=torch.int32, device=dev)
        outs, lses = [], []
        for r in range(CP_WORLD):
            kvr, btr, seqr = _paged_local(kv_rows, totals, CP_WORLD, r, dev)
            o, l = _call(
                ts,
                q,
                kvr,
                btr,
                seqr,
                seqr.max().item(),
                self.HEADS,
                CP_WORLD,
                r,
                causal_seqs,
                dev,
            )
            outs.append(o.reshape(ref.shape).float())
            lses.append(l.reshape(ref.shape[:-1]).float())
        merged = _lse_merge(outs, lses)
        rel = ((merged - ref).norm() / ref.norm()).item()
        per_tok = (merged - ref).flatten(2).norm(dim=-1) / ref.flatten(2).norm(
            dim=-1
        ).clamp_min(1e-9)
        tol = 3 * max(floor, 1e-3)
        self.assertLessEqual(
            rel, tol, f"cp8 rel {rel:.4f} vs floor {floor:.4f} (poison={poison})"
        )
        self.assertLessEqual(
            per_tok.max().item(),
            3 * tol,
            f"worst-token rel {per_tok.max().item():.4f} (poison={poison}) — "
            "a per-(request,q_tok) causal-bound error",
        )

    def test_clean_matches_fp32_reference(self):
        self._run(poison=False)

    def test_poisoned_draft_rows_pin_the_causal_bounds(self):
        self._run(poison=True)


if __name__ == "__main__":
    unittest.main()
