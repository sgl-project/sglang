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
"""SGLANG_DCP_PASS1_NO_CP contract test: the DCP verify-cascade pass-1
(NON-causal prefix fold) must be identical with and without the tokenspeed
kernel's cp path.

Why BIT-EXACT is asserted (not a tolerance): in tokenspeed_mla's fp8 decode
kernel (mla_decode_fp8.py, verified identical across wheels 0.1.7-0.2.0)
every use of ``self.cp_world`` and ``K_causal``(=``causal_seqs``) sits inside
a ``cutlass.const_expr(self.is_causal)`` branch — the per-token causal
``k_bound`` arithmetic. With ``causal_mask=False`` those branches are not
traced at all, so the (cp_world=W, causal_seqs=global) and (cp_world=1,
causal_seqs=None) compilations produce the same kernel IR over the same LOCAL
block table / seq_lens: the non-causal bound is ``k_bound = K`` (the local
seq_len) on both. The block-table / page indexing never involves cp_world
(the kernel reads the caller-compacted LOCAL slice — the "LOCAL layout"
finding of dcp-tokenspeed-contract.md §2), and the split-KV reduction is a
fixed-order warp reduction, so repeated launches are deterministic and the
two variants must agree bit-for-bit. A mismatch here means the wheel's cp
contract changed — flip the SGLANG_DCP_PASS1_NO_CP default OFF before
shipping such a wheel.

Rows whose local slice is empty are excluded: the kernel emits unspecified
NaN/garbage for seq_len==0 (dcp-tokenspeed-contract.md §3b), which production
masks out via dcp_zero_mask regardless of this flag.

Requires tokenspeed_mla + SM100 (skips elsewhere; the B200 gate executes it).

Usage:
    python -m pytest test_dcp_pass1_no_cp.py -v
"""

import importlib.util
import math
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-large")

D_LATENT, D_ROPE = 512, 64
D_QK = D_LATENT + D_ROPE
PAGE = 64


def _supported() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is required"
    if importlib.util.find_spec("tokenspeed_mla") is None:
        return False, "tokenspeed_mla python package is not installed"
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        return False, f"tokenspeed_mla requires SM100+, got SM {major}.{minor}"
    return True, ""


_SUPPORTED, _SKIP_REASON = _supported()


def _bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Byte-level equality (NaN-safe, unlike torch.equal)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8),
        b.contiguous().reshape(-1).view(torch.uint8),
    )


def _build_local_slice(global_prefix, cp_world, cp_rank, device, seed):
    """Synthetic rank-local pass-1 inputs.

    The kernel only ever sees the LOCAL compacted slice (block table +
    seq_lens); the strided global<->local mapping lives entirely in the
    framework-side index build. So a random local pool with sequential pages
    exercises exactly the kernel contract under test.
    Returns (kv_pool [pages+1,1,PAGE,D_qk] fp8, block_tables [bs,max_pages],
    local_lens [bs] int32).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Owner rule: local_len = #{i < prefix : i % W == r}.
    local_lens = torch.tensor(
        [max(0, (p - cp_rank + cp_world - 1) // cp_world) for p in global_prefix],
        dtype=torch.int32,
    )
    pages_per_seq = [math.ceil(max(int(l), 1) / PAGE) for l in local_lens]
    total_pages = sum(pages_per_seq)
    kv_pool = (
        torch.randn(total_pages + 1, PAGE, D_QK, generator=g)
        .to(torch.float8_e4m3fn)
        .to(device)
        .unsqueeze(1)
    )
    max_pages = max(pages_per_seq)
    bt = torch.zeros(len(global_prefix), max_pages, dtype=torch.int32)
    next_page = 1  # page 0 reserved so padding entries stay valid
    for b, np_ in enumerate(pages_per_seq):
        bt[b, :np_] = torch.arange(next_page, next_page + np_)
        next_page += np_
    return kv_pool, bt.to(device), local_lens.to(device)


def _ref_noncausal(q, kv_pool, bt, local_lens, softmax_scale):
    """fp32 non-causal reference over each row's local_lens[b] tokens."""
    bs, T, H, _ = q.shape
    qf = q.float().cpu()
    pool = kv_pool.squeeze(1).float().cpu()  # [pages, PAGE, D_qk]
    outs, lses = [], []
    for b in range(bs):
        L = int(local_lens[b].item())
        kv = pool[bt[b].cpu().long()].reshape(-1, D_QK)[:L]  # [L, D_qk]
        s = torch.einsum("thd,jd->thj", qf[b], kv) * softmax_scale
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("thj,jd->thd", p, kv[:, :D_LATENT]))
        lses.append(torch.logsumexp(s, dim=-1) * math.log2(math.e))
    return torch.stack(outs), torch.stack(lses)  # [bs,T,H,512], [bs,T,H] base-2


@unittest.skipIf(not _SUPPORTED, _SKIP_REASON)
class TestPass1NoCpBitExact(unittest.TestCase):
    """Old cp-path call vs plain call on identical inputs, bit-exact
    (justification in the module docstring), plus an fp32 sanity anchor."""

    # Production pass-1 shape: T=8 draft tokens, H_full=64 gathered heads.
    BS, T, H, W, RANK = 3, 8, 64, 8, 3
    # Global prefixes chosen so every row owns >= 1 local token on RANK
    # (mixed page counts: 88, 2, 1 local tokens -> 2, 1, 1 pages).
    GLOBAL_PREFIX = [700, 12, 4]

    def _run(self, q, kv_pool, bt, local_lens, ws, no_cp, causal_seqs):
        import tokenspeed_mla

        return tokenspeed_mla.tokenspeed_mla_decode(
            query=q,
            kv_cache=kv_pool,
            workspace_buffer=ws,
            kv_lora_rank=D_LATENT,
            qk_rope_head_dim=D_ROPE,
            block_tables=bt,
            seq_lens=local_lens,
            max_seq_len=max(int(local_lens.max().item()), 1),
            softmax_scale=1.0 / math.sqrt(D_QK),
            output_scale=1.0,
            enable_pdl=False,
            return_lse=True,
            cp_world=1 if no_cp else self.W,
            cp_rank=0 if no_cp else self.RANK,
            causal_seqs=None if no_cp else causal_seqs,
            causal_mask=False,  # pass-1 is NON-causal in both arms
        )

    def test_no_cp_bit_exact_and_ref_anchor(self):
        import tokenspeed_mla

        dev = torch.device("cuda")
        torch.manual_seed(41)
        kv_pool, bt, local_lens = _build_local_slice(
            self.GLOBAL_PREFIX, self.W, self.RANK, dev, seed=41
        )
        self.assertTrue(bool((local_lens > 0).all()), "test setup: zero-owned row")
        q = torch.randn(self.BS, self.T, self.H, D_QK).to(torch.float8_e4m3fn).to(dev)
        causal_seqs = torch.tensor(self.GLOBAL_PREFIX, dtype=torch.int32, device=dev)
        ws_bytes = (
            tokenspeed_mla.get_num_sm(dev)
            * self.H
            * max(self.T, 8)
            * (D_LATENT + 1)
            * 4
        )
        ws = torch.empty(ws_bytes, dtype=torch.int8, device=dev)

        out_cp, lse_cp = self._run(q, kv_pool, bt, local_lens, ws, False, causal_seqs)
        out_no, lse_no = self._run(q, kv_pool, bt, local_lens, ws, True, None)

        self.assertTrue(_bit_equal(out_no, out_cp), "pass-1 out: no-cp != cp")
        self.assertTrue(_bit_equal(lse_no, lse_cp), "pass-1 lse: no-cp != cp")

        # fp32 anchor: both arms must be REAL non-causal attention over the
        # local slice, not merely identical garbage. Relative-L2 5e-2 covers
        # the fp8 P-quantization noise floor (contract doc §2: rel ~0.0145).
        ref_out, ref_lse = _ref_noncausal(
            q, kv_pool, bt, local_lens, 1.0 / math.sqrt(D_QK)
        )
        rel = (out_no.float().cpu() - ref_out).norm() / ref_out.norm()
        self.assertLess(rel.item(), 5e-2, f"pass-1 vs fp32 ref rel {rel.item()}")
        lse_diff = (lse_no.float().cpu() - ref_lse).abs().max().item()
        self.assertLess(lse_diff, 2e-2, f"pass-1 lse vs fp32 ref {lse_diff}")

    def test_no_cp_bit_exact_repeat_determinism(self):
        """Launch-to-launch determinism of the no-cp arm itself (precondition
        for the bit-exact assertion above to be meaningful)."""
        import tokenspeed_mla

        dev = torch.device("cuda")
        torch.manual_seed(43)
        kv_pool, bt, local_lens = _build_local_slice(
            self.GLOBAL_PREFIX, self.W, self.RANK, dev, seed=43
        )
        q = torch.randn(self.BS, self.T, self.H, D_QK).to(torch.float8_e4m3fn).to(dev)
        ws_bytes = (
            tokenspeed_mla.get_num_sm(dev)
            * self.H
            * max(self.T, 8)
            * (D_LATENT + 1)
            * 4
        )
        ws = torch.empty(ws_bytes, dtype=torch.int8, device=dev)
        out_a, lse_a = self._run(q, kv_pool, bt, local_lens, ws, True, None)
        out_b, lse_b = self._run(q, kv_pool, bt, local_lens, ws, True, None)
        self.assertTrue(_bit_equal(out_a, out_b), "no-cp arm nondeterministic (out)")
        self.assertTrue(_bit_equal(lse_a, lse_b), "no-cp arm nondeterministic (lse)")


if __name__ == "__main__":
    unittest.main()
