"""GPU regression for the fused absorbed-MLA ``kv_b_proj`` LoRA correction.

Checks that the fused q/v-side kernels (:func:`q_side_fused_fwd` /
:func:`v_side_fused_fwd`) produce the same result as the split 4-kernel path
(``step_a_*`` -> ``step_b_*``) across a matrix that stresses the generality both
paths support:

  * non-uniform and empty segments,
  * ``permutation`` (adapter-sorted routing),
  * mixed and zero per-slot ranks,
  * ``use_cuda_graph`` segment grid,
  * fp16 / bf16, and a mixed FP16-activation / BF16-base-output case,
  * large rank (fused path) and a forced fallback to the split kernels,
  * many heads.

For every config it additionally checks both paths against an fp32 reference, so
a bug shared by both kernels is caught, not just divergence between them.

The fuse/fallback threshold is forced via ``SGLANG_MLA_LORA_FUSE_MAX_RANK`` so the
fused path is exercised at every rank (and to 0 for the forced-fallback case).
"""

import os
import unittest
from unittest import mock

import torch

import sglang.kernels.ops.gemm.kv_b_lora_absorbed as kvb
from sglang.kernels.ops.gemm.kv_b_lora_absorbed import (
    q_side_fused_fwd,
    step_a_q_fwd,
    step_a_v_fwd,
    step_b_q_fwd,
    step_b_v_fwd,
    v_side_fused_fwd,
)
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="nightly-1-gpu", nightly=True)

_FUSE_ENV = "SGLANG_MLA_LORA_FUSE_MAX_RANK"
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
_TOL = {"bf16": 3e-2, "fp16": 1e-2}  # rel-err vs fp32 reference, per dtype

# kv_b_proj dims for the tested model family (DeepSeek/Kimi absorbed MLA).
_QK_NOPE, _V_HEAD, _KV_LORA = 128, 128, 512


def _configs():
    """Matrix of (name, seg_lens, ranks, slot_of_seg, H, dtype, base_dtype, perm,
    cuda_graph, fuse_max_rank). ``slot_of_seg`` defaults to round-robin."""
    return [
        dict(name="decode_uniform", seg_lens=[1] * 8, ranks=[16]),
        dict(name="prefill_uniform", seg_lens=[64] * 4, ranks=[16]),
        dict(name="non_uniform_one_long", seg_lens=[1, 1, 200, 1, 1], ranks=[16]),
        dict(name="empty_segments", seg_lens=[0, 50, 0, 30, 0], ranks=[16]),
        dict(
            name="permutation_non_uniform",
            seg_lens=[1, 1, 128, 1],
            ranks=[16],
            perm=True,
        ),
        dict(
            name="mixed_ranks",
            seg_lens=[8, 8, 8, 8],
            ranks=[8, 16, 32],
            slot_of_seg=[0, 1, 2, 1],
        ),
        dict(
            name="rank0_slot_noop",
            seg_lens=[16, 16, 16],
            ranks=[0, 16, 32],
            slot_of_seg=[0, 1, 2],
        ),
        dict(name="cuda_graph_flag", seg_lens=[1] * 16, ranks=[16], cuda_graph=True),
        dict(name="dtype_fp16", seg_lens=[32] * 4, ranks=[16], dtype="fp16"),
        dict(name="big_rank_64", seg_lens=[8] * 4, ranks=[64]),
        dict(name="big_rank_128", seg_lens=[8] * 4, ranks=[128]),
        dict(name="many_heads_32", seg_lens=[4] * 8, ranks=[16], H=32),
        dict(
            name="large_batch",
            seg_lens=[1, 1, 1, 300, 1, 1, 1, 50] * 2,
            ranks=[16, 32],
            perm=True,
        ),
        dict(
            name="forced_fallback", seg_lens=[1, 1, 200, 1], ranks=[16], fuse_max_rank=0
        ),
        dict(
            name="mixed_fp16act_bf16base",
            seg_lens=[1, 1, 200, 1],
            ranks=[16],
            dtype="fp16",
            base_dtype="bf16",
        ),
    ]


def _build(cfg, device):
    seg_lens = cfg["seg_lens"]
    ranks = cfg["ranks"]
    H = cfg.get("H", 16)
    dt = _DTYPES[cfg.get("dtype", "bf16")]
    base_dt = _DTYPES[cfg.get("base_dtype", cfg.get("dtype", "bf16"))]
    qk, v, kv = _QK_NOPE, _V_HEAD, _KV_LORA
    full_K = qk + v
    nseg = len(seg_lens)
    S = int(sum(seg_lens))
    num_lora = len(ranks)
    R_max = max(1, max(ranks))
    slot_of_seg = cfg.get("slot_of_seg") or [i % num_lora for i in range(nseg)]

    g = torch.Generator(device=device).manual_seed(1234)
    rnd = (
        lambda *shape, d=dt: torch.randn(*shape, generator=g, device=device, dtype=d)
        * 0.1
    )

    seg_indptr = torch.tensor(
        [0] + torch.tensor(seg_lens).cumsum(0).tolist(),
        device=device,
        dtype=torch.int32,
    )
    weight_indices = torch.tensor(slot_of_seg, device=device, dtype=torch.int32)
    lora_ranks = torch.tensor(ranks, device=device, dtype=torch.int32)
    scalings = (0.5 + torch.rand(num_lora, generator=g, device=device)).float()
    perm = (
        torch.randperm(S, generator=g, device=device).to(torch.int32)
        if cfg.get("perm")
        else None
    )

    batch_info = LoRABatchInfo(
        use_cuda_graph=cfg.get("cuda_graph", False),
        bs=nseg,
        num_segments=nseg,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        max_len=(max(seg_lens) if seg_lens else 0),
        seg_lens=torch.tensor(seg_lens, device=device, dtype=torch.int32),
        permutation=perm,
    )
    return dict(
        q_nope=rnd(S, H, qk),
        attn_output=rnd(S, H, kv),
        A_buf=rnd(num_lora, R_max, kv),
        B_buf=rnd(num_lora, H * full_K, R_max),
        base_q=rnd(S, H, kv, d=base_dt),
        base_v=rnd(S, H, v, d=base_dt),
        batch_info=batch_info,
        perm=perm,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        nseg=nseg,
        H=H,
        qk=qk,
        v=v,
        full_K=full_K,
    )


def _fp32_reference(d):
    """fp32 ground truth honoring permutation, per-slot rank truncation, scalings."""
    qk, v, full_K, H = d["qk"], d["v"], d["full_K"], d["H"]
    q_f, a_f = d["q_nope"].float(), d["attn_output"].float()
    A_f, B_f = d["A_buf"].float(), d["B_buf"].float()
    q_out, v_out = d["base_q"].float().clone(), d["base_v"].float().clone()
    for seg in range(d["nseg"]):
        s0 = int(d["seg_indptr"][seg])
        s1 = int(d["seg_indptr"][seg + 1])
        if s1 <= s0:
            continue
        w = int(d["weight_indices"][seg])
        r = int(d["lora_ranks"][w])
        if r == 0:
            continue
        scaling = float(d["scalings"][w])
        rows = (
            d["perm"][s0:s1].long()
            if d["perm"] is not None
            else torch.arange(s0, s1, device=q_f.device)
        )
        A_slot = A_f[w, :r, :]
        for h in range(H):
            b = h * full_K
            B_kc = B_f[w, b : b + qk, :r]
            B_vc = B_f[w, b + qk : b + full_K, :r]
            q_out[rows, h, :] += (q_f[rows, h, :] @ B_kc) @ A_slot * scaling
            v_out[rows, h, :] += (a_f[rows, h, :] @ A_slot.T) @ B_vc.T * scaling
    return q_out, v_out


def _relerr(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-6)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestKvBLoraAbsorbedFused(unittest.TestCase):
    def test_fused_matches_split_and_reference(self):
        device = "cuda"
        for cfg in _configs():
            with self.subTest(cfg=cfg["name"]):
                d = _build(cfg, device)
                bi, full_K, qk, v = d["batch_info"], d["full_K"], d["qk"], d["v"]
                tol = max(
                    _TOL[cfg.get("dtype", "bf16")],
                    _TOL[cfg.get("base_dtype", cfg.get("dtype", "bf16"))],
                )

                q_ref, v_ref = _fp32_reference(d)

                # Split path (BEFORE). Uses the module-level step_* references
                # imported into this test namespace, so it is unaffected by the
                # patches applied to ``kvb`` below.
                q_split, v_split = d["base_q"].clone(), d["base_v"].clone()
                step_b_q_fwd(
                    step_a_q_fwd(d["q_nope"], d["B_buf"], bi, full_K),
                    d["A_buf"],
                    bi,
                    q_split,
                )
                step_b_v_fwd(
                    step_a_v_fwd(d["attn_output"], d["A_buf"], bi),
                    d["B_buf"],
                    bi,
                    v_split,
                    qk,
                    v,
                )

                # Fused path (AFTER). Force fused at every rank (0 => split fallback).
                forced = cfg.get("fuse_max_rank") == 0
                prev = os.environ.get(_FUSE_ENV)
                os.environ[_FUSE_ENV] = str(cfg.get("fuse_max_rank", 1_000_000))
                q_fused, v_fused = d["base_q"].clone(), d["base_v"].clone()
                try:
                    # The fused wrappers catch *any* kernel exception and silently
                    # run the split kernels instead. Patch the step-A entry points
                    # (called first on every fallback route) so a fused config that
                    # secretly falls back is caught rather than passing trivially.
                    if forced:
                        # forced_fallback MUST take the split path -- assert it did.
                        with mock.patch.object(
                            kvb, "step_a_q_fwd", wraps=kvb.step_a_q_fwd
                        ) as mq, mock.patch.object(
                            kvb, "step_a_v_fwd", wraps=kvb.step_a_v_fwd
                        ) as mv:
                            q_side_fused_fwd(
                                d["q_nope"], d["B_buf"], d["A_buf"], bi, q_fused, full_K
                            )
                            v_side_fused_fwd(
                                d["attn_output"],
                                d["A_buf"],
                                d["B_buf"],
                                bi,
                                v_fused,
                                qk,
                                v,
                            )
                        self.assertTrue(
                            mq.called, "forced_fallback did not use split q path"
                        )
                        self.assertTrue(
                            mv.called, "forced_fallback did not use split v path"
                        )
                    else:
                        # Fused configs MUST NOT fall back -- any split call fails here.
                        with mock.patch.object(
                            kvb,
                            "step_a_q_fwd",
                            side_effect=AssertionError("q fused kernel fell back"),
                        ), mock.patch.object(
                            kvb,
                            "step_a_v_fwd",
                            side_effect=AssertionError("v fused kernel fell back"),
                        ):
                            q_side_fused_fwd(
                                d["q_nope"], d["B_buf"], d["A_buf"], bi, q_fused, full_K
                            )
                            v_side_fused_fwd(
                                d["attn_output"],
                                d["A_buf"],
                                d["B_buf"],
                                bi,
                                v_fused,
                                qk,
                                v,
                            )
                finally:
                    if prev is None:
                        os.environ.pop(_FUSE_ENV, None)
                    else:
                        os.environ[_FUSE_ENV] = prev

                # Both paths must be correct vs the fp32 reference.
                self.assertLess(_relerr(q_split, q_ref), tol, "split q vs fp32 ref")
                self.assertLess(_relerr(v_split, v_ref), tol, "split v vs fp32 ref")
                self.assertLess(_relerr(q_fused, q_ref), tol, "fused q vs fp32 ref")
                self.assertLess(_relerr(v_fused, v_ref), tol, "fused v vs fp32 ref")

                # Strict fused-vs-split regression. On the tested GPUs / Triton the
                # fused output is bit-identical to the split path (the rank-contraction
                # reduction-order difference is sub-output-ULP). If a future
                # GPU/Triton produces a ~1-ULP difference, relax these to
                # torch.testing.assert_close with a tight tolerance.
                self.assertTrue(
                    torch.equal(q_fused, q_split), f"[{cfg['name']}] fused q != split q"
                )
                self.assertTrue(
                    torch.equal(v_fused, v_split), f"[{cfg['name']}] fused v != split v"
                )


if __name__ == "__main__":
    unittest.main()
