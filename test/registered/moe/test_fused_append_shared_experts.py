"""Unit tests for ``fused_append_shared_experts_with_weights``.

Covers PR #28658, which folds the shared-expert ``sigmoid`` activation and the
bf16->fp32 cast into the append kernel's prologue (``apply_sigmoid=True``), and
PR #28666, which additionally folds the ``shared_expert_gate`` GEMV into the
kernel (``fuse_gate=True``). The old (``apply_sigmoid=False, fuse_gate=False``)
path must stay byte-for-byte identical; the ``apply_sigmoid`` path must equal
the eager ``sigmoid(logits.float()) * scale`` it replaces; and the ``fuse_gate``
path must equal the eager ``sigmoid((hidden @ W_gate).float()) * scale`` GEMV it
replaces. These paths only run on the AITER shared-expert-fusion route at
serving time, so they are otherwise uncovered by CI.
"""

import unittest

import torch

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    fused_append_shared_experts_with_weights,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase, empty_gpu_cache

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")


def _eager_append(
    topk_ids, topk_weights, shared_weights, s, n_base, apply_sigmoid, scale
):
    """Reference: eager equivalent of the kernel (matches the pre-PR two-step path).

    The last ``s`` columns hold the shared expert(s); ids are ``n_base + i`` and
    weights are either the raw shared value cast to the output dtype (legacy) or
    ``sigmoid(value.float()) * scale`` cast to the output dtype (fused).
    """
    m, k = topk_ids.shape
    sw = shared_weights
    if sw.ndim == 1:
        sw = sw.unsqueeze(-1)
    if sw.shape[1] < s:
        sw = sw.expand(m, s)
    if apply_sigmoid:
        col = (torch.sigmoid(sw.float()) * scale).to(topk_weights.dtype)
    else:
        col = sw.to(topk_weights.dtype)

    shared_ids = (
        (n_base + torch.arange(s, device=topk_ids.device))
        .view(1, s)
        .expand(m, s)
        .to(topk_ids.dtype)
    )

    out_ids = torch.cat([topk_ids, shared_ids], dim=1)
    out_weights = torch.cat([topk_weights, col.contiguous()], dim=1)
    return out_ids, out_weights


def _eager_gate(topk_ids, topk_weights, hidden, gate_weight, s, n_base, scale):
    """Reference for fuse_gate=True: in-kernel GEMV + sigmoid + scale.

    ``logit[m] = hidden[m, :] . W_gate[:]`` (fp32), broadcast
    ``sigmoid(logit) * scale`` to all ``s`` shared slots.
    """
    m, k = topk_ids.shape
    logit = (hidden.float() @ gate_weight.float().reshape(-1)).view(m, 1)
    col = (torch.sigmoid(logit) * scale).expand(m, s).to(topk_weights.dtype)

    shared_ids = (
        (n_base + torch.arange(s, device=topk_ids.device))
        .view(1, s)
        .expand(m, s)
        .to(topk_ids.dtype)
    )

    out_ids = torch.cat([topk_ids, shared_ids], dim=1)
    out_weights = torch.cat([topk_weights, col.contiguous()], dim=1)
    return out_ids, out_weights


class TestFusedAppendSharedExperts(CustomTestCase):
    MS = [1, 3, 128]
    KS = [2, 6, 8]
    SS = [1, 2]
    SCALES = [1.0, 0.5]

    @staticmethod
    def _rand_inputs(m, k, s, dtype, n_base):
        device = get_device()
        topk_ids = torch.randint(0, n_base, (m, k), dtype=torch.int32, device=device)
        topk_weights = torch.rand(m, k, dtype=torch.float32, device=device).to(dtype)
        # shared gate logits: wide range so sigmoid is meaningfully exercised
        shared_logits = (
            torch.randn(m, s, dtype=torch.float32, device=device) * 4.0
        ).to(dtype)
        return topk_ids, topk_weights, shared_logits

    def _check(self, out_ids, out_w, ref_ids, ref_w, dtype, msg):
        # ids are integers -> must match exactly
        self.assertTrue(torch.equal(out_ids, ref_ids), f"ids mismatch: {msg}")
        # triton tl.sigmoid vs torch.sigmoid differ by a few ULPs, so the fused
        # activation path is only bitexact-free: tolerance, tight in fp32.
        if dtype == torch.float32:
            torch.testing.assert_close(
                out_w, ref_w, rtol=1e-5, atol=1e-6, msg=f"weights mismatch: {msg}"
            )
        else:
            torch.testing.assert_close(
                out_w, ref_w, rtol=2e-3, atol=2e-3, msg=f"weights mismatch: {msg}"
            )

    def test_fused_sigmoid_matches_eager(self):
        """apply_sigmoid=True must equal eager sigmoid(logits.float())*scale."""
        n_base = 64
        for dtype in [torch.float32, torch.bfloat16]:
            for m in self.MS:
                for k in self.KS:
                    for s in self.SS:
                        for scale in self.SCALES:
                            topk_ids, topk_w, logits = self._rand_inputs(
                                m, k, s, dtype, n_base
                            )
                            out_ids, out_w = fused_append_shared_experts_with_weights(
                                topk_ids,
                                topk_w,
                                logits,
                                s,
                                N=n_base,
                                apply_sigmoid=True,
                                scale=scale,
                            )
                            ref_ids, ref_w = _eager_append(
                                topk_ids, topk_w, logits, s, n_base, True, scale
                            )
                            self.assertEqual(out_ids.shape, (m, k + s))
                            self._check(
                                out_ids,
                                out_w,
                                ref_ids,
                                ref_w,
                                dtype,
                                f"m={m} k={k} s={s} scale={scale} dtype={dtype}",
                            )
                            empty_gpu_cache()

    def test_legacy_path_unchanged(self):
        """apply_sigmoid=False must reproduce the legacy cast-and-append exactly."""
        n_base = 64
        for dtype in [torch.float32, torch.bfloat16]:
            for m in self.MS:
                for k in self.KS:
                    for s in self.SS:
                        topk_ids, topk_w, shared = self._rand_inputs(
                            m, k, s, dtype, n_base
                        )
                        out_ids, out_w = fused_append_shared_experts_with_weights(
                            topk_ids, topk_w, shared, s, N=n_base, apply_sigmoid=False
                        )
                        ref_ids, ref_w = _eager_append(
                            topk_ids, topk_w, shared, s, n_base, False, 1.0
                        )
                        # legacy path is a pure copy/cast -> exact for all dtypes
                        self.assertTrue(torch.equal(out_ids, ref_ids))
                        self.assertTrue(torch.equal(out_w, ref_w))
                        empty_gpu_cache()

    def test_routed_columns_preserved(self):
        """First k columns must be an untouched copy of the routed topk output."""
        n_base = 32
        m, k, s = 5, 6, 1
        topk_ids, topk_w, logits = self._rand_inputs(m, k, s, torch.bfloat16, n_base)
        out_ids, out_w = fused_append_shared_experts_with_weights(
            topk_ids, topk_w, logits, s, N=n_base, apply_sigmoid=True, scale=1.0
        )
        self.assertTrue(torch.equal(out_ids[:, :k], topk_ids))
        self.assertTrue(torch.equal(out_w[:, :k], topk_w))
        # shared ids are exactly n_base .. n_base+s-1
        self.assertTrue(
            torch.equal(
                out_ids[:, k],
                torch.full((m,), n_base, dtype=topk_ids.dtype, device=topk_ids.device),
            )
        )

    def test_fuse_gate_matches_eager(self):
        """fuse_gate=True must equal eager sigmoid((hidden @ W_gate).float())*scale."""
        n_base = 64
        device = get_device()
        for dtype in [torch.float32, torch.bfloat16]:
            for hidden_dim in [512, 2048]:
                for m in self.MS:
                    for k in self.KS:
                        for s in self.SS:
                            for scale in self.SCALES:
                                topk_ids = torch.randint(
                                    0, n_base, (m, k), dtype=torch.int32, device=device
                                )
                                topk_w = torch.rand(
                                    m, k, dtype=torch.float32, device=device
                                ).to(dtype)
                                hidden = (
                                    torch.randn(
                                        m,
                                        hidden_dim,
                                        dtype=torch.float32,
                                        device=device,
                                    )
                                    * 0.1
                                ).to(dtype)
                                # shared_expert_gate weight: Linear(hidden, 1)
                                gate_w = (
                                    torch.randn(
                                        1,
                                        hidden_dim,
                                        dtype=torch.float32,
                                        device=device,
                                    )
                                    * 0.1
                                ).to(dtype)

                                out_ids, out_w = (
                                    fused_append_shared_experts_with_weights(
                                        topk_ids,
                                        topk_w,
                                        None,
                                        s,
                                        N=n_base,
                                        fuse_gate=True,
                                        hidden_states=hidden,
                                        gate_weight=gate_w,
                                        scale=scale,
                                    )
                                )
                                ref_ids, ref_w = _eager_gate(
                                    topk_ids, topk_w, hidden, gate_w, s, n_base, scale
                                )
                                self.assertEqual(out_ids.shape, (m, k + s))
                                msg = (
                                    f"m={m} k={k} s={s} h={hidden_dim} "
                                    f"scale={scale} dtype={dtype}"
                                )
                                self.assertTrue(
                                    torch.equal(out_ids, ref_ids), f"ids: {msg}"
                                )
                                # fp32 GEMV reduction order differs from torch
                                # matmul -> tolerance (tight in fp32).
                                if dtype == torch.float32:
                                    torch.testing.assert_close(
                                        out_w, ref_w, rtol=1e-4, atol=1e-5, msg=msg
                                    )
                                else:
                                    torch.testing.assert_close(
                                        out_w, ref_w, rtol=2e-3, atol=2e-3, msg=msg
                                    )
                                empty_gpu_cache()

    def test_fuse_gate_and_apply_sigmoid_mutually_exclusive(self):
        """fuse_gate already applies sigmoid; combining with apply_sigmoid asserts."""
        device = get_device()
        topk_ids = torch.randint(0, 8, (2, 2), dtype=torch.int32, device=device)
        topk_w = torch.rand(2, 2, dtype=torch.float32, device=device)
        hidden = torch.randn(2, 16, dtype=torch.float32, device=device)
        gate_w = torch.randn(1, 16, dtype=torch.float32, device=device)
        with self.assertRaises(AssertionError):
            fused_append_shared_experts_with_weights(
                topk_ids,
                topk_w,
                None,
                1,
                N=8,
                apply_sigmoid=True,
                fuse_gate=True,
                hidden_states=hidden,
                gate_weight=gate_w,
            )

    def test_fuse_gate_requires_hidden_and_gate(self):
        """fuse_gate=True without hidden_states/gate_weight asserts."""
        device = get_device()
        topk_ids = torch.randint(0, 8, (2, 2), dtype=torch.int32, device=device)
        topk_w = torch.rand(2, 2, dtype=torch.float32, device=device)
        with self.assertRaises(AssertionError):
            fused_append_shared_experts_with_weights(
                topk_ids, topk_w, None, 1, N=8, fuse_gate=True
            )

    def test_zero_shared_is_noop(self):
        """num_fused_shared_experts <= 0 returns the inputs unchanged."""
        n_base = 8
        topk_ids, topk_w, logits = self._rand_inputs(4, 2, 1, torch.float32, n_base)
        out_ids, out_w = fused_append_shared_experts_with_weights(
            topk_ids, topk_w, logits, 0, N=n_base, apply_sigmoid=True
        )
        self.assertIs(out_ids, topk_ids)
        self.assertIs(out_w, topk_w)


if __name__ == "__main__":
    unittest.main()
