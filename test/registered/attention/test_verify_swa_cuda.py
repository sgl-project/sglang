"""Independent causal-window reference for the opt-in CUDA verify kernel."""

import unittest

import torch

from sglang.kernels.ops.attention.verify_splitkv import verify_splitkv_fwd
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


@unittest.skipIf(not torch.cuda.is_available(), "GPU required")
class TestSwaSplitKVVerify(CustomTestCase):
    def test_inactive_and_masked_splits_ignore_poisoned_scratch(self):
        from sglang.kernels.ops.attention.verify_splitkv import VerifySplitKV

        if torch.version.hip or torch.cuda.get_device_capability() not in (
            (8, 9),
            (9, 0),
        ):
            self.skipTest("CUDA candidate validated on Ada and Hopper")
        q = torch.zeros(3, 16, 256, device="cuda", dtype=torch.bfloat16)
        k = torch.zeros(3, 8, 256, device="cuda", dtype=q.dtype)
        v = torch.tensor([4, 6, 8], device="cuda", dtype=q.dtype)
        v = v[:, None, None].expand_as(k).contiguous()
        kb = torch.zeros(1, 8, 256, device="cuda", dtype=q.dtype)
        vb = torch.full_like(kb, 2)
        qo = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
        kp = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        ids = torch.zeros(1, device="cuda", dtype=torch.int64)
        kernel = VerifySplitKV(1, 16, 8, 256, 256, 3, n_splits=16)
        for window, expected in ((-1, [3, 4, 5]), (1, [3, 5, 7])):
            with self.subTest(window=window):
                # Only one split contains a prefix key. A small window also
                # makes that split fully masked for the later queries.
                kernel.att_out.fill_(float("nan"))
                kernel.att_lse.fill_(float("nan"))
                out = kernel(
                    q,
                    k,
                    v,
                    kb,
                    vb,
                    qo,
                    kp,
                    ids,
                    256**-0.5,
                    sliding_window_size=window,
                )
                ref = torch.tensor(expected, device="cuda", dtype=q.dtype)
                ref = ref[:, None, None].expand_as(q)
                torch.testing.assert_close(out, ref, atol=0, rtol=0)

    def test_each_query_excludes_its_oldest_keys(self):
        if torch.version.hip or torch.cuda.get_device_capability() not in (
            (8, 9),
            (9, 0),
        ):
            self.skipTest("CUDA candidate validated on Ada and Hopper")
        for window in (1, 3, 1023):
            with self.subTest(window=window):
                q = torch.zeros(4, 16, 256, device="cuda", dtype=torch.bfloat16)
                k = torch.zeros(4, 8, 256, device="cuda", dtype=q.dtype)
                v = torch.zeros_like(k)
                kb = torch.zeros(window, 8, 256, device="cuda", dtype=q.dtype)
                vb = torch.zeros_like(kb)
                vb[: min(4, window)] = 64
                qo = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
                kp = torch.tensor([0, window], device="cuda", dtype=torch.int32)
                ids = torch.arange(window - 1, -1, -1, device="cuda", dtype=torch.int64)
                # Keep nonzero values at the logical oldest edge despite the
                # reversed physical mapping.
                vb.zero_()
                vb[ids[: min(4, window)]] = 64
                out = torch.empty_like(q)
                self.assertTrue(
                    verify_splitkv_fwd(
                        q,
                        k,
                        v,
                        out,
                        kb,
                        vb,
                        qo,
                        kp,
                        ids,
                        None,
                        True,
                        None,
                        4,
                        1.0,
                        1.0,
                        sm_scale=256**-0.5,
                        sliding_window_size=window,
                    )
                )
                expected = torch.tensor(
                    [64 * max(0, min(4, window) - j) / (window + 1) for j in range(4)],
                    device="cuda",
                    dtype=torch.float64,
                )[:, None, None].expand_as(q)
                torch.testing.assert_close(out.double(), expected, atol=1e-4, rtol=1e-4)

    def test_cuda_eligibility_contract(self):
        from sglang.srt.layers.attention.triton_backend import (
            _can_use_cuda_verify,
        )

        q = torch.zeros(5, 16, 256, device="cuda", dtype=torch.bfloat16)
        k = torch.zeros(5, 8, 256, device="cuda", dtype=q.dtype)
        v = torch.zeros_like(k)
        kb = torch.zeros(1023, 8, 256, device="cuda", dtype=q.dtype)
        vb = torch.zeros_like(kb)
        from functools import partial

        eligible = partial(_can_use_cuda_verify, arch=(9, 0), sliding_window=True)
        self.assertTrue(eligible(q, k, v, kb, vb, 5))
        self.assertFalse(eligible(q, k, v, kb, vb, 6))
        self.assertFalse(eligible(q, k, v, kb, vb, 0))
        self.assertFalse(eligible(q.float(), k, v, kb, vb, 5))
        self.assertFalse(eligible(q, None, v, kb, vb, 5))
        self.assertFalse(eligible(q, k[:, :1], v[:, :1], kb, vb, 5))
        self.assertFalse(eligible(q, k, v, kb[:, :, ::2], vb[:, :, ::2], 5))
        strided = torch.empty(5, 16, 512, device="cuda", dtype=q.dtype)[:, :, ::2]
        self.assertFalse(eligible(strided, k, v, kb, vb, 5))

    def test_window_edges_and_graph_replay(self):
        if torch.version.hip or torch.cuda.get_device_capability() not in (
            (8, 9),
            (9, 0),
        ):
            self.skipTest("CUDA candidate validated on Ada and Hopper")
        torch.manual_seed(731)
        for window in (1, 3, 1023):
            for width in (1, 2, 3, 4, 5):
                with self.subTest(window=window, width=width):
                    self._check_window_case(window, width)

    def _check_window_case(self, window, width):
        device = "cuda"
        # Logical prefix lengths cross both sides of the boundary.
        lengths = [0, 1, max(0, window - 1), window, window + 1, 4096]
        bs, hq, hkv, dim = len(lengths), 16, 8, 256
        capacity = bs * window
        q = torch.randn(bs * width, hq * 2, dim, device=device).bfloat16()[:, ::2]
        k = torch.randn(bs * width, hkv, dim, device=device).bfloat16()
        v = torch.randn_like(k)
        # Noncontiguous physical pool slots deliberately differ from
        # logical token positions and request order.
        physical = torch.randperm(capacity * 2, device=device)[:capacity]
        kb = torch.randn(capacity * 2, hkv * 2, dim, device=device).bfloat16()[:, ::2]
        vb = torch.randn_like(kb)
        qo = torch.arange(bs + 1, device=device, dtype=torch.int32) * width
        kp = torch.zeros(bs + 1, device=device, dtype=torch.int32)
        ids = torch.zeros(capacity, device=device, dtype=torch.int64)
        offsets = torch.zeros(bs, device=device, dtype=torch.int32)
        out = torch.empty_like(q)

        def metadata(prefix_lengths):
            kept = [min(p, window) for p in prefix_lengths]
            starts = [0]
            for n in kept:
                starts.append(starts[-1] + n)
            kp.copy_(torch.tensor(starts, dtype=torch.int32, device=device))
            ids.zero_()
            ids[: starts[-1]].copy_(physical[: starts[-1]])
            offsets.copy_(
                torch.tensor(
                    [p - n for p, n in zip(prefix_lengths, kept)],
                    dtype=torch.int32,
                    device=device,
                )
            )
            return kept, starts

        def reference(prefix_lengths, kept, starts):
            ref = torch.empty_like(q, dtype=torch.float64)
            for b, prefix in enumerate(prefix_lengths):
                tail_start = prefix - kept[b]
                slots = physical[starts[b] : starts[b + 1]]
                pk, pv = kb[slots].double(), vb[slots].double()
                for j in range(width):
                    # Absolute-position rule, independent of the
                    # split kernel's retained-prefix coordinates.
                    first = max(0, prefix + j - window)
                    logical = torch.arange(tail_start, prefix, device=device)
                    valid = logical >= first
                    draft_first = max(0, j - window)
                    dk = k[b * width + draft_first : b * width + j + 1].double()
                    dv = v[b * width + draft_first : b * width + j + 1].double()
                    keys = torch.cat((pk[valid], dk), dim=0)
                    values = torch.cat((pv[valid], dv), dim=0)
                    keys = keys.repeat_interleave(hq // hkv, dim=1)
                    values = values.repeat_interleave(hq // hkv, dim=1)
                    scores = (
                        torch.einsum("hd,thd->ht", q[b * width + j].double(), keys)
                        * dim**-0.5
                    )
                    ref[b * width + j] = torch.einsum(
                        "ht,thd->hd", scores.softmax(-1), values
                    )
            return ref

        def run():
            self.assertTrue(
                verify_splitkv_fwd(
                    q,
                    k,
                    v,
                    out,
                    kb,
                    vb,
                    qo,
                    kp,
                    ids,
                    None,
                    True,
                    None,
                    width,
                    1.0,
                    1.0,
                    sm_scale=dim**-0.5,
                    sliding_window_size=window,
                    window_kv_offsets=offsets,
                    max_bs=bs,
                )
            )

        kept, starts = metadata(lengths)
        run()
        torch.testing.assert_close(
            out.double(),
            reference(lengths, kept, starts),
            atol=2e-2,
            rtol=1e-2,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        # Move across the window, including a new empty-prefix row,
        # while all captured tensor addresses remain unchanged.
        changed = list(reversed(lengths))
        kept, starts = metadata(changed)
        out.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(
            out.double(),
            reference(changed, kept, starts),
            atol=2e-2,
            rtol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
