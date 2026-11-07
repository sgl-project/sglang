"""Fused Kimi-K3 DFlash post-verify kernels against their unfused references.

* ``rope_store_prefix_valid``: k-norm output -> RoPE -> committed-prefix KV
  store in one kernel; bit-identical to RMSNorm + rope + the prefix-valid store.
* ``mamba_track_steps``: Mamba state-tracking steps after acceptance.
* ``verify_argmax``: target argmax over the verify logits, NaN rows included.
* ``compact_draft_lens``: sliding-window visible/offset lengths, page aligned.
* ``fill_draft_verify_metadata``: one-launch draft verify attention metadata.
"""

import unittest

import torch

from sglang.kernels.ops.attention.k3_draft_verify_metadata import (
    fill_draft_verify_metadata,
)
from sglang.kernels.ops.speculative.k3_dflash_kv_store import rope_store_prefix_valid
from sglang.kernels.ops.speculative.k3_dflash_post import (
    compact_draft_lens,
    mamba_track_steps,
    verify_argmax,
)
from sglang.srt.layers import layernorm
from sglang.srt.layers.rotary_embedding.factory import get_rope
from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_prefix_valid_impl
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase


def mamba_track_steps_reference(commit_lens, pre, post, interval, has_track):
    last = commit_lens.to(torch.int64) - 1
    if not has_track:
        return last, None
    to_track_mask = pre // interval != post // interval
    tracking_point = post // interval * interval
    to_track_ith = torch.clamp(tracking_point - pre - 1, min=0)
    can_track_mask = to_track_mask & (to_track_ith < commit_lens.to(to_track_ith.dtype))
    steps = torch.where(
        can_track_mask,
        to_track_ith.to(torch.int64),
        torch.full_like(to_track_ith, -1, dtype=torch.int64),
    )
    return last, steps


register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEV = "cuda"
HD, L = 128, 8  # draft head dim, tokens per verify step
SLOTS = 50000


def _bits(x):
    return x.view(torch.int16)


class TestK3DflashKvStore(CustomTestCase):
    def test_rope_store_bitwise(self):
        torch.manual_seed(0)
        rope = get_rope(HD, HD, 262144, 2000000, True, dtype=torch.bfloat16).to(DEV)
        norm = layernorm.RMSNorm(HD, eps=1e-5).to(DEV).to(torch.bfloat16)
        norm.weight.data.copy_(torch.rand(HD, device=DEV) * 2)
        for bs in (1, 3, 8, 16):
            t = bs * L
            qkv = (torch.randn(t, 768, device=DEV) * 3).to(torch.bfloat16)
            k, v = qkv[:, 512:640], qkv[:, 640:768]
            pos = torch.randint(0, 200000, (bs,), device=DEV)[:, None]
            pos = (pos + torch.arange(L, device=DEV)[None]).reshape(-1)
            loc = torch.randperm(SLOTS, device=DEV)[:t].view(bs, L)
            commit = torch.randint(1, L + 1, (bs,), device=DEV, dtype=torch.int32)
            kb0 = torch.zeros(SLOTS, 1, HD, device=DEV, dtype=torch.bfloat16)
            vb0, kb1, vb1 = kb0.clone(), kb0.clone(), kb0.clone()

            kn = norm(k.reshape(-1, HD)).view_as(k)
            kn_ref = kn.reshape(-1, HD).clone()
            _, kr = rope(pos, kn.new_empty(kn.shape), kn)
            _set_kv_buffer_prefix_valid_impl(
                kr.view(-1, 1, HD).contiguous(), v.reshape(-1, 1, HD).contiguous(),
                kb0, vb0, loc, commit, row_dim=HD, store_dtype=torch.bfloat16,
            )  # fmt: skip

            kn2 = layernorm.rms_norm(k, norm.weight.data, norm.variance_epsilon)
            self.assertTrue(torch.equal(_bits(kn2), _bits(kn_ref)), bs)
            rope_store_prefix_valid(
                kn2.view(-1, 1, HD), v.reshape(-1, 1, HD), pos,
                rope.cos_sin_cache_as(torch.bfloat16), loc, commit, kb1, vb1,
            )  # fmt: skip
            self.assertTrue(torch.equal(_bits(kb0), _bits(kb1)), bs)
            self.assertTrue(torch.equal(_bits(vb0), _bits(vb1)), bs)


class TestK3DflashPost(CustomTestCase):
    def test_mamba_track_steps(self):
        torch.manual_seed(0)
        for trial in range(50):
            n = [1, 4, 8, 16, 33][trial % 5]
            commit = torch.randint(1, 9, (n,), device=DEV, dtype=torch.int32)
            pre = torch.randint(0, 100000, (n,), device=DEV)
            if trial % 4 == 0:
                pre[0] = -5
            post = pre + commit.long()
            for interval in (256, 64, 1):
                for has in (True, False):
                    a = mamba_track_steps(commit, pre, post, interval, has)
                    b = mamba_track_steps_reference(commit, pre, post, interval, has)
                    self.assertTrue(torch.equal(a[0], b[0]))
                    if has:
                        self.assertTrue(torch.equal(a[1], b[1]), (trial, interval))

    def test_verify_argmax(self):
        torch.manual_seed(0)
        for rows in (8, 32, 64, 128):
            for trial in range(10):
                x = torch.randn(rows, 163840, device=DEV)
                if trial % 3 == 1:
                    x = x.round()
                if trial % 3 == 2:
                    r = min(1, rows - 1)
                    x[0, 1000] = x[r, 5] = x[r, 3] = float("nan")
                self.assertTrue(
                    torch.equal(verify_argmax(x), torch.argmax(x, dim=-1)),
                    (rows, trial),
                )

    def test_compact_draft_lens(self):
        def ref(seq, window, page):
            vis = torch.clamp(seq.int(), max=window)
            if page <= 1:
                d = vis
            else:
                start = seq - vis.long()
                d = (seq - (start - torch.remainder(start, page))).int()
            return d, seq - d.long()

        torch.manual_seed(0)
        edges = [0, 1, 4095, 4096, 4097, 4159, 4160, 4161, 63, 64]
        for trial in range(100):
            seq = torch.randint(0, 300000, ([1, 3, 8, 16, 64][trial % 5],), device=DEV)
            seq[0] = edges[trial % len(edges)]
            for window, page in ((4096, 64), (4096, 1), (100, 16)):
                a, b = compact_draft_lens(seq, window, page), ref(seq, window, page)
                self.assertTrue(torch.equal(a[0], b[0]), (window, page))
                self.assertTrue(torch.equal(a[1], b[1]), (window, page))


class TestK3DraftVerifyMetadata(CustomTestCase):
    def test_matches_reference(self):
        torch.manual_seed(0)
        r2t = torch.randint(0, 10**6, (70, 9000), device=DEV, dtype=torch.int32)

        def filled(n, dtype):
            return torch.full((n,), -7, dtype=dtype, device=DEV)

        def cumsum0(x):
            out = torch.zeros(x.numel() + 1, dtype=torch.int64, device=DEV)
            out[1:] = torch.cumsum(x, 0)
            return out

        for bs in (1, 2, 5, 8, 16, 64):
            for window in (4096, 0):
                seq = torch.randint(1, 8900, (bs,), device=DEV, dtype=torch.int32)
                seq[0] = 4160 if bs > 1 else 4100
                rpi = torch.randperm(70, device=DEV)[:bs]
                qo, mip = filled(bs + 2, torch.int64), filled(bs + 2, torch.int64)
                kvp, wp = filled(bs + 2, torch.int32), filled(bs + 2, torch.int32)
                kvi, wi = filled(bs * 9000, torch.int64), filled(bs * 9000, torch.int64)
                wo = filled(bs + 1, torch.int32)
                fill_draft_verify_metadata(
                    bs=bs, seq_lens=seq, req_pool_indices=rpi, req_to_token=r2t,
                    num_tokens_per_req=L, qo_indptr=qo, kv_indptr=kvp,
                    kv_indices=kvi, mask_indptr=mip, window_size=window or None,
                    window_kv_indptr=wp, window_kv_indices=wi, window_kv_offsets=wo,
                )  # fmt: skip
                s64 = seq.long()
                ref_kvi = torch.cat([r2t[rpi[b], : seq[b]].long() for b in range(bs)])
                self.assertTrue(
                    torch.equal(
                        qo[: bs + 1], torch.arange(0, (bs + 1) * L, L, device=DEV)
                    )
                )
                self.assertTrue(torch.equal(kvp[: bs + 1].long(), cumsum0(s64)))
                self.assertTrue(torch.equal(kvi[: ref_kvi.numel()], ref_kvi))
                self.assertEqual(int(kvi[ref_kvi.numel()]), -7)
                self.assertTrue(torch.equal(mip[: bs + 1], cumsum0(L * (s64 + L))))
                if window:
                    wl = torch.clamp(s64, max=window)
                    off = s64 - wl
                    ref_wi = torch.cat(
                        [r2t[rpi[b], off[b] : off[b] + wl[b]].long() for b in range(bs)]
                    )
                    self.assertTrue(torch.equal(wp[: bs + 1].long(), cumsum0(wl)))
                    self.assertTrue(torch.equal(wo[:bs].long(), off))
                    self.assertTrue(torch.equal(wi[: ref_wi.numel()], ref_wi))


if __name__ == "__main__":
    unittest.main()
