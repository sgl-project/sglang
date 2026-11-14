"""Fused Kimi-K3 DFlash post-verify kernels against their unfused references.

* ``rope_store_prefix_valid``: k-norm output -> RoPE -> committed-prefix KV
  store in one kernel; bit-identical to RMSNorm + rope + the prefix-valid store.
* ``mamba_track_steps``: Mamba state-tracking steps after acceptance.
* ``verify_argmax``: target argmax over the verify logits, NaN rows included.
* ``compact_draft_lens``: sliding-window visible/offset lengths, page aligned.
"""

import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
