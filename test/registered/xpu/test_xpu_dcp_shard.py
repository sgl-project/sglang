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


import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
from sglang.srt.layers.dcp import dcp_shard_page_table, get_dcp_lens
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=120, suite="stage-b-test-1-gpu-xpu")

DEVICE = "xpu"
DTYPE = torch.bfloat16
BF16_ULP = 2.0**-8


def _kernel_emits_lse() -> bool:
    try:
        return "return_softmax_lse" in str(torch.ops.sgl_kernel.fwd.default._schema)
    except Exception:
        return False


_SKIP_REASON = None
if not is_xpu():
    _SKIP_REASON = "requires an Intel XPU"
elif not _kernel_emits_lse():
    _SKIP_REASON = "installed sgl-kernel-xpu does not emit softmax_lse"


def _ref_attn(q, k, v, scale, causal=False):
    n_q, h_q, _ = q.shape
    n_k, h_kv, _ = k.shape
    group = h_q // h_kv
    out = torch.empty(n_q, h_q, v.shape[-1], device=q.device, dtype=torch.float32)
    lse = torch.empty(n_q, h_q, device=q.device, dtype=torch.float32)
    for h in range(h_q):
        s = (q.float()[:, h] @ k.float()[:, h // group].T) * scale
        if causal:
            rows = torch.arange(n_q, device=q.device).unsqueeze(1)
            cols = torch.arange(n_k, device=q.device).unsqueeze(0)
            s = s.masked_fill(cols > rows + (n_k - n_q), -float("inf"))
        lse[:, h] = torch.logsumexp(s, dim=-1)
        out[:, h] = torch.softmax(s, dim=-1) @ v.float()[:, h // group]
    return out, lse


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class TestXPUFmhaSoftmaxLse(CustomTestCase):
    def _check_paged(self, kv_len, page_size, head_dim, h_q, h_kv, n_q=1, num_splits=0):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        torch.manual_seed(0)
        scale = head_dim**-0.5
        num_pages = (kv_len + page_size - 1) // page_size
        k = torch.zeros(
            num_pages * page_size, h_kv, head_dim, device=DEVICE, dtype=DTYPE
        )
        v = torch.zeros_like(k)
        k[:kv_len] = torch.randn(kv_len, h_kv, head_dim, device=DEVICE, dtype=DTYPE)
        v[:kv_len] = torch.randn(kv_len, h_kv, head_dim, device=DEVICE, dtype=DTYPE)
        q = torch.randn(n_q, h_q, head_dim, device=DEVICE, dtype=DTYPE)

        out, lse, *_ = flash_attn_with_kvcache(
            q=q,
            k_cache=k.view(-1, page_size, h_kv, head_dim),
            v_cache=v.view(-1, page_size, h_kv, head_dim),
            page_table=torch.arange(
                num_pages, device=DEVICE, dtype=torch.int32
            ).unsqueeze(0),
            cache_seqlens=torch.tensor([kv_len], device=DEVICE, dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, n_q], device=DEVICE, dtype=torch.int32),
            cu_seqlens_k_new=None,
            max_seqlen_q=n_q,
            softmax_scale=scale,
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            return_softmax_lse=True,
            num_splits=num_splits,
        )

        ref_out, ref_lse = _ref_attn(q, k[:kv_len], v[:kv_len], scale)
        torch.testing.assert_close(
            lse.transpose(0, 1).float(), ref_lse, atol=1e-3, rtol=1e-4
        )
        tol = 4 * BF16_ULP * max(1.0, ref_out.abs().max().item())
        torch.testing.assert_close(out.float(), ref_out, atol=tol, rtol=0)

    def test_paged_decode_lse(self):
        for head_dim in (64, 96, 128):
            for page_size in (64, 128):
                for kv_len in (1, 63, 64, 129, 512, 2000):
                    with self.subTest(d=head_dim, page=page_size, kv=kv_len):
                        self._check_paged(kv_len, page_size, head_dim, 8, 1)

    def test_paged_decode_lse_split_kv(self):
        for kv_len in (8192, 16384, 32768):
            for page_size in (64, 128):
                with self.subTest(kv=kv_len, page=page_size):
                    self._check_paged(kv_len, page_size, 128, 8, 1)

    def test_paged_decode_lse_forced_num_splits(self):
        for kv_len in (512, 8192):
            for page_size in (64, 128):
                with self.subTest(kv=kv_len, page=page_size):
                    self._check_paged(kv_len, page_size, 128, 8, 1, num_splits=4)

    def test_paged_decode_lse_serving_head_shapes(self):
        for h_q in (8, 16):
            for kv_len in (2000, 32768):
                with self.subTest(h_q=h_q, kv=kv_len):
                    self._check_paged(kv_len, 64, 128, h_q, 1)

    def test_paged_lse_grouped_heads(self):
        for page_size in (64, 128):
            for n_q in (1, 5):
                with self.subTest(page=page_size, n_q=n_q):
                    self._check_paged(300, page_size, 128, 4, 2, n_q=n_q)

    def test_varlen_prefill_lse(self):
        from sgl_kernel.flash_attn import flash_attn_varlen_func

        for head_dim in (64, 128):
            for causal in (True, False):
                with self.subTest(d=head_dim, causal=causal):
                    seqs = [7, 64, 129, 500]
                    torch.manual_seed(0)
                    scale = head_dim**-0.5
                    total = sum(seqs)
                    h_q, h_kv = 8, 1
                    q = torch.randn(total, h_q, head_dim, device=DEVICE, dtype=DTYPE)
                    k = torch.randn(total, h_kv, head_dim, device=DEVICE, dtype=DTYPE)
                    v = torch.randn(total, h_kv, head_dim, device=DEVICE, dtype=DTYPE)
                    cu = torch.tensor(
                        [0] + torch.cumsum(torch.tensor(seqs), 0).tolist(),
                        device=DEVICE,
                        dtype=torch.int32,
                    )

                    _, lse, *_ = flash_attn_varlen_func(
                        q=q,
                        k=k,
                        v=v,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=max(seqs),
                        max_seqlen_k=max(seqs),
                        softmax_scale=scale,
                        causal=causal,
                        return_softmax_lse=True,
                    )

                    ref = torch.empty(total, h_q, device=DEVICE, dtype=torch.float32)
                    for b, length in enumerate(seqs):
                        s0 = int(cu[b])
                        sl = slice(s0, s0 + length)
                        _, ref[sl] = _ref_attn(q[sl], k[sl], v[sl], scale, causal)
                    torch.testing.assert_close(
                        lse.transpose(0, 1).float(), ref, atol=1e-3, rtol=1e-4
                    )


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class TestXPUDCPPagedShard(CustomTestCase):
    HEAD_DIM = 128

    def _build(self, seq_lens, page_size, dcp_size, h_kv, seed=0):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        d = self.HEAD_DIM
        bs = len(seq_lens)
        max_len = max(seq_lens)
        widened_page = page_size * dcp_size

        req_to_token = torch.zeros(bs, max_len + widened_page, dtype=torch.int64)
        total_slots = 0
        for i, length in enumerate(seq_lens):
            base = total_slots
            req_to_token[i, :length] = base + torch.arange(length)
            total_slots = base + -(-length // widened_page) * widened_page

        phys_slots = total_slots // dcp_size
        phys_k = [
            torch.zeros(phys_slots, h_kv, d, dtype=DTYPE) for _ in range(dcp_size)
        ]
        phys_v = [torch.zeros_like(t) for t in phys_k]

        pages_per_req = -(-max_len // page_size)
        dense_k = torch.zeros(bs * pages_per_req * page_size, h_kv, d, dtype=DTYPE)
        dense_v = torch.zeros_like(dense_k)
        dense_table = torch.arange(bs * pages_per_req, dtype=torch.int32).view(
            bs, pages_per_req
        )

        for i, length in enumerate(seq_lens):
            k = torch.randn(length, h_kv, d, generator=gen)
            v = torch.randn(length, h_kv, d, generator=gen)
            locs = req_to_token[i, :length]
            pos = torch.arange(length)
            for rank in range(dcp_size):
                owned = pos % dcp_size == rank
                phys_k[rank][locs[owned] // dcp_size] = k[owned].to(DTYPE)
                phys_v[rank][locs[owned] // dcp_size] = v[owned].to(DTYPE)
            base = int(dense_table[i, 0]) * page_size
            dense_k[base : base + length] = k.to(DTYPE)
            dense_v[base : base + length] = v.to(DTYPE)

        to_dev = lambda t: t.to(DEVICE)  # noqa: E731
        return (
            to_dev(req_to_token),
            [to_dev(t) for t in phys_k],
            [to_dev(t) for t in phys_v],
            to_dev(dense_k),
            to_dev(dense_v),
            to_dev(dense_table),
        )

    def _assert_table_matches_writes(
        self, req_to_token, seq_lens, page_size, dcp_size, rank, table
    ):
        for i, length in enumerate(seq_lens):
            owned = [p for p in range(length) if p % dcp_size == rank]
            for li, pos in enumerate(owned):
                written = int(req_to_token[i, pos]) // dcp_size
                looked_up = int(table[i, li // page_size]) * page_size + li % page_size
                self.assertEqual(
                    written,
                    looked_up,
                    f"req {i} local {li} (pos {pos}): page table resolves to slot "
                    f"{looked_up}, the KV write used {written}",
                )

    def _run_case(self, seq_lens, q_lens, page_size, dcp_size, h_kv, h_q):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        d = self.HEAD_DIM
        scale = d**-0.5
        req_to_token, phys_k, phys_v, dense_k, dense_v, dense_table = self._build(
            seq_lens, page_size, dcp_size, h_kv
        )
        bs = len(seq_lens)
        torch.manual_seed(1)
        total_q = sum(q_lens)
        q = torch.randn(total_q, h_q, d, device=DEVICE, dtype=DTYPE)
        cu_q = torch.tensor(
            [0] + torch.cumsum(torch.tensor(q_lens), 0).tolist(),
            device=DEVICE,
            dtype=torch.int32,
        )
        req_pool_indices = torch.arange(bs, device=DEVICE)
        lens = torch.tensor(seq_lens, device=DEVICE)

        ref = torch.empty(total_q, h_q, d, device=DEVICE, dtype=torch.float32)
        for i, length in enumerate(seq_lens):
            base = int(dense_table[i, 0]) * page_size
            q_slice = slice(int(cu_q[i]), int(cu_q[i + 1]))
            ref[q_slice], _ = _ref_attn(
                q[q_slice],
                dense_k[base : base + length],
                dense_v[base : base + length],
                scale,
            )

        outs, lses = [], []
        for rank in range(dcp_size):
            local = get_dcp_lens(lens, dcp_size, rank)
            max_local = int(get_dcp_lens(torch.tensor(seq_lens), dcp_size, rank).max())
            table = dcp_shard_page_table(
                req_to_token, req_pool_indices, page_size, dcp_size, rank, max_local
            )
            if table is None:
                outs.append(torch.zeros_like(ref))
                lses.append(
                    torch.full(
                        (total_q, h_q),
                        -float("inf"),
                        device=DEVICE,
                        dtype=torch.float32,
                    )
                )
                continue
            self._assert_table_matches_writes(
                req_to_token, seq_lens, page_size, dcp_size, rank, table
            )
            out, lse, *_ = flash_attn_with_kvcache(
                q=q,
                k_cache=phys_k[rank].view(-1, page_size, h_kv, d),
                v_cache=phys_v[rank].view(-1, page_size, h_kv, d),
                page_table=table,
                cache_seqlens=local.to(torch.int32),
                cu_seqlens_q=cu_q,
                cu_seqlens_k_new=None,
                max_seqlen_q=max(q_lens),
                softmax_scale=scale,
                causal=False,
                window_size=(-1, -1),
                softcap=0.0,
                return_softmax_lse=True,
            )
            empty = (local == 0).repeat_interleave(cu_q[1:] - cu_q[:-1])
            outs.append(out.float())
            lses.append(
                lse.transpose(0, 1)
                .float()
                .contiguous()
                .masked_fill_(empty.unsqueeze(-1), -float("inf"))
            )

        global_lse = torch.logsumexp(torch.stack(lses), dim=0)
        merged = torch.zeros_like(ref)
        for out, lse in zip(outs, lses):
            weight = torch.nan_to_num(torch.exp(lse - global_lse), nan=0.0, neginf=0.0)
            merged += torch.nan_to_num(out, nan=0.0) * weight.unsqueeze(-1)

        self.assertTrue(torch.isfinite(merged).all(), "merged output has NaN/inf")
        tol = 4 * BF16_ULP * max(1.0, ref.abs().max().item())
        torch.testing.assert_close(merged, ref, atol=tol, rtol=0)

    def test_decode_shard(self):
        seq_lens = [1, 63, 64, 65, 200, 1000]
        for page_size in (64, 128):
            for dcp_size in (2, 4):
                for h_kv, h_q in ((1, 8), (2, 8)):
                    with self.subTest(page=page_size, dcp=dcp_size, h_kv=h_kv):
                        self._run_case(
                            seq_lens,
                            [1] * len(seq_lens),
                            page_size,
                            dcp_size,
                            h_kv,
                            h_q,
                        )

    def test_prefix_shard_multi_query(self):
        seq_lens = [64, 129, 300, 777]
        for page_size in (64, 128):
            for dcp_size in (2, 4):
                with self.subTest(page=page_size, dcp=dcp_size):
                    self._run_case(seq_lens, [3, 7, 1, 16], page_size, dcp_size, 1, 8)

    def test_ranks_with_empty_shard(self):
        seq_lens = [1, 2, 3, 130]
        for page_size in (64, 128):
            for dcp_size in (2, 4):
                with self.subTest(page=page_size, dcp=dcp_size):
                    self._run_case(
                        seq_lens, [1] * len(seq_lens), page_size, dcp_size, 1, 8
                    )


@unittest.skipIf(not is_xpu(), "requires an Intel XPU")
class TestSingleTokenSelfAttn(CustomTestCase):
    def test_matches_fp32_reference(self):
        for head_dim, h_q, h_kv, logit_cap in (
            (128, 8, 1, 0.0),
            (128, 8, 2, 0.0),
            (64, 4, 4, 0.0),
            (128, 8, 1, 30.0),
        ):
            with self.subTest(d=head_dim, hq=h_q, hkv=h_kv, cap=logit_cap):
                torch.manual_seed(0)
                tokens, scale = 5, head_dim**-0.5
                layer = SimpleNamespace(
                    scaling=scale, logit_cap=logit_cap, v_head_dim=head_dim
                )
                q = torch.randn(tokens, h_q, head_dim, device=DEVICE, dtype=DTYPE)
                k = torch.randn(tokens, h_kv, head_dim, device=DEVICE, dtype=DTYPE)
                v = torch.randn(tokens, h_kv, head_dim, device=DEVICE, dtype=DTYPE)

                out, lse = XPUAttentionBackend._single_token_self_attn(q, k, v, layer)

                # One token per sequence, so each row is its own single-key batch.
                for t in range(tokens):
                    ref_out, ref_lse = _ref_attn(
                        q[t : t + 1], k[t : t + 1], v[t : t + 1], scale
                    )
                    if logit_cap > 0:
                        ref_lse = logit_cap * torch.tanh(ref_lse / logit_cap)
                    torch.testing.assert_close(out[t], ref_out[0], atol=1e-3, rtol=1e-4)
                    torch.testing.assert_close(lse[t], ref_lse[0], atol=1e-3, rtol=1e-4)

    def test_output_is_v_broadcast_over_the_query_group(self):
        layer = SimpleNamespace(scaling=0.1, logit_cap=0.0, v_head_dim=128)
        q = torch.randn(3, 8, 128, device=DEVICE, dtype=DTYPE)
        k = torch.randn(3, 2, 128, device=DEVICE, dtype=DTYPE)
        v = torch.randn(3, 2, 128, device=DEVICE, dtype=DTYPE)

        out, _ = XPUAttentionBackend._single_token_self_attn(q, k, v, layer)

        # Softmax over a single score is 1, so out is v under the KV-head grouping.
        torch.testing.assert_close(out, v.float().repeat_interleave(4, dim=1))


class TestDCPShardPageTableCPU(CustomTestCase):
    def test_table_is_none_when_rank_owns_nothing(self):
        table = dcp_shard_page_table(
            torch.zeros(2, 64, dtype=torch.int64),
            torch.arange(2),
            page_size=64,
            dcp_size=2,
            dcp_rank=1,
            max_local_len=0,
        )
        self.assertIsNone(table)

    def test_table_columns_follow_the_owner_rule(self):
        page_size, dcp_size = 64, 4
        req_to_token = torch.arange(page_size * dcp_size * 3).unsqueeze(0)
        for rank in range(dcp_size):
            table = dcp_shard_page_table(
                req_to_token,
                torch.zeros(1, dtype=torch.int64),
                page_size,
                dcp_size,
                rank,
                max_local_len=page_size * 3,
            )
            self.assertEqual(list(table.shape), [1, 3])
            self.assertEqual(table.dtype, torch.int32)
            self.assertEqual(table[0].tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
