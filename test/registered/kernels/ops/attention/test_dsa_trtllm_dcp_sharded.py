"""Kimi-K3.1 trtllm DSA under sharded DCP: fused q concat-cast into a head-padded
fp8 buffer, and per-rank trtllm-gen partials (owned top-k rows compacted, seq_lens =
owned count, base-2 LSE) merging to the single-rank result. The prims_ts backend
reads the same raw FP8 rows and owned-row tables; its partials carry a natural-log
LSE."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention.mla_qk_prep import concat_and_cast_q_fp8_heads
from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_cast_triton
from sglang.srt.layers.attention.dsa.dsa_dcp import dcp_trtllm_sparse_attention
from sglang.srt.layers.attention.dsa.dsa_dcp_sharded import _localize_compact_kernel
from sglang.srt.layers.attention.dsa.prims_ts import (
    _EXACT_PLAN_MAX_ROWS,
    PrimsTSSparseMLA,
    check_prims_ts_sparse_mla_support,
)
from sglang.srt.layers.attention.dsa_backend import (
    _validate_dcp_backend,
    _validate_prims_ts_dcp,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

FP8 = torch.float8_e4m3fn
KV_LORA, ROPE, QK_NOPE, PAGE, TOPK = 512, 64, 128, 64, 2048
DIM = KV_LORA + ROPE
SM_SCALE = (QK_NOPE + ROPE) ** -0.5


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


class TestConcatCastQFp8Heads(CustomTestCase):
    def test_matches_cast_and_keeps_pad_heads(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        gen = torch.Generator(device="cuda").manual_seed(0)
        for heads, target in ((12, 12), (48, 48), (96, 128)):
            with self.subTest(heads=heads):
                q = torch.randn(37, heads, DIM, generator=gen, device="cuda")
                q = (q * 4).to(torch.bfloat16)
                # strided views of one [T, H, 576] q, as the model passes them
                q_nope, q_rope = q[..., :KV_LORA], q[..., KV_LORA:]
                out = torch.zeros(37, target, DIM, dtype=FP8, device="cuda")
                concat_and_cast_q_fp8_heads(out, q_nope, q_rope)
                ref = q.to(FP8)
                self.assertTrue(
                    torch.equal(out[:, :heads].view(torch.uint8), ref.view(torch.uint8))
                )
                self.assertFalse(out[:, heads:].view(torch.uint8).any().item())


class TestKvWriteCastOnStore(CustomTestCase):
    def test_matches_cast_and_owner_rule(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        gen = torch.Generator(device="cuda").manual_seed(2)
        n, slots = 300, 4096
        # strided nope / rope views of one latent row, as the model passes them
        latent = (torch.randn(n, DIM, generator=gen, device="cuda") * 3).to(
            torch.bfloat16
        )
        nope, rope = latent[:, :KV_LORA], latent[:, KV_LORA:]
        loc = torch.randperm(slots, generator=gen, device="cuda")[:n].to(torch.int64)
        loc[0] = 0  # reserved padding slot: skipped
        for world in (1, 8):
            for rank in range(world):
                with self.subTest(world=world, rank=rank):
                    buf = torch.zeros(slots // world, DIM, dtype=FP8, device="cuda")
                    set_mla_kv_buffer_cast_triton(
                        buf, loc, nope, rope, dcp_world_size=world, dcp_rank=rank
                    )
                    ref = torch.zeros_like(buf)
                    own = (loc != 0) & (loc % world == rank)
                    ref[loc[own] // world] = latent[own].to(FP8)
                    self.assertTrue(
                        torch.equal(buf.view(torch.uint8), ref.view(torch.uint8))
                    )


def _localize(slots: torch.Tensor, dcp: int, rank: int):
    out = torch.empty_like(slots)
    lens = torch.empty(slots.shape[0], dtype=torch.int32, device=slots.device)
    _localize_compact_kernel[(slots.shape[0],)](
        slots, slots.stride(0), out, out.stride(0), lens, dcp, rank, TOPK=TOPK
    )
    return out, lens


class TestShardedDcpPartials(CustomTestCase):
    def _attend(self, kv, q8, table, seq_lens, workspace, counter):
        rows, heads, _ = q8.shape
        return dcp_trtllm_sparse_attention(
            query=q8.view(rows, 1, heads, DIM),
            block_tables=table.unsqueeze(1),
            seq_lens=seq_lens,
            cu_seqlens_q=torch.arange(rows + 1, dtype=torch.int32, device="cuda"),
            kv_cache=kv.view(-1, 1, PAGE, DIM),
            workspace_buffer=workspace,
            qk_nope_head_dim=QK_NOPE,
            kv_lora_rank=KV_LORA,
            qk_rope_head_dim=ROPE,
            max_seq_len=kv.shape[0],
            sparse_mla_top_k=TOPK,
            bmm1_scale=SM_SCALE,
            backend="trtllm-gen",
            multi_ctas_kv_counter_buffer=counter,
        )

    def test_dcp8_partials_merge_to_full(self):
        if not _sm100():
            self.skipTest("trtllm-gen sparse MLA needs SM100")
        from sglang.srt.layers.attention.trtllm_mla_backend import (
            make_persistent_multi_ctas_kv_counter_buffer,
        )

        dcp, heads, padded, rows, ctx = 8, 96, 128, 24, 16384
        gen = torch.Generator(device="cuda").manual_seed(1)
        kv = (torch.randn(ctx, DIM, generator=gen, device="cuda") * 0.5).to(FP8)
        q = (torch.randn(rows, heads, DIM, generator=gen, device="cuda")).to(
            torch.bfloat16
        )
        q8 = torch.zeros(rows, padded, DIM, dtype=FP8, device="cuda")
        concat_and_cast_q_fp8_heads(q8, q[..., :KV_LORA], q[..., KV_LORA:])

        # global top-k slots, valid first then -1; row 0 has keys on rank 0
        # only, so the other ranks see an empty row
        slots = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
        counts = torch.randint(1, TOPK + 1, (rows,), generator=gen, device="cuda")
        counts[1] = TOPK
        for r in range(rows):
            n = int(counts[r])
            pick = torch.randperm(ctx, generator=gen, device="cuda")[:n]
            if r == 0:
                pick = torch.arange(0, ctx, dcp, device="cuda")[:n]
            slots[r, :n] = pick.to(torch.int32)
        counts = (slots >= 0).sum(1).to(torch.int32)

        workspace = torch.zeros(384 << 20, dtype=torch.uint8, device="cuda")
        counter = make_persistent_multi_ctas_kv_counter_buffer(
            torch.device("cuda"), padded, max_batch_size=rows
        )
        full, full_lse = self._attend(kv, q8, slots, counts, workspace, counter)
        full = full.view(rows, padded, KV_LORA)[:, :heads].float()

        outs, lses = [], []
        for rank in range(dcp):
            local_kv = kv[rank::dcp].contiguous()
            table, lens = _localize(slots, dcp, rank)
            o, lse = self._attend(local_kv, q8, table, lens, workspace, counter)
            outs.append(o.view(rows, padded, KV_LORA)[:, :heads].float())
            lses.append(lse[:, :heads])
        lse = torch.stack(lses)  # base 2
        w = torch.exp2(lse - lse.max(0).values)
        merged = (w.unsqueeze(-1) * torch.stack(outs)).sum(0) / w.sum(0).unsqueeze(-1)
        merged_lse = torch.log2(w.sum(0)) + lse.max(0).values

        # fp32 attention over the same fp8 q / KV: trtllm-gen's own error
        # (fp8 P) is ~2%, so compare both paths against it, not each other.
        qf, kvf = q8[:, :heads].float(), kv.float()
        ref = torch.empty_like(full)
        for r in range(rows):
            k = kvf[slots[r, : counts[r]].long()]
            p = torch.softmax((qf[r] @ k.T) * SM_SCALE, dim=-1)
            ref[r] = p @ k[:, :KV_LORA]

        def rel(a):
            return ((a - ref).norm() / ref.norm()).item()

        self.assertLess(rel(full), 0.03)
        self.assertLess(rel(merged), rel(full) * 1.2)
        torch.testing.assert_close(
            merged_lse, full_lse[:, :heads], atol=1e-3, rtol=1e-3
        )


class TestShardedDcpPrimsTs(CustomTestCase):
    def _require_prims_ts(self):
        if not _sm100():
            self.skipTest("Prims-TS sparse MLA needs SM100/SM103/SM107")
        try:
            check_prims_ts_sparse_mla_support(
                device=torch.device("cuda"),
                num_heads=96,
                kv_lora_rank=KV_LORA,
                qk_rope_head_dim=ROPE,
            )
        except ValueError as exc:
            self.skipTest(str(exc))
        except RuntimeError as exc:
            # The image's FlashInfer 0.7.2 lacks the D576 native-FP8 kernel.
            self.skipTest(f"Skipped until the FlashInfer bump: {exc}")

    def _check_dcp8_partials_merge_to_full(self, rows: int):
        """Per-rank Prims-TS partials (natural-log LSE; a rank owning none of a
        row's keys returns zero output and -inf LSE) merge by exp weights, the base
        TestPrimsTsDcpLseBase pins for the combine, to the single-rank result."""
        self._require_prims_ts()
        dcp, heads, ctx = 8, 96, 16384
        gen = torch.Generator(device="cuda").manual_seed(3)
        kv = (torch.randn(ctx, DIM, generator=gen, device="cuda") * 0.5).to(FP8)
        q8 = torch.randn(rows, heads, DIM, generator=gen, device="cuda").to(FP8)
        slots = torch.full((rows, TOPK), -1, dtype=torch.int32, device="cuda")
        counts = torch.randint(1, TOPK + 1, (rows,), generator=gen, device="cuda")
        counts[1] = TOPK
        for r in range(rows):
            n = int(counts[r])
            pick = torch.randperm(ctx, generator=gen, device="cuda")[:n]
            if r == 0:
                pick = torch.arange(0, ctx, dcp, device="cuda")[:n]
            slots[r, :n] = pick.to(torch.int32)
        counts = (slots >= 0).sum(1).to(torch.int32)

        runner = PrimsTSSparseMLA(torch.device("cuda"))

        def attend(cache, table, lens):
            return runner.forward(
                q=q8,
                kv_cache=cache,
                indices=table,
                lengths=lens,
                sm_scale=SM_SCALE,
                kv_scale=1.0,
            )

        full, full_lse = attend(kv, slots, counts)
        outs, lses = [], []
        for rank in range(dcp):
            table, lens = _localize(slots, dcp, rank)
            o, lse = attend(kv[rank::dcp].contiguous(), table, lens)
            if rank:
                self.assertTrue(torch.isneginf(lse[0]).all().item())
                self.assertFalse(o[0].any().item())
            outs.append(o.float())
            lses.append(lse)
        # (rows, heads, width, packed) plan keys: rows past the exact limit pack
        packed = rows > _EXACT_PLAN_MAX_ROWS
        self.assertEqual({key[3] for key in runner._plans}, {packed})
        lse = torch.stack(lses)  # natural log
        w = torch.exp(lse - lse.max(0).values)
        merged = (w.unsqueeze(-1) * torch.stack(outs)).sum(0) / w.sum(0).unsqueeze(-1)
        merged_lse = torch.log(w.sum(0)) + lse.max(0).values

        qf, kvf = q8.float(), kv.float()
        ref = torch.empty_like(full, dtype=torch.float32)
        ref_lse = torch.empty_like(full_lse)
        for r in range(rows):
            k = kvf[slots[r, : counts[r]].long()]
            s = (qf[r] @ k.T) * SM_SCALE
            ref_lse[r] = torch.logsumexp(s, dim=-1)
            ref[r] = torch.softmax(s, dim=-1) @ k[:, :KV_LORA]

        def rel(a):
            return ((a.float() - ref).norm() / ref.norm()).item()

        self.assertLess(rel(full), 0.03)
        self.assertLess(rel(merged), rel(full) * 1.2)
        torch.testing.assert_close(full_lse, ref_lse, atol=2e-2, rtol=2e-3)
        torch.testing.assert_close(merged_lse, full_lse, atol=1e-3, rtol=1e-3)

    def test_dcp8_partials_merge_to_full(self):
        self._check_dcp8_partials_merge_to_full(rows=24)

    def test_packed_rows_dcp8_partials_merge_to_full(self):
        # Past the exact-plan limit: power-of-two packed capacity with qo_indptr.
        self._check_dcp8_partials_merge_to_full(rows=_EXACT_PLAN_MAX_ROWS + 1)


class TestPrimsTsDcpLseBase(CustomTestCase):
    """Prims-TS partials carry a natural-log LSE, and the DCP combine picks the LSE
    base per DSA DCP mode; each mode must merge them base e or reject prims_ts."""

    def _combine_base_e(self, *, shards_index_k: bool, decode_backend: str) -> bool:
        exec_ = SimpleNamespace(
            kernel=SimpleNamespace(dsa_decode_backend=decode_backend)
        )
        with (
            mock.patch.object(
                forward_mla, "dsa_shards_index_k", return_value=shards_index_k
            ),
            mock.patch.object(forward_mla, "get_exec", return_value=exec_),
        ):
            return forward_mla.is_mla_dcp_lse_base_on_e("dsa")

    def test_sharded_index_k_merges_base_e(self):
        _validate_prims_ts_dcp(dcp_size=8, shards_index_k=True)
        self.assertTrue(
            self._combine_base_e(shards_index_k=True, decode_backend="prims_ts")
        )
        # trtllm-gen partials are base 2
        self.assertFalse(
            self._combine_base_e(shards_index_k=True, decode_backend="trtllm")
        )

    def test_replicated_index_k_rejects_prims_ts(self):
        # Replicated index-K (non-K3.1 DSA models) merges base 2 for every kernel.
        self.assertFalse(
            self._combine_base_e(shards_index_k=False, decode_backend="prims_ts")
        )
        with self.assertRaisesRegex(ValueError, "sharded index-K"):
            _validate_prims_ts_dcp(dcp_size=2, shards_index_k=False)
        # Without DCP there is no merge.
        _validate_prims_ts_dcp(dcp_size=1, shards_index_k=False)


class TestValidateDcpBackend(CustomTestCase):
    def test_both_phases_one_backend(self):
        common = dict(kimi_k31=True, use_fused_topk=True, use_topk_v2=True)
        for impl in ("flashmla_kv", "trtllm", "prims_ts"):
            _validate_dcp_backend(prefill_impl=impl, decode_impl=impl, **common)
        for prefill, decode in (
            ("flashmla_kv", "trtllm"),
            ("trtllm", "prims_ts"),
        ):
            with self.assertRaisesRegex(ValueError, "both flashmla_kv, both trtllm"):
                _validate_dcp_backend(
                    prefill_impl=prefill, decode_impl=decode, **common
                )


if __name__ == "__main__":
    unittest.main()
