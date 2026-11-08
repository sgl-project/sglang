"""KPool metadata kernels take the batch size at run time: one compiled variant serves every replay
batch size up to the 64-row block floor, and outputs do not depend on how requests are batched."""

import unittest

import torch

from sglang.kernels.ops.attention.dsa.kpool_metadata import decode, draft_extend, verify
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

WIDTH, PAGE, TOPK, KPOOL, NEXT_N, MAX_EXTEND = 8192, 64, 2048, 4, 6, 6
BATCH_SIZES = [1, 2, 3, 5, 8, 13, 16, 17, 31, 32, 33, 48, 63, 64]


def compiled_variants(kernel):
    caches = getattr(kernel, "device_caches", None)
    if caches is None:
        return None
    return sum(len(v[0]) if isinstance(v, tuple) else len(v) for v in caches.values())


def inputs(bs, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = bs + 5
    table = torch.randint(
        0, 1 << 20, (rows, WIDTH), generator=g, dtype=torch.int32
    ).cuda()
    req = torch.randperm(rows, generator=g)[:bs].cuda().long()
    seq = torch.randint(1, WIDTH - 2 * NEXT_N, (bs,), generator=g).cuda().long()
    ext = torch.randint(1, MAX_EXTEND + 1, (bs,), generator=g).cuda().int()
    return table, req, seq, ext


def fill(*shape):
    return torch.full(shape, -7, device="cuda", dtype=torch.int32)


def run_decode(table, req, seq):
    bs = seq.numel()
    out = dict(
        cache_seqlens=fill(bs),
        cu_seqlens_k=fill(bs + 1),
        page_table_1=fill(bs, WIDTH),
        dsa_cache_seqlens=fill(bs),
        dsa_cu_seqlens_k=fill(bs + 1),
        real_page_table=fill(bs, WIDTH // PAGE),
    )
    decode.fused_dsa_decode_metadata_kpool(
        seq_lens=seq,
        req_pool_indices=req,
        req_to_token=table,
        bs=bs,
        max_len=WIDTH,
        dsa_index_topk=TOPK,
        physical_page_size=PAGE,
        index_kpool=KPOOL,
        **out,
    )
    return out


def run_verify(table, req, seq):
    bs, n = seq.numel(), seq.numel() * NEXT_N
    out = dict(
        cache_seqlens=fill(bs),
        cu_seqlens_k=fill(bs + 1),
        page_table_1=fill(n, WIDTH),
        seqlens_expanded=fill(n),
        dsa_cache_seqlens=fill(n),
        dsa_cu_seqlens_k=fill(n + 1),
        real_page_table=fill(n, WIDTH // PAGE),
        paged_mqa_ctx_lens_2d=fill(bs, NEXT_N),
    )
    verify.fused_dsa_target_verify_metadata_kpool(
        seq_lens=seq,
        req_pool_indices=req,
        req_to_token=table,
        bs=bs,
        max_seqlen_k=WIDTH,
        dsa_index_topk=TOPK,
        physical_page_size=PAGE,
        next_n=NEXT_N,
        index_kpool=KPOOL,
        **out,
    )
    return out


def run_draft_extend(table, req, seq, ext):
    bs, total, cap = seq.numel(), int(ext.sum()), seq.numel() * MAX_EXTEND
    out = dict(
        cache_seqlens=fill(bs),
        cu_seqlens_k=fill(bs + 1),
        page_table_1=fill(cap, WIDTH),
        seqlens_expanded=fill(cap),
        dsa_cache_seqlens=fill(cap),
        dsa_cu_seqlens_k=fill(cap + 1),
        real_page_table=fill(cap, WIDTH // PAGE),
    )
    draft_extend.fused_dsa_draft_extend_metadata_kpool(
        seq_lens=seq,
        extend_seq_lens=ext,
        req_pool_indices=req,
        req_to_token=table,
        bs=bs,
        total_len=total,
        max_seqlen_k=WIDTH,
        dsa_index_topk=TOPK,
        physical_page_size=PAGE,
        max_extend_len=MAX_EXTEND,
        max_total_len=cap,
        index_kpool=KPOOL,
        **out,
    )
    return out, total


class TestKPoolMetadataBoundedVariants(CustomTestCase):
    def test_verify_matches_reference_for_every_batch_size(self):
        for bs in BATCH_SIZES:
            table, req, seq, _ = inputs(bs, 100 + bs)
            out = run_verify(table, req, seq)
            expanded = (
                seq[:, None] + torch.arange(1, NEXT_N + 1, device="cuda")
            ).flatten()
            expected = torch.minimum(expanded, TOPK + expanded % KPOOL).int()
            torch.testing.assert_close(out["seqlens_expanded"], expanded.int())
            torch.testing.assert_close(out["dsa_cache_seqlens"], expected)
            torch.testing.assert_close(
                out["dsa_cu_seqlens_k"][1:], expected.cumsum(0).int()
            )
            torch.testing.assert_close(out["cache_seqlens"], (seq + NEXT_N).int())
            pages = table[req].repeat_interleave(NEXT_N, dim=0)
            row_lens = (seq + NEXT_N).repeat_interleave(NEXT_N)
            live = torch.arange(WIDTH, device="cuda")[None, :] < row_lens[:, None]
            torch.testing.assert_close(out["page_table_1"][live], pages[live])
            real_live = (
                torch.arange(0, WIDTH, PAGE, device="cuda")[None, :] < row_lens[:, None]
            )
            torch.testing.assert_close(
                out["real_page_table"][real_live], (pages[:, ::PAGE] // PAGE)[real_live]
            )

    def test_decode_batch_equals_single_requests(self):
        for bs in BATCH_SIZES:
            table, req, seq, _ = inputs(bs, 200 + bs)
            batch = run_decode(table, req, seq)
            for i in range(bs):
                one = run_decode(table, req[i : i + 1], seq[i : i + 1])
                for key in ("cache_seqlens", "dsa_cache_seqlens"):
                    torch.testing.assert_close(batch[key][i : i + 1], one[key])
                live = int(batch["cache_seqlens"][i])
                torch.testing.assert_close(
                    batch["page_table_1"][i, :live], one["page_table_1"][0, :live]
                )
                real = (live + PAGE - 1) // PAGE
                torch.testing.assert_close(
                    batch["real_page_table"][i, :real], one["real_page_table"][0, :real]
                )
            torch.testing.assert_close(
                batch["cu_seqlens_k"][1:], batch["cache_seqlens"].cumsum(0).int()
            )
            torch.testing.assert_close(
                batch["dsa_cu_seqlens_k"][1:],
                batch["dsa_cache_seqlens"].cumsum(0).int(),
            )

    def test_draft_extend_batch_equals_single_requests(self):
        for bs in BATCH_SIZES:
            table, req, seq, ext = inputs(bs, 300 + bs)
            batch, total = run_draft_extend(table, req, seq, ext)
            offset = 0
            for i in range(bs):
                one, n = run_draft_extend(
                    table, req[i : i + 1], seq[i : i + 1], ext[i : i + 1]
                )
                torch.testing.assert_close(
                    batch["cache_seqlens"][i : i + 1], one["cache_seqlens"]
                )
                for key in ("seqlens_expanded", "dsa_cache_seqlens"):
                    torch.testing.assert_close(
                        batch[key][offset : offset + n], one[key][:n]
                    )
                offset += n
            self.assertEqual(offset, total)
            torch.testing.assert_close(
                batch["dsa_cu_seqlens_k"][1 : total + 1],
                batch["dsa_cache_seqlens"][:total].cumsum(0).int(),
            )

    def test_batch_size_sweep_compiles_no_new_variants(self):
        kernels = (
            decode._fused_dsa_decode_metadata_kpool_kernel,
            verify._fused_dsa_target_verify_metadata_kpool_kernel,
            draft_extend._fused_dsa_draft_extend_metadata_kpool_kernel,
        )

        def run_all(bs):
            table, req, seq, ext = inputs(bs, 400 + bs)
            run_decode(table, req, seq)
            run_verify(table, req, seq)
            run_draft_extend(table, req, seq, ext)

        run_all(BATCH_SIZES[0])
        before = [compiled_variants(k) for k in kernels]
        if None in before:
            self.skipTest("Triton JITFunction exposes no device_caches")
        for bs in BATCH_SIZES[1:]:
            run_all(bs)
        after = [compiled_variants(k) for k in kernels]
        # Before this change every new batch size compiled a new variant (bs was constexpr).
        self.assertEqual(
            after,
            before,
            f"variants grew from {before} to {after} over batch sizes {BATCH_SIZES}",
        )


if __name__ == "__main__":
    unittest.main()
