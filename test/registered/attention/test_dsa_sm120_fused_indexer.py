"""The SM120 fused DSA indexer against DeepGEMM logits plus an exact top-k, and
its position-to-KV-slot mapping against the page-table arithmetic."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

HEADS = 32
DIM = 128
PAGE = 64
TOPK = 2048
FP8 = torch.float8_e4m3fn


def _skip_reason():
    if not torch.cuda.is_available():
        return "no GPU"
    if torch.cuda.get_device_capability()[0] != 12:
        return "the fused indexer is SM12x-only"
    try:
        import deep_gemm  # noqa: F401
        from flashinfer.dsa_indexer import dsa_indexer_det_topk  # noqa: F401
    except ImportError as exc:
        return f"missing dependency: {exc}"
    return None


SKIP = _skip_reason()


def _inputs(rows: int, keys: int, gen: torch.Generator):
    q = (torch.randn(rows, HEADS, DIM, generator=gen, device="cuda") * 0.5).to(FP8)
    k = (torch.randn(keys, DIM, generator=gen, device="cuda") * 0.5).to(FP8)
    k_scale = torch.rand(keys, generator=gen, device="cuda") + 0.5
    w = torch.randn(rows, HEADS, generator=gen, device="cuda")
    return q, k, k_scale, w


def _reference_logits(q, k, k_scale, w, ks, ke):
    import deep_gemm

    logits = deep_gemm.fp8_mqa_logits(q, (k, k_scale), w, ks, ke, clean_logits=True)
    return logits.float()


def _assert_exact_topk(test, logits, picked, visible):
    """``picked`` holds the top ``min(TOPK, visible)`` of each row by score; the
    k-th score may tie, so compare the selected scores, not the ids."""
    for r in range(logits.shape[0]):
        n = min(TOPK, int(visible[r]))
        ids = picked[r][picked[r] >= 0].long()
        test.assertEqual(ids.numel(), n, f"row {r}")
        test.assertEqual(ids.unique().numel(), n, f"row {r}: duplicate ids")
        test.assertTrue(bool((ids < visible[r]).all()), f"row {r}: invisible key")
        if n == 0:
            continue
        want = logits[r, : int(visible[r])].topk(n).values
        got = logits[r, ids].sort(descending=True).values
        torch.testing.assert_close(got, want, rtol=0, atol=0, msg=f"row {r}")


@unittest.skipIf(SKIP is not None, SKIP)
class TestSm120FusedIndexer(unittest.TestCase):
    def test_ragged_matches_deepgemm(self):
        from sglang.srt.layers.attention.dsa.sm120_fused_indexer import topk_ragged

        gen = torch.Generator(device="cuda").manual_seed(0)
        # (prefix, extend) per request: fresh, prefix-cached, below and above TOPK.
        reqs = [(0, 300), (5000, 64), (0, 2100), (1500, 900)]
        q_parts, k_parts, s_parts, w_parts, ks, ke = [], [], [], [], [], []
        key_base = 0
        for prefix, extend in reqs:
            q, k, s, w = _inputs(extend, prefix + extend, gen)
            q_parts.append(q)
            k_parts.append(k)
            s_parts.append(s)
            w_parts.append(w)
            for u in range(extend):
                ks.append(key_base)
                ke.append(key_base + prefix + u + 1)
            key_base += prefix + extend
        q, k = torch.cat(q_parts), torch.cat(k_parts)
        k_scale, w = torch.cat(s_parts), torch.cat(w_parts)
        ks = torch.tensor(ks, dtype=torch.int32, device="cuda")
        ke = torch.tensor(ke, dtype=torch.int32, device="cuda")
        lens_q = torch.tensor([e for _, e in reqs], device="cuda")
        lens_k = torch.tensor([p + e for p, e in reqs], device="cuda")
        pad = torch.nn.functional.pad
        out = torch.full((q.shape[0], TOPK), -1, dtype=torch.int32, device="cuda")
        topk_ragged(
            q_fp8=q,
            weights=w,
            k_fp8=k,
            k_scale=k_scale,
            cu_seqlens_q=pad(lens_q.cumsum(0), (1, 0)),
            cu_seqlens_k=pad(lens_k.cumsum(0), (1, 0)),
            max_seqlen_k=int(lens_k.max()),
            topk=TOPK,
            out=out,
        )
        logits = _reference_logits(q, k, k_scale, w, ks, ke)
        # Request-local positions: shift the reference rows to their own keys.
        local = torch.full_like(logits, float("-inf"))
        for r in range(logits.shape[0]):
            n = int(ke[r] - ks[r])
            local[r, :n] = logits[r, ks[r] : ke[r]]
        _assert_exact_topk(self, local, out, (ke - ks).tolist())

    def test_paged_matches_deepgemm(self):
        from sglang.srt.layers.attention.dsa.sm120_fused_indexer import topk_paged

        gen = torch.Generator(device="cuda").manual_seed(1)
        lens = [1, 700, 2048, 2049, 9000]
        rows = len(lens)
        q, _, _, w = _inputs(rows + 2, 1, gen)  # two hidden-state padding rows
        pages_per_row = [(n + PAGE - 1) // PAGE for n in lens]
        num_pages = sum(pages_per_row) + 3
        # [pages, PAGE * 132]: PAGE FP8 keys, then their fp32 scales.
        pool = torch.zeros(
            num_pages, PAGE * (DIM + 4), dtype=torch.uint8, device="cuda"
        )
        keys = (
            torch.randn(num_pages, PAGE, DIM, generator=gen, device="cuda") * 0.5
        ).to(FP8)
        scales = torch.rand(num_pages, PAGE, generator=gen, device="cuda") + 0.5
        pool[:, : PAGE * DIM] = keys.view(torch.uint8).reshape(num_pages, -1)
        pool[:, PAGE * DIM :] = scales.view(torch.uint8).reshape(num_pages, -1)
        perm = torch.randperm(num_pages, generator=gen, device="cuda").int()
        width = max(pages_per_row)
        table = torch.zeros(rows, width, dtype=torch.int32, device="cuda")
        at = 0
        for r, p in enumerate(pages_per_row):
            table[r, :p] = perm[at : at + p]
            at += p
        out = topk_paged(
            q_fp8=q,
            weights=w,
            kv_cache=pool.view(num_pages, PAGE, 1, DIM + 4),
            context_lens=torch.tensor(lens, dtype=torch.int32, device="cuda"),
            block_table=table,
            topk=TOPK,
            num_rows=rows,
        )
        self.assertTrue(bool((out[rows:] == -1).all()))
        for r, n in enumerate(lens):
            pages = table[r, : pages_per_row[r]].long()
            k = keys[pages].reshape(-1, DIM)[:n]
            s = scales[pages].reshape(-1)[:n]
            zero = torch.zeros(1, dtype=torch.int32, device="cuda")
            end = torch.full((1,), n, dtype=torch.int32, device="cuda")
            logits = _reference_logits(q[r : r + 1], k, s, w[r : r + 1], zero, end)
            _assert_exact_topk(self, logits, out[r : r + 1], [n])

    def test_kv_slots_follow_the_page_table_in_place(self):
        """The indexer rewrites its top-k buffer in place, which prefill graph
        split ops read instead of the return value."""
        from sglang.srt.layers.attention.dsa.sm120_fused_indexer import to_kv_slots

        gen = torch.Generator(device="cuda").manual_seed(2)
        table = torch.randint(0, 1000, (3, 40), generator=gen, device="cuda").int()
        pos = torch.randint(-1, 40 * PAGE, (5, 300), generator=gen, device="cuda").int()
        rows = torch.tensor([2, 0, 1, 1, 2], dtype=torch.int32, device="cuda")
        p = pos.long().clamp(min=0)
        want = table[rows.long()].gather(1, p // PAGE).long() * PAGE + p % PAGE
        want = torch.where(pos >= 0, want, -1).int()
        to_kv_slots(pos, table, PAGE, rows, out=pos)
        torch.testing.assert_close(pos, want, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
