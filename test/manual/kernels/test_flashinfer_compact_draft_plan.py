"""Validate compact host planning through the real SGLang updater and fa2.

Includes shrink/grow rollback, remapped pages, live CUDA graphs and fallback.
"""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import flashinfer
import torch
from sglang.srt.environ import envs
from sglang.srt.layers.attention import flashinfer_backend as fb


def plan(w, indptr, indices, last, heads, kv_heads, dim, host, enabled=True):
    u = fb.FlashInferIndicesUpdaterDecode.__new__(fb.FlashInferIndicesUpdaterDecode)
    u.attn_backend = SimpleNamespace(
        kv_index_translator=SimpleNamespace(reads_are_translated=False)
    )
    u.kv_last_page_len = last
    u.num_qo_heads = heads
    u.num_kv_heads = kv_heads
    u.head_dim = dim
    u.data_type = w._cached_kv_data_type
    u.q_data_type = torch.bfloat16
    w.begin_forward = partial(flashinfer.fast_decode_plan, w)
    spec = SimpleNamespace(kv_indptr=indptr, kv_indices=indices)
    old = fb.global_override_indptr_cpu
    fb.global_override_indptr_cpu = host
    compact = fb._try_compact_draft_decode_plan
    used = []

    def observe(*a, **kw):
        result = compact(*a, **kw)
        used.append(result)
        return result

    try:
        with (
            envs.SGLANG_FLASHINFER_DRAFT_COMPACT_PLAN.override(enabled),
            patch.object(fb, "_try_compact_draft_decode_plan", observe),
        ):
            u.call_begin_forward(
                w, None, 0, indptr, None, spec, None, req_pool_indices=None
            )
    finally:
        fb.global_override_indptr_cpu = old
    assert used == ([True] if enabled else []), used


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCompactDraftPlan(unittest.TestCase):
    def test_plan_outputs_and_live_graph(self):
        torch.set_num_threads(2)
        torch.manual_seed(322)
        for dtype, batch in [
            (torch.bfloat16, 1),
            (torch.float8_e4m3fn, 1),
            (torch.float8_e4m3fn, 3),
        ]:
            capacity = 16384
            heads, kv_heads, dim = 32, 2, 128
            q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
            kv = torch.randn(
                capacity, 2, 1, kv_heads, dim, device="cuda", dtype=torch.bfloat16
            ).to(dtype)
            wrappers = []
            for _ in range(2):
                wrappers.append(
                    flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                        torch.empty(
                            256 * 1024 * 1024, device="cuda", dtype=torch.uint8
                        ),
                        use_cuda_graph=True,
                        use_tensor_cores=True,
                        paged_kv_indptr_buffer=torch.empty(
                            batch + 1, device="cuda", dtype=torch.int32
                        ),
                        paged_kv_indices_buffer=torch.empty(
                            batch * capacity, device="cuda", dtype=torch.int32
                        ),
                        paged_kv_last_page_len_buffer=torch.ones(
                            batch, device="cuda", dtype=torch.int32
                        ),
                        backend="fa2",
                    )
                )
            graphs = []
            outs = []
            for step, first in enumerate(
                [8193, 8186, 8210, 1, 13, 4095, 4097, 16380, 8192]
            ):
                lengths = [max(1, first - i * 17) for i in range(batch)]
                host = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).int()
                inds = torch.cat(
                    [
                        torch.randperm(capacity, device="cuda", dtype=torch.int32)[:n]
                        for n in lengths
                    ]
                )
                for i, w in enumerate(wrappers):
                    if step == 0:
                        w.plan(
                            host,
                            inds,
                            torch.ones(batch, dtype=torch.int32),
                            heads,
                            kv_heads,
                            dim,
                            1,
                            q_data_type=q.dtype,
                            kv_data_type=dtype,
                        )
                        w._cached_kv_data_type = dtype
                    w._paged_kv_indptr_buf.copy_(host)
                    w._paged_kv_indices_buf[: len(inds)].copy_(inds)
                    args = (
                        w,
                        w._paged_kv_indptr_buf,
                        w._paged_kv_indices_buf[: len(inds)],
                        w._paged_kv_last_page_len_buf,
                        heads,
                        kv_heads,
                        dim,
                    )
                    if i == 0:
                        flashinfer.fast_decode_plan(
                            *args,
                            1,
                            q_data_type=q.dtype,
                            kv_data_type=dtype,
                            global_override_indptr_cpu=host,
                        )
                    else:
                        plan(*args, host)
                assert wrappers[0]._plan_info == wrappers[1]._plan_info
                torch.testing.assert_close(
                    wrappers[0].run(q, kv), wrappers[1].run(q, kv), atol=0, rtol=0
                )
                if step == 0:
                    for w in wrappers:
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            o = w.run(q, kv)
                        graphs.append(g)
                        outs.append(o)
                q.normal_()
                for g in graphs:
                    g.replay()
                torch.testing.assert_close(outs[0], outs[1], atol=0, rtol=0)
                torch.testing.assert_close(
                    outs[1], wrappers[0].run(q, kv), atol=0, rtol=0
                )
                # Exercise the updater's flag-off fallback on the same live graph buffers.
                w = wrappers[1]
                args = (
                    w,
                    w._paged_kv_indptr_buf,
                    w._paged_kv_indices_buf[: len(inds)],
                    w._paged_kv_last_page_len_buf,
                    heads,
                    kv_heads,
                    dim,
                )
                plan(*args, host, enabled=False)
                assert w._plan_info == wrappers[0]._plan_info
                for badhost in [
                    None,
                    host.long(),
                    host.to("cuda"),
                    torch.empty((batch + 1) * 2, dtype=torch.int32)[::2],
                ]:
                    assert not fb._try_compact_draft_decode_plan(*args, badhost)
                torch.testing.assert_close(
                    w.run(q, kv), wrappers[0].run(q, kv), atol=0, rtol=0
                )


if __name__ == "__main__":
    unittest.main()
