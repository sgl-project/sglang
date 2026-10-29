"""The DSV4 trtllm-gen backend's persistent multi-CTA KV counter buffer.

The backend passes one zeroed buffer, sized for its query rows and heads, to
every attention launch, so outputs must equal FlashInfer's per-call counters bit
for bit and leave the counters at zero. Attention reads the lens a tail refresh
writes before griddepcontrol.wait, so a launch right after a refresh must not
use PDL.
"""

import ctypes
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# c32 TP4 DSV4.1-Flash decode: 36 rows of 16 heads over 128 + 512 KV split KV
# across CTAs, so the kernel takes and must release the counters.
ROWS, HEADS, SWA, TAIL = 36, 16, 128, 512
SCALE = 512**-0.5


def _fp8(*shape):
    return torch.randn(*shape, device="cuda").to(torch.float8_e4m3fn)


def _idx(*shape):
    return torch.randint(0, 256 * 64, shape, dtype=torch.int32, device="cuda")


def _tile(*shape, fill):
    # Tables are tile-padded to 64 rows, as TrtllmSparseTablePool hands them out.
    return torch.full((64, *shape), fill, dtype=torch.int32, device="cuda")


def _pdl_into_attention(fn):
    """Capture fn (never replayed); for each edge into an attention (fmha)
    kernel, whether it is programmatic. The DOT dump prints no edge types."""
    from cuda.bindings import driver

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        fn()
    get_edges = ctypes.CDLL("libcuda.so.1").cuGraphGetEdges_v2
    get_edges.argtypes = [ctypes.c_void_p] * 4 + [ctypes.POINTER(ctypes.c_size_t)]
    handle, num = graph.raw_cuda_graph(), ctypes.c_size_t(0)
    assert get_edges(handle, None, None, None, ctypes.byref(num)) == 0
    src, dst = (ctypes.c_void_p * num.value)(), (ctypes.c_void_p * num.value)()
    data = ((ctypes.c_uint8 * 8) * num.value)()  # CUgraphEdgeData: byte 2 = type
    assert get_edges(handle, src, dst, data, ctypes.byref(num)) == 0
    programmatic = int(
        driver.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC
    )

    def name(node):
        err, p = driver.cuGraphKernelNodeGetParams(driver.CUgraphNode(init_value=node))
        if err != driver.CUresult.CUDA_SUCCESS or not int(p.func):
            return ""
        _, raw = driver.cuFuncGetName(driver.CUfunction(init_value=int(p.func)))
        return raw.decode()

    return [
        data[i][2] == programmatic
        for i in range(num.value)
        if name(dst[i]).startswith("fmha")
    ]


def _counter_buffer(heads, capacity_rows=1, req_rows=1, max_draft=None):
    """The backend's own allocation, with fresh buffers and stubbed sizing inputs."""
    from sglang.srt.layers.attention import deepseek_v4_trtllm_backend as trtllm
    from sglang.srt.runtime_context import get_context

    runner = SimpleNamespace(
        model_config=SimpleNamespace(num_attention_heads=heads),
        req_to_token_pool=SimpleNamespace(req_to_token=torch.empty(req_rows, 1)),
    )
    with (
        get_context().resources.override(buffers={}),
        mock.patch.object(
            trtllm, "_trtllm_query_row_capacity", return_value=capacity_rows
        ),
        mock.patch.object(
            trtllm, "max_speculative_num_draft_tokens", return_value=max_draft
        ),
    ):
        return trtllm._get_trtllm_counter_buffer(runner, torch.device("cuda"))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "trtllm-gen DSV4 sparse MLA requires SM10x",
)
class TestDSV4TrtllmCounterBuffer(CustomTestCase):
    def setUp(self):
        from sglang.srt.layers.attention import deepseek_v4_trtllm_backend as trtllm

        torch.manual_seed(1020)
        self.kv = (_fp8(256, 1, 64, 512), _fp8(256, 1, 64, 512))
        self.sink = torch.randn(HEADS, device="cuda")
        self.layer = SimpleNamespace(layer_id=0, k_scale_float=None)
        b = object.__new__(trtllm.DeepseekV4TrtllmAttnBackend)
        b.softmax_scale = SCALE
        b._tail_refresh_layers = None  # layer 0 refreshes the tail
        b.trtllm_workspace_buffer = torch.zeros(
            128 << 20, dtype=torch.uint8, device="cuda"
        )
        b.trtllm_counter_buffer = _counter_buffer(HEADS)
        b._trtllm_kv_cache_views = lambda layer_id, compress_ratio: self.kv
        self.backend = b
        # The reference lets FlashInfer zero fresh counters on every call.
        self.ref_workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")

    def _check(self, run, q, table, lens, seq_lens):
        """run() equals FlashInfer with its own per-call counters, on a table and
        lens built without the backend, and leaves the shared counters at zero."""
        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

        out = run()
        ref = trtllm_batch_decode_sparse_mla_dsv4(
            query=q.view(-1, 1, HEADS, 512),
            swa_kv_cache=self.kv[0],
            workspace_buffer=self.ref_workspace,
            sparse_indices=table,
            compressed_kv_cache=self.kv[1],
            sparse_topk_lens=lens,
            seq_lens=seq_lens,
            bmm1_scale=SCALE,
            sinks=self.sink,
            backend="trtllm-gen",
        ).view(-1, HEADS, 512)
        self.assertTrue(torch.equal(out[: ref.shape[0]], ref))
        torch.cuda.synchronize()
        self.assertFalse(self.backend.trtllm_counter_buffer.any().item())
        return out

    def _assert_passes_backend_buffer(self, run):
        # FlashInfer validates a caller's buffer; without one it allocates its own.
        saved = self.backend.trtllm_counter_buffer
        self.backend.trtllm_counter_buffer = torch.zeros(
            16, dtype=torch.uint8, device="cuda"
        )
        with self.assertRaisesRegex(ValueError, "too small"):
            run()
        self.backend.trtllm_counter_buffer = saved

    def test_buffer_starts_zeroed_and_covers_rows_and_heads(self):
        from flashinfer.utils import (
            get_device_sm_count,
            get_trtllm_gen_multi_ctas_kv_counter_bytes,
        )

        # 16384 rows x 128 heads (above the helper's 8192-row, 64-head floors): from the
        # query-row capacity, or 2048 request slots x max draft 8 (adaptive spec).
        need = get_trtllm_gen_multi_ctas_kv_counter_bytes(
            16384, 128, get_device_sm_count(torch.device("cuda"))
        )
        for capacity_rows, req_rows, max_draft in ((16384, 1, None), (1, 2048, 8)):
            with (
                self.subTest(capacity_rows=capacity_rows, req_rows=req_rows),
                torch.cuda.use_mem_pool(torch.cuda.MemPool()),
            ):
                # The private pool's only free block is filled with 0xFF and goes
                # to the allocation; the kernel spins on counters not at zero.
                poison = torch.full((need,), 0xFF, dtype=torch.uint8, device="cuda")
                poison_ptr = poison.data_ptr()
                del poison
                buffer = _counter_buffer(128, capacity_rows, req_rows, max_draft)
                self.assertGreaterEqual(buffer.numel(), need)
                self.assertEqual(buffer.data_ptr(), poison_ptr)
                torch.cuda.synchronize()
                self.assertFalse(buffer.any().item())
                del buffer

    def test_decode(self):
        from sglang.kernels.ops.attention.dsv4.trtllm_metadata import (
            pack_sparse_tail,
        )

        # Dense decode: one query row per FlashInfer request. seq_lens bound only
        # the SWA window, so about half the rows are shorter than it.
        seq_lens = torch.randint(1, 2 * SWA, (ROWS,), dtype=torch.int32, device="cuda")
        table, lens = _tile(SWA + TAIL, fill=-1), _tile(fill=SWA)
        table[:ROWS, :SWA] = _idx(ROWS, SWA)
        tail_lens = torch.randint(
            1, TAIL + 1, (ROWS,), dtype=torch.int32, device="cuda"
        )
        core = SimpleNamespace(
            seq_lens_casual=seq_lens,
            swa_page_indices=table[:ROWS, :SWA],
            trtllm_c4_indices=table[:ROWS],
            trtllm_c4_lens=lens[:ROWS],
            _c4_topk_writes_table=lambda: False,
            sparse_topk_lengths=lambda compress_ratio: tail_lens,
        )
        q, extra = _fp8(ROWS, HEADS, 512), _idx(ROWS, TAIL)

        def run():
            return self.backend._forward_trtllm_decode(
                q=q,
                layer=self.layer,
                compress_ratio=4,
                core_attn_metadata=core,
                attn_sink=self.sink,
                extra_indices=extra,
            )

        ref_table = torch.cat([table[:ROWS, :SWA], extra], dim=1)
        for _ in range(2):  # the second launch reuses the first's counters
            self._check(run, q, ref_table, tail_lens + SWA, seq_lens)
        self._assert_passes_backend_buffer(run)
        # After this layer's refresh attention waits for pack (no PDL edge).
        self.assertEqual(_pdl_into_attention(run), [False])

        # Control: after the same pack, a layer that does not refresh keeps PDL,
        # so the check above can see a programmatic edge.
        def pack_then_run():
            pack_sparse_tail(extra, tail_lens, table[:ROWS], lens[:ROWS])
            run()

        self.backend._tail_refresh_layers = frozenset()
        self.assertEqual(_pdl_into_attention(pack_then_run), [True])

    def test_decode_reads_no_metadata_row_past_the_batch(self):
        """seq_lens_casual and swa_page_indices come without a tile pad, so rows
        past the batch hold whatever the allocator left there (poisoned here, as
        are the pool tables' pad rows); outputs must match exact-size tables."""
        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

        bad = 1 << 30  # out-of-range seq lens / slots: a read would show

        def live_then_poison(live):
            buf = torch.full(
                (64, *live.shape[1:]), bad, dtype=live.dtype, device="cuda"
            )
            buf[:ROWS].copy_(live)
            return buf[:ROWS]

        seq_lens = torch.randint(1, 2 * SWA, (ROWS,), dtype=torch.int32, device="cuda")
        tail_lens = torch.randint(
            1, TAIL + 1, (ROWS,), dtype=torch.int32, device="cuda"
        )
        swa, extra, q = _idx(ROWS, SWA), _idx(ROWS, TAIL), _fp8(ROWS, HEADS, 512)
        window = torch.full((ROWS,), SWA, dtype=torch.int32, device="cuda")
        # DSpark draft rows: window + 64-slot block, bounded by the block's lengths.
        wide = torch.cat([swa, _idx(ROWS, 64)], dim=1)
        block_lens = torch.randint(
            SWA, SWA + 65, (ROWS,), dtype=torch.int32, device="cuda"
        )
        for ratio, swa_table in ((0, swa), (4, swa), (0, wide)):
            dspark = swa_table.shape[1] > SWA
            core = SimpleNamespace(
                seq_lens_casual=live_then_poison(seq_lens),
                swa_page_indices=live_then_poison(swa_table),
                swa_topk_lengths=live_then_poison(block_lens),
                trtllm_swa_lens=live_then_poison(block_lens if dspark else window),
                trtllm_c4_indices=live_then_poison(
                    torch.cat([swa, torch.full_like(extra, -1)], 1)
                ),
                trtllm_c4_lens=live_then_poison(window),
                _c4_topk_writes_table=lambda: False,
                sparse_topk_lengths=lambda compress_ratio: tail_lens,
            )
            out = self.backend._forward_trtllm_decode(
                q=q,
                layer=self.layer,
                compress_ratio=ratio,
                core_attn_metadata=core,
                attn_sink=self.sink,
                extra_indices=extra if ratio else None,
            )
            ref = trtllm_batch_decode_sparse_mla_dsv4(
                query=q.view(ROWS, 1, HEADS, 512),
                swa_kv_cache=self.kv[0],
                workspace_buffer=self.ref_workspace,
                sparse_indices=torch.cat([swa, extra], 1)
                if ratio
                else swa_table.clone(),
                compressed_kv_cache=self.kv[0] if dspark else self.kv[1],
                sparse_topk_lens=tail_lens + SWA
                if ratio
                else (block_lens if dspark else window).clone(),
                seq_lens=(block_lens if dspark else seq_lens).clone(),
                bmm1_scale=SCALE,
                sinks=self.sink,
                backend="trtllm-gen",
            ).view(ROWS, HEADS, 512)
            with self.subTest(ratio=ratio, dspark=dspark):
                self.assertTrue(torch.equal(out, ref))

    def test_decode_low_ratio_tail_written_in_place(self):
        """A V4.1 index source's top-k kernel writes the shared table's tail in place,
        with no pack_sparse_tail: its picks are a top-k of the row and touch no other
        table cell, attention equals FlashInfer on [SWA | picks] with the ratio's
        own lens, and attention still launches without PDL after the write."""
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_sparse
        from sglang.srt.layers.attention.dsv4.indexer import topk_transform_paged_bf16

        seq_lens = torch.randint(1, 2 * SWA, (ROWS,), dtype=torch.int32, device="cuda")
        table, lens = _tile(SWA + TAIL, fill=-1), _tile(fill=SWA)
        table[:ROWS, :SWA] = _idx(ROWS, SWA)
        table[ROWS:] = 7  # pad rows: nothing may write them
        before = table.clone()
        tail = table[:ROWS, SWA:]
        # Scores over 4096 candidates; some rows have fewer than the top-k.
        width = 4096
        cands = torch.randint(1, width + 1, (ROWS,), dtype=torch.int32, device="cuda")
        cands[:2] = torch.tensor([TAIL - 1, 3])
        picks = cands.clamp(max=TAIL)
        lens[:ROWS] = SWA + picks  # init_trtllm_sparse_buffers, once per step
        core = SimpleNamespace(
            seq_lens_casual=seq_lens,
            swa_page_indices=table[:ROWS, :SWA],
            trtllm_c4_indices=table[:ROWS],
            trtllm_c1_lens=lens[:ROWS],
            sparse_topk_lengths=lambda compress_ratio: picks,
        )
        scores = torch.randn(ROWS, width, device="cuda").to(torch.bfloat16)
        # Identity tables: slot == candidate position, a valid compressed KV slot.
        pages = torch.arange(width // 64, dtype=torch.int32, device="cuda")
        blocks = torch.arange(width // 8, dtype=torch.int32, device="cuda")
        writers = {
            # c2 sources and the c1 candidate source (DeepSelect)
            "paged_bf16": lambda: topk_transform_paged_bf16(
                scores, cands, pages.expand(ROWS, -1), tail, 64
            ),
            # c1 candidate consumers
            "sparse_bf16_small": lambda: topk_transform_sparse(
                scores, cands, blocks.expand(ROWS, -1), tail
            ),
        }
        q = _fp8(ROWS, HEADS, 512)

        def run():
            return self.backend._forward_trtllm_decode(
                q=q,
                layer=self.layer,
                compress_ratio=1,
                core_attn_metadata=core,
                attn_sink=self.sink,
                extra_indices=tail,
            )

        for name, write in writers.items():
            with self.subTest(writer=name):
                tail.fill_(-2)
                write()
                for r in range(ROWS):
                    k, row = int(picks[r]), tail[r].sort().values
                    self.assertEqual(row[: TAIL - k].tolist(), [-1] * (TAIL - k))
                    chosen = row[TAIL - k :].long()
                    self.assertEqual(chosen.unique().numel(), k)
                    self.assertLess(int(chosen.max()), int(cands[r]))
                    kth = scores[r, : cands[r]].float().topk(k).values[-1]
                    self.assertTrue(torch.all(scores[r, chosen].float() >= kth))
                self.assertTrue(torch.equal(table[:, :SWA], before[:, :SWA]))
                self.assertTrue(torch.equal(table[ROWS:], before[ROWS:]))
                ref_table = torch.cat([table[:ROWS, :SWA], tail.clone()], dim=1)
                self._check(run, q, ref_table, picks + SWA, seq_lens)

                def write_then_run():
                    write()
                    run()

                # The write is this layer's tail refresh: attention waits for it.
                self.backend._tail_refresh_layers = None
                self.assertEqual(_pdl_into_attention(write_then_run), [False])
                # Control: a layer that reuses the tail keeps PDL.
                self.backend._tail_refresh_layers = frozenset()
                self.assertEqual(_pdl_into_attention(write_then_run), [True])
                self.backend._tail_refresh_layers = None

    def test_prefill(self):
        # A new 3-token request (inside the SWA window) and 9 tokens after an 8K
        # prefix, one dense row per token; q is padded to 16 rows.
        sum_q, pad = 12, 16
        seq_lens = torch.cat([torch.arange(1, 4), torch.arange(8193, 8202)])
        seq_lens = seq_lens.to(torch.int32).cuda()
        table, lens = _tile(SWA + TAIL, fill=-1), _tile(fill=SWA)
        tail_lens = torch.empty(sum_q, dtype=torch.int32, device="cuda")
        core = SimpleNamespace(
            trtllm_prefill_qmeta=None,  # derived from the extend lens below
            seq_lens_casual=torch.cat([seq_lens, seq_lens.new_ones(pad - sum_q)]),
            request_window_layout=None,
            trtllm_prefill_swa_indices=table[:sum_q, :SWA],
            trtllm_prefill_c4_indices=table[:sum_q],
            trtllm_prefill_c4_lens=lens[:sum_q],
            trtllm_tail_needs_pack=False,
            _c4_topk_writes_table=lambda: False,
            sparse_topk_lengths=lambda compress_ratio: tail_lens,
        )
        b = self.backend
        b.forward_metadata = SimpleNamespace(
            core_attn_metadata=core, late_layer_tail=None
        )
        # Pad rows of the output must come back zero, not this stale 7.
        b.trtllm_graph_output_buffer = torch.full(
            (pad, HEADS, 512), 7.0, dtype=torch.bfloat16, device="cuda"
        )
        b.trtllm_eager_output_buffer = b._padded_output_zeroed = None
        q, extra = _fp8(pad, HEADS, 512), _idx(sum_q, TAIL)
        batch = SimpleNamespace(extend_seq_lens_cpu=[3, 9])

        def run():
            return b._forward_trtllm_prefill(
                q=q,
                layer=self.layer,
                compress_ratio=4,
                forward_batch=batch,
                attn_sink=self.sink,
                extra_indices=extra,
            )

        # A refresh layer; then the late-layer tail's one-off pack on a layer that
        # is not an index source.
        for refresh_layers, needs_pack in ((None, False), (frozenset(), True)):
            b._tail_refresh_layers = refresh_layers
            table[:sum_q, :SWA] = _idx(sum_q, SWA)
            extra.copy_(_idx(sum_q, TAIL))
            tail_lens.random_(1, TAIL + 1)
            ref_table = torch.cat([table[:sum_q, :SWA], extra], dim=1)
            core.trtllm_tail_needs_pack = needs_pack
            out = self._check(run, q[:sum_q], ref_table, tail_lens + SWA, seq_lens)
            self.assertTrue(torch.all(out[sum_q:] == 0))
            # Captured again: attention follows pack, without PDL.
            core.trtllm_tail_needs_pack = needs_pack
            self.assertEqual(_pdl_into_attention(run), [False])
        self._assert_passes_backend_buffer(run)


if __name__ == "__main__":
    unittest.main()
