"""AITER reads real BF16 full/window arenas across compaction and graph replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

_LAYERS = (0, 2)
_HEADS, _KV_HEADS, _DIM = 16, 4, 256
_DTYPE = torch.bfloat16


@unittest.skipUnless(
    envs.SGLANG_ENABLE_AITER_UNIFIED_MEMORY.get(),
    "requires SGLANG_ENABLE_AITER_UNIFIED_MEMORY=1 and the documented AITER fixes",
)
@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "requires ROCm")
class TestAiterUnifiedMemory(CustomTestCase):
    page_size = 1
    window = None
    query_heads = _HEADS

    def setUp(self):
        super().setUp()
        from sglang.srt.configs.model_config import AttentionArch
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
        from sglang.srt.runtime_context import get_parallel, publish, reset_context
        from sglang.srt.server_args import ServerArgs
        from sglang.test.unified_allocator_fixtures import (
            UnifiedGeometry,
            build_unified_bundle,
        )

        reset_context()
        self.addCleanup(reset_context)
        args = ServerArgs(model_path="dummy", device="cuda")
        args.enable_unified_memory = True
        args.attention_backend = "aiter"
        args.page_size = self.page_size
        args._mamba_cache_chunk_size = 64
        publish(args, role="test")
        self.enterContext(get_parallel().override(attn_dcp_size=1, dcp_enabled=False))
        torch.manual_seed(123)
        context_len = 256 if self.window else 128
        self.bundle = build_unified_bundle(
            UnifiedGeometry(
                chain="swa" if self.window else "mamba",
                full_layer_ids=(_LAYERS[0],) if self.window else _LAYERS,
                swa_layer_ids=(_LAYERS[1],) if self.window else (),
                state_layer_ids=() if self.window else (1,),
                head_num=_KV_HEADS,
                head_dim=_DIM,
                swa_head_num=_KV_HEADS,
                swa_head_dim=_DIM,
                kv_cache_dtype=_DTYPE,
                page_size=self.page_size,
                window=self.window or 32,
                # Per-layer logical bytes fit int32; interleaved arena spans do not.
                full_tokens=2048 if self.window else 300_000,
                swa_tokens=2048,
                state_slots=64,
                context_len=context_len,
                max_num_reqs=6,
                device="cuda",
            )
        )
        self.alloc = self.bundle.token_to_kv_pool_allocator
        self.pool = self.bundle.token_to_kv_pool
        key = self.pool.layer_router.read(_LAYERS[0]).key()
        span = 1 + sum((n - 1) * stride for n, stride in zip(key.shape, key.stride()))
        if not self.window:
            self.assertGreater(span * key.element_size(), 2**31)
            self.assertLess(key.numel() * key.element_size(), 2**31)
        self.requests = self.bundle.req_to_token_pool.req_to_token
        self.ids = RunnerIds(
            req_to_token=self.requests,
            token_to_kv_pool_allocator=self.alloc,
            token_to_kv_pool=self.pool,
            page_size=self.page_size,
            device="cuda",
        )
        config = SimpleNamespace(
            dtype=_DTYPE,
            is_multimodal=True,  # Qwen3.5's text-serving wrapper has this stamp.
            num_attention_heads=self.query_heads,
            head_dim=_DIM,
            get_num_kv_heads=lambda tp: _KV_HEADS // tp,
            attention_arch=AttentionArch.MHA,
            context_len=context_len,
            hf_config=SimpleNamespace(model_type="qwen3_5"),
        )
        self.runner = SimpleNamespace(
            model_config=config,
            server_args=args,
            dtype=_DTYPE,
            kv_cache_dtype=_DTYPE,
            device="cuda",
            sliding_window_size=self.window,
            is_draft_worker=False,
            runner_ids=self.ids,
            token_to_kv_pool=self.pool,
            token_to_kv_pool_allocator=self.alloc,
            req_to_token_pool=self.bundle.req_to_token_pool,
        )
        # The arena selects unified attention without the legacy dispatch override.
        self.enterContext(envs.SGLANG_USE_AITER_UNIFIED_ATTN.override(False))
        self.backend = AiterAttnBackend(self.runner)
        self.assertTrue(self.backend.use_triton_unified_attention)
        self.layers = [
            RadixAttention(
                self.query_heads,
                _DIM,
                _DIM**-0.5,
                _KV_HEADS,
                layer_id,
                sliding_window_size=self.window if i == 1 and self.window else -1,
            )
            for i, layer_id in enumerate(_LAYERS)
        ]
        lengths = (37, 31, 131, 85) if self.window else (37, 31, 67, 53)
        self.groups = [
            self.alloc.alloc(-(-n // self.page_size) * self.page_size) for n in lengths
        ]
        self.assertTrue(all(g is not None for g in self.groups))
        self.live = [self.groups[i][: lengths[i]] for i in (0, 2, 3)]
        self.bands = [self.alloc.full_attn_allocator]
        if self.window:
            self.bands.append(self.alloc.swa_attn_allocator)
            self.assertFalse(
                torch.equal(
                    self._physical(self.live[0], 0), self._physical(self.live[0], 1)
                )
            )
        else:
            self._assert_high_physical_offsets()
        self.rows = [5, 2, 4]  # Request-pool indices must not become batch lanes.
        for row, slots in zip(self.rows, self.live):
            self.requests[row, : slots.numel()] = slots
        # Logical tensors are retained independently of all physical read tables.
        self.logical = [
            [
                tuple(
                    torch.randn(
                        len(slots), _KV_HEADS, _DIM, device="cuda", dtype=_DTYPE
                    )
                    for _ in range(2)
                )
                for slots in self.live
            ]
            for _ in _LAYERS
        ]

    def _physical(self, slots, layer_index):
        band = self.bands[1 if self.window and layer_index == 1 else 0]
        pages = band.ids.virtual_to_physical[slots // self.page_size]
        return pages * self.page_size + slots % self.page_size

    def _tables(self):
        tables = [self.backend.forward_metadata.kv_indices]
        if self.window:
            tables.append(self.backend.forward_metadata.swa_page_table)
        return tables

    def _assert_high_physical_offsets(self):
        # FULL grows down: low virtual ids already map to the arena's high end.
        physical = self.alloc.full_v2p_page_table[torch.cat(self.live)]
        first = int(physical.min().item())
        for layer_id in _LAYERS:
            for cache in self.pool.layer_router.read(layer_id).kv():
                offset = first * cache.stride(0) * cache.element_size()
                self.assertGreater(offset, 2**31)

    def _batch(self, lengths, query_lengths=None):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.model_executor.forward_mode import ForwardMode

        decode = query_lengths is None
        query_lengths = [1] * len(lengths) if decode else query_lengths
        prefixes = [n - q for n, q in zip(lengths, query_lengths)]
        locs = torch.cat(
            [slots[p:n] for slots, p, n in zip(self.live, prefixes, lengths)]
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
            batch_size=len(lengths),
            input_ids=torch.zeros(sum(query_lengths), dtype=torch.int64, device="cuda"),
            req_pool_indices=torch.tensor(self.rows[: len(lengths)], device="cuda"),
            seq_lens=torch.tensor(lengths, dtype=torch.int64, device="cuda"),
            out_cache_loc=locs,
            seq_lens_sum=sum(lengths),
            seq_lens_cpu=torch.tensor(lengths, dtype=torch.int64),
            extend_seq_lens=torch.tensor(query_lengths, device="cuda"),
            extend_seq_lens_cpu=query_lengths,
            extend_prefix_lens=torch.tensor(prefixes, device="cuda"),
            extend_prefix_lens_cpu=prefixes,
        )
        self.ids.bind_own_plan(fb)
        return fb

    def _reference(self, q, layer_index, lengths, query_lengths):
        q = q.view(-1, self.query_heads, _DIM)
        outputs, offset = [], 0
        for (k, v), n, qn in zip(self.logical[layer_index], lengths, query_lengths):
            keys = k[:n].float().repeat_interleave(self.query_heads // _KV_HEADS, dim=1)
            values = (
                v[:n].float().repeat_interleave(self.query_heads // _KV_HEADS, dim=1)
            )
            queries = q[offset : offset + qn].float()
            scores = torch.einsum("qhd,khd->hqk", queries, keys) * (_DIM**-0.5)
            allowed = torch.arange(n, device="cuda")[None, :] <= (
                n - qn + torch.arange(qn, device="cuda")[:, None]
            )
            if self.window and layer_index == 1:
                allowed &= torch.arange(n, device="cuda")[None, :] > (
                    n - qn + torch.arange(qn, device="cuda")[:, None] - self.window
                )
            scores.masked_fill_(~allowed, float("-inf"))
            outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), values))
            offset += qn
        return torch.cat(outputs).to(_DTYPE).flatten(1)

    def _check(self, actual, q, layer_index, lengths, query_lengths):
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(
            actual,
            self._reference(q, layer_index, lengths, query_lengths),
            atol=0.015,
            rtol=0.01,
        )

    def _eager(self, lengths, query_lengths=None, *, write=False):
        fb = self._batch(lengths, query_lengths)
        self.backend.init_forward_metadata(fb)
        qn = [1] * len(lengths) if query_lengths is None else query_lengths
        q = torch.randn(sum(qn), self.query_heads * _DIM, dtype=_DTYPE, device="cuda")
        for i, layer in enumerate(self.layers):
            k = v = None
            if write:
                k, v = [
                    torch.cat([pair[j] for pair in self.logical[i]]) for j in (0, 1)
                ]
            fn = (
                self.backend.forward_decode
                if query_lengths is None
                else self.backend.forward_extend
            )
            out = fn(q, k, v, layer, fb, save_kv_cache=write)
            self._check(out, q, i, lengths, qn)

    def test_relocation_prefill_and_captured_decode(self):
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc

        lengths = [len(slots) for slots in self.live]
        self._eager(lengths, lengths, write=True)  # Cold prefill, real physical writer.
        self._eager(lengths, [6, 7, 5])  # Cached/chunked prefill.
        self._eager(lengths)
        # Dummy writes must not contaminate the reserved finite row.
        bad = torch.full((1, _KV_HEADS, _DIM), torch.nan, dtype=_DTYPE, device="cuda")
        zero = torch.zeros(1, dtype=torch.int64, device="cuda")
        sink = KVWriteLoc(
            zero,
            physical=True,
            window_locs={self.alloc.roles.window: zero} if self.window else None,
        )
        for layer in self.layers:
            self.pool.set_kv_buffer(layer, sink, bad, bad)

        self.backend.init_cuda_graph_state(3, 3)
        fb = self._batch(lengths)
        write_slots = fb.out_cache_loc.clone()
        fb.iteration_ids.bind_replay(fb, self.ids, slots=write_slots)
        decode_kv = [
            [torch.stack([pair[j][-1] for pair in pairs]) for j in (0, 1)]
            for pairs in self.logical
        ]
        self.backend.init_forward_metadata_out_graph(fb)
        tables = self._tables()
        pointers = [table.data_ptr() for table in tables]
        # Repeated preparation must preserve a cached alias's live prefix.
        expected_tables = [table.clone() for table in tables]
        self.backend.init_forward_metadata_out_graph(fb)
        for table, expected in zip(tables, expected_tables):
            torch.testing.assert_close(table, expected, rtol=0, atol=0)
        q = torch.randn(3, self.query_heads * _DIM, dtype=_DTYPE, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                for layer, (k, v) in zip(self.layers, decode_kv):
                    self.backend.forward_decode(q, k, v, layer, fb, True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = [
                self.backend.forward_decode(q, k, v, layer, fb, True)
                for layer, (k, v) in zip(self.layers, decode_kv)
            ]
        graph.replay()
        for i, output in enumerate(outputs):
            self._check(output, q, i, lengths, [1, 1, 1])

        physical_before = [
            self._physical(torch.cat(self.live), i).clone()
            for i in range(len(self.bands))
        ]
        torch.cuda.synchronize()  # Retire attention before the allocator moves pages.
        self.alloc.free(self.groups[1])
        for band in self.bands:
            self.assertGreater(band.flush_for_allocation(), 0)
        physical_after = [
            self._physical(torch.cat(self.live), i) for i in range(len(self.bands))
        ]
        for before, after in zip(physical_before, physical_after):
            self.assertFalse(torch.equal(before, after))
        if not self.window:
            self._assert_high_physical_offsets()
        reused = self.alloc.alloc(len(self.groups[1]))
        self.assertIsNotNone(reused)
        for i, (before, after) in enumerate(zip(physical_before, physical_after)):
            vacated = before[~torch.isin(before, after)]
            self.assertTrue(torch.isin(self._physical(reused, i), vacated).any().item())
        retired_pages = 0
        if self.window:
            # Retire only whole pages older than the first of the five cached
            # queries below. FULL keeps these tokens while WINDOW drops them.
            retired_pages = (lengths[1] - 5 - self.window + 1) // self.page_size
            retired = self.live[1][: retired_pages * self.page_size]
            self.alloc.free_swa(retired)
            pages = retired[:: self.page_size] // self.page_size
            self.assertTrue(
                (self.bands[0].ids.virtual_to_physical[pages] > 0).all().item()
            )
            self.assertTrue(
                (self.bands[1].ids.virtual_to_physical[pages] == -1).all().item()
            )
        self._eager(lengths, [3, 5, 4])
        self._eager(lengths)

        # Changed values prove replay writes through the relocated physical IDs.
        for pairs, (_, v) in zip(self.logical, decode_kv):
            v.add_(1)
            for (_, logical_v), value in zip(pairs, v):
                logical_v[-1].copy_(value)
        # Replay uses stable query/length/table/write pointers with fresh IDs.
        fresh = self._batch(lengths)
        write_slots.copy_(fresh.out_cache_loc)
        # Also exercise a table first populated by another reader this iteration.
        self.ids.read_table(fresh.iteration_ids)
        if self.window:
            from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind

            self.ids.read_table(fresh.iteration_ids, kind=IdSpaceKind.SLIDING_WINDOW)
        fresh.iteration_ids.bind_replay(fb, self.ids, slots=write_slots)
        self.backend.init_forward_metadata_out_graph(fb)
        self.assertEqual([table.data_ptr() for table in self._tables()], pointers)
        if retired_pages:
            self.assertTrue((tables[1][1, :retired_pages] == 0).all().item())
            self.assertTrue((tables[0][1, :retired_pages] > 0).all().item())
        graph.replay()
        for i, output in enumerate(outputs):
            self._check(output, q, i, lengths, [1, 1, 1])

        for i, layer in enumerate(self.layers):
            physical = self._physical(
                torch.stack([slots[-1] for slots in self.live]), i
            )
            torch.testing.assert_close(
                self.pool.layer_router.read(layer.layer_id).value()[physical],
                decode_kv[i][1],
                rtol=0,
                atol=0,
            )

        # One real lane remains. Poison only its ignored suffix and unrelated reused
        # pages; the sink stays finite and padded graph lanes must read it.
        for i, layer in enumerate(self.layers):
            k, v = self.pool.layer_router.read(layer.layer_id).kv()
            # Include masked tokens inside the final owned page, not only
            # ignored page-table columns. A page1 pass cannot cover this case.
            ignored = torch.cat((self.groups[0][5:], reused))
            physical = self._physical(ignored, i)
            k[physical] = torch.nan
            v[physical] = torch.nan
        fb.seq_lens.copy_(torch.tensor([5, 1, 1], device="cuda"))
        fb.seq_lens_cpu = torch.tensor([5, 1, 1])
        fb.seq_lens_sum = 7
        fb.req_pool_indices[1:] = 0
        fresh = self._batch([5])
        write_slots.zero_()
        write_slots[:1].copy_(fresh.out_cache_loc)
        for pairs, (k, v) in zip(self.logical, decode_kv):
            pairs[0][1][4].add_(1)
            k.fill_(torch.nan)
            v.fill_(torch.nan)
            k[0].copy_(pairs[0][0][4])
            v[0].copy_(pairs[0][1][4])
        fresh.iteration_ids.bind_replay(fb, self.ids, slots=write_slots)
        self.backend.init_forward_metadata_out_graph(fb)
        self.backend.init_forward_metadata_out_graph(fb)
        self.assertEqual([table.data_ptr() for table in self._tables()], pointers)
        for table in tables:
            self.assertTrue((table[0, -(-5 // self.page_size) :] == 0).all().item())
            self.assertTrue((table[1:] == 0).all().item())
        graph.replay()
        for i, output in enumerate(outputs):
            self._check(output[:1], q[:1], i, [5], [1])
            self.assertTrue(torch.isfinite(output).all().item())
            physical = self._physical(self.live[0][4:5], i)
            torch.testing.assert_close(
                self.pool.layer_router.read(self.layers[i].layer_id).value()[physical],
                decode_kv[i][1][:1],
                rtol=0,
                atol=0,
            )
        for layer in self.layers:
            for cache in self.pool.layer_router.read(layer.layer_id).kv():
                self.assertTrue(torch.isfinite(cache[0]).all().item())

    def test_unavailable_sink_safe_writer_is_rejected(self):
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        with (
            patch(
                "sglang.kernels.ops.kvcache.kvcache.can_use_store_cache",
                # The full writer can load while the window writer cannot.
                side_effect=[True, False] if self.window else [False],
            ),
            self.assertRaisesRegex(RuntimeError, "JIT KV writer"),
        ):
            AiterAttnBackend(self.runner)


class TestAiterUnifiedWindowPage1(TestAiterUnifiedMemory):
    window = 32


class TestAiterUnifiedWindowPage16(TestAiterUnifiedWindowPage1):
    page_size = 16


class TestAiterUnifiedWindowPage64(TestAiterUnifiedWindowPage1):
    page_size = 64


class TestAiterUnifiedWindowPage16MHA(TestAiterUnifiedWindowPage16):
    query_heads = _KV_HEADS


if __name__ == "__main__":
    unittest.main()
