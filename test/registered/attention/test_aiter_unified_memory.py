"""AITER reads a real BF16 arena before and after full-KV compaction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

_LAYERS = (0, 2)
_HEADS, _KV_HEADS, _DIM = 16, 4, 256
_DTYPE = torch.bfloat16


@unittest.skipUnless(
    envs.SGLANG_ENABLE_AITER_UNIFIED_MEMORY.get(),
    "requires SGLANG_ENABLE_AITER_UNIFIED_MEMORY=1 and AITER with ROCm/aiter#6661",
)
@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "requires ROCm")
class TestAiterUnifiedMemory(CustomTestCase):
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
        args.page_size = 1
        args._mamba_cache_chunk_size = 64
        publish(args, role="test")
        self.enterContext(get_parallel().override(attn_dcp_size=1, dcp_enabled=False))
        torch.manual_seed(123)
        self.bundle = build_unified_bundle(
            UnifiedGeometry(
                chain="mamba",
                full_layer_ids=_LAYERS,
                state_layer_ids=(1,),
                head_num=_KV_HEADS,
                head_dim=_DIM,
                kv_cache_dtype=_DTYPE,
                page_size=1,
                # Per-layer logical bytes fit int32; interleaved arena spans do not.
                full_tokens=300_000,
                state_slots=64,
                context_len=128,
                max_num_reqs=6,
                device="cuda",
            )
        )
        self.alloc = self.bundle.token_to_kv_pool_allocator
        self.pool = self.bundle.token_to_kv_pool
        key = self.pool.layer_router.read(_LAYERS[0]).key()
        span = 1 + sum((n - 1) * stride for n, stride in zip(key.shape, key.stride()))
        self.assertGreater(span * key.element_size(), 2**31)
        self.assertLess(key.numel() * key.element_size(), 2**31)
        self.requests = self.bundle.req_to_token_pool.req_to_token
        self.ids = RunnerIds(
            req_to_token=self.requests,
            token_to_kv_pool_allocator=self.alloc,
            token_to_kv_pool=self.pool,
            page_size=1,
            device="cuda",
        )
        config = SimpleNamespace(
            dtype=_DTYPE,
            is_multimodal=True,  # Qwen3.5's text-serving wrapper has this stamp.
            num_attention_heads=_HEADS,
            head_dim=_DIM,
            get_num_kv_heads=lambda tp: _KV_HEADS // tp,
            attention_arch=AttentionArch.MHA,
            context_len=128,
            hf_config=SimpleNamespace(model_type="qwen3_5"),
        )
        self.runner = SimpleNamespace(
            model_config=config,
            server_args=args,
            dtype=_DTYPE,
            kv_cache_dtype=_DTYPE,
            device="cuda",
            sliding_window_size=None,
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
            RadixAttention(_HEADS, _DIM, _DIM**-0.5, _KV_HEADS, layer_id)
            for layer_id in _LAYERS
        ]
        self.groups = [self.alloc.alloc(n) for n in (37, 31, 67, 53)]
        self.assertTrue(all(g is not None for g in self.groups))
        self.live = [self.groups[i] for i in (0, 2, 3)]
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
        q = q.view(-1, _HEADS, _DIM)
        outputs, offset = [], 0
        for (k, v), n, qn in zip(self.logical[layer_index], lengths, query_lengths):
            keys = k[:n].float().repeat_interleave(_HEADS // _KV_HEADS, dim=1)
            values = v[:n].float().repeat_interleave(_HEADS // _KV_HEADS, dim=1)
            queries = q[offset : offset + qn].float()
            scores = torch.einsum("qhd,khd->hqk", queries, keys) * (_DIM**-0.5)
            allowed = torch.arange(n, device="cuda")[None, :] <= (
                n - qn + torch.arange(qn, device="cuda")[:, None]
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
        q = torch.randn(sum(qn), _HEADS * _DIM, dtype=_DTYPE, device="cuda")
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
        for layer in self.layers:
            self.pool.set_kv_buffer(layer, KVWriteLoc(zero, physical=True), bad, bad)

        self.backend.init_cuda_graph_state(3, 3)
        fb = self._batch(lengths)
        self.backend.init_forward_metadata_out_graph(fb)
        table = self.backend.forward_metadata.kv_indices
        pointer = table.data_ptr()
        # Repeated preparation must preserve a cached alias's live prefix.
        expected_table = table.clone()
        self.backend.init_forward_metadata_out_graph(fb)
        torch.testing.assert_close(table, expected_table, rtol=0, atol=0)
        q = torch.randn(3, _HEADS * _DIM, dtype=_DTYPE, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.backend.forward_decode(q, None, None, self.layers[0], fb, False)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = self.backend.forward_decode(
                q, None, None, self.layers[0], fb, False
            )
        graph.replay()
        self._check(output, q, 0, lengths, [1, 1, 1])

        physical_before = self.alloc.full_v2p_page_table[torch.cat(self.live)].clone()
        torch.cuda.synchronize()  # Retire attention before the allocator moves pages.
        self.alloc.free(self.groups[1])
        moves = self.alloc.full_attn_allocator.flush_for_allocation()
        self.assertGreater(moves, 0)
        physical_after = self.alloc.full_v2p_page_table[torch.cat(self.live)]
        self.assertFalse(torch.equal(physical_before, physical_after))
        self._assert_high_physical_offsets()
        reused = self.alloc.alloc(len(self.groups[1]))
        self.assertIsNotNone(reused)
        vacated = physical_before[~torch.isin(physical_before, physical_after)]
        reused_physical = self.alloc.full_v2p_page_table[reused]
        self.assertTrue(torch.isin(reused_physical, vacated).any().item())
        self._eager(lengths, [3, 5, 4])
        self._eager(lengths)

        # Replay uses the same query/length/table pointers with fresh iteration IDs.
        fresh = self._batch(lengths)
        # Also exercise a table first populated by another reader this iteration.
        self.ids.read_table(fresh.iteration_ids)
        fresh.iteration_ids.bind_replay(fb, self.ids, slots=fb.out_cache_loc)
        self.backend.init_forward_metadata_out_graph(fb)
        self.assertEqual(self.backend.forward_metadata.kv_indices.data_ptr(), pointer)
        graph.replay()
        self._check(output, q, 0, lengths, [1, 1, 1])

        # One real lane remains. Poison only its ignored suffix and unrelated reused
        # pages; the sink stays finite and padded graph lanes must read it.
        for layer in self.layers:
            k, v = self.pool.layer_router.read(layer.layer_id).kv()
            ignored = torch.cat((self.live[0][5:], reused))
            physical = self.alloc.full_v2p_page_table[ignored]
            k[physical] = torch.nan
            v[physical] = torch.nan
        fb.seq_lens.copy_(torch.tensor([5, 1, 1], device="cuda"))
        fb.seq_lens_cpu = torch.tensor([5, 1, 1])
        fb.seq_lens_sum = 7
        fb.req_pool_indices[1:] = 0
        fresh = self._batch([5])
        fresh.iteration_ids.bind_replay(fb, self.ids, slots=fb.out_cache_loc)
        self.backend.init_forward_metadata_out_graph(fb)
        self.backend.init_forward_metadata_out_graph(fb)
        self.assertEqual(self.backend.forward_metadata.kv_indices.data_ptr(), pointer)
        self.assertTrue((table[0, 5:] == 0).all().item())
        self.assertTrue((table[1:] == 0).all().item())
        graph.replay()
        self._check(output[:1], q[:1], 0, [5], [1])
        self.assertTrue(torch.isfinite(output).all().item())
        for layer in self.layers:
            for cache in self.pool.layer_router.read(layer.layer_id).kv():
                self.assertTrue(torch.isfinite(cache[0]).all().item())

    def test_unavailable_sink_safe_writer_is_rejected(self):
        from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

        with (
            patch(
                "sglang.kernels.ops.kvcache.kvcache.can_use_store_cache",
                return_value=False,
            ),
            self.assertRaisesRegex(RuntimeError, "JIT KV writer"),
        ):
            AiterAttnBackend(self.runner)


if __name__ == "__main__":
    unittest.main()
