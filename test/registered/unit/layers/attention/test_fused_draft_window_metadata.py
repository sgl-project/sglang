"""Window drafts keep their attention mask when placement falls back to FULL.

The old multi-step path only built window indices for SWA pools, so accepted
FULL placements reached decode with no window metadata. Exercise placement,
binding and the real index reader through eager and captured metadata gathers.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import triton_backend as tb
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.layout.fused_draft import (
    LAYER_WINDOW,
    DraftKVGeometry,
    DraftKVProfile,
    DraftLayerSet,
    PlacementContext,
    place_fused_draft,
)
from sglang.srt.mem_cache.unified_draft_pool import bind_fused_draft
from sglang.srt.mem_cache.unified_memory_pool import (
    MHASubPoolSpec,
    UnifiedKVPool,
    UnifiedSWAKVPool,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


class TestFusedDraftWindowMetadata(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.args = ServerArgs(model_path="dummy", page_size=1)
        publish(self.args, role="tokenizer")

    def _fixture(self, window, target_window):
        decision = place_fused_draft(
            profile=DraftKVProfile(
                num_layers=1,
                num_depths=1,
                kinds={
                    LAYER_WINDOW: DraftLayerSet(
                        layer_ids=(0,),
                        geometry=DraftKVGeometry(head_num=1, head_dim=8, v_head_dim=8),
                        window=window,
                    )
                },
            ),
            num_runners=1,
            ctx=PlacementContext(
                host_names=("full", "swa") if target_window else ("full", "mamba"),
                target_window=target_window,
                asymmetric_rows_ok=False,
                draft_backends=("triton",),
            ),
            store_dtype=torch.bfloat16,
        )
        self.assertIsNotNone(decision.placement, decision.declined)
        placement = decision.placement
        host = "swa" if target_window and window <= target_window else "full"
        self.assertEqual(placement.hosts(), (host,))
        specs = [
            MHASubPoolSpec(
                name=name,
                layer_num=1,
                head_num=1,
                head_dim=8,
                store_dtype=torch.bfloat16,
                grow_direction=direction,
                draft_region=placement.region(name),
            )
            for name, direction in (("full", "up"), ("swa", "down"))
        ]
        buffer = UnifiedKVPool(
            total_bytes=1024 * sum(spec.entry_bytes() for spec in specs),
            sub_pool_specs=specs,
            device="cuda",
            enable_memory_saver=False,
            page_size=1,
            fused_draft=placement,
        )
        target_pool = UnifiedSWAKVPool(
            unified_buffer=buffer,
            full_attention_layer_ids=[0],
            swa_attention_layer_ids=[1],
        )
        allocator = UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=buffer,
            kvcache=target_pool,
            device="cuda",
            full_max_total_num_tokens=1024,
            swa_max_total_num_tokens=1024,
        )
        # Different FULL/SWA addresses make a wrong storage route observable.
        old = allocator.alloc(7)
        allocator.free_swa_segment(old, start_pos=0)
        virtual = allocator.alloc(512)
        req = SimpleNamespace(size=1, req_to_token=virtual.reshape(1, -1))
        layer = RadixAttention(
            num_heads=1,
            head_dim=8,
            scaling=0.5,
            num_kv_heads=1,
            layer_id=0,
            sliding_window_size=window,
        )
        bound = bind_fused_draft(
            placement=placement,
            runner=0,
            unified_buffer=buffer,
            host_allocator=allocator,
            model=SimpleNamespace(modules=lambda: [layer]),
            model_config=SimpleNamespace(),
            req_to_token_pool=req,
            page_size=1,
        )
        reader = KVIndexTranslator(
            req_to_token=req.req_to_token,
            token_to_kv_pool_allocator=allocator,
            token_to_kv_pool=bound.token_to_kv_pool,
            page_size=1,
            device="cuda",
            is_draft_worker=True,
        )
        runner = SimpleNamespace(
            req_to_token_pool=req,
            device="cuda",
            model_config=SimpleNamespace(get_max_num_attention_heads=lambda: 1),
            server_args=self.args,
            sliding_window_size=window,
            kv_index_translator=reader,
            token_to_kv_pool=bound.token_to_kv_pool,
        )
        # The model's per-step attention is outside this metadata test. All
        # placement, allocation, translation and multi-step metadata are real.
        inner = SimpleNamespace(max_context_len=512)
        with patch.object(tb, "TritonAttnBackend", return_value=inner):
            backend = tb.TritonMultiStepDraftBackend(runner, 1, 3)
        rows = torch.zeros(1, dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor([300], dtype=torch.int32, device="cuda")
        plan = reader.plan(
            req_pool_indices=rows,
            seq_lens=seq_lens,
            seq_lens_cpu=None,
            write_virtual=None,
            read_extent=3,
        )
        batch = SimpleNamespace(
            batch_size=1,
            seq_lens_sum=None,
            seq_lens=seq_lens,
            req_pool_indices=rows,
            positions=seq_lens.to(torch.int64),
            kv_loc_plan=plan,
            spec_info=SimpleNamespace(window_kv_indices=None, window_kv_indptr=None),
        )
        full = allocator.translate_kv_loc(virtual)
        swa = allocator.translate_loc_from_full_to_swa(virtual)
        self.assertFalse(torch.equal(full, swa))
        return backend, batch, full if host == "full" else swa

    def test_window_reads_follow_placement_in_eager_and_graph(self):
        for window, target_window in ((256, 128), (64, None), (64, 128)):
            with self.subTest(window=window, target_window=target_window):
                backend, batch, physical = self._fixture(window, target_window)
                full_ids = torch.empty((3, 512), dtype=torch.int64, device="cuda")
                window_ids = torch.zeros((3, window), dtype=torch.int64, device="cuda")
                observed = []

                def collect(step, fb):
                    self.assertIsNotNone(fb.spec_info.window_kv_indices)
                    observed.append(
                        (
                            step,
                            fb.spec_info.window_kv_indptr.clone(),
                            fb.spec_info.window_kv_indices.clone(),
                        )
                    )

                def gather():
                    backend.common_template(batch, full_ids, collect, window_ids)

                def check(length):
                    for step, indptr, indices in observed:
                        end = length + step + 1
                        torch.testing.assert_close(
                            indptr,
                            torch.tensor([0, window], dtype=torch.int32, device="cuda"),
                        )
                        torch.testing.assert_close(
                            indices, physical[end - window : end]
                        )

                gather()
                check(300)
                observed.clear()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    gather()
                torch.cuda.current_stream().wait_stream(stream)
                batch.seq_lens.fill_(301)
                batch.positions.fill_(301)
                graph.replay()
                check(301)


if __name__ == "__main__":
    unittest.main()
