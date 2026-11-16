"""A staged SWA window repair on a unified arena, admitted through the PrefillAdder."""

import atexit
import json
import shutil
import tempfile
import time
import unittest
from array import array
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.tree_core_registry import _TREE_CORE_REGISTRY
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache.fixtures import write_backup
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import storage_tree_slots
from sglang.test.unified_allocator_fixtures import build_test_unified_pools

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

PAGE = 4
WINDOW = 2 * PAGE
SWA = ComponentType.of("WINDOW")


def _python_inspector(params, components):
    from sglang.test.kits.unified_radix_cache.unified_tree_core_inspector import (
        UnifiedTreeCoreInspector,
    )

    return UnifiedTreeCoreInspector(params, components)


def _rust_inspector(params, _components):
    from sglang.test.kits.unified_radix_cache.rust_unified_tree_core_inspector import (
        RustUnifiedTreeCoreInspector,
    )

    return RustUnifiedTreeCoreInspector(params)


@unittest.skipUnless(torch.cuda.is_available(), "buffer-mode transfers need CUDA")
class TestUnifiedBufferWindowRepairAdmission(CustomTestCase):
    """Under memory pressure the shared budget reclaims SWA first at admission,
    before the load is published; it may not strip the window rows a staged
    repair keeps."""

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        storage_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, storage_dir, ignore_errors=True)
        for context in (
            get_parallel().override(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1),
            envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(storage_dir),
        ):
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)

    def _cache(self, tokens: int):
        server_args = ServerArgs(
            model_path="dummy",
            page_size=PAGE,
            enable_unified_memory=True,
            enable_hierarchical_cache=True,
            hicache_write_policy="write_through",
            hicache_mem_layout="page_first",
            hicache_host_memory_mode="buffer_only",
            hicache_storage_backend="file",
            hicache_storage_backend_extra_config=json.dumps({"prefetch_threshold": 1}),
            hicache_storage_prefetch_policy="wait_complete",
        )
        set_global_server_args_for_scheduler(server_args)
        allocator = build_test_unified_pools(
            device="cuda",
            kv_cache_dtype=torch.float16,
            head_num=1,
            head_dim=8,
            v_head_dim=8,
            swa_head_num=1,
            swa_head_dim=8,
            swa_v_head_dim=8,
            page_size=PAGE,
            start_layer=0,
            end_layer=2,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            full_max_total_num_tokens=tokens,
            swa_max_total_num_tokens=tokens,
            enable_memory_saver=False,
            need_sort=False,
            lazy_compaction=True,
        ).token_to_kv_pool_allocator
        params = CacheInitParams(
            req_to_token_pool=ReqToTokenPool(
                size=4, max_context_len=tokens, device="cuda", enable_memory_saver=False
            ),
            token_to_kv_pool_allocator=allocator,
            page_size=PAGE,
            disable=False,
            sliding_window_size=WINDOW,
            tree_slots=storage_tree_slots(allocator.storage, window=WINDOW),
        )
        # The inspector cores carry the backup hooks write_backup drives.
        with mock.patch.dict(
            _TREE_CORE_REGISTRY, {"python": _python_inspector, "rust": _rust_inspector}
        ):
            cache = UnifiedRadixCache(params=params)
        cache.init_hicache(server_args, params)
        atexit.unregister(cache.shutdown)
        self.addCleanup(cache.release_host_resources)
        self.addCleanup(cache.cache_controller._stop_storage_threads)
        cache.write_through_threshold = 1 << 30
        return cache, allocator

    @staticmethod
    def _insert(cache, allocator, tokens, *, swa_evicted=0, prefix=None):
        """Insert tokens, extending the caller's own `prefix` rows; SWA is
        tombstoned on [0, swa_evicted)."""
        owned = 0 if prefix is None else len(prefix)
        new = allocator.alloc(len(tokens) - owned)
        ids = new if prefix is None else torch.cat([prefix, new])
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=ids,
                prev_prefix_len=owned,
                component_evicted_seqlens={SWA: swa_evicted} if swa_evicted else {},
            )
        )
        if swa_evicted > owned:
            allocator.free_swa(ids[owned:swa_evicted])
        return ids

    def _pump(self, cache, done, message: str):
        deadline = time.time() + 20
        while not done():
            self.assertLess(time.time(), deadline, message)
            cache.check_hicache_events()
            time.sleep(0.01)

    @staticmethod
    def _device_match(cache, tokens) -> int:
        key = RadixKey(array("q", tokens))
        return cache.match_prefix(MatchPrefixParams(key=key)).device_prefix_len

    def test_shared_budget_reclaim_keeps_the_staged_window(self):
        # The span ends one page past `head`; its window keeps the head's
        # resident SWA tail and repairs the page behind the head.
        head = list(range(1, 1 + WINDOW))
        span = head + list(range(5000, 5000 + PAGE))
        longer = span + list(range(9000, 9000 + WINDOW))

        producer, producer_allocator = self._cache(tokens=128)
        self._insert(producer, producer_allocator, span)
        leaf = producer.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", span)))
        ).last_device_node
        write_backup(producer, leaf)
        writes = producer.buffer_pipeline
        self._pump(
            producer,
            lambda: not writes.inflight_backup_node_ids and not writes.ongoing_backup,
            "the L3 write did not drain",
        )

        cache, allocator = self._cache(tokens=96)
        head_rows = self._insert(cache, allocator, head, swa_evicted=PAGE)
        request = CacheRequestHandle("window-repair-admission", 0)
        cache.prefetch_from_storage(
            request, cache.root_node_handle(), array("q", span), None, None
        )
        self._pump(
            cache,
            lambda: (
                cache.check_prefetch_progress(request)
                and cache.buffer_pipeline.has_staged(request)
            ),
            "the prefetch did not stage",
        )
        cache.pop_prefetch_loaded_tokens(request)
        # While the request waits, another one inserts a longer path through the
        # span end: the span's FULL becomes resident inside a node longer than a
        # window, with no SWA of its own.
        self._insert(
            cache, allocator, longer, swa_evicted=len(longer), prefix=head_rows
        )
        # FULL-only filler leaves no free room, so admitting the load reclaims.
        filler = 0
        while allocator.full_available_size() > 2 * PAGE:
            start = 20000 + 100 * filler
            self._insert(
                cache, allocator, list(range(start, start + PAGE)), swa_evicted=PAGE
            )
            filler += 1
        _, head_node, _ = cache.tree_core.match_full_device_prefix(
            RadixKey(array("q", head))
        )
        self.assertIsNotNone(cache.tree_core.get_component_device_value(head_node, SWA))

        sampling = SamplingParams(max_new_tokens=PAGE)
        sampling.normalize(tokenizer=None)
        req = Req(request.rid, "", array("q", longer), sampling)
        req.cache_request_handle = request
        req.init_next_round_input(cache)
        self.assertTrue(cache.buffer_pipeline.prepare_staged_prefetch(req))
        self.assertEqual(req.staged_prefetch_plan.full_tokens, 0)
        running = mock.MagicMock()
        running.reqs = []
        adder = PrefillAdder(
            page_size=PAGE,
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
            running_batch=running,
            new_token_ratio=1.0,
            rem_input_tokens=10_000,
            rem_chunk_tokens=None,
            num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )

        verdict = adder.add_one_req(
            req, has_chunked_req=False, truncation_align_size=None
        )

        self.assertIs(verdict, AddReqResult.CONTINUE)
        self.assertIsNotNone(
            cache.tree_core.get_component_device_value(head_node, SWA),
            "the reclaim stripped a window row the staged plan kept",
        )
        cache.ready_to_load_host_cache()
        self._pump(
            cache,
            lambda: not cache.buffer_pipeline.ongoing_buffer_load_back,
            "the window repair did not commit",
        )
        self.assertEqual(self._device_match(cache, span), len(span))
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
