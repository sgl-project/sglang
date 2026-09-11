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
"""How a draft worker's KVCacheConfigurator binds its KV under the unified pool.

With a fused region the draft binds `UnifiedDraftKVPool` over the target's
allocator (a compact-window DFLASH draft with a private `req_to_token` of its
own), refusing a KV dtype unlike the region's. Without one it takes the private
arm, sized by the full sub-allocator's VIRTUAL id space (`max_slots - 1`), not
`size_full`, the smaller static token budget: sizing by that would put
verify-window writes at high virtual ids out of bounds.

BUG REGRESSION (fused window reach). A multi-layer MTP draft whose window
layers live in the target's swa sub-pool starts its widened draft extend
`front` rows below the committed length, so it reads that many tokens further
back than the target. SWA eviction kept only the target's window and freed
them, and the draft read the sink page instead (NaN under a poisoned pool).
Eviction must keep the draft's reach; every other SWA rule keeps the window.

    python -m pytest test/registered/unit/mem_cache/test_kv_cache_configurator_draft_gate.py -v
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.mem_cache import unified_draft_pool
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.pure_swa_radix_cache import PureSWARadixCache
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_draft_pool import (
    UnifiedDraftKVPool,
    UnifiedDraftSWAKVPool,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    MHASubPoolSpec,
    UnifiedKVPool,
    _store_dtype_for,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, stage="weekly", runner_config="cpu")
maybe_stub_sgl_kernel()

from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleDraftWorker,
    MultiLayerEagleWorkerV2,
)


class _FakeKVCache:
    def __init__(self, max_slots):
        self.buf = torch.full((max_slots,), -1, dtype=torch.int64)
        self.allocator = None

    def attach_allocator(self, allocator):
        self.allocator = allocator


class _FakeUnifiedSWAKVPool:
    def __init__(self, shared_pool):
        self.full_kv_pool = _FakeKVCache(shared_pool.max_slots("full"))
        self.swa_kv_pool = _FakeKVCache(shared_pool.max_slots("swa"))
        self.full_to_swa_index_mapping = None

    def attach_allocators(self, *, full_allocator, swa_allocator):
        self._full_allocator = full_allocator
        self._swa_allocator = swa_allocator


class _CapturedSizes(Exception):
    """Sentinel carrying the sizes handed to the token-pool build."""

    def __init__(self, sizes):
        self.sizes = sizes


_PS = 2


def _bare(cls, **attrs):
    # Skip __init__: only the attributes the code under test reads are set.
    obj = object.__new__(cls)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


def _multi_layer_worker(pools_and_windows, *, front):
    """A multi-layer MTP worker whose draft runners hold these (pool, window)
    pairs and whose draft extend is widened by `front` rows."""
    runners = [
        SimpleNamespace(token_to_kv_pool=pool, sliding_window_size=window)
        for pool, window in pools_and_windows
    ]
    draft = _bare(
        MultiLayerEagleDraftWorker,
        draft_runner_list=runners,
        draft_extend_num_front_tokens=front,
    )
    return _bare(MultiLayerEagleWorkerV2, _draft_worker=draft)


def _swa_cache_params(*, sliding_window_size, draft_swa_window):
    return CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        sliding_window_size=sliding_window_size,
        draft_swa_window=draft_swa_window,
    )


class TestDraftBindingDispatch(CustomTestCase):
    def setUp(self):
        # `KVIndexTranslator.__init__` reads `attn_dcp_size`, a derived
        # parallel width that only exists once a config is published.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _swa_allocator(
        self,
        *,
        with_draft_region: bool,
        n_full=32,
        n_swa=16,
        kv_dtype=torch.bfloat16,
        window_draft=False,
    ):
        store_dtype = _store_dtype_for(kv_dtype)
        region = (
            DenseDraftRegion(
                lane_num=1,
                head_num=1,
                head_dim=16,
                store_dtype=store_dtype,
                kv_dtype=kv_dtype,
            )
            if with_draft_region
            else None
        )
        # A window draft's region rides in the swa sub-pool instead.
        host = "swa" if window_draft else "full"
        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=8,
            store_dtype=store_dtype,
            grow_direction="down",
            draft_region=region if host == "full" else None,
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=1,
            head_num=2,
            head_dim=8,
            store_dtype=store_dtype,
            grow_direction="up",
            draft_region=region if host == "swa" else None,
        )
        total = n_full * full_spec.entry_bytes() + n_swa * swa_spec.entry_bytes()
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device="cpu",
            enable_memory_saver=False,
            page_size=_PS,
            fused_draft=(
                FusedDraftPlacement.from_counts(
                    counts={host: [1]}, regions={host: region}
                )
                if region is not None
                else None
            ),
        )
        return UnifiedSWATokenToKVPoolAllocator(
            unified_buffer=pool,
            kvcache=_FakeUnifiedSWAKVPool(pool),
            device="cpu",
            full_max_total_num_tokens=n_full,
            swa_max_total_num_tokens=n_swa,
            page_size=_PS,
            need_sort=False,
            forward_stream=None,
        )

    def _run(
        self,
        *,
        algorithm,
        alloc,
        max_total_num_tokens,
        kv_dtype=torch.bfloat16,
        req_to_token_pool="shared",
    ):
        if req_to_token_pool == "shared":
            req_to_token_pool = object()
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        cfg.kv_cache_dtype = kv_dtype
        cfg.is_draft_worker = True
        cfg.draft_model_idx = None
        cfg.spec_algorithm = algorithm
        cfg.page_size = _PS
        cfg.is_hybrid_swa = False
        cfg.is_hybrid_swa_mtp_draft = False
        cfg.model = object()
        cfg.model_config = SimpleNamespace(hf_config=None, is_hybrid_swa=False)
        sizes = kcc._PoolSizes(
            max_total_num_tokens=max_total_num_tokens,
            max_running_requests=8,
            full_max_total_num_tokens=max_total_num_tokens,
            swa_max_total_num_tokens=16,
            c4_max_total_num_tokens=0,
            c128_max_total_num_tokens=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            c4_state_dtype=None,
            c128_state_dtype=None,
        )

        def _capture(self, *, sizes, **kw):
            raise _CapturedSizes(sizes)

        built_req_pool = SimpleNamespace(kind="private-compact")

        with (
            patch.object(
                kcc,
                "get_memory",
                return_value=SimpleNamespace(enable_unified_memory=True),
            ),
            patch.object(
                kcc,
                "get_schedule",
                return_value=SimpleNamespace(page_size=_PS),
            ),
            patch.object(kcc, "is_deepseek_dsa", return_value=False),
            patch.object(kcc, "is_deepseek_v4", return_value=False),
            patch.object(
                kcc.KVCacheConfigurator,
                "_validate_prefill_only_disable_kv_cache_pool_family",
                lambda self, *a, **kw: None,
            ),
            patch.object(kcc.KVCacheConfigurator, "_build_token_to_kv_pool", _capture),
            # The real scan walks the draft nn.Module for RadixAttention layers.
            patch.object(unified_draft_pool, "draft_kv_layer_ids", return_value=[0]),
            patch.object(
                unified_draft_pool, "draft_state_layer_classes", return_value=[]
            ),
            patch.object(
                kcc.KVCacheConfigurator,
                "_build_req_to_token_pool",
                lambda self, *, max_num_reqs: built_req_pool,
            ),
        ):
            return cfg._init_pools(
                sizes=sizes,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=alloc,
            )

    def test_non_eagle_draft_takes_the_private_arm_sized_by_the_id_space(self):
        alloc = self._swa_allocator(with_draft_region=False)
        id_space = alloc.full_attn_allocator.max_slots - 1
        # The distinction under test only exists while budget < id space.
        self.assertLess(alloc.size_full, id_space)
        with self.assertRaises(_CapturedSizes) as caught:
            self._run(
                algorithm=SpeculativeAlgorithm.DSPARK,
                alloc=alloc,
                max_total_num_tokens=alloc.size_full,
            )
        sized = caught.exception.sizes.max_total_num_tokens
        self.assertEqual(sized, (id_space + _PS - 1) // _PS * _PS)

    def test_a_draft_with_a_region_binds_the_fused_pool(self):
        """An EAGLE or DSPARK draft binds the fused pool over the target's
        allocator when the target resolved a region; the region-less private
        arm above is the fallback."""
        alloc = self._swa_allocator(with_draft_region=True)
        for algorithm in (SpeculativeAlgorithm.EAGLE3, SpeculativeAlgorithm.DSPARK):
            pools = self._run(
                algorithm=algorithm,
                alloc=alloc,
                max_total_num_tokens=alloc.size_full,
            )
            self.assertIsInstance(pools.token_to_kv_pool, UnifiedDraftKVPool)
            self.assertIs(pools.token_to_kv_pool_allocator, alloc)

    def test_a_draft_kv_dtype_unlike_the_region_refuses_to_bind(self):
        """The region stores the target's KV dtype, every fp8 flavor as uint8:
        a draft resolving the target's fp8 binds a pool that casts to it, and
        one resolving another dtype refuses instead of reading the rows as
        that dtype."""
        bf16 = self._swa_allocator(with_draft_region=True)
        with self.assertRaisesRegex(ValueError, "speculative-draft-kv-cache-dtype"):
            self._run(
                algorithm=SpeculativeAlgorithm.EAGLE3,
                alloc=bf16,
                max_total_num_tokens=bf16.size_full,
                kv_dtype=torch.float8_e4m3fn,
            )
        alloc = self._swa_allocator(
            with_draft_region=True, kv_dtype=torch.float8_e4m3fn
        )
        pools = self._run(
            algorithm=SpeculativeAlgorithm.EAGLE3,
            alloc=alloc,
            max_total_num_tokens=alloc.size_full,
            kv_dtype=torch.float8_e4m3fn,
        )
        self.assertEqual(pools.token_to_kv_pool.dtype, torch.float8_e4m3fn)
        self.assertEqual(pools.token_to_kv_pool.store_dtype, torch.uint8)
        with self.assertRaisesRegex(ValueError, "speculative-draft-kv-cache-dtype"):
            self._run(
                algorithm=SpeculativeAlgorithm.EAGLE3,
                alloc=alloc,
                max_total_num_tokens=alloc.size_full,
                kv_dtype=torch.float8_e5m2,
            )

    def test_compact_dflash_draft_fuses_with_a_private_req_table(self):
        """Compact-window DFLASH passes req_to_token_pool=None (it keeps a
        private table narrowing WHICH pages the draft reads) while the KV
        itself stays fused: the fused arm must build that table instead of
        refusing the None."""
        alloc = self._swa_allocator(with_draft_region=True)
        pools = self._run(
            algorithm=SpeculativeAlgorithm.DFLASH,
            alloc=alloc,
            max_total_num_tokens=alloc.size_full,
            req_to_token_pool=None,
        )
        self.assertIsInstance(pools.token_to_kv_pool, UnifiedDraftKVPool)
        self.assertEqual(pools.req_to_token_pool.kind, "private-compact")

    def test_window_draft_binds_the_swa_composite(self):
        """MiMoV2MTP shape: the runner's one layer is a window layer placed in
        the swa sub-pool, so it binds the SWA-shaped composite (the backends
        key their swa rail on it), with no full side at all."""
        alloc = self._swa_allocator(with_draft_region=True, window_draft=True)
        with patch.object(unified_draft_pool, "draft_swa_layer_ids", return_value=(0,)):
            pools = self._run(
                algorithm=SpeculativeAlgorithm.EAGLE3,
                alloc=alloc,
                max_total_num_tokens=alloc.size_full,
            )
        pool = pools.token_to_kv_pool
        self.assertIsInstance(pool, UnifiedDraftSWAKVPool)
        self.assertEqual(pool.layers_mapping, {0: (0, True)})
        self.assertIsNone(pool.full_kv_pool)
        self.assertIs(pools.token_to_kv_pool_allocator, alloc)

    def test_a_per_depth_window_runner_trusts_the_placement(self):
        """BUG REGRESSION. `adjust_hybrid_swa_layer_ids` clips the draft
        config's window list to the runner's own block range, so a per-depth
        head above the first block reports NO window layer. The target placed
        that runner in the swa sub-pool from the pristine config, so a binder
        re-deriving the kind from the clipped config bound the wrong side and
        died on the lane-count assert (an Inkling MTP head, depth 2).
        A runner holding only swa lanes binds swa, whatever the config says."""
        alloc = self._swa_allocator(with_draft_region=True, window_draft=True)
        # The clipped config: this runner's layer is NOT listed as a window layer.
        with patch.object(unified_draft_pool, "draft_swa_layer_ids", return_value=()):
            pools = self._run(
                algorithm=SpeculativeAlgorithm.EAGLE3,
                alloc=alloc,
                max_total_num_tokens=alloc.size_full,
            )
        pool = pools.token_to_kv_pool
        self.assertIsInstance(pool, UnifiedDraftSWAKVPool)
        self.assertEqual(pool.layers_mapping, {0: (0, True)})
        self.assertIsNone(pool.full_kv_pool)

    def _bound_draft_pool(self, *, window_draft: bool):
        alloc = self._swa_allocator(with_draft_region=True, window_draft=window_draft)
        window_ids = (0,) if window_draft else ()
        with patch.object(
            unified_draft_pool, "draft_swa_layer_ids", return_value=window_ids
        ):
            pools = self._run(
                algorithm=SpeculativeAlgorithm.EAGLE,
                alloc=alloc,
                max_total_num_tokens=alloc.size_full,
            )
        return pools.token_to_kv_pool

    def test_a_widened_window_draft_widens_the_swa_retain_window(self):
        """Inkling-Small's shape: three depths bound to the swa sub-pool at
        window 511 and a draft extend widened by 5 front rows. The draft reads
        516 tokens back, so eviction keeps 516 while the window stays 511."""
        pool = self._bound_draft_pool(window_draft=True)
        self.assertIsInstance(pool, UnifiedDraftSWAKVPool)
        worker = _multi_layer_worker([(pool, 511)] * 3, front=5)
        self.assertEqual(worker.draft_swa_window, 516)
        params = _swa_cache_params(
            sliding_window_size=511, draft_swa_window=worker.draft_swa_window
        )
        self.assertEqual(params.swa_retain_window, 516)
        self.assertEqual(params.sliding_window_size, 511)

    def test_a_draft_outside_the_swa_sub_pool_keeps_the_window(self):
        """Only a widened draft in the swa sub-pool reads past the window: one
        folded into the full sub-pool, a private pool (the baseline), or an
        extend without front rows leaves the retain window at the target's."""
        full_pool = self._bound_draft_pool(window_draft=False)
        self.assertIsInstance(full_pool, UnifiedDraftKVPool)
        cases = [
            (full_pool, 5),
            (object.__new__(SWAKVPool), 5),
            (self._bound_draft_pool(window_draft=True), 0),
        ]
        for pool, front in cases:
            worker = _multi_layer_worker([(pool, 511)], front=front)
            params = _swa_cache_params(
                sliding_window_size=511, draft_swa_window=worker.draft_swa_window
            )
            self.assertEqual(params.swa_retain_window, 511, type(pool).__name__)


_WINDOW = 4
# A window-4 draft whose extend starts 5 rows below the committed length.
_DRAFT_REACH = 9


class TestSWARetainWindow(CustomTestCase):
    """BUG REGRESSION. Both frees of a running request's SWA, decode eviction
    and the free at a checkpoint insert, must keep what a fused draft reads
    (9 tokens back here), not only the target's window (4)."""

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _fixture(self, cache_cls, *, draft_swa_window, num_tokens=40, disable=False):
        req_to_token_pool = ReqToTokenPool(
            size=4, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        kv_pool = SWAKVPool(
            size=64,
            size_swa=64,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            device="cpu",
        )
        allocator = SWATokenToKVPoolAllocator(
            size=64,
            size_swa=64,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        cache = cache_cls(
            CacheInitParams(
                disable=disable,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                sliding_window_size=_WINDOW,
                draft_swa_window=draft_swa_window,
                tree_components=(ComponentType.FULL, ComponentType.SWA),
            )
        )
        req = Req(
            rid="retain-window",
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req_to_token_pool.alloc([req])
        req.origin_input_ids = list(range(1, num_tokens + 1))
        req.output_ids = []
        kv_indices = allocator.alloc(num_tokens)
        req_to_token_pool.write((req.kv.req_pool_idx, slice(0, num_tokens)), kv_indices)
        batch = _bare(
            ScheduleBatch,
            reqs=[req],
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=req_to_token_pool,
            forward_mode=ForwardMode.DECODE,
        )
        return batch, req, kv_indices

    def _decode(self, cache_cls, *, draft_swa_window, disable=False):
        """Evict at every committed length (the worst case for the draft) and
        record the frontier and whether each slot the draft reads is mapped."""
        batch, req, kv_indices = self._fixture(
            cache_cls, draft_swa_window=draft_swa_window, disable=disable
        )
        mapping = batch.token_to_kv_pool_allocator.full_to_swa_index_mapping
        req.decode_batch_idx = 1
        steps = []
        with envs.SGLANG_SWA_EVICTION_INTERVAL.override(1):
            for committed in range(12, 40):
                req.origin_input_ids = list(range(1, committed + 2))
                batch.maybe_evict_swa()
                read = kv_indices[committed - _DRAFT_REACH : committed]
                steps.append(
                    (
                        committed,
                        req.kv.get_evicted_seqlen(ComponentType.SWA),
                        bool((mapping[read] > 0).all()),
                    )
                )
        return steps

    def test_decode_eviction_keeps_every_slot_the_draft_reads(self):
        # Radix cache on, and off (the disabled tree serves --disable-radix-cache).
        for disable in (False, True):
            with self.subTest(disable=disable):
                for committed, frontier, mapped in self._decode(
                    UnifiedRadixCache, draft_swa_window=_DRAFT_REACH, disable=disable
                ):
                    self.assertEqual(frontier, committed - _DRAFT_REACH)
                    self.assertTrue(mapped, committed)
                # The target's window alone frees 5 slots the draft still reads.
                for committed, frontier, mapped in self._decode(
                    UnifiedRadixCache, draft_swa_window=0, disable=disable
                ):
                    self.assertEqual(frontier, committed - _WINDOW)
                    self.assertFalse(mapped, committed)
        # maybe_evict_swa reads the retain window off every SWA cache.
        batch, _, _ = self._fixture(PureSWARadixCache, draft_swa_window=0)
        self.assertEqual(batch.tree_cache.swa_retain_window, _WINDOW)

    def test_the_leaf_lock_outlives_the_draft_reach(self):
        """With the prefix lock released after the window, it must wait for
        the draft's reach: until then the draft still reads the prefix."""
        for draft_swa_window, released_at in ((_DRAFT_REACH, 9), (0, 4)):
            batch, req, _ = self._fixture(
                UnifiedRadixCache, draft_swa_window=draft_swa_window
            )
            req.lock = SimpleNamespace(
                receipt=DecLockRefParams(component_lock_uuids={ComponentType.SWA: 1})
            )
            with (
                envs.SGLANG_OPT_RELEASE_PREFILL_SWA.override(True),
                patch.object(batch.tree_cache, "release_swa") as release,
            ):
                for step in range(1, 13):
                    req.decode_batch_idx = step
                    batch.maybe_evict_swa()
                    if release.called:
                        break
            self.assertEqual(step, released_at)

    def test_the_free_at_a_checkpoint_keeps_every_slot_the_draft_reads(self):
        pre_len = 20
        for draft_swa_window, frontier in (
            (_DRAFT_REACH, pre_len - 1 - _DRAFT_REACH),
            (0, pre_len - 1 - _WINDOW),
        ):
            batch, req, _ = self._fixture(
                UnifiedRadixCache, draft_swa_window=draft_swa_window, num_tokens=pre_len
            )
            cache = batch.tree_cache
            req.full_untruncated_fill_ids = array("q", req.origin_input_ids)
            req.extend_end = pre_len
            req.kv.kv_committed_len = pre_len
            req.kv.cache_protected_len = 0
            req.last_node = cache.root_node_handle()
            req.lock = None
            req.extra_key = None
            with envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(True):
                checkpoint_kv_cache(req, cache)
            self.assertEqual(req.kv.get_evicted_seqlen(ComponentType.SWA), frontier)
            cache.unlock(req.lock)
            cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
