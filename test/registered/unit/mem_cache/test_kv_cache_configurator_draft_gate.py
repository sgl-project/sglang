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

    python -m pytest test/registered/unit/mem_cache/test_kv_cache_configurator_draft_gate.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.mem_cache import unified_draft_pool
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.unified_draft_pool import UnifiedDraftKVPool
from sglang.srt.mem_cache.unified_memory_pool import (
    MHASubPoolSpec,
    UnifiedKVPool,
    _store_dtype_for,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="weekly", runner_config="cpu")


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
        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=8,
            store_dtype=store_dtype,
            grow_direction="down",
            draft_region=region,
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=1,
            head_num=2,
            head_dim=8,
            store_dtype=store_dtype,
            grow_direction="up",
        )
        total = n_full * full_spec.entry_bytes() + n_swa * swa_spec.entry_bytes()
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device="cpu",
            enable_memory_saver=False,
            page_size=_PS,
            fused_draft=(
                FusedDraftPlacement(region=region, runner_lane_counts=(1,))
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


if __name__ == "__main__":
    unittest.main()
