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
"""Draft-shaped KVCacheConfigurator dispatch under the unified pool.

BUG REGRESSION (fast path). The fast path was gated on `req_to_token_pool is
None` alone, but a compact-window DFLASH draft (`--speculative-draft-window-size`)
also passes None — it builds a private req_to_token of its own — so the draft
worker would allocate a SECOND unified byte buffer at boot. A draft-shaped
configurator must fall through to the normal pool build instead.

DERIVED PROPERTY (binding dispatch). The fused-vs-private draft binding
dispatches on the spec algorithm, not the allocator kind: a non-EAGLE draft on
a unified SWA target takes the private arm, sized by the full sub-allocator's
VIRTUAL id space (`max_slots - 1`). The SWA allocator's `size_full` reports
the static token budget — smaller than the id space — so sizing by it would
put verify-window writes at high virtual ids out of bounds.

    python -m pytest test/registered/unit/mem_cache/test_kv_cache_configurator_draft_gate.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    DenseDraftRegion,
    MHASubPoolSpec,
    UnifiedDraftKVPool,
    UnifiedKVPool,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ReachedNormalBuild(Exception):
    """Sentinel: control flow fell past the unified fast path."""


class TestUnifiedFastPathDraftGate(CustomTestCase):
    def _run(self, *, is_draft_worker: bool):
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        cfg.is_draft_worker = is_draft_worker
        cfg.mambaish_config = object()  # would select the mamba unified arm
        cfg.is_hybrid_swa = False  # not the mamba+SWA tri-pool arm
        cfg.model_config = SimpleNamespace(hf_config={})  # not DSV4
        # Every field the fast path reads: a missing one raises while the
        # arm's arguments are evaluated, which reads here as "took the fast
        # path but called nothing".
        sizes = SimpleNamespace(
            max_running_requests=8,
            max_total_num_tokens=64,
            full_max_total_num_tokens=64,
            swa_max_total_num_tokens=32,
            unified_total_bytes=1 << 20,
        )
        taken = []
        with (
            patch.object(
                kcc,
                "get_memory",
                return_value=SimpleNamespace(enable_unified_memory=True),
            ),
            patch.object(
                kcc,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="null"),
            ),
            patch.object(
                kcc.KVCacheConfigurator,
                "_init_unified_mamba_pools",
                lambda self, **kw: taken.append("mamba") or None,
            ),
            patch.object(
                kcc.KVCacheConfigurator,
                "_init_unified_swa_pools",
                lambda self, **kw: taken.append("swa") or None,
            ),
            patch.object(
                kcc.KVCacheConfigurator,
                "_build_req_to_token_pool",
                side_effect=_ReachedNormalBuild,
            ),
        ):
            try:
                cfg._init_pools(
                    sizes=sizes,
                    req_to_token_pool=None,
                    token_to_kv_pool_allocator=None,
                )
            except _ReachedNormalBuild:
                return "normal", taken
            except (AttributeError, TypeError):
                # The stubbed unified arm returned None; anything past the
                # fast-path dispatch counts as having taken it.
                return "unified", taken
        return "unified", taken

    def test_draft_worker_falls_through_to_the_normal_build(self):
        path, taken = self._run(is_draft_worker=True)
        self.assertEqual(path, "normal")
        self.assertEqual(taken, [])

    def test_target_worker_still_takes_the_fast_path(self):
        """The guard must narrow to drafts only — a target regression here
        silently turns --enable-unified-memory into a no-op."""
        path, taken = self._run(is_draft_worker=False)
        self.assertEqual(taken, ["mamba"])


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
    def _swa_allocator(self, *, with_draft_region: bool, n_full=32, n_swa=16):
        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=2,
            head_num=2,
            head_dim=4,
            store_dtype=torch.bfloat16,
            grow_direction="down",
            draft_region=(
                DenseDraftRegion(
                    layer_num=1, head_num=1, head_dim=3, store_dtype=torch.bfloat16
                )
                if with_draft_region
                else None
            ),
        )
        swa_spec = MHASubPoolSpec(
            name="swa",
            layer_num=1,
            head_num=2,
            head_dim=4,
            store_dtype=torch.bfloat16,
            grow_direction="up",
        )
        total = n_full * full_spec.entry_bytes() + n_swa * swa_spec.entry_bytes()
        pool = UnifiedKVPool(
            total_bytes=total,
            sub_pool_specs=[full_spec, swa_spec],
            device="cpu",
            enable_memory_saver=False,
            page_size=_PS,
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
        req_to_token_pool="shared",
    ):
        if req_to_token_pool == "shared":
            req_to_token_pool = object()
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        cfg.is_draft_worker = True
        cfg.spec_algorithm = algorithm
        cfg.page_size = _PS
        cfg.is_hybrid_swa_mtp_draft = False
        cfg.model_config = SimpleNamespace(hf_config=None)
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

    def test_eagle_draft_still_binds_the_fused_pool(self):
        alloc = self._swa_allocator(with_draft_region=True)
        pools = self._run(
            algorithm=SpeculativeAlgorithm.EAGLE3,
            alloc=alloc,
            max_total_num_tokens=alloc.size_full,
        )
        self.assertIsInstance(pools.token_to_kv_pool, UnifiedDraftKVPool)
        self.assertIs(pools.token_to_kv_pool_allocator, alloc)

    def test_dspark_draft_with_a_region_binds_the_fused_pool(self):
        """DSPARK's draft KV fuses when the target resolved a region (its
        block rows are indexed by the same token->page identity); the
        region-less private arm above remains the automatic fallback."""
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedDraftKVPool

        alloc = self._swa_allocator(with_draft_region=True)
        pools = self._run(
            algorithm=SpeculativeAlgorithm.DSPARK,
            alloc=alloc,
            max_total_num_tokens=alloc.size_full,
        )
        self.assertIsInstance(pools.token_to_kv_pool, UnifiedDraftKVPool)

    def test_compact_dflash_draft_fuses_with_a_private_req_table(self):
        """Compact-window DFLASH passes req_to_token_pool=None (it keeps a
        private table narrowing WHICH pages the draft reads) while the KV
        itself stays fused: the fused arm must build that table instead of
        refusing the None."""
        from sglang.srt.mem_cache.unified_memory_pool import UnifiedDraftKVPool

        alloc = self._swa_allocator(with_draft_region=True)
        pools = self._run(
            algorithm=SpeculativeAlgorithm.DFLASH,
            alloc=alloc,
            max_total_num_tokens=alloc.size_full,
            req_to_token_pool=None,
        )
        self.assertIsInstance(pools.token_to_kv_pool, UnifiedDraftKVPool)
        self.assertEqual(pools.req_to_token_pool.kind, "private-compact")

    def test_eagle_draft_without_a_region_falls_back_to_the_private_arm(self):
        """Target boot declines a region for legitimate geometry (asymmetric
        draft rows), so a region-less EAGLE draft binds the private pool -
        sized by the id space like any other private draft - instead of
        failing the boot."""
        alloc = self._swa_allocator(with_draft_region=False)
        id_space = alloc.full_attn_allocator.max_slots - 1
        with self.assertRaises(_CapturedSizes) as caught:
            self._run(
                algorithm=SpeculativeAlgorithm.EAGLE3,
                alloc=alloc,
                max_total_num_tokens=alloc.size_full,
            )
        sized = caught.exception.sizes.max_total_num_tokens
        self.assertEqual(sized, (id_space + _PS - 1) // _PS * _PS)


if __name__ == "__main__":
    unittest.main()
