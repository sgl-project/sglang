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
"""RunnerIds with a fused draft region.

A FUSED-DRAFT runner (`DraftMHAView` bound to the target's allocator)
translates exactly as the target does -- same v2p table, same physical ids,
the draft parts sitting inside the host entries -- and, routing no window
layers, has no sliding-window write ids; one with window layers in the swa
sub-pool rides the target's swa rail. A PRIVATE-POOL draft (own
virtual-indexed buffer) stays a strict passthrough. The draft writers outside
the draft's forward (DFLASH/DSPARK's target-hidden writes, the multi-step
draft decode's kv indices) read the same ids.

    python -m pytest test/registered/unit/mem_cache/test_runner_ids_draft_disposition.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.mem_cache.kv.layout.mha import DraftMHAView, MHASubPoolSpec
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.unified_draft_pool import (
    routed_draft_pool,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.mem_cache_utils import routed
from sglang.test.unified_allocator_fixtures import (
    arena_of,
    plan_stand_in,
    stand_in_views,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_DEV = "cpu"
_PS = 2


class _FakeKVCache:
    # A private pool, indexed by the allocator's ids as they are.
    requires_physical_write_loc = False

    def __init__(self, source, name=None):
        # A bare size for a private pool; else the band of ``source`` (a test
        # arena) whose unit moves this buffer follows.
        size = source if name is None else source.num_entries(name)
        if name is not None:
            source.follow(name, self)
        self.buf = torch.full((size,), -1, dtype=torch.int64)
        self.allocator = None

    def move_units(self, dst_pages, src_pages, page_size):
        slots = torch.arange(page_size, device=dst_pages.device)
        dst_loc = (dst_pages[:, None] * page_size + slots).flatten()
        src_loc = (src_pages[:, None] * page_size + slots).flatten()
        self.buf[dst_loc] = self.buf[src_loc].clone()

    def id_spaces(self):
        return {IdSpaceKind.FULL: IdSpace(key=(IdSpaceKind.FULL, None))}

    def attach_allocator(self, allocator):
        self.allocator = allocator

    def move_kv_cache(self, dst_loc, src_loc):
        self.buf[dst_loc] = self.buf[src_loc].clone()


class _FakeUnifiedSWAKVPool:
    def __init__(self, shared_pool):
        self.full_kv_pool = _FakeKVCache(shared_pool, "full")
        self.swa_kv_pool = _FakeKVCache(shared_pool, "swa")
        self.full_to_swa_index_mapping = None

    def attach_allocators(self, *, full_allocator, swa_allocator):
        self._full_allocator = full_allocator
        self._swa_allocator = swa_allocator


def _build(n_full=32, n_swa=16, *, swa_region=None):
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=2,
        head_num=2,
        head_dim=4,
        store_dtype=torch.bfloat16,
        grow_direction="down",
        draft_region=DenseDraftRegion(
            lane_num=1, head_num=1, head_dim=8, store_dtype=torch.bfloat16
        ),
    )
    swa_spec = MHASubPoolSpec(
        name="swa",
        layer_num=1,
        head_num=2,
        head_dim=4,
        store_dtype=torch.bfloat16,
        grow_direction="up",
        draft_region=swa_region,
    )
    total = n_full * full_spec.entry_bytes() + n_swa * swa_spec.entry_bytes()
    counts = {"full": [1]}
    regions = {"full": full_spec.draft_region}
    if swa_region is not None:
        counts["swa"] = [1]
        regions["swa"] = swa_region
    pool = arena_of(
        total_bytes=total,
        sub_pool_specs=[full_spec, swa_spec],
        device=_DEV,
        enable_memory_saver=False,
        page_size=_PS,
        fused_draft=FusedDraftPlacement.from_counts(counts=counts, regions=regions),
    )
    kvcache = _FakeUnifiedSWAKVPool(pool)
    allocator = ChainAllocator(
        roles=ChainRoles(owner="full", window="swa"),
        arena=pool,
        kvcache=kvcache,
        views=stand_in_views(ChainRoles(owner="full", window="swa"), kvcache),
        device=_DEV,
        full_max_total_num_tokens=n_full,
        swa_max_total_num_tokens=n_swa,
        page_size=_PS,
        need_sort=False,
        forward_stream=None,
    )
    draft_pool = DraftMHAView(
        arena=pool,
        host_spec=pool.spec("full"),
        layer_lanes={0: 0},
        page_size=_PS,
    )
    return pool, allocator, kvcache, draft_pool


def _source(allocator, pool_obj):
    return RunnerIds(
        req_to_token=torch.zeros((4, 16), dtype=torch.int32),
        token_to_kv_pool_allocator=allocator,
        token_to_kv_pool=pool_obj,
        page_size=_PS,
        device=_DEV,
    )


class TestRunnerIdsDraftDisposition(unittest.TestCase):
    def setUp(self):
        # `RunnerIds.__init__` reads `attn_dcp_size`, a derived
        # parallel width that only exists once a config is published.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_fused_draft_runner_translates_to_the_same_physical_ids(self):
        _, allocator, _, draft_pool = _build()
        src = _source(allocator, draft_pool)
        self.assertTrue(src.is_translating)
        # A dense draft pool: no sliding-window sub-pool.
        self.assertIsNone(src.space(IdSpaceKind.SLIDING_WINDOW))

        # Its own plan translates to the HOST's physical ids: the draft parts
        # live inside the same slots.
        v = allocator.alloc(2 * _PS)
        self.assertIsNotNone(v)
        fb = SimpleNamespace(out_cache_loc=v)
        src.bind_own_plan(fb)
        self.assertIsNot(fb.out_cache_loc, v)
        fa = allocator.full_attn_allocator
        expected = torch.clamp_min(
            fa.ids.virtual_to_physical[v // _PS] * _PS + v % _PS, 0
        )
        torch.testing.assert_close(fb.out_cache_loc, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(fb.out_cache_loc, allocator.translate_kv_loc(v)))
        # A dense draft pool routes no window layers: no swa write loc.
        self.assertIsNone(src.write_ids(fb, IdSpaceKind.SLIDING_WINDOW))

    def test_swa_shaped_fused_draft_rides_the_target_swa_rail(self):
        """A draft with window layers binds the swa sub-pool, so its runner
        gets the same swa rail as the target: the window write ids and the
        window read table are the target's, byte for byte."""
        swa_region = DenseDraftRegion(
            lane_num=1, head_num=1, head_dim=8, store_dtype=torch.bfloat16
        )
        pool, allocator, kvcache, _ = _build(swa_region=swa_region)
        composite = routed_draft_pool(
            arena=pool,
            plan=plan_stand_in(pool),
            owner="full",
            page_size=_PS,
            lanes_by_host={"full": {0: 0}, "swa": {1: 0}},
        )
        v = allocator.alloc(2 * _PS)
        self.assertIsNotNone(v)
        rt = torch.zeros((4, 16), dtype=torch.int32)
        rt[0, : v.numel()] = v.to(torch.int32)
        views = []
        for pool_obj in (composite, kvcache):
            runner_ids = RunnerIds(
                req_to_token=rt,
                token_to_kv_pool_allocator=allocator,
                token_to_kv_pool=pool_obj,
                page_size=_PS,
                device=_DEV,
            )
            self.assertTrue(runner_ids.is_translating)
            fb = SimpleNamespace(
                out_cache_loc=v,
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([2 * _PS], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([2 * _PS], dtype=torch.int64),
            )
            runner_ids.bind_own_plan(fb)
            views.append(
                (
                    runner_ids.write_ids(fb, IdSpaceKind.SLIDING_WINDOW),
                    runner_ids.read_table(
                        fb.iteration_ids, kind=IdSpaceKind.SLIDING_WINDOW, rows=1
                    ).ids,
                )
            )
        (draft_write, draft_read), (target_write, target_read) = views
        self.assertTrue(
            torch.equal(
                draft_write, allocator.id_spaces()[IdSpaceKind.SLIDING_WINDOW].ids(v)
            )
        )
        self.assertTrue(torch.equal(draft_write, target_write))
        self.assertTrue(torch.equal(draft_read, target_read))

    def test_a_window_depth_reads_the_plan_a_dense_depth_built(self):
        """A multi-layer draft plans each draft extend once, through its first
        depth's runner, and every depth reads that plan. When the first depth
        is dense and a later one has window layers, the later one reads its
        window ids from a plan whose runner routes no window layers."""
        swa_region = DenseDraftRegion(
            lane_num=1, head_num=1, head_dim=8, store_dtype=torch.bfloat16
        )
        pool, allocator, kvcache, dense_pool = _build(swa_region=swa_region)
        window_pool = routed_draft_pool(
            arena=pool,
            plan=plan_stand_in(pool),
            owner="full",
            page_size=_PS,
            lanes_by_host={"swa": {1: 0}},
        )
        v = allocator.alloc(2 * _PS)
        self.assertIsNotNone(v)
        rt = torch.zeros((4, 16), dtype=torch.int32)
        rt[0, : v.numel()] = v.to(torch.int32)
        dense, window, target = (
            RunnerIds(
                req_to_token=rt,
                token_to_kv_pool_allocator=allocator,
                token_to_kv_pool=pool_obj,
                page_size=_PS,
                device=_DEV,
            )
            for pool_obj in (dense_pool, window_pool, kvcache)
        )
        self.assertIsNone(dense.space(IdSpaceKind.SLIDING_WINDOW))

        def batch():
            return SimpleNamespace(
                out_cache_loc=v,
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                seq_lens=torch.tensor([2 * _PS], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([2 * _PS], dtype=torch.int64),
            )

        def window_ids(runner_ids, plan):
            # The Triton backend's sliding-window buffer fill.
            out = torch.full((2 * _PS,), -1, dtype=torch.int64)
            runner_ids.pack_read_stream(
                plan,
                req_pool_indices=plan.req_pool_indices,
                seq_lens=plan.seq_lens,
                indptr=torch.tensor([0, 2 * _PS], dtype=torch.int32),
                out=out,
                kind=IdSpaceKind.SLIDING_WINDOW,
            )
            return out

        dense_fb, target_fb = batch(), batch()
        dense.bind_own_plan(dense_fb)
        target.bind_own_plan(target_fb)
        expected = allocator.id_spaces()[IdSpaceKind.SLIDING_WINDOW].ids(v)
        self.assertTrue(
            torch.equal(window_ids(window, dense_fb.iteration_ids), expected)
        )
        self.assertTrue(
            torch.equal(window_ids(target, target_fb.iteration_ids), expected)
        )
        # A captured graph's window table, which the depth's later gathers
        # then read from the plan.
        table = window.read_table(
            dense_fb.iteration_ids, kind=IdSpaceKind.SLIDING_WINDOW, rows=1
        )
        target_table = target.read_table(
            target_fb.iteration_ids, kind=IdSpaceKind.SLIDING_WINDOW, rows=1
        )
        self.assertTrue(torch.equal(table.ids, target_table.ids))
        src = window.read_source(
            dense_fb.iteration_ids,
            req_pool_indices=dense_fb.req_pool_indices,
            bs=1,
            kind=IdSpaceKind.SLIDING_WINDOW,
        )
        self.assertIs(src.ids, table.ids)

    def test_target_hidden_writers_write_physical_ids(self):
        """DFLASH and DSPARK project the target's hidden states straight into
        the draft pool. Under fusion that pool takes the target's physical ids:
        both writers take them from the iteration's plan in the draft pool's
        space, never the virtual ids (a wrong row, with no crash)."""
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
        from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
            TargetHiddenKvInjector,
        )

        class _Shift:
            """A translating runner whose physical id is the virtual + 100."""

            is_translating = True

        virtual = torch.tensor([4, 5, 6, 9], dtype=torch.int64)
        physical = virtual + 100
        draft_translator = _Shift()
        # The iteration's plan: physical ids for the fused draft's runner_ids.
        plan = SimpleNamespace(
            write_ids=lambda reader, cols=None: (
                physical.clone() if reader is draft_translator else virtual.clone()
            )
        )
        hidden = torch.randn(4, 8)
        positions = torch.arange(4)
        commit_lens = torch.tensor([2, 1], dtype=torch.int32)

        writes = {}

        class _DflashPool:
            def set_kv_buffer(self, layer, loc, k, v, k_scale, v_scale):
                writes["loc"] = loc

            def set_kv_buffer_prefix_valid(
                self, layer, cache_loc_2d, commit_lens, k, v, k_scale, v_scale
            ):
                writes["loc_2d"] = cache_loc_2d

        attn = SimpleNamespace(
            layer_id=0,
            kv_proj_only=lambda h: (h[:, :4], h[:, 4:]),
            apply_k_norm=lambda k: k,
            apply_k_rope=lambda pos, k: k,
            num_kv_heads=1,
            head_dim=4,
            attn=SimpleNamespace(layer_id=0, k_scale=None, v_scale=None),
        )
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(device=torch.device(_DEV)),
            draft_model_runner=SimpleNamespace(
                runner_ids=draft_translator,
                token_to_kv_pool=routed(_DflashPool()),
            ),
            draft_model=SimpleNamespace(
                project_target_hidden=lambda h: h,
                prepare_context_hidden_for_kv=lambda layer, h: h,
                layers=[SimpleNamespace(self_attn=attn)],
            ),
            draft_parallel_layout=SimpleNamespace(tp_group=None),
            lilicorr=None,
            _use_fused_kv_materialize=False,
            _fused_kv_helper=None,
        )
        worker._append_target_hidden_sequential = lambda **kw: (
            DFlashWorkerV2._append_target_hidden_sequential(worker, **kw)
        )

        # Per-token writes (prefill).
        cache_loc = DFlashWorkerV2._draft_write_ids(worker, plan)
        DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc(
            worker, target_hidden=hidden, cache_loc=cache_loc, positions=positions
        )
        self.assertTrue(writes["loc"].physical)
        torch.testing.assert_close(writes["loc"].loc, physical, rtol=0, atol=0)

        # Prefix-valid writes (post-verify), from the 2-D view of the same ids.
        cache_loc = DFlashWorkerV2._draft_write_ids(worker, plan)
        DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc(
            worker,
            target_hidden=hidden,
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=cache_loc.view(2, 2),
            commit_lens=commit_lens,
        )
        torch.testing.assert_close(
            writes["loc_2d"], physical.view(2, 2), rtol=0, atol=0
        )

        # DSPARK's injector, on an MHA draft pool: the plan's ids for the
        # draft's runner_ids, written as they are.
        injected = {}

        def write_target_hidden_kv(**kwargs):
            injected.update(kwargs)

        injector = TargetHiddenKvInjector(
            draft_model=SimpleNamespace(write_target_hidden_kv=write_target_hidden_kv),
            draft_model_runner=SimpleNamespace(
                runner_ids=draft_translator,
                token_to_kv_pool=routed(SimpleNamespace()),
            ),
            model_runner=SimpleNamespace(device=torch.device(_DEV)),
            device=torch.device(_DEV),
            verify_num_draft_tokens=2,
            block_pos_offsets=torch.arange(2),
        )
        cache_loc = injector.ids_for(plan)
        injector.inject_target_hidden(
            target_hidden=hidden,
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=cache_loc.view(2, 2),
            commit_lens=commit_lens,
        )
        torch.testing.assert_close(injected["cache_loc"], physical, rtol=0, atol=0)
        torch.testing.assert_close(
            injected["cache_loc_2d"], physical.view(2, 2), rtol=0, atol=0
        )

    def test_full_v2p_table_per_disposition(self):
        """The v2p accessor must hand a DCP table builder exactly what
        `translate_kv_loc` would use: the shared v2p table on the target and
        on the fused draft, and None on a pass-through runner (a translated
        pass-through would address a private buffer with the target's ids)."""
        _, allocator, kvcache, draft_pool = _build()

        target = _source(allocator, kvcache)
        self.assertIs(target.full_v2p_table, allocator.full_v2p_page_table)

        draft = _source(allocator, draft_pool)
        self.assertIs(draft.full_v2p_table, allocator.full_v2p_page_table)

        passthrough = _source(allocator, _FakeKVCache(64))
        self.assertIsNone(passthrough.full_v2p_table)

    def test_multi_step_containers_read_the_plans_table(self):
        """Every `generate_draft_decode_kv_indices` launch, and every launch of
        its sliding-window sibling, must gather from `read_source` with its
        entry granularity. A launch over raw req_to_token emits VIRTUAL ids,
        which the fused draft pool cannot address -- silent garbage drafts."""
        import pathlib
        import re

        import sglang.srt.mem_cache.kv.alloc.runner_ids as _kit

        root = pathlib.Path(_kit.__file__).parents[3] / "layers" / "attention"
        launching = {}
        for path in sorted(root.glob("*.py")):
            text = path.read_text()
            launches = len(
                re.findall(r"generate_draft_decode_(?:window_)?kv_indices\[", text)
            )
            if launches:
                launching[path.name] = (
                    launches,
                    len(re.findall(r"ENTRY_PAGE_SIZE=src\.entry_page_size", text)),
                    "runner_ids.read_source(" in text,
                )
        self.assertGreaterEqual(
            len(launching), 4, f"launch sites disappeared: {sorted(launching)}"
        )
        for name, (launches, granular, has_source) in launching.items():
            self.assertEqual(
                launches,
                granular,
                f"{name}: {launches} kernel launch(es) but only {granular} "
                "pass the read table's entry granularity",
            )
            self.assertTrue(has_source, f"{name} launches without read_source")


if __name__ == "__main__":
    unittest.main()
