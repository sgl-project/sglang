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
"""KVIndexTranslator's three id-space dispositions with a fused draft region.

1. TARGET runner (allocator's own kvcache): translates to physical ids.
2. FUSED-DRAFT runner (`UnifiedDraftKVPool` bound to this allocator): the
   SAME translate -- same pages, same v2p table, same physical ids (the draft
   parts sit inside the host entry). A dense draft pool routes no window
   layers, so it has no swa id space: the window index table falls back to
   the one dense table and its plan binds no sliding-window write ids (a
   separate-swa assumption here crashed the read side and left the write side
   with nothing to derive).
3. PRIVATE-POOL draft (DSPARK/DFLASH shape: target's allocator, own
   virtual-indexed buffer): strict passthrough — translating such a runner
   would address a slot-count buffer with dense ids (OOB both directions),
   so the no-op is load-bearing, not a default.

    python -m pytest test/registered/unit/mem_cache/test_kv_index_translator_draft_disposition.py -v
"""

import ast
import pathlib
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache import kv_index_translator
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    FusedDraftPlacement,
)
from sglang.srt.mem_cache.unified_draft_pool import UnifiedDraftKVPool
from sglang.srt.mem_cache.unified_memory_pool import MHASubPoolSpec, UnifiedKVPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DEV = "cpu"
_PS = 2


class _FakeKVCache:
    def __init__(self, max_slots):
        self.buf = torch.full((max_slots,), -1, dtype=torch.int64)
        self.allocator = None

    def attach_allocator(self, allocator):
        self.allocator = allocator

    def move_kv_cache(self, dst_loc, src_loc):
        self.buf[dst_loc] = self.buf[src_loc].clone()


class _FakeUnifiedSWAKVPool:
    def __init__(self, shared_pool):
        self.full_kv_pool = _FakeKVCache(shared_pool.max_slots("full"))
        self.swa_kv_pool = _FakeKVCache(shared_pool.max_slots("swa"))
        self.full_to_swa_index_mapping = None

    def attach_allocators(self, *, full_allocator, swa_allocator):
        self._full_allocator = full_allocator
        self._swa_allocator = swa_allocator


def _build(n_full=32, n_swa=16):
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
    )
    total = n_full * full_spec.entry_bytes() + n_swa * swa_spec.entry_bytes()
    pool = UnifiedKVPool(
        total_bytes=total,
        sub_pool_specs=[full_spec, swa_spec],
        device=_DEV,
        enable_memory_saver=False,
        page_size=_PS,
        fused_draft=FusedDraftPlacement(
            region=full_spec.draft_region, runner_lane_counts=(1,)
        ),
    )
    kvcache = _FakeUnifiedSWAKVPool(pool)
    allocator = UnifiedSWATokenToKVPoolAllocator(
        unified_buffer=pool,
        kvcache=kvcache,
        device=_DEV,
        full_max_total_num_tokens=n_full,
        swa_max_total_num_tokens=n_swa,
        page_size=_PS,
        need_sort=False,
        forward_stream=None,
    )
    draft_pool = UnifiedDraftKVPool(
        unified_buffer=pool,
        host_sub_pool_name="full",
        host_allocator=allocator,
        layer_lanes={0: 0},
        page_size=_PS,
    )
    return pool, allocator, kvcache, draft_pool


def _source(allocator, pool_obj):
    return KVIndexTranslator(
        req_to_token=torch.zeros((4, 16), dtype=torch.int32),
        token_to_kv_pool_allocator=allocator,
        token_to_kv_pool=pool_obj,
        page_size=_PS,
        device=_DEV,
    )


class TestKVIndexTranslatorDraftDisposition(unittest.TestCase):
    def setUp(self):
        # `KVIndexTranslator.__init__` reads `attn_dcp_size`, a derived
        # parallel width that only exists once a config is published.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_target_runner_translates_to_physical_ids(self):
        _, allocator, kvcache, _ = _build()
        src = _source(allocator, kvcache)
        self.assertTrue(src.is_translating)
        v = allocator.alloc(2 * _PS)
        self.assertIsNotNone(v)
        fb = SimpleNamespace(out_cache_loc=v)
        src.bind_own_plan(fb)
        self.assertTrue(torch.equal(fb.out_cache_loc, allocator.translate_kv_loc(v)))

    def test_fused_draft_runner_translates_to_the_same_physical_ids(self):
        _, allocator, _, draft_pool = _build()
        src = _source(allocator, draft_pool)
        self.assertTrue(src.is_translating)
        self.assertIsNone(src._swa_v2p_table)  # single space: no swa id space

        # Its own plan translates to the HOST's physical ids: the draft parts
        # live inside the same slots.
        v = allocator.alloc(2 * _PS)
        self.assertIsNotNone(v)
        fb = SimpleNamespace(out_cache_loc=v)
        src.bind_own_plan(fb)
        self.assertIsNot(fb.out_cache_loc, v)
        fa = allocator.full_attn_allocator
        expected = torch.clamp_min(fa.virtual_to_physical[v // _PS] * _PS + v % _PS, 0)
        torch.testing.assert_close(fb.out_cache_loc, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(fb.out_cache_loc, allocator.translate_kv_loc(v)))
        # A dense draft pool routes no window layers: no swa write loc.
        self.assertIsNone(fb.out_cache_loc_swa)

    def test_dense_fused_draft_has_no_window_write_loc(self):
        """A dense fused-draft batch binds no sliding-window write ids: the
        pool routes no window layers, so there is nothing to derive. The
        hybrid-SWA TARGET on the same allocator keeps genuinely derived swa
        ids, so the None can never paper over a real swa id space."""
        _, allocator, kvcache, draft_pool = _build()
        src = _source(allocator, draft_pool)
        v = allocator.alloc(_PS)
        self.assertIsNotNone(v)
        fb = SimpleNamespace(out_cache_loc=v)
        src.bind_own_plan(fb)
        self.assertIsNone(fb.out_cache_loc_swa)
        # Target contrast: a real swa side derives a DIFFERENT loc.
        tgt = _source(allocator, kvcache)
        tv = allocator.alloc(_PS)
        self.assertIsNotNone(tv)
        tfb = SimpleNamespace(out_cache_loc=tv)
        tgt.bind_own_plan(tfb)
        tswa = tfb.out_cache_loc_swa
        self.assertIsNotNone(tswa)
        self.assertIsNot(tswa, tfb.out_cache_loc)
        self.assertTrue(torch.equal(tswa, allocator.translate_loc_from_full_to_swa(tv)))

    def test_private_pool_draft_stays_a_strict_passthrough(self):
        _, allocator, _, _ = _build()
        private_draft_pool = _FakeKVCache(64)  # own buffer, not the allocator's
        src = _source(allocator, private_draft_pool)
        self.assertFalse(src.is_translating)
        v = torch.arange(2 * _PS, dtype=torch.int64)
        fb = SimpleNamespace(out_cache_loc=v)
        src.bind_own_plan(fb)
        self.assertIs(fb.out_cache_loc, v)  # untouched, not even a copy

    def test_foreign_allocator_draft_pool_is_not_enabled(self):
        # A UnifiedDraftKVPool bound to a DIFFERENT allocator must not enable
        # against this one (identity, not type, decides).
        pool, allocator, kvcache, _ = _build()
        _, other_alloc, _, other_draft = _build()
        src = _source(allocator, other_draft)
        self.assertFalse(src.is_translating)

    def _one_row(self):
        _, allocator, kvcache, _ = _build()
        v = allocator.alloc(6 * _PS)
        self.assertIsNotNone(v)
        rt = torch.zeros((2, 16), dtype=torch.int32)
        rt[0, : v.numel()] = v.to(torch.int32)
        src = KVIndexTranslator(
            req_to_token=rt,
            token_to_kv_pool_allocator=allocator,
            token_to_kv_pool=kvcache,
            page_size=_PS,
            device=_DEV,
        )
        return src, torch.tensor([0], dtype=torch.int64), torch.tensor([3])

    def test_read_extent_matches_widened_lens(self):
        """A plan reading ``k`` past its lengths must be byte-identical to one
        built over ``seq_lens + k`` -- the two spellings of the whole-sequence
        verify widening. A kernel applying the extent to the page count but
        not the loads (or vice versa) silently truncates the verify tail."""
        src, rpi, seq = self._one_row()
        delta = 2 * _PS + 1
        widened = src.plan(
            req_pool_indices=rpi,
            seq_lens=seq,
            seq_lens_cpu=seq.clone(),
            write_virtual=None,
            read_extent=delta,
        ).read_table()
        by_lens = src.plan(
            req_pool_indices=rpi,
            seq_lens=seq + delta,
            seq_lens_cpu=seq + delta,
            write_virtual=None,
        ).read_table()
        self.assertEqual(widened.ids.shape, by_lens.ids.shape)
        torch.testing.assert_close(widened.ids, by_lens.ids, rtol=0, atol=0)
        # The extent genuinely widened: entries exist past the plain prefix.
        plain_pages = -(-3 // _PS)
        self.assertTrue(bool((widened.ids[0, plain_pages:] > 0).any()))

    def test_a_verify_plans_its_draft_tail(self):
        """A verify's own plan reads its `draft_token_num` past its lengths, as
        the widened-lens build over the SAME batch does."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        src, rpi, seq = self._one_row()
        delta = 2 * _PS + 1
        fb = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=SimpleNamespace(draft_token_num=delta),
            batch_size=1,
            req_pool_indices=rpi,
            seq_lens=seq,
            seq_lens_cpu=seq.clone(),
            out_cache_loc=torch.zeros(delta, dtype=torch.int64),
        )
        widened = src.own_plan(fb).read_table()
        by_lens = src.plan(
            req_pool_indices=rpi,
            seq_lens=seq + delta,
            seq_lens_cpu=seq + delta,
            write_virtual=None,
        ).read_table()
        torch.testing.assert_close(widened.ids, by_lens.ids, rtol=0, atol=0)

    def test_target_hidden_writers_write_physical_ids(self):
        """BUG REGRESSION. DFLASH and DSPARK do not compute their draft KV from
        the draft's own forward -- they PROJECT the target's hidden states and
        write them straight into the draft pool, at locs read off the target's
        req_to_token (VIRTUAL). Under fusion the draft pool takes the target's
        physical ids, so an untranslated write lands at the wrong row: the
        draft attends over the wrong KV and accept length collapses to 1.0
        with no crash. Both writers take the ids from the iteration's plan in
        the draft pool's space -- the target's physical ids under fusion --
        and write them as they are."""
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
        # The iteration's plan: physical ids for the fused draft's translator.
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
            kv_proj_only=lambda h: (h[:, :4], h[:, 4:]),
            apply_k_norm=lambda k: k,
            apply_k_rope=lambda pos, k: k,
            num_kv_heads=1,
            head_dim=4,
            attn=SimpleNamespace(k_scale=None, v_scale=None),
        )
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(device=torch.device(_DEV)),
            draft_model_runner=SimpleNamespace(
                kv_index_translator=draft_translator, token_to_kv_pool=_DflashPool()
            ),
            draft_model=SimpleNamespace(
                project_target_hidden=lambda h: h,
                prepare_context_hidden_for_kv=lambda layer, h: h,
                layers=[SimpleNamespace(self_attn=attn)],
            ),
            draft_owns_attention=False,
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
        # draft's translator, written as they are.
        injected = {}

        def write_target_hidden_kv(**kwargs):
            injected.update(kwargs)

        injector = TargetHiddenKvInjector(
            draft_model=SimpleNamespace(write_target_hidden_kv=write_target_hidden_kv),
            draft_model_runner=SimpleNamespace(
                kv_index_translator=draft_translator, token_to_kv_pool=SimpleNamespace()
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
        """Every `generate_draft_decode_kv_indices` launch must gather from
        `read_source` with its entry granularity. A launch over raw
        req_to_token emits VIRTUAL ids, which the fused draft pool cannot
        address -- silent garbage drafts."""
        import pathlib
        import re

        import sglang.srt.mem_cache.kv_index_translator as _kit

        root = pathlib.Path(_kit.__file__).parent.parent / "layers" / "attention"
        launching = {}
        for path in sorted(root.glob("*.py")):
            text = path.read_text()
            launches = len(re.findall(r"generate_draft_decode_kv_indices\[", text))
            if launches:
                launching[path.name] = (
                    launches,
                    len(re.findall(r"ENTRY_PAGE_SIZE=src\.entry_page_size", text)),
                    "kv_index_translator.read_source(" in text,
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


class TestDispositionBranchesAgree(unittest.TestCase):
    """Every `__init__` disposition must assign the SAME attribute set.

    BUG REGRESSION. `KVIndexTranslator.__init__` picks a disposition --
    translating (unified target or fused draft) or passthrough -- and later
    methods read attributes off `self` unconditionally. When upstream adds an
    attribute it naturally adds it to the branches IT knows about; a branch
    added here is silently left short, and nothing fails until the missing
    attribute is read at RUNTIME. That is exactly how the fused-draft branch
    lost `_translate_write_full` and `defer_read_translate` across a rebase:
    py_compile passes (the attribute is only ever read, never declared), the
    undefined-NAME check passes (it is an attribute, not a bare name), and the
    target and passthrough paths both work -- only a fused-draft forward raises
    `AttributeError: 'KVIndexTranslator' object has no attribute
    '_translate_write_full'`.

    Comparing the branches against EACH OTHER needs no list to maintain: a new
    attribute is covered the moment any one branch sets it.

        python -m pytest test/registered/unit/mem_cache/test_kv_index_translator_draft_disposition.py -v
    """

    def setUp(self):
        # `KVIndexTranslator.__init__` reads `attn_dcp_size`, a derived
        # parallel width that only exists once a config is published.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _init_branch_assignments(self):
        """{branch index: {attr names it assigns}} for __init__'s if/elif/else."""
        src = pathlib.Path(kv_index_translator.__file__).read_text()
        tree = ast.parse(src)
        cls = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == "KVIndexTranslator"
        )
        init = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )

        # The disposition chain is the `if` whose body assigns the id-space
        # fields; find it by the attribute every disposition must set.
        def assigns(stmts):
            out = set()
            for st in stmts:
                for node in ast.walk(st):
                    if isinstance(node, ast.Assign):
                        for t in node.targets:
                            if (
                                isinstance(t, ast.Attribute)
                                and isinstance(t.value, ast.Name)
                                and t.value.id == "self"
                            ):
                                out.add(t.attr)
            return out

        for node in init.body:
            if not isinstance(node, ast.If):
                continue
            branches, cur = [], node
            while True:
                branches.append(assigns(cur.body))
                if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
                    cur = cur.orelse[0]
                    continue
                if cur.orelse:
                    branches.append(assigns(cur.orelse))
                break
            if len(branches) >= 2 and all("_full_v2p_table" in b for b in branches):
                return branches
        self.fail("could not locate the disposition if/elif/else in __init__")

    def test_every_disposition_assigns_the_same_attributes(self):
        branches = self._init_branch_assignments()
        union = set().union(*branches)
        missing = {i: sorted(union - b) for i, b in enumerate(branches) if union - b}
        self.assertEqual(
            missing,
            {},
            "a KVIndexTranslator.__init__ disposition does not assign every "
            "attribute its siblings do; the branch raises AttributeError only "
            "when that path is taken at runtime: " + repr(missing),
        )


if __name__ == "__main__":
    unittest.main()
