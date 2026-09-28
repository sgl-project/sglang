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
"""The write loc's physical mark.

Under the token-major views a virtual id is in range and, by value, the same
kind of integer as a physical one, so a unified pool cannot tell a skipped
rebind from a translated loc by looking at it. `rebind_write_loc` marks the
batch's loc physical, backends carry the mark through `KVWriteLoc.for_batch`,
composites forward it, and the unified write doors refuse an unmarked loc
(under SGLANG_ENABLE_ASYNC_ASSERT).

    python -m pytest test/registered/unit/mem_cache/test_physical_write_loc.py -v
"""

import ast
import pathlib
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.memory_pool import (
    KVWriteLoc,
    MHATokenToKVPool,
    write_loc_is_physical,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    MHASubPoolSpec,
    UnifiedKVPool,
    UnifiedMHATokenToKVPool,
    init_unified_swa_pools,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

# `set_kv_buffer` dispatches on the platform, so the pools that get written live
# on the platform's device; the marker logic itself is device-free.
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_L, _H, _D = 2, 2, 8
_DTYPE = torch.float16


def _plain_translator():
    return KVIndexTranslator(
        req_to_token=torch.zeros((1, 4), dtype=torch.int64),
        token_to_kv_pool_allocator=SimpleNamespace(),
        token_to_kv_pool=SimpleNamespace(),
        page_size=1,
        device="cpu",
    )


def _translating_translator(v2p):
    src = _plain_translator()
    src.is_translating = True
    src._translate_write_full = lambda t, out=None: v2p[t.to(torch.int64)]
    return src


def _batch(loc):
    return SimpleNamespace(out_cache_loc=loc, out_cache_loc_is_physical=False)


def _unified_mha_pool(ps=1):
    full = MHASubPoolSpec(
        name="full",
        layer_num=_L,
        head_num=_H,
        head_dim=_D,
        store_dtype=_DTYPE,
        grow_direction="down",
    )
    swa = MHASubPoolSpec(
        name="swa",
        layer_num=_L,
        head_num=_H,
        head_dim=_D,
        store_dtype=_DTYPE,
        grow_direction="up",
    )
    kv = UnifiedKVPool(
        total_bytes=full.entry_bytes() * 32 + swa.entry_bytes() * 16,
        sub_pool_specs=[full, swa],
        device=_DEV,
        enable_memory_saver=False,
        page_size=ps,
    )
    return UnifiedMHATokenToKVPool(
        unified_buffer=kv, sub_pool_name="full", page_size=ps, enable_alt_stream=False
    )


class TestRebindMarksTheBatch(unittest.TestCase):
    def test_non_translating_pool_marks_physical_without_rebinding(self):
        loc = torch.tensor([3, 5], dtype=torch.int64)
        fb = _batch(loc)
        _plain_translator().rebind_write_loc(fb)
        self.assertIs(fb.out_cache_loc, loc)  # physical by allocation: untouched
        self.assertTrue(fb.out_cache_loc_is_physical)

    def test_translating_pool_rebinds_and_marks_physical(self):
        v2p = torch.tensor([7, 6, 5, 4], dtype=torch.int64)
        loc = torch.tensor([1, 2], dtype=torch.int64)
        fb = _batch(loc)
        _translating_translator(v2p).rebind_write_loc(fb)
        self.assertTrue(torch.equal(fb.out_cache_loc, v2p[loc]))
        self.assertTrue(fb.out_cache_loc_is_physical)

    def test_no_loc_stays_unmarked(self):
        fb = _batch(None)
        _translating_translator(torch.arange(4)).rebind_write_loc(fb)
        self.assertFalse(fb.out_cache_loc_is_physical)


class TestKVWriteLocCarriesTheMark(unittest.TestCase):
    def test_for_batch_copies_the_batch_mark_and_default_loc(self):
        fb = _batch(torch.tensor([1, 2]))
        self.assertFalse(KVWriteLoc.for_batch(fb).physical)
        fb.out_cache_loc_is_physical = True
        info = KVWriteLoc.for_batch(fb, swa_loc=torch.tensor([9, 9]))
        self.assertIs(info.loc, fb.out_cache_loc)
        self.assertTrue(info.physical)
        sliced = KVWriteLoc.for_batch(fb, fb.out_cache_loc[:1])
        self.assertTrue(sliced.physical)

    def test_bare_and_default_are_not_physical(self):
        self.assertFalse(write_loc_is_physical(torch.tensor([1])))
        self.assertFalse(write_loc_is_physical(KVWriteLoc(torch.tensor([1]))))
        self.assertTrue(
            write_loc_is_physical(KVWriteLoc(torch.tensor([1]), physical=True))
        )


class TestUnifiedDoorsRefuseUnmarkedLocs(unittest.TestCase):
    def _kv(self, n=2):
        k = torch.ones((n, _H, _D), dtype=_DTYPE, device=_DEV)
        return k, k * 2

    def test_unified_mha_door(self):
        pool = _unified_mha_pool()
        layer = SimpleNamespace(layer_id=0)
        loc = torch.tensor([3, 4], dtype=torch.int64, device=_DEV)
        k, v = self._kv()
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True):
            with self.assertRaisesRegex(AssertionError, "not marked physical"):
                pool.set_kv_buffer(layer, loc, k, v)
            with self.assertRaisesRegex(AssertionError, "not marked physical"):
                pool.set_kv_buffer(layer, KVWriteLoc(loc), k, v)
            pool.set_kv_buffer(layer, KVWriteLoc(loc, physical=True), k, v)
        self.assertTrue(torch.all(pool.k_buffer[0][3] == 1))
        self.assertTrue(torch.all(pool.v_buffer[0][4] == 2))

    def test_check_is_gated_by_the_async_assert_env(self):
        pool = _unified_mha_pool()
        loc = torch.tensor([3], dtype=torch.int64, device=_DEV)
        k, v = self._kv(1)
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(False):
            pool.set_kv_buffer(SimpleNamespace(layer_id=1), loc, k, v)
        self.assertTrue(torch.all(pool.k_buffer[1][3] == 1))

    def test_plain_pool_ignores_the_marker(self):
        pool = MHATokenToKVPool(
            size=8,
            page_size=1,
            head_num=_H,
            head_dim=_D,
            dtype=_DTYPE,
            layer_num=1,
            device=_DEV,
            enable_memory_saver=False,
        )
        loc = torch.tensor([2], dtype=torch.int64, device=_DEV)
        k, v = self._kv(1)
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True):
            pool.set_kv_buffer(SimpleNamespace(layer_id=0), loc, k, v)
        self.assertTrue(torch.all(pool.k_buffer[0][2] == 1))

    def test_swa_composite_forwards_the_space(self):
        b = init_unified_swa_pools(
            device=_DEV,
            kv_cache_dtype=_DTYPE,
            head_num=_H,
            head_dim=_D,
            v_head_dim=_D,
            swa_head_num=_H,
            swa_head_dim=_D,
            swa_v_head_dim=_D,
            page_size=1,
            start_layer=0,
            end_layer=4,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            full_max_total_num_tokens=64,
            swa_max_total_num_tokens=32,
            enable_memory_saver=False,
            need_sort=False,
        )
        pool = b.token_to_kv_pool
        loc = torch.tensor([2, 3], dtype=torch.int64, device=_DEV)
        k, v = self._kv()
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True):
            for layer_id in (0, 1):  # a full layer and a swa layer
                layer = SimpleNamespace(layer_id=layer_id)
                with self.assertRaisesRegex(AssertionError, "not marked physical"):
                    pool.set_kv_buffer(layer, KVWriteLoc(loc, loc), k, v)
                pool.set_kv_buffer(layer, KVWriteLoc(loc, loc, physical=True), k, v)


_PRODUCER_DIRS = ("models", "layers/cp", "speculative")


def _untagged_producers():
    """Model-layer write-door calls that declare no id space.

    A bare loc is not marked physical, so a producer that hands the pool
    `forward_batch.out_cache_loc` or a `KVWriteLoc` without `physical` trips the
    unified door's check on a write whose values are already physical.
    """
    root = pathlib.Path(__file__).resolve().parents[4] / "python/sglang/srt"
    bad, seen = [], 0
    for sub in _PRODUCER_DIRS:
        assert (root / sub).is_dir(), root / sub
        for f in sorted((root / sub).rglob("*.py")):
            seen += 1
            try:
                tree = ast.parse(f.read_text())
            except SyntaxError:
                continue
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call):
                    continue
                fn = n.func
                if isinstance(fn, ast.Attribute):
                    name = fn.attr
                elif isinstance(fn, ast.Name):
                    name = fn.id
                else:
                    name = ""
                if name in ("set_kv_buffer", "set_mla_kv_buffer") and len(n.args) >= 2:
                    if ast.unparse(n.args[1]).endswith(".out_cache_loc"):
                        bad.append(f"{f.name}:{n.lineno} bare out_cache_loc")
                if name == "KVWriteLoc" and not any(
                    k.arg == "physical" for k in n.keywords
                ):
                    bad.append(f"{f.name}:{n.lineno} KVWriteLoc without physical")
    assert seen > 100, seen
    return bad


class TestWriteLocProducersDeclarePhysical(unittest.TestCase):
    def test_every_model_layer_producer_declares_physical(self):
        self.assertEqual(_untagged_producers(), [])


if __name__ == "__main__":
    unittest.main()
