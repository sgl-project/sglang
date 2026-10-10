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
bind from a translated loc by looking at it. A plan's `bind` marks the
batch's loc physical, producers carry the mark in `KVWriteLoc`, composites
forward it, and the unified write doors refuse a loc without it.

    python -m pytest test/registered/unit/mem_cache/test_physical_write_loc.py -v
"""

import ast
import pathlib
import unittest
from collections import defaultdict
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.mem_cache.kv_loc_plan import IdSpace, IdSpaceKind
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

register_cpu_ci(est_time=14, suite="base-a-test-cpu")

# `set_kv_buffer` dispatches on the platform, so the pools that get written live
# on the platform's device; the mark itself is device-free.
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
    src._spaces = {
        IdSpaceKind.FULL: IdSpace(
            key=(IdSpaceKind.FULL, "test"), write=lambda t: v2p[t.to(torch.int64)]
        )
    }
    return src


def _batch(loc, physical=False):
    return SimpleNamespace(
        out_cache_loc=loc,
        out_cache_loc_is_physical=physical,
        encoder_out_cache_loc=torch.tensor([8, 9]),
    )


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
        _plain_translator().bind_own_plan(fb)
        self.assertIs(fb.out_cache_loc, loc)  # physical by allocation: untouched
        self.assertTrue(fb.out_cache_loc_is_physical)

    def test_translating_pool_rebinds_and_marks_physical(self):
        v2p = torch.tensor([7, 6, 5, 4], dtype=torch.int64)
        loc = torch.tensor([1, 2], dtype=torch.int64)
        fb = _batch(loc)
        _translating_translator(v2p).bind_own_plan(fb)
        self.assertTrue(torch.equal(fb.out_cache_loc, v2p[loc]))
        self.assertTrue(fb.out_cache_loc_is_physical)

    def test_no_loc_stays_unmarked(self):
        fb = _batch(None)
        _translating_translator(torch.arange(4)).bind_own_plan(fb)
        self.assertFalse(fb.out_cache_loc_is_physical)


class TestKVWriteLocCarriesTheMark(unittest.TestCase):
    def test_for_batch_wraps_out_cache_loc_with_the_batch_mark(self):
        fb = _batch(torch.tensor([1, 2]))
        self.assertFalse(KVWriteLoc.for_batch(fb).physical)
        fb.out_cache_loc_is_physical = True
        info = KVWriteLoc.for_batch(fb, swa_loc=torch.tensor([9, 9]))
        self.assertIs(info.loc, fb.out_cache_loc)
        self.assertTrue(info.physical)

    def test_for_layer_leaves_the_encoder_loc_unmarked(self):
        fb = _batch(torch.tensor([1, 2]), physical=True)
        decoder = KVWriteLoc.for_layer(fb, SimpleNamespace(is_cross_attention=False))
        self.assertIs(decoder.loc, fb.out_cache_loc)
        self.assertTrue(decoder.physical)
        cross = KVWriteLoc.for_layer(fb, SimpleNamespace(is_cross_attention=True))
        self.assertIs(cross.loc, fb.encoder_out_cache_loc)
        self.assertFalse(cross.physical)

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
        with self.assertRaisesRegex(ValueError, "not marked physical"):
            pool.set_kv_buffer(layer, loc, k, v)
        with self.assertRaisesRegex(ValueError, "not marked physical"):
            pool.set_kv_buffer(layer, KVWriteLoc(loc), k, v)
        pool.set_kv_buffer(layer, KVWriteLoc(loc, physical=True), k, v)
        self.assertTrue(torch.all(pool.k_buffer[0][3] == 1))
        self.assertTrue(torch.all(pool.v_buffer[0][4] == 2))

    def test_plain_pool_ignores_the_mark(self):
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
        pool.set_kv_buffer(SimpleNamespace(layer_id=0), loc, k, v)
        self.assertTrue(torch.all(pool.k_buffer[0][2] == 1))

    def test_swa_composite_forwards_the_mark(self):
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
        for layer_id in (0, 1):  # a full layer and a swa layer
            layer = SimpleNamespace(layer_id=layer_id)
            with self.assertRaisesRegex(ValueError, "not marked physical"):
                pool.set_kv_buffer(layer, KVWriteLoc(loc, loc), k, v)
            pool.set_kv_buffer(layer, KVWriteLoc(loc, loc, physical=True), k, v)


# --------------------------------------------------------------------------
# Every producer that can reach a unified pool states the mark.
# --------------------------------------------------------------------------

_SRT = pathlib.Path(__file__).resolve().parents[4] / "python/sglang/srt"
_WRITE_DOORS = ("set_kv_buffer", "set_mla_kv_buffer")
_MARKED_CONSTRUCTORS = ("for_batch", "for_layer")
# The attention backends --enable-unified-memory admits, by resolved name.
_UNIFIED_BACKEND_FILES = {
    "triton": ("layers/attention/triton_backend.py",),
    "fa3": ("layers/attention/flashattention_backend.py",),
    "fa4": ("layers/attention/flashattention_backend.py",),
    "flashinfer": (
        "layers/attention/flashinfer_backend.py",
        "layers/attention/flashinfer_mla_backend.py",
    ),
    "trtllm_mha": ("layers/attention/trtllm_mha_backend.py",),
    "trtllm_mla": ("layers/attention/trtllm_mla_backend.py",),
    "cutedsl_mla": ("layers/attention/cutedsl_mla_backend.py",),
    "tokenspeed_mla": ("layers/attention/tokenspeed_mla_backend.py",),
    "flashmla": ("layers/attention/flashmla_backend.py",),
}
# Model code and the rest of the forward path also write KV directly.
_PRODUCER_DIRS = (
    "models",
    "layers/cp",
    "speculative",
    "model_executor",
    "disaggregation",
)
# kv_cache_hook's attention-backend allowlists for the unified pool.
_ALLOWLIST_NAMES = ("allowed_full", "spec_allowed", "dcp_allowed")


class _WriteLocCensus:
    """Resolves the loc argument of every write-door call to how it is built.

    A loc is marked when it is `KVWriteLoc(..., physical=...)`,
    `KVWriteLoc.for_batch(...)` or `KVWriteLoc.for_layer(...)`; a local name
    whose every assignment is marked; a parameter every in-scope caller passes
    marked; or a call to a function every `return` of which is marked.
    """

    def __init__(self, sources):
        self.enclosing = {}
        self.defs = defaultdict(list)
        self.calls = defaultdict(list)
        self.doors = []
        for where, src in sources.items():
            tree = ast.parse(src)
            for fn in ast.walk(tree):
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.defs[fn.name].append(fn)
                    for node in ast.walk(fn):
                        self.enclosing[node] = fn  # innermost wins: walk is BFS
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _callee_name(node)
                self.calls[name].append(node)
                if name in _WRITE_DOORS:
                    self.doors.append((where, node))

    def unmarked(self):
        bad = []
        for where, call in self.doors:
            loc = _loc_argument(call)
            fn = self.enclosing.get(call)
            if loc is None or not self._marked(loc, fn, set()):
                bad.append(f"{where}:{call.lineno}")
        return bad

    def _marked(self, expr, fn, seen) -> bool:
        if isinstance(expr, ast.Call):
            f = expr.func
            if _is_name(f, "KVWriteLoc"):
                return any(k.arg == "physical" for k in expr.keywords)
            if isinstance(f, ast.Attribute) and _is_name(f.value, "KVWriteLoc"):
                return f.attr in _MARKED_CONSTRUCTORS
            return self._returns_marked(_callee_name(expr), seen)
        if isinstance(expr, ast.Name) and fn is not None:
            return self._name_marked(expr.id, fn, seen)
        return False

    def _returns_marked(self, name, seen) -> bool:
        if not name or ("def", name) in seen or not self.defs[name]:
            return False
        seen = seen | {("def", name)}
        for d in self.defs[name]:
            returns = [
                r.value
                for r in ast.walk(d)
                if isinstance(r, ast.Return) and self.enclosing.get(r) is d
            ]
            if not returns or not all(
                r is not None and self._marked(r, d, seen) for r in returns
            ):
                return False
        return True

    def _name_marked(self, name, fn, seen) -> bool:
        key = ("name", id(fn), name)
        if key in seen:
            return True  # a cycle adds no new value
        seen = seen | {key}
        values = [
            node.value
            for node in ast.walk(fn)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and self.enclosing.get(node) is fn
            and node.value is not None
            and any(
                _is_name(t, name)
                for t in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
        ]
        params = [a.arg for a in fn.args.posonlyargs + fn.args.args]
        is_param = name in params or name in [a.arg for a in fn.args.kwonlyargs]
        if not values and not is_param:
            return False
        if not all(self._marked(v, fn, seen) for v in values):
            return False
        return not is_param or self._callers_mark(fn, name, params, seen)

    def _callers_mark(self, fn, name, params, seen) -> bool:
        callers = self.calls[fn.name]
        if not callers:
            return False
        bound = params[1:] if params[:1] in (["self"], ["cls"]) else params
        for call in callers:
            arg = next((k.value for k in call.keywords if k.arg == name), None)
            if arg is None and name in bound and bound.index(name) < len(call.args):
                arg = call.args[bound.index(name)]
            if arg is None or not self._marked(arg, self.enclosing.get(call), seen):
                return False
        return True


def _is_name(node, name) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _callee_name(call) -> str:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    return f.id if isinstance(f, ast.Name) else ""


def _loc_argument(call):
    for k in call.keywords:
        if k.arg in ("loc_info", "loc"):
            return k.value
    return call.args[1] if len(call.args) >= 2 else None


def _unified_producer_files():
    files = {_SRT / f for group in _UNIFIED_BACKEND_FILES.values() for f in group}
    for sub in _PRODUCER_DIRS:
        assert (_SRT / sub).is_dir(), _SRT / sub
        files.update((_SRT / sub).rglob("*.py"))
    return sorted(files)


def _unified_backend_names():
    """Every attention backend the unified-memory allowlists name."""
    tree = ast.parse((_SRT / "arg_groups/kv_cache_hook.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(_is_name(t, n) for t in node.targets for n in _ALLOWLIST_NAMES)
            and isinstance(node.value, ast.Set)
        ):
            names.update(
                e.value for e in node.value.elts if isinstance(e, ast.Constant)
            )
    return names


class TestEveryUnifiedProducerMarksItsLoc(unittest.TestCase):
    def test_the_census_covers_every_allowlisted_backend(self):
        names = _unified_backend_names()
        self.assertIn("fa3", names)  # the parse found the allowlists
        self.assertLessEqual(names, set(_UNIFIED_BACKEND_FILES))
        for group in _UNIFIED_BACKEND_FILES.values():
            for f in group:
                self.assertTrue((_SRT / f).is_file(), f)

    def test_every_write_door_call_passes_a_marked_loc(self):
        files = _unified_producer_files()
        census = _WriteLocCensus(
            {str(f.relative_to(_SRT)): f.read_text() for f in files}
        )
        self.assertGreater(len(census.doors), 20)  # the census saw the doors
        self.assertEqual(census.unmarked(), [])

    def test_the_census_tells_marked_from_unmarked(self):
        src = """
def good(self, pool, fb, layer):
    pool.set_kv_buffer(layer, KVWriteLoc.for_batch(fb), 1, 2)
    loc = KVWriteLoc(fb.out_cache_loc // 2, physical=True)
    pool.set_mla_kv_buffer(layer, loc, 1, 2)
    pool.set_kv_buffer(layer, self._helper(fb), 1, 2)
    self._write(fb, layer, KVWriteLoc.for_layer(fb, layer))

def _helper(self, fb):
    return KVWriteLoc.for_batch(fb)

def _write(self, fb, layer, loc_info):
    pool.set_kv_buffer(layer, loc_info, 1, 2)

def bad(self, pool, fb, layer):
    pool.set_kv_buffer(layer, fb.out_cache_loc, 1, 2)
    cache_loc = fb.out_cache_loc
    pool.set_mla_kv_buffer(layer, cache_loc, 1, 2)
    pool.set_kv_buffer(layer, KVWriteLoc(cache_loc), 1, 2)
"""
        census = _WriteLocCensus({"m.py": src})
        self.assertEqual(census.unmarked(), ["m.py:16", "m.py:18", "m.py:19"])


def _only_raises(fn) -> bool:
    body = [
        s
        for s in fn.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]
    return bool(body) and all(isinstance(s, ast.Raise) for s in body)


def _door_takes_a_write_loc(fn) -> bool:
    """A pool's write door unwraps its loc argument, forwards it untouched to
    another write door, takes only `*args`, or only raises."""
    params = [a.arg for a in fn.args.posonlyargs + fn.args.args]
    if _only_raises(fn):
        return True
    if len(params) < 3:  # (self, layer, loc_info, ...)
        return fn.args.vararg is not None
    loc = params[2]
    reassigned = any(
        _is_name(t, loc)
        for node in ast.walk(fn)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
    )
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = _callee_name(node)
        if name == "unwrap_write_loc" and node.args and _is_name(node.args[0], loc):
            return True
        if name in _WRITE_DOORS and not reassigned:
            if _is_name(_loc_argument(node), loc):
                return True
    return False


def _pool_doors_taking_a_bare_loc(sources):
    doors, bad = 0, []
    for where, src in sources.items():
        for cls in ast.walk(ast.parse(src)):
            if not isinstance(cls, ast.ClassDef):
                continue
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name in _WRITE_DOORS:
                    doors += 1
                    if not _door_takes_a_write_loc(fn):
                        bad.append(f"{where}:{cls.name}.{fn.name}")
    return doors, bad


class TestEveryPoolDoorTakesAWriteLoc(unittest.TestCase):
    """Producers now hand every pool a `KVWriteLoc`, static pools included, so
    an override that still indexes with its loc argument breaks on the first
    write."""

    def test_every_pool_write_door_unwraps_or_forwards_its_loc(self):
        doors, bad = _pool_doors_taking_a_bare_loc(
            {
                str(f.relative_to(_SRT)): f.read_text()
                for f in sorted(_SRT.rglob("*.py"))
            }
        )
        self.assertGreater(doors, 20)  # the census saw the doors
        self.assertEqual(bad, [])

    def test_the_census_tells_unwrapping_from_bare(self):
        src = """
class Good(KVCache):
    def set_kv_buffer(self, layer, loc_info, k, v):
        loc, _, _ = unwrap_write_loc(loc_info)
        self.buf[loc] = k

    def set_mla_kv_buffer(self, layer, loc_info, k, v):
        self.inner.set_mla_kv_buffer(layer, loc_info, k, v)


class PassThrough(KVCache):
    def set_kv_buffer(self, *args, **kwargs):
        self.inner.set_kv_buffer(*args, **kwargs)


class Bare(KVCache):
    def set_kv_buffer(self, layer, loc, k, v):
        self.buf[loc] = k

    def set_mla_kv_buffer(self, layer, loc, k, v):
        loc = self.translate(loc)
        super().set_mla_kv_buffer(layer, loc, k, v)
"""
        doors, bad = _pool_doors_taking_a_bare_loc({"m.py": src})
        self.assertEqual(doors, 5)
        self.assertEqual(
            bad, ["m.py:Bare.set_kv_buffer", "m.py:Bare.set_mla_kv_buffer"]
        )


if __name__ == "__main__":
    unittest.main()
