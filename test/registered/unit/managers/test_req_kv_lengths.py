"""ReqKvInfo owns its lengths: code outside the class changes them only
through its named methods."""

import ast
import dataclasses
import tempfile
import unittest
import warnings
from pathlib import Path

from sglang.srt.managers import schedule_batch
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SRT_DIR = Path(schedule_batch.__file__).resolve().parents[1]

_LENGTH_FIELDS = {
    "cache_protected_len",
    "cache_inserted_len",
    "kv_committed_len",
    "kv_allocated_len",
    "swa_evict_floor",
    "component_evicted_seqlens",
}
# A mutable field: any use outside ReqKvInfo can change it through an alias.
_OWNED_ONLY_FIELDS = {"component_evicted_seqlens"}
_SETATTR_FUNCS = {"setattr", "delattr", "__setattr__", "__delattr__"}
# Calls whose keyword arguments set fields: the constructor and dataclasses.replace.
_FIELD_KWARG_FUNCS = {"ReqKvInfo", "replace"}

# (file, scope, field) sites outside ReqKvInfo that set a length. Only shrinks.
_ALLOWED_WRITERS = {
    # A Req always builds its ReqKvInfo, so these `req.kv is None` branches
    # look unreachable; they build a fresh record holding the matched prefix.
    (
        "mem_cache/unified_cache/components/swa.py",
        "SWAComponent.update_external_linker_load",
        "kv_allocated_len",
    ),
    (
        "mem_cache/unified_cache/unified_cache_linker.py",
        "UnifiedCacheLinkerWrapper.load_back",
        "kv_allocated_len",
    ),
}


def _func_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_own_field(node, own_fields):
    """`self.field` where the enclosing class declares a field of that name."""
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in own_fields
    )


def _target_fields(target, own_fields):
    """Fields a store or delete target names: `x.field`, `x.field[k]`, and
    `x.__dict__["field"]` / `vars(x)["field"]`."""
    for sub in ast.walk(target):
        if isinstance(sub, ast.Attribute) and sub.attr in _LENGTH_FIELDS:
            if not _is_own_field(sub, own_fields):
                yield sub.attr
        elif (
            isinstance(sub, ast.Subscript)
            and isinstance(sub.slice, ast.Constant)
            and sub.slice.value in _LENGTH_FIELDS
        ):
            yield sub.slice.value


def _writes(node, own_fields=frozenset()):
    """The length fields a node writes, or could write, from outside ReqKvInfo.
    `own_fields` are the enclosing class's own fields of the same names."""
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
        if isinstance(node, (ast.Assign, ast.Delete)):
            targets = node.targets
        else:
            targets = [node.target]
        for target in targets:
            yield from _target_fields(target, own_fields)
    elif isinstance(node, ast.Call):
        name = _func_name(node.func)
        if name in _SETATTR_FUNCS:
            for arg in node.args[:2]:
                if isinstance(arg, ast.Constant) and arg.value in _LENGTH_FIELDS:
                    yield arg.value
        elif name in _FIELD_KWARG_FUNCS:
            for keyword in node.keywords:
                if keyword.arg in _LENGTH_FIELDS:
                    yield keyword.arg
    elif (
        isinstance(node, ast.Attribute)
        and node.attr in _OWNED_ONLY_FIELDS
        and isinstance(node.ctx, ast.Load)
        and not _is_own_field(node, own_fields)
    ):
        yield node.attr


def _class_fields(node):
    """The length-field names a class body declares as its own fields."""
    fields = set()
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            fields.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
    return frozenset(fields & _LENGTH_FIELDS)


def _scan(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    found = []

    def visit(node, scope, own_fields):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = f"{scope}.{node.name}" if scope else node.name
        if isinstance(node, ast.ClassDef):
            own_fields = _class_fields(node)
        if scope.split(".")[0] == "ReqKvInfo":
            return
        for field in _writes(node, own_fields):
            found.append((scope, field, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, scope, own_fields)

    visit(tree, "", frozenset())
    return found


class TestReqKvLengthWriters(CustomTestCase):
    def test_only_req_kv_info_methods_assign_lengths(self):
        found = []
        allowed_seen = set()
        for path in sorted(_SRT_DIR.rglob("*.py")):
            rel = path.relative_to(_SRT_DIR).as_posix()
            for scope, field, line in _scan(path):
                if (rel, scope, field) in _ALLOWED_WRITERS:
                    allowed_seen.add((rel, scope, field))
                    continue
                found.append(f"{rel}:{line} {scope} writes {field}")
        self.assertFalse(
            found,
            "Write these lengths through a ReqKvInfo method:\n  " + "\n  ".join(found),
        )
        self.assertEqual(
            allowed_seen,
            _ALLOWED_WRITERS,
            "Drop the allowlist entries that no longer write a length",
        )

    def test_scan_catches_indirect_writes(self):
        cases = {
            "req.kv.kv_committed_len = 1": "kv_committed_len",
            "kv.kv_allocated_len += 2": "kv_allocated_len",
            "self.cache_protected_len = 0": "cache_protected_len",
            "a, b.swa_evict_floor = 1, 2": "swa_evict_floor",
            "setattr(kv, 'kv_committed_len', 3)": "kv_committed_len",
            "object.__setattr__(kv, 'cache_inserted_len', 3)": "cache_inserted_len",
            "delattr(kv, 'kv_allocated_len')": "kv_allocated_len",
            "kv.__dict__['kv_committed_len'] = 3": "kv_committed_len",
            "vars(kv)['kv_allocated_len'] = 3": "kv_allocated_len",
            "ReqKvInfo(kv_allocated_len=4)": "kv_allocated_len",
            "dataclasses.replace(kv, kv_committed_len=4)": "kv_committed_len",
            "del kv.component_evicted_seqlens[k]": "component_evicted_seqlens",
            "d = kv.component_evicted_seqlens": "component_evicted_seqlens",
            "f(kv.component_evicted_seqlens)": "component_evicted_seqlens",
        }
        for source, field in cases.items():
            with self.subTest(source=source):
                fields = [
                    f for node in ast.walk(ast.parse(source)) for f in _writes(node)
                ]
                self.assertIn(field, fields)

        for source in (
            "x = kv.kv_committed_len",
            "kv.set_evicted_seqlen(t, 3)",
            "s.replace('a', 'b')",
            "ReqKvInfo()",
        ):
            with self.subTest(source=source):
                fields = [
                    f for node in ast.walk(ast.parse(source)) for f in _writes(node)
                ]
                self.assertEqual(fields, [])

    def test_scan_skips_a_class_own_field_of_the_same_name(self):
        source = (
            "class Params:\n"
            "    component_evicted_seqlens: dict\n"
            "    def set(self, k, v):\n"
            "        self.component_evicted_seqlens[k] = v\n"
            "    def take(self, req):\n"
            "        return req.kv.component_evicted_seqlens\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "params.py"
            path.write_text(source)
            found = _scan(path)
        self.assertEqual(found, [("Params.take", "component_evicted_seqlens", 6)])


_LENGTHS = (
    "cache_protected_len",
    "cache_inserted_len",
    "kv_committed_len",
    "kv_allocated_len",
    "swa_evict_floor",
)


def _record(**lengths):
    kv = ReqKvInfo(**lengths)
    kv.set_evicted_seqlen(ComponentType.SWA, 6)
    return kv


def _lengths(kv):
    lengths = {name: getattr(kv, name) for name in _LENGTHS}
    lengths["swa_evicted"] = kv.get_evicted_seqlen(ComponentType.SWA)
    return lengths


class TestReqKvLengthMethods(CustomTestCase):
    def test_each_method_changes_only_its_lengths(self):
        start = dict(
            cache_protected_len=4,
            cache_inserted_len=8,
            kv_committed_len=8,
            kv_allocated_len=10,
            swa_evict_floor=2,
        )
        cases = [
            ("allocate_extend", (12,), dict(kv_committed_len=12, kv_allocated_len=12)),
            # Commits at allocation, not when the step's token is known.
            ("allocate_decode", (1,), dict(kv_committed_len=9, kv_allocated_len=11)),
            ("allocate_spec_decode", (12,), dict(kv_allocated_len=12)),
            ("allocate_spec_decode", (9,), {}),
            ("allocate_prealloc", (5,), dict(kv_allocated_len=5)),
            ("commit_prealloc", (5,), dict(kv_committed_len=5)),
            ("commit_decode", (2,), dict(kv_committed_len=10)),
            ("commit_finished", (9,), dict(kv_committed_len=9)),
            ("commit_finished", (12,), dict(kv_committed_len=10)),
            ("truncate_committed", (7,), dict(kv_committed_len=7)),
            ("truncate_committed", (9,), {}),
            (
                "truncate",
                (5,),
                dict(kv_committed_len=5, kv_allocated_len=5, swa_evicted=5),
            ),
            ("truncate", (12,), {}),
            (
                "free_tail",
                (5,),
                dict(kv_committed_len=5, kv_allocated_len=5, swa_evicted=5),
            ),
            # Unlike truncate, allocated becomes end even when below it.
            ("free_tail", (12,), dict(kv_allocated_len=12)),
            ("rewind", (5,), dict(kv_committed_len=5, kv_allocated_len=5)),
            ("protect_prefix", (6,), dict(cache_protected_len=6)),
            ("record_insert", (9,), dict(cache_inserted_len=9)),
            ("record_insert", (6,), {}),
            (
                "clear_tree_prefix",
                (),
                dict(cache_protected_len=0, cache_inserted_len=0),
            ),
            ("shield_swa_prefill", (9,), dict(swa_evict_floor=9)),
        ]
        for method, args, changed in cases:
            with self.subTest(method=method, args=args):
                kv = _record(**start)
                expected = {**_lengths(kv), **changed}
                getattr(kv, method)(*args)
                self.assertEqual(_lengths(kv), expected)

    def test_reset_for_retract_resets_only_its_subset(self):
        kv = ReqKvInfo(
            cache_protected_len=4,
            cache_inserted_len=8,
            kv_committed_len=8,
            kv_allocated_len=10,
            swa_evict_floor=2,
            retraction_backup=object(),
            restore_done_event=object(),
            mamba_ping_pong_empty=[True, False],
            mamba_next_track_idx=1,
            mamba_last_track_idx=0,
            mamba_last_track_seqlen=8,
            mamba_prev_track_seqlen=4,
            mamba_untracked_decode_batch_idx=3,
            mamba_needs_clear=True,
        )
        kv.set_evicted_seqlen(ComponentType.SWA, 6)
        reset = {
            "cache_protected_len",
            "cache_inserted_len",
            "kv_committed_len",
            "mamba_pool_idx",
            "mamba_ping_pong_track_buffer",
            "mamba_ping_pong_empty",
            "mamba_next_track_idx",
            "mamba_last_track_idx",
            "mamba_last_track_seqlen",
            "mamba_prev_track_seqlen",
            "mamba_untracked_decode_batch_idx",
            "mamba_cow_src_index",
            "mamba_needs_clear",
        }
        fresh = ReqKvInfo()
        before = {f.name: getattr(kv, f.name) for f in dataclasses.fields(kv)}
        kv.reset_for_retract()
        for name, value in before.items():
            expected = getattr(fresh, name) if name in reset else value
            with self.subTest(field=name):
                self.assertEqual(getattr(kv, name), expected)

        kv.req_pool_idx = 0
        with self.assertRaises(AssertionError):
            kv.reset_for_retract()


if __name__ == "__main__":
    unittest.main()
