"""Nobody reaches the startup record through another object for a resolved value.

The supplied-instance census counts three spellings, all of which start from a
`server_args` parameter -- the caller chose the object, which is the contract
that makes those reads defensible. This pins the fourth: `model_runner.
server_args.field`, `self.scheduler.server_args.field`, `tokenizer_manager.
server_args.field`. A reference lifted off whatever object happened to hold the
record carries no contract at all, and it was invisible to every census, which
is how it grew to 105 reads across 36 files unnoticed.

They are gone, and this is what keeps them gone. Only fields resolution writes
are pinned: reading `model_runner.server_args.host` off the record answers with
what the caller asked for, which is what the record is for. The written set is
derived from the declaration sites rather than listed, so a field that stops
being resolution-written drops out on its own -- and a field that stops being
*declared* cannot slip out that way, because bare assignment during resolution
is refused by `server_args/test_resolution_declarations.py`.
"""

import ast
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PACKAGE = pathlib.Path(sglang.__file__).resolve().parent
_SRT = _PACKAGE / "srt"

# The pipeline and its extension points: reading the in-flight record is their
# job, and they run before anything is published.
_OWNERS = ("server_args.py", "runtime_context.py", "arg_groups/")

# Where the writers are. Resolution lives in `srt` -- nothing outside it
# declares -- so the written-field derivations scan `srt` while the *reads* are
# counted across the whole shipped package: a borrowed read answers with the
# startup default wherever it is written, and `benchmark/` ships too.
_READS_SCANNED = _PACKAGE

_DECLARERS = ("_declare", "declare_resolution", "declare_late_resolution")


def _declared_by_keyword():
    """Fields named as a keyword at a declaration site."""
    written = set()
    for path in sorted(_SRT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            raise AssertionError(f"unparsable module in the census: {path}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            else:
                continue
            if name in _DECLARERS:
                written |= {kw.arg for kw in node.keywords if kw.arg}
    return written


def _returned_field_names(function):
    """Field names a provider/pass writes: the keys of the mapping it returns.

    Only the returned mapping counts -- walking every `ast.Dict` in the body
    also collects a dict-valued field's *nested* keys and any unrelated local
    mapping, and those stray names would reject valid borrowed reads of fields
    resolution never writes. The mapping is traced through four spellings: a
    returned literal, assignments (annotated or not) to a returned name, a
    literal-key subscript write on it, and `.update(field=...)` on it. A
    spelling this cannot see raises instead of skipping.
    """
    names = set()
    returned = set()

    def top_level_keys(mapping):
        for key in mapping.keys:
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                raise AssertionError(f"non-literal key in {function.name}")
            names.add(key.value)

    for node in ast.walk(function):
        if isinstance(node, ast.Return) and node.value is not None:
            value = node.value
            if isinstance(value, ast.Dict):
                top_level_keys(value)
            elif isinstance(value, ast.Name):
                returned.add(value.id)
            elif isinstance(value, ast.Constant) and value.value is None:
                pass
            else:
                raise AssertionError(
                    f"opaque return in {function.name}: {ast.unparse(value)}"
                )
    for node in ast.walk(function):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in returned
                    and isinstance(node.value, ast.Dict)
                ):
                    top_level_keys(node.value)
                if isinstance(target, ast.Subscript) and (
                    isinstance(target.value, ast.Name) and target.value.id in returned
                ):
                    if not isinstance(target.slice, ast.Constant):
                        raise AssertionError(f"non-literal key in {function.name}")
                    names.add(target.slice.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in returned
        ):
            names |= {kw.arg for kw in node.keywords if kw.arg}
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and receiver.id == "overrides":
                # Positional dict literals are collected by the Dict walk;
                # anything else is invisible.
                for arg in node.args:
                    if not isinstance(arg, ast.Dict):
                        raise AssertionError(
                            f"opaque overrides.update() argument in {function.name}"
                        )
                if any(kw.arg is None for kw in node.keywords):
                    raise AssertionError(
                        f"**kwargs overrides.update() in {function.name}"
                    )
    return names


def _declared_by_registry_and_passes():
    """Fields the model-override registry and the post-process passes write.

    These field names are *data* -- dict keys, not keywords -- so a keyword
    scan misses every one of them. The callables are collected from the live
    registries rather than by matching decorator names: 26 of the 27 providers
    register through a `_register_for(...)` helper, so a scan for
    `@register_model_override*` sees exactly one of them and reports a healthy
    census over a channel it cannot see.
    """
    from sglang.srt.arg_groups import overrides

    tree = ast.parse((_SRT / "arg_groups/overrides.py").read_text(encoding="utf-8-sig"))
    bodies = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    callables = {fn for fns in overrides._MODEL_OVERRIDE_FNS.values() for fn in fns}
    callables |= {
        fn for _predicate, fn in getattr(overrides, "_PREDICATE_OVERRIDE_FNS", ())
    }
    callables |= set(overrides.POST_PROCESS_PASSES)

    fields = set()
    for fn in callables:
        body = bodies.get(getattr(fn, "__name__", ""))
        if body is not None:
            fields |= _returned_field_names(body)
    # The literal arch -> {field: value} table, which has no callable at all.
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        if target != "MODEL_OVERRIDES" or node.value is None:
            continue
        for inner in ast.walk(node.value):
            if not isinstance(inner, ast.Dict):
                continue
            for key, value in zip(inner.keys, inner.values):
                if isinstance(value, ast.Dict):
                    continue
                if not isinstance(key, ast.Constant):
                    raise AssertionError("non-literal override key")
                fields.add(key.value)
    return fields


def _declared_by_late_resolution():
    """Keywords of `self._late_resolution(...)`, the fourth declarer spelling.

    It forwards `**fields` to `declare_late_resolution`, so the keywords sit at
    its call sites and a scan for the declarer's own name finds none of them.
    """
    tree = ast.parse((_SRT / "server_args.py").read_text(encoding="utf-8-sig"))
    fields = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_late_resolution"
        ):
            fields |= {keyword.arg for keyword in node.keywords if keyword.arg}
    return fields


def _written_after_publish():
    """Fields the runtime overrides once the bags exist.

    Imported from the supplied-instance ratchet rather than re-derived: it
    already enumerates `get_context().override(...)` and its named wrapper, and
    a second derivation of the same channel is what drifts narrower. A borrowed
    read of one of these answers with the startup value the same way a
    resolution-written one does -- the write just lands later.
    """
    import importlib.util

    companion = (
        pathlib.Path(__file__).resolve().parent
        / "test_supplied_instance_exposure_ratchet.py"
    )
    spec = importlib.util.spec_from_file_location("_exposure_for_ratchet", companion)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.TestSuppliedInstanceExposure._override_written_fields())


def _resolution_written():
    """Every field the startup record answers wrong once it stays raw."""
    return (
        _declared_by_keyword()
        | _declared_by_registry_and_passes()
        | _declared_by_late_resolution()
        | _written_after_publish()
    )


def _borrowed_parking_spans(tree):
    """[(span, parked attribute names)] for classes that park a borrowed record.

    The supplied-instance census counts `self.server_args.<field>` only where
    the class was handed the record as a parameter -- `self.server_args =
    server_args` inside a method that takes one. A class that borrows it off
    another object instead (`self.server_args = scheduler.server_args`) is
    covered by neither census, and a read through that attribute is exactly the
    borrowed-record chain read this file is about. The parked name is whatever
    the class chose -- `self.args = scheduler.server_args` hides the same read,
    so the assignment target is recorded, not assumed.
    """
    spans = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        parked_names = set()
        for node in ast.walk(cls):
            # Both assignment spellings -- `self.args = x.server_args` and the
            # annotated `self.args: ServerArgs = x.server_args`.
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
            else:
                continue
            # A bare name on the right is the parameter the companion census
            # follows; an attribute chain ending in `.server_args` is a record
            # taken off another object.
            if not (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "server_args"
            ):
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    parked_names.add(target.attr)
        if parked_names:
            spans.append(((cls.lineno, cls.end_lineno), parked_names))
    return spans


def _subtrees_with_their_own_record():
    """Top-level package directories that define a second `ServerArgs`.

    `multimodal_gen` ships one, so `x.server_args.<field>` inside it names a
    field of *that* class -- and `model_path` is a field of both. Keying on the
    spelling alone once put a resolution call into a diffusion entry point.
    Derived from the class definitions rather than named here, so a third
    record would be excluded the same way instead of silently counting.
    """
    roots = set()
    for path in _PACKAGE.rglob("*.py"):
        rel = path.relative_to(_PACKAGE).as_posix()
        if rel.startswith("srt/") or "/" not in rel:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        if any(
            isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
            for node in ast.walk(tree)
        ):
            roots.add(rel.split("/")[0])
    return roots


def _reads_the_startup_record(tree):
    """True when this module's `server_args` is the one `srt` resolves.

    Inside a subtree that owns another record, only a module that imports the
    startup record is talking about it.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("sglang.srt"):
                # The *original* names: `ServerArgs as SrtServerArgs` is still
                # the startup record, whatever this module calls it.
                names = {alias.name for alias in node.names}
                if names & {"ServerArgs", "server_args", "prepare_server_args"}:
                    return True
    return False


def _chain_reads(written):
    """`<expression>.server_args.<field>` where the field is resolution-written."""
    found = []
    other_records = _subtrees_with_their_own_record()
    for path in sorted(_READS_SCANNED.rglob("*.py")):
        rel = path.relative_to(_READS_SCANNED).as_posix()
        in_srt = rel.startswith("srt/")
        if in_srt:
            under_srt = rel[len("srt/") :]
            if path.name in _OWNERS or under_srt.startswith(_OWNERS[-1]):
                continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            raise AssertionError(f"unparsable module in the census: {rel}")
        if (
            not in_srt
            and rel.split("/")[0] in other_records
            and not _reads_the_startup_record(tree)
        ):
            continue
        parked = _borrowed_parking_spans(tree)

        def parked_alias(lineno, name):
            return any(
                start <= lineno <= end and name in names
                for (start, end), names in parked
            )

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and node.attr in written
            ):
                continue
            base = node.value
            if not isinstance(base, ast.Attribute):
                continue
            through_self = isinstance(base.value, ast.Name) and base.value.id == "self"
            if base.attr == "server_args":
                # `self.server_args.field` is the parked spelling the
                # supplied-instance census counts -- but only where the record
                # arrived as a parameter.
                if through_self and not parked_alias(node.lineno, base.attr):
                    continue
            elif not (through_self and parked_alias(node.lineno, base.attr)):
                # Any other attribute counts only as a recorded parked alias
                # (`self.args = scheduler.server_args` and later `self.args.x`).
                continue
            suffix = " (parked borrowed record)" if through_self else ""
            found.append(f"{rel}:{node.lineno} {ast.unparse(base)}.{node.attr}{suffix}")
        # `getattr(model_runner.server_args, "field", default)` reads the same
        # borrowed record through a `Call`, with the same stale default.
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in written
            ):
                continue
            base = node.args[0]
            if not isinstance(base, ast.Attribute):
                continue
            through_self = isinstance(base.value, ast.Name) and base.value.id == "self"
            if base.attr == "server_args":
                if through_self and not parked_alias(node.lineno, base.attr):
                    continue
            elif not (through_self and parked_alias(node.lineno, base.attr)):
                continue
            found.append(
                f"{rel}:{node.lineno} getattr({ast.unparse(base)}, "
                f"{node.args[1].value!r})"
            )
    return sorted(found)


class TestNoChainReadsOfResolvedConfig(CustomTestCase):
    def test_the_census_has_something_to_count(self):
        """A written set that collapsed would make the pin vacuous.

        Each mechanism is checked on its own, because they fail
        independently: the keyword scan cannot see a field name that is data,
        and a scan for `@register_model_override*` sees one provider out of
        twenty-seven because the rest register through a helper. A hand-written
        expectation of the resulting field names is what hid that -- it stayed
        green while a whole channel went unscanned -- so each mechanism is
        pinned by a floor derived from the live registry instead.
        """
        from sglang.srt.arg_groups import overrides

        by_keyword = _declared_by_keyword()
        by_data = _declared_by_registry_and_passes()
        by_late = _declared_by_late_resolution()

        self.assertGreater(
            len(by_keyword),
            100,
            f"only {len(by_keyword)} fields are declared by keyword; the scan broke",
        )
        providers = {fn for fns in overrides._MODEL_OVERRIDE_FNS.values() for fn in fns}
        providers |= {
            fn for _predicate, fn in getattr(overrides, "_PREDICATE_OVERRIDE_FNS", ())
        }
        self.assertGreater(
            len(providers) + len(overrides.POST_PROCESS_PASSES),
            50,
            "the registry and pass tables collapsed; the data-channel scan is "
            "reading an empty registry",
        )
        self.assertGreater(
            len(by_data),
            25,
            f"only {len(by_data)} fields come from the registry and the passes, "
            f"across {len(providers)} providers and "
            f"{len(overrides.POST_PROCESS_PASSES)} passes; the scan of the "
            "dict-key channel broke",
        )
        self.assertGreaterEqual(
            len(by_late),
            3,
            f"only {len(by_late)} fields are declared late; the "
            "`_late_resolution` keyword scan broke",
        )
        # The three mechanisms are not the same set: if any became a subset of
        # the keyword scan, that scan would be doing all the work and a
        # regression in the others would be invisible.
        self.assertTrue(by_data - by_keyword, "the data channel adds nothing")
        self.assertTrue(by_late - by_keyword, "late resolution adds nothing")

    def test_nothing_reads_a_resolved_field_off_a_borrowed_record(self):
        found = _chain_reads(_resolution_written())
        self.assertEqual(
            found,
            [],
            "these reach the startup record through another object for a value "
            "resolution decides, so they answer with the CLI default once the "
            "record stays raw; read the config bag instead:\n  " + "\n  ".join(found),
        )


if __name__ == "__main__":
    unittest.main()
