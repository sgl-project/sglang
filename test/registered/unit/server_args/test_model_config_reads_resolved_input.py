"""`ModelConfig` is built from values resolution has already decided.

Resolution builds a `ModelConfig` partway through and keys later decisions off
it, so the pipeline reads its own output through that object. The loop is only
benign while every field `ModelConfig.from_server_args` reads has been resolved
by the time it is built -- otherwise the model configuration describes a
half-resolved input, and every handler downstream of it inherits that.

Nothing enforces the ordering today; it holds because the path and quantization
handlers happen to run early. So this derives both sides from the source -- the
fields the constructor reads, and the step each is declared at -- and pins the
one field that is deliberately read before resolution touches it.
"""

import ast
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).resolve().parent / "srt"

# Read for what the caller asked for: the constructor passes it through and
# never stores it, while resolution later overwrites the field with the value
# the architecture implies. Two quantities sharing one name.
_READ_BEFORE_RESOLUTION = frozenset({"is_embedding"})

# Declared after the first `get_model_config()`, so the cached configuration
# holds the earlier value. Nothing reads the stale copy today (its one consumer
# is on the `is_draft_model` branch, built after resolution), and fixing it
# means moving the build or the hook. Pinned so a second field in this position
# has to be looked at.
_STALE_IN_THE_MODEL_CONFIG = frozenset({"speculative_algorithm"})


def _server_args_names(tree, path):
    names = {"self"} if path.name == "server_args.py" else {"server_args"}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            annotation = arg.annotation
            if isinstance(annotation, ast.Constant):
                text = annotation.value
            elif isinstance(annotation, ast.Name):
                text = annotation.id
            elif isinstance(annotation, ast.Attribute):
                text = annotation.attr
            else:
                continue
            if text == "ServerArgs":
                names.add(arg.arg)
    return names


def _constructor_reads():
    """Fields `ModelConfig.from_server_args` takes off the record."""
    path = _SRT / "configs/model_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    constructor = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "from_server_args"
    )
    names = _server_args_names(tree, path)
    reads = {
        node.attr
        for node in ast.walk(constructor)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in names
        and isinstance(node.ctx, ast.Load)
    }
    # `getattr(server_args, "field", default)` is the normal spelling for an
    # optional input and is a `Call`, not an `Attribute`.
    for node in ast.walk(constructor):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in names
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            reads.add(node.args[1].value)
    return reads


def _late_resolution_fields():
    """Fields written through `_late_resolution` / `declare_late_resolution`.

    All of them land after the model configuration is built: the launcher's
    validation stage runs long after `__post_init__`.
    """
    fields = set()
    for name in (
        "server_args.py",
        "arg_groups/overrides.py",
        "utils/template_detection.py",
    ):
        path = _SRT / name
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if called in ("_late_resolution", "declare_late_resolution"):
                fields |= {kw.arg for kw in node.keywords if kw.arg}
    return fields


def _hook_declarations(dispatch, source_module):
    """{field: dispatcher line} for hooks the dispatch calls on other objects.

    `handle_speculative_decoding(self)` and `current_platform.
    apply_server_args_defaults(self)` are not `self.<handler>()` calls, so a
    scan of the dispatcher's own method calls never reaches their
    `declare_resolution` sites -- and the speculative hooks decide
    `speculative_algorithm`, which the model configuration reads.
    """
    imported = {}
    for node in ast.walk(ast.parse(source_module.read_text(encoding="utf-8-sig"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported[alias.asname or alias.name] = node.module

    out = {}
    for node in ast.walk(dispatch):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else (node.func.attr if isinstance(node.func, ast.Attribute) else None)
        )
        module = imported.get(name)
        if not module or not module.startswith("sglang.srt."):
            continue
        path = _SRT / (module[len("sglang.srt.") :].replace(".", "/") + ".py")
        if not path.exists():
            continue
        for inner in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "declare_resolution"
            ):
                for keyword in inner.keywords:
                    if keyword.arg:
                        out[keyword.arg] = max(out.get(keyword.arg, 0), node.lineno)
    return out


def _pipeline():
    """(ordered steps, {step: methods it reaches}) for the resolution dispatch."""
    source = (_SRT / "server_args.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    record = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
    )
    methods = {
        node.name: node for node in record.body if isinstance(node, ast.FunctionDef)
    }
    dispatch = methods["_run_resolution_pipeline"]
    steps = [
        name
        for _line, name in sorted(
            (node.lineno, node.func.attr)
            for node in ast.walk(dispatch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        )
    ]

    def reaches(name, seen=None):
        seen = seen if seen is not None else set()
        if name in seen or name not in methods:
            return seen
        seen.add(name)
        for node in ast.walk(methods[name]):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr in methods
            ):
                reaches(node.func.attr, seen)
        return seen

    step_lines = {}
    for node in ast.walk(dispatch):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            step_lines.setdefault(node.func.attr, node.lineno)
    return steps, methods, {name: reaches(name) for name in steps}, step_lines


class TestModelConfigReadsResolvedInput(CustomTestCase):
    def test_every_field_it_reads_is_resolved_before_it_is_built(self):
        steps, methods, reached, step_lines = _pipeline()
        wanted = _constructor_reads()

        first_build = None
        declared_at = {}
        for index, step in enumerate(steps):
            for method in reached[step]:
                body = methods[method]
                for node in ast.walk(body):
                    if not isinstance(node, ast.Call):
                        continue
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "get_model_config"
                        and first_build is None
                    ):
                        first_build = (index, step)
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "_declare"
                    ):
                        for keyword in node.keywords:
                            if keyword.arg in wanted:
                                # The *last* declaration is the one that has
                                # to precede the build.
                                declared_at[keyword.arg] = max(
                                    declared_at.get(keyword.arg, index), index
                                )
        self.assertIsNotNone(
            first_build, "no handler builds a ModelConfig; the scan broke"
        )

        # Hooks the dispatch calls on other objects declare too, and a hook
        # below the first build is late by definition. Both positions are read
        # *inside the dispatcher*: a handler body sits further down the file
        # than the dispatcher that calls it, so a line number taken from one
        # scope says nothing about ordering against the other.
        dispatch = methods["_run_resolution_pipeline"]
        build_line = step_lines[first_build[1]]
        for field, line in _hook_declarations(
            dispatch, _SRT / "server_args.py"
        ).items():
            if field in wanted and line > build_line:
                declared_at[field] = max(declared_at.get(field, first_build[0]), 10**6)

        # Late resolution is the other channel that can decide a field the
        # constructor reads, and it runs at the launcher's validation stage --
        # after every build. Without this the check iterates `declared_at`
        # only, so a field written only there is never even a candidate.
        for field in _late_resolution_fields():
            if field in wanted:
                declared_at[field] = 10**6

        known = _READ_BEFORE_RESOLUTION | _STALE_IN_THE_MODEL_CONFIG
        late = sorted(
            field
            for field, index in declared_at.items()
            if index >= first_build[0] and field not in known
        )
        self.assertEqual(
            late,
            [],
            "resolution decides these after it builds the ModelConfig that reads "
            f"them, so the model configuration describes a half-resolved input "
            f"(first build: step {first_build[0]}, {first_build[1]}): {late}",
        )

    def test_the_pinned_stale_field_is_still_stale(self):
        """If the ordering gets fixed, this pin has to be retired, not kept.

        A pin that outlives the defect it describes is worse than none: it
        documents a hazard that no longer exists and hides the day one appears.
        """
        steps, methods, reached, step_lines = _pipeline()
        dispatch = methods["_run_resolution_pipeline"]
        hooks = _hook_declarations(dispatch, _SRT / "server_args.py")
        build_line = min(
            step_lines[step]
            for step in steps
            for method in reached[step]
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_model_config"
                for node in ast.walk(methods[method])
            )
        )
        for field in _STALE_IN_THE_MODEL_CONFIG:
            self.assertIn(
                field,
                hooks,
                f"{field} is pinned as decided after the build, but no hook "
                "declares it any more; retire the pin",
            )
            self.assertGreater(
                hooks[field],
                build_line,
                f"{field} is now decided before the model configuration is "
                "built; retire the pin",
            )

    def test_the_documented_exception_is_still_the_only_one(self):
        """A field pinned as read-before-resolution has to still be both."""
        steps, methods, reached, step_lines = _pipeline()
        wanted = _constructor_reads()
        declared = set()
        for step in steps:
            for method in reached[step]:
                for node in ast.walk(methods[method]):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "_declare"
                    ):
                        declared |= {kw.arg for kw in node.keywords if kw.arg}
        stale = sorted(
            field
            for field in _READ_BEFORE_RESOLUTION
            if field not in wanted or field not in declared
        )
        self.assertEqual(
            stale,
            [],
            "these are pinned as read-before-resolution but are no longer both "
            f"read by the constructor and written by resolution: {stale}",
        )


if __name__ == "__main__":
    unittest.main()
