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
import functools
import pathlib
import unittest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).resolve().parent / "srt"

# Read for what the caller asked for: the constructor passes it through and
# never stores it, while resolution declares the value the architecture implies.
# Two quantities sharing one name.
_READ_BEFORE_RESOLUTION = frozenset({"is_embedding"})

# Declared after the first `model_config_of()`, so the cached configuration
# holds the earlier value. Nothing reads the stale copy today (its one consumer
# is on the `is_draft_model` branch, built after resolution), and fixing it
# means moving the build or the hook. Pinned so a second field in this position
# has to be looked at.
_STALE_IN_THE_MODEL_CONFIG = frozenset({"speculative_algorithm"})

# Behind the expert-pack build. `expert_pack_hook.handle_expert_pack` builds a
# model configuration, and it always did -- the walk stopped at the record's
# file and never saw it, so these three read as decided before the first build.
# The call sits behind `load_format != "expert_pack": return`, so it is the
# first build only on an expert-pack launch. Pre-existing; named rather than
# fixed, because fixing it means moving the build or the hook.
_STALE_BEHIND_THE_EXPERT_PACK_BUILD = frozenset(
    {
        "_speculative_draft_quantization_explicitly_set",
        "model_path",
        "speculative_draft_model_quantization",
    }
)

# The same staleness through the registries: `_handle_model_specific_adjustments`
# builds the model configuration and *then* collects the override declarations,
# both inside one handler body. Named rather than fixed (that means moving the
# build or the collection), so a fifth field here has to be looked at -- and so
# does fixing the ordering.
_STALE_FROM_THE_REGISTRIES = frozenset(
    {
        "disable_hybrid_swa_memory",
        "dtype",
        "enable_multi_layer_eagle",
        "quantization",
    }
)


@functools.lru_cache(maxsize=None)
def _parsed(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


@functools.lru_cache(maxsize=None)
def _declared_resolution_fields(path):
    fields = set()
    for node in ast.walk(_parsed(path)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "declare_resolution"
        ):
            fields |= {kw.arg for kw in node.keywords if kw.arg}
    return frozenset(fields)


def _registry_declared_fields():
    """What the live registries and passes declare.

    Imported from the chain ratchet by path instead of re-derived: two
    derivations of the same set drift, and the one that drifts narrower makes
    this check quietly vacuous. Keying on `self._declare(...)` alone is what
    hid these four -- 26 of the providers register through a helper call, and
    none of them spell a keyword this file can see.
    """
    import importlib.util

    ratchet = (
        pathlib.Path(__file__).resolve().parent.parent / "test_chain_read_ratchet.py"
    )
    spec = importlib.util.spec_from_file_location("_chain_ratchet_for_pin", ratchet)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._declared_by_registry_and_passes()


def _registry_collection_is_after_the_build():
    """(collection line, first build line) inside the model-specific handler.

    Handler-local ordering only -- the caller still has to compare against the
    pipeline-wide first build, which sits in an *earlier* step: hoisting the
    collection above this handler's own `model_config_of()` call does not move
    it above the configuration another handler already cached.
    """
    handler = None
    for source, wanted in (
        (_SRT / "server_args.py", "_handle_model_specific_adjustments"),
        *(
            (path, "handle_model_specific_adjustments")
            for path in sorted((_SRT / "arg_groups").glob("*.py"))
        ),
    ):
        for node in ast.walk(_parsed(source)):
            if isinstance(node, ast.FunctionDef) and node.name == wanted:
                if any(
                    isinstance(child, ast.Call)
                    and getattr(child.func, "attr", getattr(child.func, "id", None))
                    == "collect_model_override_declarations"
                    for child in ast.walk(node)
                ):
                    handler = node
                    break
        if handler is not None:
            break
    assert handler is not None, "the model-specific handler was not found"
    build = collect = None
    for node in ast.walk(handler):
        if not isinstance(node, ast.Call):
            continue
        # Both spellings: an Attribute call and a bare Name call.
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        else:
            continue
        if name == "model_config_of" and build is None:
            build = node.lineno
        if name == "collect_model_override_declarations" and collect is None:
            collect = node.lineno
    return collect, build


def _server_args_names(tree, path):
    """Every local that names the record, including the read views over it.

    A resolution-time reader reads through `resolving_view(server_args)` (the
    declaration stash over the fields): declaration-only resolvers write no
    field, so a field read there answers with the raw input. `cfg.dtype` after `cfg = resolving_view(sa)` is
    the same read this scan is looking for, so the local it binds counts.
    """
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
    # `cfg = resolving_view(server_args)` / `resolved_view(server_args)`
    for _ in range(2):  # a view over a view-holding local is still one
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            bare = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in ("resolving_view", "resolved_view")
                and value.args
                and isinstance(value.args[0], ast.Name)
                and value.args[0].id in names
            )
            # `resolved = self._resolved()` is the same view, spelled as the
            # resolution vocabulary.
            member = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and isinstance(value.func, ast.Name)
                and value.func.id == "resolved_view"
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id in names
            )
            if not (bare or member):
                continue
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


def _constructor_reads():
    """Fields `ModelConfig.from_server_args` takes off the record."""
    path = _SRT / "configs/model_config.py"
    tree = _parsed(path)
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
        "parser/template_detection.py",
    ):
        path = _SRT / name
        # A named file that moved away has to be loud; skipping it silently
        # leaves the scan believing it read a module it never opened.
        assert path.exists(), f"{name} is not where this scan looks for it"
        tree = _parsed(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if called == "declare_late_resolution":
                fields |= {kw.arg for kw in node.keywords if kw.arg}
    return fields


def _hook_declarations(dispatch, source_module):
    """{field: dispatcher line} for hooks the dispatch calls on other objects.

    `handle_speculative_decoding(self)` is not a `self.<handler>()` call, so a
    scan of the dispatcher's own method calls never reaches its
    `declare_resolution` sites -- and the speculative hooks decide
    `speculative_algorithm`, which the model configuration reads.

    The platform hook is *not* covered here: it reaches the pipeline as a
    callback argument, so there is no call node to follow and its writes live
    outside this tree. Its position is pinned instead --
    `test_every_opaque_callback_is_still_late`.
    """
    imported = {}
    for node in ast.walk(_parsed(source_module)):
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
        for field in _declared_resolution_fields(path):
            out[field] = max(out.get(field, 0), node.lineno)
    return out


# The dispatcher's own file: its imports are what map a bare-name call in it
# to the family that defines the callable.
_DISPATCH_MODULE = _SRT / "arg_groups" / "pipeline.py"


def _hook_functions():
    """Module-level resolution functions under `arg_groups/`.

    A handler that moved out of the record leaves a slot behind that imports
    one of these and calls it. Without following that hop the scan stops at
    the slot and silently loses everything the handler does.
    """
    functions = {}
    for path in sorted((_SRT / "arg_groups").glob("*.py")):
        for node in _parsed(path).body:
            if isinstance(node, ast.FunctionDef):
                functions.setdefault(node.name, node)
    return functions


def _pipeline():
    """(ordered steps, {step: methods it reaches}) for the resolution dispatch."""
    tree = _parsed(_SRT / "server_args.py")
    record = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
    )
    methods = {
        node.name: node for node in record.body if isinstance(node, ast.FunctionDef)
    }
    # The dispatcher calls its hooks by bare name, so the walk resolves those
    # against `arg_groups/` alongside the record's own methods.
    hooks = _hook_functions()
    methods.update({name: node for name, node in hooks.items() if name not in methods})
    dispatch = methods["run_resolution_pipeline"]
    # A step is either a record method (`self._x()`) or a bare-name hook call.
    steps = [
        name
        for _line, name in sorted(
            (
                node.lineno,
                (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else node.func.id
                ),
            )
            for node in ast.walk(dispatch)
            if isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                )
                or (isinstance(node.func, ast.Name) and node.func.id in hooks)
            )
        )
    ]

    def reaches(name, seen=None):
        seen = seen if seen is not None else set()
        if name in seen or name not in methods:
            return seen
        seen.add(name)
        for node in ast.walk(methods[name]):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr in methods
            ):
                reaches(node.func.attr, seen)
            elif isinstance(node.func, ast.Name) and node.func.id in hooks:
                reaches(node.func.id, seen)
        return seen

    step_lines = {}
    for node in ast.walk(dispatch):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            step_lines.setdefault(node.func.attr, node.lineno)
        elif isinstance(node.func, ast.Name) and node.func.id in hooks:
            step_lines.setdefault(node.func.id, node.lineno)
    return steps, methods, {name: reaches(name) for name in steps}, step_lines


def _opaque_callback_positions(dispatch, source_module):
    """{callback spelling: dispatcher line} for every resolver handed in.

    `declare_direct_writes(record, source, callback)` runs a callable instead of
    code in this tree -- a platform plugin, a registered speculative algorithm.
    Which fields such a callback writes is not a static question; only *when* it
    runs is, so the position is what gets pinned.

    Two spellings reach the pipeline: the dispatcher wraps a callback itself, or
    it calls a hook in this tree that wraps one. The line recorded is always the
    dispatcher's, because that is where the ordering against the build is
    decided -- a hook body sits further down its own file and says nothing about
    it.
    """
    imported = {}
    for node in ast.walk(_parsed(source_module)):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported[alias.asname or alias.name] = node.module

    def callbacks_in(tree):
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else (node.func.attr if isinstance(node.func, ast.Attribute) else None)
            )
            if name == "declare_direct_writes" and len(node.args) > 2:
                found.append(ast.unparse(node.args[2]))
        return found

    positions = {}
    for spelling in callbacks_in(dispatch):
        positions[spelling] = min(
            positions.get(spelling, 10**9),
            next(
                node.lineno
                for node in ast.walk(dispatch)
                if isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "declare_direct_writes"
            ),
        )
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
        for spelling in callbacks_in(_parsed(path)):
            positions[spelling] = min(positions.get(spelling, 10**9), node.lineno)
    return positions


def _declaration_positions():
    """({field: position}, first_build) over the fields the constructor reads.

    A position is `(step index, rank)`, and `rank` is 0 only for a declaration
    that sits *above* the build in the very method that builds: a declaration
    applies where it is written, so one statement earlier in the same body is
    genuinely earlier. Everything else in the build's step gets rank 1 and
    counts as late -- line numbers say nothing across two method bodies, since
    a handler sits further down the file than the dispatcher that calls it.

    One derivation, two callers: the check below asks which fields land after
    the build, and the pin check asks whether an exempted field is still one
    of them. Two derivations of that answer drift apart.
    """
    steps, methods, reached, step_lines = _pipeline()
    wanted = _constructor_reads()

    def build_site():
        """(step index, method name, line) of the first `model_config_of()`."""
        for index, step in enumerate(steps):
            for method in reached[step]:
                for node in ast.walk(methods[method]):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "model_config_of"
                    ):
                        return index, step, method, node.lineno
        return None

    site = build_site()
    if site is None:
        return {}, None
    build_index, build_step, build_method, build_line_in_body = site
    first_build = (build_index, build_step)

    source_module = _DISPATCH_MODULE
    imported = {}
    for node in ast.walk(_parsed(source_module)):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported[alias.asname or alias.name] = node.module

    def _hook_declared_fields(name):
        """Fields a hook imported from `sglang.srt` declares, by callable name."""
        module = imported.get(name)
        if not module or not module.startswith("sglang.srt."):
            return frozenset()
        path = _SRT / (module[len("sglang.srt.") :].replace(".", "/") + ".py")
        if not path.exists():
            return frozenset()
        return _declared_resolution_fields(path)

    declared_at = {}
    for index, step in enumerate(steps):
        for method in reached[step]:
            for node in ast.walk(methods[method]):
                if not isinstance(node, ast.Call):
                    continue
                same_body = index == build_index and method == build_method
                rank = 0 if same_body and node.lineno < build_line_in_body else 1
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "declare_resolution"
                ):
                    fields = {kw.arg for kw in node.keywords if kw.arg}
                # A handler that calls an imported hook (the Kimi and DeepSeek
                # defaults live in arg_groups modules) declares through it, and
                # the hook can sit below the build inside the same handler.
                elif isinstance(node.func, ast.Name):
                    fields = _hook_declared_fields(node.func.id)
                else:
                    continue
                for field in fields:
                    if field in wanted:
                        # The *last* declaration is the one that has to precede
                        # the build.
                        declared_at[field] = max(
                            declared_at.get(field, (index, rank)), (index, rank)
                        )

    # Hooks the dispatch calls on other objects declare too, and a hook below
    # the first build is late by definition. Both positions are read *inside
    # the dispatcher*: a handler body sits further down the file than the
    # dispatcher that calls it, so a line number taken from one scope says
    # nothing about ordering against the other.
    dispatch = methods["run_resolution_pipeline"]
    build_line = step_lines[first_build[1]]
    for field, line in _hook_declarations(dispatch, _DISPATCH_MODULE).items():
        if field in wanted and line > build_line:
            declared_at[field] = max(
                declared_at.get(field, (build_index, 1)), (10**6, 1)
            )

    # Late resolution is the other channel that can decide a field the
    # constructor reads, and it runs after every build.
    for field in _late_resolution_fields():
        if field in wanted:
            declared_at[field] = (10**6, 1)
    return declared_at, first_build


class TestModelConfigReadsResolvedInput(CustomTestCase):
    def test_every_field_it_reads_is_resolved_before_it_is_built(self):
        declared_at, first_build = _declaration_positions()
        self.assertIsNotNone(
            first_build, "no handler builds a ModelConfig; the scan broke"
        )

        known = (
            _READ_BEFORE_RESOLUTION
            | _STALE_IN_THE_MODEL_CONFIG
            | _STALE_FROM_THE_REGISTRIES
            | _STALE_BEHIND_THE_EXPERT_PACK_BUILD
        )
        late = sorted(
            field
            for field, position in declared_at.items()
            if position >= (first_build[0], 1) and field not in known
        )
        self.assertEqual(
            late,
            [],
            "resolution decides these after it builds the ModelConfig that reads "
            f"them, so the model configuration describes a half-resolved input "
            f"(first build: step {first_build[0]}, {first_build[1]}): {late}",
        )

    def test_the_registry_stale_set_is_exactly_what_is_late(self):
        """Equality, not membership.

        A fifth field the registries decide after the build fails here, and so
        does fixing the ordering -- either way someone has to come back and
        read this. The earlier version of this file derived declarations only
        from `self._declare(...)` keywords, so it passed while these four were
        already stale.
        """
        collect_line, build_line = _registry_collection_is_after_the_build()
        self.assertIsNotNone(
            collect_line, "the handler no longer collects registry declarations"
        )
        reads = _constructor_reads()
        registry = _registry_declared_fields()
        self.assertGreater(
            len(registry), 20, "the registry-declared set collapsed; nothing to compare"
        )
        # Late against the *pipeline-wide* first build, not only the build in
        # the collection's own handler: `_handle_gpu_memory_settings` builds
        # the configuration many steps earlier, so hoisting the collection
        # above the local build still leaves that cache describing raw input.
        steps, methods, reached, _step_lines = _pipeline()
        _declared_at, first_build = _declaration_positions()
        self.assertIsNotNone(
            first_build, "no handler builds a ModelConfig; the scan broke"
        )
        collecting_steps = [
            index
            for index, step in enumerate(steps)
            for method in reached[step]
            if any(
                isinstance(node, ast.Call)
                and (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else getattr(node.func, "id", None)
                )
                == "collect_model_override_declarations"
                for node in ast.walk(methods[method])
            )
        ]
        self.assertTrue(collecting_steps, "no pipeline step collects the registry")
        collection_is_late = min(collecting_steps) > first_build[0] or (
            build_line is not None and collect_line > build_line
        )
        late = frozenset(reads & registry) if collection_is_late else frozenset()
        self.assertEqual(
            sorted(late),
            sorted(_STALE_FROM_THE_REGISTRIES),
            "the set of ModelConfig-read fields the registries decide after the "
            f"build changed (collection at line {collect_line}, build at line "
            f"{build_line}); read the comment on _STALE_FROM_THE_REGISTRIES "
            "before editing it",
        )

    def test_the_pinned_stale_field_is_still_stale(self):
        """If the ordering gets fixed, this pin has to be retired, not kept.

        A pin that outlives the defect it describes is worse than none: it
        documents a hazard that no longer exists and hides the day one appears.
        """
        steps, methods, reached, step_lines = _pipeline()
        dispatch = methods["run_resolution_pipeline"]
        hooks = _hook_declarations(dispatch, _DISPATCH_MODULE)
        build_line = min(
            step_lines[step]
            for step in steps
            for method in reached[step]
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "model_config_of"
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

    def test_every_opaque_callback_is_still_late(self):
        """The opaque resolvers all run after the model configuration is built.

        A plugin that rewrites `dtype` or `model_path` in one of them is
        invisible to the configuration already cached, and no scan can say
        whether it does: the implementations are out of tree. So the positions
        are the pin, and the set of callbacks is pinned with them -- a new one
        has to be placed against the build by whoever adds it. Moving them all
        above the build fixes the hazard and fails this test; retire the pin
        then, rather than keeping a note about a hazard that is gone.
        """
        steps, methods, reached, step_lines = _pipeline()
        dispatch = methods["run_resolution_pipeline"]
        positions = _opaque_callback_positions(dispatch, _DISPATCH_MODULE)
        self.assertEqual(
            sorted(positions),
            [
                "algo.handle_server_args",
                "algo.validate_server_args",
                "current_platform.apply_server_args_defaults",
            ],
            "the set of resolvers handed to declare_direct_writes changed; each "
            "one needs its position against the ModelConfig build looked at",
        )
        _declared_at, first_build = _declaration_positions()
        self.assertIsNotNone(
            first_build, "no handler builds a ModelConfig; the scan broke"
        )
        build_line = step_lines[first_build[1]]
        for spelling, line in sorted(positions.items()):
            self.assertGreater(
                line,
                build_line,
                f"{spelling} now runs before the model configuration is built, "
                "so a plugin's writes reach it; retire the pin",
            )

    def test_the_documented_exception_is_still_the_only_one(self):
        """A field pinned as read-before-resolution has to still be all three.

        Read by the constructor, written by resolution, and written *after* the
        build -- the last one is what makes the exemption load-bearing. Without
        it, moving the declaration earlier leaves the name sitting in the
        exempt set with nothing to exempt, and the next field that lands in
        this position gets waved through by a pin nobody re-read.
        """
        wanted = _constructor_reads()
        declared_at, first_build = _declaration_positions()
        self.assertIsNotNone(
            first_build, "no handler builds a ModelConfig; the scan broke"
        )
        for field in sorted(_READ_BEFORE_RESOLUTION):
            self.assertIn(
                field,
                wanted,
                f"{field} is pinned as read before resolution, but the "
                "constructor no longer reads it; retire the pin",
            )
            self.assertIn(
                field,
                declared_at,
                f"{field} is pinned as read before resolution, but resolution "
                "no longer writes it; retire the pin",
            )
            self.assertGreaterEqual(
                declared_at[field],
                (first_build[0], 1),
                f"{field} is now decided before the model configuration is "
                "built, so the exemption covers nothing; retire the pin",
            )


if __name__ == "__main__":
    unittest.main()
