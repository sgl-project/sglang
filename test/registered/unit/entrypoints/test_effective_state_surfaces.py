"""Every serving surface can answer what is running, not only what was asked.

`/server_info` and its gRPC and in-process twins report the startup record,
parsers included -- the launcher resolves `auto` into the record before
publishing. What changes after publication -- the model a weight update
swapped in, its load format, an operator-set weight version -- is reported by
the model-info surface, and there is one per entry point: HTTP, gRPC and
`Engine`. Adding a field to one and forgetting the others leaves that entry
point's users with no way to see it, which no test notices because each
surface passes its own tests.

The required set has two halves. The derived half comes from the control-plane
writers: whatever a process writes with `override` after publication is exactly
what can differ from the record, and both the keyword and the `**`-expansion
shapes resolve statically here. The second half is a policy, not a derivation --
what any one surface reports effectively, all of them owe their users -- so a
field every surface drops at once leaves the set with it. Each surface must both
carry the key and take its value from the effective config: reading
`server_args.<field>` under the right key reports the startup value with a
straight face.
"""

import ast
import inspect
import pathlib
import re
import unittest

import sglang
from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")

_PACKAGE_ROOT = pathlib.Path(sglang.__file__).resolve().parent
_RUST_MODEL_INFO = (
    pathlib.Path(__file__).resolve().parents[4]
    / "rust/sglang-server/src/api_server/common.rs"
)

# The tokenizer process is the one whose control-plane writes a served request
# can observe; a field it overrides there is a post-launch fact.
_TOKENIZER_WRITERS = (
    "srt/managers/tokenizer_manager.py",
    "srt/managers/tokenizer_control_mixin.py",
    "srt/entrypoints/http_server.py",
    "srt/entrypoints/engine.py",
)

# The model path and the served name stay manager attributes: a weight update
# moves them on the manager rather than through `override`, so the writer
# derivation cannot see them. They are subtracted from the derived half and
# asserted directly instead -- an exemption whose premise ("every surface
# reports them") is checked, not assumed. It was not true when it was written:
# only `Engine` carried `served_model_name`, and the router had to read the
# launch record off `/server_info` to learn a name a weight update had moved.
_MANAGER_ATTRIBUTES = {"model_path", "served_model_name"}


def _hicache_status_fields() -> set:
    """The fields `GET /hicache/storage-backend` answers with.

    The HiCache mirror is written post-publish and reported by its own
    endpoint, so the model-info surfaces do not owe it. Taking the set from
    that handler is what keeps the exemption honest: a field it stops
    reporting falls back to them.
    """
    tree = ast.parse((_PACKAGE_ROOT / "srt/entrypoints/http_server.py").read_text())
    for fn in ast.walk(tree):
        if (
            not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            or fn.name != "hicache_storage_backend_status"
        ):
            continue
        fields = {
            elt.value
            for comp in ast.walk(fn)
            if isinstance(comp, ast.DictComp)
            for gen in comp.generators
            for elt in getattr(gen.iter, "elts", [])
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        }
        assert fields, "the HiCache status handler names no field"
        return fields
    raise AssertionError(
        "no `hicache_storage_backend_status` handler: the HiCache fields have "
        "no endpoint of their own and fall to the model-info surfaces"
    )


def _expanded_write_keys(rel: str, tree: ast.AST, call: ast.Call, kw: ast.keyword):
    """The field names behind a `**` at a control-plane writer call.

    Resolves a dict literal -- constant keys, or a key bound by an enclosing
    literal `for` -- and a name bound to a dict literal in the enclosing
    function, including constant-subscript stores onto it. A `**` that
    forwards its own function's `**kwargs` names no field: its callers do.
    Anything else raises, because a skipped expansion shrinks the required
    set instead of failing.
    """
    enclosing = None
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            fn.lineno <= call.lineno <= (fn.end_lineno or fn.lineno)
        ):
            if enclosing is None or fn.lineno > enclosing.lineno:
                enclosing = fn

    def loop_bound(name: str) -> set:
        """The values a `for name, ... in (<literal tuples>)` around the call binds."""
        values = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            target = node.target
            names = (
                [target]
                if isinstance(target, ast.Name)
                else list(getattr(target, "elts", []))
            )
            if not names or not isinstance(names[0], ast.Name) or names[0].id != name:
                continue
            if not (node.lineno <= call.lineno <= (node.end_lineno or node.lineno)):
                continue
            for item in getattr(node.iter, "elts", []):
                first = (
                    item.elts[0] if isinstance(item, ast.Tuple) and item.elts else item
                )
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    values.add(first.value)
        return values

    def dict_keys(node: ast.Dict) -> set:
        keys = set()
        for key in node.keys:
            if isinstance(key, ast.Constant):
                keys.add(key.value)
                continue
            assert isinstance(
                key, ast.Name
            ), f"non-literal dict key in a writer expansion at {rel}:{call.lineno}"
            bound = loop_bound(key.id)
            assert bound, (
                f"dict key {key.id!r} at {rel}:{call.lineno} is not bound by a "
                "literal loop; extend the resolver"
            )
            keys |= bound
        return keys

    if isinstance(kw.value, ast.Dict):
        return dict_keys(kw.value)
    assert isinstance(
        kw.value, ast.Name
    ), f"unresolvable writer expansion at {rel}:{call.lineno}"
    assert (
        enclosing is not None
    ), f"writer expansion outside any function at {rel}:{call.lineno}"
    name = kw.value.id
    if enclosing.args.kwarg is not None and enclosing.args.kwarg.arg == name:
        return set()
    keys = set()
    found = False
    for node in ast.walk(enclosing):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            assert isinstance(node.value, ast.Dict), (
                f"writer expansion {name!r} at {rel}:{call.lineno} is assigned "
                "something other than a dict literal; extend the resolver"
            )
            found = True
            keys |= dict_keys(node.value)
        elif (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == name
            and isinstance(target.slice, ast.Constant)
        ):
            keys.add(target.slice.value)
    assert found, (
        f"writer expansion {name!r} at {rel}:{call.lineno} has no dict-literal "
        "assignment in its function; extend the resolver"
    )
    return keys


def _overridden_fields() -> set:
    """Fields the tokenizer process writes after publication."""
    fields = set()
    for rel in _TOKENIZER_WRITERS:
        tree = ast.parse((_PACKAGE_ROOT / rel).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if name not in ("override", "record_config_updates"):
                continue
            for kw in node.keywords:
                if kw.arg == "source":
                    # The provenance label, not a config field.
                    continue
                if kw.arg:
                    fields.add(kw.arg)
                else:
                    fields |= _expanded_write_keys(rel, tree, node, kw)
    # Only the mirror's own fields earn the endpoint exemption: adding an
    # unrelated field to that handler must not buy it a pass here.
    hicache = {f for f in _hicache_status_fields() if f.startswith("hicache_")}
    return fields - _MANAGER_ATTRIBUTES - hicache


def _effective_reads_in(source: str, func_name: str) -> set:
    """Fields the function reports *and* reads through the effective config.

    A key whose value comes off the `ServerArgs` record does not count: that is
    the startup value under a name that promises the running one.
    """
    tree = ast.parse(source)
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name != func_name:
            continue
        reported = set()
        for node in ast.walk(fn):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                    continue
                for inner in ast.walk(value):
                    if (
                        isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr in ("config_value", "config_leaf")
                        and inner.args
                        and isinstance(inner.args[0], ast.Constant)
                        and inner.args[0].value == key.value
                    ):
                        reported.add(key.value)
        return reported
    return set()


def _rust_model_info_keys() -> set:
    """The keys the rust server's `/model_info` handler answers with.

    Scanned as text, from the handler's signature to the next item in the
    file, with each line cut at its first `//`. A key is always left of its
    value, so cutting inside a string can only drop keys, never invent one.
    The signature must appear exactly once: a rust handler that moved or was
    renamed would otherwise contribute an empty set and let the parity check
    pass on nothing.
    """
    source = _RUST_MODEL_INFO.read_text()
    marker = "async fn model_info("
    found = source.count(marker)
    assert found == 1, (
        f"{_RUST_MODEL_INFO.name} declares `{marker}` {found} times; the rust "
        "/model_info surface is the one users reach under SGLANG_RUST_SERVER=1 "
        "and is no longer being read"
    )
    body = source[source.index(marker) + len(marker) :]
    following = re.search(r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s", body, re.M)
    if following is not None:
        body = body[: following.start()]
    code = "\n".join(line.split("//")[0] for line in body.splitlines())
    keys = set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', code))
    assert keys, "the rust /model_info handler names no field"
    return keys


def _reported_keys_in(source: str, func_name: str) -> set:
    """Every string key the function's response dicts carry, whatever the value.

    The manager-owned attributes are read off the manager, not through
    `config_value`, so `_effective_reads_in` does not see them; this is how they
    are checked.
    """
    tree = ast.parse(source)
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name != func_name:
            continue
        return {
            key.value
            for node in ast.walk(fn)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    return set()


def _model_info_sources() -> dict:
    """`(source, function name)` per Python model-info surface."""
    return {
        "http /model_info": (
            (_PACKAGE_ROOT / "srt/entrypoints/http_server.py").read_text(),
            "model_info",
        ),
        "grpc get_model_info": (
            (_PACKAGE_ROOT / "srt/entrypoints/grpc_bridge.py").read_text(),
            "get_model_info",
        ),
        "Engine.get_model_info": (
            inspect.getsource(Engine.get_model_info).lstrip(),
            "get_model_info",
        ),
    }


def _model_info_surfaces() -> dict:
    """Each Python entry point's model-info surface, by what it reports
    effectively."""
    return {
        name: _effective_reads_in(source, func_name)
        for name, (source, func_name) in _model_info_sources().items()
    }


class TestEffectiveStateSurfaces(CustomTestCase):
    def test_each_entry_point_reports_the_post_launch_facts(self):
        surfaces = _model_info_surfaces()
        written = _overridden_fields()
        # Both writer shapes are in reach of the scan: `load_format` is a
        # literal keyword, the parsers arrive as `**{attr: ...}` under a key
        # the enclosing loop binds.
        self.assertLessEqual(
            {"load_format", "reasoning_parser", "tool_call_parser"},
            written,
            "the derivation stopped finding the control-plane writers",
        )
        # What any surface reports effectively, all of them owe their users;
        # what the control plane overrides, every surface owes regardless.
        required = set().union(*surfaces.values()) | written
        missing = {
            name: sorted(required - reported)
            for name, reported in surfaces.items()
            if required - reported
        }
        self.assertEqual(
            missing,
            {},
            "a serving surface cannot report what it is running: " f"{missing}",
        )

    def test_every_surface_reports_the_manager_owned_identity(self):
        """The identity a weight update moves is answered where it is read.

        `model_path` and `served_model_name` live on the tokenizer manager, so
        the writer derivation cannot reach them and they are subtracted from the
        required set. This is the assertion that pays for that subtraction. A
        surface that drops one sends its clients back to the launch record --
        which is what the rust router had to read, under a name that had since
        moved.
        """
        surfaces = {
            name: _reported_keys_in(source, func_name)
            for name, (source, func_name) in _model_info_sources().items()
        }
        surfaces["rust /model_info"] = _rust_model_info_keys()
        missing = {
            name: sorted(_MANAGER_ATTRIBUTES - reported)
            for name, reported in surfaces.items()
            if _MANAGER_ATTRIBUTES - reported
        }
        self.assertEqual(
            missing,
            {},
            f"a model-info surface does not say which model it serves: {missing}",
        )

    def test_the_rust_model_info_answers_the_same_keys(self):
        """`SGLANG_RUST_SERVER=1` swaps the whole HTTP server, not one handler.

        The keys are owed there too, or the endpoint's contract depends on
        which server the operator launched. The values are the launch record:
        that process parses `server_args` once and mounts no route that can
        change weights or parsers, so this is a key-set check and the handler
        states which it reports.
        """
        required = set().union(*_model_info_surfaces().values()) | _overridden_fields()
        missing = sorted(required - _rust_model_info_keys())
        self.assertEqual(
            missing,
            [],
            "the rust /model_info answers a different contract than the Python "
            f"one it replaces: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
