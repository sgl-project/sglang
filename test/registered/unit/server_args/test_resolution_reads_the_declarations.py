"""Resolution reads its own decisions, not the record's fields.

`declare_resolution` records a decision in the declaration stash and writes
nothing. The fields keep what the caller passed, so a resolver that reads a
field another resolver may have decided reads the raw input -- silently, and
only on the configurations where that other resolver fires. The whole pipeline
therefore reads through `resolving_view` (or `resolved_view`, which is
the same view after resolution has finished), and this pins that there is
nothing left reading a field directly.

Subjects: every function in `arg_groups/` that takes a config, every
`ServerArgs` handler the dispatcher reaches, and every member of `ServerArgs` /
`PortArgs` -- the members are reached from the hooks and from business code,
which the handler walk cannot see, and a member that recomputes from a raw field
decides from what was typed. All three
are derived -- a new hook file, a new handler or a new member is covered the
moment it is written. Readers *outside* those
two -- the platform defaults, `ModelConfig`, the spec-algo hook -- are reached by
resolution too and have moved to the view as well, but enumerating them needs
the call-graph derivation `test_resolution_reads_no_bag` owns; this file pins
the two scopes it can derive exactly.
"""

import ast
import dataclasses
import pathlib
import re

import sglang
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).resolve().parent / "srt"
_FIELDS = frozenset(field.name for field in dataclasses.fields(ServerArgs))

# Names a config travels under. `args` is included because the platform hooks
# use it; a false positive would be a function taking an argparse Namespace and
# reading an attribute that happens to be a ServerArgs field name, which the
# allowlist below would then have to carry.
_HOLDER_NAMES = frozenset({"server_args", "sa", "args"})


def _holders(fn):
    names = {
        arg.arg
        for arg in list(fn.args.posonlyargs)
        + list(fn.args.args)
        + list(fn.args.kwonlyargs)
        if arg.arg in _HOLDER_NAMES
    }
    for arg in (
        list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs)
    ):
        annotation = arg.annotation
        text = (
            annotation.value
            if isinstance(annotation, ast.Constant)
            else (
                annotation.id
                if isinstance(annotation, ast.Name)
                else annotation.attr
                if isinstance(annotation, ast.Attribute)
                else None
            )
        )
        if text == "ServerArgs":
            names.add(arg.arg)
    return names


def _field_reads(fn, holders):
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _FIELDS
            and isinstance(node.value, ast.Name)
            and node.value.id in holders
            and isinstance(node.ctx, ast.Load)
        ):
            yield node.lineno, node.attr


_DECLARERS = frozenset(
    {
        "declare_resolution",
        "declare_late_resolution",
        "declare_direct_writes",
    }
)


def _declared_fields():
    """The fields resolution decides, read off every shape that reaches the stash.

    A keyword on a `declare_*` call is only one shape: the model-override and
    post-process passes build a mapping instead (`MODEL_OVERRIDES` literals,
    `overrides["dtype"] = ...`, a returned dict), and late resolution splats a
    variable-keyed one. Deriving from keywords alone leaves nineteen fields
    outside the subject set, `dtype` and `reasoning_parser` among them.
    """
    fields = set()
    # The declaration calls live wherever a resolver does; the mapping channels
    # only exist where the override providers and post-process passes are.
    keyword_sources = [_SRT / "server_args.py"]
    for sub in ("arg_groups", "hardware_backend", "parser"):
        keyword_sources += sorted((_SRT / sub).rglob("*.py"))
    mapping_sources = {_SRT / "server_args.py", *(_SRT / "arg_groups").rglob("*.py")}
    for path in keyword_sources:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            # 1. `declare_resolution(sa, src, page_size=64)` and its siblings
            if isinstance(node, ast.Call):
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else getattr(node.func, "attr", None)
                )
                if name in _DECLARERS:
                    for keyword in node.keywords:
                        if keyword.arg:
                            fields.add(keyword.arg)
                        elif isinstance(keyword.value, ast.Dict):
                            fields.update(_string_keys(keyword.value))
            # 2. every mapping literal in the files that declare through one:
            #    the MODEL_OVERRIDES tables, the dicts the override providers
            #    return, the ones the post-process passes build. Scanning
            #    unrelated files here would collect a plain kwarg dict
            #    (`tokenizer_config={"trust_remote_code": ...}`) and turn a
            #    passthrough read into a violation.
            if isinstance(node, ast.Dict) and path in mapping_sources:
                fields.update(_string_keys(node))
            # 3. `overrides["field"] = ...`
            if (
                path in mapping_sources
                and isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].slice, ast.Constant)
                and isinstance(node.targets[0].slice.value, str)
            ):
                fields.add(node.targets[0].slice.value)
    return frozenset(fields & _FIELDS)


def _string_keys(node: ast.Dict) -> set:
    return {
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _record_members():
    """Every member of `ServerArgs` / `PortArgs`, by class and name."""
    tree = ast.parse((_SRT / "server_args.py").read_text(encoding="utf-8-sig"))
    members = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in ("ServerArgs", "PortArgs"):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members[f"{node.name}.{member.name}"] = member
    return members


def _config_reading_helpers():
    """Module functions that load a decided field off the config they are handed.

    A member that hands them `self`, or a call site that hands them a record,
    reads the raw input through the callee -- the shape neither an attribute
    scan nor a `getattr` scan can see, because the field name is spelled in the
    helper and the record is spelled at the call site.
    """
    decided = _declared_fields()
    helpers = {}
    sources = [_SRT / "server_args.py"] + sorted((_SRT / "arg_groups").rglob("*.py"))
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = {
                arg.arg
                for arg in list(fn.args.posonlyargs)
                + list(fn.args.args)
                + list(fn.args.kwonlyargs)
            } - {"self", "cls"}
            if not params:
                continue
            reads = {
                node.attr
                for node in ast.walk(fn)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in params
                and isinstance(node.ctx, ast.Load)
                and node.attr in decided
            }
            if reads:
                helpers[fn.name] = sorted(reads)
    return helpers


# The accessors that hand back the process-global record itself. A helper that
# is handed one of these reads the raw input exactly as a bare `self` would.
_RECORD_ACCESSORS = frozenset({"get_server_args", "global_server_args"})

# `self._server_args` is the same record under a private name; the scan has to
# see it or a reader inside the context object escapes every shape above.
_RECORD_ATTR = re.compile(r"^_*(server_args|sa)$")


def _record_arguments(node, aliases=frozenset()):
    """The bare-record arguments of a call.

    Four spellings reach a helper with a record: the bare name (`self`, `sa`),
    an attribute (`runner.server_args`), the process-global accessor called
    inline (`get_server_args()`), and a local bound to either of the last two
    earlier in the same function.
    """
    out = []
    for arg in node.args:
        if isinstance(arg, ast.Name) and arg.id in ("self", "server_args", "sa"):
            out.append(arg.id)
        elif isinstance(arg, ast.Attribute) and _RECORD_ATTR.match(arg.attr or ""):
            out.append(ast.unparse(arg))
        elif (
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Name)
            and arg.func.id in _RECORD_ACCESSORS
        ):
            out.append(ast.unparse(arg))
        elif isinstance(arg, ast.Name) and arg.id in aliases:
            out.append(arg.id)
    return out


def _record_aliases(function):
    """Locals bound to the record under a name of their own.

    `_sa = getattr(runner, "server_args", None)`, `cfg = get_server_args()` and
    `engine_args = ServerArgs.from_cli_args(args)` all put the record behind a
    name the argument scan does not recognise, so a later
    `getattr(_sa, "<decided leaf>")` reads what the operator typed.
    """
    aliases = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if isinstance(value, ast.Attribute):
            if _RECORD_ATTR.match(value.attr or ""):
                aliases.add(target.id)
            continue
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        if isinstance(func, ast.Name):
            if func.id in _RECORD_ACCESSORS or func.id == "ServerArgs":
                aliases.add(target.id)
            elif (
                func.id == "getattr"
                and len(value.args) >= 2
                and isinstance(value.args[1], ast.Constant)
                and _RECORD_ATTR.match(str(value.args[1].value))
            ):
                aliases.add(target.id)
        elif (
            isinstance(func, ast.Attribute)
            and func.attr in ("from_cli_args", "replace_resolved")
            and isinstance(func.value, ast.Name)
            and func.value.id == "ServerArgs"
        ):
            aliases.add(target.id)
    return aliases


# The one reader for which the raw field is the right answer. The gateway sizes
# its worker pool from the operator's requested replica count; `--dwdp-size`
# makes resolution declare a `dp_size` describing one multi-rank server's
# internal topology, so reading the decision there would spawn dp_size
# single-rank children and ask for dp_size^2 GPUs. A new entry here needs that
# kind of reason next to it.
_NO_RESOLVED_SURFACE = frozenset(
    {
        (
            "sgl-model-gateway/bindings/python/src/sglang_router/launch_server.py",
            "server_args.dp_size",
        ),
    }
)


def _is_record_base(node, aliases):
    """Is this expression the record itself?

    A local bound to one, a parameter that carries one (`server_args`, `sa`,
    `engine_args`), or an attribute holding one (`self._server_args`).
    """
    if isinstance(node, ast.Name):
        return node.id in aliases
    if isinstance(node, ast.Attribute):
        return bool(_RECORD_ATTR.match(node.attr or ""))
    return False


def _record_handoff_offenders(rel, tree, helpers, decided, is_record=False):
    """Every way a decided leaf is reached through a record in one module.

    Two shapes, both scanned under the record aliases the function binds:
    handing the record to a helper that loads a decided field, and loading one
    off the alias directly (`alias.<leaf>` or `getattr(alias, "<leaf>")`). The
    second is what the MiniMax backend spelled, and an argument scan cannot see
    it -- the leaf never appears at a call site.
    """
    offenders, seen = [], set()

    def record(lineno, text):
        if (lineno, text) in seen:
            return
        seen.add((lineno, text))
        offenders.append(f"{rel}:{lineno} {text}")

    scopes = [(tree, frozenset())] + [
        (fn, _record_aliases(fn))
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for scope, aliases in scopes:
        for node in ast.walk(scope):
            if isinstance(node, ast.Call):
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else getattr(node.func, "attr", None)
                )
                if name in helpers:
                    for arg in _record_arguments(node, aliases):
                        if arg == "self" and not is_record:
                            continue
                        record(
                            node.lineno,
                            f"{name}({arg}) reads {', '.join(helpers[name])}",
                        )
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in aliases
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in decided
                ):
                    record(
                        node.lineno,
                        f'getattr({node.args[0].id}, "{node.args[1].value}")',
                    )
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and node.attr in decided
                and _is_record_base(node.value, aliases)
            ):
                record(node.lineno, f"{ast.unparse(node.value)}.{node.attr}")
    return offenders


# Source the scanner must read the same way whether or not the tree happens to
# contain these shapes today. The first four are the spellings that reached
# production and were converted; the last two are the legal forms next to them,
# which have to stay quiet or the guard is unusable.
_SPELLINGS = """
def hands_the_alias_to_a_helper(runner):
    _sa = getattr(runner, "server_args", None)
    return m3_fp8_attn_gemm_enabled(_sa)


def loads_a_leaf_off_the_alias(runner):
    _sa = getattr(runner, "server_args", None)
    return getattr(_sa, "speculative_num_draft_tokens", None)


def reads_a_leaf_through_the_alias(runner):
    sa_local = runner.server_args
    return sa_local.attention_backend


def hands_the_accessor_to_a_helper():
    return attention_backends_of(get_server_args())


def reads_the_view(runner):
    cfg = resolving_view(runner.server_args)
    return cfg.attention_backend


def reads_an_undecided_leaf(runner):
    _sa = runner.server_args
    return _sa.tp_size


def reads_a_leaf_off_a_private_attribute(self):
    return self._server_args.attention_backend


def reads_a_leaf_off_a_constructed_record(cli):
    engine_args = ServerArgs.from_cli_args(cli)
    engine_args.resolve_once()
    return engine_args.attention_backend
"""


class TestResolutionReadsTheDeclarations(CustomTestCase):
    def test_no_hook_reads_a_field_off_the_record(self):
        offenders = []
        files = sorted((_SRT / "arg_groups").glob("*.py"))
        self.assertGreater(len(files), 5, "the hook scan found almost nothing")
        for path in files:
            rel = f"arg_groups/{path.name}"
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                holders = _holders(fn)
                if not holders:
                    continue
                for lineno, field in _field_reads(fn, holders):
                    offenders.append(f"{rel}:{lineno} {fn.name} reads .{field}")
        self.assertEqual(
            offenders,
            [],
            "a resolution hook reads a field off the record; the field holds the "
            "raw input, so this decides from what was typed rather than from "
            "what resolution decided. Read `resolving_view(server_args)`:\n  "
            + "\n  ".join(offenders),
        )

    def test_the_record_hosts_no_resolution_handler(self):
        """The pipeline and every step it runs live under `arg_groups/`.

        While a step was a method, it could read a raw field off `self` and
        `test_no_handler_reads_a_field_off_self` had to say it could not. There
        is no such method left, so the invariant is now the stronger one: the
        record hosts none of them. What the steps read is checked on the
        package side, by `test_no_hook_reads_a_field_off_the_record`.
        """
        handlers = sorted(
            name
            for name, node in _record_members().items()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (
                name.split(".")[-1].startswith(("_handle_", "_validate_"))
                or "resolution_pipeline" in name
            )
        )
        self.assertEqual(
            handlers,
            [],
            "a resolution handler is back on the record; it belongs in an "
            "`arg_groups` family, where the package-side guards can see it:\n  "
            + "\n  ".join(handlers),
        )

    def test_no_member_recomputes_from_a_raw_field(self):
        decided = _declared_fields()
        self.assertGreater(
            len(decided), 100, f"the declaration set derived only {len(decided)} fields"
        )
        members = _record_members()
        # The floor is here to catch the scan collapsing, not to pin the
        # class's size.
        self.assertGreater(len(members), 15, f"only {len(members)} members were found")
        offenders = []
        for name, fn in sorted(members.items()):
            holders = _holders(fn) | {"self"}
            for lineno, field in _field_reads(fn, holders):
                if field in decided:
                    offenders.append(f"server_args.py:{lineno} {name} reads .{field}")
        self.assertEqual(
            offenders,
            [],
            "a record member recomputes from a field resolution decides; the "
            "field holds the raw input, so the member answers for what was "
            "typed. Bind `cfg = resolving_view(self)` and read that:\n  "
            + "\n  ".join(offenders),
        )

    def test_no_reader_hands_the_record_to_a_config_helper(self):
        helpers = _config_reading_helpers()
        self.assertGreater(
            len(helpers), 5, f"the helper derivation found only {len(helpers)}"
        )
        decided = _declared_fields()
        offenders = []
        # `scripts/`, `examples/` and the gateway binding are outside the
        # package but hold records they resolve themselves, and every reader
        # this scan found in them was reading a field resolution fills in.
        _REPO = _SRT.parent.parent.parent
        roots = (
            [_SRT]
            + [_SRT.parent / d for d in ("benchmark", "lang")]
            + [
                _REPO / d
                for d in (
                    "scripts",
                    "examples",
                    "sgl-model-gateway/bindings/python/src",
                )
            ]
        )
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.py")):
                # Package files keep their `srt/...` spelling (the skips below
                # key on it); the repo-level roots are named from the repo.
                try:
                    rel = path.relative_to(_SRT.parent).as_posix()
                except ValueError:
                    rel = path.relative_to(_REPO).as_posix()
                if rel.startswith(("srt/arg_groups/", "multimodal_gen/")):
                    continue
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                except SyntaxError:
                    continue
                offenders += _record_handoff_offenders(
                    rel, tree, helpers, decided, is_record=rel == "srt/server_args.py"
                )
        offenders = [
            line
            for line in offenders
            if (line.split(":", 1)[0], line.split(" ", 1)[1])
            not in _NO_RESOLVED_SURFACE
        ]
        self.assertEqual(
            offenders,
            [],
            "a caller hands the record to a helper that loads a field "
            "resolution decides; the helper then reads the raw input. Hand it "
            "`resolving_view(record)` (or the published bag):\n  "
            + "\n  ".join(offenders),
        )

    def test_the_scan_sees_every_spelling_that_reached_production(self):
        """Every spelling that reached production, pinned next to the scanner.

        A shape the scan stops seeing is a silent hole, so each one is listed
        here with the legal forms beside it and the flagged set compared
        exactly.

        What it does not reach: a record that arrives as a *parameter* and was
        resolved by the caller (`scripts/playground/bench_speculative.py` hands
        `main(args, server_args)` one). Binding that would need the call graph,
        and naming a parameter `server_args` is also how the resolution-time
        readers spell a view.
        """
        helpers = _config_reading_helpers()
        decided = _declared_fields()
        for name in ("m3_fp8_attn_gemm_enabled", "attention_backends_of"):
            self.assertIn(name, helpers, f"the helper derivation lost {name}")
        for field in ("speculative_num_draft_tokens", "attention_backend"):
            self.assertIn(field, decided, f"the declared set lost {field}")

        offenders = _record_handoff_offenders(
            "sample.py", ast.parse(_SPELLINGS), helpers, decided
        )
        flagged = {line.split(" ", 1)[1] for line in offenders}
        self.assertEqual(
            flagged,
            {
                "m3_fp8_attn_gemm_enabled(_sa)"
                " reads " + ", ".join(helpers["m3_fp8_attn_gemm_enabled"]),
                'getattr(_sa, "speculative_num_draft_tokens")',
                "sa_local.attention_backend",
                "self._server_args.attention_backend",
                "engine_args.attention_backend",
                "attention_backends_of(get_server_args())"
                " reads " + ", ".join(helpers["attention_backends_of"]),
            },
            "the scan lost a spelling, or started flagging a legal one:\n  "
            + "\n  ".join(sorted(flagged)),
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
