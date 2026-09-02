"""Every `server_args.<name>()` in the tree names something the record has.

Removing a member from `ServerArgs` means rewriting its callers, and the ones
inside `server_args.py` are the ones you fix by reflex. The cross-file caller is
what bites: `ServerArgs.ssl_verify()` moved to `serving_hook.ssl_verify_of()` and
one call site kept the old spelling as `self.server_args.ssl_verify()` -- a grep
for `server_args.ssl_verify()` does not find that, and nothing else looks. Every
`HttpServerEngineAdapter` request raised `AttributeError` before sending.

So this resolves the call sites instead of grepping for them: every attribute
*called* on something statically known to be a record has to exist on the record.
It is deliberately not limited to methods the refactor touched -- the next
removal gets the same check for free.

`multimodal_gen` carries a different, same-named class outside this contract, as
the other record ratchets also record.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import ast
import dataclasses
import pathlib
import unittest

import sglang
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

_ROOTS = (
    pathlib.Path(next(iter(sglang.__path__))) / "srt",
    pathlib.Path(__file__).resolve().parents[3],  # test/
)
_EXCLUDED = ("multimodal_gen",)

# Attribute names that hold a `ServerArgs`. `resolving_view` and `resolved_view`
# proxy the record but answer for names it does not carry, so they are not here.
_RECORD_NAMES = ("server_args", "_server_args")


def _is_record(node) -> bool:
    """`server_args`, `self.server_args`, `self._server_args`, `cls.server_args`."""
    if isinstance(node, ast.Name):
        return node.id in _RECORD_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _RECORD_NAMES
    return False


def _rebound_locally(tree) -> set:
    """Names assigned something that is plainly not a record.

    `server_args` is also a natural name for a dict of CLI flags or a list of
    argv strings in test helpers, and those legitimately answer `.update()` and
    `.items()`. A function that assigns one of those to the name is not talking
    about the record in that scope.
    """
    literal = (ast.Dict, ast.List, ast.DictComp, ast.ListComp)
    builders = {"dict", "list", "tuple", "set"}

    def _not_a_record(value) -> bool:
        if isinstance(value, literal):
            return True
        # `dict(...)` / `list(...)`, and an annotated `server_args: list[str] = [...]`
        return (
            isinstance(value, ast.Call) and getattr(value.func, "id", None) in builders
        )

    rebound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if value is None or not _not_a_record(value):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in _RECORD_NAMES:
                rebound.add(target.id)
    return rebound


def _called_members():
    """{name: [file:line]} for every `<record>.<name>(...)` in the tree."""
    found: dict[str, list[str]] = {}
    for root in _ROOTS:
        for path in sorted(root.rglob("*.py")):
            text = path.as_posix()
            if any(part in text for part in _EXCLUDED):
                continue
            source = path.read_text(encoding="utf-8-sig")
            if "server_args" not in source:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            rebound = _rebound_locally(tree)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and _is_record(node.func.value)
                    and getattr(node.func.value, "id", None) not in rebound
                ):
                    found.setdefault(node.func.attr, []).append(
                        f"{path.name}:{node.lineno}"
                    )
    return found


class TestRecordMemberCallsResolve(CustomTestCase):
    def test_every_called_member_exists_on_the_record(self):
        called = _called_members()
        self.assertGreater(
            len(called),
            5,
            f"only {len(called)} members called on a record; the scan is broken, "
            "not the tree",
        )
        available = set(dir(ServerArgs)) | {
            field.name for field in dataclasses.fields(ServerArgs)
        }
        missing = {
            name: sites
            for name, sites in sorted(called.items())
            if name not in available
        }
        self.assertEqual(
            {},
            missing,
            "these are called on a ServerArgs but the record has no such member -- "
            "each one raises AttributeError at the call. A member that moved out of "
            "the record has to be rewritten at every call site, including the ones "
            f"reached through `self.server_args`: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
