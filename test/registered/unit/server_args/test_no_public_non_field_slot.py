"""The record grows no attribute the projection cannot see.

A publicly-named attribute that is not a dataclass field is invisible to every
other guard here: the namespace coverage walks fields, the projection walks
fields, and the read ratchets watch field reads. Three of them accumulated that
way -- a `ModelConfig` cache, an `moe_ep_size` that only a log line read, and an
env-derived `grpc_worker_threads` that one entry point read across the boundary.

Leading-underscore names are the record's own bookkeeping and stay: the
read-only guard classifies writability by that spelling, so a private name is
already outside the config tier by construction.
"""

import ast
import dataclasses
import pathlib
import unittest

import sglang
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _self_written_attributes() -> set:
    """Names `ServerArgs` writes on itself, by either spelling."""
    source = (
        pathlib.Path(next(iter(sglang.__path__))) / "srt" / "server_args.py"
    ).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
    )
    written = set()
    for node in ast.walk(cls):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    written.add(target.attr)
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "__setattr__"
            and getattr(getattr(node.func, "value", None), "id", None) == "object"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
        ):
            written.add(node.args[1].value)
    return written


class TestNoPublicNonFieldSlot(CustomTestCase):
    def test_every_public_attribute_is_a_field(self):
        written = _self_written_attributes()
        self.assertGreater(
            len(written),
            3,
            f"only {len(written)} self-writes found; the scan is broken, not the "
            "record",
        )
        fields = {field.name for field in dataclasses.fields(ServerArgs)}
        stray = sorted(
            name for name in written if not name.startswith("_") and name not in fields
        )
        self.assertEqual(
            [],
            stray,
            "these are written on the record under a public name but are not "
            "fields, so the projection cannot see them and no other guard "
            "watches them: make each a field, or give it the leading underscore "
            f"that says it is the record's own bookkeeping: {stray}",
        )


if __name__ == "__main__":
    unittest.main()
