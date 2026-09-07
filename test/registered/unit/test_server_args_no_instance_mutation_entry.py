"""``ServerArgs`` has no in-place mutation entry, and nothing calls one.

``ServerArgs.override(source, **fields)`` used to mutate a resolved instance,
and ``ServerArgs.derive(source, **fields)`` used to copy-and-edit one; after
resolution the fields are the record the config bags were projected from, so a
write desyncs every namespace reader, and a copy invites publishing stale
variants. Both are gone: post-publish changes go to the bags
(``get_context().override``), a value one runner or worker owns travels as a
constructor argument, and late launcher-stage resolution declares through
``arg_groups.overrides.declare_late_resolution``, which writes no field and
refuses the published instance.

The textual half of this guard matters because the resolution pipeline's own file
is exempt from the mutation ratchet: a ``self.override(...)`` there — exactly
what the LoRA normalization used — is invisible to it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import re
import unittest
from pathlib import Path

import sglang
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

_SGLANG_ROOT = Path(next(iter(sglang.__path__)))

# ``x.override(`` on anything that is a ServerArgs by name, including the
# pipeline's own ``self.override(`` inside server_args.py.
_PATTERNS = [
    re.compile(r"\bself\.override\("),
    re.compile(r"\bserver_args\.override\("),
    re.compile(r"\bsa\.override\("),
    re.compile(r"\bargs\.override\("),
]

_EXCLUDED = ("multimodal_gen",)


class TestNoServerArgsMutationEntry(CustomTestCase):
    def test_the_methods_are_gone(self):
        self.assertFalse(
            hasattr(ServerArgs, "override"),
            "ServerArgs.override is back; post-publish changes belong on the bags "
            "(get_context().override); a value one runner owns travels as a "
            "constructor argument.",
        )
        self.assertFalse(
            hasattr(ServerArgs, "derive"),
            "ServerArgs.derive is back; a value one runner or worker owns travels "
            "as a constructor argument (draft_attention_backend, MMEncoder "
            "gpu_id), and test doubles copy via "
            "sglang.test.test_utils.server_args_variant.",
        )

    def test_nothing_calls_an_instance_override(self):
        offenders = []
        for path in sorted(_SGLANG_ROOT.rglob("*.py")):
            rel = path.relative_to(_SGLANG_ROOT).as_posix()
            if rel.startswith(_EXCLUDED):
                continue
            source = path.read_text()
            for pattern in _PATTERNS:
                for match in pattern.finditer(source):
                    line = source.count("\n", 0, match.start()) + 1
                    offenders.append(f"{rel}:{line}: {match.group(0)}")
        if offenders:
            self.fail(
                "in-place ServerArgs mutation call-sites:\n"
                + "\n".join(offenders)
                + "\n\nUse get_context().override(source, ...) for resolved config "
                "or declare_late_resolution(...) for pre-publish launcher "
                "resolution."
            )

    def test_nothing_derives(self):
        """A config is never copied-and-edited in the package: a value one
        runner consumes travels as a constructor argument, and test doubles
        copy via ``server_args_variant`` (test_utils)."""
        derive_pattern = re.compile(r"\.derive\(")
        offenders = []
        for path in sorted(_SGLANG_ROOT.rglob("*.py")):
            rel = path.relative_to(_SGLANG_ROOT).as_posix()
            if rel.startswith(_EXCLUDED):
                continue
            source = path.read_text()
            for match in derive_pattern.finditer(source):
                line = source.count("\n", 0, match.start()) + 1
                offenders.append(f"{rel}:{line}")
        self.assertFalse(
            offenders,
            ".derive( call-sites in the package (the method no longer exists):\n"
            + "\n".join(offenders),
        )

    def test_late_resolution_refuses_the_published_config(self):
        from sglang.srt.arg_groups.overrides import (
            declare_late_resolution,
            resolution_result,
        )
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(tp_size=2)
        published = override.install()
        self.addCleanup(override.restore)

        with self.assertRaises(ValueError):
            declare_late_resolution(published, "test", tp_size=4)
        # The refusal left the resolution alone: the hook's declaration stands,
        # and the record still carries the operator's input.
        self.assertEqual(resolution_result(published, "tp_size"), 2)


if __name__ == "__main__":
    unittest.main()
