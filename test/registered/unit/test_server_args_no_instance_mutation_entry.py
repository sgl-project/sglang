"""``ServerArgs`` has no in-place mutation entry, and nothing calls one.

``ServerArgs.override(source, **fields)`` used to mutate a resolved instance;
after resolution the fields are the record the config bags were projected from,
so such a write desyncs every namespace reader. The method is gone and the two
sanctioned replacements are ``get_context().override`` (post-publish, writes the
bags) and ``ServerArgs.derive`` (a variant for another runner / process). Late
launcher-stage resolution writes in place through
``arg_groups.overrides.declare_late_resolution``, which refuses the published
instance.

The textual half of this guard matters because the resolution pipeline's own file
is exempt from the mutation ratchet: a ``self.override(...)`` there — exactly
what the LoRA normalization used — is invisible to it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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
    def test_the_method_is_gone(self):
        self.assertFalse(
            hasattr(ServerArgs, "override"),
            "ServerArgs.override is back; post-publish changes belong on the bags "
            "(get_context().override) and per-runner values on a derive() variant.",
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
                + "\n\nUse get_context().override(source, ...) for resolved config, "
                "server_args.derive(source, ...) for a per-runner variant, or "
                "declare_late_resolution(...) for pre-publish launcher resolution."
            )

    def test_late_resolution_refuses_the_published_config(self):
        from sglang.srt.arg_groups.overrides import declare_late_resolution
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(tp_size=2)
        published = override.install()
        self.addCleanup(override.restore)

        with self.assertRaises(ValueError):
            declare_late_resolution(published, "test", tp_size=4)
        self.assertEqual(published.tp_size, 2)


if __name__ == "__main__":
    unittest.main()
