"""``ServerArgs.derive`` is the only way one config becomes another.

After resolution the instance is the process's read-only startup record and the
object the config bags were projected from, so it cannot be mutated: a change to
resolved config goes to the bags (``get_context().override``), and a config that
differs for one runner or worker — a draft's context length, an encode worker's
device — is a second object.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.runtime_context import get_context, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


class TestServerArgsDerive(CustomTestCase):
    def tearDown(self):
        reset_context()

    def _resolved(self) -> ServerArgs:
        """A config in the post-resolution state, without a real model."""
        server_args = ServerArgs(model_path="dummy")
        object.__setattr__(server_args, "_declarations_materialized", True)
        return server_args

    def test_the_receiver_is_untouched(self):
        server_args = self._resolved()
        variant = server_args.derive("draft_worker.copy", context_length=1024)

        self.assertEqual(variant.context_length, 1024)
        self.assertIsNot(variant, server_args)
        self.assertIsNone(server_args.context_length)

    def test_a_published_config_rejects_assignment_but_still_derives(self):
        override = get_context().override_server_args(tp_size=2)
        published = override.install()
        self.addCleanup(override.restore)

        with self.assertRaises(AttributeError):
            published.tp_size = 4

        self.assertEqual(published.derive("worker", tp_size=4).tp_size, 4)
        self.assertEqual(published.tp_size, 2)

    def test_the_variant_is_not_published_by_deriving(self):
        override = get_context().override_server_args(tp_size=2)
        published = override.install()
        self.addCleanup(override.restore)

        published.derive("worker", tp_size=4)

        self.assertIs(get_context().server_args, published)

    def test_the_variant_is_frozen_too(self):
        variant = self._resolved().derive("worker", context_length=1024)
        with self.assertRaises(AttributeError):
            variant.context_length = 2048

    def test_provenance_is_recorded(self):
        variant = self._resolved().derive("draft_worker.copy", watchdog_timeout=1.0)
        self.assertIn(
            ("draft_worker.copy", {"watchdog_timeout": 1.0}),
            getattr(variant, "_runtime_mutations", []),
        )

    def test_a_resolvable_field_joins_the_declaration_stash(self):
        """So a re-resolution of the variant keeps the derived value."""
        variant = self._resolved().derive("draft_worker.build", kv_cache_dtype="bf16")
        stash = getattr(variant, "_resolved_overrides", [])
        self.assertIn(("draft_worker.build", {"kv_cache_dtype": "bf16"}), stash)


if __name__ == "__main__":
    unittest.main()
