"""A draft bundled inside its target's repo is addressed by subfolder.

The Hub has no single-string form for this: a repo ID may contain at most one
"/", so "org/target/DFlash" is not a valid ID and the sub-path has to travel
separately. `resolve_model_subfolder` collapses the pair back to one real
directory during path resolution, before the speculative hook reads the draft
config, so config loading and the weight loader cannot end up reading from
different places.

These are directory-only cases; the Hub branch needs the network and is left to
integration coverage.
"""

import os
import tempfile
import unittest

from sglang.srt.server_args import ServerArgs, resolve_model_subfolder
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestResolveModelSubfolder(CustomTestCase):
    def _target_with_draft(self):
        """A target directory holding its draft in ./DFlash, as we publish them."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = tmp.name
        draft = os.path.join(root, "DFlash")
        os.makedirs(draft)
        with open(os.path.join(root, "config.json"), "w") as f:
            f.write('{"model_type": "target"}')
        with open(os.path.join(draft, "config.json"), "w") as f:
            f.write('{"model_type": "draft"}')
        return root, draft

    def test_a_subfolder_resolves_to_the_directory_inside_the_target(self):
        root, draft = self._target_with_draft()
        self.assertEqual(resolve_model_subfolder(root, "DFlash"), draft)
        # Surrounding slashes are the same request.
        self.assertEqual(resolve_model_subfolder(root, "/DFlash/"), draft)

    def test_an_absent_subfolder_returns_the_path_untouched(self):
        root, _ = self._target_with_draft()
        for absent in (None, "", "/"):
            self.assertEqual(resolve_model_subfolder(root, absent), root)

    def test_a_missing_subfolder_fails_loudly(self):
        """Silently falling back to the target would load the wrong weights."""
        root, _ = self._target_with_draft()
        with self.assertRaises(ValueError) as ctx:
            resolve_model_subfolder(root, "NotThere")
        self.assertIn("NotThere", str(ctx.exception))

    def test_the_draft_path_is_folded_during_server_args_resolution(self):
        """The contract: the draft path collapses to the subfolder and the
        target is untouched, so every downstream consumer sees one plain dir."""
        root, draft = self._target_with_draft()
        args = ServerArgs(
            model_path=root,
            speculative_draft_model_path=root,
            speculative_draft_model_subfolder="DFlash",
        )
        self.assertEqual(args.speculative_draft_model_path, draft)
        self.assertEqual(args.model_path, root)


if __name__ == "__main__":
    unittest.main()
