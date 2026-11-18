"""DSA canonical CLI flags, env vars, and the attention registry key."""

import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))


class TestDSAChoicesAndFields(unittest.TestCase):
    """Verify DSA CLI choices and ServerArgs field renaming."""

    def setUp(self):
        from sglang.srt.server_args import ServerArgs

        self.ServerArgs = ServerArgs
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        self.actions = {
            option: action
            for action in parser._actions
            for option in action.option_strings
        }

    def test_dsa_choices_is_canonical(self):
        choices = self.actions["--dsa-prefill-backend"].choices
        self.assertIn("fa3", choices)
        self.assertIn("tilelang", choices)
        self.assertIn("flashinfer_sparse_mla", choices)

    def test_serverargs_has_dsa_fields(self):
        sa = self.ServerArgs
        self.assertTrue(hasattr(sa, "dsa_prefill_backend"))
        self.assertTrue(hasattr(sa, "dsa_decode_backend"))

    def test_serverargs_no_nsa_fields(self):
        """The nsa_* attributes should no longer exist on ServerArgs."""
        sa = self.ServerArgs
        self.assertFalse(
            hasattr(sa, "nsa_prefill_backend"),
            "nsa_prefill_backend should have been renamed",
        )
        self.assertFalse(
            hasattr(sa, "nsa_decode_backend"),
            "nsa_decode_backend should have been renamed",
        )


class TestCLICanonicalFlags(unittest.TestCase):
    """Canonical flags write to canonical attributes with no warning."""

    def setUp(self):
        from sglang.srt.server_args import ServerArgs

        self.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(self.parser)

    def _parse(self, extra_args):
        return self.parser.parse_args(["--model", "dummy"] + extra_args)

    def test_dsa_prefill_backend_canonical(self):
        args = self._parse(["--dsa-prefill-backend", "fa3"])
        self.assertEqual(args.dsa_prefill_backend, "fa3")

    def test_dsa_decode_backend_canonical(self):
        args = self._parse(["--dsa-decode-backend", "tilelang"])
        self.assertEqual(args.dsa_decode_backend, "tilelang")

    def test_defaults_are_none_or_false(self):
        args = self._parse([])
        self.assertIsNone(args.dsa_prefill_backend)
        self.assertIsNone(args.dsa_decode_backend)

    def test_attention_backend_dsa_key_in_choices(self):
        args = self._parse(["--attention-backend", "dsa"])
        self.assertEqual(args.attention_backend, "dsa")


class TestAttentionRegistry(unittest.TestCase):
    """Registry: the 'dsa' key creates the backend."""

    def test_dsa_key_registered(self):
        from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS

        self.assertIn("dsa", ATTENTION_BACKENDS)


class TestEnvVarAliases(unittest.TestCase):
    """SGLANG_DSA_* canonical env vars."""

    def setUp(self):
        # Clean state for every test
        for key in [
            "SGLANG_DSA_FUSE_TOPK",
            "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD",
        ]:
            os.environ.pop(key, None)
        # Re-import to reset descriptor state
        from importlib import reload

        import sglang.srt.environ as e

        reload(e)
        from sglang.srt.environ import envs

        self.envs = envs

    def tearDown(self):
        for key in [
            "SGLANG_DSA_FUSE_TOPK",
            "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD",
        ]:
            os.environ.pop(key, None)

    def test_dsa_fuse_topk_default(self):
        self.assertTrue(self.envs.SGLANG_DSA_FUSE_TOPK.get())

    def test_dsa_fuse_topk_canonical_set(self):
        os.environ["SGLANG_DSA_FUSE_TOPK"] = "0"
        self.assertFalse(self.envs.SGLANG_DSA_FUSE_TOPK.get())

    def test_dsa_threshold_default(self):
        self.assertEqual(
            self.envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get(), 2048
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
