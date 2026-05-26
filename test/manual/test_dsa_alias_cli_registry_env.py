"""
Manual test for step 01: NSA → DSA user-facing alias layer.

Tests:
  1. CLI: --dsa-* canonical flags write to dsa_* attrs
  2. CLI: --nsa-* deprecated flags write to dsa_* attrs + log deprecation warning
  3. Registry: "dsa" key creates the backend; "nsa" key triggers DeprecationWarning
  4. Env: SGLANG_DSA_* canonical vars work
  5. Env: SGLANG_NSA_* deprecated vars fall back to SGLANG_DSA_* with DeprecationWarning

Run:
    python test/manual/test_dsa_alias_cli_registry_env.py
"""

import argparse
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))


class TestDSAChoicesAndFields(unittest.TestCase):
    """Verify DSA_CHOICES constant and ServerArgs field renaming."""

    def setUp(self):
        from sglang.srt.server_args import (
            DSA_CHOICES,
            DSA_PREFILL_CP_SPLIT_CHOICES,
            NSA_CHOICES,
            NSA_PREFILL_CP_SPLIT_CHOICES,
            ServerArgs,
        )

        self.ServerArgs = ServerArgs
        self.DSA_CHOICES = DSA_CHOICES
        self.NSA_CHOICES = NSA_CHOICES
        self.DSA_PREFILL_CP_SPLIT_CHOICES = DSA_PREFILL_CP_SPLIT_CHOICES
        self.NSA_PREFILL_CP_SPLIT_CHOICES = NSA_PREFILL_CP_SPLIT_CHOICES

    def test_dsa_choices_is_canonical(self):
        self.assertIn("fa3", self.DSA_CHOICES)
        self.assertIn("tilelang", self.DSA_CHOICES)

    def test_nsa_choices_is_alias(self):
        self.assertIs(
            self.NSA_CHOICES,
            self.DSA_CHOICES,
            "NSA_CHOICES must be the same object as DSA_CHOICES",
        )

    def test_nsa_cp_split_choices_is_alias(self):
        self.assertIs(
            self.NSA_PREFILL_CP_SPLIT_CHOICES,
            self.DSA_PREFILL_CP_SPLIT_CHOICES,
        )

    def test_serverargs_has_dsa_fields(self):
        sa = self.ServerArgs
        self.assertTrue(hasattr(sa, "dsa_prefill_backend"))
        self.assertTrue(hasattr(sa, "dsa_decode_backend"))
        self.assertTrue(hasattr(sa, "enable_dsa_prefill_context_parallel"))
        self.assertTrue(hasattr(sa, "dsa_prefill_cp_mode"))

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
        self.assertFalse(hasattr(sa, "enable_nsa_prefill_context_parallel"))
        self.assertFalse(hasattr(sa, "nsa_prefill_cp_mode"))


class TestCLICanonicalFlags(unittest.TestCase):
    """--dsa-* canonical flags write to dsa_* attributes with no warning."""

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

    def test_enable_dsa_prefill_cp_canonical(self):
        args = self._parse(["--enable-dsa-prefill-context-parallel"])
        self.assertTrue(args.enable_dsa_prefill_context_parallel)

    def test_dsa_prefill_cp_mode_canonical(self):
        args = self._parse(["--dsa-prefill-cp-mode", "in-seq-split"])
        self.assertEqual(args.dsa_prefill_cp_mode, "in-seq-split")

    def test_defaults_are_none_or_false(self):
        args = self._parse([])
        self.assertIsNone(args.dsa_prefill_backend)
        self.assertIsNone(args.dsa_decode_backend)
        self.assertFalse(args.enable_dsa_prefill_context_parallel)
        self.assertEqual(args.dsa_prefill_cp_mode, "round-robin-split")

    def test_attention_backend_dsa_key_in_choices(self):
        args = self._parse(["--attention-backend", "dsa"])
        self.assertEqual(args.attention_backend, "dsa")


class TestCLIDeprecatedFlags(unittest.TestCase):
    """--nsa-* deprecated flags write to dsa_* attributes and emit logger warning."""

    def setUp(self):
        import logging

        from sglang.srt.server_args import ServerArgs

        self.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(self.parser)

        # Capture log output to detect deprecation warnings
        self.log_records = []
        handler = (
            logging.handlers_collector(self.log_records)
            if hasattr(logging, "handlers_collector")
            else None
        )

    def _parse(self, extra_args):
        return self.parser.parse_args(["--model", "dummy"] + extra_args)

    def _parse_capture_warnings(self, extra_args):
        """Parse and capture both warnings.warn and logger output."""
        import io
        import logging

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.WARNING)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            args = self._parse(extra_args)
        finally:
            root.removeHandler(handler)
        return args, log_stream.getvalue()

    def test_nsa_prefill_backend_deprecated_writes_to_dsa(self):
        args, log_output = self._parse_capture_warnings(
            ["--nsa-prefill-backend", "fa3"]
        )
        self.assertEqual(args.dsa_prefill_backend, "fa3")
        self.assertIn(
            "deprecated",
            log_output.lower(),
            f"Expected deprecation warning in log; got: {log_output!r}",
        )

    def test_nsa_decode_backend_deprecated_writes_to_dsa(self):
        args, log_output = self._parse_capture_warnings(
            ["--nsa-decode-backend", "tilelang"]
        )
        self.assertEqual(args.dsa_decode_backend, "tilelang")
        self.assertIn("deprecated", log_output.lower())

    def test_enable_nsa_prefill_cp_deprecated(self):
        args, log_output = self._parse_capture_warnings(
            ["--enable-nsa-prefill-context-parallel"]
        )
        self.assertTrue(args.enable_dsa_prefill_context_parallel)
        self.assertIn("deprecated", log_output.lower())

    def test_nsa_prefill_cp_mode_deprecated(self):
        args, log_output = self._parse_capture_warnings(
            ["--nsa-prefill-cp-mode", "in-seq-split"]
        )
        self.assertEqual(args.dsa_prefill_cp_mode, "in-seq-split")
        self.assertIn("deprecated", log_output.lower())

    def test_attention_backend_nsa_still_accepted(self):
        """attention_backend='nsa' still parses without error (registry handles the deprecation)."""
        args = self._parse(["--attention-backend", "nsa"])
        self.assertEqual(args.attention_backend, "nsa")


class TestAttentionRegistry(unittest.TestCase):
    """Registry: 'dsa' key creates backend; 'nsa' key emits DeprecationWarning."""

    def test_dsa_key_registered(self):
        from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS

        self.assertIn("dsa", ATTENTION_BACKENDS)

    def test_nsa_key_still_registered(self):
        from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS

        self.assertIn("nsa", ATTENTION_BACKENDS, "nsa must remain as deprecated alias")

    def test_nsa_key_emits_deprecation_warning(self):
        """Calling the nsa factory should emit DeprecationWarning."""
        from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS

        nsa_factory = ATTENTION_BACKENDS.get("nsa")
        self.assertIsNotNone(nsa_factory)

        class _FakeRunner:
            server_args = type("S", (), {"attention_backend": "nsa"})()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                nsa_factory(_FakeRunner())
            except Exception:
                pass  # import errors OK; we only care about DeprecationWarning
            dep_warns = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertTrue(
                len(dep_warns) > 0,
                "Expected DeprecationWarning when using 'nsa' registry key",
            )
            self.assertIn("deprecated", str(dep_warns[0].message).lower())


class TestEnvVarAliases(unittest.TestCase):
    """SGLANG_DSA_* canonical; SGLANG_NSA_* fall back with DeprecationWarning."""

    def setUp(self):
        # Clean state for every test
        for key in [
            "SGLANG_DSA_FUSE_TOPK",
            "SGLANG_NSA_FUSE_TOPK",
            "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD",
            "SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD",
            "SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA",
            "SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA",
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
            "SGLANG_NSA_FUSE_TOPK",
            "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD",
            "SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD",
        ]:
            os.environ.pop(key, None)

    def test_dsa_fuse_topk_default(self):
        self.assertTrue(self.envs.SGLANG_DSA_FUSE_TOPK.get())

    def test_dsa_fuse_topk_canonical_set(self):
        os.environ["SGLANG_DSA_FUSE_TOPK"] = "0"
        self.assertFalse(self.envs.SGLANG_DSA_FUSE_TOPK.get())

    def test_nsa_fuse_topk_deprecated_fallback(self):
        """SGLANG_NSA_FUSE_TOPK=0 should be read by SGLANG_DSA_FUSE_TOPK with DeprecationWarning."""
        os.environ["SGLANG_NSA_FUSE_TOPK"] = "0"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = self.envs.SGLANG_DSA_FUSE_TOPK.get()
            self.assertFalse(val)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertTrue(
                len(dep) > 0, "Expected DeprecationWarning for SGLANG_NSA_FUSE_TOPK"
            )

    def test_dsa_threshold_default(self):
        self.assertEqual(
            self.envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get(), 2048
        )

    def test_nsa_threshold_deprecated_fallback(self):
        os.environ["SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD"] = "1024"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = self.envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()
            self.assertEqual(val, 1024)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertTrue(len(dep) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
