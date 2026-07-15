"""Tests that the DSA decode backend argument rejects the prefill-only
``flashmla_auto`` value at argparse time instead of deferring the failure to the
first decode step.

``flashmla_auto`` is only resolvable on the prefill path
(``enable_auto_select_prefill_impl`` in ``dsa_backend.py``); the decode dispatch
has no branch for it and would ``raise ValueError(f"Unsupported ...")``. It used
to pass argparse because the decode arg (and its deprecated ``--nsa-decode-backend``
alias) shared the full ``DSA_CHOICES`` list with the prefill arg, so the server
booted and crashed only at the first decode forward. The decode arg now uses the
narrowed ``DSA_DECODE_CHOICES`` list.
"""

import argparse
import contextlib
import io
import unittest

from sglang.srt.server_args import DSA_CHOICES, DSA_DECODE_CHOICES, ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDsaDecodeBackendValidation(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def _parse(self, args_list):
        # argparse writes usage to stderr on a bad choice; keep test output clean.
        with contextlib.redirect_stderr(io.StringIO()):
            return self.parser.parse_args(["--model", "dummy"] + args_list)

    def test_decode_choices_exclude_only_flashmla_auto(self):
        """The narrowed decode list drops flashmla_auto and nothing else."""
        self.assertNotIn("flashmla_auto", DSA_DECODE_CHOICES)
        self.assertEqual(set(DSA_DECODE_CHOICES), set(DSA_CHOICES) - {"flashmla_auto"})

    def test_decode_backend_rejects_flashmla_auto(self):
        """--dsa-decode-backend flashmla_auto fails fast at argparse."""
        with self.assertRaises(SystemExit):
            self._parse(["--dsa-decode-backend", "flashmla_auto"])

    def test_nsa_decode_alias_rejects_flashmla_auto(self):
        """The deprecated --nsa-decode-backend alias is narrowed too."""
        with self.assertRaises(SystemExit):
            self._parse(["--nsa-decode-backend", "flashmla_auto"])

    def test_decode_backend_accepts_valid_values(self):
        """Every non-auto DSA backend is still accepted for decode."""
        for backend in DSA_DECODE_CHOICES:
            with self.subTest(backend=backend):
                args = self._parse(["--dsa-decode-backend", backend])
                self.assertEqual(args.dsa_decode_backend, backend)

    def test_prefill_backend_still_accepts_flashmla_auto(self):
        """flashmla_auto remains valid on the prefill path (unchanged)."""
        args = self._parse(["--dsa-prefill-backend", "flashmla_auto"])
        self.assertEqual(args.dsa_prefill_backend, "flashmla_auto")


if __name__ == "__main__":
    unittest.main()
