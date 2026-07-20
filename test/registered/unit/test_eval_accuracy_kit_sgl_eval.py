"""Unit tests for the GSM8K backend dispatch + sgl-eval skip in eval_accuracy_kit.

Hermetic (no server, no real sgl-eval install). These guard the behavior that
existing consumers rely on -- not the sgl-eval happy path, which the live
accuracy runs already cover:

  1. The default GSM8K backend stays on ``run_eval`` (OpenAI completion API);
     the ~47 existing GSM8K consumers must never be silently rerouted.
  2. The legacy ``gsm8k_accuracy_thres`` alias is still honored as the pass/fail
     gate when the canonical ``gsm8k_score_threshold`` is unset.
  3. The sgl-eval reasoning path skips (does not error) when sgl-eval is absent,
     so CI without the optional dependency stays green.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import requests

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits import eval_accuracy_kit as kit
from sglang.test.kits.eval_accuracy_kit import GPQAMixin, GSM8KMixin
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _fake_get(url, *args, **kwargs):
    # flush_cache must succeed (GSM8K calls it unguarded); /server_info is probed
    # by _check_accept_length, which swallows RequestException.
    if str(url).endswith("/flush_cache"):
        return MagicMock()
    raise requests.RequestException()


def _make_host(mixin, method):
    """Build a throwaway mixin host bound to ``method``.

    Created dynamically (never bound at module scope) so it is collected by
    neither runner: CI executes this file via ``python3 <file>`` ->
    ``unittest.main()``, whose loader ignores pytest's ``__test__`` flag, and
    pytest only collects module-level ``Test*`` classes. The host runs only when
    a test below instantiates and drives it directly.
    """
    return type(f"_{mixin.__name__}Host", (mixin, CustomTestCase), {})(method)


class TestEvalKitBackendDispatch(CustomTestCase):
    def _run_gsm8k_default(self, score, **attrs):
        """Run GSM8K on the default (run_eval) backend with run_eval faked to
        return ``score``; returns the SimpleNamespace args run_eval received."""
        captured = {}

        def fake_run_eval(args):
            captured["args"] = args
            return {"score": score}

        host = _make_host(GSM8KMixin, "test_gsm8k")
        host.base_url = "http://127.0.0.1:0"
        host.model = "m"
        for k, v in attrs.items():
            setattr(host, k, v)
        with patch.object(kit, "run_eval", side_effect=fake_run_eval), patch.object(
            kit.requests, "get", side_effect=_fake_get
        ):
            host.test_gsm8k()
        return captured["args"]

    def test_default_backend_uses_run_eval_completion(self):
        # The default path that all existing GSM8K consumers rely on must stay on
        # run_eval's OpenAI completion API -- it must not touch sgl-eval.
        args = self._run_gsm8k_default(0.95, gsm8k_accuracy_thres=0.5)
        self.assertEqual(args.eval_name, "gsm8k")
        self.assertEqual(args.api, "completion")

    def test_legacy_accuracy_thres_alias_gates_score(self):
        # Canonical gsm8k_score_threshold left unset (NaN) -> the legacy
        # gsm8k_accuracy_thres must still be the pass/fail gate.
        self._run_gsm8k_default(0.95, gsm8k_accuracy_thres=0.90)  # above -> passes
        with self.assertRaises(AssertionError):
            self._run_gsm8k_default(0.80, gsm8k_accuracy_thres=0.90)  # below -> fails

    def test_sgl_eval_path_skips_when_not_installed(self):
        # GPQA/AIME25 and the sgl_eval backend must skip -- not error -- when
        # sgl-eval is not installed. None in sys.modules makes the import raise.
        host = _make_host(GPQAMixin, "test_gpqa")
        host.base_url = "http://127.0.0.1:0"
        host.model = "m"
        host.gpqa_score_threshold = 0.5
        absent = {
            k: None for k in ("sgl_eval.registry", "sgl_eval.sampler", "sgl_eval.types")
        }
        with patch.dict(sys.modules, absent):
            with self.assertRaises(unittest.SkipTest):
                host.test_gpqa()


if __name__ == "__main__":
    unittest.main()
