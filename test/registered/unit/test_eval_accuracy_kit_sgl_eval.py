"""Unit tests for sgl-eval-backed accuracy mixin dispatch.

Hermetic (no server, no real sgl-eval install). These guard the dispatch that
existing consumers rely on, not the sgl-eval happy path the live runs cover.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import requests

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits import eval_accuracy_kit as kit
from sglang.test.kits.eval_accuracy_kit import GPQAMixin, GSM8KMixin, MMMUProMixin
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _fake_get(url, *args, **kwargs):
    # flush_cache must succeed (GSM8K calls it unguarded); /server_info is probed
    # by _check_accept_length, which swallows RequestException.
    if str(url).endswith("/flush_cache"):
        return MagicMock()
    raise requests.RequestException()


def _make_host(mixin, method):
    """Build a throwaway mixin host bound to ``method``.

    Created dynamically so neither runner collects it: ``unittest.main()`` ignores
    pytest's ``__test__`` flag, and pytest only collects module-level ``Test*``.
    """
    return type(f"_{mixin.__name__}Host", (mixin, CustomTestCase), {})(method)


class TestEvalKitBackendDispatch(CustomTestCase):
    def _run_gsm8k_default(self, score, **attrs):
        """Run GSM8K on the default backend; returns the args run_eval received."""
        captured = {}

        def fake_run_eval(args):
            captured["args"] = args
            return {"score": score}

        host = _make_host(GSM8KMixin, "test_gsm8k")
        host.base_url = "http://127.0.0.1:0"
        host.model = "m"
        for k, v in attrs.items():
            setattr(host, k, v)
        with (
            patch.object(kit, "run_eval", side_effect=fake_run_eval),
            patch.object(kit.requests, "get", side_effect=_fake_get),
        ):
            host.test_gsm8k()
        return captured["args"]

    def test_default_backend_uses_run_eval_completion(self):
        args = self._run_gsm8k_default(0.95, gsm8k_accuracy_thres=0.5)
        self.assertEqual(args.eval_name, "gsm8k")
        self.assertEqual(args.api, "completion")

    def test_legacy_accuracy_thres_alias_gates_score(self):
        # Canonical gsm8k_score_threshold left unset (NaN).
        self._run_gsm8k_default(0.95, gsm8k_accuracy_thres=0.90)  # above -> passes
        with self.assertRaises(AssertionError):
            self._run_gsm8k_default(0.80, gsm8k_accuracy_thres=0.90)  # below -> fails

    def test_sgl_eval_path_skips_when_not_installed(self):
        # None in sys.modules makes ``import sgl_eval`` raise ImportError.
        host = _make_host(GPQAMixin, "test_gpqa")
        host.base_url = "http://127.0.0.1:0"
        host.model = "m"
        host.gpqa_score_threshold = 0.5
        with patch.dict(sys.modules, {"sgl_eval": None}):
            with self.assertRaises(unittest.SkipTest):
                host.test_gpqa()

    def _run_mmmu_pro(self, score):
        captured = {}

        def fake_run_sgl_eval(args):
            captured["args"] = args
            return {"score": score}

        host = _make_host(MMMUProMixin, "test_mmmu_pro")
        host.base_url = "http://127.0.0.1:0"
        host.model = "deployment-model"
        host.mmmu_pro_score_threshold = 0.75
        host.mmmu_pro_load_preset_from_model_id = "moonshotai/Kimi-K3"
        with (
            patch.object(kit, "run_sgl_eval", side_effect=fake_run_sgl_eval),
            patch.object(kit.requests, "get", side_effect=_fake_get),
        ):
            host.test_mmmu_pro()
        return captured["args"]

    def test_mmmu_pro_uses_kimi_preset_and_300_examples(self):
        args = self._run_mmmu_pro(0.80)

        self.assertEqual(args.eval_name, "mmmu_pro")
        self.assertEqual(args.load_preset_from_model_id, "moonshotai/Kimi-K3")
        self.assertEqual(args.num_examples, 300)
        self.assertIsNone(args.num_threads)
        self.assertIsNone(args.model)
        for attr in ("temperature", "top_p", "max_tokens", "reasoning_effort"):
            self.assertFalse(hasattr(args, attr))

    def test_mmmu_pro_score_threshold_gates_result(self):
        with self.assertRaises(AssertionError):
            self._run_mmmu_pro(0.74)


if __name__ == "__main__":
    unittest.main()
