"""Accuracy mixin dispatch, generation settings and score gates."""

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
        with (
            patch.object(kit, "run_eval", side_effect=fake_run_eval),
            patch.object(kit.requests, "get", side_effect=_fake_get),
        ):
            host.test_gsm8k()
        return captured["args"]

    def test_gsm8k_forwards_sgl_eval_generation_settings(self):
        args = self._run_gsm8k_default(
            0.95,
            gsm8k_accuracy_thres=0.5,
            gsm8k_thinking=True,
            gsm8k_temperature=0.7,
            gsm8k_top_p=0.9,
            gsm8k_max_tokens=8192,
            gsm8k_n_repeats=3,
        )
        self.assertEqual(args.eval_name, "gsm8k")
        self.assertFalse(hasattr(args, "api"))
        self.assertTrue(args.sgl_eval_thinking)
        self.assertEqual(args.max_tokens, 8192)
        self.assertEqual(args.repeat, 3)
        self.assertEqual(args.temperature, 0.7)
        self.assertEqual(args.top_p, 0.9)

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

    def _run_mmmu_pro(self, score):
        captured = {}

        def fake_run_eval(args):
            captured["args"] = args
            return {"score": score}

        host = _make_host(MMMUProMixin, "test_mmmu_pro")
        host.base_url = "http://127.0.0.1:0"
        host.model = "deployment-model"
        host.mmmu_pro_score_threshold = 0.75
        host.mmmu_pro_load_preset_from_model_id = "moonshotai/Kimi-K3"
        with (
            patch.object(kit, "run_eval", side_effect=fake_run_eval),
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
